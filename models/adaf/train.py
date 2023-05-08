import sys
import os
from datetime import datetime
from io import StringIO
import torch.nn as nn
import torch.optim as optim
import torch
import logging
from tqdm import tqdm

from pathlib import Path
from omegaconf import DictConfig
from models.adaf.model import ADAF
from models.train_utils import _get_loss_obj, _set_regularization, _get_data_loader, _get_dataset, _get_folder_structure, \
    _get_tester, data_to_device, _save_weights_and_optimizer, get_tester
from utils import get_model_epoch, get_config_map, get_test_results, get_metric_values, get_metric_median, get_metric_mean

from neuralhydrology.utils.config import Config
from neuralhydrology.training.logger import Logger
from neuralhydrology.modelzoo.inputlayer import InputLayer
import wandb

LOGGER = logging.getLogger(__name__)

class ADAF_Trainer(nn.Module):
    ''' Train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, cfg: DictConfig, src_cfg_path: str, tgt_cfg_path: str):
        super(ADAF_Trainer, self).__init__()
        self.cfg = cfg
        self.src_cfg_path = src_cfg_path
        self.tgt_cfg_path = tgt_cfg_path
        
        self.src_cfg = Config(Path(self.src_cfg_path))
        self.tgt_cfg = Config(Path(self.tgt_cfg_path))
        self.device = self.tgt_cfg.device
        self._scaler = {}
        self.validator = None
        self.is_wandb = False
        
        # Hyper parameters
        self._lambda = -0.1 
        self.hidden_size = self.tgt_cfg.hidden_size 
        self.batch_size = self.tgt_cfg.batch_size 
        self.epochs = self.tgt_cfg.epochs
        
        # Model components
        self.embedding_net = InputLayer(self.tgt_cfg)
        self.input_size = self.embedding_net.output_size  # the number of expected features in the input X
        
        self.model_name = self.cfg.model
        
        # Set up WanDB for running experiments
        # This helps generate a background process 
        # that syncs and logs data for experiment runs
        if self.is_wandb:
            now = datetime.now()
            run_name = f'{self.model_name}-epochs{self.tgt_cfg.epochs}_{now.month}-{now.day}_{now.hour-5}:{now.minute}:{now.second}'
            self.run = wandb.init(
                project="adaf-streamflow",
                entity="rro2q2",
                name= run_name,
                notes="first experiment",
                tags=["main model", "ml4hydro"]
            )
            
            
            wandb.config = {
                "epochs": self.tgt_cfg.epochs,
                "learning_rate": self.tgt_cfg.learning_rate,
                "batch_size": self.tgt_cfg.batch_size,
                "hidden_size": self.tgt_cfg.hidden_size,
                "num_workers": self.tgt_cfg.num_workers
            }
            self.epochs = wandb.config["epochs"]
        
            # self.model_artifact = wandb.Artifact(name="adaf-model", type="model")
            # self.data_artifact = wandb.Artifact(name="camels-dataset", type="dataset")
            # self.data_artifact.add_dir(local_path='data_dir')
        
        # Create folder structure for target
        stream = StringIO()
        LOGGER.addHandler(logging.StreamHandler(stream))
        _get_folder_structure(self.tgt_cfg, LOGGER)
        cur_stream = self.get_stream(stream)
        # Convert the stream of outputs of model configurations into a Dict type
        self.tgt_config_map = get_config_map(cur_stream)
        
        # Create folder structure for source
        _get_folder_structure(self.src_cfg, LOGGER)
        
        # Instantiate model
        self.src_model = ADAF(cfg=self.src_cfg)
        self.tgt_model = ADAF(cfg=self.tgt_cfg)
    
    def get_stream(self, stream):
        stream.seek(0)
        return stream.read().strip().split('\n')

    def init_train(self) -> None:
        # NSE Loss function
        self.criterion = _get_loss_obj(self.tgt_cfg).to(self.device)
        self.discriminator_criterion = nn.BCELoss().to(self.device)
        
        # Optimizer 
        self.optimizer = optim.Adam(
            list(self.src_model.encoder.parameters()) +
            list(self.src_model.decoder.attention.parameters()) +
            list(self.src_model.decoder.parameters()) +
            list(self.tgt_model.encoder.parameters()) +
            list(self.tgt_model.decoder.attention.parameters()) +
            list(self.tgt_model.decoder.parameters()) +
            list(self.tgt_model.discriminator.parameters()),
            lr=self.tgt_cfg.learning_rate[0])
        
        # Add possible regularization terms to the loss function.
        _set_regularization(self.criterion, self.tgt_cfg)
        
        # Load source dataset
        print(f"----- Getting source dataset {self.src_cfg.dataset} -----")
        source_ds = _get_dataset(self.src_cfg, "train", self._scaler)
        if len(source_ds) == 0:
            raise ValueError("Source dataset contains no samples.")
        self.source_loader = _get_data_loader(self.src_cfg, ds=source_ds)
        print(f"Length of source loader: {len(self.source_loader)}")
        
        # Load target dataset
        print(f"----- Getting target dataset {self.tgt_cfg.dataset} -----")
        target_ds = _get_dataset(self.tgt_cfg, "train", self._scaler)
        if len(target_ds) == 0:
            raise ValueError("Target dataset contains no samples.")
        self.target_loader = _get_data_loader(self.tgt_cfg, ds=target_ds)
            
        print(f"Length of target loader: {len(self.target_loader)}")
        
        self.experiment_logger = Logger(cfg=self.tgt_cfg)
        if self.tgt_cfg.log_tensorboard:
            self.experiment_logger.start_tb()
            
        if self.tgt_cfg.validate_every is not None:
            if self.tgt_cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Validation set to validate every {self.tgt_cfg.validate_every} epoch(s), but ",
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set."
                ]
                LOGGER.warning("".join(warn_msg))
                self.tgt_cfg.validate_n_random_basins = self.tgt_cfg.number_of_basins
            self.validator = _get_tester(domain_cfg=self.tgt_cfg)

    def train(self) -> None:
        # Initialize train
        self.init_train()
    
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            if epoch in self.tgt_cfg.learning_rate.keys():
                LOGGER.info(f"Setting learning rate to {self.tgt_cfg.learning_rate[epoch]}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.tgt_cfg.learning_rate[epoch]
                
            # Set model to train mode
            self.src_model.train()
            self.tgt_model.train()
            
            pbar = tqdm(enumerate(zip(self.source_loader, self.target_loader)), file=sys.stdout)
            pbar.set_description(desc=f'# Epoch {epoch}')
            
            for _, (source_data, target_data) in pbar:
                # Set source and target data to CUDA device
                source_data = data_to_device(source_data, self.device)
                target_data = data_to_device(target_data, self.device)
                
                # Embed dynamic and static features
                source_data['x_d'] = self.embedding_net(source_data)
                target_data['x_d'] = self.embedding_net(target_data)

                # Zero the gradients for each optimizer
                self.optimizer.zero_grad()
                
                # Run forward function for encoder decoder model for source and target network
                source_pred = self.src_model(source_data)
                # Source ground truth
                source_gt = dict()
                source_gt['x_d'] = source_data['x_d'][self.src_model.iw:, :, :]
                source_gt['x_s'] = source_data['x_s']
                source_gt['y'] = source_data['y'][:, self.src_model.iw:, :]
                source_gt['per_basin_target_stds'] = source_data['per_basin_target_stds']
                
                target_pred = self.tgt_model(target_data)
                # Target ground truth
                target_gt = dict()
                target_gt['x_d'] = target_data['x_d'][self.tgt_model.iw:, :, :]
                target_gt['x_s'] = target_data['x_s']
                target_gt['y'] = target_data['y'][:, self.tgt_model.iw:, :]
                target_gt['per_basin_target_stds'] = target_data['per_basin_target_stds']
                
                # Compute the loss for source and target network
                source_loss, source_all_losses = self.criterion(source_pred, source_gt)
                target_loss, target_all_losses = self.criterion(target_pred, target_gt)
                
                # Combine attention features from source and target networks
                # to pass though a dense linear layer for domain invariance
                combined_feature = torch.cat((source_pred['attn_data'], target_pred['attn_data']), dim=0) 
                
                # Compute loss for discriminator network
                domain_pred = self.tgt_model.discriminator(combined_feature)
                domain_source_labels = torch.zeros(source_pred['attn_data'].shape[0]).type(torch.LongTensor)
                domain_target_labels = torch.ones(target_pred['attn_data'].shape[0]).type(torch.LongTensor)
                domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), dim=0).reshape((-1, 1)).to(self.device)
                domain_loss = self.discriminator_criterion(domain_pred.squeeze(-1), domain_combined_label.to(self.device).float().squeeze(-1))
                
                # Compute loss with adversarial training
                tot_loss = source_loss + target_loss + (self._lambda * domain_loss)
                
                if self.is_wandb:
                    wandb.log({"loss": tot_loss})
                
                # Backpropagation
                tot_loss.backward()
                
                if self.tgt_cfg.clip_gradient_norm is not None:
                    # Clip gradient for sequence generators 
                    torch.nn.utils.clip_grad_norm_(self.src_model.encoder.parameters(), self.src_cfg.clip_gradient_norm)
                    torch.nn.utils.clip_grad_norm_(self.src_model.decoder.parameters(), self.src_cfg.clip_gradient_norm)
                    torch.nn.utils.clip_grad_norm_(self.tgt_model.encoder.parameters(), self.tgt_cfg.clip_gradient_norm)
                    torch.nn.utils.clip_grad_norm_(self.tgt_model.decoder.parameters(), self.tgt_cfg.clip_gradient_norm)
                    # Clip gradient for attention module
                    torch.nn.utils.clip_grad_norm_(self.src_model.decoder.attention.parameters(), self.src_cfg.clip_gradient_norm)
                    torch.nn.utils.clip_grad_norm_(self.tgt_model.decoder.attention.parameters(), self.tgt_cfg.clip_gradient_norm)
                    # Clip gradient for discriminator module
                    torch.nn.utils.clip_grad_norm_(self.tgt_model.discriminator.parameters(), self.tgt_cfg.clip_gradient_norm)
                
                self.optimizer.step()
                
                # Progress bar 
                pbar.set_postfix_str(f"Loss: {tot_loss.item():.4f}")
                self.experiment_logger.log_step(loss=tot_loss.item())
                
            print(f"Target loss: {tot_loss}")
            avg_loss = self.experiment_logger.summarise()
            LOGGER.info(f"Epoch {epoch} average loss: {avg_loss}")
            
            # Run validation during training
            self.validate(epoch)
        
        # 4. Log an artifact to W&B
        # wandb.log_artifact(self.tgt_model)
        # self.run.log_artifact(self.data_artifact)
        
        # make sure to close tensorboard to avoid losing the last epoch
        if self.tgt_cfg.log_tensorboard:
            self.experiment_logger.stop_tb()
    
    def validate(self, epoch) -> None:
        if epoch % self.tgt_cfg.save_weights_every == 0:
                _save_weights_and_optimizer(model=self.tgt_model, epoch=epoch, domain_cfg=self.tgt_cfg)
        
        if (self.validator is not None) and (epoch % self.tgt_cfg.validate_every == 0):
                self.validator.evaluate(epoch=epoch,
                                        save_results=self.tgt_cfg.save_validation_results,
                                        metrics=self.tgt_cfg.metrics,
                                        model=self.tgt_model,
                                        experiment_logger=self.experiment_logger.valid())

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_loss']:.5f}"
                if self.tgt_cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_loss')
                    LOGGER.info(print_msg)
                    
    def start_evaluation(self, cfg: Config, run_dir: Path, epoch: int = None, period: str = "test"):
        """Start evaluation of a trained network
        Parameters
        ----------
        cfg : Config
            The run configuration, read from the run directory.
        run_dir : Path
            Path to the run directory.
        epoch : int, optional
            Define a specific epoch to evaluate. By default, the weights of the last epoch are used.
        period : {'train', 'validation', 'test'}, optional
            The period to evaluate, by default 'test'.
        """
        tester = get_tester(cfg=cfg, run_dir=run_dir, period=period, init_model=True)
        tester.evaluate(epoch=epoch, save_results=True, metrics=["NSE", "KGE", "alpha-nse", "beta-nse"])
                    
    def eval_run(self, run_dir: Path, period: str, epoch: int = None, gpu: int = None):
        """Start evaluating a trained model.
        
        Parameters
        ----------
        run_dir : Path
            Path to the run directory.
        period : {'train', 'validation', 'test'}
            The period to evaluate.
        epoch : int, optional
            Define a specific epoch to use. By default, the weights of the last epoch are used.  
        gpu : int, optional
            GPU id to use. Will override config argument 'device'. A value less than zero indicates CPU.
        """
        config = Config(run_dir / "config.yml")

        # check if a GPU has been specified as command line argument. If yes, overwrite config
        if gpu is not None and gpu >= 0:
            config.device = f"cuda:{gpu}"
        if gpu is not None and gpu < 0:
            config.device = "cpu"
        
        self.start_evaluation(cfg=config, run_dir=run_dir, epoch=epoch, period=period)
    
    
    def evaluate(self) -> None:
        run_dir = Path(self.tgt_config_map['run_dir'])
        
        ##############################################
        #                 EVALUATION                 #
        ##############################################
        self.eval_run(run_dir=run_dir, period="test", gpu=0)
        epoch = self.current_epoch #int(config_map["epochs"])-1
        model_epoch = get_model_epoch(str(epoch))
        self.results = get_test_results(run_dir, model_epoch)
        
    def save_at_epoch(self, model_name, epoch, runs):
        nse_median = get_metric_median(self.results)
        if f"{model_name}-epoch-{epoch}" not in runs:
            runs[f"{model_name}-epoch-{epoch}"] = nse_median
        return runs
    
    def save(self) -> None:
        # Create stream from stdout to collect model results
        stream = StringIO()
        LOGGER.addHandler(logging.StreamHandler(stream))
        
        nse_vals = get_metric_values(self.results)
        nse_median = get_metric_median(self.results)
        nse_mean = get_metric_mean(self.results)
        
        LOGGER.info(f"nse_median:  {nse_median}")
        LOGGER.info(f"nse_mean: {nse_mean}")

        kge_vals = get_metric_values(self.results, metric_type="KGE")
        kge_median = get_metric_median(self.results, metric_type="KGE")
        kge_mean = get_metric_mean(self.results, metric_type="KGE")

        LOGGER.info(f"kge_median: {kge_median}")
        LOGGER.info(f"kge_mean: {kge_mean}")
        
        aNSE_vals = get_metric_values(self.results, metric_type="Alpha-NSE")
        aNSE_median = get_metric_median(self.results, metric_type="Alpha-NSE")
        aNSE_mean = get_metric_mean(self.results, metric_type="Alpha-NSE")

        LOGGER.info(f"alpha_nse_median: {aNSE_median}")
        LOGGER.info(f"alpha_nse_mean: {aNSE_mean}")
        
        
        bNSE_vals = get_metric_values(self.results, metric_type="Beta-NSE")
        bNSE_median = get_metric_median(self.results, metric_type="Beta-NSE")
        bNSE_mean = get_metric_mean(self.results, metric_type="Beta-NSE")

        LOGGER.info(f"beta_nse_median: {bNSE_median}")
        LOGGER.info(f"beta_nse_mean: {bNSE_mean}")
        
        # Number of basins where NSE < 0
        neg_vals = [val for val in nse_vals if val < 0.0]
        LOGGER.info(f"nse_less_than_zero: {len(neg_vals)}")
        
        if self.is_wandb:
            wandb.log({
                "nse_median": nse_median,
                "nse_mean": nse_mean,
                "kge_median": kge_median,
                "kge_mean": kge_mean,
                "alpha_nse_median": aNSE_median,
                "alpha_nse_mean": aNSE_mean,
                "beta_nse_median": bNSE_median,
                "beta_nse_mean": bNSE_mean,
                "nse_less_than_zero": neg_vals
            })
        
        # Create model folder in runs if it does not exist
        model_folder = self.model_name.lower()
        model_path = 'runs/' + model_folder
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # Store results of the models into test log file 
        cur_stream = self.get_stream(stream)
        
        epochs_text = f"epochs: {self.tgt_config_map['epochs']}"
        model_text = f"model: {self.tgt_config_map['model']}" 
        run_dir = '/'.join(self.tgt_config_map['run_dir'].split('/')[4:])
        run_dir_text = f"run_dir: {run_dir}"
        
        cur_stream.insert(0, run_dir_text)
        cur_stream.insert(0, epochs_text)
        cur_stream.insert(0, model_text)
        
        run_dir_ID = '_'.join(self.tgt_config_map['run_dir'].split('/')[-1].split('_')[-2:])
        results_log_path = model_path + f'/{run_dir_ID}_test_results.log'
        
        file_contents = '\n'.join(cur_stream)
        with open(results_log_path, 'w') as fp:
            fp.write(file_contents)
