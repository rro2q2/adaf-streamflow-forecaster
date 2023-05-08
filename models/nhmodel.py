import torch
import logging
from pathlib import Path
from typing import Dict, Any
from io import StringIO
import shutil
from omegaconf import DictConfig

from constants import *
from utils import get_test_results, get_metric_values, get_metric_median, get_metric_mean, get_model_epoch, get_config_map

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.nh_run import start_run, finetune
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.gru import GRU
from neuralhydrology.training.basetrainer import BaseTrainer, LOGGER
from neuralhydrology.evaluation import get_tester


class NH_Model():
    """
    Class of NeuralHydrology models referenced from (link: https://github.com/neuralhydrology/neuralhydrology).
    
    @param cfg: DictConfig      - Base configuration
    @param src_config_path: str - Configuration path of source dataset
    @param tar_config_path: str - Configuration path of target dataset
    ____
    returns None
    """
    def __init__(self, cfg: DictConfig, src_config_path: str, tar_config_path: str) -> None:
        super(NH_Model, self).__init__()
        self.cfg = cfg
        self.src_config_path = src_config_path
        self.tar_config_path = tar_config_path
        
        self.src_config_map = None
        self.tar_config_map = None
        self.results = None
        
        
    def get_stream(self, stream):
        stream.seek(0)
        return stream.read().strip().split('\n')
        
    def run_train(self) -> Dict[str, Any]:
        """
        Runs training and validation for given model. Stores ouput of the train/val run in
        log file.
        
        ______
        returns Dict[str, Any]
        """
        config_file = Path(self.src_config_path)

        # Create stream through stdout and store config info of current model
        stream = StringIO()
        LOGGER.addHandler(logging.StreamHandler(stream))
        
        # Start training and validation
        start_run(config_file=config_file)
        
        cur_stream = self.get_stream(stream)
        
        # Convert the stream of outputs of model configurations into a Dict type
        config_map = get_config_map(cur_stream)
        
        # Store file contents from into a log file
        model_log_path = config_map['run_dir'] + '/model_train_info.log'
        
        file_contents = '\n'.join(cur_stream)
        with open(model_log_path, 'w') as fp:
            fp.write(file_contents)
                
        return config_map
    
    def transfer_weights(self, run_dir, transfer_model_config, cfg):
        """
        Performs the transfer function between the pre-trained model and new model by
        transferring the weights of the pre-trained model and reinitializing a new output
        layer in the new model.
        @param run_dir: Run directory of the pre-trained model
        @param transfer_model_config: Config type of target model
        @param cfg: Config type of the pre-trained source model
        ______
        return TBD
        """

        if self.cfg.model.lower() == "cudalstm":
            transfer_model = CudaLSTM(cfg=transfer_model_config)
        elif self.cfg.model.lower() == "gru":
            transfer_model = GRU(cfg=transfer_model_config)

        # get the model epoch
        model_epoch = get_model_epoch(str(cfg.epochs))
        model_path_ext = model_epoch + '.pt'
        
        # load the trained weights into the new model. 
        model_path = run_dir / model_path_ext
        model_weights = torch.load(str(model_path))

        transfer_model.load_state_dict(model_weights) # set the new model's weights to the values loaded from file

        # Replace head of the model
        # This would be helpful of reinitializing the head and adjust for target dataset distribution 
        transfer_model.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=len(cfg.target_variables))

        print("Type of model:")
        print(type(transfer_model))
        return transfer_model
    
    
    def run_transfer_learning_train(self, trainer, model) -> None:
        """
        Trains and validates the transfer learning model.
        @param trainer: BaseTrainer instance of the the TL model
        @param model: Model state (weights) of the pre-trained model
        ______
        return None
        """
        trainer.model = model
        trainer.device = "cuda:0"
        trainer.initialize_training()
        trainer.train_and_validate()
        
    def run_finetune(self, run_dir, ft_config_path) -> Dict[str, Any]:
        """
        Runs finetune function after training transfer learning model.
        @param run_dir: The run directory of the transfer learning model
        @param ft_config_path: Specified configuration path of the finetuned TL model
        ______
        return Dict[str, Any]
        """
        # Add the path to the pre-trained model to the finetune config
        with open(ft_config_path, "a") as fp:
            fp.write(f"base_run_dir: {run_dir.absolute()}")
        
        # Create stream through stdout and store config info of current model
        stream = StringIO()
        LOGGER.addHandler(logging.StreamHandler(stream))
        
        finetune(Path(ft_config_path))
        
        cur_stream = self.get_stream(stream)
        
        # Convert the stream of outputs of model configurations into a Dict type
        config_map = get_config_map(cur_stream)
        
        # Store file contents from into a log file
        model_log_path = config_map['run_dir'] + '/model_transfer_info.log'
        
        file_contents = '\n'.join(cur_stream)
        with open(model_log_path, 'w') as fp:
            fp.write(file_contents)
            
        return config_map
        
        
    def train(self) -> None:
        """
        Performs transfer learning with finetuning using a pretrained CAMELS-US  model.
        ______
        Return: None
        """
        ########## 1. Load pretrained model ##########
        self.src_config_map = self.run_train()

        ########################################
        ######## FOR TRANSFER LEARNING #########
        ########################################
        # Create source config object from .yml file
        src_config_file = Path(self.src_config_map['run_dir'] + '/config.yml')
        src_model_cfg = Config(src_config_file)
        # Get run directory from source config map
        src_run_dir = Path(self.src_config_map['run_dir'])
        
        ########## 3. Transfer model to target dataset ##########
        tar_config_file = Path(self.tar_config_path)
        # Create config for target domain dataset
        transfer_model_cfg = Config(tar_config_file)
        
        # Transfer weights to new model 
        model = self.transfer_weights(src_run_dir, transfer_model_cfg, src_model_cfg)
        
        print("############ START TRANSFER LEARNING ############")
        trainer = BaseTrainer(cfg=transfer_model_cfg)
        # Perform tranfer learning
        self.run_transfer_learning_train(trainer, model)
        
        # Store run_dir from transfer learning to use for finetuning and evaluation
        run_name = '/'.join(Path(transfer_model_cfg.run_dir).stem.split('/')[-2:])
        tar_run_dir = Path('runs/' + run_name)
    
        print("############ START FINETUNING ############")
        # Copy finetune config path into the target (transfer learning) run directory directory
        
        shutil.copy2(f'configs/{self.cfg.tgt_data.lower()}/finetune.yml', tar_run_dir)
        ft_config_path = tar_run_dir / Path('finetune.yml')
        
        self.tar_config_map = self.run_finetune(tar_run_dir, ft_config_path)
        
        
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
        # FOR SOURCE AND TARGET
        config_map = self.tar_config_map 
        run_dir = Path(self.tar_config_map['run_dir'])
        
        ##############################################
        #                 EVALUATION                 #
        ##############################################
        self.eval_run(run_dir=run_dir, period="test")
        model_epoch = get_model_epoch(config_map["epochs"])
        self.results = get_test_results(run_dir, model_epoch)
        print(self.results)
        
    def save(self) -> None:  
        # Create stream from stdout to collect model results
        stream = StringIO()
        LOGGER.addHandler(logging.StreamHandler(stream))
        
        nse_vals = get_metric_values(self.results)
        nse_median = get_metric_median(self.results)
        nse_mean = get_metric_mean(self.results)
        
        LOGGER.info(f"NSE median:  {nse_median}")
        LOGGER.info(f"NSE mean: {nse_mean}")

        kge_vals = get_metric_values(self.results, metric_type="KGE")
        kge_median = get_metric_median(self.results, metric_type="KGE")
        kge_mean = get_metric_mean(self.results, metric_type="KGE")

        LOGGER.info(f"KGE median: {kge_median}")
        LOGGER.info(f"KGE mean: {kge_mean}")
        
        aNSE_vals = get_metric_values(self.results, metric_type="Alpha-NSE")
        aNSE_median = get_metric_median(self.results, metric_type="Alpha-NSE")
        aNSE_mean = get_metric_mean(self.results, metric_type="Alpha-NSE")

        LOGGER.info(f"Alpha-NSE median: {aNSE_median}")
        LOGGER.info(f"Alpha-NSE mean: {aNSE_mean}")
        
        
        bNSE_vals = get_metric_values(self.results, metric_type="Beta-NSE")
        bNSE_median = get_metric_median(self.results, metric_type="Beta-NSE")
        bNSE_mean = get_metric_mean(self.results, metric_type="Beta-NSE")

        LOGGER.info(f"Beta-NSE median: {bNSE_median}")
        LOGGER.info(f"Beta-NSE mean: {bNSE_mean}")
        
        # Number of basins where NSE < 0
        neg_vals = [val for val in nse_vals if val < 0.0]
        LOGGER.info(f"No. of basins with NSE < 0: {len(neg_vals)}")
        
        # Create model folder in runs if not exists
        model_folder = self.cfg.model.lower() + '_tl'
        model_path = 'runs/' + model_folder
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # Store results of the models into test log file 
        cur_stream = self.get_stream(stream)
        
        epochs_text = f"epochs: {self.tar_config_map['epochs']}"
        model_text = f"model: {self.tar_config_map['model']}"
        run_dir = '/'.join(self.tar_config_map['run_dir'].split('/')[4:])
        run_dir_text = f"run_dir: {run_dir}"
        
        cur_stream.insert(0, run_dir_text)
        cur_stream.insert(0, epochs_text)
        cur_stream.insert(0, model_text)
        
        run_dir_ID = '_'.join(self.tar_config_map['run_dir'].split('/')[-1].split('_')[-2:])
        results_log_path = model_path + f'/{run_dir_ID}_test_results.log'
        
        file_contents = '\n'.join(cur_stream)
        with open(results_log_path, 'w') as fp:
             fp.write(file_contents)
