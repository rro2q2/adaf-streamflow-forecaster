import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from yellowbrick.text import TSNEVisualizer

from neuralhydrology.neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.neuralhydrology.utils.config import Config
import neuralhydrology.neuralhydrology.training.loss as loss
from neuralhydrology.neuralhydrology.utils.logging_utils import setup_logging
from neuralhydrology.neuralhydrology.training import get_loss_obj, get_regularization_obj
from neuralhydrology.neuralhydrology.evaluation import get_tester
from neuralhydrology.neuralhydrology.evaluation.tester import BaseTester

def _get_folder_structure(domain_cfg: Config, logger: logging):
        _create_folder_structure(domain_cfg)
        setup_logging(str(domain_cfg.run_dir / "output.log"))
        logger.info(f"### Folder structure created at {domain_cfg.run_dir}")

        logger.info(f"### Run configurations for {domain_cfg.experiment_name}")
        for key, val in domain_cfg.as_dict().items():
            logger.info(f"{key}: {val}")
        
# Import functions from Neuralhydrology for training initialization
def _get_loss_obj(domain_cfg: Config) -> loss.BaseLoss:
    return get_loss_obj(cfg=domain_cfg)

def _set_regularization(criterion, domain_cfg: Config):
    criterion.set_regularization_terms(get_regularization_obj(cfg=domain_cfg))
    
def _get_dataset(domain_cfg: Config, period: str, _scaler) -> BaseDataset:
    return get_dataset(cfg=domain_cfg, period=period, is_train=True, scaler=_scaler)

def _get_data_loader(domain_cfg: Config, ds: BaseDataset) -> torch.utils.data.DataLoader:
    return DataLoader(ds, batch_size=domain_cfg.batch_size, shuffle=True, num_workers=domain_cfg.num_workers, drop_last=True)

def _get_tester(domain_cfg: Config) -> BaseTester:
    return get_tester(cfg=domain_cfg, run_dir=domain_cfg.run_dir, period="validation", init_model=False)

def _create_folder_structure(domain_cfg):
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    second = f"{now.second}".zfill(2)
    run_name = f'{domain_cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

    # if no directory for the runs is specified, a 'runs' folder will be created in the current working dir
    if domain_cfg.run_dir is None:
        domain_cfg.run_dir = Path().cwd() / "runs" / run_name
    else:
        domain_cfg.run_dir = domain_cfg.run_dir / run_name

    # create folder + necessary subfolder
    if not domain_cfg.run_dir.is_dir():
        domain_cfg.train_dir = domain_cfg.run_dir / "train_data"
        domain_cfg.train_dir.mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {domain_cfg.run_dir}")
    if domain_cfg.log_n_figures is not None:
        domain_cfg.img_log_dir = domain_cfg.run_dir / "img_log"
        domain_cfg.img_log_dir.mkdir(parents=True)
        
def data_to_device(data, device):
    for key in data.keys():
        data[key] = data[key].to(device)

    return data

def get_stream(stream):
    stream.seek(0)
    return stream.read().strip().split('\n')

def _save_weights_and_optimizer(model, epoch: int, domain_cfg: Config):
    weight_path = domain_cfg.run_dir / f"model_epoch{epoch:03d}.pt"
    torch.save(model.state_dict(), str(weight_path))
    
def get_station_id(results):
        nse_median = np.median(results["NSE"].values)
        station_id = results.loc[results['NSE'] == nse_median]["basin"].values[0]
        return station_id
