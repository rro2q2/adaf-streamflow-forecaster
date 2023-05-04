from constants import *
from models.nhmodel import NH_Model
from models.adaf.train import ADAF_Trainer
from omegaconf import DictConfig

def get_data(cfg: DictConfig) -> dict:
    """
    Selects the source and target dataset config paths.
    @param cfg: base config object
    _____
    Returns dictionary of lists of the source and target dataset file paths
    """
    datasets = {"camels-us", "camels-cl", "camels-aus" "camels-gb", "tnc-kenya"}
    data = dict()
    config_dir = 'configs/'
    if cfg.src_data.lower() not in datasets:
        SystemExit("\nError: Dataset not found. Please select correct source dataset.\n")
    # Set source dataset directory
    src_data = cfg.src_data.lower()
    data["source"] = [config_dir, f"{src_data}/", None, f"-{src_data}-static.yml"]
    
    if cfg.tgt_data.lower() not in datasets:
        SystemExit("\nError. Please select correct target dataset.\n")
    # Set target dataset directory
    tgt_data = cfg.tgt_data.lower()
    data["target"] = [config_dir, f"{tgt_data}/", None, f"-{tgt_data}-static.yml"] 
    
    return data

def get_model(cfg: DictConfig):
    """
    Selects the transfer learning model to use for model training.
    @param cfg: base config object
    @param data: tuple of source and target dataset config paths
    _____
    Returns Any [nn.Module or BaseModule] type of model
    """
    data = get_data(cfg)
    # Check if model exists
    if not cfg.model:
        return None
    data["source"][2] = data["target"][2] = cfg.model.lower()
    model = None
    if cfg.model.lower() == "cudalstm":
        model = NH_Model(cfg=cfg,
                         src_config_path=''.join(data["source"]), 
                         tar_config_path=''.join(data["target"]))
    elif cfg.model.lower() == "gru":
        model = NH_Model(cfg=cfg,
                         src_config_path=''.join(data["source"]), 
                         tar_config_path=''.join(data["target"]))
    elif cfg.model.lower() == "adaf":
        model = ADAF_Trainer(cfg=cfg,
                             src_cfg_path=''.join(data["source"]),
                             tgt_cfg_path=''.join(data["target"]))
    else:
        SystemExit("\nError. Please select correct transfer learning model.\n")
    
    return model
