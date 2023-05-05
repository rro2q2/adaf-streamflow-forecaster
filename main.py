from models import get_model
from constants import *
from neuralhydrology.neuralhydrology.modelzoo.cudalstm import CudaLSTM
import torch

from omegaconf import DictConfig
import hydra

@hydra.main(version_base="1.2.0", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # 1. Load transfer learning model
    model = get_model(cfg)
    
    # 2. Train the model
    print("######## Start train ########")
    model.train()
    
    # Empty cuda cache 
    torch.cuda.empty_cache()
    
    # 3. Evaluate the model
    print("######## Start evaluation ########")
    model.evaluate()
    
    # Empty cuda cache 
    torch.cuda.empty_cache()
    
    # 4. Save the model results
    print("######## Save ########")
    model.save()
    
     
if __name__ == '__main__':
    main()
