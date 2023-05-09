


This is the official repository for the ICLR 2023 workshop paper on Climate Change AI titled [Attention-based Domain Adaptation Forecasting of Streamflow in Data Sparse Regions](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/iclr2023/14/paper.pdf).

## Prerequisites
### Python environment
- Python: 3.8+
- We recommend to use Anaconda/Miniconda. With one of the two installed, a dedicated environment with all requirements will be added to the project.

### Hardware Requirements
For improving model training and evaluation, we recommend using a multiple GPU cores. The CPU cores may take a long time too training or lead to an error.

### Installing Neuralhydrology Project
Add repo as a submodule
```
git submodule add https://github.com/neuralhydrology/neuralhydrology.git 
```

Implement own models and datasets in Neuralhydrology
```
cd neuralhydrology
pip install -e .
```

### Download dataset
Download dataset from [Kaggle](https://www.kaggle.com/datasets/rolandoruche/camels). Add downloaded files to root directory as ` data_dir `.


## How to run
The following commands show how to run our proposed model and baselines:

ADAF: 
```
python main.py
```

LSTM-TL:
```
python main.py model=cudalstm
```

GRU-TL:
```
python main.py model=gru
```

## Citing this work
If you would like to cite this work, please use the BibTeX syntax shown below:
```
@article{oruche2023attention,
  title={Attention-based Domain Adaption Forecasting of Streamflow in Data Sparse Regions},
  author={Oruche, Roland and O'Donncha, Fearghal},
  journal={arXiv preprint arXiv:2302.05386},
  year={2023}
}
```
## Contact
- Primary contact: Roland Oruche (roruche23@gmail.com)
- Other contacts: Fearghal O'Donncha (feardonn@ie.ibm.com)

