# Transfer learning to improve streamflow forecasts in data sparse regions
This repository implements the transfer learning algorithm in the pre-print manuscript "Transfer learning to improve streamflow forecasts in data sparse regions".

## Prerequisites
### Neuralhydrology Project
The Python enviroment that uses transfer learning for hydrological time series modeling for streamflow basins is based on [Neuralhydrology GitHub](https://github.com/neuralhydrology/neuralhydrology) repository. In order to run the experiments, you will need to first follow the instillation steps of the following links. For installation purposes, you will need to do a "Non-Editable" installation in your Python environment, as this will allow you to add the model and dataset we have included. Please stop when you reach "Training the Model" in the [Neuralhydrology Documentation](https://neuralhydrology.readthedocs.io) link. Please see the following links below for installing the Neuralhydrology repoository:

- Neuralhydrology Repo: [neuralhydrology/neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) 
- Neuralhydrology Documentation: [neuralhydrology.readthedocs.io](https://neuralhydrology.readthedocs.io)

### Package Requirements TODO
- Python: 3.7+
- We recommend to use Anaconda/Miniconda. With one of the two installed, a dedicated environment with all requirements will be added to the project

### Hardware Requirements
For improving model training and evaluation, we recommend using a multiple GPU cores. The CPU cores may take a long time too training or lead to an error. For future releases of this project, we will demonstrate how to excute this project without the need of a local GPU desktop computer.

## Project Contents
```
transfer-learning-aaai21
    data_dir
    docs
    environments
    examples
    expir_runs
    neuralhydrology
    output
    pretrained_models
    runs
    test
    contsants.py
    main.py
    readthedocs.yml
    setup.py
    train.py
    transferlearning.py
    utils.py
    visualizations.py
```
- `data_dir` - a directory that contains the dataset for both the CAMELS-US and Kenya Hydrology dataset. This includes meteorological forcings data as well and static catchment attributes. The CAMELS-US dataset can be found on [CAMELS US (NCAR)](https://ral.ucar.edu/solutions/products/camels)
- `environments` - a directory that holds all Python high-performance computing configurations. For faster data processing and model training, we recommend using CUDA 10.2
- `examples` - contains tutorial examples of adding datasets and model experimentation. We've included the `CAMELS-US` and `TNC-Kenya` folders that will be used for model (pre-)training and testing
- `expir_runs` - entails the files in which the final model stores it's results. We formatted them in JSON file to be retrieved for displaying and visualizing the results
- `neuralhydrology` - main neuralhydrology directory that contains the set of dataset, model, training, and evaluation utility functions for the basis of our model
- `output` - a directory that displays model outputs based on our `visualization.py` file
- `pretrained_models` - contains the pre-trained LSTM models used for transfer learning, with and without static catchment attributes
- `test` - a directory that has the testing utility functions and configurations for the final model
- `constants.py` - a file that holds the necessary Python packages and paths for models, configuration, and results
- `main.py` - main file for pre-training, training, and evaluation
- `readthedocs.yml` - YML file for install the required Python packages
- `train.py` - for model training
- `transferlearning.py` - a file for that performs transferlearning from the pre-trained model to a new LSTM model
- `utils.py` - utility functions for model training and validation
- `visualizations.py` - a file that contains functions for visualizing the final model results

## Getting Started
### Kenya Hydrology Dataset
For storage purposes, we have pubically uploaded our dataset on Kaggle. Please download the [Kenya Hydrology Dataset](https://www.kaggle.com/rolandoruche/transferlearningaaai21) and move the extracted contents into the `data_dir` directory.

### Train the LSTM Model
We present multiple options for training and validation the LSTM model. The deep learning LSTM to can train over the Kenya dataset or the CAMELS US dataset. Static descriptors can be used as additional input with the meteorological forcing data as more features to for the model.  To reproduce the results of this project, we suggest you train on the CAMELS_US dataset to develop the pre-trained LSTM model for transfer learning.

To run the training module, using the following command:

`python train.py --static <str> --dataset <str>`

- `--static <str>`: Allowing static descriptors to be used as input to the pre-trained/standard LSTM model by replacing `<str>` with "yes" or "y" (**not** case sensitive)
- `--dataset <str>`: Select which dataset to train the LSTM model over by replacing `<str>` with "camels_us" or "tnc_kenya" (**not** case sensitive)

### Perform Transfer Learning
Once the training/validation period is finished, transfer learning can be performed based on the pre-trained LSTM model. It is important to note that the transfer learning module cannot be ran without running the train module first.

To run the transfer learning module, using the following command:

`python transferlearning.py`


### Run the Evaluation Module
The evaluation module presents two options for testing the LSTM model during the hold-off set. The model can either be evaluated on the training/validation module or the transferlearning module.


To run the evaluation module, using the following command:

`python evaluation.py --module <str>`

- `--module <str>`: Select which module to evaluate using the test dataset by replacing `<str>` with "train" or "transfer_learning" (**not** case sensitive)
    
#### Storing Expirement Runs
These experiment runs will be stored in JSON files where they will be retrieved for reporting the performance scores and visualization. These files are unique based on the specifications of the experiment run and are stored in expir_runs.

### Reporting Results and Visualization
For visualizing the results, we allow the user to select both the model results from the JSON file and the type of visualization. Herein, we present two types of visualizations:
- prediction: shows the predcitive performance of the model using the Nash-Sutcliffe model efficiency (NSE) estimation method. This compares between the target discharge (streamflow) with the predictive model discharge.
- colormap: demonstrates the performance of the model over the Kenyan river stations. A higher value colored in red indicates a high NSE score, and a low color in blue represents a low NSE score. 

You can run the model based on the options using the below command:

`python visualizations.py --model <str> --viz <str>`

The options for running the model are as follows:
- `--model <str>`: Chose the model to visualize. LSTM with transfer learning (lstm_tl), LSTM with transfer learning and static catchment attributes (lstm_tl_sca), LSTM with static catchment attributes (lstm_sca), or LSTM with no specifications (lstm). `<str>` is replace with one of these options.
- `--viz <str>`: Chose the type of visualization. Prediction (prediction) or colormap (colormap). `<str>` is replace with one of these options.

The final visualization is store in the output folder.

### Output
As mentioned in the previous subsection, we have developed two types of visualizations to show the performance of our model. The first is a streamflow prediction graph, which shows the predictive skill of our model at particular basins. The second is a colormap, which highlights the NSE values of each basin accross a fixed number of trial runs (ensembles).

The output of the predicitive streamflow visualization is shown below:

<p align="center">
  <img src="output/compare_predictions.png" />
</p>

This example compares the streamflow prediction between a standard LSTM and an LSTM model with transfer learning.

The output of the colormap visualization is shown below:

<p align="center">
  <img src="output/lstm_tl_colormap.png" />
</p>


This demonstrates the performance of the fine-tuned LSTM model that uses TL and its response to generalize over sparse data on 30 runs.

**Note:** We plan to add more documentation/examples in the next coming weeks
- Bug reports/Feature requests [https://github.com/neuralhydrology/neuralhydrology/issues](https://github.com/neuralhydrology/neuralhydrology/issues)

## Contact
- Primary contact: Fearghal O'Donncha (feardonn(at)ie(dot)ibm(dot)com)
- Other contacts: Roland Oruche (rro2q2(at)umsystem(dot)edu)

