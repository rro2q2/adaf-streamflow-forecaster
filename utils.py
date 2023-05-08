from constants import *
# from neuralhydrology.utils.config import Config
# from neuralhydrology.nh_run import start_run, eval_run, finetune
# from neuralhydrology.modelzoo.cudalstm import CudaLSTM
# from neuralhydrology.datasetzoo import get_dataset
# from neuralhydrology.training.basetrainer import BaseTrainer

def get_model_epoch(epoch) -> str:
    """
    Gets epoch from model.
    @param epoch: Number of epochs model uses for training.
    ______
    return str
    """
    epoch_base = 'model_epoch'
    epoch_num = ''
    if int(epoch) < 10:
        epoch_num = '00' + epoch
    elif int(epoch) < 100:
        epoch_num = '0' + epoch
    else:
        epoch_num = epoch
    return epoch_base + epoch_num


def get_validation_results(run_dir, model_epoch):
    """
    Gets validation results of the LSTM model.
    @param run_dir: Run directory of the chosen model
    @param model_epoch: Last epoch of the model
    ______
    return results
    """
    validation_path_ext = "validation/" + model_epoch + "/validation_results.p"
    with open(run_dir / validation_path_ext, "rb") as fp:
        results = pickle.load(fp)
    return results

def get_test_results(run_dir, model_epoch):
    """
    Gets test results of the LSTM model.
    @param run_dir: Run directory of the chosen model
    @param model_epoch: Lstm epoch of the model
    ______
    return results
    """
    test_path_ext = "test/" + model_epoch + "/test_results.p"
    with open(run_dir / test_path_ext, "rb") as fp:
        results = pickle.load(fp)
    return results


def clear_config_info_log(filename) -> None:
    """
    Clears the configuration output log of the model.
    @param filename: File name of the model output (train or transfer learning)
    ______
    return None
    """
    # Remove file contents   
    with open(filename, "w"):
        pass
        
def get_config_map(stream: list) -> dict:
    """
    Gets configuration of the model from current stream.
    @param stream: Current stream of the model output
    ______
    return config_map: dict
    """
    config_map = dict()
    for line in stream:
        if line.startswith('run_dir:'):
            config_map["run_dir"] = line.split(':')[1].strip()
        elif line.startswith('epochs:'):
            config_map["epochs"] = line.split(':')[1].strip()
        elif line.startswith("model:"):
            config_map["model"] = line.split(':')[1].strip()

    return config_map


def get_metric_values(results, metric_type="NSE"):
    """
    Get the metric values of all the basins; values from the last epoch
    @param results: form model evaluation
    ______
    Return metric_values
    """
    return [v['1D'][metric_type] for v in results.values() if metric_type in v['1D'].keys()]

def get_metric_median(results, metric_type="NSE"):
    """
    Gets the metric median of all the basins; values from the last epoch
    @param results: from model evaluation
    ______
    Return metric_median
    """
    return np.median([v['1D'][metric_type] for v in results.values() if metric_type in v['1D'].keys()])

def get_metric_mean(results, metric_type="NSE"):
    """
    Gets the metric mean of all the basins; values from the last epoch
    @param results: from model evaluation
    ______
    Return metric_mean
    """
    return np.mean([v['1D'][metric_type] for v in results.values() if metric_type in v['1D'].keys()])


def print_results(results, metric_type="NSE"):
    """
    Prints the  median, mean, and metric values of all the basins
    @param results: Results from model evaluation
    ______
    Return None
    """
    print(f"{metric_type.upper()} values: ", get_metric_values(results))
    print(f"{metric_type.upper()} median: ", get_metric_median(results))
    print(f"{metric_type.upper()} mean: ", get_metric_mean(results))
