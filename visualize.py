import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pickle
import ast
from pathlib import Path

from omegaconf import DictConfig
import hydra

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
        if int(epoch) == 0:
            epoch = '1'
        epoch_num = '00' + epoch
    elif int(epoch) < 100:
        epoch_num = '0' + epoch
    else:
        epoch_num = epoch
    return epoch_base + epoch_num

def parse_file(file):
    results_map = {}
    with open(file) as fp:
        contents = fp.readlines()
        for line in contents:
            if line.replace(":", "") != line: # check for colons
                items = line.split(':', 1) 
                key, value = items[0].strip(), items[1].strip()
                if key.replace(" ", "") == key: # check for whitespaces
                    if '{' in value or '[' in value:
                        value = ast.literal_eval(value)
                    results_map[key] = value
    return results_map

def get_predictions(files):
    if not files:
        SystemError("No test files were found.\n")
    # Get Station ID
    with open(files[0], 'r') as fp:
        for line in fp:
            if line.startswith("station_id"):
                station_id = line.split(':')[1].strip()
            else:
                SystemError("No station ID found.\n")
                
    # Get test results for each model
    model_res = {}
    for i in range(len(files)):
        results_map = parse_file(files[i])
        epoch = str(int(results_map["epochs"]) - 1)     
        epoch = get_model_epoch(epoch)
        run_dir = results_map['run_dir'].split('/')
        run_dir.insert(1, 'final')
        run_dir_final = '/'.join(run_dir)
        with open(f"{run_dir_final}/test/{epoch}/test_results.p", "rb") as fp:
            test_results = pickle.load(fp)
            model_name = files[i].split('/')[1]
            model_res[model_name] = test_results
                
    return model_res, station_id


def plot_predictions(model_res: dict, station_id):
    # Plot prediction  
    sns.set()
    qobs = model_res["adaf_attn_adv"][str(station_id)]['1D']['xr']['QObs(mm/d)_obs']
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(qobs['date'], qobs, label="Observed")
    i = 0
    for m in model_res:
        qsim = model_res[m][str(station_id)]['1D']['xr']['QObs(mm/d)_sim']
        if m == 'cudalstm_tl':
            label = 'LSTM-TL'
        elif m == "gru_tl":
            label = "GRU-TL"
        else:
            label = "Our approach"  
        ax.plot(qsim['date'], qsim, alpha=0.7, label=label)
        i += 1
        
    ax.grid(True)
    ax.set_ylabel("Streamflow (mm/d)")
    ax.set_xlabel("Test Period")
    #ax.set_title(f"Test Period for Sation # {station}")
    ax.legend(title='Models')
    ax.legend(loc="upper right")
    # Save plot
    fig.autofmt_xdate()
    fig.savefig('figs/compare_predictions.png')


def get_ablation(folders, epoch_list):
    mean_scores = {f: [] for f in folders}
    std_scores = {f: [] for f in folders}
    for f in folders:
        score_list = []
        for i in range(len(epoch_list)):
            score_list.append([])
        
        files = os.listdir(f"runs/{f}")
        for ff in files:
            fname = f"runs/{f}/{ff}"
            results_map = parse_file(fname)
            for i in range(len(epoch_list)):
                score_list[i].append(float(results_map[f"{f}-epoch-{epoch_list[i]}"]))
        # Process each model results to get mean and std across the evaluated markers
        for i in range(len(epoch_list)):
            mean_scores[f].append(np.mean(score_list[i]))
            std_scores[f].append(np.std(score_list[i]))
            
    return mean_scores, std_scores

def plot_ablation(folders, mean_scores, std_scores, epoch_list):
    sns.set()
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['b', 'r', 'g']
    legend = []
    model_name = []
    i = 0
    for model in folders:
        mean = np.array(mean_scores[model])
        std = np.array(std_scores[model])
        line = ax.plot(epoch_list, mean, f"{colors[i]}-", label=folders[model])
        fill = ax.fill_between(epoch_list, mean - std, mean + std, color=colors[i], alpha=0.2)
        legend.append((line[0], fill))
        model_name.append(folders[model])
        i+= 1
    
    # ax.grid(True)
    ax.set_ylabel("Nash-Sutcliffe Efficiency (NSE)")
    ax.set_xlabel("Epochs")
    #ax.set_title(f"Test Period for Sation # {station}")
    ax.legend(legend, model_name, title='Models', loc="lower right")
    # Save plot
    # fig.autofmt_xdate()
    fig.savefig('figs/ablation.png')

@hydra.main(version_base="1.2.0", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    epoch_list = [1, 5, 10, 20, 50, 100]
    if cfg.viz == "prediction":
        # sf_pred_files - manually storing the logs for each model run
        # runs/adaf_attn_adv/1304_043806_test_results.log
        sf_pred_files = ['runs/adaf_attn_adv/1304_043806_test_results.log', 'runs/cudalstm_tl/1204_134644_test_results.log',
                         'runs/gru_tl/1204_035500_test_results.log']
        model_results, station_id = get_predictions(sf_pred_files)
        plot_predictions(model_results, station_id)
    elif cfg.viz == "ablation":
        # TODO work on ablation
        # model_folders = ['adaf_attn_adv', 'adaf_attn', 'adaf_seq2seq']
        model_folders = {'adaf_attn_adv': 'Our approach', 'seq2seq': 'Seq2Seq-TL'}
        mean_scores, std_scores = get_ablation(model_folders, epoch_list)
        plot_ablation(model_folders, mean_scores, std_scores, epoch_list)
    else:
        SystemError("Incorrect input")


if __name__ == "__main__":
    main()
