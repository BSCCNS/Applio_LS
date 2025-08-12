import glob
import joblib
import os 
import json

import time

import pandas as pd
#from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from phonetics import utils as u
#from phonetics import phone_info as ph_i 
from phonetics import plots as plots
from phonetics import metrics as ph_metrics


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#############################################################

parser = argparse.ArgumentParser(
                    prog='LS',
                    description='Extract LS',
                    epilog='Ask me for help')

# Define named arguments
parser.add_argument('--parfile', type=str, required=True, help="Path to the parameter file")
#parser.add_argument('--stage', type=str, required=True, help="Optional argument")

def read_param_dict(parser):
    args = parser.parse_args()
    json_file = open(args.parfile)
    param_dict = json.load(json_file)
    json_file.close()
    return param_dict

param_dict = read_param_dict(parser)


#DATA_SET = param_dict["dataset"] #'GTSinger_ES'
#DATA_SET_TP = param_dict["dataset_tp"] #'gt'
#EXCLUDE_PHONES = param_dict["exclude_phones"] #None #['<AP>']
#tp_algn = param_dict["text_grid"]  #'text_grid'
#K_MI = param_dict["K_MI"] # 50

# DATA_SET = 'songs'
# DATA_SET_TP = None 
# EXCLUDE_PHONES =  None 
# tp_algn = 'lab' 


#############################################################

def boiler_plate(param_dict):

    input_path = param_dict["input_path"]
    experiment = param_dict["experiment"]
    
    tp_algn = param_dict['tp_algn']

    if tp_algn == 'text_grid':
        algn_paths = glob.glob(f'{input_path}/TextGrid/*.TextGrid')
    elif tp_algn == 'lab':
        algn_paths = glob.glob(f'{input_path}/lab/*.lab')

    experiment_folder = f'experiments/{experiment}'
    folder_dict = {'experiment_folder': experiment_folder}

    if param_dict.get("output_feat_768d", False):
        folder_dict.update({'feat_768d_folder': f'{experiment_folder}/feat_768d'})

    if param_dict.get("projection_2d", True):
        folder_dict.update(
            {'feat_2d_folder': f'{experiment_folder}/feat_2d',
              'plots_folder' : f'{experiment_folder}/plots'})
        
    print(f'folders to be created : {list(folder_dict.values())}')
    
    for fo in list(folder_dict.values()):
        os.makedirs(fo, exist_ok=True)

    with open(f"{experiment_folder}/metadata.json", "w") as outfile: 
        json.dump(param_dict, outfile, indent=2)

    return algn_paths, folder_dict 

def make_df_annotated(layer, param_dict):
    print(f'-------- df annotated')

    input_path = param_dict["input_path"]
    tp_algn = param_dict["tp_algn"]
    dataset_tp = param_dict["dataset_tp"]

    feat_paths = glob.glob(f'{input_path}/feat/layer_{layer}/*.csv')

    df_anotated = u.make_anotated_feat_df(feat_paths, 
                                          algn_paths, 
                                          tp_algn = tp_algn,
                                          dataset = dataset_tp)
    
    if param_dict.get("output_feat_768d", True):
        df_anotated.to_csv(f'{folder_dict["feat_768d_folder"]}/feat_768d_layer_{layer}.csv')

    return df_anotated

def make_df_projected_annotated_2d(df_anotated, param_dict):
    
    exclude_phones = param_dict['exclude_phones_plot']
    print(f'Excluding phones {exclude_phones} from plot')

    metric = param_dict.get('umap_metric', 'euclidean')
    print(f'UMAP projection using metric {metric}')

    normalize_vectors = param_dict.get('umap_normalize_vectors', False)
    print(f'UMAP projection normalize vectors {normalize_vectors}')
    
    print(f'-------- umap')
    umap2 = u.train_umap(
        df_anotated,
        exclude_phones = exclude_phones,
        n_components=2, 
        n_neighbors=100, 
        min_dist=0.1,
        metric = metric,
        normalize_vectors = normalize_vectors,
        save_model = False,
        folder = None)
        
    df_proj_anotated = u.make_proj_anotated_feat_df(df_anotated, 
                                                    umap2,
                                                    save_df = False,
                                                    folder = None)
    
    df_proj_anotated.to_csv(f'{folder_dict["feat_2d_folder"]}/feat_2d_layer_{layer}.csv')

    return df_proj_anotated

def make_plot(df_proj_anotated):
    print(f'-------- plot')
    my_phones = [k for k in df_proj_anotated['phone_base'].value_counts().keys() if k != 'AP']
    plots.make_tagged_LS_plot(df_proj_anotated,
            phones = my_phones,
            alpha = 0.25, 
            s = 0.1,
            show_global=True)
    plt.savefig(f'{folder_dict["plots_folder"]}/LS_layer_{layer}')


#############################################################
#### Pipeline
#############################################################

T0 = time.time()

algn_paths, folder_dict = boiler_plate(param_dict)
metric_dict = {}

for layer in range(1,13):

    t0 = time.time()
    print(f'-------- Working on layer {layer}')

    df_anotated = make_df_annotated(layer, param_dict) 
    metric_dict[layer] = ph_metrics.compute_metric_for_layer(df_anotated, param_dict)

    if param_dict.get("projection_2d", True):
        df_proj_anotated = make_df_projected_annotated_2d(df_anotated, param_dict) 
        make_plot(df_proj_anotated)
    else:
        print('Skipping 2d projection and plots')
    
    t1 = time.time()
    dt = t1 - t0
    print(f'------------- Time for layer {layer}: {dt}')

exp_folder = folder_dict["experiment_folder"]
df_metric = ph_metrics.make_df_metric(metric_dict)
df_metric.to_csv(f'{exp_folder}/metric_layers.csv')

T1 = time.time()    
DT = T1 - T0

print(f'------------- Total Time: {DT}')