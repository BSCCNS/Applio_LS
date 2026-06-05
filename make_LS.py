import glob
import joblib
import os 
import json
import logging

import time

import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')   # must come before importing pyplot

import matplotlib.pyplot as plt

from phonetics import utils as u
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

def read_param_dict(parser):
    args = parser.parse_args()
    json_file = open(args.parfile)
    param_dict = json.load(json_file)
    json_file.close()
    return param_dict

param_dict = read_param_dict(parser)

def setup_logs(logs_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_path),   # Logs to a file
            logging.StreamHandler()           # Prints to console
        ]
    )
    logging.info(f'------------- logs output to {logs_path}')

#############################################################

def boiler_plate(param_dict):

    input_path = param_dict["input_path"]
    experiment = param_dict["experiment"]
    
    tp_algn = param_dict['tp_algn']

    if tp_algn == 'text_grid':
        algn_paths = glob.glob(f'{input_path}/TextGrid/*.TextGrid')
    elif tp_algn == 'lab':
        algn_paths = glob.glob(f'{input_path}/lab/*.lab')
    elif tp_algn is None:
        algn_paths = None

    experiment_folder = f'experiments/{experiment}'
    folder_dict = {'experiment_folder': experiment_folder}

    if param_dict.get("output_feat_768d", False):
        folder_dict.update({'feat_768d_folder': f'{experiment_folder}/feat_768d'})

    if param_dict.get("umap_projection", True):
        umap_dim = param_dict['umap'].get("dim")
        logging.info(f'Doing umap projection to dim {umap_dim}')

        folder_dict.update(
            {'feat_umap_folder': f'{experiment_folder}/feat_{umap_dim}d',
              'plots_umap_folder' : f'{experiment_folder}/plots_{umap_dim}d'})
                
    for fo in list(folder_dict.values()):
        os.makedirs(fo, exist_ok=True)

    with open(f"{experiment_folder}/metadata.json", "w") as outfile: 
        json.dump(param_dict, outfile, indent=2)

    logs_path = f'{experiment_folder}/output.log'
    setup_logs(logs_path)

    logging.info(f'Folders created : {list(folder_dict.values())}')

    return algn_paths, folder_dict 

def make_df_annotated(layer, param_dict):
    logging.info(f'-------- df annotated')

    input_path = param_dict["input_path"]
    tp_algn = param_dict["tp_algn"]
    dataset_tp = param_dict.get("dataset_tp", None)
    add_transitions = param_dict.get("add_transitions", False)
    pad_seconds = param_dict.get("pad_seconds", None)
    sort_annotated_data = param_dict.get("sort_annotated_data", False)

    feat_paths = glob.glob(f'{input_path}/feat/layer_{layer}/*.csv')

    df_anotated = u.make_anotated_feat_df(feat_paths, 
                                        algn_paths, 
                                        tp_algn = tp_algn,
                                        dataset = dataset_tp,
                                        add_transitions = add_transitions,
                                        pad_seconds = pad_seconds)
    
    if sort_annotated_data:
        logging.info(f'-------- sorting annotated data')
        phoneme_order = list(df_anotated['phone_base'].value_counts().keys())
        rank = {p: i for i, p in enumerate(phoneme_order)}
        df_anotated = (
            df_anotated
            .assign(_phoneme_rank=df_anotated["phone_base"].map(rank))
            .sort_values(
                ["_phoneme_rank", "duration"],
                kind="mergesort"
            )
            .drop(columns="_phoneme_rank")
            .reset_index(drop=True)
        )
    
    if param_dict.get("output_feat_768d", False):
        df_anotated.to_csv(f'{folder_dict["feat_768d_folder"]}/feat_768d_layer_{layer}.csv')

    len_minutes = len(df_anotated)*0.02/60
    logging.info(f'-------- Produced df_annotated with shape {df_anotated.shape}, amounts to {len_minutes} minutes')

    return df_anotated

def make_df_projected_annotated(df_anotated, param_dict, layer):

    logging.info(f'--------- UMAP')

    umap_dict = param_dict['umap']

    dim = umap_dict.get("dim")
    min_dist = umap_dict.get("min_dist", 0.1)
    n_neighbors = umap_dict.get("n_neighbors", 100)
    metric = umap_dict.get('metric', 'euclidean')
    n_jobs = umap_dict.get('n_jobs', 1)

    logging.info(f'UMAP: dim {dim} | n_neighbors {n_neighbors} | min_dist {min_dist} | metric {metric} | n_jobs {n_jobs}')

    # TODO: rename exclude_phones_plot -> exclude_phones_umap
    exclude_phones = umap_dict.get('exclude_phones_plot', [])
    logging.info(f'Excluding phones {exclude_phones} from projection')

    normalize_vectors = umap_dict.get('normalize_vectors', False)
    logging.info(f'UMAP projection normalize vectors {normalize_vectors}')

    use_gpu_umap = umap_dict.get('use_gpu', False)
    logging.info(f'Using gpu umap {use_gpu_umap}')

    fix_random_state_umap = umap_dict.get('fix_random_state', True)
    logging.info(f'Fixing umap random state {fix_random_state_umap}. Only matters if use_gpu_umap is set to False')

    umap = u.train_umap(
        df_anotated,
        exclude_phones = exclude_phones,
        n_components=dim, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        n_jobs = n_jobs,
        metric = metric,
        normalize_vectors = normalize_vectors,
        use_gpu = use_gpu_umap,
        fix_random_sate = fix_random_state_umap,
        save_model = False,
        folder = None)
        
    df_proj_anotated = u.make_proj_anotated_feat_df(df_anotated, 
                                                    umap,
                                                    save_df = False,
                                                    folder = None)
    
    df_proj_anotated.to_csv(f'{folder_dict[f"feat_umap_folder"]}/feat_{dim}d_layer_{layer}.csv')

    return df_proj_anotated

def make_plot(df_proj_anotated):
    logging.info(f'-------- plot')
    my_phones = [k for k in df_proj_anotated['phone_base'].value_counts().keys() if k != 'SP']
    plots.make_tagged_LS_plot(df_proj_anotated,
            phones = my_phones,
            alpha = 0.25, 
            s = 0.1,
            show_global=True)

    plt.savefig(f'{folder_dict[f"plots_umap_folder"]}/LS_layer_{layer}')


#############################################################
#### Pipeline
#############################################################

T0 = time.time()

algn_paths, folder_dict = boiler_plate(param_dict)
metric_dict = {}

single_layer = param_dict.get('single_layer', None)

if single_layer is None:
    logging.info(f'-------- Computing for all layers')
    min_layer = 1
    max_layer = 12
else:
    logging.info(f'-------- Computing for single layer {single_layer}')
    min_layer = single_layer
    max_layer = single_layer

for layer in range(min_layer, max_layer + 1):

    t0 = time.time()
    logging.info(f'-------- Working on layer {layer}')

    df_anotated = make_df_annotated(layer, param_dict) 
    
    if param_dict.get('compute_metrics', False):
        logging.info('Computating metrics')
        metric_dict[layer] = ph_metrics.compute_metric_for_layer(df_anotated, param_dict)
    else:
        logging.info('Skipping metric computation')

    if param_dict.get("umap_projection", True):
        df_proj_anotated = make_df_projected_annotated(df_anotated, 
                                                          param_dict, 
                                                          layer)
        make_plot(df_proj_anotated)
    else:
        logging.info('Skipping umap projection')

    t1 = time.time()
    dt = t1 - t0
    logging.info(f'------------- Time for layer {layer}: {dt}')

if param_dict.get('compute_metrics', False):
    exp_folder = folder_dict["experiment_folder"]
    df_metric = ph_metrics.make_df_metric(metric_dict)
    df_metric.to_csv(f'{exp_folder}/metric_layers.csv')

T1 = time.time()    
DT = T1 - T0

logging.info(f'------------- Total Time: {DT}')