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
from phonetics import phone_info as ph_i 
from phonetics import plots as plots
from umap import UMAP
from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

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

#root = f'/media/HDD_disk/tomas/ICHOIR/Applio_LS/assets/datasets/GTSinger_ES'

def boiler_plate(param_dict):

    input_path = param_dict["input_path"]
    experiment = param_dict["experiment"]
    
    tp_algn = param_dict['tp_algn']

    if tp_algn == 'text_grid':
        algn_paths = glob.glob(f'{input_path}/TextGrid/*.TextGrid')
    elif tp_algn == 'lab':
        algn_paths = glob.glob(f'{input_path}/lab/*.lab')

    experiment_folder = f'experiments/{experiment}'
    plots_folder = f'{experiment_folder}/plots'
    feat_2d_folder = f'{experiment_folder}/feat_2d'
    feat_768d_folder = f'{experiment_folder}/feat_768d'

    folders = [experiment_folder, plots_folder, feat_2d_folder, feat_768d_folder]

    for fo in folders:
        os.makedirs(fo, exist_ok=True)

    with open(f"{experiment_folder}/metadata.json", "w") as outfile: 
        json.dump(param_dict, outfile, indent=4)

    folder_dict = {
        'experiment_folder': experiment_folder,
        'plots_folder': plots_folder,
        'feat_2d_folder': feat_2d_folder,
        'feat_768d_folder': feat_768d_folder
        }

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
    
    df_anotated.to_csv(f'{folder_dict["feat_768d_folder"]}/feat_768d_layer_{layer}.csv')

    return df_anotated

def make_df_projected_annotated_2d(df_anotated, param_dict):
    
    exclude_phones = param_dict['exclude_phones_plot']
    print(f'Excluding phones {exclude_phones} from plot')
    
    print(f'-------- umap')
    umap2 = u.train_umap(
        df_anotated,
        exclude_phones = exclude_phones,
        n_components=2, 
        n_neighbors=100, 
        min_dist=0.1,
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

def extract_Xy(df_anotated):
    X = df_anotated.drop(columns = ['phone_base', 'song']).values
    y = df_anotated['phone_base'].values

    return X, y

def compute_silhouette(df_anotated):
    print(f'-------- Computing silhouette')

    X, y = extract_Xy(df_anotated)
    sil_score = silhouette_score(X, y, metric='cosine')
    print(f'--------- sil_score {sil_score}')
    return sil_score 

def compute_MI(df_anotated, n_clusters = 50):
    print(f'-------- Computing MI')

    X, y = extract_Xy(df_anotated)

    kmeans = KMeans(n_clusters = n_clusters, 
                    random_state=42)
    
    cluster_assignments = kmeans.fit_predict(X)

    mi = mutual_info_score(y, cluster_assignments)

    print(f"'-------- Mutual Information (MI-phone): {mi:.4f}")
    return mi

def compute_metric_for_layer(df_anotated, param_dict):
    
    K_MI = param_dict["K_MI"]
    exclude_phones = param_dict['exclude_phones_metric']

    print(f'Excluding phones {exclude_phones} from metric computations')
    
    if exclude_phones is None:
        df = df_anotated
    else:
        mask = ~df_anotated["phone_base"].isin(exclude_phones)
        df = df_anotated[mask]

    sil = compute_silhouette(df)
    mi = compute_MI(df, n_clusters = K_MI)

    return {'sil': sil, 'mi': mi}


def make_df_metric(metric_dict):
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df = df.reset_index().rename(columns={'index': 'layer'})

    return df

#############################################################


T0 = time.time()

algn_paths, folder_dict = boiler_plate(param_dict)
metric_dict = {}

for layer in range(1,13):

    t0 = time.time()
    print(f'-------- Working on layer {layer}')

    df_anotated = make_df_annotated(layer, param_dict) 
    metric_dict[layer] = compute_metric_for_layer(df_anotated, param_dict)

    if param_dict.get("projection_2d", True):
        df_proj_anotated = make_df_projected_annotated_2d(df_anotated, param_dict) 
        make_plot(df_proj_anotated)
    else:
        print('Skipping 2d projection and plots')
    
    t1 = time.time()
    dt = t1 - t0
    print(f'------------- Time for layer {layer}: {dt}')

exp_folder = folder_dict["experiment_folder"]
df_metric = make_df_metric(metric_dict)
df_metric.to_csv(f'{exp_folder}/metric_layers.csv')

T1 = time.time()    
DT = T1 - T0

print(f'------------- Total Time: {DT}')