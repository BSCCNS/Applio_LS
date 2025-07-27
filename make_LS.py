import glob
import joblib
import os 

import time

import pandas as pd
#from pathlib import Path
import matplotlib.pyplot as plt

from phonetics import utils as u
from phonetics import phone_info as ph_i 
from phonetics import plots as plots
from umap import UMAP
from sklearn.metrics import silhouette_score, silhouette_samples

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# DATA_SET = 'GTSinger_ES'
# DATA_SET_TP = 'gt'
# EXCLUDE_PHONES = ['<AP>']
# tp_algn = 'text_grid'

DATA_SET = 'songs' #'GTSinger_ES'
DATA_SET_TP = None #'gt'
EXCLUDE_PHONES =  ['AP'] #['<AP>']
tp_algn = 'lab' #'text_grid'

root = f'/media/HDD_disk/tomas/ICHOIR/Applio_LS/assets/datasets/{DATA_SET}'
output_folder = f'{DATA_SET}_output'

if tp_algn == 'text_grid':
    algn_paths = glob.glob(f'{root}/TextGrid/*.TextGrid')
elif tp_algn == 'lab':
    algn_paths = glob.glob(f'{root}/lab/*.lab')

silhouette_dict = {}

folder_plots = f'{output_folder}/plots'
folder_feat_2d = f'{output_folder}/feat_2d'
folder_feat_768d = f'{output_folder}/feat_768d'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(folder_plots, exist_ok=True)
os.makedirs(folder_feat_2d, exist_ok=True)
os.makedirs(folder_feat_768d, exist_ok=True)

T0 = time.time()

for layer in range(1,13):

    t0 = time.time()
    print(f'-------- Working on layer {layer}')
    feat_paths = glob.glob(f'{root}/feat/layer_{layer}/*.csv')
    df_anotated = u.make_anotated_feat_df(feat_paths, 
                                          algn_paths, 
                                          tp_algn = tp_algn,
                                          dataset = DATA_SET_TP)
    
    df_anotated.to_csv(f'{folder_feat_768d}/feat_768d_layer_{layer}.csv')
    
    ### umap
    print(f'-------- umap')
    umap2 = u.train_umap(
        df_anotated,
        exclude_phones = EXCLUDE_PHONES,
        n_components=2, 
        n_neighbors=100, 
        min_dist=0.1,
        save_model = False,
        folder = None)
    
    df_proj_anotated = u.make_proj_anotated_feat_df(df_anotated, 
                                                    umap2,
                                                    save_df = False,
                                                    folder = None)
    
    df_proj_anotated.to_csv(f'{folder_feat_2d}/feat_2d_layer_{layer}.csv')


    ### plots
    print(f'-------- plot')
    my_phones = list(df_anotated['phone_base'].value_counts().to_dict().keys())[0:15]
    plots.make_tagged_LS_plot(df_proj_anotated,
            phones = my_phones,
            alpha = 0.25, 
            s = 0.1,
            show_global=True)
    plt.savefig(f'{folder_plots}/LS_{DATA_SET}_layer_{layer}')
    
    ### silhouettes
    print(f'-------- silhouette')
    X1 = df_anotated.drop(columns = ['phone_base', 'song']).values
    y1 = df_anotated['phone_base'].values
    sil_score = silhouette_score(X1, y1, metric='cosine')

    silhouette_dict[layer] = sil_score
    print(f'--------- sil_score {sil_score}')

    t1 = time.time()
    dt = t1 - t0
    print(f'------------- Time for layer {layer}: {dt}')

df = pd.DataFrame.from_dict(silhouette_dict, orient='index', columns=['silhouette'])
df_sil = df.reset_index().rename(columns={'index': 'layer'})

df_sil.to_csv(f'{output_folder}/silhouette_layers.csv')

T1 = time.time()    
DT = T1 - T0

print(f'------------- Total Time: {DT}')