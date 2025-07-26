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

root = '/media/HDD_disk/tomas/ICHOIR/Applio_LS/assets/datasets/GTSinger_ES'
#feat_paths = glob.glob(f'{root}/feat/layer_2/*.csv')
text_grid_paths = glob.glob(f'{root}/TextGrid/*.TextGrid')

silhouette_dict = {}

folder_plots = 'GT_SingerES_output/plots'
folder_feat_annotated = 'GT_SingerES_output/feat_2d'

os.makedirs('GT_SingerES_output', exist_ok=True)

os.makedirs(folder_plots, exist_ok=True)
os.makedirs(folder_feat_annotated, exist_ok=True)

T0 = time.time()

for layer in range(1,13):

    t0 = time.time()
    print(f'-------- Working on layer {layer}')
    feat_paths = glob.glob(f'{root}/feat/layer_{layer}/*.csv')
    df_anotated = u.make_anotated_feat_df(feat_paths, 
                                          text_grid_paths, 
                                          tp_algn = 'text_grid',
                                          dataset = 'gt')
    
    ### umap
    print(f'-------- umap')
    umap2 = u.train_umap(
        df_anotated,
        exclude_phones = ['<AP>'],
        n_components=2, 
        n_neighbors=100, 
        min_dist=0.1,
        save_model = False,
        folder = None)
    
    df_proj_anotated = u.make_proj_anotated_feat_df(df_anotated, 
                                                    umap2,
                                                    save_df = False,
                                                    folder = None)
    
    df_proj_anotated.to_csv(f'{folder_feat_annotated}/feat_2d_layer_{layer}.csv')


    ### plots
    print(f'-------- plot')
    my_phones = list(df_anotated['phone_base'].value_counts().to_dict().keys())[0:15]
    plots.make_tagged_LS_plot(df_proj_anotated,
            phones = my_phones,
            alpha = 0.25, 
            s = 0.1,
            show_global=True)
    plt.savefig(f'{folder_plots}/LS_GTSinger_ES_layer_{layer}')
    
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

df_sil.to_csv(f'{folder_feat_annotated}/silhouette_layers.csv')

T1 = time.time()    
DT = T1 - T0

print(f'------------- Total Time: {DT}')