import glob
import joblib
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

# df_anotated_all = u.make_anotated_feat_df(feat_paths, 
#                                           text_grid_paths, 
#                                           tp_algn = 'text_grid',
#                                           dataset = 'gt')

# df_info = df_anotated_all['song'].str.split('-', expand=True)
# df_info.columns = ['speaker', 'book', 'line']
# df_anotated_all = pd.concat([df_anotated_all, df_info], axis = 1)

# print(df_anotated_all)

silhouette_dict = {}

for layer in [2,3]:

    print(f'-------- Working on layer {layer}')
    #feat_paths = glob.glob(f'{root}/{layers_dict[layer]}/*.csv')
    feat_paths = glob.glob(f'{root}/feat/layer_{layer}/*.csv')
    #df_anotated = u.make_anotated_feat_df(feat_paths, lab_paths)

    df_anotated = u.make_anotated_feat_df(feat_paths, 
                                          text_grid_paths, 
                                          tp_algn = 'text_grid',
                                          dataset = 'gt')
    # df_info = df_anotated['song'].str.split('-', expand=True)
    # df_info.columns = ['speaker', 'book', 'line']
    # df_anotated = pd.concat([df_anotated, df_info], axis = 1)
    # c1 = df_anotated_all['phone_base'] != '<AP>'
    # drop_cols = ['phone_base', 'song', 'speaker', 'book', 'line']
    # df_no_sil = df_anotated_all[c1].reset_index(drop = True)
    # X_no_sil = df_no_sil.drop(columns=drop_cols).values

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
    
    ### plots
    print(f'-------- plot')
    #unique_phones = [k for k in df_anotated['phone_base'].value_counts().keys() if k != '<AP>']
    my_phones = list(df_anotated['phone_base'].value_counts().to_dict().keys())[0:15]
    plots.make_tagged_LS_plot(df_proj_anotated,
            phones = my_phones,
            alpha = 0.25, 
            s = 0.1,
            show_global=True)
    plt.savefig(f'LS_GTSinger_ES_layer_{layer}')
    
    ### silhouettes
    print(f'-------- silhouette')
    X1 = df_anotated.drop(columns = ['phone_base', 'song']).values
    y1 = df_anotated['phone_base'].values
    sil_score = silhouette_score(X1, y1, metric='cosine')

    silhouette_dict[layer] = sil_score
    print(f'--------- sil_score {sil_score}')

    print(f'-------- o -------- \n')