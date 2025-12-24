import os
import pandas as pd
import matplotlib.pyplot as plt

from phonetics import utils as u
from phonetics import plots as plots

#############################################################

root = '/media/HDD_disk/tomas/ICHOIR/Applio_LS/experiments'

#root = '/home/bsc/bsc270816/Applio_LS/experiments'
experiment = 'maria_v2_NEW'
file = f'{root}/{experiment}/feat_768d/feat_768d_layer_12.csv'

OUTDIR = './umap_test'

print(f'-------- reding data')
df_anotated = pd.read_csv(file, index_col=0)

print(f'-------- sorting data')
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

#######################################################################

def make_ls(df_anotated, dim, min_dist = 0.1, nn = 100):
    print(f'-------- umap {dim}d')
    umap = u.train_umap(
    df_anotated,
    exclude_phones = ['SP'],
    n_components = dim, 
    n_neighbors = nn, 
    min_dist = min_dist,
    n_jobs = 1,
    save_model = False,
    folder = None)
    print(f'-------- umap {dim}d finished')
        
    df_proj_anotated = u.make_proj_anotated_feat_df(df_anotated, 
                                                    umap,
                                                    save_df = False,
                                                    folder = None)

    outfile = f'{OUTDIR}/df_proj_anotated_nn_{nn}_{dim}d.csv'
    df_proj_anotated.to_csv(outfile)

#######################################################################

dim = 2
make_ls(df_anotated, 
        dim, 
        min_dist = 0.1, 
        nn = 100)

dim = 3
for min_dist in [0.1, 0.2, 0.3]:
    make_ls(df_anotated, 
            dim, 
            min_dist = min_dist, 
            nn = 100)





