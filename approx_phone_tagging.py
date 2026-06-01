import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree

# root_exp = '/media/HDD_disk/tomas/ICHOIR/Applio_LS/experiments/maria_3d'
# feat_path = f'{root_exp}/feat_768d/feat_768d_layer_12.csv'
# feat_projected_path = f'{root_exp}/feat_3d/feat_3d_layer_12.csv'

# root_song = '/media/HDD_disk/tomas/ICHOIR/Applio_LS/assets/datasets/pellizco/feat/layer_12'
# song_name = 'feats_04 PELLIZCO_LIVE__PELLIZCO_BVS'
# feat_path_song = f'{root_song}/{song_name}.csv'

ROOT = "/home/bsc/bsc270816/Applio_LS/experiments"
LAYER = 8

exp_libri = 'libri_768d_v2'
feat_path = f"{ROOT}/{exp_libri}/feat_768d/feat_768d_layer_{LAYER}.csv"
feat_projected_path = f"{ROOT}/{exp_libri}/feat_2d/feat_2d_layer_{LAYER}.csv"

exp_asv = 'ASVpreproc_768d_small_v1'
feat_path_song = f"{ROOT}/{exp_asv}/feat_768d/feat_768d_layer_{LAYER}.csv"

outfile = f"{ROOT}/{exp_asv}/feat_2d/feat_2d_layer_{LAYER}_approx_projection_{exp_libri}.csv"
outfile2 = f"{ROOT}/{exp_asv}/feat_2d/feat_2d_layer_{LAYER}_approx_projection_tagged_{exp_libri}.csv"

NON_EMB_COLS = ['phone_base', 'duration', 'start' , 'song']

t0 = time.time()
##########################################################################
print('----- Reading full LS data')

df_anotated = pd.read_csv(feat_path, index_col=0, low_memory=False)
df_anotated_projected = pd.read_csv(feat_projected_path, index_col=0, low_memory=False)

##########################################################################
print('----- Reading song LS data')
df_song_feat = pd.read_csv(feat_path_song, index_col=0)

##########################################################################
print('----- Constructing distance tree')

X_full_values = df_anotated.drop(columns=NON_EMB_COLS).to_numpy()
tree = cKDTree(X_full_values)

X_target = df_song_feat.drop(columns=NON_EMB_COLS).to_numpy()
# Query nearest df1 point for each df2 point
dist, idx = tree.query(X_target, k=1, workers=-1)  # k=1 = nearest

# Assign tag (and optionally distance / matched df1 index)
df2_tagged = df_song_feat.copy()
df2_tagged['phone_base'] = df_anotated.iloc[idx]['phone_base'].to_numpy()

df2_tagged['nn_distance'] = dist
df2_tagged['nn_df1_index'] = df_anotated.index.to_numpy()[idx]  # keeps original df1 index

df2_tagged[['phone_base', 'nn_distance', 'nn_df1_index']].to_csv(outfile2)

##########################################################################
print('----- Applying annotated projection by proximity')

df_song_projected_proximity = df_anotated_projected.iloc[df2_tagged['nn_df1_index']]

#outfile = f'{root_song}/{song_name}_approx_projection.csv'
print(f'----- Saving output to {outfile}')
df_song_projected_proximity.to_csv(outfile)
##########################################################################

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')