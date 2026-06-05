import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree
import faiss

import scipy
import os

import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU in use:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

print('------ Preamble')
print(scipy.__version__)
print(f"CPUs available to process: {os.cpu_count()}")

ROOT = "/home/bsc/bsc270816/Applio_LS/experiments"
LAYER = 8

exp_libri = 'libri_768d_v3'
feat_path = f"{ROOT}/{exp_libri}/feat_768d/feat_768d_layer_{LAYER}.csv"

exp_asv = 'ASVpreproc_768d_full'
feat_path_song = f"{ROOT}/{exp_asv}/feat_768d/feat_768d_layer_{LAYER}.csv"

output_tag_dir = f"{ROOT}/{exp_asv}/tag"
print(f'----- Making output tag dir {output_tag_dir}')
os.makedirs(output_tag_dir, exist_ok=True)

outfile_tag = f"{output_tag_dir}/layer_{LAYER}_tagged_{exp_libri}.csv"

NON_EMB_COLS = ['phone_base', 'duration', 'start' , 'song']

t0 = time.time()
##########################################################################
print('----- Reading LS data')

df_anotated = pd.read_csv(feat_path, index_col=0, low_memory=False)
df_song_feat = pd.read_csv(feat_path_song, index_col=0)

def tree_tag(df_anotated, df_song_feat):
    print('----- Constructing distance tree')

    X_full_values = df_anotated.drop(columns=NON_EMB_COLS).to_numpy()
    X_target = df_song_feat.drop(columns=NON_EMB_COLS).to_numpy()

    # Build index
    t0 = time.time()
    tree = cKDTree(X_full_values)
    print(f"Build cKDTree: {time.time()-t0:.2f}s")

    t0 = time.time()
    dist, idx = tree.query(X_target, k=1, workers=112)  # k=1 = nearest
    print(f'dist shape {dist.shape} | idx shape {idx.shape}')
    print(f"Query cKDTree: {time.time()-t0:.2f}s")

    df2_tagged = df_song_feat.copy()
    df2_tagged['phone_base'] = df_anotated.iloc[idx]['phone_base'].to_numpy()
    df2_tagged['nn_distance'] = dist

    return df2_tagged

def faiss_tag(df_anotated, df_song_feat):

    X_full_values = df_anotated.drop(columns=NON_EMB_COLS).to_numpy()
    X_target = df_song_feat.drop(columns=NON_EMB_COLS).to_numpy()

    # FAISS requires float32
    Xf        = X_full_values.astype(np.float32)
    Xf_target = X_target.astype(np.float32)

    # Build index
    t0 = time.time()
    index = faiss.IndexFlatL2(Xf.shape[1])   # exact L2, equivalent to cKDTree
    index.add(Xf)
    print(f"Build faiss: {time.time()-t0:.2f}s")

    # Set threads
    faiss.omp_set_num_threads(112)

    # Query
    t0 = time.time()
    dist, idx = index.search(Xf_target, k=1)
    dist = np.squeeze(dist)
    idx = np.squeeze(idx)
    print(f'dist shape {dist.shape} | idx shape {idx.shape}')
    print(f"Query faiss: {time.time()-t0:.2f}s")

    df2_tagged = df_song_feat.copy()
    df2_tagged['phone_base'] = df_anotated.iloc[idx]['phone_base'].to_numpy()
    df2_tagged['nn_distance'] = dist

    return df2_tagged

df2_tagged = faiss_tag(df_anotated, df_song_feat)

print(f'----- Saving output to {outfile_tag}')
df2_tagged[['phone_base', 'nn_distance']].to_csv(outfile_tag)


##########################################################################

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')