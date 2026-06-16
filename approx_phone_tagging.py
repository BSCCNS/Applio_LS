import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree
import faiss

#from mpi4py import MPI

import scipy
import os
import sys

import torch

print(f"CPUs available to process: {os.cpu_count()}")

if torch.cuda.is_available():
    print("CUDA is available. GPU in use:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

# Usage 
# python approx_phone_tagging.py <experiment_name>
# python approx_phone_tagging.py ASV_dev_preproc_768d
# python approx_phone_tagging.py ASV_train_preproc_768d

exp_asv = sys.argv[1]

ROOT = "/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019/experiments"
LAYER = 8

exp_libri = 'libri_768d_v3'
feat_path = f"{ROOT}/{exp_libri}/feat_768d/feat_768d_layer_{LAYER}.csv"

#exp_asv = 'ASV_dev_preproc_768d_full'
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

def faiss_mpi_tag(df_anotated, df_song_feat):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    X_full_values = df_anotated.drop(columns=NON_EMB_COLS).to_numpy()
    X_target = df_song_feat.drop(columns=NON_EMB_COLS).to_numpy()

    X        = X_full_values.astype(np.float32)
    X_target = X_target.astype(np.float32)

    local_chunk = np.array_split(X_target, size)[rank]

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    faiss.omp_set_num_threads(112)

    local_dist, local_idx = index.search(local_chunk, k=1)

    all_dist = comm.gather(local_dist, root=0)
    all_idx  = comm.gather(local_idx,  root=0)

    # only rank 0 has the full results
    if rank == 0:
        dist = np.concatenate(all_dist).flatten()   # fix 2: flatten to 1D
        idx  = np.concatenate(all_idx).flatten()    # fix 2: flatten to 1D

        print(f'dist shape {dist.shape} | idx shape {idx.shape}')  # fix 1: moved inside
        print(f"Query faiss: {time.time()-t0:.2f}s")

        df2_tagged = df_song_feat.copy()
        df2_tagged['phone_base'] = df_anotated.iloc[idx]['phone_base'].to_numpy()
        df2_tagged['nn_distance'] = dist

        return df2_tagged

    return None  # other ranks return nothing

t0 = time.time()
df2_tagged = faiss_tag(df_anotated, df_song_feat[0:15000])
t1 = time.time()
dt1 = t0 - t1
print(f'dt1 : {dt1}')

# t0 = time.time()
# print(f'Full size {len(df_song_feat)}')
# result = faiss_mpi_tag(df_anotated, df_song_feat[:15000])
# if result is not None:  # only rank 0 has the result
#     print(result.head()) 
# t1 = time.time()

# dt2 = t1 - t0
# print(f'dt2 MPI : {dt2}')

#print(f'----- Saving output to {outfile_tag}')
#df2_tagged[['phone_base', 'nn_distance']].to_csv(outfile_tag)


##########################################################################

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')