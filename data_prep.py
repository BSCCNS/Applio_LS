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

ROOT_EXP = "/home/bsc/bsc270816/Applio_LS/experiments/ASVpreproc_768d_full"

feat_path = f"{ROOT_EXP}/feat_768d/feat_768d_layer_8.csv"
tag_path = f"{ROOT_EXP}/tag/layer_8_tagged_libri_768d_v3.csv"

dataset_dir = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019/ASVspoof2019_LA_train_preproc'
output_data_prep_dir = f"{dataset_dir}/data_prep"
output_data_prep = f"{output_data_prep_dir}/feat_768d_layer_8_tag.parquet"

asv_folder = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019/LA/'
cm_train_file = 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
cm_path = f'{asv_folder}/{cm_train_file}'

print(f'----- Making output dir {output_data_prep_dir}')
os.makedirs(output_data_prep_dir, exist_ok=True)

t0 = time.time()
##########################################################################
print('----- Reading cm data')
df_cm = pd.read_csv(
    cm_path,
    sep=' ',
    header=None,
    names=['speaker_id', 'name', 'placeholder', 'system_id', 'key']
)
df_cm = df_cm.drop(columns=['placeholder'])

print('df_cm')
print(df_cm.head())

##########################################################################
print('----- Reading tag data')
df_tag = pd.read_csv(tag_path, index_col=0)

print('df_tag')
print(df_tag.head())

df_tag = df_tag.set_index('name').join(df_cm.set_index('name')).reset_index()

print('df_tag after')
print(df_tag.head())

##########################################################################

print('----- Reading LS data')

df_anotated = pd.read_csv(feat_path, index_col=0, low_memory=False)

# emb cols and song
df_anotated = df_anotated.drop(columns=['phone_base', 'duration', 'start'])
df_anotated = df_anotated.rename(columns = {'song': 'name'})

##########################################################################

print('----- join')
df_tagged = df_anotated.join(df_tag)

##########################################################################

print(f'----- Saving output to {output_data_prep}')
df_tagged.to_parquet(output_data_prep)

##########################################################################

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')