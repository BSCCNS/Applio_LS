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

cm_file_dict = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt"
}

tp = "train"
layer = 12
ROOT = "/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019"
ROOT_EXP = f"{ROOT}/experiments/ASV_{tp}_preproc_768d"
feat_path = f"{ROOT_EXP}/feat_768d/feat_768d_layer_{layer}.csv"

output_data_prep_dir = f"{ROOT}/data_prep"
output_data_prep = f"{output_data_prep_dir}/feat_768d_{tp}_layer_{layer}_tag.parquet"

asv_folder = f'{ROOT}/LA/ASVspoof2019_LA_cm_protocols'
cm_train_file = cm_file_dict[tp]
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
# print('----- Reading tag data')
# df_tag = pd.read_csv(tag_path, index_col=0)

# print('df_tag')
# print(df_tag.head())
##########################################################################

print('----- Reading LS data')

df_anotated = pd.read_csv(feat_path, index_col=0, low_memory=False)

# emb cols and song
df_anotated = df_anotated.drop(columns=['phone_base', 'duration', 'start'])
df_anotated = df_anotated.rename(columns = {'song': 'name'})

print('df_anotated.columns')
print(df_anotated.columns)

##########################################################################

print('----- join')
df_anotated = df_anotated.set_index('name').join(df_cm.set_index('name')).reset_index()

##########################################################################

print(f'----- Saving output to {output_data_prep}')
df_anotated.to_parquet(output_data_prep)

##########################################################################

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')