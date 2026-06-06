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

output_data_prep_dir = f"{ROOT_EXP}/data_prep"
output_data_prep = f"{output_data_prep_dir}/feat_768d_layer_8_tag.csv"

print(f'----- Making output dir {output_data_prep_dir}')
os.makedirs(output_data_prep_dir, exist_ok=True)

t0 = time.time()
##########################################################################
print('----- Reading LS data')

df_anotated = pd.read_csv(feat_path, index_col=0, low_memory=False)
df_tag = pd.read_csv(tag_path, index_col=0)

# emb cols and song
df_anotated = df_anotated.drop(columns=['phone_base', 'duration', 'start'])
df_anotated = df_anotated.rename(columns = {'song': 'name'})

print('----- join')
df_tagged = df_anotated.join(df_tag)

print(f'----- Saving output to {output_data_prep}')
df_tagged.to_csv(output_data_prep)

##########################################################################

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')