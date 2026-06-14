import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree
import faiss
import argparse

import scipy
import os

import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU in use:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

parser = argparse.ArgumentParser(
                    prog='data_prep with phone data',
                    description='combine features with asv and phone tagging data',
                    epilog='Ask me for help')

# Define named arguments
parser.add_argument('--tp', type=str, required=True, help="train or dev")
parser.add_argument('--layer', type=int, required=True, help="layer of contentvec")

args = parser.parse_args()

cm_file_dict = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt"
}

tp = args.tp
layer = int(args.layer)

print(f'----- Using dataset {tp}, layer {layer}')

ROOT = "/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019"
ROOT_EXP = f"{ROOT}/experiments/ASV_{tp}_preproc_768d"
feat_path = f"{ROOT_EXP}/feat_768d/feat_768d_layer_{layer}.csv"
tag_path = f"{ROOT_EXP}/tag/layer_{layer}_tagged_libri_768d_v3.csv"

output_data_prep_dir = f"{ROOT}/data_prep"

asv_folder = f'{ROOT}/LA/ASVspoof2019_LA_cm_protocols'
cm_train_file = cm_file_dict[tp]
cm_path = f'{asv_folder}/{cm_train_file}'

print(f'----- Making output dir {output_data_prep_dir}')
os.makedirs(output_data_prep_dir, exist_ok=True)

##########################################################################

def make_df_cm(cm_path):
    print('----- Reading cm data')
    df_cm = pd.read_csv(
        cm_path,
        sep=' ',
        header=None,
        names=['speaker_id', 'name', 'placeholder', 'system_id', 'key']
    )
    df_cm = df_cm.drop(columns=['placeholder'])

    return df_cm

def make_df_tag(tag_path):
    print('----- Reading tag data')
    if tag_path is None:
        print('---- No phone tag file provided')
        return None
    else:
        return pd.read_csv(tag_path, index_col=0)
    
def make_df_feat(feat_path):
    print('----- Reading LS data')
    df_feat = pd.read_csv(feat_path, index_col=0, low_memory=False)

    # emb cols and song --> name
    df_feat = df_feat.drop(columns=['phone_base', 'duration', 'start'])
    df_feat = df_feat.rename(columns = {'song': 'name'})

    return df_feat

##########################################################################

t0 = time.time()

df_cm = make_df_cm(cm_path)
df_tag = make_df_tag(tag_path)
df_feat = make_df_feat(feat_path)

print('----- join')
df_anotated = df_feat.set_index('name').join(df_cm.set_index('name')).reset_index()

if df_tag is None:
    print('---- No phone tag df')
    df_tagged = df_anotated
    output_data_prep = f"{output_data_prep_dir}/feat_768d_{tp}_layer_{layer}.parquet"
else:
    df_tagged = df_anotated.join(df_tag)
    output_data_prep = f"{output_data_prep_dir}/feat_768d_{tp}_layer_{layer}_libri_tag.parquet"

print(f'----- Output columns {df_tagged.columns}')
print(f'----- Saving output to {output_data_prep}')
df_tagged.to_parquet(output_data_prep)

t1 = time.time()
dt = t1 - t0
print(f'Total time: {dt}')