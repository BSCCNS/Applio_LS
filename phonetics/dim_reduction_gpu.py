#import cupy as cp
from cuml.manifold import UMAP

import pandas as pd
import numpy as np
import time
import os

from tools.utils import make_X

circ_file = 'input/min_cl_user_50_min_cl_item_100/circ_processed.csv'
out_folder = 'output/sample_2d'
os.makedirs(out_folder, exist_ok=True)

N_NEIGHBORS = 100

T0 = time.time()
X = make_X(circ_file)

# 1) CPU-side prep
X32 = X.astype("float32")
X_np = np.asarray(X32.values, dtype=np.float32, order="C")

# 2) UMAP with batched nn-descent
umap = UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.0,
    metric="cosine",
    init="random", # avoid spectral-init bug
)

X_2d = umap.fit_transform(X_np)
df_X_2d = pd.DataFrame(X_2d, columns=["x", "y"], index=X.index)

print(f'df_X_2d.shape : {df_X_2d.shape}')

file = f'{out_folder}/user_emb_2d_gpu_{N_NEIGHBORS}.csv'
df_X_2d.to_csv(file)

T1 = time.time()
DT = T1 - T0

print(f'Total time {DT}')
