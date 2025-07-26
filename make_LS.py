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
feat_paths = glob.glob(f'{root}/feat_layer_2/*.csv')
text_grid_paths = glob.glob(f'{root}/TextGrid/*.TextGrid')

df_anotated_all = u.make_anotated_feat_df(feat_paths, 
                                          text_grid_paths, 
                                          tp_algn = 'text_grid',
                                          dataset = 'gt')

df_info = df_anotated_all['song'].str.split('-', expand=True)
df_info.columns = ['speaker', 'book', 'line']
df_anotated_all = pd.concat([df_anotated_all, df_info], axis = 1)

print(df_anotated_all)