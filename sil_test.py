import pandas as pd
from phonetics import utils as u
import time

from sklearn.metrics import silhouette_samples

experiment = 'maria_v0'

root = '/media/HDD_disk/tomas/ICHOIR/Applio_LS/experiments'
experiment_path = f'{root}/{experiment}'
feat_path = f'{experiment_path}/feat_768d/feat_768d_layer_12.csv'

t0 = time.time()


df_anotated = pd.read_csv(feat_path, index_col=0)
df_grouped, ph_group = u.group_vowels_consonants_ap(df_anotated)

X = df_grouped.drop(columns=['phone_base', 'song', 'group']).values
y = df_grouped['group'].values

print('-------------- Computing silhouette scores')
sil_samples = silhouette_samples(X, y, metric='cosine')

sil_score = sil_samples.mean()

print(f'Sil score : {sil_score}')

df = pd.DataFrame(data = sil_samples, columns=['sil'])

df.to_csv(f'{experiment}_sil_samples_768d_layer_12.csv')


t1 = time.time()
dt = t1 - t0

print(f'Total time {dt}')