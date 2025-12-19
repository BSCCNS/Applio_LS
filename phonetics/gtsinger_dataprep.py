import glob
import pandas as pd
import os
from pathlib import Path
import glob
import shutil
import re
import unicodedata
from textgrids import TextGrid

lang = 'ZH'
ROOT = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets'
output_folder = f'{ROOT}/GTSinger_{lang}_flat'

SELECT_STYLES = ['Vibrato', 'Glissando']
EXCLUDE_GROUPS = ['Paired_Speech_Group', 'Control_Group']

#####################################################################################

def normalize(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove accents (e.g., é → e)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing spaces
    return text.strip().replace(' ', '_')

def check_matching_files(lang):
    tg_root = f'{ROOT}/GTSinger_{lang}_flat/TextGrid'
    wav_root = f'{ROOT}/GTSinger_{lang}_flat/wav'

    tg_files = sorted(glob.glob(f'{tg_root}/*'))
    wav_files = sorted(glob.glob(f'{wav_root}/*'))

    tg_stems = [file.split('/')[-1].split('.TextGrid')[0] for file in tg_files]
    wav_stems = [file.split('/')[-1].split('.wav')[0] for file in wav_files]

    df_tg = pd.DataFrame(data = tg_stems)
    df_wav = pd.DataFrame(data = wav_stems)

    n_wav = len(df_wav)
    n_tg = len(df_tg)

    matches = (df_tg == df_wav).sum().values[0]

    return n_wav, n_tg, matches

def get_all_tg_files(lang):
    tg_root = f'{ROOT}/GTSinger_{lang}_flat/TextGrid'
    tg_files = sorted(glob.glob(f'{tg_root}/*'))

    return tg_files

def get_total_length(lang):
    files = get_all_tg_files(lang)
    secs = sum([get_tg_length(f) for f in files])
    return secs/60.

def get_tg_length(file):
    tg = TextGrid()
    tg.read(file)

    return tg['global'].xmax


#####################################################################################

os.makedirs(f'{output_folder}/wav')
os.makedirs(f'{output_folder}/TextGrid')

all_files = glob.glob(f'{ROOT}/{lang}/*/**/***/****/*****')
data = [file.split(f'{ROOT}/{lang}/')[1].split('/') for file in all_files]

df = pd.DataFrame(data = data, columns=['singer', 'style', 'song', 'group', 'file'])
df['input_path'] = all_files

# prepare output paths
df['song'] = df['song'].apply(normalize)
df['new_song'] = df['style'] + '_' +  df['song'] + '_' + df['group']
df['out_path'] = df['singer'] + '-' + df['new_song'] + '-' + df['file'].str.replace('_TextGrid', '.TextGrid')

print(df['style'].unique())

## Select 
df_songs = df[~df['group'].isin(EXCLUDE_GROUPS)]

df_songs = df_songs[df_songs['style'].isin(SELECT_STYLES)]

df_wav = df_songs[df_songs['file'].str.contains('wav')].sort_values('input_path')
df_wav['out_path'] = f'{ROOT}/GTSinger_{lang}_flat/wav/' + df_wav['out_path']

df_TextGrid = df_songs[df_songs['file'].str.contains('TextGrid')].sort_values('input_path')
df_TextGrid['out_path'] = f'{ROOT}/GTSinger_{lang}_flat/TextGrid/' + df_TextGrid['out_path']

# check 1
print(df_wav.shape, df_TextGrid.shape)

# output files
for input_path, out_path in zip(df_wav['input_path'], df_wav['out_path']):
    shutil.copy2(input_path, out_path)

for input_path, out_path in zip(df_TextGrid['input_path'], df_TextGrid['out_path']):
    shutil.copy2(input_path, out_path)


#################################################################
### checks

check_matching_files(lang)
get_total_length(lang)