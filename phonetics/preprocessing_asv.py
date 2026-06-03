import os
import glob

import numpy as np

import librosa
import soundfile as sf
from pydub import AudioSegment

ROOT = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019'

root_audios = f'{ROOT}/LA/ASVspoof2019_LA_train/flac'
files = sorted(glob.glob(f'{root_audios}/^LA_T_[12]\d*\.flac$'))

output_dir = f'{ROOT}/ASVspoof2019_LA_train_preproc/flac'
os.makedirs(output_dir, exist_ok=True)

def clip_audio(audio_path, out_folder = 'pre-process'):

    #print(f'working on file {audio_path}')

    t0, t1 = get_t0_t1(audio_path, threshold = 0.05)

    audio = AudioSegment.from_file(audio_path)
    clip = audio[int(t0*1000):int(t1*1000)]  
    clip = clip.fade_in(20).fade_out(20)  

    name = audio_path.split('/')[-1]
    file_out = f'{out_folder}/{name}'
    clip.export(file_out, format="flac")

def get_t0_t1(audio_path, threshold = 0.05):
    t, rms = get_rms(audio_path, target_sr = 16_000)

    real_threshold = threshold*rms.max()

    t0 = t[rms > real_threshold][0] 
    t1 = t[rms > real_threshold][-1] 
    return t0, t1

def get_rms(audio_path, target_sr = 16_000):

    data, samplerate = librosa.load(audio_path, sr=target_sr, mono=True)

    hop = target_sr // 50       
    rms = librosa.feature.rms(y=data, hop_length = hop)[0]
    time = np.arange(len(rms)) / 50.0
    
    return time, rms

def get_duration_flac(path):
    info = sf.info(path)
    return info.duration

if __name__ == "__main__":

    len_in_data = sum([get_duration_flac(f) for f in files])/3600.
    print(f'Total input dataset {len_in_data} hours')

    for file in files:
        clip_audio(file, out_folder = output_dir)

    out_files = glob.glob(f'{output_dir}/*.flac')
    len_out_data = sum([get_duration_flac(f) for f in out_files])/3600
    print(f'Total output dataset {len_out_data} hours')