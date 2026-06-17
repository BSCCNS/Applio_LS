import os
import glob

import numpy as np

import librosa
import soundfile as sf
from pydub import AudioSegment

ROOT = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019'

dev_input_folder  = f"{ROOT}/LA/ASVspoof2019_LA_dev/flac"
dev_output_folder = f"{ROOT}/ASVspoof2019_LA_dev_preproc/flac"

train_input_folder  = f"{ROOT}/LA/ASVspoof2019_LA_train/flac"
train_output_folder = f"{ROOT}/ASVspoof2019_LA_train_preproc/flac"

eval_input_folder  = f"{ROOT}/LA/ASVspoof2019_LA_eval/flac"
eval_output_folder = f"{ROOT}/ASVspoof2019_LA_eval_preproc/flac"

def process_files(input_folder, output_folder):

    print(f'---- Working on input folder {input_folder}')

    files = sorted(glob.glob(f'{input_folder}/*.flac'))
    os.makedirs(output_folder, exist_ok=True)

    print(f'---- Created folder {output_folder}')

    len_in_data = sum([get_duration_flac(f) for f in files])/3600.
    print(f'---- Total input dataset {len_in_data} hours')

    for i, file in enumerate(files):
        print(f'{i}/{len(files)}')
        clip_audio(file, out_folder = output_folder)

    out_files = glob.glob(f'{output_folder}/*.flac')
    len_out_data = sum([get_duration_flac(f) for f in out_files])/3600
    print(f'---- Total output dataset {len_out_data} hours')

def clip_audio(audio_path, out_folder = 'pre-process'):
    t0, t1 = get_t0_t1(audio_path, threshold = 0.05)

    audio = AudioSegment.from_file(audio_path)
    clip = audio[int(t0*1000):int(t1*1000)]  
    clip = clip.fade_in(20).fade_out(20)  

    name = audio_path.split('/')[-1]
    file_out = f'{out_folder}/{name}'
    clip.export(file_out, format="flac")

def get_t0_t1(audio_path, threshold=0.05, min_speech_frames=3):
    t, rms = get_rms(audio_path, target_sr=16_000)
    real_threshold = threshold * rms.max()
    above = rms > real_threshold

    # Remove isolated spikes — require at least min_speech_frames
    # consecutive frames above threshold to count as real speech
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(above.astype(float), size=min_speech_frames)
    sustained = smoothed >= 1.0

    valid_frames = t[sustained]
    if len(valid_frames) == 0:
        return t[0], t[-1]  # fallback: keep everything

    t0 = valid_frames[0]
    t1 = valid_frames[-1]
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

    #process_files(train_input_folder, train_output_folder)
    #process_files(dev_input_folder, dev_output_folder)
    process_files(eval_input_folder, eval_output_folder)