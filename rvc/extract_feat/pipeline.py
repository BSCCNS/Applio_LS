import os
import sys
import torch
import numpy as np
from scipy import signal
import pandas as pd

now_dir = os.getcwd()
sys.path.append(now_dir)


import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

import itertools
def unique_file(basename, ext):
    actualname = f"{basename}_00000.{ext}" 
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = f"{basename}_{str(next(c)).zfill(5)}.{ext}" 
    return actualname

# Constants for high-pass filter
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="highpass", fs=SAMPLE_RATE
)

input_audio_path2wav = {}

class Pipeline:
    """
    The main pipeline class for performing voice conversion, including preprocessing, F0 estimation,
    voice conversion using a model, and post-processing.
    """

    def __init__(self, tgt_sr, config):
        """
        Initializes the Pipeline class with target sampling rate and configuration parameters.

        Args:
            tgt_sr: The target sampling rate for the output audio.
            config: A configuration object containing various parameters for the pipeline.
        """
        print(f'tgt_sr {tgt_sr}')
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.device = config.device

    def pipeline(
        self,
        model,
        audio,
        basefilename="",
        padding_for_features = True
    ):

        audio = signal.filtfilt(bh, ah, audio)

        if padding_for_features:
            df_feats = self.convert_with_padding(model, audio)
        else:
            df_feats = self.voice_conversion(model, audio)
            
        pathname = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features"
        fname = unique_file(f"{pathname}/feats_{basefilename}", "csv")

        print("feats contentvec:",df_feats.shape)
        df_feats.to_csv(fname)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    def voice_conversion(
        self,
        model,
        audio,
    ):
        with torch.no_grad():
            audio_torch = torch.from_numpy(audio.copy()).float()
            audio_torch = audio_torch.mean(-1) if audio_torch.dim() == 2 else audio_torch
            assert audio_torch.dim() == 1, audio_torch.dim()
            audio_torch = audio_torch.view(1, -1).to(self.device)

            # extract features
            feats = model(audio_torch)["last_hidden_state"]
            df_feats = pd.DataFrame(feats[0].cpu())
            return df_feats
            
    def convert_with_padding(
        self, 
        model, 
        audio
    ):
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        
        df_feats_list = []
        for t in opt_ts:
            t = t // self.window * self.window
            df_feats = self.voice_conversion(
                            model,
                            audio_pad[s : t + self.t_pad2 + self.window])
            
            df_feats = df_feats[50:-50] #HARD CODED!!! adjust to padding length
            print("Partial feats contentvec:",df_feats.shape)

            df_feats_list.append(df_feats)
            s = t
        df_feats = self.voice_conversion(
                        model,
                        audio_pad[t:])
        df_feats = df_feats[50:-50] #HARD CODED!!! adjust to padding length 

        print("Partial feats contentvec:",df_feats.shape)
        df_feats_list.append(df_feats)

        df_feats_all = pd.concat(df_feats_list)
        return df_feats_all
        