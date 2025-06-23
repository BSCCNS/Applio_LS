import os
# import gc
# import re
import sys
import torch
#import torch.nn.functional as F
# import torchcrepe
# import faiss
# import librosa
import numpy as np
from scipy import signal
#from torch import Tensor
import pandas as pd

now_dir = os.getcwd()
sys.path.append(now_dir)

# from rvc.lib.predictors.RMVPE import RMVPE0Predictor
# from rvc.lib.predictors.FCPE import FCPEF0Predictor

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

    def voice_conversion(
        self,
        model,
        audio0,
        basefilename=""
    ):
        """
        Performs voice conversion on a given audio segment.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio0: The input audio segment.
            pitch: Quantized F0 contour for pitch guidance.
            pitchf: Original F0 contour for pitch guidance.
            index: FAISS index for speaker embedding retrieval.
            big_npy: Speaker embeddings stored in a NumPy array.
            index_rate: Blending rate for speaker embedding retrieval.
            version: Model version ("v1" or "v2").
            protect: Protection level for preserving the original pitch.
        """
        with torch.no_grad():
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1).to(self.device)
            # extract features
        
            feats = model(feats)["last_hidden_state"]
            
            pathname = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features"

            print("Feats del modelo again:",feats.shape)
            fname = unique_file(f"{pathname}/feats_pre_index_{basefilename}", "csv")
            exportable = pd.DataFrame(feats[0].cpu())
            exportable.to_csv(fname)
            
    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        filter_radius,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_autotune,
        f0_autotune_strength,
        f0_file,
        basefilename="",
        padding_pipeline = True
    ):
        """
        The main pipeline function for performing voice conversion.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio: The input audio signal.
            input_audio_path: Path to the input audio file.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            file_index: Path to the FAISS index file for speaker embedding retrieval.
            index_rate: Blending rate for speaker embedding retrieval.
            pitch_guidance: Whether to use pitch guidance during voice conversion.
            filter_radius: Radius for median filtering the F0 contour.
            tgt_sr: Target sampling rate for the output audio.
            resample_sr: Resampling rate for the output audio.
            volume_envelope: Blending rate for adjusting the RMS level of the output audio.
            version: Model version.
            protect: Protection level for preserving the original pitch.
            hop_length: Hop length for F0 estimation methods.
            f0_autotune: Whether to apply autotune to the F0 contour.
            f0_file: Path to a file containing an F0 contour to use.
        """

        audio = signal.filtfilt(bh, ah, audio)

        if padding_pipeline:
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
            
            for t in opt_ts:
                t = t // self.window * self.window
                self.voice_conversion(
                        model,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        basefilename=basefilename)
                s = t
            self.voice_conversion(
                model,
                audio_pad[t:],
                basefilename=basefilename) 
        else:
            self.voice_conversion(
                model,
                audio,
                basefilename=basefilename+"no_pad") 

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

