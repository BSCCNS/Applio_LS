import os
import sys
import time
import torch
#import librosa
#import logging
import traceback
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

from transformers import HubertModel
from torch import nn

from scipy import signal

# Constants for high-pass filter
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="highpass", fs=SAMPLE_RATE
)


now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.extract_feat.pipeline import Pipeline as VC
from rvc.lib.utils import load_audio_infer #, load_embedding
#from rvc.lib.tools.split_audio import process_audio, merge_audio
#from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

EMBEDDERS_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/rvc/models/embedders/'
#FEAT_PATH = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features"

class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self, 
        embedder_model: str = "contentvec",
        use_window: bool = False,
        use_hi_filter: bool = True,
        output_feat_path = '',
        extract_inner_layers = False,
        n_layer = -1):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load RVC configuration
        self.use_window = use_window
        self.use_hi_filter = use_hi_filter
        self.output_feat_path = output_feat_path

        self.extract_inner_layers = extract_inner_layers
        self.n_layer = n_layer

        self.load_hubert(embedder_model)
        self.make_output_dir()

    def make_output_dir(self):
        print(f'----- Saving outputs to folder {self.output_feat_path}')
        os.makedirs(self.output_feat_path, exist_ok=True)

    def load_hubert(self, embedder_model: str):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.
        """
        emb_path = f'{EMBEDDERS_PATH}/{embedder_model}'
        self.hubert_model = HubertModelWithFinalProj.from_pretrained(emb_path)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    def convert_audio(
        self,
        audio_input_path: str,
        **kwargs,
    ):
        
        try:
            start_time = time.time()
            print(f"Converting audio '{audio_input_path}'...")

            audio = load_audio_infer(
                audio_input_path,
                16000,
                **kwargs,
            )

            #basefilename = os.path.basename(audio_input_path)[:-4]
            basefilename = Path(audio_input_path).stem
            feat_extraction(
                self.hubert_model, 
                audio, 
                self.config, 
                basefilename = basefilename,
                use_window = self.use_window,
                use_hi_filter = self.use_hi_filter,
                out_folder = self.output_feat_path,
                extract_inner_layers = self.extract_inner_layers,
                n_layer = self.n_layer
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed in {elapsed_time:.2f} seconds."
            )
        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def feat_extraction(
    model, 
    audio, 
    config, 
    basefilename = '',
    use_window = False,
    use_hi_filter = True,
    out_folder = '',
    extract_inner_layers = False,
    n_layer = -1
):

    if use_hi_filter:
        audio = signal.filtfilt(bh, ah, audio)
    else:
        pass

    if use_window:
        print('Extraction with window')
        df_feats = extraction_with_window(model, 
                                          audio, 
                                          config, 
                                          extract_inner_layers = extract_inner_layers,
                                          n_layer = n_layer)
    else:
        print('Extraction without window')
        df_feats = single_extraction(model, 
                                    audio, 
                                    config, 
                                    extract_inner_layers = extract_inner_layers,
                                    n_layer = n_layer)
        
    fname = f"{out_folder}/feats_{basefilename}.csv"

    print(f"feats contentvec: {df_feats.shape}")
    df_feats.to_csv(fname)

def single_extraction(
    model, 
    audio, 
    config, 
    extract_inner_layers = False,
    n_layer = -1
):
    #extract_inner_layers = kwargs.get('extract_inner_layers', False)
    #n_layer = kwargs.get('extract_inner_layers', False)

    with torch.no_grad():
        audio_torch = torch.from_numpy(audio.copy()).float()
        audio_torch = audio_torch.mean(-1) if audio_torch.dim() == 2 else audio_torch
        assert audio_torch.dim() == 1, audio_torch.dim()
        audio_torch = audio_torch.view(1, -1).to(config.device)
        
        if extract_inner_layers:
            # Forward pass with output_hidden_states=True
            with torch.no_grad():
                outputs = model(audio_torch, output_hidden_states=True)

            # All hidden states (includes embeddings + outputs of each transformer layer)
            hidden_states = outputs["hidden_states"] 
            feats = hidden_states[n_layer]

        else:
            feats = model(audio_torch)["last_hidden_state"]

    df_feats = pd.DataFrame(feats[0].cpu())
    return df_feats

def extraction_with_window(model, audio, config, **kwargs):

    x_pad = config.x_pad
    x_query = config.x_query
    x_center = config.x_center
    x_max = config.x_max
    sample_rate = 16000
    window = 160
    t_pad = sample_rate * x_pad
    #self.t_pad_tgt = tgt_sr * self.x_pad
    t_pad2 = t_pad * 2
    t_query = sample_rate * x_query
    t_center = sample_rate * x_center
    t_max = sample_rate * x_max
    #time_step = window / sample_rate * 1000

    audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
    opt_ts = []
    if audio_pad.shape[0] > t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(window):
            audio_sum += audio_pad[i : i - window]
        for t in range(t_center, audio.shape[0], t_center):
            opt_ts.append(
                t
                - t_query
                + np.where(
                    np.abs(audio_sum[t - t_query : t + t_query])
                    == np.abs(audio_sum[t - t_query : t + t_query]).min()
                )[0][0]
            )
    s = 0
    t = None
    audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")
    
    df_feats_list = []
    for t in opt_ts:
        t = t // window * window
        df_feats = single_extraction(
                    model,
                    audio_pad[s : t + t_pad2 + window],
                    config)
                
        df_feats = df_feats[50:-50] #HARD CODED!!! adjust to padding length
        print("Partial feats contentvec:",df_feats.shape)

        df_feats_list.append(df_feats)
        s = t
    df_feats = single_extraction(
                    model,
                    audio_pad[t:],
                    config)
    df_feats = df_feats[50:-50] #HARD CODED!!! adjust to padding length 

    print("Partial feats contentvec:",df_feats.shape)
    df_feats_list.append(df_feats)

    df_feats_all = pd.concat(df_feats_list).reset_index(drop = True)
    return df_feats_all