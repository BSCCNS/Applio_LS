import os
import sys
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import pandas as pd
import soundfile as sf

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
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

EMBEDDERS_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/rvc/models/embedders/'
FEAT_PATH = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features"

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load RVC configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )

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
        audio_output_path: str,
        embedder_model: str = "contentvec",
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

            self.load_hubert(embedder_model)

            basefilename = os.path.basename(audio_input_path)[:-4]

            feat_extraction_external(
                self.hubert_model, 
                audio, 
                self.config.device, 
                basefilename = basefilename
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
            )
        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())

def feat_extraction_external(model, audio, device, basefilename = ''):

    audio = signal.filtfilt(bh, ah, audio)

    with torch.no_grad():
        audio_torch = torch.from_numpy(audio.copy()).float()
        audio_torch = audio_torch.mean(-1) if audio_torch.dim() == 2 else audio_torch
        assert audio_torch.dim() == 1, audio_torch.dim()
        audio_torch = audio_torch.view(1, -1).to(device)

        # extract features
        feats = model(audio_torch)["last_hidden_state"]
        df_feats = pd.DataFrame(feats[0].cpu())
        return df_feats
    
    fname = unique_file(f"{FEAT_PATH}/feats_{basefilename}", "csv")

    print("feats contentvec:",df_feats.shape)
    df_feats.to_csv(fname)

import itertools
def unique_file(basename, ext):
    actualname = f"{basename}_00000.{ext}" 
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = f"{basename}_{str(next(c)).zfill(5)}.{ext}" 
    return actualname