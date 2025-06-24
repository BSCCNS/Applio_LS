import os
import sys
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf

from transformers import HubertModel
from torch import nn


now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.extract_feat.pipeline import Pipeline as VC
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

EMBEDDERS_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/rvc/models/embedders/'

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
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0
        self.loaded_model = None

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
        model_path: str,
        embedder_model: str = "contentvec",
        resample_sr: int = 0,
        sid: int = 0,
        **kwargs,
    ):
        """
        Performs voice conversion on the input audio.

        Args:
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            embedder_model (str): Path to the embedder model.
            embedder_model_custom (str): Path to the custom embedder model.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        self.get_vc(model_path, sid)
        try:
            start_time = time.time()
            print(f"Converting audio '{audio_input_path}'...")

            audio = load_audio_infer(
                audio_input_path,
                16000,
                **kwargs,
            )

            self.load_hubert(embedder_model)

            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            basefilename = os.path.basename(audio_input_path)[:-4]

            audio_opt = self.vc.pipeline(
                model=self.hubert_model,
                audio=audio,
                basefilename=basefilename,
                padding_for_features=True
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
            )
        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())

    def get_vc(self, weight_root, sid):
        """
        Loads the voice conversion model and sets up the pipeline.

        Args:
            weight_root (str): Path to the model weights.
            sid (int): Speaker ID.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root)
            if self.cpt is not None:
                self.setup_network()
                self.setup_vc_instance()
            self.loaded_model = weight_root

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu")
            if os.path.isfile(weight_root)
            else None
        )

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            # extra TA-changes
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                #is_half=self.config.is_half,
                vocoder=self.vocoder,
            )
            # del self.net_g.enc_q
            # self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            # self.net_g.eval().to(self.config.device)
            # self.net_g = (
            #     self.net_g.half() if self.config.is_half else self.net_g.float()
            # )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]
