import sys
import glob
import os
from rvc.extract_feat.infer import VoiceConverter
import torch

import time

#INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/study_phonemes_contentvec/libri/flac'
#OUTPUT_FEAT_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/study_phonemes_contentvec/libri/feat'

if torch.cuda.is_available():
    print("CUDA is available. GPU in use:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

args = sys.argv[1:]

INPUT_WAV_PATH = args[0]
ext = 'wav'
input_files = glob.glob(f'{INPUT_WAV_PATH}/*.{ext}')

# OUTPUT_FEAT_PATH = args[1]
# ext = args[2]

OUTPUT_FEAT_PATH_LAST = './feat_test_last'
OUTPUT_FEAT_PATH_12 = './feat_test_12'

print('------ Extracting last layer')
os.makedirs(OUTPUT_FEAT_PATH_LAST, exist_ok=True)
infer_pipeline_last = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = False,
                output_feat_path= f'{OUTPUT_FEAT_PATH_LAST}',
                extract_inner_layers = False,
                n_layer = None
                ) 
for input in input_files:
    infer_pipeline_last.convert_audio(input)

print('------ Extracting 12th layer')
os.makedirs(OUTPUT_FEAT_PATH_12, exist_ok=True)
infer_pipeline_12 = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = False,
                output_feat_path= f'{OUTPUT_FEAT_PATH_12}',
                extract_inner_layers = True,
                n_layer = 12
                ) 
for input in input_files:
    infer_pipeline_12.convert_audio(input)


