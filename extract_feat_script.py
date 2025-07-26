import sys
import glob
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
OUTPUT_FEAT_PATH = args[1]
ext = args[2]

t0 = time.time()

infer_pipeline = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = False,
                output_feat_path= f'{OUTPUT_FEAT_PATH}'
                ) 

files = glob.glob(f'{INPUT_WAV_PATH}/*.{ext}')

for input in files:
    infer_pipeline.convert_audio(input)

t1 = time.time()
DT = t1 - t0
print(f'-------- Total time {DT:.4f}')