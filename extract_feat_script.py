import sys
import glob
import os
from rvc.extract_feat.infer import VoiceConverter

#INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/study_phonemes_contentvec/libri/flac'
#OUTPUT_FEAT_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/study_phonemes_contentvec/libri/feat'

args = sys.argv[1:]

INPUT_WAV_PATH = args[0]
OUTPUT_FEAT_PATH = args[1]
ext = args[2]

os.makedirs(OUTPUT_FEAT_PATH, exist_ok=True)

N_LAYER = 9

infer_pipeline = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = False,
                output_feat_path= f'{OUTPUT_FEAT_PATH}',
                extract_inner_layers = True,
                n_layer = N_LAYER
                ) 

files = glob.glob(f'{INPUT_WAV_PATH}/*.{ext}')

for input in files:
    infer_pipeline.convert_audio(input)

