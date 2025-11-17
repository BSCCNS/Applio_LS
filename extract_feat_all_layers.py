import sys
import glob
import os
from rvc.extract_feat.infer import VoiceConverter
import time

#INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/study_phonemes_contentvec/libri/flac'
#OUTPUT_FEAT_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/study_phonemes_contentvec/libri/feat'

args = sys.argv[1:]

INPUT_WAV_PATH = args[0]
OUTPUT_FEAT_PATH = args[1]
ext = args[2]

n_layers = list(range(1,13))
files = glob.glob(f'{INPUT_WAV_PATH}/*.{ext}')

os.makedirs(OUTPUT_FEAT_PATH, exist_ok=True)

T0 = time.time()

for n_layer in n_layers:

    t0 = time.time()
    print(f'------- working on layer {n_layer}')
    output_layer = f'{OUTPUT_FEAT_PATH}/layer_{n_layer}'
    os.makedirs(output_layer, exist_ok=True)

    infer_pipeline = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = False,
                output_feat_path= output_layer,
                extract_inner_layers = True,
                n_layer = n_layer
                ) 

    for input in files:
        infer_pipeline.convert_audio(input)

    t1 = time.time()
    dt = t1 - t0

    print(f'------ Time for layer {n_layer}: {dt}')

T1 = time.time()

DT = T1 - T0
print(f'------ Total Time: {DT}')