import sys
import glob
import os
from rvc.extract_feat.infer import VoiceConverter
import time
import logging
import argparse

parser = argparse.ArgumentParser(
                    prog='extract feat',
                    description='extract contentvec features',
                    epilog='Ask me for help')

# Define named arguments
parser.add_argument('--input', type=str, required=True, help="Path to the input audio folder")
parser.add_argument('--output', type=str, required=True, help="Path to the output feat folder")
parser.add_argument('--ext', type=str, required=True, help="Extension of the audio file e.g flac")
parser.add_argument('--layer', default=None)

args = parser.parse_args()

###############################################
# args = sys.argv[1:]

INPUT_WAV_PATH = args.input
OUTPUT_FEAT_PATH = args.output
ext = args.ext
layer = args.layer
###############################################

def setup_logs(logs_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_path),   # Logs to a file
            logging.StreamHandler()           # Prints to console
        ]
    )
    logging.info(f'------------- logs output to {logs_path}')

###############################################

logs_path = 'output.log'
setup_logs(logs_path)

###############################################

if layer is None:
    logging.info(f'-------- Computing for all layers')
    min_layer = 1
    max_layer = 12
else:
    logging.info(f'-------- Computing for single layer {layer}')
    min_layer = layer
    max_layer = layer

files = glob.glob(f'{INPUT_WAV_PATH}/*.{ext}')

os.makedirs(OUTPUT_FEAT_PATH, exist_ok=True)

T0 = time.time()

#for n_layer in n_layers:
for n_layer in range(min_layer, max_layer + 1):

    t0 = time.time()
    logging.info(f'------- working on layer {n_layer}')
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