
import glob
from rvc.extract_feat.infer import VoiceConverter

INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios'
OUTPUT_FEAT_PATH = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features"

infer_pipeline = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = True,
                output_feat_path= f'{OUTPUT_FEAT_PATH}/experiment_1'
                ) 

files = glob.glob(f'{INPUT_WAV_PATH}/*.wav')
print(files)

for input in files:
    infer_pipeline.convert_audio(input)

