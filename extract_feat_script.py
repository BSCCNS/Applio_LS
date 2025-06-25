
import glob
from rvc.extract_feat.infer import VoiceConverter

INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/Ha/U'
OUTPUT_FEAT_PATH = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features/Ha_maria/U"

infer_pipeline = VoiceConverter(
                embedder_model = "contentvec",
                use_window = False,
                use_hi_filter = False,
                output_feat_path= f'{OUTPUT_FEAT_PATH}'
                ) 

files = glob.glob(f'{INPUT_WAV_PATH}/*.wav')

for input in files:
    infer_pipeline.convert_audio(input)

