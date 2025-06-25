
import glob
from rvc.extract_feat.infer import VoiceConverter

#INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/Ha/U'
#OUTPUT_FEAT_PATH = "/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features/Ha_maria/U"

INPUT_WAV_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/ls_features/1_16k_wavs_diffsinger'
OUTPUT_FEAT_PATH = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/features/diffsinger_window_filter'

infer_pipeline = VoiceConverter(
                embedder_model = "contentvec",
                use_window = True,
                use_hi_filter = True,
                output_feat_path= f'{OUTPUT_FEAT_PATH}'
                ) 

files = glob.glob(f'{INPUT_WAV_PATH}/*.wav')

for input in files:
    infer_pipeline.convert_audio(input)

