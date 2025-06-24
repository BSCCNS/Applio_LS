import sys
import numpy as np

from rvc.extract_feat.infer import VoiceConverter

#input_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/ES_milagro.wav'
input_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/tomas_vowels_1.wav'
output_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/tomas_vowels_1_output.wav'

params = {
'input_path': input_path, 
'output_path': output_path,
'pth_path': 'logs/maria-20/maria-20.pth', 
'embedder_model': 'contentvec', 
'sid': 0}

print(params)

def run_infer_script(
    input_path = str,
    output_path = str,
    pth_path = str,
    embedder_model = str,
    sid: int = 0,
):
    kwargs = {
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": pth_path,
        "embedder_model": embedder_model,
        "sid": sid,
    }
    infer_pipeline = VoiceConverter() 
    infer_pipeline.convert_audio(
        **kwargs,
    )
    
    return f"File {input_path} inferred successfully."

run_infer_script(**params)