import sys
import numpy as np

from rvc.extract_feat.infer import VoiceConverter

#input_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/ES_milagro.wav'
input_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/tomas_vowels_1.wav'

params = {
'input_path': input_path, 
'embedder_model': 'contentvec',
"use_window": True,
"use_hi_filter": True
}

print(params)

def run_infer_script(
    input_path : str = '',
    embedder_model : str = '',
    use_window : bool = False,
    use_hi_filter : bool = True,
):
    kwargs = {
        "audio_input_path": input_path,
        "embedder_model": embedder_model,
        "use_window": use_window,
        "use_hi_filter": use_hi_filter
    }
    infer_pipeline = VoiceConverter() 
    infer_pipeline.convert_audio(
        **kwargs,
    )
    
    return f"File {input_path} converted successfully."

run_infer_script(**params)