import sys
import numpy as np

#from core import run_infer_script
from params_template import params


from rvc.extract_feat.infer import VoiceConverter

#input_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/tomas_vowels_1.wav'
input_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/ES_milagro.wav'
output_path = '/Users/tomasandrade/Documents/BSC/ICHOIR/applio/Applio_LS/assets/audios/tomas_vowels_1_output.wav'

input_params = {
'input_path': input_path, 
'output_path': output_path, 
'pitch': 0}

params.update(input_params)
print(params)

# def import_voice_converter():
#     from rvc.infer.infer import VoiceConverter
#     return VoiceConverter()

def get_config():
    from rvc.configs.config import Config
    return Config()

# Infer
def run_infer_script(
    pitch = int,
    index_rate = float,
    volume_envelope = int,
    protect = float,
    hop_length = int,
    f0_method = str,
    input_path = str,
    output_path = str,
    pth_path = str,
    index_path = str,
    split_audio = bool,
    f0_autotune = bool,
    f0_autotune_strength = float,
    clean_audio = bool,
    clean_strength = float,
    export_format = str,
    f0_file = str,
    embedder_model = str,
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6,
    compressor_threshold: float = 0,
    compressor_ratio: float = 1,
    compressor_attack: float = 1.0,
    compressor_release: float = 100,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
):
    kwargs = {
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": pth_path,
        "index_path": index_path,
        "pitch": pitch,
        "index_rate": index_rate,
        "volume_envelope": volume_envelope,
        "protect": protect,
        "hop_length": hop_length,
        "f0_method": f0_method,
        "pth_path": pth_path,
        "index_path": index_path,
        "split_audio": split_audio,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "f0_file": f0_file,
        "embedder_model": embedder_model,
        "embedder_model_custom": embedder_model_custom,
        "post_process": post_process,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "reverb": reverb,
        "pitch_shift": pitch_shift,
        "limiter": limiter,
        "gain": gain,
        "distortion": distortion,
        "chorus": chorus,
        "bitcrush": bitcrush,
        "clipping": clipping,
        "compressor": compressor,
        "delay": delay,
        "reverb_room_size": reverb_room_size,
        "reverb_damping": reverb_damping,
        "reverb_wet_level": reverb_wet_gain,
        "reverb_dry_level": reverb_dry_gain,
        "reverb_width": reverb_width,
        "reverb_freeze_mode": reverb_freeze_mode,
        "pitch_shift_semitones": pitch_shift_semitones,
        "limiter_threshold": limiter_threshold,
        "limiter_release": limiter_release_time,
        "gain_db": gain_db,
        "distortion_gain": distortion_gain,
        "chorus_rate": chorus_rate,
        "chorus_depth": chorus_depth,
        "chorus_delay": chorus_center_delay,
        "chorus_feedback": chorus_feedback,
        "chorus_mix": chorus_mix,
        "bitcrush_bit_depth": bitcrush_bit_depth,
        "clipping_threshold": clipping_threshold,
        "compressor_threshold": compressor_threshold,
        "compressor_ratio": compressor_ratio,
        "compressor_attack": compressor_attack,
        "compressor_release": compressor_release,
        "delay_seconds": delay_seconds,
        "delay_feedback": delay_feedback,
        "delay_mix": delay_mix,
        "sid": sid,
    }
    infer_pipeline = VoiceConverter() 
    infer_pipeline.convert_audio(
        **kwargs,
    )
    
    return f"File {input_path} inferred successfully.", output_path.replace(
        ".wav", f".{export_format.lower()}"
    )

run_infer_script(**params)