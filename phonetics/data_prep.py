from pathlib import Path
import pandas as pd
from pydub import AudioSegment

import os

from utils import utils as u

def splice_single_song_from_lab_file(lab_file,
                                     wav_paths,
                                     remove_ph = ['SP'],
                                     out_sr = 16000,
                                     fade_duration = 20,
                                     folder = '',
                                     suffix = ''):

    df_algn = u.df_alignments_from_lab_file(lab_file)

    path = Path(lab_file)
    song_name = path.stem

    #wavs_found = [p for p in wav_paths if song_name in p]
    files_match = [f for f in wav_paths if song_name == Path(f).stem]
    assert len(files_match) == 1, f'Issue with song {song_name}'
    wav_file = files_match[0]

    audio = AudioSegment.from_file(wav_file, format='wav')
    df_remove = df_algn[df_algn['phone_base'].isin(remove_ph)]

    # Apply the silence removal and merging
    edited_audio = remove_silences(audio, 
                                df_remove, 
                                fade_ms=fade_duration)

    edited_audio = edited_audio.set_frame_rate(out_sr)

    os.makedirs(folder, exist_ok=True)
    out_file = f'{folder}/{song_name}_spliced{suffix}.wav'
    edited_audio.export(out_file, format="wav", codec="pcm_f32le")


def remove_silences(audio: AudioSegment, 
                    silence_df: pd.DataFrame, 
                    fade_ms=20) -> AudioSegment:
    segments = []
    current_pos = 0

    for _, row in silence_df.iterrows():
        start_ms = int(row['start'] * 1000)
        end_ms = int(row['end'] * 1000)

        # Extract and apply fade out to the segment before the silence
        if start_ms > current_pos:
            segment = audio[current_pos:start_ms]
            segment = segment.fade_out(fade_ms)
            segments.append(segment)

        # Move past the silence (and apply fade in to the next chunk)
        current_pos = end_ms

    # Add the final remaining segment
    if current_pos < len(audio):
        segment = audio[current_pos:]
        segment = segment.fade_in(fade_ms)
        segments.append(segment)

    # Concatenate all remaining parts
    output_audio = sum(segments)
    return output_audio