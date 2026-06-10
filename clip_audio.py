"""
Audio preprocessing with internal silence removal
--------------------------------------------------
Removes:
  1. Leading / trailing silence (existing behaviour)
  2. Internal silence segments longer than MAX_INTERNAL_SILENCE_S

For long internal silences, the audio is split into speech segments
and concatenated with a short natural-sounding pause in between.
This preserves natural breath pauses (< threshold) while removing
artefact silences (> threshold).
"""

import os
import numpy as np
import librosa
from pydub import AudioSegment
from scipy.ndimage import uniform_filter1d


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

MAX_INTERNAL_SILENCE_S = 0.5    # silences longer than this are artefacts
NATURAL_PAUSE_MS       = 200    # replace long silences with this short pause
                                 # (sounds natural, ~1 breath pause)
RMS_THRESHOLD_FACTOR   = 0.1   # fraction of peak RMS → silence
MIN_SPEECH_FRAMES      = 3      # min consecutive frames to count as speech


# -----------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------

def get_rms(audio_path, target_sr=16_000):
    data, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    hop      = target_sr // 50        # 20ms frames
    rms      = librosa.feature.rms(y=data, hop_length=hop)[0]
    time     = np.arange(len(rms)) / 50.0
    return time, rms


def get_speech_segments(audio_path, threshold=RMS_THRESHOLD_FACTOR,
                        min_speech_frames=MIN_SPEECH_FRAMES,
                        max_internal_silence_s=MAX_INTERNAL_SILENCE_S):
    """
    Returns list of (t_start, t_end) tuples for each speech segment,
    after merging silence gaps shorter than max_internal_silence_s.

    Long internal silences (artefacts) create separate segments.
    Short silences (natural pauses / breath) are merged into one segment.
    """
    t, rms = get_rms(audio_path)
    real_threshold = threshold * rms.max()
    above    = rms > real_threshold

    # Smooth to remove isolated spikes
    smoothed  = uniform_filter1d(above.astype(float), size=min_speech_frames)
    is_speech = smoothed >= 1.0

    frame_dur = t[1] - t[0]   # seconds per frame (0.02s)
    max_sil_frames = int(max_internal_silence_s / frame_dur)

    # Find contiguous speech runs
    segments = []
    in_speech = False
    seg_start = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            in_speech  = True
            seg_start  = i
        elif not speech and in_speech:
            in_speech  = False
            segments.append((seg_start, i - 1))
    if in_speech:
        segments.append((seg_start, len(is_speech) - 1))

    if not segments:
        return [(0, len(t) - 1)]   # fallback: keep everything

    # Merge segments separated by short silences
    merged = [segments[0]]
    for start, end in segments[1:]:
        gap = start - merged[-1][1]
        if gap <= max_sil_frames:
            # Short gap — merge into current segment
            merged[-1] = (merged[-1][0], end)
        else:
            # Long gap — keep as separate segment (will be concatenated)
            merged.append((start, end))

    # Convert frame indices to times
    return [(t[s], t[e]) for s, e in merged]


def get_t0_t1(audio_path, threshold=RMS_THRESHOLD_FACTOR,
              min_speech_frames=MIN_SPEECH_FRAMES):
    """
    Original boundary trimming — finds first and last sustained speech.
    Used as fallback / for compatibility.
    """
    t, rms = get_rms(audio_path)
    real_threshold = threshold * rms.max()
    above    = rms > real_threshold
    smoothed = uniform_filter1d(above.astype(float), size=min_speech_frames)
    sustained = smoothed >= 1.0

    valid_frames = t[sustained]
    if len(valid_frames) == 0:
        return t[0], t[-1]

    return valid_frames[0], valid_frames[-1]


def clip_audio(audio_path, out_folder='pre-process',
               max_internal_silence_s=MAX_INTERNAL_SILENCE_S,
               natural_pause_ms=NATURAL_PAUSE_MS):
    """
    Trim boundary silences AND remove long internal silence artefacts.

    If multiple speech segments are found after removing long silences,
    they are concatenated with a short natural pause between them.

    Parameters
    ----------
    audio_path            : path to input flac/wav
    out_folder            : output directory
    max_internal_silence_s: silences longer than this are removed
    natural_pause_ms      : duration of replacement pause between segments
    """
    import os
    os.makedirs(out_folder, exist_ok=True)

    segments = get_speech_segments(
        audio_path,
        max_internal_silence_s=max_internal_silence_s
    )

    audio = AudioSegment.from_file(audio_path)
    pause = AudioSegment.silent(duration=natural_pause_ms)

    # Extract and concatenate speech segments
    clips = []
    for t0, t1 in segments:
        t0_ms = int(t0 * 1000)
        t1_ms = int(t1 * 1000)
        if t1_ms > t0_ms:
            clips.append(audio[t0_ms:t1_ms])

    if not clips:
        # Fallback: just boundary trim
        t0, t1 = get_t0_t1(audio_path)
        clip = audio[int(t0 * 1000):int(t1 * 1000)]
    elif len(clips) == 1:
        clip = clips[0]
    else:
        # Join segments with a short natural pause
        clip = clips[0]
        for c in clips[1:]:
            clip = clip + pause + c

    clip = clip.fade_in(20).fade_out(20)

    name     = os.path.basename(audio_path)
    file_out = os.path.join(out_folder, name)
    clip.export(file_out, format="flac")

    return len(clips)   # return number of segments (useful for logging)


# -----------------------------------------------------------------------
# Batch processing with logging
# -----------------------------------------------------------------------

def process_dataset(audio_paths, out_folder='pre-process',
                    max_internal_silence_s=MAX_INTERNAL_SILENCE_S):
    """
    Process a list of audio files, logging any multi-segment cases.
    """
    import os
    from tqdm import tqdm

    os.makedirs(out_folder, exist_ok=True)
    multi_segment_log = []

    for path in tqdm(audio_paths, desc='Clipping'):
        try:
            n_segs = clip_audio(
                path, out_folder,
                max_internal_silence_s=max_internal_silence_s
            )
            if n_segs > 1:
                multi_segment_log.append((os.path.basename(path), n_segs))
        except Exception as e:
            print(f"ERROR {path}: {e}")

    if multi_segment_log:
        print(f"\n{len(multi_segment_log)} files had long internal silences removed:")
        for name, n in sorted(multi_segment_log, key=lambda x: -x[1])[:20]:
            print(f"  {name}: {n} segments")

    return multi_segment_log


# -----------------------------------------------------------------------
# Entry point — pass input folder, optional output folder
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import glob

    usage = """
Usage:
  python clip_audio.py <in_folder> [out_folder]

Arguments:
  in_folder   : folder containing .flac files (searched recursively)
  out_folder  : where to save processed files (default: pre-process)

Examples:
  python clip_audio.py /data/ASVspoof/LA/train/flac
  python clip_audio.py /data/ASVspoof/LA/train/flac /data/processed/train

Config (edit at top of script):
  MAX_INTERNAL_SILENCE_S = {max_sil}s  — silences longer than this are removed
  NATURAL_PAUSE_MS       = {pause}ms  — replacement pause between segments
  RMS_THRESHOLD_FACTOR   = {thresh}   — fraction of peak RMS → silence
""".format(
        max_sil=MAX_INTERNAL_SILENCE_S,
        pause=NATURAL_PAUSE_MS,
        thresh=RMS_THRESHOLD_FACTOR,
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(0)

    in_folder  = sys.argv[1]
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'pre-process'

    if not os.path.isdir(in_folder):
        print(f"ERROR: '{in_folder}' is not a directory")
        sys.exit(1)

    # Find all flac files recursively
    files = sorted(glob.glob(os.path.join(in_folder, '**', '*.flac'),
                             recursive=True))

    # Also catch wav files if any
    files += sorted(glob.glob(os.path.join(in_folder, '**', '*.wav'),
                              recursive=True))

    if len(files) == 0:
        print(f"No .flac or .wav files found in '{in_folder}'")
        sys.exit(1)

    print(f"Input folder  : {in_folder}")
    print(f"Output folder : {out_folder}")
    print(f"Files found   : {len(files)}")
    print(f"Max internal silence threshold: {MAX_INTERNAL_SILENCE_S}s")
    print()

    log = process_dataset(files, out_folder=out_folder)

    # Save log of multi-segment files
    if log:
        log_path = os.path.join(out_folder, 'multi_segment_log.txt')
        with open(log_path, 'w') as f:
            f.write("# Files where long internal silences were removed\n")
            f.write("# name | n_segments\n\n")
            for name, n in sorted(log, key=lambda x: -x[1]):
                f.write(f"{name}  {n} segments\n")
        print(f"\nMulti-segment log saved to {log_path}")

    print(f"\nDone. {len(files)} files processed → {out_folder}")
