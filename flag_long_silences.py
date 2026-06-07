"""
Flag utterances with potentially problematic internal silences
--------------------------------------------------------------
After RMS-based boundary trimming, some utterances may still contain
long internal silences that are artefacts rather than natural breath pauses.

This script analyses the RMS energy of each audio file and flags cases where:
  1. Internal silence duration exceeds a threshold (likely artefact)
  2. Silence fraction is unusually high (mostly silence)
  3. Multiple long silence segments (fragmented/concatenated audio)

Outputs:
  - flagged_silences.csv  : all flagged utterances with stats
  - silence_report.png    : distribution plots for manual inspection

Usage:
  Point AUDIO_DIR to your preprocessed flac files.
  Adjust thresholds to match your expected utterance characteristics.
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

AUDIO_DIR   = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019/ASVspoof2019_LA_train_preproc/flac/'   # ← change this
OUT_DIR     = 'silence_diagnostics'
TARGET_SR   = 16_000
FRAME_LEN   = 0.02                 # 20ms RMS frames (matches HuBERT)
HOP_LEN     = 0.02                 # non-overlapping

# Silence detection
RMS_THRESHOLD_FACTOR = 0.05        # fraction of file peak RMS → silence
                                   # same as your clip_audio preprocessing

# Flagging thresholds — adjust based on your data
MAX_INTERNAL_SILENCE_S  = 0.5     # flag if any internal silence > 500ms
MAX_SILENCE_FRACTION    = 0.40    # flag if >40% of utterance is silence
MIN_SPEECH_DURATION_S   = 0.5     # flag if <500ms of actual speech
MAX_SILENCE_SEGMENTS    = 3       # flag if more than 3 separate silence runs

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Per-file analysis
# -----------------------------------------------------------------------

def analyse_file(audio_path):
    """
    Returns a dict of silence statistics for one audio file.
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR)
    duration = len(y) / sr

    # Frame-level RMS
    frame_samples = int(FRAME_LEN * sr)
    hop_samples   = int(HOP_LEN  * sr)
    rms = librosa.feature.rms(
        y=y, frame_length=frame_samples, hop_length=hop_samples
    )[0]

    # Silence mask (same threshold as clip_audio)
    threshold    = RMS_THRESHOLD_FACTOR * rms.max()
    is_silence   = rms < threshold
    frame_dur    = HOP_LEN  # seconds per frame

    # Find contiguous silence runs
    silence_runs = []
    in_silence   = False
    run_start    = 0

    for i, sil in enumerate(is_silence):
        if sil and not in_silence:
            in_silence = True
            run_start  = i
        elif not sil and in_silence:
            in_silence = False
            silence_runs.append((run_start, i - 1))
    if in_silence:
        silence_runs.append((run_start, len(is_silence) - 1))

    # Separate boundary vs internal silences
    # Boundary = leading (before first speech) or trailing (after last speech)
    speech_frames = np.where(~is_silence)[0]
    if len(speech_frames) == 0:
        # Entirely silent — definitely flag
        return dict(
            duration_s          = duration,
            silence_fraction    = 1.0,
            speech_duration_s   = 0.0,
            n_silence_segments  = 1,
            max_internal_sil_s  = duration,
            max_boundary_sil_s  = 0.0,
            n_internal_sil_segs = 1,
            flag_all_silent     = True,
            flag_long_internal  = True,
            flag_high_fraction  = True,
            flag_short_speech   = True,
            flag_many_segments  = False,
        )

    first_speech = speech_frames[0]
    last_speech  = speech_frames[-1]

    # Classify runs as boundary or internal
    internal_runs  = []
    boundary_runs  = []
    for start, end in silence_runs:
        if end < first_speech or start > last_speech:
            boundary_runs.append((start, end))
        else:
            internal_runs.append((start, end))

    # Statistics
    internal_durations = [(e - s + 1) * frame_dur for s, e in internal_runs]
    boundary_durations = [(e - s + 1) * frame_dur for s, e in boundary_runs]

    max_internal_sil = max(internal_durations) if internal_durations else 0.0
    max_boundary_sil = max(boundary_durations) if boundary_durations else 0.0
    total_silence    = sum(internal_durations) + sum(boundary_durations)
    silence_fraction = total_silence / duration if duration > 0 else 0.0
    speech_duration  = duration - total_silence
    n_internal_segs  = len(internal_runs)

    # Flags
    flag_long_internal = max_internal_sil > MAX_INTERNAL_SILENCE_S
    flag_high_fraction = silence_fraction  > MAX_SILENCE_FRACTION
    flag_short_speech  = speech_duration   < MIN_SPEECH_DURATION_S
    flag_many_segments = n_internal_segs   > MAX_SILENCE_SEGMENTS

    return dict(
        duration_s          = round(duration, 3),
        silence_fraction    = round(silence_fraction, 3),
        speech_duration_s   = round(speech_duration, 3),
        n_silence_segments  = len(silence_runs),
        n_internal_sil_segs = n_internal_segs,
        max_internal_sil_s  = round(max_internal_sil, 3),
        max_boundary_sil_s  = round(max_boundary_sil, 3),
        flag_all_silent     = False,
        flag_long_internal  = flag_long_internal,
        flag_high_fraction  = flag_high_fraction,
        flag_short_speech   = flag_short_speech,
        flag_many_segments  = flag_many_segments,
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':

    audio_dir = Path(AUDIO_DIR)
    files     = sorted(audio_dir.glob('**/*.flac'))
    print(f"Found {len(files)} flac files in {audio_dir}")

    if len(files) == 0:
        raise FileNotFoundError(f"No flac files found in {AUDIO_DIR}")

    # Analyse all files
    rows = []
    for f in tqdm(files, desc='Analysing'):
        try:
            stats       = analyse_file(f)
            stats['name']  = f.stem
            stats['path']  = str(f)
            rows.append(stats)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")
            rows.append({'name': f.stem, 'path': str(f), 'error': str(e)})

    df = pd.DataFrame(rows)

    # Overall flag column
    flag_cols = ['flag_all_silent', 'flag_long_internal',
                 'flag_high_fraction', 'flag_short_speech',
                 'flag_many_segments']
    df['flagged'] = df[flag_cols].any(axis=1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Total files analysed : {len(df)}")
    print(f"Flagged              : {df['flagged'].sum()} "
          f"({df['flagged'].mean()*100:.1f}%)")
    print(f"\nFlag breakdown:")
    for col in flag_cols:
        if col in df.columns:
            n = df[col].sum() if col in df else 0
            print(f"  {col:30s}: {n:5d} ({n/len(df)*100:.1f}%)")

    print(f"\nTop flagged by max internal silence:")
    top = df[df['flag_long_internal']].nlargest(20, 'max_internal_sil_s')
    print(top[['name', 'duration_s', 'max_internal_sil_s',
               'silence_fraction', 'n_internal_sil_segs']].to_string())

    # -----------------------------------------------------------------------
    # Save flagged cases
    # -----------------------------------------------------------------------
    flagged_path = os.path.join(OUT_DIR, 'flagged_silences.csv')
    df[df['flagged']].sort_values(
        'max_internal_sil_s', ascending=False
    ).to_csv(flagged_path, index=False)
    print(f"\nFlagged cases saved to {flagged_path}")

    # Full stats for all files
    all_path = os.path.join(OUT_DIR, 'all_silence_stats.csv')
    df.to_csv(all_path, index=False)
    print(f"Full stats saved to {all_path}")

    # -----------------------------------------------------------------------
    # Diagnostic plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    def plot_hist(ax, col, label, threshold=None, log=False):
        data = df[col].dropna()
        ax.hist(data, bins=60, color='steelblue', alpha=0.7, density=True)
        if threshold is not None:
            ax.axvline(threshold, color='red', linestyle='--', lw=2,
                       label=f'threshold ({threshold})')
            ax.legend(fontsize=8)
        if log:
            ax.set_yscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(label)

    plot_hist(axes[0], 'duration_s',
              'Utterance duration (s)')

    plot_hist(axes[1], 'silence_fraction',
              'Silence fraction',
              threshold=MAX_SILENCE_FRACTION)

    plot_hist(axes[2], 'speech_duration_s',
              'Speech duration (s)',
              threshold=MIN_SPEECH_DURATION_S)

    plot_hist(axes[3], 'max_internal_sil_s',
              'Max internal silence (s)',
              threshold=MAX_INTERNAL_SILENCE_S)

    plot_hist(axes[4], 'n_internal_sil_segs',
              'Number of internal silence segments',
              threshold=MAX_SILENCE_SEGMENTS)

    # Flag summary bar chart
    flag_counts = {c.replace('flag_', ''): df[c].sum()
                   for c in flag_cols if c in df.columns}
    axes[5].barh(list(flag_counts.keys()),
                 list(flag_counts.values()),
                 color='tomato', alpha=0.75)
    axes[5].set_xlabel('Number of files flagged')
    axes[5].set_title('Flags summary')
    for i, (k, v) in enumerate(flag_counts.items()):
        axes[5].text(v + 0.5, i, str(v), va='center', fontsize=9)

    fig.suptitle(f'Silence Diagnostic Report — {len(df)} files',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    report_path = os.path.join(OUT_DIR, 'silence_report.png')
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    print(f"Report saved to {report_path}")

    # -----------------------------------------------------------------------
    # Quick listen list — sorted by worst offenders
    # -----------------------------------------------------------------------
    listen_path = os.path.join(OUT_DIR, 'manual_check_list.txt')
    with open(listen_path, 'w') as f:
        f.write("# Files to check manually — sorted by max internal silence\n")
        f.write("# Columns: name | duration_s | max_internal_sil_s | "
                "silence_fraction | flags\n\n")
        for _, row in df[df['flagged']].sort_values(
                'max_internal_sil_s', ascending=False).iterrows():
            flags = [c.replace('flag_', '') for c in flag_cols
                     if c in row and row[c]]
            f.write(f"{row['name']:<30s}  "
                    f"dur={row.get('duration_s', '?'):>6.2f}s  "
                    f"max_sil={row.get('max_internal_sil_s', '?'):>5.2f}s  "
                    f"sil_frac={row.get('silence_fraction', '?'):>4.1%}  "
                    f"flags=[{', '.join(flags)}]\n")
    print(f"Manual check list saved to {listen_path}")
