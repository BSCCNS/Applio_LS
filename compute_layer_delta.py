"""
Compute ContentVec layer12 - layer8 delta features
-----------------------------------------------------
Reads two parquet files (same utterances, different layers),
computes the per-frame, per-dimension difference, and saves
a new parquet with the delta embeddings.

Both input parquets are expected to have columns:
    ['name', EMBEDDING_COLS, 'speaker_id', 'system_id', 'key']

where EMBEDDING_COLS = ['0', '1', ..., '767'] (768D ContentVec/HuBERT).

Output parquet has the same structure, with embedding columns replaced by:
    delta_i = layer12_i - layer8_i   for i in 0..767

Assumptions:
  - Both parquets have the same 'name' + row ordering per utterance
    (i.e. frame i in layer8 corresponds to frame i in layer12)
  - Metadata columns (speaker_id, system_id, key) are identical
    across both files — taken from the layer8 file

Usage:
    Edit LAYER8_PATH, LAYER12_PATH, OUT_PATH below, then run.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Config — update paths
# -----------------------------------------------------------------------

ROOT = Path("/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019/data_prep")

LAYER8_PATH  = ROOT / "feat_768d_train_layer_8_tag.parquet"
LAYER12_PATH = ROOT / "feat_768d_train_layer_12_tag.parquet"
OUT_PATH     = ROOT / "feat_768d_train_layer_delta_tag.parquet"

EMBEDDING_COLS = [str(i) for i in range(768)]
META_COLS      = ['speaker_id', 'system_id', 'key']


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def compute_delta(layer8_path, layer12_path, out_path,
                  embedding_cols=EMBEDDING_COLS, meta_cols=META_COLS):

    print(f"Loading layer 8 : {layer8_path}")
    df8 = pd.read_parquet(layer8_path)

    print(f"Loading layer 12: {layer12_path}")
    df12 = pd.read_parquet(layer12_path)

    print(f"  Layer 8  shape: {df8.shape}")
    print(f"  Layer 12 shape: {df12.shape}")

    # --- Sanity checks ---
    missing8  = [c for c in embedding_cols + meta_cols + ['name']
                  if c not in df8.columns]
    missing12 = [c for c in embedding_cols + ['name']
                  if c not in df12.columns]
    if missing8:
        raise ValueError(f"Layer 8 parquet missing columns: {missing8}")
    if missing12:
        raise ValueError(f"Layer 12 parquet missing columns: {missing12}")

    if len(df8) != len(df12):
        raise ValueError(
            f"Frame count mismatch: layer8={len(df8)} vs layer12={len(df12)}. "
            f"Both parquets must have the same utterances with the same "
            f"number of frames in the same order."
        )

    # Check utterance sets match
    names8  = set(df8['name'].unique())
    names12 = set(df12['name'].unique())
    if names8 != names12:
        only8  = names8 - names12
        only12 = names12 - names8
        raise ValueError(
            f"Utterance mismatch between files. "
            f"Only in layer8: {len(only8)}, only in layer12: {len(only12)}. "
            f"First few: {list(only8)[:3]} / {list(only12)[:3]}"
        )

    # --- Verify row-level alignment ---
    # Sort both by name (stable) to ensure consistent ordering before
    # checking — groupby+concat below assumes per-utterance frame order
    # is already consistent within each file.
    if not (df8['name'].values == df12['name'].values).all():
        print("  Row order differs between files — reordering layer12 "
              "to match layer8 per-utterance frame order...")
        # Reindex df12 frame-by-frame to match df8's (name) sequence.
        # This assumes within each utterance, frames are in the same
        # temporal order in both files (just possibly different
        # utterance grouping order).
        df12 = (
            df12.set_index('name', drop=False)
                .groupby('name', sort=False, group_keys=False)
                .apply(lambda g: g.reset_index(drop=True))
        )
        df8_order = df8[['name']].reset_index(drop=True)
        df12 = (
            df12.groupby('name', sort=False)
                .apply(lambda g: g.reset_index(drop=True))
                .reset_index(drop=True)
        )
        # Final check
        if not (df8['name'].values == df12['name'].values).all():
            raise ValueError(
                "Could not align row order between layer8 and layer12 "
                "parquets — please ensure both files have identical "
                "per-utterance frame ordering."
            )

    print("  Row alignment verified ✓")

    # --- Compute delta ---
    print("Computing delta = layer12 - layer8 ...")
    delta = (
        df12[embedding_cols].values.astype(np.float32)
        - df8[embedding_cols].values.astype(np.float32)
    )

    # --- Plot distribution ---
    delta_flat = delta.flatten()
    plt.hist(delta_flat, bins=100)
    plt.title('Layer12 - Layer8 delta distribution')
    plt.savefig('delta_hist.png')

    # --- Build output dataframe ---
    out = pd.DataFrame(delta, columns=embedding_cols)
    out.insert(0, 'name', df8['name'].values)
    for col in meta_cols:
        out[col] = df8[col].values

    # Reorder columns: name, embeddings, metadata
    out = out[['name'] + embedding_cols + meta_cols]

    print(f"  Output shape: {out.shape}")
    print(f"  Delta stats — mean: {delta.mean():.4f}, "
          f"std: {delta.std():.4f}, "
          f"abs max: {np.abs(delta).max():.4f}")

    print(f"Saving to {out_path}")
    out.to_parquet(out_path, index=False)
    print("Done.")

    return out


if __name__ == '__main__':
    compute_delta(LAYER8_PATH, LAYER12_PATH, OUT_PATH)

    
    
