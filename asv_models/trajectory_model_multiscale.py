"""
Trajectory Anomaly Detection — Multi-Scale LSTM Predictor (768D)
-----------------------------------------------------------------
Extension of the clean single-scale predictor with multi-horizon
next-frame prediction.

Key hypothesis:
  Bonafide speech has predictable dynamics at ALL timescales:
    - Short horizon (1-2 frames, 20-40ms):  local coarticulation
    - Mid   horizon (5-10 frames, 100-200ms): phoneme transitions
    - Long  horizon (25-50 frames, 500ms-1s): respiratory rhythm / breathing

  Synthetic speech may fool the 1-frame predictor but should fail
  at longer horizons where breathing structure is encoded.
  The multi-scale loss forces the LSTM to learn the bonafide manifold
  at all these timescales simultaneously.

Architecture:
  Shared LSTM encoder (768D → hidden_dim)
  Separate linear projection head per horizon
  Loss = weighted sum of MSE across all horizons
         (longer horizons weighted more — breathing hypothesis)

Outputs saved to: experiments/multiscale/

Usage:
  Set USE_OFFICIAL_SPLIT = True and point to your parquet files.
  Compare results/trajectory_report_multiscale_*.png against
  experiments/single_scale/ baseline.
"""

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------
# Input 
# -----------------------------------------------------------------------

ROOT = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019'
train_file = f'{ROOT}/ASVspoof2019_LA_train_preproc/data_prep/feat_768d_layer_8_tag.parquet'
dev_file = f'{ROOT}/ASVspoof2019_LA_dev_preproc/data_prep_dev/feat_768d_dev_layer_8_tag.parquet'

# -----------------------------------------------------------------------
# Output folder
# -----------------------------------------------------------------------

EXPERIMENT   = 'multiscale'
OUT_DIR      = f'experiments/{EXPERIMENT}'
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# 0. Config
# -----------------------------------------------------------------------

HUBERT_COLS = [str(i) for i in range(768)]

# Prediction horizons in frames (1 frame = 20ms)
# 1  frame  =  20ms  local coarticulation
# 5  frames = 100ms  phoneme transition
# 10 frames = 200ms  syllable boundary
# 25 frames = 500ms  breath group onset
# 50 frames =   1s   respiratory cycle
HORIZONS = [1, 5, 10, 25, 50]

# Horizon weights — longer horizons weighted more to emphasise
# respiratory / breathing dynamics over local coarticulation
HORIZON_WEIGHTS = {
    1:  1.0,
    5:  1.5,
    10: 2.0,
    25: 3.0,
    50: 3.0,
}

CFG = dict(
    features          = HUBERT_COLS,   # 768D pure HuBERT
    hidden_dim        = 512,
    num_layers        = 2,
    dropout           = 0.1,
    max_len           = 500,
    batch_size        = 256,
    epochs            = 350,
    lr                = 1e-3,
    clip_grad         = 1.0,
    device            = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_bf16          = True,
    num_workers       = 8,
    eval_every        = 10,
    seed              = 42,
    warmrestart_T0    = 50,
    warmrestart_Tmult = 2,
    horizons          = HORIZONS,
    horizon_weights   = HORIZON_WEIGHTS,
)


# -----------------------------------------------------------------------
# 1. Datasets  (identical to clean single-scale version)
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Eager dataset for train set."""

    def __init__(self, df, scaler=None, fit_scaler=False, max_len=500):
        self.max_len  = max_len
        feat_cols     = CFG['features']

        if fit_scaler:
            self.scaler = StandardScaler()
            df = df.copy()
            df[feat_cols] = self.scaler.fit_transform(df[feat_cols])
        elif scaler is not None:
            self.scaler = scaler
            df = df.copy()
            df[feat_cols] = scaler.transform(df[feat_cols])
        else:
            self.scaler = None

        print("  Building train utterance index (eager)...")
        self.names     = []
        self.sequences = []
        self.lengths   = []

        for name, grp in df.groupby('name', sort=False):
            feats = grp[feat_cols].values.astype(np.float32)
            T     = min(len(feats), max_len)
            seq     = np.zeros((max_len, len(feat_cols)), dtype=np.float32)
            seq[:T] = feats[:T]
            self.names.append(name)
            self.sequences.append(seq)
            self.lengths.append(T)

        self.sequences = np.stack(self.sequences)
        self.lengths   = np.array(self.lengths)
        print(f"  Train dataset ready: {len(self.names)} utterances  "
              f"({self.sequences.nbytes / 1e9:.1f} GB)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),
            int(self.lengths[idx]),
            self.names[idx],
        )


class LazyTrajectoryDataset(Dataset):
    """Lazy dataset for eval set — no large tensor pre-allocation."""

    def __init__(self, df, scaler, max_len=500):
        self.max_len   = max_len
        self.feat_cols = CFG['features']

        print("  Scaling eval features...")
        df = df.copy()
        df[self.feat_cols] = scaler.transform(
            df[self.feat_cols].values.astype(np.float32)
        )

        print("  Building eval utterance index (lazy)...")
        self.names  = []
        self.groups = {}
        for name, grp in df.groupby('name', sort=False):
            self.names.append(name)
            self.groups[name] = grp[self.feat_cols].values.astype(np.float32)

        print(f"  Eval dataset ready: {len(self.names)} utterances "
              f"(lazy — no pre-allocation)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name  = self.names[idx]
        feats = self.groups[name]
        T     = min(len(feats), self.max_len)
        seq     = np.zeros((self.max_len, len(self.feat_cols)), dtype=np.float32)
        seq[:T] = feats[:T]
        return (torch.from_numpy(seq), T, name)


def collate_fn(batch):
    seqs, lengths, names = zip(*batch)
    return (torch.stack(seqs), torch.tensor(lengths), list(names))


# -----------------------------------------------------------------------
# 2. Model — Multi-Scale LSTM Predictor
# -----------------------------------------------------------------------

class MultiScaleLSTMPredictor(nn.Module):
    """
    Multi-horizon next-frame predictor.

    Shared LSTM encoder processes the full sequence.
    Separate linear head per horizon predicts the frame H steps ahead.

    For horizon H, given frame t, predicts frame t+H.
    Input sequence: frames 0 .. T-H-1
    Target:         frames H .. T-1

    The shared encoder must learn a representation that supports
    prediction at all timescales simultaneously — forcing it to
    capture both local coarticulation dynamics and slow respiratory
    rhythm in the same hidden state.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, horizons):
        super().__init__()
        self.horizons   = horizons
        self.max_horizon = max(horizons)

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # One projection head per horizon
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, input_dim)
            for h in horizons
        })

    def forward(self, x, lengths):
        """
        x:       (B, T, 768)
        lengths: (B,) actual sequence lengths

        Returns dict: {horizon: predictions (B, T-max_horizon, 768)}
        All predictions aligned to the same input frames 0..T-max_horizon-1
        so losses can be summed directly.
        """
        H    = self.max_horizon
        T    = x.shape[1]

        # Input: frames 0..T-H-1 (same for all horizons, aligned to max)
        x_in       = x[:, :T - H, :]                    # (B, T-H, 768)
        lengths_in = (lengths - H).clamp(min=1)

        packed = pack_padded_sequence(
            x_in, lengths_in.cpu(),
            batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(
            out, batch_first=True, total_length=T - H
        )
        # out: (B, T-H, hidden_dim)

        # Each head predicts the frame h steps ahead
        predictions = {}
        for h in self.horizons:
            predictions[h] = self.heads[str(h)](out)    # (B, T-H, 768)

        return predictions


# -----------------------------------------------------------------------
# 3. Multi-scale loss
# -----------------------------------------------------------------------

def multiscale_prediction_mse(predictions, x, lengths, horizons,
                               horizon_weights):
    """
    Weighted sum of masked MSE across all prediction horizons.

    For horizon h:
      input  frames: 0 .. T-H-1   (where H = max_horizon)
      target frames: h .. T-H+h-1
      mask: valid frames only (no padding)

    predictions: dict {h: (B, T-H, 768)}
    x:           (B, T, 768) original sequence
    lengths:     (B,) on device
    """
    H      = max(horizons)
    T      = x.shape[1]
    device = x.device

    # Valid frame mask: input positions 0..T-H-1 that have a valid target
    # Position t is valid if t + h < length (i.e. target frame exists)
    total_loss   = 0.0
    total_weight = 0.0

    for h in horizons:
        pred   = predictions[h]                          # (B, T-H, 768)
        target = x[:, h:T - H + h, :]                   # (B, T-H, 768)

        # Mask: position t valid if t + h < lengths  →  t < lengths - h
        valid_len = (lengths - h).clamp(min=0)           # (B,)
        mask      = torch.arange(T - H, device=device)[None, :] \
                    < valid_len[:, None]                  # (B, T-H)

        err    = ((pred - target) ** 2).mean(dim=-1)     # (B, T-H)
        denom  = mask.sum(dim=-1).float().clamp(min=1.0)
        loss_h = ((err * mask).sum(dim=-1) / denom).mean()

        w = horizon_weights[h]
        total_loss   += w * loss_h
        total_weight += w

    return total_loss / total_weight


# -----------------------------------------------------------------------
# 4. Per-horizon anomaly score (for analysis)
# -----------------------------------------------------------------------

@torch.no_grad()
def score_per_horizon(predictions, x, lengths, horizons):
    """
    Returns per-utterance anomaly score broken down by horizon.
    Useful for understanding which timescale drives discrimination.
    """
    H      = max(horizons)
    T      = x.shape[1]
    device = x.device
    scores = {}

    for h in horizons:
        pred      = predictions[h].float()
        target    = x[:, h:T - H + h, :].float()
        valid_len = (lengths - h).clamp(min=0)
        mask      = torch.arange(T - H, device=device)[None, :] \
                    < valid_len[:, None]
        err       = ((pred - target) ** 2).mean(dim=-1)
        denom     = mask.sum(dim=-1).float().clamp(min=1.0)
        scores[h] = ((err * mask).sum(dim=-1) / denom).cpu().numpy()

    return scores


# -----------------------------------------------------------------------
# 5. Training
# -----------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, use_bf16):
    model.train()
    total_loss  = 0.0
    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.autocast(device_type='cuda', dtype=autocast_dt)
           if device == 'cuda'
           else torch.autocast(device_type='cpu', enabled=False))

    for seqs, lengths, _ in loader:
        seqs    = seqs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        optimizer.zero_grad()
        with ctx:
            predictions = model(seqs, lengths)
            loss        = multiscale_prediction_mse(
                predictions, seqs, lengths,
                CFG['horizons'], CFG['horizon_weights']
            )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------------------------------------------------
# 6. Scoring
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_scores(model, loader, device, use_bf16):
    """
    Per-utterance anomaly scores:
      - combined: weighted average across all horizons (primary score)
      - per_horizon: individual score per horizon (for analysis)
    """
    model.eval()
    all_names          = []
    all_combined       = []
    all_per_horizon    = {h: [] for h in CFG['horizons']}

    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.autocast(device_type='cuda', dtype=autocast_dt)
           if device == 'cuda'
           else torch.autocast(device_type='cpu', enabled=False))

    for seqs, lengths, names in loader:
        seqs    = seqs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with ctx:
            predictions = model(seqs, lengths)

        # Cast to float32 for scoring
        predictions_f = {h: predictions[h].float() for h in CFG['horizons']}
        seqs_f        = seqs.float()

        # Combined score (same weighting as training loss)
        H      = max(CFG['horizons'])
        T      = seqs_f.shape[1]
        device_ = seqs_f.device
        combined = torch.zeros(seqs_f.shape[0], device=device_)
        total_w  = 0.0

        for h in CFG['horizons']:
            pred      = predictions_f[h]
            target    = seqs_f[:, h:T - H + h, :]
            valid_len = (lengths - h).clamp(min=0)
            mask      = torch.arange(T - H, device=device_)[None, :] \
                        < valid_len[:, None]
            err       = ((pred - target) ** 2).mean(dim=-1)
            denom     = mask.sum(dim=-1).float().clamp(min=1.0)
            w         = CFG['horizon_weights'][h]
            combined += w * ((err * mask).sum(dim=-1) / denom)
            total_w  += w

            all_per_horizon[h].extend(
                ((err * mask).sum(dim=-1) / denom).cpu().numpy().tolist()
            )

        combined = (combined / total_w).cpu().numpy()
        all_names.extend(names)
        all_combined.extend(combined.tolist())

    scores_df = pd.DataFrame({'name': all_names, 'score': all_combined})
    for h in CFG['horizons']:
        scores_df[f'score_h{h}'] = all_per_horizon[h]

    return scores_df


# -----------------------------------------------------------------------
# 7. Evaluation
# -----------------------------------------------------------------------

def evaluate(scores_df, score_col='score', verbose=True):
    y_true      = (scores_df['key'] == 'spoof').astype(int)
    y_score     = scores_df[score_col]
    auc         = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr         = 1 - tpr
    eer_idx     = np.argmin(np.abs(fpr - fnr))
    eer         = (fpr[eer_idx] + fnr[eer_idx]) / 2
    if verbose:
        print(f"  AUC-ROC : {auc:.4f}   EER: {eer*100:.2f}%")
    return auc, eer, fpr, tpr


# -----------------------------------------------------------------------
# 8. Diagnostic report
# -----------------------------------------------------------------------

def full_diagnostic_report(scores_df, fpr, tpr, auc, train_losses,
                            eval_log=None, save_path=None):
    bonafide_mean = scores_df[scores_df['key'] == 'bonafide']['score'].mean()
    systems       = sorted([s for s in scores_df['system_id'].unique()
                            if s is not None and s != '-'])

    n_rows = 5 if eval_log else 4
    fig    = plt.figure(figsize=(16, n_rows * 5))
    gs     = gridspec.GridSpec(n_rows, 2, figure=fig,
                               hspace=0.5, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    ax6 = fig.add_subplot(gs[3, :])

    # Training loss
    ax1.plot(train_losses, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss — Multi-Scale Prediction (bonafide only)')

    # ROC
    ax2.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curve (combined score)')
    ax2.legend()

    # Overall score distributions
    for label, color in [('bonafide', 'steelblue'), ('spoof', 'tomato')]:
        subset = scores_df[scores_df['key'] == label]['score']
        ax3.hist(subset, bins=60, alpha=0.55, color=color,
                 label=f'{label} (n={len(subset)})', density=True)
    ax3.set_xlabel('Anomaly score (combined)')
    ax3.set_ylabel('Density')
    ax3.set_title('Bonafide vs Spoof — Combined Score')
    ax3.legend()

    # Per-system box plots
    box_data   = [scores_df[scores_df['system_id'] == s]['score'].values
                  for s in ['-'] + systems]
    box_labels = ['bonafide'] + systems
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['steelblue'] + ['tomato'] * len(systems)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.axhline(bonafide_mean, color='black', linestyle='--', lw=1)
    ax4.set_ylabel('Anomaly score')
    ax4.set_title('Per-System Score Distribution')
    ax4.tick_params(axis='x', rotation=45)

    # Per-system histograms
    colors_sys = plt.cm.tab10(np.linspace(0, 1, len(systems)))
    for sys_id, color in zip(systems, colors_sys):
        subset = scores_df[scores_df['system_id'] == sys_id]['score']
        ax5.hist(subset, bins=40, alpha=0.4, color=color,
                 label=f'{sys_id} (n={len(subset)})', density=True)
    ax5.axvline(bonafide_mean, color='black', linestyle='--', lw=2,
                label=f'bonafide mean ({bonafide_mean:.2f})')
    ax5.set_xlabel('Anomaly score')
    ax5.set_ylabel('Density')
    ax5.set_title('Per-System Score Distributions')
    ax5.legend(fontsize=8, ncol=3)

    # Per-horizon AUC breakdown — the key diagnostic for multi-scale
    horizon_aucs = []
    for h in CFG['horizons']:
        col = f'score_h{h}'
        if col in scores_df.columns:
            auc_h, _, _, _ = evaluate(scores_df, score_col=col, verbose=False)
            horizon_aucs.append((h, auc_h))

    if horizon_aucs:
        hs, aucs = zip(*horizon_aucs)
        colors_h = ['steelblue' if a < auc else 'tomato' for a in aucs]
        bars = ax6.bar([f'{h}f\n({h*20}ms)' for h in hs], aucs,
                       color=colors_h, alpha=0.75, edgecolor='black', lw=0.5)
        ax6.axhline(auc, color='black', linestyle='--', lw=1.5,
                    label=f'combined AUC ({auc:.3f})')
        ax6.axhline(0.671, color='gray', linestyle=':', lw=1.5,
                    label='single-scale baseline (0.671)')
        ax6.set_ylim(0.5, 1.0)
        ax6.set_xlabel('Prediction horizon (frames / ms)')
        ax6.set_ylabel('AUC-ROC')
        ax6.set_title('AUC per Prediction Horizon\n'
                      '(red = above combined, blue = below combined)')
        ax6.legend()
        # Label bars
        for bar, (h, a) in zip(bars, horizon_aucs):
            ax6.text(bar.get_x() + bar.get_width()/2, a + 0.005,
                     f'{a:.3f}', ha='center', va='bottom', fontsize=9)

    # Score separation during training
    if eval_log:
        ax7 = fig.add_subplot(gs[4, :])
        epochs_logged = [e for e, *_ in eval_log]
        bf_means      = [bf for _, bf, _,  _  in eval_log]
        bf_stds       = [s  for _, _,  s,  _  in eval_log]
        sp_means      = [sp for _, _,  _,  sp in eval_log]
        ax7.plot(epochs_logged, bf_means, color='steelblue',
                 label='bonafide mean')
        ax7.fill_between(
            epochs_logged,
            [m - s for m, s in zip(bf_means, bf_stds)],
            [m + s for m, s in zip(bf_means, bf_stds)],
            alpha=0.2, color='steelblue', label='bonafide ±1σ'
        )
        ax7.plot(epochs_logged, sp_means, color='tomato', label='spoof mean')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Anomaly score')
        ax7.set_title('Score Separation During Training')
        ax7.legend()

    fig.suptitle('Multi-Scale LSTM Predictor — Full Diagnostic Report',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------
# 9. Main
# -----------------------------------------------------------------------

if __name__ == '__main__':

    torch.manual_seed(CFG['seed'])
    device   = CFG['device']
    use_bf16 = CFG['use_bf16'] and device == 'cuda'
    print(f"Experiment : {EXPERIMENT}")
    print(f"Output dir : {OUT_DIR}")
    print(f"Device     : {device}")
    print(f"BF16       : {use_bf16}")
    print(f"Horizons   : {CFG['horizons']} frames "
          f"({[h*20 for h in CFG['horizons']]} ms)")
    print(f"Weights    : {CFG['horizon_weights']}")

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    USE_OFFICIAL_SPLIT = True   # ← False for random 80/20 split

    USE_OFFICIAL_SPLIT = True   # ← set True when dev features are ready

    if USE_OFFICIAL_SPLIT:
        print("Loading official train/dev split...")
        df_train_raw = pd.read_parquet(train_file)
        df_eval_raw  = pd.read_parquet(dev_file)
    else:
        print("Loading data (random 80/20 bonafide split)...")
        df_train_raw = pd.read_parquet(train_file)
        df_eval_raw  = None   # built from split below

    # Verify HuBERT columns
    missing = [c for c in HUBERT_COLS if c not in df_train_raw.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} HuBERT columns — "
                         f"first missing: {missing[0]}.")

    n_utt  = df_train_raw['name'].nunique()
    n_feat = len(CFG['features'])
    mem_gb = (n_utt * CFG['max_len'] * n_feat * 4) / 1e9
    print(f"Utterances : {n_utt}")
    print(f"Frames     : {len(df_train_raw)}")
    print(f"Tensor mem : ~{mem_gb:.1f} GB")
    print(f"Key counts : "
          f"{df_train_raw.groupby('name')['key'].first().value_counts().to_dict()}")

    # -------------------------------------------------------------------
    # Train / eval split
    # -------------------------------------------------------------------
    if USE_OFFICIAL_SPLIT:
        df_train = df_train_raw[df_train_raw['key'] == 'bonafide']
        df_eval  = df_eval_raw
        print(f"\nOfficial split:")
        print(f"  Train bonafide : {df_train['name'].nunique()}")
        print(f"  Eval  bonafide : {df_eval[df_eval['key']=='bonafide']['name'].nunique()}")
        print(f"  Eval  spoof    : {df_eval[df_eval['key']=='spoof']['name'].nunique()}")
    else:
        rng            = np.random.default_rng(CFG['seed'])
        bonafide_names = df_train_raw[df_train_raw['key'] == 'bonafide']['name'].unique()
        rng.shuffle(bonafide_names)
        n_train              = int(len(bonafide_names) * 0.8)
        train_bonafide_names = bonafide_names[:n_train]
        eval_bonafide_names  = bonafide_names[n_train:]
        spoof_names          = df_train_raw[df_train_raw['key'] == 'spoof']['name'].unique()
        eval_names           = np.concatenate([eval_bonafide_names, spoof_names])
        df_train = df_train_raw[df_train_raw['name'].isin(train_bonafide_names)]
        df_eval  = df_train_raw[df_train_raw['name'].isin(eval_names)]
        print(f"\nRandom 80/20 split:")
        print(f"  Train bonafide : {len(train_bonafide_names)}")
        print(f"  Eval  bonafide : {len(eval_bonafide_names)}")
        print(f"  Eval  spoof    : {len(spoof_names)}")

    # Sanity checks
    assert df_train['key'].nunique() == 1
    assert df_train['key'].unique()[0] == 'bonafide'
    print("Sanity checks passed ✓")

    # -------------------------------------------------------------------
    # Datasets & loaders
    # -------------------------------------------------------------------
    print("\nBuilding train dataset (eager)...")
    train_set = TrajectoryDataset(df_train, fit_scaler=True,
                                  max_len=CFG['max_len'])
    print("Building eval dataset (lazy)...")
    eval_set  = LazyTrajectoryDataset(df_eval, scaler=train_set.scaler,
                                      max_len=CFG['max_len'])

    loader_kwargs = dict(
        collate_fn         = collate_fn,
        num_workers        = CFG['num_workers'],
        pin_memory         = device == 'cuda',
        persistent_workers = CFG['num_workers'] > 0,
    )
    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'],
                              shuffle=True,  **loader_kwargs)
    eval_loader  = DataLoader(eval_set,  batch_size=CFG['batch_size'],
                              shuffle=False, **loader_kwargs)

    # -------------------------------------------------------------------
    # Model, optimiser, scheduler
    # -------------------------------------------------------------------
    model = MultiScaleLSTMPredictor(
        n_feat, CFG['hidden_dim'], CFG['num_layers'],
        CFG['dropout'], CFG['horizons']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0    = CFG['warmrestart_T0'],
        T_mult = CFG['warmrestart_Tmult'],
        eta_min= 1e-6,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Heads: {len(CFG['horizons'])} × Linear({CFG['hidden_dim']}, 768)")

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    if USE_OFFICIAL_SPLIT:
        meta = df_eval.groupby('name')[['key', 'system_id']].first().reset_index()
    else:
        meta = df_train_raw.groupby('name')[['key', 'system_id']].first().reset_index()

    train_losses = []
    eval_log     = []

    for epoch in range(CFG['epochs']):
        loss = train_epoch(model, train_loader, optimizer, device, use_bf16)
        train_losses.append(loss)
        scheduler.step(epoch)

        if (epoch + 1) % CFG['eval_every'] == 0:
            scores_tmp = compute_scores(model, eval_loader, device, use_bf16)
            scores_tmp = scores_tmp.merge(meta, on='name')
            scores_tmp['system_id'] = scores_tmp['system_id'].fillna('-')
            bf     = scores_tmp[scores_tmp['key'] == 'bonafide']['score']
            sp     = scores_tmp[scores_tmp['key'] == 'spoof']['score']
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}/{CFG['epochs']}  "
                  f"loss {loss:.4f}  "
                  f"bonafide {bf.mean():.3f}±{bf.std():.3f}  "
                  f"spoof {sp.mean():.3f}  "
                  f"lr {lr_now:.2e}")
            eval_log.append((epoch + 1, bf.mean(), bf.std(), sp.mean()))
        elif (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}/{CFG['epochs']}  "
                  f"loss {loss:.4f}  "
                  f"lr {lr_now:.2e}")

    # -------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------
    print("\nFinal evaluation...")
    scores_df = compute_scores(model, eval_loader, device, use_bf16)
    scores_df = scores_df.merge(meta, on='name')
    scores_df['system_id'] = scores_df['system_id'].fillna('-')

    print(scores_df.groupby('system_id')['score']
          .agg(['mean', 'std', 'count'])
          .sort_values('mean', ascending=False))

    print("\nCombined score:")
    auc, eer, fpr, tpr = evaluate(scores_df)

    print("\nPer-horizon AUC:")
    for h in CFG['horizons']:
        col = f'score_h{h}'
        auc_h, eer_h, _, _ = evaluate(scores_df, score_col=col, verbose=False)
        print(f"  h={h:2d} ({h*20:4d}ms)  AUC {auc_h:.4f}  EER {eer_h*100:.2f}%")

    # -------------------------------------------------------------------
    # Report & model
    # -------------------------------------------------------------------
    report_path = os.path.join(OUT_DIR, 'trajectory_report_multiscale.png')
    full_diagnostic_report(
        scores_df, fpr, tpr, auc, train_losses,
        eval_log  = eval_log,
        save_path = report_path,
    )

    model_path = os.path.join(OUT_DIR, 'trajectory_model_multiscale.pt')
    torch.save({
        'model_state': model.state_dict(),
        'scaler_mean': train_set.scaler.mean_,
        'scaler_std':  train_set.scaler.scale_,
        'cfg':         CFG,
        'horizons':    CFG['horizons'],
    }, model_path)
    print(f"Model saved to {model_path}")

    # Save scores CSV for further analysis
    scores_path = os.path.join(OUT_DIR, 'scores_multiscale.csv')
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores saved to {scores_path}")
