"""
Trajectory Anomaly Detection — LSTM Next-Frame Predictor (768D, clean)
-----------------------------------------------------------------------
Trains on bonafide utterance trajectories only.
Anomaly score = mean next-frame prediction error (length-masked MSE).
Higher score = more anomalous = more likely spoof.

Key findings from ablation experiments:
  - Phone weights make no difference — dropped
  - nn_distance makes no difference — dropped
  - Input: pure 768D HuBERT embeddings only
  - The physiological signal lives entirely in the trajectory
    geometry and dynamics of the HuBERT embedding space

This simplification removes the LibriSpeech phone tagging dependency
entirely, significantly reducing the data preparation pipeline.

Input features: HuBERT layer 8 embeddings, dims 0..767  [768-dim]

Usage:
  - Prepare a DataFrame with columns:
      name       : utterance file ID
      0..767     : HuBERT frame embeddings (float32)
      key        : 'bonafide' or 'spoof'
      system_id  : 'A01'..'A19' or '-' for bonafide
  - Rows in temporal order within each utterance
  - One row per 20ms HuBERT frame

Optimised for Hopper (H100) GPUs:
  - BF16 mixed precision
  - Batch size 256
  - num_workers=8 with persistent_workers
  - Eager train dataset / lazy eval dataset
"""

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

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

ROOT = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019'
train_file = f'{ROOT}/ASVspoof2019_LA_train_preproc/data_prep/feat_768d_layer_8_tag.parquet'
dev_file = f'{ROOT}/ASVspoof2019_LA_dev_preproc/data_prep_dev/feat_768d_dev_layer_8_tag.parquet'

# -----------------------------------------------------------------------
# 0. Config
# -----------------------------------------------------------------------

HUBERT_COLS = [str(i) for i in range(768)]  # '0', '1', ... '767'

CFG = dict(
    features          = HUBERT_COLS,   # 768 input channels — pure HuBERT
    hidden_dim        = 512,
    num_layers        = 2,
    dropout           = 0.1,
    max_len           = 500,
    batch_size        = 256,
    epochs            = 500,
    lr                = 1e-3,
    clip_grad         = 1.0,
    device            = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_bf16          = True,
    num_workers       = 8,
    eval_every        = 10,
    seed              = 42,
    warmrestart_T0    = 50,
    warmrestart_Tmult = 2,
)


# -----------------------------------------------------------------------
# 1. Datasets
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Eager dataset — pre-builds all padded sequences in memory.
    Use for TRAIN set (small, accessed every epoch).
    Stores only HuBERT embeddings — no phone labels needed.
    """

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

        self.sequences = np.stack(self.sequences)   # (N, max_len, 768)
        self.lengths   = np.array(self.lengths)     # (N,)
        print(f"  Train dataset ready: {len(self.names)} utterances  "
              f"({self.sequences.nbytes / 1e9:.1f} GB)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),  # (max_len, 768)
            int(self.lengths[idx]),                  # scalar
            self.names[idx],                         # str
        )


class LazyTrajectoryDataset(Dataset):
    """
    Lazy dataset — builds each sequence on demand in __getitem__.
    No large tensor pre-allocation. Use for EVAL set.
    """

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

        return (
            torch.from_numpy(seq),  # (max_len, 768)
            T,                       # scalar
            name,                    # str
        )


def collate_fn(batch):
    seqs, lengths, names = zip(*batch)
    return (
        torch.stack(seqs),
        torch.tensor(lengths),
        list(names),
    )


# -----------------------------------------------------------------------
# 2. Model — LSTM Next-Frame Predictor
# -----------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """
    Next-frame predictor.

    Given frames 0..T-2, predicts frames 1..T-1.
    Trained on bonafide only — learns the dynamics of real speech.

    The model learns: "given where I am on the bonafide HuBERT manifold,
    where should the trajectory go next?" Spoofed trajectories have higher
    prediction error because their frame-to-frame transitions don't follow
    the learned physiological dynamics of real speech production.

    Architecture:
      LSTM:  768D → hidden_dim (sequence dynamics)
      proj:  hidden_dim → 768D (reconstruct in original space)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        """
        x:       (B, T, 768)
        lengths: (B,) actual sequence lengths

        Returns predicted next frames: (B, T-1, 768)
        Input frames 0..T-2, target frames 1..T-1.
        """
        x_in       = x[:, :-1, :]               # (B, T-1, 768)
        lengths_in = (lengths - 1).clamp(min=1)

        packed = pack_padded_sequence(
            x_in, lengths_in.cpu(),
            batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(
            out, batch_first=True, total_length=x.shape[1] - 1
        )
        return self.proj(out)                    # (B, T-1, 768)


# -----------------------------------------------------------------------
# 3. Loss — length-masked MSE (no phone weighting)
# -----------------------------------------------------------------------

def prediction_mse(pred, target, lengths):
    """
    Plain MSE on next-frame predictions, masked by sequence length.
    No phone weighting — ablation confirmed weights make no difference.

    pred:    (B, T-1, 768)
    target:  (B, T-1, 768)  — x[:, 1:, :]
    lengths: (B,)            — original sequence lengths
    """
    # Mask: True for valid (non-padding) target frames
    T_pred = pred.shape[1]
    mask   = torch.arange(T_pred, device=pred.device)[None, :] \
             < (lengths[:, None] - 1).clamp(min=0)       # (B, T-1)

    err    = ((pred - target) ** 2).mean(dim=-1)          # (B, T-1)
    denom  = mask.sum(dim=-1).float().clamp(min=1.0)
    return ((err * mask).sum(dim=-1) / denom).mean()


# -----------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, use_bf16):
    model.train()
    total_loss  = 0.0
    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.autocast(device_type='cuda', dtype=autocast_dt)
           if device == 'cuda'
           else torch.autocast(device_type='cpu', enabled=False))

    for seqs, lengths, _ in loader:
        seqs = seqs.to(device, non_blocking=True)

        optimizer.zero_grad()
        with ctx:
            pred   = model(seqs, lengths)
            target = seqs[:, 1:, :]
            loss   = prediction_mse(pred, target, lengths)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------------------------------------------------
# 5. Scoring
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_scores(model, loader, device, use_bf16):
    """Per-utterance anomaly score = mean next-frame prediction MSE."""
    model.eval()
    all_names, all_scores = [], []
    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.autocast(device_type='cuda', dtype=autocast_dt)
           if device == 'cuda'
           else torch.autocast(device_type='cpu', enabled=False))

    for seqs, lengths, names in loader:
        seqs = seqs.to(device, non_blocking=True)

        with ctx:
            pred = model(seqs, lengths)

        pred   = pred.float()
        target = seqs[:, 1:, :].float()

        T_pred = pred.shape[1]
        mask   = torch.arange(T_pred, device=pred.device)[None, :] \
                 < (lengths[:, None] - 1).clamp(min=0)
        err    = ((pred - target) ** 2).mean(dim=-1)
        denom  = mask.sum(dim=-1).float().clamp(min=1.0)
        scores = ((err * mask).sum(dim=-1) / denom).cpu().numpy()

        all_names.extend(names)
        all_scores.extend(scores.tolist())

    return pd.DataFrame({'name': all_names, 'score': all_scores})


# -----------------------------------------------------------------------
# 6. Evaluation
# -----------------------------------------------------------------------

def evaluate(scores_df, verbose=True):
    y_true      = (scores_df['key'] == 'spoof').astype(int)
    y_score     = scores_df['score']
    auc         = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr         = 1 - tpr
    eer_idx     = np.argmin(np.abs(fpr - fnr))
    eer         = (fpr[eer_idx] + fnr[eer_idx]) / 2
    if verbose:
        print(f"AUC-ROC : {auc:.4f}")
        print(f"EER     : {eer*100:.2f}%")
    return auc, eer, fpr, tpr


# -----------------------------------------------------------------------
# 7. Diagnostic report
# -----------------------------------------------------------------------

def full_diagnostic_report(scores_df, fpr, tpr, auc, train_losses,
                            eval_log=None, save_path=None):
    bonafide_mean = scores_df[scores_df['key'] == 'bonafide']['score'].mean()
    systems       = sorted([s for s in scores_df['system_id'].unique()
                            if s != '-'])

    n_rows = 4 if eval_log else 3
    fig    = plt.figure(figsize=(16, n_rows * 5))
    gs     = gridspec.GridSpec(n_rows, 2, figure=fig,
                               hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # Training loss
    ax1.plot(train_losses, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss — Next-Frame Prediction (bonafide only)')

    # ROC
    ax2.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curve')
    ax2.legend()

    # Score distributions
    for label, color in [('bonafide', 'steelblue'), ('spoof', 'tomato')]:
        subset = scores_df[scores_df['key'] == label]['score']
        ax3.hist(subset, bins=60, alpha=0.55, color=color,
                 label=f'{label} (n={len(subset)})', density=True)
    ax3.set_xlabel('Anomaly score (prediction error)')
    ax3.set_ylabel('Density')
    ax3.set_title('Bonafide vs Spoof Score Distributions')
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

    # Score separation during training
    if eval_log:
        ax6 = fig.add_subplot(gs[3, :])
        epochs_logged = [e for e, *_ in eval_log]
        bf_means      = [bf for _, bf, _,  _  in eval_log]
        bf_stds       = [s  for _, _,  s,  _  in eval_log]
        sp_means      = [sp for _, _,  _,  sp in eval_log]
        ax6.plot(epochs_logged, bf_means, color='steelblue',
                 label='bonafide mean')
        ax6.fill_between(
            epochs_logged,
            [m - s for m, s in zip(bf_means, bf_stds)],
            [m + s for m, s in zip(bf_means, bf_stds)],
            alpha=0.2, color='steelblue', label='bonafide ±1σ'
        )
        ax6.plot(epochs_logged, sp_means, color='tomato',
                 label='spoof mean')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Anomaly score')
        ax6.set_title('Score Separation During Training')
        ax6.legend()

    fig.suptitle('Trajectory LSTM Predictor — Full Diagnostic Report',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------

if __name__ == '__main__':

    torch.manual_seed(CFG['seed'])
    device   = CFG['device']
    use_bf16 = CFG['use_bf16'] and device == 'cuda'
    print(f"Device  : {device}")
    print(f"BF16    : {use_bf16}")

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    # Two modes:
    #
    # MODE A — random 80/20 bonafide split (current, for feasibility testing)
    #   df = pd.read_parquet('train_features.parquet')
    #   Expected columns: name, 0..767, key, system_id
    #
    # MODE B — official ASVspoof train/dev split (for paper results)
    #   df_train_raw = pd.read_parquet('train_features.parquet')
    #   df_eval_raw  = pd.read_parquet('dev_features.parquet')
    #   Set USE_OFFICIAL_SPLIT = True below
    # -------------------------------------------------------------------

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

    # Info
    n_utt  = df_train_raw['name'].nunique()
    n_feat = len(CFG['features'])
    mem_gb = (n_utt * CFG['max_len'] * n_feat * 4) / 1e9
    print(f"Utterances : {n_utt}")
    print(f"Frames     : {len(df_train_raw)}")
    print(f"Tensor mem : ~{mem_gb:.1f} GB")
    print(f"Key counts : "
          f"{df_train_raw.groupby('name')['key'].first().value_counts().to_dict()}")

    # -------------------------------------------------------------------
    # Build train / eval splits
    # -------------------------------------------------------------------

    if USE_OFFICIAL_SPLIT:
        # Official split — all bonafide from train, all of dev for eval
        df_train      = df_train_raw[df_train_raw['key'] == 'bonafide']
        df_eval       = df_eval_raw
        n_train_bf    = df_train['name'].nunique()
        n_eval_bf     = (df_eval_raw['key'] == 'bonafide').sum()
        n_eval_spoof  = (df_eval_raw['key'] == 'spoof').sum()
        print(f"\nOfficial split:")
        print(f"  Train bonafide : {n_train_bf}")
        print(f"  Eval  bonafide : {df_eval['name'][df_eval['key']=='bonafide'].nunique()}")
        print(f"  Eval  spoof    : {df_eval['name'][df_eval['key']=='spoof'].nunique()}")
    else:
        # Random 80/20 split on bonafide utterances
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
    assert df_train['key'].nunique() == 1, "Spoof files in train set!"
    assert df_train['key'].unique()[0] == 'bonafide', "Train not bonafide!"
    if not USE_OFFICIAL_SPLIT:
        overlap = set(df_train['name'].unique()) & \
                  set(df_eval[df_eval['key']=='bonafide']['name'].unique())
        assert len(overlap) == 0, f"Train/eval overlap: {len(overlap)}!"
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
    model = LSTMPredictor(n_feat, CFG['hidden_dim'],
                          CFG['num_layers'], CFG['dropout']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0    = CFG['warmrestart_T0'],
        T_mult = CFG['warmrestart_Tmult'],
        eta_min= 1e-6,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    meta = (df_eval if USE_OFFICIAL_SPLIT else df_train_raw) \
           .groupby('name')[['key', 'system_id']].first().reset_index()

    train_losses = []
    eval_log     = []

    for epoch in range(CFG['epochs']):
        loss = train_epoch(model, train_loader, optimizer, device, use_bf16)
        train_losses.append(loss)
        scheduler.step(epoch)

        if (epoch + 1) % CFG['eval_every'] == 0:
            scores_tmp = compute_scores(model, eval_loader, device, use_bf16)
            scores_tmp = scores_tmp.merge(meta, on='name')
            bf         = scores_tmp[scores_tmp['key'] == 'bonafide']['score']
            sp         = scores_tmp[scores_tmp['key'] == 'spoof']['score']
            lr_now     = optimizer.param_groups[0]['lr']
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

    print(scores_df.groupby('system_id')['score']
          .agg(['mean', 'std', 'count'])
          .sort_values('mean', ascending=False))

    auc, eer, fpr, tpr = evaluate(scores_df)

    split_tag = 'official' if USE_OFFICIAL_SPLIT else 'random'
    full_diagnostic_report(
        scores_df, fpr, tpr, auc, train_losses,
        eval_log  = eval_log,
        save_path = f'trajectory_report_{split_tag}.png'
    )

    # Save model + scaler
    torch.save({
        'model_state':       model.state_dict(),
        'scaler_mean':       train_set.scaler.mean_,
        'scaler_std':        train_set.scaler.scale_,
        'cfg':               CFG,
        'use_official_split':USE_OFFICIAL_SPLIT,
    }, f'trajectory_model_{split_tag}.pt')
    print(f"Model saved to trajectory_model_{split_tag}.pt")
