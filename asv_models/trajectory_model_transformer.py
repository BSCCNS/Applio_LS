"""
Trajectory Anomaly Detection — Masked Acoustic Transformer (768D)
-----------------------------------------------------------------
Trains on bonafide utterance trajectories only.
Anomaly score = mean reconstruction error at masked frame positions.
Higher score = more anomalous = more likely spoof.

Architecture: BERT-style masked prediction (MAM — Masked Acoustic Modelling)
  - Randomly mask 15% of frames per utterance
  - Transformer encoder reconstructs masked frames from bidirectional context
  - Train on bonafide only — model learns the physiological manifold
  - At inference: mask frames N times, average reconstruction error

Masking strategies:
  'random' : standard BERT — mask random individual frames (15%)
  'span'   : mask contiguous spans of 1-10 frames — harder task

Changes from v1:
  - lr 5e-4 → 1e-3    (faster convergence)
  - warmup 20 → 5     (less warmup needed)
  - epochs 700 → 1000 (model still converging at 700)
  - n_masks 5 → 20    (reduces score variance)
  - checkpoint save/resume (every CHECKPOINT_EVERY epochs)
  - RESCORE_ONLY mode: load checkpoint, re-score with more masks

Input: ContentVec/HuBERT layer 8 embeddings [768-dim]
"""

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import os
import random
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT       = "/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019"
train_file = f"{ROOT}/data_prep/feat_768d_train_layer_8_tag.parquet"
dev_file   = f"{ROOT}/data_prep/feat_768d_dev_layer_8_tag.parquet"


# -----------------------------------------------------------------------
# 0. Config
# -----------------------------------------------------------------------

HUBERT_COLS = [str(i) for i in range(768)]

CFG = dict(
    features          = HUBERT_COLS,
    # Transformer architecture
    d_model           = 256,
    nhead             = 8,
    num_layers        = 4,
    dim_feedforward   = 1024,
    dropout           = 0.1,
    # Masking
    mask_strategy     = 'random',  # 'random' easier/faster than 'span'
    mask_prob         = 0.15,
    span_min          = 1,
    span_max          = 10,
    # Scoring
    n_masks           = 20,        # masks averaged per utterance at eval
    # Training
    max_len           = 500,
    batch_size        = 128,
    epochs            = 1000,
    lr                = 1e-3,      # increased from 5e-4
    warmup_epochs     = 5,         # reduced from 20
    clip_grad         = 1.0,
    device            = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_bf16          = True,
    num_workers       = 8,
    eval_every        = 10,
    checkpoint_every  = 100,       # save checkpoint every N epochs
    seed              = 42,
)

# -----------------------------------------------------------------------
# Run modes
# -----------------------------------------------------------------------
USE_OFFICIAL_SPLIT = True

# RESCORE_ONLY: skip training, load checkpoint, re-score with n_masks
# Set to path of saved checkpoint, or None to train from scratch
RESCORE_ONLY       = 'checkpoints/trajectory_model_transformer_official.pt'
# Example: RESCORE_ONLY = 'checkpoints/transformer_official_epoch700.pt'

# RESUME_FROM: continue training from a checkpoint
RESUME_FROM        = None
# Example: RESUME_FROM = 'checkpoints/transformer_official_epoch700.pt'

CHECKPOINT_DIR     = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Datasets
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Eager dataset for TRAIN set."""

    def __init__(self, df, scaler=None, fit_scaler=False, max_len=500):
        self.max_len = max_len
        feat_cols    = CFG['features']

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
        self.names, self.sequences, self.lengths = [], [], []

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
    """Lazy dataset for EVAL set."""

    def __init__(self, df, scaler, max_len=500):
        self.max_len   = max_len
        self.feat_cols = CFG['features']

        print("  Scaling eval features...")
        df = df.copy()
        df[self.feat_cols] = scaler.transform(
            df[self.feat_cols].values.astype(np.float32)
        )

        print("  Building eval utterance index (lazy)...")
        self.names, self.groups = [], {}
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
# 2. Masking
# -----------------------------------------------------------------------

def create_mask(lengths, mask_prob=0.15, strategy='random',
                span_min=1, span_max=10):
    """
    Returns (B, max_len) bool mask — True = masked.
    Masking only applied within valid (non-padding) frames.
    """
    B       = len(lengths)
    max_len = CFG['max_len']
    mask    = torch.zeros(B, max_len, dtype=torch.bool)

    for b in range(B):
        L = int(lengths[b])
        if L < 2:
            continue

        if strategy == 'random':
            frame_mask   = torch.rand(L) < mask_prob
            mask[b, :L]  = frame_mask

        elif strategy == 'span':
            n_to_mask = max(1, int(L * mask_prob))
            masked, attempts = 0, 0
            while masked < n_to_mask and attempts < 100:
                span  = random.randint(span_min, min(span_max, L - 1))
                start = random.randint(0, L - span)
                mask[b, start:start + span] = True
                masked   += span
                attempts += 1

    return mask


# -----------------------------------------------------------------------
# 3. Model
# -----------------------------------------------------------------------

class MaskedAcousticTransformer(nn.Module):
    """
    BERT-style masked acoustic model.

    Bidirectional transformer encoder reconstructs masked frames from
    full context. Trained on bonafide only — learns the physiological
    manifold. High reconstruction error at inference = anomalous = spoof.
    """

    def __init__(self, input_dim, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.pos_enc    = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, mask, lengths):
        B, T, _ = x.shape
        device  = x.device

        h = self.input_proj(x)

        # Pad mask to max_len and apply
        if mask.shape[1] < T:
            pad        = torch.zeros(B, T - mask.shape[1],
                                     dtype=torch.bool, device=device)
            mask_padded = torch.cat([mask.to(device), pad], dim=1)
        else:
            mask_padded = mask.to(device)[:, :T]

        h[mask_padded] = self.mask_token.to(device=device, dtype=h.dtype)

        positions = torch.arange(T, device=device).unsqueeze(0)
        h = h + self.pos_enc(positions)

        pad_mask = torch.arange(T, device=device)[None, :] \
                   >= lengths.to(device)[:, None]
        h = self.transformer(h, src_key_padding_mask=pad_mask)

        return self.output_proj(h)


# -----------------------------------------------------------------------
# 4. Loss
# -----------------------------------------------------------------------

def masked_reconstruction_mse(recon, target, mask, lengths):
    """MSE at masked positions only, ignoring padding."""
    device      = recon.device
    length_mask = torch.arange(recon.shape[1], device=device)[None, :] \
                  < lengths.to(device)[:, None]
    valid_mask  = mask.to(device) & length_mask

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    err = ((recon - target) ** 2).mean(dim=-1)
    return err[valid_mask].mean()


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

        mask = create_mask(
            lengths.cpu(),
            mask_prob = CFG['mask_prob'],
            strategy  = CFG['mask_strategy'],
            span_min  = CFG['span_min'],
            span_max  = CFG['span_max'],
        ).to(device)

        optimizer.zero_grad()
        with ctx:
            recon = model(seqs, mask, lengths)
            loss  = masked_reconstruction_mse(recon, seqs, mask, lengths)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------------------------------------------------
# 6. Scoring
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_scores(model, loader, device, use_bf16,
                   n_masks=None):
    """
    Per-utterance anomaly score averaged over n_masks random masks.
    More masks = more stable score, slower eval.
    """
    if n_masks is None:
        n_masks = CFG['n_masks']

    model.eval()
    all_names, all_scores = [], []
    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.autocast(device_type='cuda', dtype=autocast_dt)
           if device == 'cuda'
           else torch.autocast(device_type='cpu', enabled=False))

    for seqs, lengths, names in loader:
        seqs    = seqs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        B       = seqs.shape[0]

        batch_scores = torch.zeros(B, device=device)

        for _ in range(n_masks):
            mask = create_mask(
                lengths.cpu(),
                mask_prob = CFG['mask_prob'],
                strategy  = CFG['mask_strategy'],
                span_min  = CFG['span_min'],
                span_max  = CFG['span_max'],
            ).to(device)

            with ctx:
                recon = model(seqs, mask, lengths)

            recon  = recon.float()
            target = seqs.float()

            length_mask = torch.arange(seqs.shape[1], device=device)[None, :] \
                          < lengths[:, None]
            valid_mask  = mask & length_mask
            err         = ((recon - target) ** 2).mean(dim=-1)
            denom       = valid_mask.sum(dim=-1).float().clamp(min=1.0)
            batch_scores += (err * valid_mask).sum(dim=-1) / denom

        batch_scores /= n_masks
        all_names.extend(names)
        all_scores.extend(batch_scores.cpu().numpy().tolist())

    return pd.DataFrame({'name': all_names, 'score': all_scores})


# -----------------------------------------------------------------------
# 7. Evaluation
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
# 8. Checkpoint helpers
# -----------------------------------------------------------------------

def save_checkpoint(epoch, model, optimizer, scheduler,
                    train_losses, eval_log, scaler, tag='official'):
    path = os.path.join(
        CHECKPOINT_DIR,
        f'transformer_{tag}_epoch{epoch+1}.pt'
    )
    torch.save({
        'epoch':        epoch,
        'model_state':  model.state_dict(),
        'optimizer':    optimizer.state_dict(),
        'scheduler':    scheduler.state_dict(),
        'train_losses': train_losses,
        'eval_log':     eval_log,
        'scaler_mean':  scaler.mean_,
        'scaler_std':   scaler.scale_,
        'cfg':          CFG,
    }, path)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch  = ckpt.get('epoch', -1) + 1
    train_losses = ckpt.get('train_losses', [])
    eval_log     = ckpt.get('eval_log', [])
    scaler_mean  = ckpt.get('scaler_mean', None)
    scaler_std   = ckpt.get('scaler_std', None)
    print(f"  Resumed from epoch {start_epoch}")
    return start_epoch, train_losses, eval_log, scaler_mean, scaler_std


# -----------------------------------------------------------------------
# 9. Diagnostic report
# -----------------------------------------------------------------------

def full_diagnostic_report(scores_df, fpr, tpr, auc, train_losses,
                            eval_log=None, save_path=None):
    bonafide_mean = scores_df[scores_df['key'] == 'bonafide']['score'].mean()
    systems       = sorted([s for s in scores_df['system_id'].unique()
                            if s is not None and s != '-'])

    n_rows = 4 if eval_log else 3
    fig    = plt.figure(figsize=(16, n_rows * 5))
    gs     = gridspec.GridSpec(n_rows, 2, figure=fig,
                               hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    ax1.plot(train_losses, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(
        f'Training Loss — Masked Acoustic Transformer\n'
        f'({CFG["mask_strategy"]} masking, '
        f'{CFG["mask_prob"]*100:.0f}% masked, '
        f'{CFG["num_layers"]} layers, d={CFG["d_model"]}, '
        f'n_masks={CFG["n_masks"]})'
    )

    ax2.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.axhline(0.661, color='gray', linestyle=':', lw=1.5,
                label='LSTM baseline (0.661)')
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curve')
    ax2.legend()

    for label, color in [('bonafide', 'steelblue'), ('spoof', 'tomato')]:
        subset = scores_df[scores_df['key'] == label]['score']
        ax3.hist(subset, bins=60, alpha=0.55, color=color,
                 label=f'{label} (n={len(subset)})', density=True)
    ax3.set_xlabel('Anomaly score (reconstruction error)')
    ax3.set_ylabel('Density')
    ax3.set_title('Bonafide vs Spoof Score Distributions')
    ax3.legend()

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
        ax6.plot(epochs_logged, sp_means, color='tomato', label='spoof mean')
        ax6.axhline(0.661, color='gray', linestyle=':', lw=1,
                    label='LSTM baseline')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Anomaly score')
        ax6.set_title('Score Separation During Training')
        ax6.legend()

    fig.suptitle('Masked Acoustic Transformer — Full Diagnostic Report',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------
# 10. Main
# -----------------------------------------------------------------------

if __name__ == '__main__':

    torch.manual_seed(CFG['seed'])
    device   = CFG['device']
    use_bf16 = CFG['use_bf16'] and device == 'cuda'
    split_tag = 'official' if USE_OFFICIAL_SPLIT else 'random'

    print(f"Device        : {device}")
    print(f"BF16          : {use_bf16}")
    print(f"Architecture  : Transformer d={CFG['d_model']} "
          f"h={CFG['nhead']} layers={CFG['num_layers']}")
    print(f"Mask strategy : {CFG['mask_strategy']} "
          f"({CFG['mask_prob']*100:.0f}% masked, n_masks={CFG['n_masks']})")
    print(f"Mode          : "
          f"{'RESCORE' if RESCORE_ONLY else 'RESUME' if RESUME_FROM else 'TRAIN'}")

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    if USE_OFFICIAL_SPLIT:
        print("\nLoading official train/dev split...")
        df_train_raw = pd.read_parquet(train_file)
        df_eval_raw  = pd.read_parquet(dev_file)
    else:
        print("\nLoading data (random 80/20 bonafide split)...")
        df_train_raw = pd.read_parquet(train_file)
        df_eval_raw  = None

    missing = [c for c in HUBERT_COLS if c not in df_train_raw.columns]
    if missing:
        raise ValueError(f"Missing HuBERT columns — first: {missing[0]}")

    n_utt  = df_train_raw['name'].nunique()
    n_feat = len(CFG['features'])
    mem_gb = (n_utt * CFG['max_len'] * n_feat * 4) / 1e9
    print(f"Utterances : {n_utt}  |  Frames: {len(df_train_raw)}  "
          f"|  Tensor mem: ~{mem_gb:.1f} GB")
    print(f"Key counts : "
          f"{df_train_raw.groupby('name')['key'].first().value_counts().to_dict()}")

    # -------------------------------------------------------------------
    # Build splits
    # -------------------------------------------------------------------
    if USE_OFFICIAL_SPLIT:
        df_train = df_train_raw[df_train_raw['key'] == 'bonafide']
        df_eval  = df_eval_raw
        print(f"\nOfficial split — "
              f"train bonafide: {df_train['name'].nunique()}  "
              f"eval bonafide: {df_eval[df_eval['key']=='bonafide']['name'].nunique()}  "
              f"eval spoof: {df_eval[df_eval['key']=='spoof']['name'].nunique()}")
    else:
        rng            = np.random.default_rng(CFG['seed'])
        bonafide_names = df_train_raw[df_train_raw['key'] == 'bonafide']['name'].unique()
        rng.shuffle(bonafide_names)
        n_train              = int(len(bonafide_names) * 0.8)
        train_bonafide_names = bonafide_names[:n_train]
        eval_bonafide_names  = bonafide_names[n_train:]
        spoof_names          = df_train_raw[df_train_raw['key'] == 'spoof']['name'].unique()
        df_train = df_train_raw[df_train_raw['name'].isin(train_bonafide_names)]
        df_eval  = df_train_raw[df_train_raw['name'].isin(
                       np.concatenate([eval_bonafide_names, spoof_names]))]
        print(f"\nRandom 80/20 — train: {len(train_bonafide_names)}  "
              f"eval bonafide: {len(eval_bonafide_names)}  "
              f"spoof: {len(spoof_names)}")

    assert df_train['key'].nunique() == 1
    assert df_train['key'].unique()[0] == 'bonafide'
    print("Sanity checks passed ✓")

    # -------------------------------------------------------------------
    # Datasets & loaders
    # -------------------------------------------------------------------
    print("\nBuilding datasets...")

    # If resuming/rescoring, restore scaler from checkpoint
    if (RESCORE_ONLY or RESUME_FROM):
        ckpt_path = RESCORE_ONLY or RESUME_FROM
        ckpt_peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        pre_scaler = StandardScaler()
        pre_scaler.mean_  = ckpt_peek['scaler_mean']
        pre_scaler.scale_ = ckpt_peek['scaler_std']
        pre_scaler.var_   = pre_scaler.scale_ ** 2
        pre_scaler.n_features_in_ = len(CFG['features'])
        train_set = TrajectoryDataset(df_train, scaler=pre_scaler,
                                      max_len=CFG['max_len'])
    else:
        train_set = TrajectoryDataset(df_train, fit_scaler=True,
                                      max_len=CFG['max_len'])

    eval_set = LazyTrajectoryDataset(df_eval, scaler=train_set.scaler,
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
    # Model
    # -------------------------------------------------------------------
    model = MaskedAcousticTransformer(
        input_dim      = n_feat,
        d_model        = CFG['d_model'],
        nhead          = CFG['nhead'],
        num_layers     = CFG['num_layers'],
        dim_feedforward= CFG['dim_feedforward'],
        dropout        = CFG['dropout'],
        max_len        = CFG['max_len'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=1e-4
    )

    def lr_lambda(epoch):
        if epoch < CFG['warmup_epochs']:
            return (epoch + 1) / CFG['warmup_epochs']
        progress = (epoch - CFG['warmup_epochs']) / \
                   max(1, CFG['epochs'] - CFG['warmup_epochs'])
        return max(1e-6 / CFG['lr'], 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if USE_OFFICIAL_SPLIT:
        meta = df_eval.groupby('name')[['key','system_id']].first().reset_index()
    else:
        meta = df_train_raw.groupby('name')[['key','system_id']].first().reset_index()

    # -------------------------------------------------------------------
    # RESCORE ONLY mode — load checkpoint, score, report, exit
    # -------------------------------------------------------------------
    if RESCORE_ONLY:
        load_checkpoint(RESCORE_ONLY, model)
        model.to(device)
        print(f"\nRescoring with n_masks={CFG['n_masks']}...")
        scores_df = compute_scores(model, eval_loader, device, use_bf16)
        scores_df = scores_df.merge(meta, on='name')
        scores_df['system_id'] = scores_df['system_id'].fillna('-')
        print(scores_df.groupby('system_id')['score']
              .agg(['mean','std','count']).sort_values('mean', ascending=False))
        auc, eer, fpr, tpr = evaluate(scores_df)
        tag = os.path.splitext(os.path.basename(RESCORE_ONLY))[0]
        full_diagnostic_report(
            scores_df, fpr, tpr, auc, [],
            save_path=f'trajectory_report_rescore_{tag}.png'
        )
        scores_df.to_csv(f'scores_rescore_{tag}.csv', index=False)
        print("Done.")
        exit(0)

    # -------------------------------------------------------------------
    # Training — with optional resume
    # -------------------------------------------------------------------
    start_epoch  = 0
    train_losses = []
    eval_log     = []

    if RESUME_FROM:
        start_epoch, train_losses, eval_log, _, _ = load_checkpoint(
            RESUME_FROM, model, optimizer, scheduler
        )

    for epoch in range(start_epoch, CFG['epochs']):
        loss = train_epoch(model, train_loader, optimizer, device, use_bf16)
        train_losses.append(loss)
        scheduler.step()

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
                  f"loss {loss:.4f}  lr {lr_now:.2e}")

        # Periodic checkpoint
        if (epoch + 1) % CFG['checkpoint_every'] == 0:
            save_checkpoint(epoch, model, optimizer, scheduler,
                            train_losses, eval_log,
                            train_set.scaler, tag=split_tag)

    # -------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------
    print("\nFinal evaluation...")
    scores_df = compute_scores(model, eval_loader, device, use_bf16)
    scores_df = scores_df.merge(meta, on='name')
    scores_df['system_id'] = scores_df['system_id'].fillna('-')

    print(scores_df.groupby('system_id')['score']
          .agg(['mean','std','count']).sort_values('mean', ascending=False))
    auc, eer, fpr, tpr = evaluate(scores_df)

    full_diagnostic_report(
        scores_df, fpr, tpr, auc, train_losses,
        eval_log  = eval_log,
        save_path = f'trajectory_report_transformer_{split_tag}.png'
    )

    # Final checkpoint
    save_checkpoint(CFG['epochs'] - 1, model, optimizer, scheduler,
                    train_losses, eval_log,
                    train_set.scaler, tag=split_tag)

    scores_df.to_csv(f'scores_transformer_{split_tag}.csv', index=False)
    print(f"Scores saved to scores_transformer_{split_tag}.csv")
