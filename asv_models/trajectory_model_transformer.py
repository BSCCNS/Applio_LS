"""
Trajectory Anomaly Detection — Masked Acoustic Transformer (768D)
-----------------------------------------------------------------
Trains on bonafide utterance trajectories only.
Anomaly score = mean reconstruction error at masked frame positions.
Higher score = more anomalous = more likely spoof.

Architecture: BERT-style masked prediction (MAM — Masked Acoustic Modelling)
  - Randomly mask 15% of frames per utterance
  - Transformer encoder reconstructs masked frames from full bidirectional context
  - Train on bonafide only — model learns the physiological manifold
  - At inference: mask frames, score by reconstruction error

Key advantage over LSTM next-frame predictor:
  - Bidirectional attention — can use both past AND future context
  - Direct long-range attention — breath frame at t=20 attends directly
    to phonation at t=100, no gradient through 80 sequential steps
  - Span masking option forces model to reconstruct entire phoneme
    segments including breath-to-phonation transitions

Masking strategies (set MASK_STRATEGY in CFG):
  'random' : standard BERT — mask random individual frames (15%)
  'span'   : mask contiguous spans of 1-10 frames — harder, tests
             whether the model has learned segment-level dynamics

Input features: ContentVec/HuBERT layer 8 embeddings [768-dim]

Same data format and pipeline as trajectory_model_clean.py.
"""

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

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

ROOT = "/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019"
train_file = f"{ROOT}/data_prep/feat_768d_train_layer_8_tag.parquet"
dev_file   = f"{ROOT}/data_prep/feat_768d_dev_layer_8_tag.parquet"


# -----------------------------------------------------------------------
# 0. Config
# -----------------------------------------------------------------------

HUBERT_COLS = [str(i) for i in range(768)]

CFG = dict(
    features          = HUBERT_COLS,   # 768D ContentVec/HuBERT
    # Transformer architecture
    d_model           = 256,           # transformer hidden dim
    nhead             = 8,             # attention heads (d_model must be divisible)
    num_layers        = 4,             # transformer encoder layers
    dim_feedforward   = 1024,          # FFN hidden dim (4 × d_model)
    dropout           = 0.1,
    # Masking
    mask_strategy     = 'span',        # 'random' or 'span'
    mask_prob         = 0.15,          # fraction of frames to mask
    span_min          = 1,             # min span length (frames)
    span_max          = 10,            # max span length (frames)
    # Training
    max_len           = 500,
    batch_size        = 128,           # smaller than LSTM — attention is O(T²)
    epochs            = 350,
    lr                = 5e-4,          # slightly lower than LSTM — transformers
                                       # benefit from more conservative lr
    warmup_epochs     = 20,            # linear warmup before cosine decay
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
# 1. Datasets (identical to clean LSTM version)
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
# 2. Masking
# -----------------------------------------------------------------------

def create_mask(lengths, mask_prob=0.15, strategy='span',
                span_min=1, span_max=10):
    """
    Create boolean mask tensor — True = masked (to be predicted).

    'random': independently mask each frame with probability mask_prob
    'span':   mask contiguous spans of frames — harder task, forces model
              to reconstruct entire phoneme segments from context

    Returns mask: (B, T) bool tensor on CPU
    """
    B       = len(lengths)
    max_len = CFG['max_len']   # always match padded tensor shape
    mask    = torch.zeros(B, max_len, dtype=torch.bool)

    for b in range(B):
        L = int(lengths[b])
        if L < 2:
            continue

        if strategy == 'random':
            # Independent Bernoulli masking — only within valid frames
            frame_mask = torch.rand(L) < mask_prob
            mask[b, :L] = frame_mask

        elif strategy == 'span':
            # Span masking — mask contiguous runs within valid frames
            n_to_mask = max(1, int(L * mask_prob))
            masked    = 0
            attempts  = 0
            while masked < n_to_mask and attempts < 100:
                span  = random.randint(span_min, min(span_max, L - 1))
                start = random.randint(0, L - span)
                mask[b, start:start + span] = True
                masked   += span
                attempts += 1

    return mask   # (B, max_len) — matches padded tensor shape


# -----------------------------------------------------------------------
# 3. Model — Masked Acoustic Transformer
# -----------------------------------------------------------------------

class MaskedAcousticTransformer(nn.Module):
    """
    BERT-style masked acoustic model for trajectory anomaly detection.

    Given a sequence of ContentVec/HuBERT frames with some positions masked,
    reconstruct the original frames at masked positions using bidirectional
    attention over the full sequence.

    Trained on bonafide only:
      - Model learns what physiologically plausible trajectories look like
      - At inference: high reconstruction error = off-manifold = likely spoof

    Architecture:
      input_proj  : 768D → d_model  (linear projection)
      pos_enc     : learnable positional embeddings (max_len × d_model)
      transformer : N × TransformerEncoderLayer (pre-norm, bidirectional)
      output_proj : d_model → 768D  (reconstruct in original space)

    The mask_token is a learnable vector replacing masked frame content.
    It is the same for all masked positions — the model must use positional
    and contextual information to reconstruct each masked frame.
    """

    def __init__(self, input_dim, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_len):
        super().__init__()

        # Project 768D ContentVec to transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable mask token — replaces masked frame embeddings
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Learnable positional encoding — one vector per position
        self.pos_enc = nn.Embedding(max_len, d_model)

        # Transformer encoder — pre-norm for training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-norm: more stable than post-norm
            activation     = 'gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,  # avoid warning with padding mask
        )

        # Project back to input space for loss computation
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, mask, lengths):
        """
        x:       (B, T, 768)   original frames (unmasked)
        mask:    (B, T) bool   True = masked position
        lengths: (B,)          actual sequence lengths

        Returns recon: (B, T, 768) reconstructed frames at all positions
        Loss is computed only at masked positions.
        """
        B, T, _ = x.shape
        device  = x.device

        # 1. Project input to d_model
        h = self.input_proj(x)                            # (B, T, d_model)

        # 2. Replace masked positions with learnable mask token
        # mask is (B, T_actual) but h is (B, max_len, d_model)
        # Pad mask to max_len along dim=1 to match h
        if mask.shape[1] < T:
            pad = torch.zeros(B, T - mask.shape[1], dtype=torch.bool,
                              device=device)
            mask_padded = torch.cat([mask.to(device), pad], dim=1)
        else:
            mask_padded = mask.to(device)[:, :T]
        # Expand to (B, T, d_model) for assignment
        h[mask_padded] = self.mask_token.to(device=device, dtype=h.dtype)

        # 3. Add positional encoding
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        h = h + self.pos_enc(positions)                   # (B, T, d_model)

        # 4. Padding mask — transformer ignores padded positions
        #    True = ignore this position
        pad_mask = torch.arange(T, device=device)[None, :] \
                   >= lengths.to(device)[:, None]         # (B, T)

        # 5. Bidirectional transformer — attends over full sequence
        h = self.transformer(h, src_key_padding_mask=pad_mask)

        # 6. Project back to input space
        return self.output_proj(h)                        # (B, T, 768)


# -----------------------------------------------------------------------
# 4. Loss — reconstruction error at masked positions only
# -----------------------------------------------------------------------

def masked_reconstruction_mse(recon, target, mask, lengths):
    """
    MSE between reconstructed and original frames, computed only at
    masked positions within valid (non-padding) sequence length.

    recon:   (B, T, 768)
    target:  (B, T, 768)   — original unmasked frames
    mask:    (B, T) bool   — True = masked (compute loss here)
    lengths: (B,)          — valid frame counts
    """
    device = recon.device

    # Valid mask: masked AND within sequence length (not padding)
    length_mask = torch.arange(recon.shape[1], device=device)[None, :] \
                  < lengths.to(device)[:, None]           # (B, T)
    valid_mask  = mask.to(device) & length_mask           # (B, T)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    err  = ((recon - target) ** 2).mean(dim=-1)           # (B, T)
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

        # Create fresh mask for each batch — different mask every pass
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
# 6. Scoring — average over multiple random masks for stability
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_scores(model, loader, device, use_bf16, n_masks=5):
    """
    Per-utterance anomaly score = mean reconstruction error at masked positions.

    Using n_masks > 1 averages over multiple random masks per utterance,
    giving a more stable score than a single mask pass.
    Higher score = more anomalous = more likely spoof.
    """
    model.eval()
    all_names  = []
    all_scores = []

    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.autocast(device_type='cuda', dtype=autocast_dt)
           if device == 'cuda'
           else torch.autocast(device_type='cpu', enabled=False))

    for seqs, lengths, names in loader:
        seqs    = seqs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        B       = seqs.shape[0]

        # Accumulate scores over multiple mask samples
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

            # Per-utterance score — mean error at masked valid positions
            length_mask = torch.arange(seqs.shape[1], device=device)[None, :] \
                          < lengths[:, None]
            valid_mask  = mask & length_mask

            err   = ((recon - target) ** 2).mean(dim=-1)   # (B, T)
            denom = valid_mask.sum(dim=-1).float().clamp(min=1.0)
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
# 8. Diagnostic report (same structure as LSTM version)
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
    ax1.set_title(f'Training Loss — Masked Acoustic Transformer\n'
                  f'({CFG["mask_strategy"]} masking, '
                  f'{CFG["mask_prob"]*100:.0f}% masked, '
                  f'{CFG["num_layers"]} layers, d={CFG["d_model"]})')

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
                    label='LSTM spoof mean baseline')
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
# 9. Main (same structure as LSTM version)
# -----------------------------------------------------------------------

if __name__ == '__main__':

    torch.manual_seed(CFG['seed'])
    device   = CFG['device']
    use_bf16 = CFG['use_bf16'] and device == 'cuda'
    print(f"Device        : {device}")
    print(f"BF16          : {use_bf16}")
    print(f"Architecture  : Transformer d={CFG['d_model']} "
          f"h={CFG['nhead']} layers={CFG['num_layers']}")
    print(f"Mask strategy : {CFG['mask_strategy']} "
          f"({CFG['mask_prob']*100:.0f}% masked)")

    USE_OFFICIAL_SPLIT = True

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

    if USE_OFFICIAL_SPLIT:
        df_train = df_train_raw[df_train_raw['key'] == 'bonafide']
        df_eval  = df_eval_raw
        print(f"\nOfficial split:")
        print(f"  Train bonafide : {df_train['name'].nunique()}")
        print(f"  Eval  bonafide : "
              f"{df_eval[df_eval['key']=='bonafide']['name'].nunique()}")
        print(f"  Eval  spoof    : "
              f"{df_eval[df_eval['key']=='spoof']['name'].nunique()}")
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

    assert df_train['key'].nunique() == 1,            "Spoof in train!"
    assert df_train['key'].unique()[0] == 'bonafide', "Train not bonafide!"
    if not USE_OFFICIAL_SPLIT:
        overlap = set(df_train['name'].unique()) & \
                  set(df_eval[df_eval['key']=='bonafide']['name'].unique())
        assert len(overlap) == 0, f"Train/eval overlap: {len(overlap)}!"
    print("Sanity checks passed ✓")

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

    # Model
    model = MaskedAcousticTransformer(
        input_dim      = n_feat,
        d_model        = CFG['d_model'],
        nhead          = CFG['nhead'],
        num_layers     = CFG['num_layers'],
        dim_feedforward= CFG['dim_feedforward'],
        dropout        = CFG['dropout'],
        max_len        = CFG['max_len'],
    ).to(device)

    optimizer = torch.optim.AdamW(   # AdamW preferred for transformers
        model.parameters(),
        lr           = CFG['lr'],
        weight_decay = 1e-4,
    )

    # Warmup then cosine annealing — standard for transformers
    def lr_lambda(epoch):
        if epoch < CFG['warmup_epochs']:
            return (epoch + 1) / CFG['warmup_epochs']
        progress = (epoch - CFG['warmup_epochs']) / \
                   max(1, CFG['epochs'] - CFG['warmup_epochs'])
        return max(1e-6 / CFG['lr'],
                   0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    if USE_OFFICIAL_SPLIT:
        meta = df_eval.groupby('name')[['key', 'system_id']].first().reset_index()
    else:
        meta = df_train_raw.groupby('name')[['key', 'system_id']].first().reset_index()

    train_losses = []
    eval_log     = []

    for epoch in range(CFG['epochs']):
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
                  f"loss {loss:.4f}  "
                  f"lr {lr_now:.2e}")

    print("\nFinal evaluation...")
    scores_df = compute_scores(model, eval_loader, device, use_bf16)
    scores_df = scores_df.merge(meta, on='name')
    scores_df['system_id'] = scores_df['system_id'].fillna('-')

    print(scores_df.groupby('system_id')['score']
          .agg(['mean', 'std', 'count'])
          .sort_values('mean', ascending=False))

    auc, eer, fpr, tpr = evaluate(scores_df)

    split_tag = 'official' if USE_OFFICIAL_SPLIT else 'random'
    full_diagnostic_report(
        scores_df, fpr, tpr, auc, train_losses,
        eval_log  = eval_log,
        save_path = f'trajectory_report_transformer_{split_tag}.png'
    )

    torch.save({
        'model_state': model.state_dict(),
        'scaler_mean': train_set.scaler.mean_,
        'scaler_std':  train_set.scaler.scale_,
        'cfg':         CFG,
    }, f'trajectory_model_transformer_{split_tag}.pt')
    print(f"Model saved to trajectory_model_transformer_{split_tag}.pt")
