"""
Trajectory Anomaly Detection — LSTM Autoencoder (768D)
-------------------------------------------------------
Trains on bonafide utterance trajectories only.
Anomaly score = phone-weighted reconstruction error.
Higher score = more anomalous = more likely spoof.

Input features per frame: HuBERT dims 0..767 + nn_distance  [769-dim]

Optimised for Hopper (H100) GPUs:
  - BF16 mixed precision via torch.autocast
  - Batch size 256
  - num_workers=8 with persistent_workers
  - dict-based dataset (fast per-utterance lookup, no repeated df scans)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -----------------------------------------------------------------------
# 0. Config
# -----------------------------------------------------------------------

HUBERT_COLS = [str(i) for i in range(768)]  # '0', '1', ... '767'

CFG = dict(
    features         = HUBERT_COLS + ['nn_distance'],  # 769 input channels
    hidden_dim       = 256,
    num_layers       = 2,
    dropout          = 0.1,
    max_len          = 500,
    batch_size       = 256,   # large batch — H100 has headroom
    epochs           = 200,
    lr               = 1e-3,
    clip_grad        = 1.0,
    device           = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_bf16         = True,  # BF16 autocast — faster on Hopper, ignored on CPU
    num_workers      = 8,     # dataloader workers — tune to your node's CPU count
    eval_every       = 10,    # evaluate on held-out set every N epochs
    seed             = 42,
)

# Phone weights — applied to per-frame reconstruction error
PHONE_WEIGHTS = {
    'sp': 3.0, 'sil': 3.0,
    'F':  2.5, 'K':   2.5, 'D': 2.5,
    'AY': 2.0, 'M':   2.0,
    'S':  2.0, 'SH':  2.0,
    'Z':  1.5, 'V':   1.5, 'TH': 1.5, 'DH': 1.5,
}
DEFAULT_WEIGHT = 1.0


# -----------------------------------------------------------------------
# 1. Dataset — two variants: eager (train) and lazy (eval)
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Eager dataset — pre-builds all padded sequences in memory.
    Use for the TRAIN set (small, needs fast repeated access).
    """

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
        self.weights   = []

        for name, grp in df.groupby('name', sort=False):
            feats  = grp[feat_cols].values.astype(np.float32)
            phones = grp['phone_base'].values
            T      = min(len(feats), max_len)

            seq     = np.zeros((max_len, len(feat_cols)), dtype=np.float32)
            seq[:T] = feats[:T]
            w       = np.zeros(max_len, dtype=np.float32)
            w[:T]   = [PHONE_WEIGHTS.get(p, DEFAULT_WEIGHT) for p in phones[:T]]

            self.names.append(name)
            self.sequences.append(seq)
            self.lengths.append(T)
            self.weights.append(w)

        self.sequences = np.stack(self.sequences)
        self.weights   = np.stack(self.weights)
        self.lengths   = np.array(self.lengths)
        print(f"  Train dataset ready: {len(self.names)} utterances  "
              f"({self.sequences.nbytes / 1e9:.1f} GB)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),
            int(self.lengths[idx]),
            torch.from_numpy(self.weights[idx]),
            self.names[idx],
        )


class LazyTrajectoryDataset(Dataset):
    """
    Lazy dataset — stores the scaled DataFrame and builds each sequence
    on demand in __getitem__. No large tensor pre-allocation.
    Use for the EVAL set (large, each utterance accessed only once per eval).
    """

    def __init__(self, df, scaler, max_len=500):
        self.max_len   = max_len
        self.feat_cols = CFG['features']

        # Apply scaler in one vectorised pass — no copy of padded tensors
        print("  Scaling eval features...")
        df = df.copy()
        df[self.feat_cols] = scaler.transform(df[self.feat_cols].values
                                               .astype(np.float32))

        # Store per-utterance slices as a dict for O(1) lookup
        print("  Building eval utterance index (lazy)...")
        self.names  = []
        self.groups = {}
        for name, grp in df.groupby('name', sort=False):
            self.names.append(name)
            self.groups[name] = (
                grp[self.feat_cols].values.astype(np.float32),
                grp['phone_base'].values,
            )
        print(f"  Eval dataset ready: {len(self.names)} utterances  "
              f"(lazy — no pre-allocation)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name         = self.names[idx]
        feats, phones = self.groups[name]
        T            = min(len(feats), self.max_len)

        seq     = np.zeros((self.max_len, len(self.feat_cols)), dtype=np.float32)
        seq[:T] = feats[:T]
        w       = np.zeros(self.max_len, dtype=np.float32)
        w[:T]   = [PHONE_WEIGHTS.get(p, DEFAULT_WEIGHT) for p in phones[:T]]

        return (
            torch.from_numpy(seq),
            T,
            torch.from_numpy(w),
            name,
        )


def collate_fn(batch):
    seqs, lengths, weights, names = zip(*batch)
    return (
        torch.stack(seqs),
        torch.tensor(lengths),
        torch.stack(weights),
        list(names),
    )


# -----------------------------------------------------------------------
# 2. Model — LSTM Autoencoder
# -----------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, c) = self.lstm(packed)
        return h, c   # (num_layers, B, hidden_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj       = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, h, c, target_len):
        B         = h.shape[1]
        dec_input = torch.zeros(B, target_len, self.hidden_dim, device=h.device)
        out, _    = self.lstm(dec_input, (h, c))
        return self.proj(out)   # (B, T, output_dim)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_dim, num_layers, dropout, input_dim)

    def forward(self, x, lengths):
        h, c  = self.encoder(x, lengths)
        recon = self.decoder(h, c, x.shape[1])
        return recon


# -----------------------------------------------------------------------
# 3. Weighted reconstruction loss
# -----------------------------------------------------------------------

def weighted_mse(recon, target, weights):
    """
    Per-frame MSE weighted by phone_weight, masked on padding (weight=0).
    Normalised by total weight per utterance to remove length bias.
    """
    err      = ((recon - target) ** 2).mean(dim=-1)   # (B, T)
    weighted = err * weights
    denom    = weights.sum(dim=-1).clamp(min=1e-6)
    return (weighted.sum(dim=-1) / denom).mean()


# -----------------------------------------------------------------------
# 4. Training — BF16 autocast
# -----------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, use_bf16):
    model.train()
    total_loss  = 0.0
    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    # autocast is a no-op on CPU
    ctx = torch.autocast(device_type='cuda', dtype=autocast_dt) \
          if device == 'cuda' else torch.autocast(device_type='cpu', enabled=False)

    for seqs, lengths, weights, _ in loader:
        seqs    = seqs.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad()
        with ctx:
            recon = model(seqs, lengths)
            loss  = weighted_mse(recon, seqs, weights)

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
    model.eval()
    all_names, all_scores = [], []
    autocast_dt = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.autocast(device_type='cuda', dtype=autocast_dt) \
          if device == 'cuda' else torch.autocast(device_type='cpu', enabled=False)

    for seqs, lengths, weights, names in loader:
        seqs    = seqs.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        with ctx:
            recon = model(seqs, lengths)

        # Per-utterance weighted MSE — cast to float32 for stability
        recon   = recon.float()
        seqs    = seqs.float()
        err     = ((recon - seqs) ** 2).mean(dim=-1)
        weighted = err * weights
        denom   = weights.sum(dim=-1).clamp(min=1e-6)
        scores  = (weighted.sum(dim=-1) / denom).cpu().numpy()

        all_names.extend(names)
        all_scores.extend(scores.tolist())

    return pd.DataFrame({'name': all_names, 'score': all_scores})


# -----------------------------------------------------------------------
# 6. Evaluation
# -----------------------------------------------------------------------

def evaluate(scores_df, verbose=True):
    y_true  = (scores_df['key'] == 'spoof').astype(int)
    y_score = scores_df['score']
    auc     = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2
    if verbose:
        print(f"AUC-ROC : {auc:.4f}")
        print(f"EER     : {eer*100:.2f}%")
    return auc, eer, fpr, tpr


# -----------------------------------------------------------------------
# 7. Diagnostic report
# -----------------------------------------------------------------------

def full_diagnostic_report(scores_df, fpr, tpr, auc, train_losses,
                            eval_log=None, save_path=None):
    """
    eval_log: list of (epoch, bf_mean, bf_std, sp_mean) tuples
              from periodic eval during training — shown as extra panel if provided
    """
    bonafide_mean = scores_df[scores_df['key'] == 'bonafide']['score'].mean()
    systems       = sorted([s for s in scores_df['system_id'].unique() if s != '-'])

    n_rows = 4 if eval_log else 3
    fig    = plt.figure(figsize=(16, n_rows * 5))
    gs     = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # Training loss
    ax1.plot(train_losses, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (bonafide only)')

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
    ax3.set_xlabel('Anomaly score')
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

    # Optional: bonafide vs spoof mean scores over training
    if eval_log:
        ax6 = fig.add_subplot(gs[3, :])
        epochs_logged = [e for e, *_ in eval_log]
        bf_means      = [bf for _, bf, _, _ in eval_log]
        bf_stds       = [s  for _, _, s, _ in eval_log]
        sp_means      = [sp for _, _, _, sp in eval_log]
        ax6.plot(epochs_logged, bf_means, color='steelblue', label='bonafide mean')
        ax6.fill_between(epochs_logged,
                         [m - s for m, s in zip(bf_means, bf_stds)],
                         [m + s for m, s in zip(bf_means, bf_stds)],
                         alpha=0.2, color='steelblue', label='bonafide ±1σ')
        ax6.plot(epochs_logged, sp_means, color='tomato', label='spoof mean')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Anomaly score')
        ax6.set_title('Score Separation During Training')
        ax6.legend()

    fig.suptitle('Trajectory LSTM Autoencoder — Full Diagnostic Report',
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
    device  = CFG['device']
    use_bf16 = CFG['use_bf16'] and device == 'cuda'
    print(f"Device  : {device}")
    print(f"BF16    : {use_bf16}")

    # --- Load data ---
    # Replace with your actual path. Parquet recommended for speed.
    # df = pd.read_parquet('train_features.parquet')
    df = pd.read_csv('your_data.csv')
    # Expected columns: name, 0..767, nn_distance, phone_base, key, system_id

    # Verify HuBERT columns
    missing = [c for c in HUBERT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} HuBERT columns — "
                         f"first missing: {missing[0]}.")

    # Memory estimate
    n_utt  = df['name'].nunique()
    n_feat = len(CFG['features'])
    mem_gb = (n_utt * CFG['max_len'] * n_feat * 4) / 1e9
    print(f"Utterances : {n_utt}")
    print(f"Frames     : {len(df)}")
    print(f"Tensor mem : ~{mem_gb:.1f} GB")
    print(f"Key counts : {df.groupby('name')['key'].first().value_counts().to_dict()}")

    # --- Train / eval split on bonafide (no leakage) ---
    rng            = np.random.default_rng(CFG['seed'])
    bonafide_names = df[df['key'] == 'bonafide']['name'].unique()
    rng.shuffle(bonafide_names)
    n_train              = int(len(bonafide_names) * 0.8)
    train_bonafide_names = bonafide_names[:n_train]
    eval_bonafide_names  = bonafide_names[n_train:]
    spoof_names          = df[df['key'] == 'spoof']['name'].unique()
    eval_names           = np.concatenate([eval_bonafide_names, spoof_names])

    df_train = df[df['name'].isin(train_bonafide_names)]
    df_eval  = df[df['name'].isin(eval_names)]

    print(f"\nTrain bonafide : {len(train_bonafide_names)}")
    print(f"Eval  bonafide : {len(eval_bonafide_names)}")
    print(f"Eval  spoof    : {len(spoof_names)}")

    # --- Datasets ---
    # Train: eager (pre-built tensors, fast repeated access over 200 epochs)
    # Eval:  lazy  (on-demand, avoids ~35 GB pre-allocation for 23k utterances)
    print("\nBuilding train dataset (eager)...")
    train_set = TrajectoryDataset(df_train, fit_scaler=True,
                                  max_len=CFG['max_len'])
    print("Building eval dataset (lazy)...")
    eval_set  = LazyTrajectoryDataset(df_eval, scaler=train_set.scaler,
                                      max_len=CFG['max_len'])

    # --- DataLoaders ---
    loader_kwargs = dict(
        collate_fn        = collate_fn,
        num_workers       = CFG['num_workers'],
        pin_memory        = device == 'cuda',
        persistent_workers= CFG['num_workers'] > 0,
    )
    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'],
                              shuffle=True,  **loader_kwargs)
    eval_loader  = DataLoader(eval_set,  batch_size=CFG['batch_size'],
                              shuffle=False, **loader_kwargs)

    # --- Model ---
    model = LSTMAutoencoder(n_feat, CFG['hidden_dim'],
                            CFG['num_layers'], CFG['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training loop ---
    meta         = df.groupby('name')[['key', 'system_id']].first().reset_index()
    train_losses = []
    eval_log     = []   # (epoch, bf_mean, bf_std, sp_mean)

    for epoch in range(CFG['epochs']):
        loss = train_epoch(model, train_loader, optimizer, device, use_bf16)
        train_losses.append(loss)
        scheduler.step(loss)

        # Periodic eval — tracks separation opening up during training
        if (epoch + 1) % CFG['eval_every'] == 0:
            scores_tmp = compute_scores(model, eval_loader, device, use_bf16)
            scores_tmp = scores_tmp.merge(meta, on='name')
            bf         = scores_tmp[scores_tmp['key'] == 'bonafide']['score']
            sp         = scores_tmp[scores_tmp['key'] == 'spoof']['score']
            lr_now     = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}/{CFG['epochs']} "
                  f"loss {loss:.4f}  "
                  f"bonafide {bf.mean():.3f}±{bf.std():.3f}  "
                  f"spoof {sp.mean():.3f}  "
                  f"lr {lr_now:.2e}")
            eval_log.append((epoch + 1, bf.mean(), bf.std(), sp.mean()))
        elif (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}/{CFG['epochs']} "
                  f"loss {loss:.4f}  lr {lr_now:.2e}")

    # --- Final scoring & evaluation ---
    print("\nFinal evaluation...")
    scores_df = compute_scores(model, eval_loader, device, use_bf16)
    scores_df = scores_df.merge(meta, on='name')

    print(scores_df.groupby('system_id')['score']
          .agg(['mean', 'std', 'count'])
          .sort_values('mean', ascending=False))

    auc, eer, fpr, tpr = evaluate(scores_df)

    # --- Report ---
    full_diagnostic_report(
        scores_df, fpr, tpr, auc, train_losses,
        eval_log  = eval_log,
        save_path = 'trajectory_report.png'
    )

    # --- Save model ---
    torch.save({
        'model_state': model.state_dict(),
        'scaler_mean': train_set.scaler.mean_,
        'scaler_std':  train_set.scaler.scale_,
        'cfg':         CFG,
    }, 'trajectory_model.pt')
    print("Model saved to trajectory_model.pt")
