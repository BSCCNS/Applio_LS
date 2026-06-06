"""
Trajectory Anomaly Detection — LSTM Autoencoder (2D version)
-------------------------------------------------------------
Trains on bonafide utterance trajectories only.
Anomaly score = phone-weighted reconstruction error.
Higher score = more anomalous = more likely spoof.

Input features per frame: (x, y, nn_distance)  [3-dim]
"""

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

ROOT = '/gpfs/scratch/bsc21/bsc270816/ls_data/datasets/ASVspoof2019/ASVspoof2019_LA_train_preproc'
file_data_prep = f'{ROOT}/data_prep/feat_768d_layer_8_tag.parquet'

# -----------------------------------------------------------------------
# 0. Config — edit these
# -----------------------------------------------------------------------

HUBERT_COLS = [str(i) for i in range(768)]  # '0', '1', ... '767'

CFG = dict(
    features       = HUBERT_COLS + ['nn_distance'],  # 769 input channels
    hidden_dim     = 256,   # larger capacity for 768D input
    num_layers     = 2,     # LSTM depth
    dropout        = 0.1,
    max_len        = 500,   # pad/truncate to this length
    batch_size     = 32,    # reduced — sequences are much larger now
    epochs         = 200,
    lr             = 1e-3,
    clip_grad      = 1.0,
    device         = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed           = 42,
)

# Phone weights — same as static benchmark, applied to reconstruction error
PHONE_WEIGHTS = {
    'sp': 3.0, 'sil': 3.0,
    'F':  2.5, 'K':   2.5, 'D': 2.5,
    'AY': 2.0, 'M':   2.0,
    'S':  2.0, 'SH':  2.0,
    'Z':  1.5, 'V':   1.5, 'TH': 1.5, 'DH': 1.5,
}
DEFAULT_WEIGHT = 1.0


# -----------------------------------------------------------------------
# 1. Dataset
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    One sample = one utterance trajectory, padded to max_len.
    Returns (sequence, lengths, phone_weights, name).
    """

    def __init__(self, df, scaler=None, fit_scaler=False, max_len=500):
        self.max_len = max_len
        self.names = df['name'].unique()

        # Fit or apply scaler on raw features
        feat_cols = CFG['features']
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

        # Pre-build sequences
        self.sequences = []
        self.lengths   = []
        self.weights   = []

        for name in self.names:
            utt = df[df['name'] == name]
            feats   = utt[feat_cols].values.astype(np.float32)
            phones  = utt['phone_base'].values
            T       = min(len(feats), max_len)

            # Pad / truncate
            seq = np.zeros((max_len, len(feat_cols)), dtype=np.float32)
            seq[:T] = feats[:T]

            # Phone weight mask — zero on padding frames
            w = np.zeros(max_len, dtype=np.float32)
            w[:T] = [PHONE_WEIGHTS.get(p, DEFAULT_WEIGHT) for p in phones[:T]]

            self.sequences.append(seq)
            self.lengths.append(T)
            self.weights.append(w)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),   # (max_len, n_feat)
            self.lengths[idx],                   # int
            torch.tensor(self.weights[idx]),     # (max_len,)
            self.names[idx],                     # str
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
        return h, c  # (num_layers, B, hidden_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, h, c, target_len):
        B = h.shape[1]
        # Decoder input: zeros, driven purely by hidden state
        dec_input = torch.zeros(B, target_len, self.hidden_dim,
                                device=h.device)
        out, _ = self.lstm(dec_input, (h, c))
        return self.proj(out)  # (B, T, output_dim)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, num_layers,
                                   dropout, input_dim)

    def forward(self, x, lengths):
        h, c = self.encoder(x, lengths)
        recon = self.decoder(h, c, x.shape[1])
        return recon


# -----------------------------------------------------------------------
# 3. Weighted reconstruction loss
# -----------------------------------------------------------------------

def weighted_mse(recon, target, weights):
    """
    Per-frame MSE weighted by phone_weight and masked by padding.
    weights: (B, T) — zero on padding frames
    """
    err = ((recon - target) ** 2).mean(dim=-1)  # (B, T)
    weighted = err * weights
    # Normalise by sum of weights (not max_len) to avoid padding bias
    denom = weights.sum(dim=-1).clamp(min=1e-6)
    return (weighted.sum(dim=-1) / denom).mean()


# -----------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for seqs, lengths, weights, _ in loader:
        seqs    = seqs.to(device)
        weights = weights.to(device)
        optimizer.zero_grad()
        recon = model(seqs, lengths)
        loss  = weighted_mse(recon, seqs, weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_scores(model, loader, device):
    """
    Return per-utterance anomaly scores and metadata.
    """
    model.eval()
    all_names, all_scores = [], []

    for seqs, lengths, weights, names in loader:
        seqs    = seqs.to(device)
        weights = weights.to(device)
        recon   = model(seqs, lengths)

        # Per-utterance weighted MSE
        err     = ((recon - seqs) ** 2).mean(dim=-1)   # (B, T)
        weighted = err * weights
        denom   = weights.sum(dim=-1).clamp(min=1e-6)
        scores  = (weighted.sum(dim=-1) / denom).cpu().numpy()

        all_names.extend(names)
        all_scores.extend(scores)

    return pd.DataFrame({'name': all_names, 'score': all_scores})


# -----------------------------------------------------------------------
# 5. Evaluation
# -----------------------------------------------------------------------

def evaluate(scores_df):
    y_true  = (scores_df['key'] == 'spoof').astype(int)
    y_score = scores_df['score']
    auc     = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"AUC-ROC : {auc:.4f}")
    print(f"EER     : {eer*100:.2f}%")
    return auc, eer, fpr, tpr


# -----------------------------------------------------------------------
# 6. Diagnostic plots (same structure as static benchmark)
# -----------------------------------------------------------------------

def plot_training_curve(train_losses, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted MSE loss')
    ax.set_title('Training Loss (bonafide only)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def full_diagnostic_report(scores_df, fpr, tpr, auc, train_losses,
                            save_path=None):
    bonafide_mean = scores_df[scores_df['key'] == 'bonafide']['score'].mean()
    systems = sorted([s for s in scores_df['system_id'].unique() if s != '-'])

    fig = plt.figure(figsize=(16, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # training loss
    ax2 = fig.add_subplot(gs[0, 1])  # ROC
    ax3 = fig.add_subplot(gs[1, 0])  # overall distributions
    ax4 = fig.add_subplot(gs[1, 1])  # per-system box plots
    ax5 = fig.add_subplot(gs[2, :])  # per-system distributions

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

    # Overall distributions
    for label, color in [('bonafide', 'steelblue'), ('spoof', 'tomato')]:
        subset = scores_df[scores_df['key'] == label]['score']
        ax3.hist(subset, bins=60, alpha=0.55, color=color,
                 label=f'{label} (n={len(subset)})', density=True)
    ax3.set_xlabel('Anomaly score')
    ax3.set_ylabel('Density')
    ax3.set_title('Bonafide vs Spoof Score Distributions')
    ax3.legend()

    # Per-system box plots
    box_data  = [scores_df[scores_df['system_id'] == s]['score'].values
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

    # Per-system overlapping histograms
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

    fig.suptitle('Trajectory LSTM Autoencoder — Full Diagnostic Report',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------

if __name__ == '__main__':

    torch.manual_seed(CFG['seed'])
    device = CFG['device']
    print(f"Using device: {device}")

    # --- Load data ---
    # Replace with your actual path
    df = pd.read_parquet(file_data_prep)
    print('df.columns')
    print(df.columns)
    # Expected columns: name, 0, 1, ..., 767, nn_distance, phone_base, key, system_id

    # Verify HuBERT columns are present
    missing = [c for c in HUBERT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} HuBERT columns — "
                         f"first missing: {missing[0]}. "
                         f"Check column names are '0'..'767'.")

    # Memory estimate — warn if large
    n_utt  = df['name'].nunique()
    n_feat = len(CFG['features'])
    mem_gb = (n_utt * CFG['max_len'] * n_feat * 4) / 1e9
    print(f"Estimated dataset tensor memory: {mem_gb:.2f} GB")

    print(f"Total frames : {len(df)}")
    print(f"Utterances   : {df['name'].nunique()}")
    print(f"Key counts   : {df.groupby('name')['key'].first().value_counts().to_dict()}")

    # --- Split bonafide into train / eval (no leakage) ---
    rng = np.random.default_rng(CFG['seed'])
    bonafide_names = df[df['key'] == 'bonafide']['name'].unique()
    rng.shuffle(bonafide_names)

    n_train = int(len(bonafide_names) * 0.8)
    train_bonafide_names = bonafide_names[:n_train]
    eval_bonafide_names  = bonafide_names[n_train:]

    # Train: bonafide train split only
    df_train = df[df['name'].isin(train_bonafide_names)]

    # Eval: held-out bonafide + all spoof — NO training utterances
    spoof_names = df[df['key'] == 'spoof']['name'].unique()
    eval_names  = np.concatenate([eval_bonafide_names, spoof_names])
    df_eval     = df[df['name'].isin(eval_names)]

    print(f"\nBonafide utterances — train: {len(train_bonafide_names)}, "
          f"eval: {len(eval_bonafide_names)}")
    print(f"Spoof utterances   — eval:  {len(spoof_names)}")

    # --- Datasets & loaders ---
    train_set = TrajectoryDataset(df_train, fit_scaler=True,
                                  max_len=CFG['max_len'])
    eval_set  = TrajectoryDataset(df_eval,  scaler=train_set.scaler,
                                  max_len=CFG['max_len'])

    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'],
                              shuffle=True,  collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_set,  batch_size=CFG['batch_size'],
                              shuffle=False, collate_fn=collate_fn)

    # --- Model ---
    n_feat = len(CFG['features'])
    model  = LSTMAutoencoder(n_feat, CFG['hidden_dim'],
                             CFG['num_layers'], CFG['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training loop ---
    train_losses = []
    for epoch in range(CFG['epochs']):
        loss = train(model, train_loader, optimizer, device)
        train_losses.append(loss)
        scheduler.step(loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{CFG['epochs']} — loss: {loss:.4f}")

    # --- Scoring ---
    scores_df = compute_scores(model, eval_loader, device)

    # Attach labels
    meta = df.groupby('name')[['key', 'system_id']].first().reset_index()
    scores_df = scores_df.merge(meta, on='name')

    print(f"\nScored {len(scores_df)} utterances")
    print(scores_df.groupby('system_id')['score']
          .agg(['mean', 'std', 'count'])
          .sort_values('mean', ascending=False))

    # --- Evaluate ---
    auc, eer, fpr, tpr = evaluate(scores_df)

    # --- Report ---
    full_diagnostic_report(
        scores_df, fpr, tpr, auc, train_losses,
        save_path='trajectory_report.png'
    )

    # --- Save model ---
    torch.save({
        'model_state': model.state_dict(),
        'scaler_mean': train_set.scaler.mean_,
        'scaler_std':  train_set.scaler.scale_,
        'cfg':         CFG,
    }, 'trajectory_model.pt')
    print("Model saved to trajectory_model.pt")
