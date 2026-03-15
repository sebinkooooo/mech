"""
sebi.py  –  Temporal Gait Transformer (TGT) utilities
=====================================================
Importable module containing:
  * Insole CSV preprocessing (merge L/R, resample to 60 Hz)
  * Sliding-window segmentation
  * PyTorch Dataset
  * TGT model class
  * Training / validation / testing helpers
  * Plotting helpers (curves, confusion matrix, per-class bar chart)
"""

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── non-interactive matplotlib so headless / subprocess works ────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ═══════════════════════════════════════════════════════════════════════════
#  DATA  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def get_frame_rate(data):
    avg_interval = data["timestamp"].diff().mean()
    return 1 / avg_interval if avg_interval > 0 else None


def downsample_to_60Hz(data, target_fps=60):
    total_time = data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]
    total_frames = len(data)
    current_fps = total_frames / total_time if total_time > 0 else None
    if current_fps and current_fps > target_fps:
        target_frame_count = int(total_time * target_fps)
        indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
        return data.iloc[indices].reset_index(drop=True)
    return data


def upsample_to_60Hz(data, target_fps=60):
    target_interval = 1.0 / target_fps
    new_ts = np.arange(data["timestamp"].min(), data["timestamp"].max(), target_interval)
    orig = data.set_index("timestamp")
    combined = orig.reindex(orig.index.union(new_ts)).sort_index()
    combined = combined.interpolate(method="index")
    data_interp = combined.reindex(new_ts).reset_index()
    data_interp.rename(columns={"index": "timestamp"}, inplace=True)
    return data_interp


def insole_process(file_path):
    """Read a raw insole CSV, merge L/R feet, resample to 60 Hz."""
    target_fps = 60
    data = pd.read_csv(file_path).dropna()
    left = data[data["ele_36"] == 1]
    right = data[data["ele_36"] == 0]

    # 0-17: pressure, 18-20: accelerometer, 30-33: quaternion
    cols = [f"ele_{i}" for i in range(18)] + [
        "ele_18", "ele_19", "ele_20",
        "ele_30", "ele_31", "ele_32", "ele_33",
    ]
    left = left[["timestamp"] + cols]
    right = right[["timestamp"] + cols]

    left_fps = get_frame_rate(left)
    right_fps = get_frame_rate(right)

    if left_fps < right_fps:
        base, high, suffixes = left, right, ("_left", "_right")
    else:
        base, high, suffixes = right, left, ("_right", "_left")

    merged = pd.merge_asof(
        base.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True),
        high.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True),
        on="timestamp",
        direction="nearest",
        suffixes=suffixes,
        tolerance=0.02,
    )

    if any(c.endswith("_right") for c in merged.columns[1:]):
        lcols = [c for c in merged.columns if "_left" in c]
        rcols = [c for c in merged.columns if "_right" in c]
        merged = merged[["timestamp"] + lcols + rcols]

    merged = merged.dropna()
    fps = get_frame_rate(merged)
    if fps < target_fps:
        return upsample_to_60Hz(merged, target_fps)
    if fps > target_fps:
        return downsample_to_60Hz(merged, target_fps)
    return merged


# ── windowing ────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Left pressure (18)
    *[f"ele_{i}_left" for i in range(18)],
    # Left accelerometer (3)
    "ele_18_left", "ele_19_left", "ele_20_left",
    # Left quaternion (4)
    "ele_30_left", "ele_31_left", "ele_32_left", "ele_33_left",
    # Right pressure (18)
    *[f"ele_{i}_right" for i in range(18)],
    # Right accelerometer (3)
    "ele_18_right", "ele_19_right", "ele_20_right",
    # Right quaternion (4)
    "ele_30_right", "ele_31_right", "ele_32_right", "ele_33_right",
]

LABEL_MAPPING = {
    "Normal_walking": 0,
    "Injury_walking": 1,
    "Stepping": 2,
    "Swaying": 3,
    "Jumping": 4,
}

CLASS_NAMES = ["Normal_walking", "Injury_walking", "Stepping", "Swaying", "Jumping"]


def load_and_window(data_dir, feature_columns, label_mapping,
                    window_size, stride):
    """
    Walk *data_dir* (expects sub-folders whose names contain a label key),
    preprocess every CSV, and return windowed numpy arrays.

    Returns (features, labels) as numpy arrays.
    """
    all_features, all_labels = [], []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} does not exist")

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for keyword, label_id in label_mapping.items():
            if keyword in folder_name:
                csvs = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
                for fname in tqdm(csvs, desc=f"{keyword}", unit="file"):
                    fpath = os.path.join(folder_path, fname)
                    try:
                        feats = insole_process(fpath)[feature_columns]
                    except Exception as e:
                        print(f"  ⚠ skipping {fname}: {e}")
                        continue
                    n = feats.shape[0]
                    for s in range(0, n - window_size + 1, stride):
                        all_features.append(feats.iloc[s : s + window_size].values)
                        all_labels.append(label_id)
                break  # matched keyword; no need to check remaining

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

class GaitDataset(Dataset):
    """
    Parameters
    ----------
    features : ndarray (N, W, F)
    labels   : ndarray (N,)
    scaler   : fitted sklearn scaler, optional
        If provided, this scaler is used to *transform* (not fit) the data.
        If None and ``standardize`` or ``normalize`` is True, a new scaler
        is fitted on this data and stored in ``self.scaler``.
    standardize : bool
        Fit / apply StandardScaler (z-score).
    normalize : bool
        Fit / apply MinMaxScaler ([0, 1]).
    """

    def __init__(self, features, labels,
                 scaler=None, standardize=False, normalize=False):
        n, w, f = features.shape
        flat = features.reshape(-1, f)

        if scaler is not None:
            flat = scaler.transform(flat)
            self.scaler = scaler
        elif standardize:
            sc = StandardScaler().fit(flat)
            flat = sc.transform(flat)
            self.scaler = sc
        elif normalize:
            sc = MinMaxScaler().fit(flat)
            flat = sc.transform(flat)
            self.scaler = sc
        else:
            self.scaler = None

        features = flat.reshape(n, w, f)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_class_weights(labels, num_classes):
    """Return inverse-frequency weights as a float32 tensor (for CrossEntropyLoss)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL  –  Temporal Gait Transformer
# ═══════════════════════════════════════════════════════════════════════════

class TGTModel(nn.Module):
    """
    Architecture (from paper):
      1.  Linear projection  →  d_model-dimensional embedding per timestep
      2.  + fixed sinusoidal positional encoding
      3.  Prepend learnable [CLS] token
      4.  N × Transformer encoder layers  (multi-head self-attention + FFN w/ GELU)
      5.  LayerNorm on [CLS] representation
      6.  Linear  →  num_classes logits
    """

    def __init__(self, num_features, window_size, num_classes=5,
                 d_model=128, num_heads=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.d_model = d_model

        self.input_proj = nn.Linear(num_features, d_model)

        pe = self._make_pe(window_size, d_model)
        self.register_buffer("pe", pe, persistent=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    # ── helpers ──────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _make_pe(length, d_model):
        pe = torch.zeros(length, d_model)
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (length, d_model)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x):
        """x: (B, window_size, num_features) → logits (B, num_classes)"""
        x = self.input_proj(x)                             # (B, W, d)
        x = x + self.pe.unsqueeze(0)                       # add pos enc
        cls = self.cls_token.expand(x.size(0), -1, -1)     # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                     # (B, W+1, d)
        x = self.encoder(x)                                # (B, W+1, d)
        cls_rep = self.cls_norm(x[:, 0, :])                # (B, d)
        return self.head(cls_rep)                           # (B, C)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING  /  EVALUATION  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    """Return (all_preds, all_labels) as numpy arrays."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train Acc", color="royalblue")
    ax2.plot(epochs, val_accs, label="Val Acc", color="crimson")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {save_path}")


def plot_confusion(model, loader, class_names, device, save_path,
                   title="Confusion Matrix"):
    preds, labels = predict_all(model, loader, device)
    present = sorted(set(labels.tolist()))
    names = [class_names[i] for i in present]
    cm = confusion_matrix(labels, preds, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {save_path}")

    report = classification_report(
        labels, preds, labels=present, target_names=names, zero_division=0
    )
    print(report)
    return report


def plot_per_class_accuracy(model, loader, class_names, device, save_path):
    preds, labels = predict_all(model, loader, device)
    present = sorted(set(labels.tolist()))
    names = [class_names[i] for i in present]
    accs = []
    for c in present:
        mask = labels == c
        accs.append(100.0 * (preds[mask] == c).sum() / mask.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color="steelblue", edgecolor="black")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{a:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {save_path}")
