"""
generate_meta_dataset.py — Stage 1 of ModelGuard
Train 120 small models with injected failure modes and extract training-dynamic features.
Output: data/meta_dataset.csv
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import extract_features, compute_gradient_norm

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
RAW_DIR = "./raw"

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY  = DEVICE.type == "cuda"   # faster host→device transfers
NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0
BATCH_SIZE  = 256 if DEVICE.type == "cuda" else 64

print(f"[Device] Using: {DEVICE}" + (
    f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""
))

# ── Failure-mode constants ────────────────────────────────────────────────────
FAILURE_MODES = {
    0: "OVERFIT",
    1: "CLASS_IMBALANCE",
    2: "LABEL_NOISE",
    3: "HEALTHY",
    4: "VANISHING_GRADIENT",
    5: "CATASTROPHIC_FORGETTING",
    6: "DATA_DRIFT",
}
MODELS_PER_CLASS = 250   # 7 × 250 = 1750 total (Overnight Run)


# ═══════════════════════════════════════════════════════════════════════════════
# MLP architecture
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int,
                 dropout: float = 0.0, activation='relu'):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

        if activation == 'sigmoid':
            # tiny weight initialization for vanishing gradient
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_sklearn_dataset():
    """make_classification — 1000 samples, 10 features, 5 classes."""
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        n_classes=5, n_clusters_per_class=1, random_state=SEED
    )
    X = X.astype(np.float32)
    return X, y, 10, 5


def load_fashion_mnist():
    """FashionMNIST — 5000 samples, flattened 784 features, 10 classes."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten
    ])
    ds = torchvision.datasets.FashionMNIST(
        root=RAW_DIR, train=True, download=True, transform=transform
    )
    indices = list(range(5000))
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=5000, shuffle=False, num_workers=0)
    X_t, y_t = next(iter(loader))
    return X_t.numpy(), y_t.numpy(), 784, 10


def load_cifar10():
    """CIFAR-10 — 5000 samples, flattened 3072 features, 10 classes."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten
    ])
    ds = torchvision.datasets.CIFAR10(
        root=RAW_DIR, train=True, download=True, transform=transform
    )
    indices = list(range(5000))
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=5000, shuffle=False, num_workers=0)
    X_t, y_t = next(iter(loader))
    return X_t.numpy(), y_t.numpy(), 3072, 10


DATASET_LOADERS = [load_sklearn_dataset, load_fashion_mnist, load_cifar10]


# ═══════════════════════════════════════════════════════════════════════════════
# Failure-mode data transformations
# ═══════════════════════════════════════════════════════════════════════════════

def apply_class_imbalance(X, y, num_classes, imbalance_ratio=0.90):
    """Keep imbalance_ratio from 2 dominant classes, distribute the rest among the rest."""
    dominant = np.random.choice(num_classes, size=2, replace=False)

    dom_mask  = np.isin(y, dominant)
    other_mask = ~dom_mask

    dom_X,   dom_y   = X[dom_mask],   y[dom_mask]
    other_X, other_y = X[other_mask], y[other_mask]

    n_total    = len(X)
    n_dominant = int(n_total * imbalance_ratio)
    n_other    = n_total - n_dominant

    dom_idx   = np.random.choice(len(dom_X),   size=min(n_dominant, len(dom_X)),   replace=False)
    other_idx = np.random.choice(len(other_X), size=min(n_other,    len(other_X)), replace=False)

    new_X = np.concatenate([dom_X[dom_idx],   other_X[other_idx]], axis=0)
    new_y = np.concatenate([dom_y[dom_idx],   other_y[other_idx]], axis=0)

    perm = np.random.permutation(len(new_X))
    return new_X[perm], new_y[perm]


def apply_label_noise(y, num_classes, noise_rate=0.30):
    """Randomly flip noise_rate fraction of labels to a random wrong class."""
    y_noisy = y.copy()
    n_noisy = int(len(y) * noise_rate)
    noisy_idx = np.random.choice(len(y), size=n_noisy, replace=False)
    for idx in noisy_idx:
        wrong = np.random.choice([c for c in range(num_classes) if c != y[idx]])
        y_noisy[idx] = wrong
    return y_noisy


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def make_loaders(X, y, val_split=0.2):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_split, random_state=SEED, stratify=None
    )
    def to_dl(Xa, ya, shuffle):
        ds = TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(ya, dtype=torch.long)
        )
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            persistent_workers=(NUM_WORKERS > 0),
        )
    return to_dl(X_tr, y_tr, True), to_dl(X_val, y_val, False), y_tr


def train_one_model(model, train_loader, val_loader, n_epochs, failure_class=None, train_loader2=None):
    """Train model, return epoch logs and per-epoch gradient norms."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logs = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    grad_norms = []
    
    current_train_loader = train_loader

    for epoch_idx in range(n_epochs):
        if failure_class == 5 and epoch_idx == n_epochs // 2 and train_loader2 is not None:
            current_train_loader = train_loader2

        # ── Train ──
        model.train()
        tl, ta, n = 0.0, 0.0, 0
        epoch_grad_norms = []
        for xb, yb in current_train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            epoch_grad_norms.append(compute_gradient_norm(model))
            optimizer.step()
            tl += loss.item() * len(xb)
            ta += (out.argmax(1) == yb).sum().item()
            n  += len(xb)
        logs["train_loss"].append(tl / n)
        logs["train_acc"].append(ta / n)
        grad_norms.append(float(np.mean(epoch_grad_norms)))

        # ── Validate ──
        model.eval()
        vl, va, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                
                if failure_class == 6: # DATA_DRIFT
                    noise_level = (epoch_idx / n_epochs) * 2.0  # Scale appropriately
                    xb = xb + torch.randn_like(xb) * noise_level

                out  = model(xb)
                loss = criterion(out, yb)
                vl += loss.item() * len(xb)
                va += (out.argmax(1) == yb).sum().item()
                nv += len(xb)
        logs["val_loss"].append(vl / nv)
        logs["val_acc"].append(va / nv)

    return logs, grad_norms


# ═══════════════════════════════════════════════════════════════════════════════
# Per-failure-mode model factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_experiment(failure_class: int, input_size: int, num_classes: int,
                     X: np.ndarray, y: np.ndarray):
    """
    Returns (model, train_loader, val_loader, y_train, n_epochs)
    after applying the relevant failure injection.
    """
    X_work = X.copy().astype(np.float32)
    y_work = y.copy()

    if failure_class == 0:  # OVERFIT
        num_layers = random.randint(3, 5)
        hidden = [random.choice([256, 512, 1024]) for _ in range(num_layers)]
        dropout = random.uniform(0.0, 0.1)
        n_epochs = random.randint(40, 60)
        
        subset_idx = np.random.choice(len(X_work), size=max(100, int(len(X_work)*0.20)), replace=False)
        X_work, y_work = X_work[subset_idx], y_work[subset_idx]

    elif failure_class == 1:  # CLASS_IMBALANCE
        imb_ratio = random.uniform(0.70, 0.95)
        X_work, y_work = apply_class_imbalance(X_work, y_work, num_classes, imbalance_ratio=imb_ratio)
        
        num_layers = random.randint(1, 3)
        hidden = [random.choice([32, 64, 128]) for _ in range(num_layers)]
        dropout = random.uniform(0.0, 0.2)
        n_epochs = random.randint(20, 40)

    elif failure_class == 2:  # LABEL_NOISE
        noise = random.uniform(0.15, 0.45)
        y_work = apply_label_noise(y_work, num_classes, noise_rate=noise)
        
        num_layers = random.randint(1, 3)
        hidden = [random.choice([32, 64, 128]) for _ in range(num_layers)]
        dropout = random.uniform(0.0, 0.2)
        n_epochs = random.randint(20, 40)

    elif failure_class == 4:  # VANISHING_GRADIENT
        num_layers = random.randint(6, 8)
        hidden = [random.choice([64, 128]) for _ in range(num_layers)]
        dropout = 0.0
        n_epochs = random.randint(30, 50)
        model = MLP(input_size, hidden, num_classes, dropout, activation='sigmoid').to(DEVICE)
        train_loader, val_loader, y_train = make_loaders(X_work, y_work)
        return model, train_loader, val_loader, y_train, n_epochs, None

    elif failure_class == 5:  # CATASTROPHIC_FORGETTING
        num_layers = random.randint(2, 3)
        hidden = [random.choice([64, 128]) for _ in range(num_layers)]
        dropout = random.uniform(0.0, 0.2)
        n_epochs = random.randint(30, 50)
        
        # Split into two disjoint sets of classes if possible.
        half = num_classes // 2
        mask1 = y_work < half
        mask2 = y_work >= half
        
        # If we couldn't split (e.g. classes too few), fallback to random subset
        if np.sum(mask1) < 10 or np.sum(mask2) < 10:
            mask1 = np.random.rand(len(y_work)) > 0.5
            mask2 = ~mask1

        train_loader, val_loader, y_train = make_loaders(X_work[mask1], y_work[mask1])
        train_loader2, _, _ = make_loaders(X_work[mask2], y_work[mask2])
        model = MLP(input_size, hidden, num_classes, dropout).to(DEVICE)
        return model, train_loader, val_loader, y_train, n_epochs, train_loader2

    elif failure_class == 6:  # DATA_DRIFT
        num_layers = random.randint(2, 3)
        hidden = [random.choice([64, 128, 256]) for _ in range(num_layers)]
        dropout = random.uniform(0.2, 0.4)
        n_epochs = random.randint(30, 50)
        model = MLP(input_size, hidden, num_classes, dropout).to(DEVICE)
        train_loader, val_loader, y_train = make_loaders(X_work, y_work)
        return model, train_loader, val_loader, y_train, n_epochs, None

    else:  # HEALTHY
        num_layers = random.randint(2, 3)
        hidden = [random.choice([64, 128, 256]) for _ in range(num_layers)]
        dropout = random.uniform(0.2, 0.5)
        n_epochs = random.randint(25, 45)

    model = MLP(input_size, hidden, num_classes, dropout).to(DEVICE)
    train_loader, val_loader, y_train = make_loaders(X_work, y_work)
    return model, train_loader, val_loader, y_train, n_epochs, None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ModelGuard — Stage 1: Generating Meta-Dataset")
    print("=" * 60)

    # Pre-load all datasets (downloads happen once)
    print("\n[1/2] Loading datasets (FashionMNIST and CIFAR-10 will auto-download)...")
    datasets = []
    for i, loader_fn in enumerate(DATASET_LOADERS):
        print(f"  Loading dataset {i} ...", end=" ", flush=True)
        X, y, inp_sz, n_cls = loader_fn()
        datasets.append((X, y, inp_sz, n_cls))
        print(f"done  ->  shape={X.shape}, classes={n_cls}")

    # Re-encode labels to 0..N-1 (sklearn dataset already OK; torchvision too)
    datasets_enc = []
    for X, y, inp_sz, n_cls in datasets:
        le = LabelEncoder()
        y   = le.fit_transform(y)
        datasets_enc.append((X, y, inp_sz, n_cls))

    # Build experiment list: 30 × 7 failure modes, cycling over 3 datasets
    experiments = []
    for fc in range(7):
        for run_idx in range(MODELS_PER_CLASS):
            ds_idx = run_idx % len(datasets_enc)
            experiments.append((fc, ds_idx))

    print(f"\n[2/2] Training {len(experiments)} models...\n")

    rows = []
    for fc, ds_idx in tqdm(experiments, desc="Training models", unit="model"):
        X, y, inp_sz, n_cls = datasets_enc[ds_idx]

        model, tr_loader, val_loader, y_train, n_epochs, tr_loader2 = build_experiment(
            fc, inp_sz, n_cls, X, y
        )

        logs, grad_norms = train_one_model(
            model, tr_loader, val_loader, n_epochs, failure_class=fc, train_loader2=tr_loader2
        )

        # Extract curve-based features
        feats = extract_features(logs)

        # Gradient norm features
        feats["gradient_norm_mean"] = float(np.mean(grad_norms))
        feats["gradient_norm_std"]  = float(np.std(grad_norms))

        # failure mode label
        feats["failure_mode"]   = fc


        rows.append(feats)

    df = pd.DataFrame(rows)
    out_path = os.path.join("data", "meta_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[DONE] Meta-dataset saved  ->  {out_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print(df["failure_mode"].value_counts().sort_index().rename(FAILURE_MODES))


if __name__ == "__main__":
    main()
