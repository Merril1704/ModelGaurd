"""
utils.py — Shared helpers for ModelGuard pipeline
"""

import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt

# Only force Agg backend when not already set (avoids overriding Streamlit's backend)
import os as _os
if _os.environ.get("STREAMLIT_SERVER_PORT") is None:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Feature extraction from training epoch logs
# ---------------------------------------------------------------------------

def extract_features(epoch_logs: dict) -> dict:
    """
    Compute all curve-based scalar features from recorded training dynamics.

    Parameters
    ----------
    epoch_logs : dict with keys:
        'train_loss', 'val_loss', 'train_acc', 'val_acc'
        Each value is a list of floats (one per epoch).

    Returns
    -------
    dict of scalar features (keys match meta-dataset column names)
    """
    train_loss = np.array(epoch_logs["train_loss"], dtype=float)
    val_loss   = np.array(epoch_logs["val_loss"],   dtype=float)
    train_acc  = np.array(epoch_logs["train_acc"],  dtype=float)
    val_acc    = np.array(epoch_logs["val_acc"],    dtype=float)

    n = len(train_loss)
    window = min(10, n)  # use full curve if < 10 epochs

    # Basic finals
    final_train_loss = float(train_loss[-1])
    final_val_loss   = float(val_loss[-1])
    final_train_acc  = float(train_acc[-1])
    final_val_acc    = float(val_acc[-1])

    # Gap metrics
    loss_gap    = final_val_loss - final_train_loss
    acc_gap     = final_train_acc - final_val_acc
    overfit_score = max(0.0, acc_gap)

    # Trend slopes over last `window` epochs
    x = np.arange(window)
    val_loss_trend   = float(np.polyfit(x, val_loss[-window:],   1)[0])
    train_loss_trend = float(np.polyfit(x, train_loss[-window:], 1)[0])

    # Volatility
    loss_volatility = float(np.std(val_loss))
    acc_volatility  = float(np.std(val_acc))

    # Best val acc / epoch
    best_val_acc   = float(np.max(val_acc))
    best_val_epoch = int(np.argmax(val_acc))

    # Convergence speed: first epoch train_loss < 0.5
    below = np.where(train_loss < 0.5)[0]
    convergence_speed = int(below[0]) if len(below) > 0 else -1

    # Pearson correlation between train_loss and val_loss curves
    if n > 1 and np.std(train_loss) > 1e-8 and np.std(val_loss) > 1e-8:
        corr, _ = stats.pearsonr(train_loss, val_loss)
        train_val_loss_corr = float(corr) if not np.isnan(corr) else 0.0
    else:
        train_val_loss_corr = 0.0

    return {
        "final_train_loss":     final_train_loss,
        "final_val_loss":       final_val_loss,
        "final_train_acc":      final_train_acc,
        "final_val_acc":        final_val_acc,
        "loss_gap":             loss_gap,
        "acc_gap":              acc_gap,
        "overfit_score":        overfit_score,
        "val_loss_trend":       val_loss_trend,
        "train_loss_trend":     train_loss_trend,
        "loss_volatility":      loss_volatility,
        "acc_volatility":       acc_volatility,
        "best_val_acc":         best_val_acc,
        "best_val_epoch":       best_val_epoch,
        "convergence_speed":    convergence_speed,
        "train_val_loss_corr":  train_val_loss_corr,
    }


# ---------------------------------------------------------------------------
# Gradient norm helper
# ---------------------------------------------------------------------------

def compute_gradient_norm(model) -> float:
    """
    Compute the mean L2 gradient norm across all parameters that have gradients.
    Call this after loss.backward() and before optimizer.step().
    """
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
            count += 1
    return total_norm / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Curve plotting (returns matplotlib Figure)
# ---------------------------------------------------------------------------

def plot_curves(train_loss, val_loss, train_acc, val_acc):
    """
    Plot training/validation loss and accuracy curves.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_loss) + 1)

    # Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss,   label="Val Loss",   linewidth=2, linestyle="--")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, label="Train Acc", linewidth=2)
    axes[1].plot(epochs, val_acc,   label="Val Acc",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig
