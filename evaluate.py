"""
evaluate.py — Stage 3 of ModelGuard
Generate evaluation artifacts: confusion matrices, ROC-AUC, feature importance chart, scatter plot.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FAILURE_NAMES   = {0: "OVERFIT", 1: "CLASS_IMBALANCE", 2: "LABEL_NOISE", 3: "HEALTHY"}
TARGET_NAMES    = [FAILURE_NAMES[i] for i in sorted(FAILURE_NAMES)]
CLASSES         = sorted(FAILURE_NAMES.keys())

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_split():
    path = os.path.join("data", "test_split.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test split not found at '{path}'. "
            "Run train_meta_classifier.py first."
        )
    df      = pd.read_csv(path)
    y_test  = df["failure_mode"].values
    X_test  = df.drop(columns=["failure_mode"]).values
    feature_names = [c for c in df.columns if c != "failure_mode"]
    return X_test, y_test, feature_names


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm  = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
        ax=ax, linewidths=0.5
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved  →  {save_path}")


def compute_roc_auc(y_true, y_prob):
    y_bin = label_binarize(y_true, classes=CLASSES)
    return roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")


def plot_feature_importance(feature_names, importances, save_path):
    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_df)))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1])
    ax.set_title("Top-15 Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature",    fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved  →  {save_path}")


def plot_scatter(df_full, save_path):
    """overfit_score vs loss_gap colored by failure_mode."""
    palette = {0: "#e63946", 1: "#f4a261", 2: "#2a9d8f", 3: "#457b9d"}
    labels  = {0: "OVERFIT", 1: "CLASS_IMBALANCE", 2: "LABEL_NOISE", 3: "HEALTHY"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in CLASSES:
        mask = df_full["failure_mode"] == cls
        ax.scatter(
            df_full.loc[mask, "loss_gap"],
            df_full.loc[mask, "overfit_score"],
            label=labels[cls],
            color=palette[cls],
            alpha=0.75, edgecolors="white", linewidths=0.4, s=70
        )
    ax.set_title("Overfit Score vs Loss Gap by Failure Mode",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Loss Gap (val_loss − train_loss)", fontsize=12)
    ax.set_ylabel("Overfit Score (max(0, train_acc − val_acc))", fontsize=12)
    ax.legend(title="Failure Mode", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved  →  {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ModelGuard — Stage 3: Evaluation")
    print("=" * 60)

    # Load test split
    X_test, y_test, feature_names = load_test_split()
    print(f"\nTest set: {len(X_test)} samples")

    # Load scaler + models
    scaler    = joblib.load(os.path.join("models", "scaler.joblib"))
    rf_model  = joblib.load(os.path.join("models", "rf_model.joblib"))
    mlp_model = joblib.load(os.path.join("models", "mlp_model.joblib"))

    X_test_s = scaler.transform(X_test)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n── Random Forest ──────────────────────────────────────────")
    y_pred_rf  = rf_model.predict(X_test_s)
    y_prob_rf  = rf_model.predict_proba(X_test_s)

    print(classification_report(y_test, y_pred_rf, target_names=TARGET_NAMES))
    roc_rf = compute_roc_auc(y_test, y_prob_rf)
    print(f"ROC-AUC (macro OvR):  {roc_rf:.4f}")

    plot_confusion_matrix(
        y_test, y_pred_rf,
        title="Random Forest — Confusion Matrix",
        save_path=os.path.join("results", "confusion_matrix_rf.png")
    )

    # ── MLP Classifier ────────────────────────────────────────────────────────
    print("\n── MLP Classifier ─────────────────────────────────────────")
    y_pred_mlp = mlp_model.predict(X_test_s)
    y_prob_mlp = mlp_model.predict_proba(X_test_s)

    print(classification_report(y_test, y_pred_mlp, target_names=TARGET_NAMES))
    roc_mlp = compute_roc_auc(y_test, y_prob_mlp)
    print(f"ROC-AUC (macro OvR):  {roc_mlp:.4f}")

    plot_confusion_matrix(
        y_test, y_pred_mlp,
        title="MLP Classifier — Confusion Matrix",
        save_path=os.path.join("results", "confusion_matrix_mlp.png")
    )

    # ── Feature importance chart ───────────────────────────────────────────────
    print("\n── Feature Importance ─────────────────────────────────────")
    plot_feature_importance(
        feature_names, rf_model.feature_importances_,
        save_path=os.path.join("results", "feature_importance.png")
    )

    # ── Scatter plot ───────────────────────────────────────────────────────────
    print("\n── Scatter Plot ───────────────────────────────────────────")
    df_full = pd.read_csv(os.path.join("data", "meta_dataset.csv"))
    plot_scatter(
        df_full,
        save_path=os.path.join("results", "scatter.png")
    )

    print("\n✅  Stage 3 complete. All plots saved to results/")


if __name__ == "__main__":
    main()
