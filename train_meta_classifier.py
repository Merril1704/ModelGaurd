"""
train_meta_classifier.py — Stage 2 of ModelGuard
Train a meta-classifier on data/meta_dataset.csv to predict failure_mode.
"""

import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FAILURE_NAMES = {
    0: "OVERFIT", 1: "CLASS_IMBALANCE", 2: "LABEL_NOISE", 3: "HEALTHY",
    4: "VANISHING_GRADIENT", 5: "CATASTROPHIC_FORGETTING", 6: "DATA_DRIFT"
}

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


def main():
    print("=" * 60)
    print("ModelGuard — Stage 2: Training Meta-Classifier")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    csv_path = os.path.join("data", "meta_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Meta-dataset not found at '{csv_path}'. "
            "Run generate_meta_dataset.py first."
        )

    df = pd.read_csv(csv_path)
    print(f"\nLoaded '{csv_path}'  ->  {len(df)} rows, {len(df.columns)} columns")
    print("Failure-mode distribution:")
    print(df["failure_mode"].value_counts().sort_index().rename(FAILURE_NAMES))

    # ── Features / labels ────────────────────────────────────────────────────
    y = df["failure_mode"].values
    X = df.drop(columns=["failure_mode"]).values

    feature_names = [c for c in df.columns if c != "failure_mode"]
    target_names  = [FAILURE_NAMES[i] for i in sorted(FAILURE_NAMES)]

    # ── Train / test split ───────────────────────────────────────────────────
    min_required = len(FAILURE_NAMES) / 0.20
    stratify_arg = y if len(X) >= min_required else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=stratify_arg
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join("models", "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved  ->  {scaler_path}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n[1/2] Training Random Forest ...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_path = os.path.join("models", "rf_model.joblib")
    joblib.dump(rf, rf_path)
    print(f"RF model saved  ->  {rf_path}")

    y_pred_rf = rf.predict(X_test_s)
    print("\n── Random Forest — Classification Report ──")
    print(classification_report(
        y_test, y_pred_rf,
        labels=range(len(target_names)),
        target_names=target_names,
        zero_division=0
    ))

    # ── MLP Classifier ────────────────────────────────────────────────────────
    print("[2/2] Training MLP Classifier ...", flush=True)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED
    )
    mlp.fit(X_train_s, y_train)
    mlp_path = os.path.join("models", "mlp_model.joblib")
    joblib.dump(mlp, mlp_path)
    print(f"MLP model saved  ->  {mlp_path}")

    y_pred_mlp = mlp.predict(X_test_s)
    print("\n── MLP Classifier — Classification Report ──")
    print(classification_report(
        y_test, y_pred_mlp,
        labels=range(len(target_names)),
        target_names=target_names,
        zero_division=0
    ))

    # ── Feature importances (from RF) ─────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature":   feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_path = os.path.join("results", "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"\nFeature importances saved  ->  {fi_path}")
    print(fi_df.head(10).to_string(index=False))

    # ── Also persist test split for evaluate.py ───────────────────────────────
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["failure_mode"] = y_test
    test_df_path = os.path.join("data", "test_split.csv")
    test_df.to_csv(test_df_path, index=False)
    print(f"\nTest split saved  ->  {test_df_path}  (used by evaluate.py)")

    print("\n[DONE] Stage 2 complete.")


if __name__ == "__main__":
    main()
