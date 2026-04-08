"""
validate_generalization.py — ModelGuard Generalization Audit
============================================================
Goal: Determine if the meta-classifier is learning GENUINE training
dynamics vs. just memorizing the MATH of our injection functions.

Three Key Tests:
  1. Leave-One-Dataset-Out CV  — Can the classifier generalize across
     datasets it has NEVER seen during training?
  2. Feature Distribution Overlap — Are any features ambiguous symptoms
     that could indicate multiple failure modes?
  3. Misclassification Forensics — What do the WRONG predictions tell us
     about the classifier's decision boundaries?
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

FAILURE_NAMES = {0: "OVERFIT", 1: "CLASS_IMBALANCE", 2: "LABEL_NOISE", 3: "HEALTHY"}
TARGET_NAMES  = [FAILURE_NAMES[i] for i in sorted(FAILURE_NAMES)]
DATASET_NAMES = {0: "Sklearn_Synthetic", 1: "FashionMNIST", 2: "CIFAR-10"}

METADATA_COLS = ["num_params", "class_entropy", "dataset_source"]


def load_data():
    df = pd.read_csv("data/meta_dataset.csv")
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Leave-One-Dataset-Out Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: If we train on Sklearn+FashionMNIST and test on CIFAR-10, the classifier
# has NEVER seen CIFAR-10 training dynamics. If it still works, it's learning
# universal failure patterns. If it collapses, it's memorizing dataset-specific
# injection artifacts.

def test_leave_one_dataset_out(df):
    print("=" * 70)
    print("TEST 1: Leave-One-Dataset-Out Cross-Validation")
    print("=" * 70)
    print("Purpose: Can the classifier detect failures on a dataset")
    print("         it has NEVER seen during training?\n")

    y_col = "failure_mode"
    # Test with and without metadata
    feature_sets = {
        "ALL features (20)":  [c for c in df.columns if c != y_col],
        "PURE telemetry (17)": [c for c in df.columns if c != y_col and c not in METADATA_COLS],
    }

    results = []

    for fs_name, feature_cols in feature_sets.items():
        print(f"--- Feature Set: {fs_name} ---")
        for held_out_ds in range(3):
            train_mask = df["dataset_source"] != held_out_ds
            test_mask  = df["dataset_source"] == held_out_ds

            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, y_col].values
            X_test  = df.loc[test_mask, feature_cols].values
            y_test  = df.loc[test_mask, y_col].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            # RF
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train_s, y_train)
            rf_f1 = f1_score(y_test, rf.predict(X_test_s), average='macro')

            # MLP
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
            mlp.fit(X_train_s, y_train)
            mlp_f1 = f1_score(y_test, mlp.predict(X_test_s), average='macro')

            ds_name = DATASET_NAMES[held_out_ds]
            print(f"  Held-out: {ds_name:20s}  |  RF F1={rf_f1:.3f}  |  MLP F1={mlp_f1:.3f}  "
                  f"|  Train={train_mask.sum()}, Test={test_mask.sum()}")

            results.append({
                "feature_set": fs_name,
                "held_out": ds_name,
                "rf_f1": rf_f1,
                "mlp_f1": mlp_f1,
                "train_size": train_mask.sum(),
                "test_size": test_mask.sum()
            })
        print()

    results_df = pd.DataFrame(results)

    # Verdict
    avg_rf_all  = results_df[results_df["feature_set"].str.contains("ALL")]["rf_f1"].mean()
    avg_mlp_all = results_df[results_df["feature_set"].str.contains("ALL")]["mlp_f1"].mean()
    avg_rf_pure  = results_df[results_df["feature_set"].str.contains("PURE")]["rf_f1"].mean()
    avg_mlp_pure = results_df[results_df["feature_set"].str.contains("PURE")]["mlp_f1"].mean()

    print("VERDICT:")
    print(f"  Avg Leave-One-Out F1 (ALL features):  RF={avg_rf_all:.3f}, MLP={avg_mlp_all:.3f}")
    print(f"  Avg Leave-One-Out F1 (PURE telemetry): RF={avg_rf_pure:.3f}, MLP={avg_mlp_pure:.3f}")

    if avg_mlp_pure >= 0.85:
        print("  -> PASS: Classifier generalizes across unseen datasets on pure telemetry.")
        print("           This means it's learning dynamics, not dataset-specific artifacts.")
    elif avg_mlp_pure >= 0.70:
        print("  -> WARN: Partial generalization. Some injection patterns may be dataset-specific.")
    else:
        print("  -> FAIL: Classifier collapses on unseen datasets. It's memorizing injection math.")
    print()

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Feature Distribution Overlap Analysis
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: If `class_entropy` perfectly separates CLASS_IMBALANCE from everything
# else, the classifier doesn't need to learn dynamics — it just thresholds one
# feature. We need to find which features are GENUINELY discriminative vs. which
# are just artifacts of our injection function.
#
# METHOD: For each feature, compute pairwise KS-test p-values between classes.
# If a feature has p < 0.001 for ALL pairs involving one class, it's likely
# an injection fingerprint rather than a genuine dynamic.

def test_feature_overlap(df):
    print("=" * 70)
    print("TEST 2: Feature Distribution Overlap (Symptom Ambiguity)")
    print("=" * 70)
    print("Purpose: Which features are UNIQUE to one failure mode vs.")
    print("         shared across multiple modes (ambiguous symptoms)?\n")

    y = df["failure_mode"]
    feature_cols = [c for c in df.columns if c != "failure_mode"]

    # For each feature, compute separability per class pair
    print("--- Per-Feature Class Separability (KS-test D-statistic) ---\n")
    print(f"{'Feature':>25s}  | {'OVF-IMB':>8s} {'OVF-NOS':>8s} {'OVF-HLT':>8s} "
          f"{'IMB-NOS':>8s} {'IMB-HLT':>8s} {'NOS-HLT':>8s} | {'Diagnosis':>20s}")
    print("-" * 120)

    class_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    pair_names = ["OVF-IMB", "OVF-NOS", "OVF-HLT", "IMB-NOS", "IMB-HLT", "NOS-HLT"]

    feature_diagnostics = []

    for feat in feature_cols:
        d_stats = []
        for c1, c2 in class_pairs:
            vals_c1 = df.loc[y == c1, feat].values
            vals_c2 = df.loc[y == c2, feat].values
            ks_stat, _ = stats.ks_2samp(vals_c1, vals_c2)
            d_stats.append(ks_stat)

        # Diagnosis logic
        max_d = max(d_stats)
        min_d = min(d_stats)
        avg_d = np.mean(d_stats)

        if max_d < 0.15:
            diagnosis = "USELESS (no separation)"
        elif min_d > 0.50:
            diagnosis = "UNIVERSAL SEPARATOR"
        elif max_d > 0.60 and min_d < 0.15:
            diagnosis = "ONE-CLASS FINGERPRINT"
        else:
            diagnosis = "PARTIAL SIGNAL"

        # Check if it's an injection artifact
        # A feature is suspicious if it PERFECTLY separates exactly one class
        # from all others but shows no signal between the remaining classes
        high_pairs = sum(1 for d in d_stats if d > 0.40)
        low_pairs = sum(1 for d in d_stats if d < 0.10)

        if high_pairs >= 3 and low_pairs >= 2:
            diagnosis += " [INJECTION?]"

        d_str = "  ".join(f"{d:8.3f}" for d in d_stats)
        print(f"{feat:>25s}  | {d_str} | {diagnosis:>20s}")

        feature_diagnostics.append({
            "feature": feat,
            "avg_ks": avg_d,
            "max_ks": max_d,
            "min_ks": min_d,
            "diagnosis": diagnosis,
        })

    print()

    # Highlight the most suspicious features
    diag_df = pd.DataFrame(feature_diagnostics)
    suspicious = diag_df[diag_df["diagnosis"].str.contains("INJECTION")]
    if len(suspicious) > 0:
        print("WARNING: These features may be INJECTION FINGERPRINTS:")
        for _, row in suspicious.iterrows():
            print(f"  - {row['feature']}: avg KS={row['avg_ks']:.3f}, "
                  f"max KS={row['max_ks']:.3f}, min KS={row['min_ks']:.3f}")
        print("  These features perfectly separate one class but add nothing")
        print("  between other classes -> likely detecting OUR FORMULA, not real dynamics.\n")
    else:
        print("No features flagged as obvious injection fingerprints.\n")

    return diag_df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Misclassification Forensics
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: By studying WHAT the classifier gets wrong, we can understand what
# genuine ambiguity looks like and whether any misclassifications reveal
# symptom overlap.

def test_misclassification_forensics(df):
    print("=" * 70)
    print("TEST 3: Misclassification Forensics")
    print("=" * 70)
    print("Purpose: What do the WRONG predictions reveal about")
    print("         the classifier's blind spots?\n")

    from sklearn.model_selection import train_test_split

    y = df["failure_mode"].values
    feature_cols = [c for c in df.columns if c != "failure_mode"]
    X = df[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Use RF for interpretability
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)

    # Find misclassified samples
    wrong_mask = y_pred != y_test
    n_wrong = wrong_mask.sum()

    print(f"Total test samples: {len(y_test)}")
    print(f"Misclassified:      {n_wrong} ({100*n_wrong/len(y_test):.1f}%)\n")

    if n_wrong == 0:
        print("No misclassifications to analyze.\n")
        return

    # For each misclassification, show what happened
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["true_label"] = [FAILURE_NAMES[y] for y in y_test]
    test_df["predicted"]  = [FAILURE_NAMES[y] for y in y_pred]
    test_df["confidence"] = y_prob.max(axis=1)
    test_df["correct"]    = ~wrong_mask

    wrong_df = test_df[~test_df["correct"]].copy()

    print("--- Misclassification Breakdown ---")
    confusion_pairs = wrong_df.groupby(["true_label", "predicted"]).size().reset_index(name="count")
    confusion_pairs = confusion_pairs.sort_values("count", ascending=False)
    for _, row in confusion_pairs.iterrows():
        print(f"  {row['true_label']:>20s} -> predicted as {row['predicted']:<20s}  ({row['count']} cases)")
    print()

    # For each confusion pair, show the feature profile of misclassified samples
    print("--- Feature Profile of Misclassified Samples ---\n")
    key_features = ["loss_gap", "acc_gap", "overfit_score", "class_entropy",
                    "gradient_norm_std", "loss_volatility", "num_params",
                    "train_val_loss_corr", "final_val_loss", "val_loss_trend"]

    for _, row in confusion_pairs.iterrows():
        true_cls = row["true_label"]
        pred_cls = row["predicted"]
        mask = (wrong_df["true_label"] == true_cls) & (wrong_df["predicted"] == pred_cls)
        subset = wrong_df[mask]

        print(f"  {true_cls} misclassified as {pred_cls} ({len(subset)} samples):")

        # Compare misclassified samples to class centroids
        true_cls_id = [k for k, v in FAILURE_NAMES.items() if v == true_cls][0]
        pred_cls_id = [k for k, v in FAILURE_NAMES.items() if v == pred_cls][0]

        true_centroid = df[df["failure_mode"] == true_cls_id][key_features].mean()
        pred_centroid = df[df["failure_mode"] == pred_cls_id][key_features].mean()
        misclass_mean = subset[key_features].mean()

        print(f"  {'Feature':>25s}  {'Misclassified':>14s}  {'True Class Avg':>14s}  {'Pred Class Avg':>14s}  {'Closer To':>10s}")
        for feat in key_features:
            mc_val = misclass_mean[feat]
            true_val = true_centroid[feat]
            pred_val = pred_centroid[feat]
            dist_true = abs(mc_val - true_val)
            dist_pred = abs(mc_val - pred_val)
            closer = "TRUE" if dist_true < dist_pred else "PRED <--"
            print(f"  {feat:>25s}  {mc_val:14.4f}  {true_val:14.4f}  {pred_val:14.4f}  {closer:>10s}")
        print()

    # Low-confidence predictions (even correct ones)
    print("--- Low-Confidence Predictions (Confidence < 0.60) ---")
    low_conf = test_df[test_df["confidence"] < 0.60]
    if len(low_conf) == 0:
        print("  None. All predictions have confidence >= 0.60.\n")
    else:
        print(f"  {len(low_conf)} samples with confidence < 0.60:")
        for _, row in low_conf.iterrows():
            status = "CORRECT" if row["correct"] else "WRONG"
            print(f"    {row['true_label']:>20s} -> {row['predicted']:<20s}  "
                  f"conf={row['confidence']:.3f}  [{status}]")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Injection Formula Detection
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: If we can predict the failure mode using ONLY the metadata features
# (class_entropy, num_params, dataset_source) with high accuracy, then
# the classifier COULD be learning our injection formula, not dynamics.
# Conversely, if metadata alone fails, the classifier MUST be using dynamics.

def test_injection_formula_detection(df):
    print("=" * 70)
    print("TEST 4: Can Metadata ALONE Predict Failure Mode?")
    print("=" * 70)
    print("Purpose: If class_entropy + num_params alone achieve high F1,")
    print("         the classifier might just be detecting our injection")
    print("         formula, not genuine training dynamics.\n")

    from sklearn.model_selection import cross_val_score, StratifiedKFold

    y = df["failure_mode"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Feature sets to compare
    feature_sets = {
        "ONLY metadata (3)":         METADATA_COLS,
        "ONLY class_entropy (1)":    ["class_entropy"],
        "ONLY num_params (1)":       ["num_params"],
        "PURE telemetry (17)":       [c for c in df.columns if c != "failure_mode" and c not in METADATA_COLS],
        "ALL features (20)":         [c for c in df.columns if c != "failure_mode"],
    }

    print(f"{'Feature Set':>30s}  |  {'RF F1':>8s}  {'MLP F1':>8s}  |  {'Interpretation'}")
    print("-" * 100)

    for fs_name, cols in feature_sets.items():
        X = StandardScaler().fit_transform(df[cols])

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)

        rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_macro')
        mlp_scores = cross_val_score(mlp, X, y, cv=cv, scoring='f1_macro')

        rf_f1 = rf_scores.mean()
        mlp_f1 = mlp_scores.mean()

        # Interpretation
        if "metadata" in fs_name.lower() or "entropy" in fs_name.lower() or "num_params" in fs_name.lower():
            if max(rf_f1, mlp_f1) > 0.80:
                interp = "DANGER: Metadata alone is highly predictive!"
            elif max(rf_f1, mlp_f1) > 0.50:
                interp = "CAUTION: Metadata carries significant signal"
            else:
                interp = "SAFE: Metadata alone can't predict failure mode"
        else:
            interp = ""

        print(f"{fs_name:>30s}  |  {rf_f1:8.3f}  {mlp_f1:8.3f}  |  {interp}")

    print()
    print("INTERPRETATION GUIDE:")
    print("  If 'ONLY metadata' F1 is close to 'ALL features' F1:")
    print("    -> The classifier is mostly using injection artifacts, NOT dynamics.")
    print("  If 'PURE telemetry' F1 is close to 'ALL features' F1:")
    print("    -> The classifier genuinely relies on training dynamics.")
    print("  If 'ONLY class_entropy' F1 is high for CLASS_IMBALANCE:")
    print("    -> Our apply_class_imbalance() function leaves a fingerprint")
    print("       that a real-world imbalance detector wouldn't have.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  ModelGuard — Generalization Audit")
    print("  'Is the classifier learning DYNAMICS or INJECTION MATH?'")
    print("=" * 70 + "\n")

    df = load_data()

    results_lodo = test_leave_one_dataset_out(df)
    diag_df      = test_feature_overlap(df)
    test_injection_formula_detection(df)
    test_misclassification_forensics(df)

    # Save results
    os.makedirs("results", exist_ok=True)
    results_lodo.to_csv("results/leave_one_dataset_out.csv", index=False)
    diag_df.to_csv("results/feature_diagnostics.csv", index=False)

    print("=" * 70)
    print("  Audit complete. Results saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
