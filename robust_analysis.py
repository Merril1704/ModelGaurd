import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

def main():
    print("="*70)
    print("ROBUST DIAGNOSTIC: Generation Artifacts vs. Model Dynamics")
    print("="*70)
    
    df = pd.read_csv("data/meta_dataset.csv")
    y = df["failure_mode"]
    X_all = df.drop(columns=["failure_mode"])
    
    # The Suspects: These metadata features might allow the classifier to cheat 
    # without ever looking at the actual training loss curves.
    metadata_cols = [c for c in ["num_params", "class_entropy", "dataset_source"] if c in X_all.columns]
    X_pure = X_all.drop(columns=metadata_cols) if metadata_cols else X_all.copy()
    
    print(f"Total features (Including potentially leaky metadata): {X_all.shape[1]}")
    print(f"Pure telemetry features (Loss curves & gradients only): {X_pure.shape[1]}\n")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. 5-Fold Cross Validation on All Features
    print("--- BASELINE: 5-Fold CV on ALL Features (Including Metadata) ---")
    rf_all_scores = cross_val_score(rf, StandardScaler().fit_transform(X_all), y, cv=cv, scoring='f1_macro')
    mlp_all_scores = cross_val_score(mlp, StandardScaler().fit_transform(X_all), y, cv=cv, scoring='f1_macro')
    print(f"Random Forest F1: {rf_all_scores.mean():.3f} (± {rf_all_scores.std():.3f})")
    print(f"MLP Classifier F1 : {mlp_all_scores.mean():.3f} (± {mlp_all_scores.std():.3f})\n")
    
    # 2. 5-Fold Cross Validation on Pure Telemetry (The Acid Test)
    # If the F1 score collapses here, the 95% was an illusion built on generated metadata.
    print("--- ACID TEST: 5-Fold CV on PURE LOG TELEMETRY (No Metadata) ---")
    rf_pure_scores = cross_val_score(rf, StandardScaler().fit_transform(X_pure), y, cv=cv, scoring='f1_macro')
    mlp_pure_scores = cross_val_score(mlp, StandardScaler().fit_transform(X_pure), y, cv=cv, scoring='f1_macro')
    print(f"Random Forest F1: {rf_pure_scores.mean():.3f} (± {rf_pure_scores.std():.3f})")
    print(f"MLP Classifier F1 : {mlp_pure_scores.mean():.3f} (± {mlp_pure_scores.std():.3f})\n")
    
    # 3. Permutation Importance (True dependency testing)
    print("--- Permutation Importance on All Features (What is the model relying on?) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    rf.fit(X_scaled, y)
    
    # Permutation importance randomly shuffles one column at a time and measures accuracy drop.
    result = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42)
    
    importances = pd.DataFrame({
        'feature': X_all.columns,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
    
    print(importances.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
