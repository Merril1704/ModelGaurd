======================================================================
ROBUST DIAGNOSTIC: Generation Artifacts vs. Model Dynamics
======================================================================
Total features (Including potentially leaky metadata): 20
Pure telemetry features (Loss curves & gradients only): 17

--- BASELINE: 5-Fold CV on ALL Features (Including Metadata) ---
Random Forest F1: 0.992 (▒ 0.005)
MLP Classifier F1 : 0.994 (▒ 0.007)

--- ACID TEST: 5-Fold CV on PURE LOG TELEMETRY (No Metadata) ---
Random Forest F1: 0.952 (▒ 0.016)
MLP Classifier F1 : 0.968 (▒ 0.011)

--- Permutation Importance on All Features (What is the model relying on?) ---
            feature  importance
      class_entropy      0.0371
  gradient_norm_std      0.0221
    loss_volatility      0.0208
   train_loss_trend      0.0022
     dataset_source      0.0014
         num_params      0.0013
           loss_gap      0.0009
train_val_loss_corr      0.0009
     final_val_loss      0.0005
            acc_gap      0.0002
