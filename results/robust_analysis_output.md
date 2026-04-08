======================================================================
ROBUST DIAGNOSTIC: Generation Artifacts vs. Model Dynamics
======================================================================
Total features (Including potentially leaky metadata): 17
Pure telemetry features (Loss curves & gradients only): 17

--- BASELINE: 5-Fold CV on ALL Features (Including Metadata) ---
Random Forest F1: 0.946 (▒ 0.009)
MLP Classifier F1 : 0.941 (▒ 0.010)

--- ACID TEST: 5-Fold CV on PURE LOG TELEMETRY (No Metadata) ---
Random Forest F1: 0.946 (▒ 0.009)
MLP Classifier F1 : 0.941 (▒ 0.010)

--- Permutation Importance on All Features (What is the model relying on?) ---
            feature  importance
  gradient_norm_std    0.191886
train_val_loss_corr    0.063657
    loss_volatility    0.043600
 gradient_norm_mean    0.030686
     acc_volatility    0.024229
   train_loss_trend    0.023714
     best_val_epoch    0.007314
     final_val_loss    0.003714
           loss_gap    0.003657
       best_val_acc    0.003371
