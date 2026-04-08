# ModelGuard — Future Work & Planned Extensions

This document outlines what has been completed, what is scoped for near-term
implementation, and what longer-term research directions exist for ModelGuard.

---

## Current Status (Completed)

| Component | Status | Notes |
|---|---|---|
| Meta-dataset generation (120 models, 4 failure classes) | ✅ Done | GPU-accelerated, seeded |
| Feature extraction (20 scalars per run) | ✅ Done | Gradient norms, curve trends, entropy |
| Meta-classifier training (RF + MLP) | ✅ Done | sklearn, saved with joblib |
| Evaluation plots (confusion matrix, ROC-AUC, scatter) | ✅ Done | Saved to results/ |
| Streamlit demo app | ✅ Done | Dark glassmorphism UI, CSV upload |
| Documentation | ✅ Done | This docs/ folder |

---

## Near-Term Extensions (Next Steps)

### 1. Scale the Meta-Dataset

**What:** Increase from 120 training runs to 500–1000.

**Why it matters:**
- The current 120-sample meta-dataset is small for training a robust meta-classifier.
- More runs = better generalisation of the meta-classifier to unseen model architectures.
- Include more epoch lengths (10, 20, 30, 50, 100) to capture more curve shapes.

**Implementation:**
- Add a `--n-runs` CLI argument to `generate_meta_dataset.py`
- Add more datasets: MNIST, SVHN, STL-10, custom tabular datasets
- Consider caching intermediate results so runs can be resumed if interrupted

---

### 2. Add More Failure Modes

**Current gap:** Only 4 failure classes. Real-world failures are more diverse.

**Proposed new classes:**

| New Class | How to inject | Real-world counterpart |
|---|---|---|
| `GRADIENT_EXPLOSION` | Remove batch norm, high LR (0.1+), deep network | Poor initialisation, missing grad clipping |
| `UNDERFITTING` | Tiny network [8, 8], very low LR, few epochs | Model too simple for task complexity |
| `DATA_LEAKAGE` | Val samples included in training set | Preprocessing bug leaking test data into train |
| `DISTRIBUTION_SHIFT` | Train/val from different class distributions | Production data differs from training data |
| `CATASTROPHIC_FORGETTING` | Sequential training on disjoint tasks without replay | Continual learning without memory mechanisms |

---

### 3. Support Real Model Architectures

**Current gap:** All 120 underlying models are fully-connected MLPs.

**Why it matters:**
- CNNs, Transformers, and RNNs produce different training curve signatures.
- A meta-classifier trained only on MLP runs may not generalise to ResNet training logs.

**Implementation:**
- Add a `model_type` configuration to `generate_meta_dataset.py`
- Implement a simple CNN (e.g. 3-conv + 2-FC) for vision datasets
- Add `model_type` as a feature in the meta-dataset (one-hot encoded)
- Retrain meta-classifier on mixed-architecture meta-dataset

---

### 4. Real-Run Validation (External Training Logs)

**What:** Validate that the meta-classifier predicts correctly on training logs
from real model runs (not synthetically generated).

**How:**
- Take a model from a Kaggle competition or HuggingFace that is known to overfit
- Export its per-epoch training log
- Feed it to ModelGuard → verify the prediction is `OVERFIT`
- Repeat for a known imbalanced dataset (e.g. Credit Card Fraud Detection)

**This is the key step to move from proof-of-concept to validated system.**

---

### 5. Compound Failure Detection

**Current gap:** Each model has exactly one injected failure. Real models often
fail for multiple simultaneous reasons.

**Example compound failures:**
- Overfitting *and* label noise simultaneously
- Imbalanced data *and* a too-simple model (underfitting the minority class)

**Implementation:**
- Change the label from single-class to multi-label
- Inject compound failures: e.g., 20% noise + deep wide net (overfit+noise)
- Switch meta-classifier to multi-label classification (RF with `MultiOutputClassifier`)

---

### 6. Time-Series Prediction (Early Warning)

**What:** Instead of predicting failure *after* training completes, predict it
*during* training — from just the first K epochs.

**Why it matters:**
- Current system requires the full training curve
- An early-warning system could stop a clearly-doomed training run at epoch 10
- Saves compute and enables real-time MLOps integration

**Implementation:**
- Extract features from only the first 10/20/30% of epochs
- Train meta-classifier on truncated curve features
- Evaluate: at what epoch can we reliably predict the final failure mode?
- Add a `--early-detection-at` flag to the app

---

### 7. MLflow / W&B Integration

**What:** Instead of uploading a CSV, automatically pull training logs from
MLflow or Weights & Biases runs.

**Implementation:**
- Add `mlflow` and `wandb` as optional dependencies
- Implement a `pull_from_mlflow(run_id)` function in `utils.py`
- Allow the Streamlit app to accept a W&B run URL directly
- The app fetches the epoch metrics via the API and runs prediction

---

### 8. Confidence Calibration

**Current gap:** The Random Forest `predict_proba` outputs are not calibrated —
a confidence of 90% may not actually mean the model is right 90% of the time.

**Implementation:**
- Apply `CalibratedClassifierCV` (sklearn) with Platt scaling or isotonic regression
- Plot reliability diagrams (calibration curves) in `evaluate.py`
- Save calibrated model instead of raw RF

---

### 9. REST API Wrapper

**What:** Expose the meta-classifier as an HTTP API endpoint so it can be called
from any CI/CD pipeline, MLflow hook, or model training framework.

**Implementation (FastAPI):**
```python
POST /predict
Content-Type: application/json

{
  "epoch_logs": {
    "train_loss": [1.5, 1.2, ...],
    "val_loss": [1.6, 1.4, ...],
    "train_acc": [0.3, 0.45, ...],
    "val_acc": [0.28, 0.40, ...]
  },
  "num_params": 50000,
  "class_entropy": 2.1
}

→ {"prediction": "OVERFIT", "confidence": 0.87, "probabilities": {...}}
```

---

### 10. Report Generation Module

**What:** Auto-generate a PDF/HTML report summarising a model's failure diagnosis,
curve plots, and recommended fixes.

**Implementation:**
- Use `reportlab` or `weasyprint` for PDF generation
- Template: failure badge → curve plots → feature table → recommendations
- Integrate into Streamlit as a "Download Report" button

---

## Long-Term Research Directions

| Direction | Description |
|---|---|
| **Attention over training curves** | Use a Transformer/LSTM to process raw per-epoch sequences directly instead of engineered scalar features |
| **Few-shot failure detection** | Meta-learn a model that can identify new failure modes from just 5–10 examples |
| **Cross-architecture transfer** | Train on MLP/CNN runs, evaluate on Transformer runs without retraining |
| **Causal failure attribution** | Go beyond classification — identify *which training decision* caused the failure (e.g., specific hyperparameter) |
| **Benchmark dataset** | Publish the meta-dataset as a public benchmark for the community |

---

## Priority Order (Recommended)

For a research paper or production demo, tackle in this order:

1. ✅ ~~Core pipeline (Stages 1–4)~~ — **Done**
2. 🔜 Real-run validation (external training logs)
3. 🔜 Scale meta-dataset to 500+ runs
4. 🔜 Add 2–3 more failure modes
5. 🔜 Early-warning / partial curve prediction
6. 🔜 Compound failure detection
7. 🔜 REST API wrapper
8. 🔜 MLflow/W&B integration
