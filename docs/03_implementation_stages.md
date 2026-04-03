# ModelGuard — Implementation Stages

This document describes every stage of the ModelGuard pipeline as currently implemented,
including what each script does, how it works internally, and what outputs it produces.

---

## Stage 1 — `generate_meta_dataset.py`

**Goal:** Train 120 small MLPs with deliberately injected failure conditions.
Extract 20 scalar features per run. Save a meta-dataset CSV.

### How it works

```
3 Real Datasets  ──→  Failure Injection  ──→  Train MLP (GPU)  ──→  Extract Features  ──→  meta_dataset.csv
(sklearn/FMNIST/  (overfit / imbalance /    (per-epoch logs)    (20 scalars per run)
 CIFAR-10)         noise / healthy)
```

### Step-by-step flow

1. **Load datasets** — sklearn `make_classification` (auto), FashionMNIST and CIFAR-10
   (auto-download to `./raw/` via torchvision on first run).

2. **Build experiment list** — 30 runs per failure class × 4 classes = 120 experiments.
   Each run cycles through the 3 datasets (run%3 determines which dataset).

3. **Apply failure injection** per class:
   - `OVERFIT`: deep wide MLP [512×4], no dropout, 50 epochs
   - `CLASS_IMBALANCE`: skew data to 90/10 split between 2 dominant vs rest
   - `LABEL_NOISE`: flip 30% of labels to a random wrong class
   - `HEALTHY`: balanced data, dropout=0.3, [128,128] MLP, 30 epochs

4. **Train on GPU** (auto-detected via `torch.cuda.is_available()`).
   Records `train_loss`, `val_loss`, `train_acc`, `val_acc` at every epoch.
   Also captures per-batch gradient norms.

5. **Extract features** via `utils.extract_features()` — computes all 20 scalars
   from the recorded epoch curves.

6. **Save** all rows to `data/meta_dataset.csv`.

### Key implementation details

- Device: CUDA auto-detected. Falls back to CPU gracefully.
- Batch size: 256 on GPU, 64 on CPU.
- DataLoaders: `pin_memory=True`, `num_workers=2`, `persistent_workers=True` on GPU.
- All randomness seeded: `random`, `numpy`, `torch`, `torch.cuda`.
- tqdm progress bar: `Training models: 43%|████ | 52/120 [12:30<16:20, 14.4s/model]`
- Edge case: if a training curve has fewer than 10 epochs, trend/slope uses full curve.

### Outputs

```
data/meta_dataset.csv       ← 120 rows × 21 columns (20 features + label)
raw/FashionMNIST/           ← auto-downloaded dataset
raw/cifar-10-batches-py/    ← auto-downloaded dataset
```

### Runtime

- GPU (RTX 4050): ~35–45 minutes
- CPU only: ~3–4 hours

---

## Stage 2 — `train_meta_classifier.py`

**Goal:** Train a meta-classifier on the 120-row meta-dataset to predict failure mode.

### How it works

```
meta_dataset.csv  ──→  80/20 Split  ──→  StandardScaler  ──→  Train RF + MLP  ──→  Save Models
                        (stratified)      (fit on train)       (sklearn)            (.joblib)
```

### Step-by-step flow

1. **Load** `data/meta_dataset.csv`.
2. **Split** 80% train / 20% test, stratified by failure class.
3. **Scale** features with `StandardScaler` (fit on train only, apply to both).
4. **Train two classifiers:**
   - `RandomForestClassifier(n_estimators=100, max_depth=10)`
   - `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)`
5. **Print classification reports** for both on the test set.
6. **Save feature importances** (from RF) to `results/feature_importances.csv`.
7. **Persist test split** to `data/test_split.csv` (used by Stage 3).

### Why two classifiers?

| Classifier | Strength |
|---|---|
| **Random Forest** | Interpretable feature importances; handles non-linear boundaries; robust to scale |
| **MLP (sklearn)** | Comparison baseline; learns feature interactions via hidden layers |

The RF is the primary model used in the Streamlit app because it provides
`predict_proba` natively and feature importances for interpretability.

### Outputs

```
models/rf_model.joblib          ← trained Random Forest
models/mlp_model.joblib         ← trained MLP Classifier
models/scaler.joblib            ← fitted StandardScaler
results/feature_importances.csv ← RF feature importance scores
data/test_split.csv             ← held-out test data for Stage 3
```

### Runtime

< 1 minute.

---

## Stage 3 — `evaluate.py`

**Goal:** Generate evaluation visualisations and metrics from the trained meta-classifiers.

### Outputs produced

| File | Description |
|---|---|
| `results/confusion_matrix_rf.png` | Confusion matrix for Random Forest predictions on test set |
| `results/confusion_matrix_mlp.png` | Confusion matrix for MLP predictions on test set |
| `results/feature_importance.png` | Horizontal bar chart of top-15 RF feature importances |
| `results/scatter.png` | Scatter: `overfit_score` vs `loss_gap` coloured by failure mode |

### Metrics computed (printed to console)

- Per-class precision, recall, F1 for both models
- Macro-averaged ROC-AUC (one-vs-rest) for both models

### Scatter plot rationale

The `overfit_score` vs `loss_gap` scatter is an important diagnostic:
- OVERFIT models cluster in the high `overfit_score` / high `loss_gap` quadrant
- HEALTHY models cluster near the origin
- IMBALANCE and NOISE occupy intermediate regions
This validates that the feature space is **geometrically separable** — a necessary
condition for the meta-classifier to work.

### Runtime

< 30 seconds.

---

## Stage 4 — `app.py` (Streamlit Demo)

**Goal:** A browser-based UI where any practitioner can upload their own training log
CSV and get an instant failure-mode diagnosis.

### Application layout

```
┌─────────────────────────────────────────────────────────┐
│  Sidebar: Manual Feature Overrides                      │
│  (gradient_norm_mean, gradient_norm_std,                │
│   class_entropy, num_params, dataset_source)            │
├─────────────────────────────────────────────────────────┤
│  Hero: 🛡️ ModelGuard — Pre-Deployment Failure Predictor │
├─────────────────────────────────────────────────────────┤
│  Upload Card: CSV file upload + Sample CSV download     │
├─────────────────────────────────────────────────────────┤
│  Prediction Result Card:                                │
│  [badge: OVERFIT / IMBALANCE / NOISE / HEALTHY]         │
│  Confidence % + colour bar                              │
│  Plain-English explanation + suggested fixes            │
├─────────────────────────────────────────────────────────┤
│  Class Probabilities Bar Chart (all 4 classes)          │
├─────────────────────────────────────────────────────────┤
│  Training Curve Plots (loss + accuracy)                 │
├─────────────────────────────────────────────────────────┤
│  [Expandable] Feature vector JSON viewer                │
└─────────────────────────────────────────────────────────┘
```

### Input format

CSV with columns: `epoch, train_loss, val_loss, train_acc, val_acc`

### Feature computation in the app

The app computes all curve-based features (15 scalars) from the uploaded CSV.
The remaining 5 features (gradient norms, class entropy, dataset source, num params)
are either set to 0.0 (safe default) or provided manually via the sidebar sliders.

### Design

- Dark glassmorphism theme with gradient background
- Google Inter font via CDN
- Colour-coded prediction badges per failure class
- Animated confidence fill-bar
- Warning banner if models have not been trained yet

### Launch command

```bash
& "C:\Users\Merril\miniconda3\envs\torch\python.exe" -m streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Shared Utilities — `utils.py`

Contains four functions used across all stages:

| Function | Used in | Description |
|---|---|---|
| `extract_features(epoch_logs)` | Stage 1, Stage 4 | Computes all 15 curve-based scalars |
| `compute_gradient_norm(model)` | Stage 1 | Mean L2 gradient norm across all parameters (called after `loss.backward()`) |
| `get_class_entropy(labels)` | Stage 1 | Shannon entropy of training label distribution (scipy) |
| `plot_curves(...)` | Stage 4 | Returns a matplotlib Figure of loss + accuracy curves |

---

## Execution Order Summary

```bash
# Stage 1 — ~35-45 min on GPU
python generate_meta_dataset.py

# Stage 2 — < 1 min
python train_meta_classifier.py

# Stage 3 — < 30 sec
python evaluate.py

# Stage 4 — live app
streamlit run app.py
```

Each stage is **independently runnable** — it loads its inputs from disk
and fails gracefully with a clear error message if prerequisites are missing.
