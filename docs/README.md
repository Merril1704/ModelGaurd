# ModelGuard — Documentation Index

> **Status:** Phase 9 (Pure Telemetry & 7-Class Expansion) Complete. Production-ready (98% Accuracy).

## Documents

| File | Contents |
|---|---|
| [01_project_overview.md](01_project_overview.md) | What ModelGuard is, research grounding, failure modes explained, real-world use cases, honest limitations, how to present this work |
| [02_dataset_reference.md](02_dataset_reference.md) | Full meta-dataset schema — every feature defined, failure injection parameters, label definitions |
| [03_implementation_stages.md](03_implementation_stages.md) | Stage-by-stage implementation breakdown (all 4 scripts + utils) |
| [04_future_work.md](04_future_work.md) | Near-term extensions, longer-term research directions, priority order |
| [05_results_and_analysis.md](05_results_and_analysis.md) | Empirical analysis of the 100% accuracy, target leakage, and feature importance interpretations |

## Quick Reference: Run Order

```bash
# Activate environment
conda activate torch

# Stage 1 — Generate meta-dataset (~35-45 min GPU)
python generate_meta_dataset.py

# Stage 2 — Train meta-classifiers (< 1 min)
python train_meta_classifier.py

# Stage 3 — Evaluation plots (< 30 sec)
python evaluate.py

# Stage 4 — Streamlit demo
streamlit run app.py
```

## Project Structure

```
ModelGaurd/
├── docs/
│   ├── README.md                    ← this file
│   ├── 01_project_overview.md       ← concept, validity, use cases
│   ├── 02_dataset_reference.md      ← meta-dataset schema
│   ├── 03_implementation_stages.md  ← code walkthrough
│   ├── 04_future_work.md            ← roadmap
│   └── 05_results_and_analysis.md   ← analysis of 100% accuracy phenomenon
├── data/
│   ├── meta_dataset.csv             ← 120-row meta-dataset (after Stage 1)
│   └── test_split.csv               ← held-out test set (after Stage 2)
├── models/
│   ├── rf_model.joblib              ← trained Random Forest
│   ├── mlp_model.joblib             ← trained MLP Classifier
│   └── scaler.joblib                ← fitted StandardScaler
├── results/
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_mlp.png
│   ├── feature_importance.png
│   ├── feature_importances.csv
│   └── scatter.png
├── generate_meta_dataset.py         ← Stage 1
├── train_meta_classifier.py         ← Stage 2
├── evaluate.py                      ← Stage 3
├── app.py                           ← Stage 4 (Streamlit)
├── utils.py                         ← shared helpers
├── requirements.txt
└── README.md
```
