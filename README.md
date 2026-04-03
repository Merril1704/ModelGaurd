# ModelGuard 🛡️ — Pre-Deployment ML Failure Predictor

## Overview

ModelGuard is a proof-of-concept system that predicts ML model failure modes directly from training dynamics — *before* you deploy to production. By training 120 small neural networks with deliberately injected failure conditions (overfitting, class imbalance, label noise, healthy baseline), it extracts a rich set of scalar features from each model's training curves and trains a meta-classifier to detect which failure class a new model belongs to. A Streamlit demo lets you upload any training log CSV and get an instant failure-mode diagnosis with plain-English remediation advice.

## Project Structure

```
modelguard/
├── data/                        # saved meta-dataset CSV + test split
├── models/                      # trained meta-classifiers + scaler
├── results/                     # evaluation plots and feature importances
├── generate_meta_dataset.py     # Stage 1: train 120 models, save meta-dataset
├── train_meta_classifier.py     # Stage 2: train RF + MLP meta-learners
├── evaluate.py                  # Stage 3: confusion matrices, ROC-AUC, plots
├── app.py                       # Stage 4: Streamlit demo UI
├── utils.py                     # shared feature extraction helpers
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create / activate a conda environment (Python 3.9+)
conda activate torch

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA is supported but not required; the pipeline runs on CPU only.

## How to Run (in order)

### Stage 1 — Generate Meta-Dataset (~15–30 min on CPU)
```bash
python generate_meta_dataset.py
```
Trains 120 small MLPs across three datasets with four injected failure modes.
Saves `data/meta_dataset.csv`.

### Stage 2 — Train Meta-Classifier
```bash
python train_meta_classifier.py
```
Trains a Random Forest and an MLP classifier on the meta-dataset.
Saves models to `models/` and feature importances to `results/`.

### Stage 3 — Evaluate
```bash
python evaluate.py
```
Generates confusion matrices, feature importance bar chart, and scatter plots in `results/`.

### Stage 4 — Launch Streamlit App
```bash
streamlit run app.py
```
Opens the ModelGuard web UI in your browser at `http://localhost:8501`.

## How It Works

- **Failure injection** — 120 MLPs are trained across three datasets (tabular, FashionMNIST, CIFAR-10), each with one of four conditions: overfitting (deep network, no regularisation), class imbalance (90/10 label skew), label noise (30% random label flips), or healthy baseline.
- **Feature extraction** — After each training run, 20 scalar features are extracted from epoch curves: loss/accuracy gaps, trend slopes, volatility, convergence speed, gradient norm statistics, class entropy, and more.
- **Meta-classification** — A Random Forest and an MLP classifier are trained on these 20-feature vectors to predict which failure mode a model belongs to.
- **Evaluation** — Confusion matrices, macro ROC-AUC, per-class precision/recall/F1, feature importances, and diagnostic scatter plots are generated automatically.
- **Streamlit demo** — Users upload a training log CSV; the app computes features in real-time, runs the Random Forest prediction, displays a confidence score, shows training curves, and gives actionable remediation advice.

## Failure Mode Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | `OVERFIT` | Model memorises training data; large train/val accuracy gap |
| 1 | `CLASS_IMBALANCE` | Training labels are heavily skewed toward a few classes |
| 2 | `LABEL_NOISE` | 30% of training labels are randomly corrupted |
| 3 | `HEALTHY` | Balanced data, dropout regularisation, stable convergence |
