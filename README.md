# ModelGuard 🛡️ — Pre-Deployment ML Failure Predictor

## Overview

ModelGuard is a robust, pre-deployment diagnostic system that predicts machine learning model failure modes directly from training dynamics. By monitoring the telemetry of the training process—such as loss curves, accuracy trends, and gradient norms—ModelGuard can detect whether a model is likely to fail in production *before* it is deployed. 

The system leverages a meta-dataset of 1,750 small neural networks trained under seven deliberately injected conditions. Using features extracted strictly from training epoch logs, it trains a meta-classifier to identify complex failure modes with a high degree of accuracy. A Streamlit dashboard allows for rapid submission of any training log CSV to instantly receive failure-mode diagnosis and practical remediation advice.

## Project Structure

```text
modelguard/
├── data/                        # Contains the generated meta-dataset (meta_dataset.csv) and test splits
├── models/                      # Saved trained meta-classifiers (RF, MLP) and StandardScaler joblib
├── results/                     # Diagnostic outputs, generalization audits, graphs and metrics
├── docs/                        # Research reports and supplementary documentation
├── generate_meta_dataset.py     # Stage 1: Trains 1,750 models and builds the meta-dataset telemetry
├── train_meta_classifier.py     # Stage 2: Trains Random Forest and MLP classifiers to predict failures
├── evaluate.py                  # Stage 3: Produces confusion matrices, ROC curves, feature importances
├── robust_analysis.py           # Robustness verification and metadata leakage analysis (Acid Test)
├── validate_generalization.py   # Leave-one-dataset-out generalization tests across domain shifts
├── app.py                       # Stage 4: Interactive Streamlit UI for ad-hoc diagnostics
├── utils.py                     # Feature extraction functions for the pipelines
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Setup & Installation

```bash
# Create and activate a virtual environment (Python 3.9+)
conda create -n modelguard python=3.9
conda activate modelguard

# Install required dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA is utilized if available for faster training, but the entire pipeline—including the meta-classifiers—can still fall back to CPU.

## How It Works

1. **Failure Injection & Meta-Data Generation**: 1,750 Multi-Layer Perceptrons (MLPs) are trained across three distinct datasets (Sklearn Synthetic, FashionMNIST, CIFAR-10), each deliberately afflicted with one of 7 conditions:
   - Overfitting
   - Class Imbalance
   - Label Noise
   - Vanishing Gradient
   - Catastrophic Forgetting
   - Data Drift
   - Healthy Status (Baseline)
2. **Telemetry Extraction**: For each trained sub-model, ModelGuard extracts 20 critical scalar features from the training process (e.g., train-val loss gap, gradient norm statistics, loss volatility, convergence speed, class entropy).
3. **Meta-Classification**: Using Scikit-Learn, a Random Forest and an MLP classifier consume these dynamic "fingerprints" to predict the underlying failure condition. They achieve over 95% F1-score on pure log telemetry, proving they learn real training dynamics rather than memorizing dataset metadata.
4. **Generalization Auditing**: The framework uses leave-one-dataset-out evaluations (e.g., train on Synthetic/FashionMNIST, test on CIFAR-10) to verify cross-domain robustness.
5. **Streamlit UI**: Users interact with ModelGuard locally by dropping training log CSVs to receive instant diagnostics, confidence scores, plotted charts, and actionable remedies.

## Usage Pipeline

### Stage 1 — Generate the Meta-Dataset
```bash
python generate_meta_dataset.py
```
*Generates the 1,750 model telemetry entries mapping them to one of the 7 failure targets.*

### Stage 2 — Train the Meta-Classifier
```bash
python train_meta_classifier.py
```
*Trains and persists the Random Forest and MLP classification models (`models/rf_model.joblib`, `models/mlp_model.joblib`), saving out features' importance.*

### Stage 3 — Evaluate Performance
```bash
python evaluate.py
```
*Creates confusion matrices, scatter visualizations, and raw classification metrics.*

### Stage 4 — Robustness & Generalization (Optional but Recommended)
```bash
python robust_analysis.py
python validate_generalization.py
```
*Executes the acid test by stripping out metadata to verify the true diagnostic capability on loss curves and gradients, followed by a domain generalization test.*

### Stage 5 — Launch the Web Dashboard
```bash
streamlit run app.py
```
*Opens the GUI dashboard at `http://localhost:8501` to use the pre-trained ModelGuard.*

## The 7 Failure Modes

| Index | Name | Description |
|-------|------|-------------|
| 0 | `OVERFIT` | The model heavily memorizes training data, characterized by diverging train and validation losses. |
| 1 | `CLASS_IMBALANCE` | Training set has an artificially skewed label distribution. |
| 2 | `LABEL_NOISE` | Substantial random label corruption (e.g., 30%) introduced during training. |
| 3 | `HEALTHY` | The baseline—balanced data, standard regularization, expected convergence behavior. |
| 4 | `VANISHING_GRADIENT` | Diminished gradient flow resulting in stagnant, suboptimal training. |
| 5 | `CATASTROPHIC_FORGETTING` | Progressive loss of previously acquired pattern behavior during sequential/prolonged fitting. |
| 6 | `DATA_DRIFT` | Distributional shifts synthetically induced to mirror real-world non-stationarity. |
