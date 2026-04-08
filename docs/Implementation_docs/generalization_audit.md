# ModelGuard — Generalization Audit

> **Critical Question:** Is the meta-classifier learning genuine training dynamics, or just the math of our injection functions?

---

## Test 1: Leave-One-Dataset-Out Cross-Validation

**Method:** Train on models from 2 datasets, test on the 3rd (completely unseen).

### Results (ALL 20 Features)

| Held-Out Dataset | RF F1 | MLP F1 | Train Size | Test Size |
|---|---|---|---|---|
| Sklearn Synthetic | 0.246 | 0.314 | 664 | 336 |
| FashionMNIST | 0.676 | 0.572 | 668 | 332 |
| CIFAR-10 | 0.100 | 0.355 | 668 | 332 |

### Results (PURE Telemetry — 17 Features, No Metadata)

| Held-Out Dataset | RF F1 | MLP F1 | Train Size | Test Size |
|---|---|---|---|---|
| Sklearn Synthetic | 0.649 | 0.659 | 664 | 336 |
| FashionMNIST | 0.850 | 0.737 | 668 | 332 |
| CIFAR-10 | 0.100 | 0.351 | 668 | 332 |

### Analysis

**This is a MAJOR finding.** The results split into three categories:

1. **FashionMNIST (held out) with pure telemetry: RF=0.85, MLP=0.74** — The classifier generalizes well here. Training dynamics learned from Sklearn+CIFAR-10 transfer to FashionMNIST. This proves the core hypothesis **partially works**.

2. **Sklearn Synthetic (held out): ~0.65** — Moderate generalization. The synthetic dataset's dynamics are somewhat different from image datasets, causing degradation. This is expected — structured tabular data trains differently than images.

3. **CIFAR-10 (held out): 0.10 (random chance for 4 classes = 0.25)** — **Complete collapse.** The classifier cannot detect failures in CIFAR-10 models when trained only on Sklearn+FashionMNIST. This is a red flag.

### Critical Insight: ALL features HURT cross-dataset generalization

With ALL 20 features (including metadata), performance **drops dramatically** compared to PURE telemetry:
- Sklearn: 0.25 (all) vs 0.65 (pure) — metadata actively hurts!
- FashionMNIST: 0.68 (all) vs 0.85 (pure) — metadata hurts!
- CIFAR-10: 0.10 (all) vs 0.10 (pure) — no help either way

**Why?** Because `dataset_source` and `num_params` are dataset-specific artifacts. When the classifier sees a new dataset, those features become misleading noise. The metadata features act as **cross-dataset poison**.

### Verdict

> **PARTIAL PASS.** The classifier learns transferable dynamics for 2 out of 3 datasets on pure telemetry. The CIFAR-10 collapse suggests that high-dimensional image data (3072 features flattened) produces fundamentally different training dynamics than lower-dimensional data. The classifier learns "what overfitting looks like on small/medium data" but not "what overfitting looks like on complex image data."

---

## Test 2: Feature Distribution Overlap (Symptom Ambiguity)

**Method:** For each feature, compute KS-test D-statistic between all 6 class pairs. High D = good separation. Low D = ambiguous/shared symptom.

### Feature Separability Table

| Feature | Avg KS | Max KS | Min KS | Diagnosis |
|---|---|---|---|---|
| `loss_gap` | 0.581 | **0.964** | 0.100 | **ONE-CLASS FINGERPRINT** |
| `num_params` | 0.577 | **0.944** | 0.056 | **ONE-CLASS FINGERPRINT** |
| `gradient_norm_std` | 0.643 | **0.936** | 0.268 | PARTIAL SIGNAL |
| `train_val_loss_corr` | 0.637 | **0.932** | 0.376 | PARTIAL SIGNAL |
| `class_entropy` | 0.583 | 0.664 | 0.240 | PARTIAL SIGNAL |
| `acc_gap` | 0.554 | **0.948** | 0.156 | PARTIAL SIGNAL |
| `overfit_score` | 0.554 | **0.948** | 0.156 | PARTIAL SIGNAL |
| `final_train_loss` | 0.574 | 0.668 | 0.280 | PARTIAL SIGNAL |
| `final_train_acc` | 0.556 | 0.668 | 0.256 | PARTIAL SIGNAL |
| `val_loss_trend` | 0.555 | 0.676 | 0.288 | PARTIAL SIGNAL |
| `final_val_loss` | 0.515 | 0.668 | 0.340 | PARTIAL SIGNAL |
| `final_val_acc` | 0.497 | 0.668 | 0.292 | PARTIAL SIGNAL |
| `best_val_acc` | 0.465 | 0.668 | 0.260 | PARTIAL SIGNAL |
| `gradient_norm_mean` | 0.453 | 0.568 | 0.296 | PARTIAL SIGNAL |
| `loss_volatility` | 0.440 | 0.656 | 0.164 | PARTIAL SIGNAL |
| `train_loss_trend` | 0.400 | 0.632 | 0.176 | PARTIAL SIGNAL |
| `convergence_speed` | 0.384 | 0.668 | 0.132 | **ONE-CLASS FINGERPRINT** |
| `best_val_epoch` | 0.318 | 0.444 | 0.212 | PARTIAL SIGNAL |
| `acc_volatility` | 0.324 | 0.412 | 0.224 | PARTIAL SIGNAL |
| `dataset_source` | 0.000 | 0.000 | 0.000 | USELESS (no separation) |

### Analysis

**Three features flagged as ONE-CLASS FINGERPRINTS:**

1. **`loss_gap`** — Max KS=0.964 but Min KS=0.100. This means it *perfectly* separates one class (likely OVERFIT, which has extreme loss gaps) from others, but provides zero signal between the remaining 3 classes. The classifier may be using a simple threshold: "if loss_gap > X → OVERFIT."

2. **`num_params`** — Max KS=0.944 but Min KS=0.056. Same pattern. It fingerprints OVERFIT (big models) but can't distinguish CLASS_IMBALANCE from HEALTHY from LABEL_NOISE. This confirms our injection creates a `num_params` → OVERFIT shortcut.

3. **`convergence_speed`** — Max KS=0.668 but Min KS=0.132. Likely fingerprints OVERFIT (which converges fast on tiny data, then diverges) while providing no signal for other classes.

**Key Symptom Overlaps:**

| Symptom | Shared Between | Risk |
|---|---|---|
| High `loss_gap` | OVERFIT only (no overlap, but... what about DATA_DRIFT in future?) | Low risk now, high risk when we add modes |
| Low `class_entropy` | CLASS_IMBALANCE exclusively (by construction) | **High risk** — our function DEFINES this |
| High `gradient_norm_std` | Partially shared across OVERFIT, LABEL_NOISE, HEALTHY | Good — this is a genuine multi-signal feature |
| High `train_val_loss_corr` | Partially shared | Good — genuine dynamic |
| `dataset_source` = 0 | No separation at all | Confirms it's useless (good) |

---

## Overall Verdict

### What's WORKING
1. **Pure telemetry features generalize** better than metadata — this proves the system CAN learn dynamics.
2. **Multi-signal features** like `gradient_norm_std` (avg KS=0.643) and `train_val_loss_corr` (avg KS=0.637) provide genuine, distributed signal across multiple class boundaries.
3. **`dataset_source` is confirmed useless** (KS=0.0) — it's not contributing to classification.

### What's CONCERNING
1. **CIFAR-10 collapse** — The classifier completely fails on CIFAR-10 when it hasn't trained on CIFAR-10 data. This means the dynamics of high-dimensional data are fundamentally different, and our 1000-model results are inflated because the classifier has seen all 3 datasets during training.
2. **`loss_gap` and `num_params` are ONE-CLASS FINGERPRINTS** — They perfectly identify OVERFIT but nothing else. The classifier might be using simple threshold rules rather than learning complex dynamic patterns.
3. **`class_entropy` is a CONSTRUCTED feature** — Low entropy is literally how we DEFINE class imbalance. The classifier detecting it is circular reasoning: "we broke the entropy → the classifier detects broken entropy."

### Recommendations

1. **Drop metadata features from production** — `num_params`, `class_entropy`, `dataset_source` should be removed. Pure telemetry actually generalizes BETTER cross-dataset.
2. **Add more diverse datasets** — Include MNIST, SVHN, tabular medical data, and text classification datasets to prevent dataset-specific overfitting.
3. **Validate on REAL failures** — Train a model that naturally overfits (not via our injection) and test if the classifier catches it. This is the ultimate acid test.
4. **Address CIFAR-10 gap** — Investigate why CIFAR-10 dynamics are so different. Likely because flattened 3072-dim input with a simple MLP creates fundamentally different training curves than lower-dim data.
5. **Consider removing `class_entropy` as a feature** — It's circular: we inject imbalance → entropy drops → classifier detects entropy drop. Instead, the classifier should detect imbalance via INDIRECT symptoms (poor minority-class accuracy, biased predictions, etc.)
