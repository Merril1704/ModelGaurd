# ModelGuard — Project Overview & Conceptual Validity

## What is ModelGuard?

ModelGuard is a **proof-of-concept meta-learning system** that predicts ML model failure modes
directly from training dynamics — *before deployment to production*.

The core idea: instead of waiting until a deployed model performs badly in production,
ModelGuard analyzes the *training curves* of a model (loss over epochs, accuracy over
epochs, gradient norms) and predicts which failure class the model belongs to
with high confidence — at training time.

---

## The Research Concept: Learning About Models from Their Training Signals

### Why training curves are informative

Every experienced ML engineer reads TensorBoard or W&B dashboards because training
dynamics carry real diagnostic information:

- A **widening train/val accuracy gap** → model is memorising, not generalising
- A **rising validation loss** while training loss falls → classic overfitting signature
- **High loss volatility** with low final accuracy → label noise or conflicting gradients
- **Flat validation accuracy** from early on → model never learned from under-represented classes

These are not abstract intuitions — they are physical signals from the optimization
process that manifest consistently and repeatably across architectures and datasets.

### Grounding in existing research

ModelGuard is a lightweight re-implementation of ideas from several published papers:

| Paper | Relevance |
|---|---|
| **Dataset Cartography** (Swayamdipta et al., EMNLP 2020) | Characterises training data quality using per-epoch model confidence dynamics |
| **Learning Curve Extrapolation** (Domhan et al., 2015) | Predicts final model performance from early training curve shape |
| **Meta-Learning / Learning to Learn** (Finn et al., MAML 2017) | Trains a model that learns from the behaviour of other models |
| **Neural Architecture Search (NAS)** | Uses training proxies to avoid training every candidate to completion |

ModelGuard occupies the same conceptual space: **use training-time signals as features
to make model-level predictions**.

---

## What Data Is the System Based On?

### The 120 training runs

The meta-dataset is built from **120 independently trained neural networks**.
Each network was deliberately configured to express one of four failure conditions.
After training, 20 scalar features are extracted from the training curves of each run.
One CSV row = one trained model.

### The input datasets (what the 120 models were trained on)

| Dataset | Type | Size used | Content |
|---|---|---|---|
| `sklearn make_classification` | Synthetic tabular | 1000 samples, 10 numeric features | Continuous-valued artificial feature vectors — represents any generic tabular ML task |
| **FashionMNIST** | Real greyscale images | 5000 samples, flattened to 784 dims | Clothing/footwear items: T-shirts, trousers, dresses, shoes, bags (10 classes) |
| **CIFAR-10** | Real RGB images | 5000 samples, flattened to 3072 dims | Natural objects: aeroplanes, cars, birds, cats, ships, trucks, etc. (10 classes) |

> **Important**: FashionMNIST and CIFAR-10 are **widely-used, real-world benchmark
> datasets** from the computer vision community — not synthetic noise. The data
> fed to the 120 models is real and meaningful.

Note: all three datasets are intentionally processed through *fully-connected MLPs*
(not CNNs). This is deliberate — MLPs on image data are sub-optimal by design,
which makes failure mode signals more pronounced and easier for the meta-classifier
to separate.

---

## The Four Failure Modes

### Class 0 — OVERFIT

**What was done:** A very deep and wide MLP (4 hidden layers × 512 units each),
no dropout, no weight decay, trained for 50 epochs on small data (800 training samples).

**What this mimics in the real world:**
- A practitioner who chose an over-parameterised model without regularisation
- Fine-tuning a large pre-trained model on a tiny dataset without early stopping
- Any scenario where model capacity far exceeds data size

**Signature in training curves:**
- Train accuracy → 95–99%
- Val accuracy → plateaus or declines after a few epochs
- `acc_gap` and `overfit_score` are large
- `val_loss_trend` is positive (val loss rising)

---

### Class 1 — CLASS_IMBALANCE

**What was done:** 90% of training samples come from 2 dominant classes, remaining
10% split across all other classes. Standard MLP, no class reweighting or resampling.

**What this mimics in the real world:**
- **Medical diagnosis**: disease cases are rare, healthy cases dominate
- **Fraud detection**: fraudulent transactions are <1% of the dataset
- **Content moderation**: most content is benign, flagged content is rare
- **Industrial defect detection**: defect-free products far outnumber defective ones

**Signature in training curves:**
- Val accuracy appears reasonable (the model gets the majority class right)
- `class_entropy` is low (label distribution is skewed)
- Model never properly learns minority class boundaries

---

### Class 2 — LABEL_NOISE

**What was done:** 30% of training labels are randomly flipped to a wrong class.
Standard MLP, unmodified training procedure.

**What this mimics in the real world:**
- **Crowdsourced annotation** (MTurk, Scale AI) with unreliable annotators
- **Auto-labelling pipelines** where a weak model generates training labels (noisy pseudo-labels)
- **Web scraping** where image–tag correspondence is imperfect
- **Legacy datasets** where historical records have entry errors

**Signature in training curves:**
- High `loss_volatility` (model keeps seeing contradictory labels)
- Poor final val accuracy despite potentially low train loss
- `train_val_loss_corr` may be lower — the two curves decouple

---

### Class 3 — HEALTHY

**What was done:** Balanced class distribution, moderate architecture (2 layers × 128 units),
dropout = 0.3, trained for 30 epochs.

**What this mimics in the real world:**
- A well-designed training run following best practices
- The baseline "all clear" signal — model is safe to proceed to evaluation/deployment

**Signature in training curves:**
- Train and val accuracy converge smoothly
- `acc_gap` and `loss_gap` remain small
- `val_loss_trend` is negative or near-zero (val loss still decreasing or flat)

---

## Exact Real-World Use Case Where This Makes Sense

> **Scenario**: An ML team at a mid-size company runs 50+ model training experiments
> per week (hyperparameter sweeps, architecture experiments, new data pipelines).
> Before any model gets promoted to staging/QA, the team engineer runs ModelGuard
> on the training log. If ModelGuard flags OVERFIT or LABEL_NOISE, the model is
> automatically rejected and the engineer is given a plain-English explanation
> and remediation steps — *before* a single inference is served to users.

This is the pre-deployment triage role. It is specifically useful when:

1. **The team trains many models** and cannot manually inspect every training curve
2. **Junior practitioners** are running experiments and may not recognise failure signatures
3. **Automated CI/CD pipelines for ML** (MLOps) need machine-readable go/no-go signals
4. **Fast iteration cycles** where training completes overnight and needs automated flagging by morning

---

## Honest Limitations (What This Is NOT)

| Limitation | Explanation |
|---|---|
| 120 training runs is a small meta-dataset | A production system would have thousands of runs across diverse tasks |
| MLP-only underlying models | Failure signatures for CNNs, Transformers, and RNNs may differ |
| Synthetically injected failures | Real-world failures emerge organically and may be subtler or compound |
| No architecture features | Meta-classifier doesn't know model type — only sees training curves |
| Cross-dataset generalisation untested | The meta-classifier was trained and tested on the same 3 base datasets |

---

## How to Present This Work

The correct framing for any report, presentation, or interview:

> *"ModelGuard is a proof-of-concept meta-learning system that demonstrates training
> dynamics signals — loss curves, accuracy gaps, gradient norms — are sufficient to
> discriminate four common ML failure modes with high classification accuracy.
> The system was trained on 120 deliberately-conditioned model runs across two
> real benchmark vision datasets (FashionMNIST, CIFAR-10) and one tabular dataset,
> using synthetically injected failure conditions (overfitting, class imbalance,
> label noise) consistent with failure modes documented in the Dataset Cartography
> and training dynamics literature."*

This is **honest, specific, grounded, and defensible**.
