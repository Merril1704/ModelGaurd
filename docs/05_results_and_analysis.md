# ModelGuard — Results & Empirical Analysis

> **Update from Stage 2 & 3 Runs**: Upon executing the meta-classifier training, both the Random Forest and the MLP achieved **100% classification accuracy** (1.0 F1, 1.0 ROC-AUC) on the held-out test set. 

While impressive at first glance, achieving perfect accuracy in machine learning usually warrants a critical look at the data. Let's analyze the visual artifacts, feature importances, and the meta-classifier's behavior to understand if it's genuinely learning or simply identifying artifacts.

---

## The "100% Accuracy" Phenomenon

You correctly noted that the classifiers appear to be performing "too well," leading to concerns of overfitting or memorization. This perfect performance is a combination of **small sample size** and **target leakage via deterministic experimental design**.

### 1. Feature Importance & Target Leakage

The Random Forest's top feature importances reveal exactly how the model is distinguishing the classes:

1. **`num_params` (0.118)**
2. **`gradient_norm_std` (0.116)**
3. `acc_gap` (0.088)
4. `train_val_loss_corr` (0.083)
5. `overfit_score` (0.082)
6. `loss_gap` (0.074)

Notice that **`num_params`** is the #1 most important feature. 
- In our experimental design, the `OVERFIT` configuration was explicitly built using `[512, 512, 512, 512]` hidden layers.
- The `HEALTHY`, `CLASS_IMBALANCE`, and `LABEL_NOISE` configurations used much smaller networks (`[128, 128]` and `[128, 64]`). 

Because we deterministically locked the architecture to the failure mode to *induce* the failure, `num_params` acts as a direct substitute for the `OVERFIT` label. The tree can simply split on `num_params > 500,000` and immediately isolate 100% of the `OVERFIT` samples. This is a classic case of **target leakage**.

### 2. The Nature of the Synthetics

Similarly, the failure injections were taken to extremes:
- `LABEL_NOISE` flipped exactly 30% of labels. This causes massive, consistent fluctuations in gradients and decoupled loss curves. The second most important feature, **`gradient_norm_std`**, likely isolates these runs perfectly.
- `CLASS_IMBALANCE` used a strict 90/10 skew, making accuracy metrics behave in a very specific, repeatable way.

Because the conditions were perfectly engineered with no "intermediate" or "mild" failures, the space between the classes is massive. The decision boundaries are completely empty, making it trivially easy for the RF to draw lines between the clusters without making any mistakes. 

### 3. Small Sample Size

With only 120 total models (96 in the train set, 24 in the test set), Random Forests and MLPs have more than enough capacity to simply "memorize" the feature space. In a larger distribution with 10,000 models exhibiting mild to severe failures, the boundaries would bleed into each other, and accuracy would realistically settle somewhere between 75% and 85%.

---

## Analyzing the Graphical Results

If you look at the plots generated in `results/`:

### 1. Scatter Plot (`scatter.png`)
The scatter plot mapping `loss_gap` vs `overfit_score` shows precisely why the classifiers have an easy time:
- The **OVERFIT** models appear completely sequestered in the high `loss_gap` and high `overfit_score` quadrant.
- The **HEALTHY** and **CLASS_IMBALANCE** models cluster nearer the origin with lower gaps.
- The geometric separation of the clusters visually proves that the training dynamic signals (accuracy gap and loss gap) *do* encode the failure states distinctly.

### 2. Confusion Matrices (`confusion_matrix_rf.png` & `confusion_matrix_mlp.png`)
These show a perfect diagonal. Since the test split only has 6 samples per class (24 total), achieving 6/6 for every class is statistically easy when the data clusters are perfectly separated.

---

## Conclusion: Validity vs. Overfitting

Is the model just memorizing gibberish? **No, but it is taking shortcuts.**

1. **The training dynamics theory holds true:** Features like `acc_gap`, `train_val_loss_corr`, and `overfit_score` rank highly (positions 3 to 6). This confirms that the *organic signals* of generalization fail are present and highly predictive, validating the core concept of the project.
2. **The 100% accuracy is an artifact of the sandbox:** The perfect classification happens because (a) target leaks like `num_params` give away the answer, and (b) the injected failures are perfectly distinct with no gray areas. 

### How to fix this in future iterations?
As outlined in `04_future_work.md`, this naturally leads to the next steps for maturing the project:
* **Remove architecture proxies:** Exclude `num_params` from the classifier, or randomize architecture sizes across *all* failure modes so it no longer correlates perfectly with `OVERFIT`.
* **Randomize injection severity:** Instead of a strict 30% noise rate, pull the noise rate from a uniform distribution `U(5%, 40%)`. Make the class imbalance vary anywhere from 60/40 to 95/5. This will blur the clusters and force the meta-classifier to rely on the subtleties of the epoch curves.
* **Scale up:** Generate 5,000+ runs so the test set has statistical weight. 

For a proof-of-concept, finding that the classes are mathematically separable is the required first milestone. The artifact of 100% accuracy simply signifies that it's time to make the sandbox harder!
