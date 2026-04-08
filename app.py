"""
app.py — Stage 4 of ModelGuard
Streamlit demo: upload training logs → failure-mode prediction with explanations.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import extract_features, plot_curves

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ModelGuard — Pre-Deployment Failure Predictor",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FAILURE_LABELS = {
    0: "OVERFIT", 1: "CLASS_IMBALANCE", 2: "LABEL_NOISE", 3: "HEALTHY",
    4: "VANISHING_GRADIENT", 5: "CATASTROPHIC_FORGETTING", 6: "DATA_DRIFT"
}

FAILURE_COLORS = {
    "OVERFIT":          "#e63946",
    "CLASS_IMBALANCE":  "#f4a261",
    "LABEL_NOISE":      "#2a9d8f",
    "HEALTHY":          "#457b9d",
    "VANISHING_GRADIENT": "#8b5cf6",
    "CATASTROPHIC_FORGETTING": "#ec4899",
    "DATA_DRIFT":       "#14b8a6",
}

FAILURE_EXPLANATIONS = {
    "OVERFIT": (
        "⚠️ **Model memorized training data.**\n\n"
        "Your model is fitting noise rather than learning generalisable patterns. "
        "You'll see a large gap between training and validation accuracy.\n\n"
        "**Suggested fixes:**\n"
        "- Add Dropout layers (0.3–0.5)\n"
        "- Apply L1/L2 weight regularisation\n"
        "- Gather more training data\n"
        "- Reduce model capacity (fewer layers/units)\n"
        "- Use early stopping based on val loss"
    ),
    "CLASS_IMBALANCE": (
        "⚠️ **Training data is skewed across classes.**\n\n"
        "One or a few classes dominate your dataset. "
        "The model learns to predict the majority class and ignores minority classes.\n\n"
        "**Suggested fixes:**\n"
        "- Use class-weighted loss (`weight` param in CrossEntropyLoss)\n"
        "- Oversample minority classes (SMOTE, random oversampling)\n"
        "- Undersample majority classes\n"
        "- Collect more data for under-represented classes"
    ),
    "LABEL_NOISE": (
        "⚠️ **Labels may be corrupted or inconsistently annotated.**\n\n"
        "A significant portion of your training samples may have wrong labels, "
        "causing the model to learn contradictory signals.\n\n"
        "**Suggested fixes:**\n"
        "- Audit your labelling pipeline / annotation process\n"
        "- Use label-smoothing in your loss function\n"
        "- Apply confident-learning or cleanlab to detect mislabelled samples\n"
        "- Cross-validate labels with a second annotator"
    ),
    "HEALTHY": (
        "✅ **No failure detected. Model dynamics look stable.**\n\n"
        "Train/val curves are converging nicely. "
        "The class distribution appears balanced and labels seem consistent.\n\n"
        "**Next steps:**\n"
        "- Proceed with hyperparameter tuning\n"
        "- Run a full evaluation suite on a held-out test set\n"
        "- Consider ONNX export or quantisation for production deployment"
    ),
    "VANISHING_GRADIENT": (
        "⚠️ **Model is stalling due to vanishing gradients.**\n\n"
        "Your model's weights are not updating effectively because the gradient signal is too weak.\n\n"
        "**Suggested fixes:**\n"
        "- Use ReLU or LeakyReLU activations instead of Sigmoid/Tanh\n"
        "- Add Batch Normalization layers\n"
        "- Use residual connections (ResNet style)\n"
        "- Check weight initialization (e.g. use He initialization)\n"
    ),
    "CATASTROPHIC_FORGETTING": (
        "⚠️ **Model forgot previously learned information.**\n\n"
        "The current training data distribution differs completely from earlier data, causing the model to overwrite earlier representations.\n\n"
        "**Suggested fixes:**\n"
        "- Randomize/shuffle your training dataset completely before training\n"
        "- Use experience replay if doing continual learning\n"
        "- Lower the learning rate\n"
    ),
    "DATA_DRIFT": (
        "⚠️ **Validation data distribution is shifting.**\n\n"
        "The characteristics of your validation set are consistently drifting away from your training set distribution.\n\n"
        "**Suggested fixes:**\n"
        "- Update your training set with recent real-world examples\n"
        "- Add strong data augmentation to improve robustness\n"
        "- Monitor input data distributions in production\n"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    rf_path     = os.path.join("models", "rf_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    if not os.path.exists(rf_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(rf_path), joblib.load(scaler_path)


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder for user-uploaded CSV
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "final_train_loss", "final_val_loss", "final_train_acc", "final_val_acc",
    "loss_gap", "acc_gap", "overfit_score", "val_loss_trend", "train_loss_trend",
    "loss_volatility", "acc_volatility", "best_val_acc", "best_val_epoch",
    "convergence_speed", "train_val_loss_corr",
    "gradient_norm_mean", "gradient_norm_std",
]


def build_feature_vector(df_log: pd.DataFrame, overrides: dict) -> np.ndarray:
    """
    Compute features from uploaded epoch-log CSV plus manual overrides.
    Returns a 1-D numpy array in FEATURE_ORDER.
    """
    epoch_logs = {
        "train_loss": df_log["train_loss"].tolist(),
        "val_loss":   df_log["val_loss"].tolist(),
        "train_acc":  df_log["train_acc"].tolist(),
        "val_acc":    df_log["val_acc"].tolist(),
    }
    curve_feats = extract_features(epoch_logs)

    # Merge overrides for non-curve features
    curve_feats["gradient_norm_mean"] = overrides.get("gradient_norm_mean", 0.0)
    curve_feats["gradient_norm_std"]  = overrides.get("gradient_norm_std",  0.0)

    return np.array([curve_feats[f] for f in FEATURE_ORDER], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #12122b 60%, #0a1628 100%); }

/* Cards */
.mg-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}

/* Hero title */
.mg-hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.mg-hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.mg-hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0;
}

/* Prediction badge */
.mg-badge {
    display: inline-block;
    padding: 0.5rem 1.4rem;
    border-radius: 40px;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4);
    margin-bottom: 0.8rem;
}

/* Confidence bar container */
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 14px;
    width: 100%;
    margin-top: 6px;
}
.conf-bar-fill {
    height: 14px;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Warning banner */
.mg-warning {
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid rgba(251, 191, 36, 0.35);
    border-radius: 10px;
    padding: 0.75rem 1.2rem;
    color: #fbbf24;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Manual Feature Overrides")
    st.markdown(
        "These features cannot be derived from a plain epoch-log CSV. "
        "Set them manually if known, or leave at defaults."
    )

    gradient_norm_mean = st.slider(
        "gradient_norm_mean", 0.0, 10.0, 0.0, 0.01,
        help="Mean L2 gradient norm per epoch averaged over training."
    )
    gradient_norm_std = st.slider(
        "gradient_norm_std", 0.0, 50.0, 0.0, 0.01,
        help="Std-dev of per-epoch gradient norms."
    )

    st.markdown("---")
    st.markdown(
        "**Expected CSV columns:**\n"
        "`epoch, train_loss, val_loss, train_acc, val_acc`"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mg-hero">
  <h1>🛡️ ModelGuard</h1>
  <p>Pre-Deployment Failure Predictor &nbsp;|&nbsp; Detect failure modes from training dynamics — before you ship.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
rf_model, scaler = load_models()

if rf_model is None:
    st.markdown("""
    <div class="mg-warning">
    ⚠️ <strong>Models not found.</strong>
    Run <code>generate_meta_dataset.py</code> → <code>train_meta_classifier.py</code> first,
    then relaunch this app.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="mg-card">', unsafe_allow_html=True)
st.markdown("### 📂 Upload Training Log CSV")
st.markdown(
    "CSV must have columns: `epoch, train_loss, val_loss, train_acc, val_acc`"
)

SAMPLE_CSV = (
    "epoch,train_loss,val_loss,train_acc,val_acc\n"
    "1,1.85,1.92,0.31,0.28\n"
    "2,1.60,1.75,0.42,0.38\n"
    "3,1.35,1.55,0.55,0.48\n"
    "4,1.10,1.45,0.63,0.52\n"
    "5,0.88,1.38,0.72,0.54\n"
    "6,0.65,1.42,0.80,0.53\n"
    "7,0.48,1.50,0.87,0.52\n"
    "8,0.32,1.61,0.91,0.51\n"
    "9,0.21,1.73,0.95,0.50\n"
    "10,0.14,1.82,0.97,0.49\n"
)

col_up, col_dl = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader("Choose a CSV…", type=["csv"], label_visibility="collapsed")
with col_dl:
    st.download_button(
        "⬇️ Sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_training_log.csv",
        mime="text/csv",
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main prediction flow
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        df_log = pd.read_csv(uploaded)
        required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
        missing  = required - set(df_log.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df_log = df_log.sort_values("epoch").reset_index(drop=True)

        # ── Prediction ────────────────────────────────────────────────────────
        overrides = {
            "gradient_norm_mean": gradient_norm_mean,
            "gradient_norm_std":  gradient_norm_std,
        }

        feat_vec = build_feature_vector(df_log, overrides).reshape(1, -1)
        feat_scaled = scaler.transform(feat_vec)

        proba        = rf_model.predict_proba(feat_scaled)[0]
        pred_class   = int(np.argmax(proba))
        pred_label   = FAILURE_LABELS[pred_class]
        confidence   = float(proba[pred_class])
        badge_color  = FAILURE_COLORS[pred_label]

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown('<div class="mg-card">', unsafe_allow_html=True)
        st.markdown("### 🔮 Prediction Result")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f'<div class="mg-badge" style="background:{badge_color};">'
                f'{pred_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
            st.markdown(
                f'<div class="conf-bar-bg">'
                f'<div class="conf-bar-fill" style="width:{confidence*100:.1f}%;'
                f'background:{badge_color};"></div></div>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown("#### What does this mean?")
            st.markdown(FAILURE_EXPLANATIONS[pred_label])

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Probability breakdown ─────────────────────────────────────────────
        st.markdown('<div class="mg-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Class Probabilities")
        prob_df = pd.DataFrame({
            "Failure Mode": [FAILURE_LABELS[i] for i in range(7)],
            "Probability":  proba,
        }).set_index("Failure Mode")
        st.bar_chart(prob_df)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Curve plots ───────────────────────────────────────────────────────
        st.markdown('<div class="mg-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Uploaded Training Curves")
        fig = plot_curves(
            df_log["train_loss"].tolist(),
            df_log["val_loss"].tolist(),
            df_log["train_acc"].tolist(),
            df_log["val_acc"].tolist(),
        )
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Key computed features ─────────────────────────────────────────────
        with st.expander("🔬 View computed feature vector"):
            feat_dict = build_feature_vector.__wrapped__(df_log, overrides) \
                if hasattr(build_feature_vector, "__wrapped__") \
                else dict(zip(FEATURE_ORDER, feat_vec.flatten()))
            from utils import extract_features as _ef
            raw_feats = _ef({
                "train_loss": df_log["train_loss"].tolist(),
                "val_loss":   df_log["val_loss"].tolist(),
                "train_acc":  df_log["train_acc"].tolist(),
                "val_acc":    df_log["val_acc"].tolist(),
            })
            raw_feats.update(overrides)
            st.json(raw_feats)

    except Exception as exc:
        st.error(f"Error processing file: {exc}")
else:
    st.info(
        "👆 Upload your training log CSV above to get started. "
        "Download the **Sample CSV** to see the expected format."
    )
