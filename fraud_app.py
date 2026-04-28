import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | Khang Le",
    page_icon="🔍",
    layout="wide"
)

# ── Load all artifacts (cached so only runs once) ─────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    cfg  = json.load(open(os.path.join(base, "model_config.json")))
    mdl  = lgb.Booster(model_file=os.path.join(base, "final_model_lgbm.txt"))
    imp  = joblib.load(os.path.join(base, "imputer.pkl"))
    scl  = joblib.load(os.path.join(base, "scaler.pkl"))
    exp  = joblib.load(os.path.join(base, "shap_explainer.pkl"))
    Xtr  = pd.read_csv(os.path.join(base, "X_train.csv"))
    return cfg, mdl, imp, scl, exp, Xtr

cfg, mdl, imp, scl, exp, Xtr = load_artifacts()

FEATS = cfg["features"]
THR   = cfg["threshold"]
TOP   = cfg["top_features"]
MEANS = Xtr[FEATS].mean().to_dict()

# ── Header ────────────────────────────────────────────────────
st.title("🔍 IEEE-CIS Fraud Detection")
st.markdown(
    f"**Model:** {cfg['model_name']} &nbsp;|&nbsp; "
    f"**Test AUC:** {cfg['test_auc']:.4f} &nbsp;|&nbsp; "
    f"**Avg Precision:** {cfg['test_ap']:.4f} &nbsp;|&nbsp; "
    f"**F1:** {cfg['test_f1']:.4f}"
)
st.markdown(
    "Enter values for the most important features below. "
    "All other features are automatically filled from training-data averages."
)
st.divider()

# ── Sidebar inputs (top features only) ───────────────────────
st.sidebar.header("Transaction Inputs")
st.sidebar.markdown("Adjust the sliders to describe the transaction:")

user_inputs = {}
for feat in TOP[:8]:
    if feat not in Xtr.columns:
        continue
    mn  = float(Xtr[feat].min())
    mx  = float(Xtr[feat].max())
    med = float(Xtr[feat].median())
    user_inputs[feat] = st.sidebar.slider(
        label=feat,
        min_value=mn,
        max_value=mx,
        value=med,
        help=f"Training median: {med:.2f}"
    )

score_btn = st.sidebar.button("Score Transaction", type="primary", use_container_width=True)

# ── Scoring logic ─────────────────────────────────────────────
def build_and_score(user_inputs):
    # Fill all features with training means, override with user inputs
    row = {f: MEANS.get(f, 0.0) for f in FEATS}
    row.update(user_inputs)
    X = pd.DataFrame([row])[FEATS]
    X = pd.DataFrame(imp.transform(X),  columns=FEATS)
    X = pd.DataFrame(scl.transform(X),  columns=FEATS)
    prob = float(mdl.predict(X)[0])
    return prob, X

# ── Main panel ────────────────────────────────────────────────
if score_btn:
    prob, X_scored = build_and_score(user_inputs)
    label = "FRAUD" if prob >= THR else "LEGITIMATE"

    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", label)
    col2.metric("Fraud Probability", f"{prob:.4f}")
    col3.metric("Classification Threshold", f"{THR:.4f}")

    if prob >= THR:
        st.error(
            f"Transaction FLAGGED for review. "
            f"Fraud probability {prob:.2%} exceeds threshold {THR:.2%}."
        )
    else:
        st.success(
            f"Transaction appears LEGITIMATE. "
            f"Fraud probability {prob:.2%} is below threshold {THR:.2%}."
        )

    st.divider()

    # ── SHAP waterfall (local explainability) ─────────────────
    st.subheader("Why this prediction? — SHAP Feature Contributions")
    st.caption(
        "The waterfall chart shows which features pushed the score "
        "above or below the base rate. Red = increases fraud risk, Blue = decreases it."
    )

    shap_vals = exp.shap_values(X_scored)

# Handle both old (2D) and new (3D) SHAP output shapes
sv_arr = np.array(shap_vals)
if sv_arr.ndim == 3:
    # New SHAP: (n_samples, n_features, n_classes) - take fraud class
    values_for_plot = sv_arr[0, :, 1] if sv_arr.shape[2] > 1 else sv_arr[0, :, 0]
    base_val = exp.expected_value[1] if hasattr(exp.expected_value, '__len__') else exp.expected_value
elif isinstance(shap_vals, list):
    # Older SHAP: list of arrays per class - take fraud class
    values_for_plot = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    base_val = exp.expected_value[1] if hasattr(exp.expected_value, '__len__') else exp.expected_value
else:
    # 2D array
    values_for_plot = sv_arr[0]
    base_val = exp.expected_value if not hasattr(exp.expected_value, '__len__') else exp.expected_value[0]

fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=values_for_plot,
        base_values=base_val,
        data=X_scored.iloc[0].values,
        feature_names=FEATS
    ),
    max_display=12,
    show=False
)
)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

else:
    # Default view before any scoring
    st.info("Adjust the sliders in the sidebar and click **Score Transaction** to get a prediction.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Performance Summary")
        perf = pd.DataFrame({
            "Metric": ["AUC-ROC", "Avg Precision", "F1-Score", "Accuracy"],
            "Test Score": [
                f"{cfg['test_auc']:.4f}",
                f"{cfg['test_ap']:.4f}",
                f"{cfg['test_f1']:.4f}",
                f"{cfg['test_acc']:.4f}",
            ]
        })
        st.dataframe(perf, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Top Predictive Features")
        fi = pd.DataFrame({
            "Feature": TOP[:10],
            "Rank": [f"#{i}" for i in range(1, 11)]
        })
        st.dataframe(fi, hide_index=True, use_container_width=True)

st.divider()
st.caption("Milestone 4 — IEEE-CIS Fraud Detection | Author: Khang Le | Streamlit Community Cloud")
