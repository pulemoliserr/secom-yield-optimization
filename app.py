# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="SECOM Yield Optimization Cockpit", 
    page_icon="🏭", 
    layout="wide"
)

# --- LOAD CACHED PRODUCTION MODEL ARTIFACTS ---
@st.cache_resource
def load_production_assets():
    rf_pipe = joblib.load('production_rf_pipeline.pkl')
    xgb_pipe = joblib.load('production_xgb_pipeline.pkl')
    test_cache = joblib.load('test_dataset_cache.pkl')
    return rf_pipe, xgb_pipe, test_cache['X_test'], test_cache['y_test']

try:
    rf_pipeline, xgb_pipeline, X_test, y_test = load_production_assets()
    y_test_binary = (y_test == 1).astype(int)
except FileNotFoundError:
    st.error("🚨 Model artifacts missing! Please run `python train_model.py` first to generate pipeline binaries.")
    st.stop()

# --- BUSINESS RISK VARIABLES ---
COST_FP = 100
COST_FN = 400

st.title("🏭 SECOM Semiconductor Yield Optimization Engine")
st.markdown("Operational evaluation dashboard using real-time inference across out-of-sample manufacturing runs.")

# --- SIDEBAR STYLE FIX & CONTROL PANEL ---
# This CSS strip removes the default massive top padding inside the sidebar container
st.markdown(
    """
    <style>
    [data-testid="stSidebarUserContent"] {
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🕹️ Simulation Control Tower")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox("Choose Tournament Model:", ["Tuned Random Forest", "Optimized XGBoost"])

slider_threshold = st.sidebar.slider(
    "Adjust Classification Threshold:", 
    min_value=0.05, max_value=0.95, value=0.42, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Threshold Relevance:** "
    "In data science, this trades off Precision vs. Recall by shifting the decision boundary. "
    "In business terms, lowering it catches more defects to reduce escape leakage, but spikes costly false alarms that halt production."
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Cost Parameters:**\n* 🔴 Missed Defect (FN): **${COST_FN}**\n* 🟡 False Alarm (FP): **${COST_FP}**")

# --- INFERENCE RUNTIME EVALUATION ---
proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

active_proba = proba_rf if selected_model == "Tuned Random Forest" else proba_xgb
pred_sim = (active_proba >= slider_threshold).astype(int)

# Matrix Compilations
tn_sim, fp_sim, fn_sim, tp_sim = confusion_matrix(y_test_binary, pred_sim).ravel()
cost_sim = (fp_sim * COST_FP) + (fn_sim * COST_FN)
recall_sim = (tp_sim / (tp_sim + fn_sim)) * 100 if (tp_sim + fn_sim) > 0 else 0.0

# Fixed Champion References
tn_champ, fp_champ, fn_champ, tp_champ = 272, 21, 15, 6
cost_champ = 8100
recall_champ = 28.57

# Helper for matrices with dynamic size scaling
def draw_matrix(tn, fp, fn, tp, figsize=(4.5, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pass", "Fail"], yticklabels=["Actual Normal", "Actual Defect"],
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)
    plt.tight_layout()
    return fig

# ==========================================
# 4. RENDER DASHBOARD LAYOUT (OPTIMIZED HORIZONTAL EXPANSION)
# ==========================================
st.markdown(
    """
    <style>
    .cream-box {
        background-color: #FFFDD0; /* Soft Cream */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #D4AF37; /* Gold accent line */
        margin-top: 10px;
        color: #333333;
    }
    .cream-box h3, .cream-box h4 {
        color: #111111 !important;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# [Left Spacer, Center Stage, Middle Divider, Extended Right Sidebar]
col_space1, col_center, col_divide, col_right = st.columns([0.1, 1.8, 0.3, 0.8])

# --- LEFT MARGIN BUFFER ---
with col_space1:
    st.write("")

# --- MAIN CENTER STAGE: LIVE SIMULATOR TOURNAMENT ---
with col_center:
    st.subheader(f"⚔️ Interactive Arena: {selected_model}")
    
    m1, m2 = st.columns(2)
    m1.metric("Simulated Business Cost", f"${cost_sim:,}", delta=f"${cost_sim - cost_champ:,} vs Champion", delta_color="inverse")
    m2.metric("Recall (Defect Catch Rate)", f"{recall_sim:.1f}%")
    
    st.pyplot(draw_matrix(tn_sim, fp_sim, fn_sim, tp_sim, figsize=(5.5, 3.8)))

# --- MIDDLE DIVIDER ---
with col_divide:
    st.write("")

# --- RIGHT COLUMN: EXTENDED PRODUCTION CHAMPION ---
with col_right:
    st.markdown(
        f"""
        <div class="cream-box">
            <h3>🏆 Production Champion</h3>
            <p>Balancing asymmetric manufacturing penalties requires tracking strict floor limits:</p>
            <ul>
                <li><strong>Floor Cost:</strong> ${cost_champ:,}</li>
                <li><strong>Baseline Recall:</strong> {recall_champ:.1f}%</li>
                <li><strong>Cost-Optimized Threshold:</strong><br>t = 0.420</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("")
    st.markdown("**Locked Target Distribution:**")
    st.pyplot(draw_matrix(tn_champ, fp_champ, fn_champ, tp_champ, figsize=(3.4, 2.2)))
    
    st.info(f"💡 **Note:** Standard metrics like Youden's J assume symmetric errors. By using a cost-optimized threshold framework instead, we actively protect the assembly line bottom line.")
# --- BOTTOM PROFILE LINE TOURNAMENT ---
st.markdown("---")
st.subheader("📈 Financial Trajectory Matrix: Cost vs. Threshold Sweep")
th_space = np.linspace(0.05, 0.95, 50)
curve_rf = [confusion_matrix(y_test_binary, (proba_rf >= t).astype(int)).ravel()[1]*100 + confusion_matrix(y_test_binary, (proba_rf >= t).astype(int)).ravel()[2]*400 for t in th_space]
curve_xgb = [confusion_matrix(y_test_binary, (proba_xgb >= t).astype(int)).ravel()[1]*100 + confusion_matrix(y_test_binary, (proba_xgb >= t).astype(int)).ravel()[2]*400 for t in th_space]

fig_curve, ax = plt.subplots(figsize=(10, 3.0))
ax.plot(th_space, curve_rf, label="Random Forest (Champion)", color="#1f77b4", linewidth=2.5)
ax.plot(th_space, curve_xgb, label="XGBoost", color="#ff7f0e", linestyle="--")
ax.axvline(slider_threshold, color="red", linestyle="-.", linewidth=1.5, label=f"Simulator Marker ({slider_threshold:.2f})")
ax.set_xlabel("Decision Threshold Boundary")
ax.set_ylabel("Total Financial Cost ($)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
st.pyplot(fig_curve)