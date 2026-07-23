"""
SECOM Semiconductor Yield Optimization Engine
-----------------------------------------------
Interactive dashboard for exploring how the classification threshold and
the Type I / Type II cost assumptions change a model's confusion matrix,
recall, and total business cost -- without retraining anything.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SECOM Yield Optimization Engine", page_icon="🏭", layout="wide")

PREDICTIONS_FILE = Path(__file__).parent / "secom_test_predictions.csv"
MODEL_PARAMS_FILE = Path(__file__).parent / "secom_model_params.json"

MODEL_LABELS = {
    "proba_rf": "Tuned Random Forest",
    "proba_xgb": "XGBoost",
}

DEMO_MODEL_PARAMS = {
    "proba_rf": {
        "Algorithm": "Random Forest (scikit-learn)",
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_neighbors (imputer)": 5,
        "random_state": 42,
    },
    "proba_xgb": {
        "Algorithm": "XGBoost",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
    },
}


# --------------------------------------------------------------------------- #
# Data loading & Caching
# --------------------------------------------------------------------------- #
@st.cache_data
def make_demo_predictions(seed: int = 42, n: int = 314, defect_rate: float = 0.0669) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < defect_rate).astype(int)
    proba_rf = np.clip(rng.beta(2, 22, n) + y * rng.beta(3, 4, n) * 0.6, 0, 1)
    proba_xgb = np.clip(rng.beta(2, 19, n) + y * rng.beta(3, 3.5, n) * 0.55, 0, 1)
    return pd.DataFrame({"y_test": y, "proba_rf": proba_rf, "proba_xgb": proba_xgb})


@st.cache_data
def load_predictions():
    if PREDICTIONS_FILE.exists():
        return pd.read_csv(PREDICTIONS_FILE), "loaded from secom_test_predictions.csv"
    return make_demo_predictions(), "demo data -- secom_test_predictions.csv not found next to app.py"


@st.cache_data
def load_model_params():
    if MODEL_PARAMS_FILE.exists():
        with open(MODEL_PARAMS_FILE) as f:
            return json.load(f), "loaded from secom_model_params.json"
    return DEMO_MODEL_PARAMS, "demo values (illustrative) -- secom_model_params.json not found next to app.py"


def build_params_table(model_params: dict, available_models: dict) -> pd.DataFrame:
    ordered_keys = []
    for col in available_models:
        for k in model_params.get(col, {}):
            if k not in ordered_keys:
                ordered_keys.append(k)
    table = {
        label: [str(model_params.get(col, {}).get(k, "\u2014")) for k in ordered_keys]
        for col, label in available_models.items()
    }
    return pd.DataFrame(table, index=ordered_keys)


df, data_source_note = load_predictions()
available_models = {col: label for col, label in MODEL_LABELS.items() if col in df.columns}
if not available_models:
    st.error("No probability columns found. Expected at least one of: " + ", ".join(MODEL_LABELS))
    st.stop()

model_params, params_source_note = load_model_params()
model_params = {col: params for col, params in model_params.items() if col in available_models}


# --------------------------------------------------------------------------- #
# Cost & Metric Calculations
# --------------------------------------------------------------------------- #
def confusion_counts(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float, cost_fp: float, cost_fn: float) -> dict:
    c = confusion_counts(y_true, proba, threshold)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    cost = fp * cost_fp + fn * cost_fn
    return {**c, "recall": recall, "precision": precision, "specificity": specificity, "f1": f1, "cost": cost}


def cost_curve(y_true: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float) -> tuple:
    grid = np.linspace(0.01, 0.99, 99)
    costs = np.array([metrics_at(y_true, proba, t, cost_fp, cost_fn)["cost"] for t in grid])
    return grid, costs


def sweep_all_metrics(y_true: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float):
    grid = np.linspace(0.01, 0.99, 99)
    rows = [metrics_at(y_true, proba, t, cost_fp, cost_fn) for t in grid]
    return grid, pd.DataFrame(rows)


def sweep_thresholds(y_true: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float, min_recall: float = 0.0):
    grid = np.linspace(0.01, 0.99, 99)
    rows = [metrics_at(y_true, proba, t, cost_fp, cost_fn) for t in grid]
    costs = np.array([r["cost"] for r in rows])
    recalls = np.array([r["recall"] for r in rows])

    best_idx = int(np.argmin(costs))

    feasible = np.where(recalls >= min_recall)[0]
    if len(feasible) > 0:
        best_feasible_idx = feasible[np.argmin(costs[feasible])]
    else:
        best_feasible_idx = best_idx

    return grid, costs, recalls, grid[best_idx], grid[best_feasible_idx]


# --------------------------------------------------------------------------- #
# Sidebar Controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🎛️ Simulation Control Tower")
    st.divider()

    model_col = st.selectbox(
        "Choose Tournament Model:",
        options=list(available_models.keys()),
        format_func=lambda c: available_models[c],
    )

    threshold = st.slider("Adjust Classification Threshold:", 0.0, 1.0, 0.42, 0.01)

    st.caption(
        "**Threshold relevance:** lowering it catches more defects (higher "
        "recall) but raises false alarms; raising it does the reverse."
    )

    st.divider()
    st.markdown("**Cost Parameters**")
    cost_fp = st.number_input("Cost per False Alarm (Type I)", min_value=1, value=100, step=10)
    cost_fn = st.number_input("Cost per Missed Defect (Type II)", min_value=1, value=400, step=10)

    st.divider()
    st.markdown("**Business Constraint**")
    min_recall_pct = st.slider(
        "Target Recall SLA Constraint (%)", 
        min_value=0, 
        max_value=100, 
        value=80, 
        step=1,
        help="Finds the lowest-cost threshold (t) that guarantees at least this percentage of true defects are caught."
    )

    st.divider()
    st.caption(f"Data source: {data_source_note}")


# --------------------------------------------------------------------------- #
# Core Computation
# --------------------------------------------------------------------------- #
y_true = df["y_test"].to_numpy()
proba = df[model_col].to_numpy()

current = metrics_at(y_true, proba, threshold, cost_fp, cost_fn)
grid, costs, recalls, best_t, best_t_sla = sweep_thresholds(
    y_true, proba, cost_fp, cost_fn, min_recall=min_recall_pct / 100
)
champion = metrics_at(y_true, proba, best_t, cost_fp, cost_fn)
champion_sla = metrics_at(y_true, proba, best_t_sla, cost_fp, cost_fn)

model_curves = {}
all_metrics_at_current = {}
full_sweeps = {}

for col, label in available_models.items():
    p = df[col].to_numpy()
    g, c = cost_curve(y_true, p, cost_fp, cost_fn)
    at_current = metrics_at(y_true, p, threshold, cost_fp, cost_fn)
    model_curves[col] = {"label": label, "grid": g, "costs": c, "cost_at_current": at_current["cost"]}
    all_metrics_at_current[col] = at_current
    _, full_sweeps[col] = sweep_all_metrics(y_true, p, cost_fp, cost_fn)

cheapest_col = min(model_curves, key=lambda c: model_curves[c]["cost_at_current"])


# --------------------------------------------------------------------------- #
# Header Section
# --------------------------------------------------------------------------- #
st.title("🏭 SECOM Semiconductor Yield Optimization Engine")
st.caption(
    "Operational evaluation dashboard driven by cached out-of-sample predictions -- "
    "moving the threshold or cost parameters recomputes everything below instantly, "
    "with no retraining involved."
)
st.markdown("---")


# --------------------------------------------------------------------------- #
# Dashboard Main Content Body
# --------------------------------------------------------------------------- #
left, right = st.columns([2.1, 1])

with left:
    st.subheader(f"⚔️ Interactive Arena: {available_models[model_col]}")

    # Top Central Cards: Active Model Cost & Recall
    m1, m2 = st.columns(2)
    delta_cost = current["cost"] - champion["cost"]
    m1.metric(
        f"{available_models[model_col]} Simulated Cost", 
        f"${current['cost']:,}",
        delta=f"${delta_cost:+,} vs cost-optimal threshold", 
        delta_color="inverse",
    )
    m2.metric(f"{available_models[model_col]} Recall", f"{current['recall']*100:.1f}%")

    # Dynamic Comparison Cards: Figures for all models at threshold t
    st.markdown(f"**Model Cost Comparison @ t = {threshold:.2f}**")
    cost_cols = st.columns(len(model_curves))
    for i, (col, mc) in enumerate(model_curves.items()):
        is_cheapest = col == cheapest_col and len(model_curves) > 1
        cost_cols[i].metric(
            f"{mc['label']} Cost",
            f"${mc['cost_at_current']:,}",
            delta="cheaper" if is_cheapest else None,
            delta_color="normal",
        )

    # Central Financial Trajectory Graph
    curve = go.Figure()
    palette = {"proba_rf": "#1E2761", "proba_xgb": "#F5A623"}
    dash = {"proba_rf": "solid", "proba_xgb": "dash"}
    for col, mc in model_curves.items():
        name = mc["label"] + ("  (cheaper now)" if col == cheapest_col and len(model_curves) > 1 else "")
        curve.add_trace(go.Scatter(
            x=mc["grid"], y=mc["costs"], mode="lines", name=name,
            line=dict(color=palette.get(col, "#6B7280"), width=3, dash=dash.get(col, "solid")),
        ))
    curve.add_vline(x=threshold, line_dash="dashdot", line_color="#C0392B", line_width=2,
                    annotation_text=f"Current threshold ({threshold:.2f})", annotation_position="top")
    
    curve.update_layout(
        title=dict(
            text="📈 Financial Trajectory Matrix: Cost vs. Threshold Sweep",
            y=0.96,
            x=0,
            xanchor="left",
            font=dict(size=18),
        ),
        height=450,
        xaxis_title="Decision Threshold Boundary",
        yaxis_title="Total Financial Cost ($)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=10, r=10, t=70, b=60),
    )
    st.plotly_chart(curve, use_container_width=True)

    # Side-by-Side Confusion Matrices
    with st.expander("📊 Dynamic Confusion Matrices (Side-by-Side)", expanded=False):
        cm_cols = st.columns(len(available_models))
        for idx, (col, label) in enumerate(available_models.items()):
            with cm_cols[idx]:
                st.markdown(f"**{label}**")
                m = all_metrics_at_current[col]
                cm = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted Normal", "Predicted Defect"],
                    y=["Actual Normal", "Actual Defect"],
                    colorscale=[[0, "#EAF0FB"], [1, palette.get(col, "#1E2761")]],
                    showscale=False, text=cm, texttemplate="%{text}", textfont={"size": 18},
                ))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown(
        f"""
        <div style="background:#FDF6D8;padding:1.1rem 1.3rem;border-radius:0.6rem;
                    border-left:5px solid #E8B923;">
        <h4 style="margin-top:0;">🏆 Global Unconstrained Champion</h4>
        <p style="font-size:0.88rem; margin-bottom:0.5rem;">Absolute lowest-cost threshold across all recall levels for this model.</p>
        <ul style="font-size:0.88rem; padding-left:1.2rem; margin-bottom:0;">
            <li><b>Cost-Optimal Threshold:</b> t = {best_t:.3f}</li>
            <li><b>Minimum Cost:</b> ${champion['cost']:,}</li>
            <li><b>Recall Achieved:</b> {champion['recall']*100:.1f}%</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic SLA Card updating continuously with the SLA slider target
    sla_card_title = (
        f"🎯 Optimal Parameters @ {min_recall_pct}% Target Recall SLA"
        if min_recall_pct > 0
        else "🎯 Baseline (No SLA Target Applied)"
    )

    st.markdown(
        f"""
        <div style="background:#EAF6F2;padding:1.1rem 1.3rem;border-radius:0.6rem;
                    border-left:5px solid #00A896;margin-top:0.8rem;">
        <h4 style="margin-top:0;">{sla_card_title}</h4>
        <ul style="font-size:0.88rem; padding-left:1.2rem; margin-bottom:0;">
            <li><b>Required Threshold:</b> t = {best_t_sla:.3f}</li>
            <li><b>Total Cost at SLA:</b> ${champion_sla['cost']:,}</li>
            <li><b>Actual Recall Achieved:</b> {champion_sla['recall']*100:.1f}%</li>
            <li><b>False Alarms (FP):</b> {champion_sla['fp']:,}</li>
            <li><b>Missed Defects (FN):</b> {champion_sla['fn']:,}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#####")
    st.markdown("**Class distribution in this test set**")
    dist = pd.DataFrame({
        "Class": ["Normal", "Defect"],
        "Count": [int((y_true == 0).sum()), int((y_true == 1).sum())],
    }).set_index("Class")
    st.bar_chart(dist, use_container_width=True, color="#1E2761")

st.markdown("---")

# --------------------------------------------------------------------------- #
# All 7 Detailed Metrics Comparison Table
# --------------------------------------------------------------------------- #
st.markdown(f"### 📋 All 7 Key Metrics Comparison at t = {threshold:.2f}")

metrics_table_data = {
    "Metric": [
        "Recall (Defect Catch Rate)", 
        "Precision (True Defect Share)", 
        "Specificity (True Normal Share)", 
        "F1 Score", 
        "False Alarms (FP)", 
        "Missed Defects (FN)", 
        "Total Financial Cost ($)"
    ]
}

for col, label in available_models.items():
    m = all_metrics_at_current[col]
    metrics_table_data[label] = [
        f"{m['recall']*100:.1f}%",
        f"{m['precision']*100:.1f}%",
        f"{m['specificity']*100:.1f}%",
        f"{m['f1']*100:.1f}%",
        f"{m['fp']:,}",
        f"{m['fn']:,}",
        f"${m['cost']:,}",
    ]

st.table(pd.DataFrame(metrics_table_data).set_index("Metric"))

# --------------------------------------------------------------------------- #
# Metric Variation Across Thresholds Chart
# --------------------------------------------------------------------------- #
st.markdown("### 📊 Metric Dynamics Across Decision Threshold Sweeps")

selected_metric = st.selectbox(
    "Select Metric to Visualize Across Thresholds (t):",
    options=["recall", "precision", "specificity", "f1", "fp", "fn", "cost", "All Normalized Metrics (0-1)"],
    format_func=lambda x: {
        "recall": "Recall",
        "precision": "Precision",
        "specificity": "Specificity",
        "f1": "F1 Score",
        "fp": "False Alarms (FP)",
        "fn": "Missed Defects (FN)",
        "cost": "Total Financial Cost ($)",
        "All Normalized Metrics (0-1)": "All Metrics Combined (0 to 1 Scale)",
    }.get(x, x),
)

metric_fig = go.Figure()

if selected_metric == "All Normalized Metrics (0-1)":
    df_active = full_sweeps[model_col]
    metric_colors = {
        "recall": "#27AE60",
        "precision": "#2980B9",
        "specificity": "#8E44AD",
        "f1": "#F39C12"
    }
    for m_key, m_color in metric_colors.items():
        metric_fig.add_trace(go.Scatter(
            x=grid, y=df_active[m_key], mode="lines", name=m_key.capitalize(),
            line=dict(color=m_color, width=2.5)
        ))
    chart_title = f"Rate Metrics Trade-off for {available_models[model_col]}"
    y_label = "Metric Value (0.0 - 1.0)"
else:
    for col, label in available_models.items():
        df_sweep = full_sweeps[col]
        metric_fig.add_trace(go.Scatter(
            x=grid, y=df_sweep[selected_metric], mode="lines", name=label,
            line=dict(color=palette.get(col, "#6B7280"), width=3, dash=dash.get(col, "solid"))
        ))
    chart_title = f"{selected_metric.capitalize()} Trajectory Across Thresholds"
    y_label = selected_metric.upper() if selected_metric in ["fp", "fn"] else selected_metric.capitalize()

metric_fig.add_vline(x=threshold, line_dash="dashdot", line_color="#C0392B", line_width=2,
                     annotation_text=f"Current t ({threshold:.2f})", annotation_position="top")

metric_fig.update_layout(
    title=dict(
        text=chart_title,
        y=0.96,
        x=0,
        xanchor="left",
        font=dict(size=18),
    ),
    height=450,
    xaxis_title="Decision Threshold (t)",
    yaxis_title=y_label,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=10, r=10, t=70, b=60),
)

st.plotly_chart(metric_fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Base Section: Hyperparameters Collapsible Expander
# --------------------------------------------------------------------------- #
with st.expander("🔧 Model Configuration — Hyperparameters & Architecture", expanded=False):
    params_df = build_params_table(model_params, available_models)
    st.dataframe(params_df, use_container_width=True)
    st.caption(
        f"Data source: {params_source_note}. `n_estimators`, `max_depth`, and "
        f"`min_samples_leaf` for the Random Forest -- and the equivalent XGBoost "
        f"parameters -- were selected via cross-validated `RandomizedSearchCV` "
        f"scored on Average Precision, not chosen by hand. The currently selected "
        f"model in the sidebar is **{available_models[model_col]}**."
    )