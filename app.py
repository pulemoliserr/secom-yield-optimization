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
    min_recall_pct = st.slider("Minimum Required Recall (SLA)", 0, 100, 0, 5,
                                help="Restrict the 'cheapest threshold' search to thresholds that still catch at least this share of true defects.")

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

for col, label in available_models.items():
    p = df[col].to_numpy()
    g, c = cost_curve(y_true, p, cost_fp, cost_fn)
    at_current = metrics_at(y_true, p, threshold, cost_fp, cost_fn)
    model_curves[col] = {"label": label, "grid": g, "costs": c, "cost_at_current": at_current["cost"]}
    all_metrics_at_current[col] = at_current

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
    st.subheader("📈 Financial Trajectory Matrix: Cost vs. Threshold Sweep")
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
        height=420,
        xaxis_title="Decision Threshold Boundary", yaxis_title="Total Financial Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
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
        <h4 style="margin-top:0;">🏆 Production Champion</h4>
        <p style="font-size:0.9rem;">Cost-minimizing threshold found by sweeping this
        model's probabilities against your current cost parameters.</p>
        <ul style="font-size:0.9rem;">
            <li><b>Champion cost:</b> ${champion['cost']:,}</li>
            <li><b>Champion recall:</b> {champion['recall']*100:.1f}%</li>
            <li><b>Cost-optimized threshold:</b> t = {best_t:.3f}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if min_recall_pct > 0:
        st.markdown(
            f"""
            <div style="background:#EAF6F2;padding:1.1rem 1.3rem;border-radius:0.6rem;
                        border-left:5px solid #00A896;margin-top:0.8rem;">
            <h4 style="margin-top:0;">🎯 Cheapest at {min_recall_pct}% Recall SLA</h4>
            <ul style="font-size:0.9rem;">
                <li><b>Cost:</b> ${champion_sla['cost']:,}</li>
                <li><b>Recall achieved:</b> {champion_sla['recall']*100:.1f}%</li>
                <li><b>Threshold:</b> t = {best_t_sla:.3f}</li>
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

# Side-by-Side Detailed Metrics Comparison Table
st.markdown(f"**Detailed metrics comparison at t = {threshold:.2f}**")

metrics_table_data = {"Metric": ["Recall", "Precision", "Specificity", "F1 Score", "Total Cost"]}
for col, label in available_models.items():
    m = all_metrics_at_current[col]
    metrics_table_data[label] = [
        f"{m['recall']*100:.1f}%",
        f"{m['precision']*100:.1f}%",
        f"{m['specificity']*100:.1f}%",
        f"{m['f1']*100:.1f}%",
        f"${m['cost']:,}",
    ]

st.table(pd.DataFrame(metrics_table_data).set_index("Metric"))

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