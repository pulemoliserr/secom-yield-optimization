"""
SECOM Semiconductor Yield Optimization Engine
-----------------------------------------------
Interactive dashboard for exploring how the classification threshold and
the Type I / Type II cost assumptions change a model's confusion matrix,
recall, and total business cost -- without retraining anything.

Run with:
    streamlit run app.py

Expects a CSV called `secom_test_predictions.csv` next to this file, with
one row per TEST-set wafer and columns:
    y_test     -> true label, 0 = pass, 1 = defect
    proba_rf   -> Tuned Random Forest's predicted P(defect) for that wafer
    proba_xgb  -> XGBoost's predicted P(defect) for that wafer          (optional)

Optionally also expects a `secom_model_params.json` file next to this file,
describing each model's hyperparameters (see the export snippet at the
bottom of this file). Without it, the app shows clearly-labelled demo
values so the "Model Configuration" section is still explorable.

If secom_test_predictions.csv isn't found, the app falls back to a small
synthetic demo dataset so the UI is still fully explorable.
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

# Clearly-labelled placeholder hyperparameters, used only if secom_model_params.json
# isn't found next to this file -- see load_model_params() and the export snippet
# at the bottom of this file for how to generate the real one from your notebook.
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
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data
def make_demo_predictions(seed: int = 42, n: int = 314, defect_rate: float = 0.0669) -> pd.DataFrame:
    """Synthetic stand-in so the dashboard is explorable before you've
    exported real predictions. Shaped like the SECOM test partition
    (roughly 6.7% defect rate) with imperfect-but-informative scores."""
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
    """One row per parameter, one column per available model, in the order
    each parameter first appears (Algorithm/architecture fields first)."""
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
# Cost / metric helpers
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
    """Total cost at every candidate threshold for ONE model -- used to plot
    multiple models' curves on the same chart."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = np.array([metrics_at(y_true, proba, t, cost_fp, cost_fn)["cost"] for t in grid])
    return grid, costs


def sweep_thresholds(y_true: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float,
                      min_recall: float = 0.0):
    """Total cost at every candidate threshold, plus the cheapest threshold
    (a) unconstrained and (b) subject to a minimum-recall SLA."""
    grid = np.linspace(0.01, 0.99, 99)
    rows = [metrics_at(y_true, proba, t, cost_fp, cost_fn) for t in grid]
    costs = np.array([r["cost"] for r in rows])
    recalls = np.array([r["recall"] for r in rows])

    best_idx = int(np.argmin(costs))

    feasible = np.where(recalls >= min_recall)[0]
    if len(feasible) > 0:
        best_feasible_idx = feasible[np.argmin(costs[feasible])]
    else:
        best_feasible_idx = best_idx  # no threshold meets the SLA; fall back

    return grid, costs, recalls, grid[best_idx], grid[best_feasible_idx]


# --------------------------------------------------------------------------- #
# Sidebar controls
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
        "recall) but raises false alarms; raising it does the reverse. "
        "There is no threshold that improves both at once -- the point below "
        "is chosen by minimizing total cost, not by eye."
    )

    st.divider()
    st.markdown("**Cost Parameters**")
    cost_fp = st.number_input("Cost per False Alarm (Type I)", min_value=1, value=100, step=10)
    cost_fn = st.number_input("Cost per Missed Defect (Type II)", min_value=1, value=400, step=10)

    st.divider()
    st.markdown("**Business Constraint**")
    min_recall_pct = st.slider("Minimum Required Recall (SLA)", 0, 100, 0, 5,
                                help="Restrict the 'cheapest threshold' search to thresholds that still "
                                     "catch at least this share of true defects.")

    st.divider()
    st.caption(f"Data source: {data_source_note}")


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
y_true = df["y_test"].to_numpy()
proba = df[model_col].to_numpy()

current = metrics_at(y_true, proba, threshold, cost_fp, cost_fn)
grid, costs, recalls, best_t, best_t_sla = sweep_thresholds(
    y_true, proba, cost_fp, cost_fn, min_recall=min_recall_pct / 100
)
champion = metrics_at(y_true, proba, best_t, cost_fp, cost_fn)
champion_sla = metrics_at(y_true, proba, best_t_sla, cost_fp, cost_fn)

# Cost curve + current-threshold cost for EVERY available model, not just the
# one selected in the sidebar -- this is what powers the comparison chart.
model_curves = {}
for col, label in available_models.items():
    p = df[col].to_numpy()
    g, c = cost_curve(y_true, p, cost_fp, cost_fn)
    at_current = metrics_at(y_true, p, threshold, cost_fp, cost_fn)
    model_curves[col] = {"label": label, "grid": g, "costs": c, "cost_at_current": at_current["cost"]}

# Whichever model is cheaper AT THE CURRENT THRESHOLD -- recomputed on every
# slider move, so this label tracks the live comparison rather than a fixed,
# one-time winner.
cheapest_col = min(model_curves, key=lambda c: model_curves[c]["cost_at_current"])


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🏭 SECOM Semiconductor Yield Optimization Engine")
st.caption(
    "Operational evaluation dashboard driven by cached out-of-sample predictions -- "
    "moving the threshold or cost parameters recomputes everything below instantly, "
    "with no retraining involved."
)
st.markdown("---")

with st.expander("🔧 Model Configuration \u2014 Hyperparameters & Architecture", expanded=True):
    params_df = build_params_table(model_params, available_models)
    st.dataframe(params_df, use_container_width=True)
    st.caption(
        f"Data source: {params_source_note}. `n_estimators`, `max_depth`, and "
        f"`min_samples_leaf` for the Random Forest -- and the equivalent XGBoost "
        f"parameters -- were selected via cross-validated `RandomizedSearchCV` "
        f"scored on Average Precision, not chosen by hand. The currently selected "
        f"model in the sidebar is **{available_models[model_col]}**."
    )

left, right = st.columns([2.1, 1])

with left:
    st.subheader(f"⚔️ Interactive Arena: {available_models[model_col]}")

    m1, m2 = st.columns(2)
    delta_cost = current["cost"] - champion["cost"]
    m1.metric(
        "Simulated Business Cost", f"${current['cost']:,}",
        delta=f"${delta_cost:+,} vs cost-optimal threshold", delta_color="inverse",
    )
    m2.metric("Recall (Defect Catch Rate)", f"{current['recall']*100:.1f}%")

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

    # Live cost of each model at exactly the current threshold -- updates on every slider move.
    cost_cols = st.columns(len(model_curves))
    for i, (col, mc) in enumerate(model_curves.items()):
        is_cheapest = col == cheapest_col and len(model_curves) > 1
        cost_cols[i].metric(
            f"{mc['label']} cost @ t={threshold:.2f}",
            f"${mc['cost_at_current']:,}",
            delta="cheaper" if is_cheapest else None,
            delta_color="normal",
        )

    with st.expander("📊 Confusion matrix for the selected model", expanded=False):
        cm = [[current["tn"], current["fp"]], [current["fn"], current["tp"]]]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Defect"],
            y=["Actual Normal", "Actual Defect"],
            colorscale=[[0, "#EAF0FB"], [1, "#1E2761"]],
            showscale=False, text=cm, texttemplate="%{text}", textfont={"size": 22},
        ))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
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
st.markdown("**Detailed metrics at the current threshold**")
st.table(pd.DataFrame({
    "Metric": ["Recall", "Precision", "Specificity", "F1 Score", "Total cost"],
    "Value": [
        f"{current['recall']*100:.1f}%", f"{current['precision']*100:.1f}%",
        f"{current['specificity']*100:.1f}%", f"{current['f1']*100:.1f}%",
        f"${current['cost']:,}",
    ],
}).set_index("Metric"))


# --------------------------------------------------------------------------- #
# HOW TO EXPORT secom_model_params.json FROM YOUR NOTEBOOK
# --------------------------------------------------------------------------- #
# Add this to the corrected pipeline notebook, after Sections 15/16 have
# produced `best_rf_pipeline` and `xgb_pipeline`, alongside the existing
# secom_test_predictions.csv export -- then drop the resulting JSON file
# next to app.py:
#
#   import json
#
#   def safe_get_params(pipeline, step_name, keys):
#       if step_name not in pipeline.named_steps:
#           return {}
#       p = pipeline.named_steps[step_name].get_params()
#       return {k: p.get(k) for k in keys if k in p}
#
#   rf_params = {"Algorithm": "Random Forest (scikit-learn)"}
#   rf_params.update(safe_get_params(best_rf_pipeline, "rf", [
#       "n_estimators", "max_depth", "min_samples_split",
#       "min_samples_leaf", "max_features", "random_state",
#   ]))
#   rf_params.update(safe_get_params(best_rf_pipeline, "impute", ["n_neighbors"]))
#   if "winsorize" in best_rf_pipeline.named_steps:
#       w = best_rf_pipeline.named_steps["winsorize"]
#       rf_params["winsorize_lower_quantile"] = w.lower_quantile
#       rf_params["winsorize_upper_quantile"] = w.upper_quantile
#
#   xgb_params = {"Algorithm": "XGBoost"}
#   xgb_params.update(safe_get_params(xgb_pipeline, "xgb", [
#       "n_estimators", "max_depth", "learning_rate",
#       "subsample", "colsample_bytree", "random_state",
#   ]))
#
#   with open("secom_model_params.json", "w") as f:
#       json.dump({"proba_rf": rf_params, "proba_xgb": xgb_params}, f, indent=2)
#
# `safe_get_params` skips a pipeline step entirely if it isn't present, so
# this works whether Section 13's extended search (with its own winsorizer
# and tunable imputer) was adopted or the simpler Section 12 pipeline was
# kept -- no manual editing required either way.
