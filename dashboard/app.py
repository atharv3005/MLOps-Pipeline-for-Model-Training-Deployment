"""
Streamlit MLOps Monitoring Dashboard:
Displays:
1. Current Production Model Metrics & Model Registry lifecycle status.
2. MLflow Run History, parameters, and accuracy progression over runs.
3. Continuous Prediction Drift Monitor with 3% threshold visualizer, baseline vs. production distribution comparisons, KS p-value, and PSI scores.
4. Auto-Retraining and Revalidation Audit logs.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import (
    METRICS_PATH,
    BEST_METRICS_PATH,
    DRIFT_REPORT_PATH,
    RETRAIN_REPORT_PATH,
    BASELINE_STATS_PATH,
)

EXPERIMENT_NAME = "sentiment-classifier"

st.set_page_config(page_title="End-to-End MLOps Pipeline Dashboard", layout="wide")
st.title("Sentiment Classification — End-to-End MLOps Dashboard")

# ---------------------------------------------------------------- Sidebar --
st.sidebar.header("Configuration & Tracking")
tracking_uri = st.sidebar.text_input(
    "MLflow Tracking URI",
    value=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
)

# ------------------------------------------------------- Model Status ----
st.header("1. Production Model & Baseline")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Active Model Metrics")
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.upper(), f"{value:.4f}")
    else:
        st.info("No metrics.json found — run `python src/train.py` first.")

with col_b:
    st.subheader("Training-Time Baseline")
    if BASELINE_STATS_PATH.exists():
        with open(BASELINE_STATS_PATH) as f:
            b_stats = json.load(f)
        st.write(
            f"**Samples:** {b_stats.get('n_samples', 'N/A')} | "
            f"**Positive Rate:** {b_stats.get('positive_rate', 0.5):.1%} | "
            f"**Avg Length:** {b_stats.get('text_length_mean', 0.0):.1f} words"
        )
        st.caption(f"Dataset SHA256: `{b_stats.get('dataset_hash', 'N/A')[:16]}...`")
    else:
        st.info("No baseline_stats.json found — run `python src/data_prep.py` first.")

# --------------------------------------------------------- Drift Monitor --
st.header("2. Scheduled Drift Monitoring (3% Threshold)")
if DRIFT_REPORT_PATH.exists():
    with open(DRIFT_REPORT_PATH) as f:
        report = json.load(f)

    is_drift = report.get("drift_detected", False)
    if is_drift:
        st.error(f"STATUS: DRIFT DETECTED — Action: {report.get('action', 'trigger_retraining').upper()}")
    else:
        st.success(f"STATUS: HEALTHY — Action: {report.get('action', 'continue_monitoring').upper()}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        drift_val = report.get("prediction_drift_value", 0.0)
        st.metric("Prediction Drift (|Delta P|)", f"{drift_val:.2%}")
        st.caption("3.0% Threshold Gate")
    with m2:
        st.metric("Drift Threshold", f"{report.get('drift_threshold', 0.03):.1%}")
        st.caption("Configured in params.yaml")
    with m3:
        ks_pval = report.get("text_length_ks_pvalue", 1.0)
        st.metric("Text Length KS p-value", f"{ks_pval:.4f}")
        st.caption("Threshold: < 0.05 indicates input drift")
    with m4:
        psi = report.get("psi_score", 0.0)
        st.metric("Class Balance PSI Score", f"{psi:.4f}")
        st.caption("Threshold: > 0.20 indicates population drift")

    st.subheader("Class Distribution: Baseline vs. Current Production Window")
    if "class_distribution_baseline" in report and "class_distribution_current" in report:
        dist_df = pd.DataFrame(
            {
                "Baseline (Training)": report["class_distribution_baseline"],
                "Current (Production)": report["class_distribution_current"],
            }
        ).fillna(0)
        st.bar_chart(dist_df)
        st.caption(f"Evaluated on {report.get('n_predictions_checked', 0)} recent production prediction logs.")
else:
    st.info("No drift report found yet. Run `python src/simulate_drift.py` and `python src/monitor_drift.py`.")

# ------------------------------------------------------ Retraining & Revalidation -
st.header("3. Automated Retraining & Revalidation Audit")
if RETRAIN_REPORT_PATH.exists():
    with open(RETRAIN_REPORT_PATH) as f:
        retrain_rep = json.load(f)

    status_tag = retrain_rep.get("status", "UNKNOWN")
    (st.success if status_tag == "PROMOTED" else st.warning)(f"Last Retraining Status: **{status_tag}**")

    r1, r2, r3 = st.columns(3)
    r1.metric("Candidate Accuracy", f"{retrain_rep.get('candidate_accuracy', 0.0):.4f}")
    r2.metric("Champion Accuracy", f"{retrain_rep.get('champion_accuracy', 0.0):.4f}")
    r3.metric("Min Threshold Gate", f"{retrain_rep.get('min_accuracy_threshold', 0.70):.4f}")

    if "rejection_reason" in retrain_rep:
        st.caption(f"Rejection Rationale: {retrain_rep['rejection_reason']}")
else:
    st.info("No auto-retrain events recorded yet.")

# ------------------------------------------------------ MLflow Run History -
st.header("4. MLflow Experiment Tracking & Model Registry")
try:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        st.warning(f"No experiment named '{EXPERIMENT_NAME}' found at {tracking_uri}.")
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        if not runs:
            st.info("Experiment exists but has no logged runs yet.")
        else:
            rows = []
            for r in runs:
                row = {
                    "run_id": r.info.run_id[:8],
                    "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
                }
                row.update(r.data.metrics)
                row.update({f"param_{k}": v for k, v in r.data.params.items()})
                rows.append(row)
            df = pd.DataFrame(rows).sort_values("start_time")
            st.dataframe(df, use_container_width=True)

            if "accuracy" in df.columns:
                st.subheader("Accuracy Across Training Runs")
                st.line_chart(df.set_index("start_time")["accuracy"])

            # Model Registry
            try:
                versions = client.search_model_versions(f"name='{EXPERIMENT_NAME}'")
                if versions:
                    st.subheader("Model Registry Versions & Stages")
                    reg_df = pd.DataFrame(
                        [
                            {
                                "version": v.version,
                                "stage": v.current_stage,
                                "run_id": v.run_id[:8],
                                "last_updated": pd.to_datetime(v.last_updated_timestamp, unit="ms"),
                            }
                            for v in versions
                        ]
                    ).sort_values("version", ascending=False)
                    st.dataframe(reg_df, use_container_width=True)
            except Exception:
                pass
except Exception as e:
    st.warning(f"Could not connect to MLflow server at `{tracking_uri}` ({e}). Local fallback mode active.")

