"""
Production Prediction Drift Monitoring Module:
Compares recent production prediction logs (data/prediction_logs.jsonl)
against the stored training-time baseline (data/processed/baseline_stats.json).

Metrics & Evaluation:
1. Prediction Distribution Drift (TVD / Absolute Positive Rate Shift):
   Delta P = |P_current(positive) - P_baseline(positive)|
   Flagged if Delta P > drift_threshold (configured at 0.03, i.e., 3.0%).
2. Feature Distribution Drift:
   Two-sample Kolmogorov-Smirnov test on text lengths (p-value < threshold).
3. Population Stability Index (PSI):
   PSI across predicted class distributions (PSI > threshold).

Outputs models/drift_report.json and exits with code 1 if drift exceeds threshold,
triggering the auto-retraining workflow in GitHub Actions.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp

from utils import (
    BASELINE_STATS_PATH,
    PREDICTION_LOGS_PATH,
    DRIFT_REPORT_PATH,
    load_json,
    save_json,
    load_params,
)


def compute_psi(baseline_dist: Dict[str, float], current_dist: Dict[str, float], eps: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) across category distributions:
    PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))
    """
    labels = set(baseline_dist.keys()) | set(current_dist.keys())
    score = 0.0
    for label in labels:
        b = float(baseline_dist.get(label, eps) or eps)
        c = float(current_dist.get(label, eps) or eps)
        score += (c - b) * np.log(c / b)
    return float(score)


def compute_drift(
    baseline_stats: Dict[str, Any],
    prediction_records: List[Dict[str, Any]],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates prediction distribution drift and statistical shift between baseline and current records.
    """
    drift_threshold = float(thresholds.get("drift_threshold", 0.03))
    min_samples = int(thresholds.get("min_predictions_for_check", 20))
    ks_pvalue_threshold = float(thresholds.get("text_length_ks_pvalue_threshold", 0.05))
    psi_threshold = float(thresholds.get("class_balance_psi_threshold", 0.2))

    n_records = len(prediction_records)
    if n_records < min_samples:
        return {
            "status": "insufficient_data",
            "n_predictions_checked": n_records,
            "min_predictions_required": min_samples,
            "drift_detected": False,
            "action": "continue_monitoring",
            "message": f"Only {n_records} predictions logged (need {min_samples}). Skipping drift check.",
        }

    # 1. Prediction Class Distribution & 3% Threshold
    counts: Dict[str, int] = {}
    for r in prediction_records:
        label = str(r["predicted_label"])
        counts[label] = counts.get(label, 0) + 1

    current_dist = {k: v / n_records for k, v in counts.items()}
    baseline_dist = baseline_stats["class_distribution"]

    # Calculate absolute delta on positive rate (Total Variation Distance for binary classes)
    baseline_pos = float(baseline_dist.get("positive", 0.5))
    current_pos = float(current_dist.get("positive", 0.0))
    prediction_drift_value = abs(current_pos - baseline_pos)
    prediction_drift_flagged = bool(prediction_drift_value > drift_threshold)

    # 2. Input Feature (Text Length) Kolmogorov-Smirnov Test
    current_lengths = [int(r["text_length"]) for r in prediction_records if "text_length" in r]
    baseline_lengths = baseline_stats.get("text_lengths_sample", [])

    if baseline_lengths and current_lengths:
        ks_stat, ks_pvalue = ks_2samp(baseline_lengths, current_lengths)
        ks_stat = float(ks_stat)
        ks_pvalue = float(ks_pvalue)
        length_drift_flagged = bool(ks_pvalue < ks_pvalue_threshold)
    else:
        ks_stat, ks_pvalue = 0.0, 1.0
        length_drift_flagged = False

    # 3. Population Stability Index (PSI)
    psi_score = compute_psi(baseline_dist, current_dist)
    psi_drift_flagged = bool(psi_score > psi_threshold)

    # Combined drift trigger: primary trigger is prediction drift > 3% threshold
    drift_detected = bool(prediction_drift_flagged or length_drift_flagged or psi_drift_flagged)
    action = "trigger_retraining" if drift_detected else "continue_monitoring"

    report = {
        "timestamp": time.time(),
        "n_predictions_checked": n_records,
        "drift_threshold": drift_threshold,
        "drift_metric": "prediction_distribution_shift_tvd",
        "prediction_drift_value": float(round(prediction_drift_value, 4)),
        "prediction_drift_flagged": prediction_drift_flagged,
        "class_distribution_baseline": baseline_dist,
        "class_distribution_current": current_dist,
        "text_length_ks_statistic": ks_stat,
        "text_length_ks_pvalue": ks_pvalue,
        "text_length_ks_threshold": ks_pvalue_threshold,
        "text_length_drift_flagged": length_drift_flagged,
        "psi_score": float(round(psi_score, 4)),
        "psi_threshold": psi_threshold,
        "psi_drift_flagged": psi_drift_flagged,
        "drift_detected": drift_detected,
        "action": action,
    }
    return report


def check_drift(
    baseline_path: Path = BASELINE_STATS_PATH,
    logs_path: Path = PREDICTION_LOGS_PATH,
    report_path: Path = DRIFT_REPORT_PATH,
    params_path: str = "params.yaml",
) -> Tuple[Dict[str, Any], int]:
    """
    Executes the drift evaluation workflow, saves drift_report.json, and returns (report, exit_code).
    """
    if not baseline_path.exists():
        print(f"Error: Baseline statistics file not found at {baseline_path}. Run data_prep.py first.")
        return {"error": "Baseline file missing"}, 2

    baseline_stats = load_json(baseline_path)
    thresholds = load_params(params_path)["drift"]

    if not logs_path.exists():
        print("No prediction logs found yet at data/prediction_logs.jsonl.")
        report = {
            "n_predictions_checked": 0,
            "drift_detected": False,
            "action": "continue_monitoring",
            "message": "No prediction logs found.",
        }
        return report, 0

    with open(logs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    report = compute_drift(baseline_stats, records, thresholds)
    save_json(report, report_path)

    print("=== Drift Monitoring Report ===")
    print(json.dumps(report, indent=2))

    if report.get("drift_detected", False):
        print(f"\n[ALERT] Drift detected! Action: {report['action']}. Exiting with code 1.")
        return report, 1
    else:
        print(f"\n[OK] Model healthy. Drift within configured thresholds. Action: {report['action']}.")
        return report, 0


def main() -> None:
    _, exit_code = check_drift()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

