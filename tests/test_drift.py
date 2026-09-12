import json
import pytest
from pathlib import Path

from monitor_drift import compute_psi, compute_drift, check_drift
from utils import BASELINE_STATS_PATH, load_json


@pytest.fixture
def mock_baseline_stats():
    return {
        "n_samples": 1000,
        "positive_rate": 0.50,
        "negative_rate": 0.50,
        "class_distribution": {"positive": 0.50, "negative": 0.50},
        "text_length_mean": 200.0,
        "text_length_std": 50.0,
        "text_lengths_sample": [150, 180, 200, 220, 250] * 20,
    }


@pytest.fixture
def mock_thresholds():
    return {
        "drift_threshold": 0.03,
        "min_predictions_for_check": 20,
        "text_length_ks_pvalue_threshold": 0.05,
        "class_balance_psi_threshold": 0.20,
    }


def test_psi_identical_distributions():
    dist = {"positive": 0.5, "negative": 0.5}
    score = compute_psi(dist, dist)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_psi_divergent_distributions():
    base = {"positive": 0.5, "negative": 0.5}
    curr = {"positive": 0.9, "negative": 0.1}
    score = compute_psi(base, curr)
    assert score > 0.3


def test_drift_insufficient_samples(mock_baseline_stats, mock_thresholds):
    records = [{"predicted_label": "positive", "text_length": 200}] * 10
    report = compute_drift(mock_baseline_stats, records, mock_thresholds)
    assert report["status"] == "insufficient_data"
    assert report["drift_detected"] is False
    assert report["action"] == "continue_monitoring"


def test_drift_normal_in_distribution(mock_baseline_stats, mock_thresholds):
    # 50% positive, 50% negative + representative text lengths -> Delta P = 0.0 <= 0.03, KS p-val > 0.05
    sample_lens = [150, 180, 200, 220, 250]
    records = [
        {"predicted_label": "positive", "text_length": sample_lens[i % 5], "confidence": 0.9}
        for i in range(25)
    ] + [
        {"predicted_label": "negative", "text_length": sample_lens[i % 5], "confidence": 0.9}
        for i in range(25)
    ]
    report = compute_drift(mock_baseline_stats, records, mock_thresholds)
    assert report["drift_detected"] is False
    assert report["prediction_drift_flagged"] is False
    assert report["prediction_drift_value"] == 0.0
    assert report["action"] == "continue_monitoring"


def test_drift_above_3_percent_threshold(mock_baseline_stats, mock_thresholds):
    # 60% positive, 40% negative -> Delta P = |0.60 - 0.50| = 0.10 (10%) > 0.03 (3%)
    records = [
        {"predicted_label": "positive", "text_length": 200, "confidence": 0.9}
        for _ in range(30)
    ] + [
        {"predicted_label": "negative", "text_length": 200, "confidence": 0.9}
        for _ in range(20)
    ]
    report = compute_drift(mock_baseline_stats, records, mock_thresholds)
    assert report["prediction_drift_value"] == pytest.approx(0.10, abs=1e-3)
    assert report["prediction_drift_flagged"] is True
    assert report["drift_detected"] is True
    assert report["action"] == "trigger_retraining"


def test_drift_boundary_condition(mock_baseline_stats, mock_thresholds):
    # 54% positive, 46% negative -> Delta P = |0.54 - 0.50| = 0.04 (4.0%) > 0.03 (3.0%)
    records = [
        {"predicted_label": "positive", "text_length": 200, "confidence": 0.9}
        for _ in range(27)
    ] + [
        {"predicted_label": "negative", "text_length": 200, "confidence": 0.9}
        for _ in range(23)
    ]
    report = compute_drift(mock_baseline_stats, records, mock_thresholds)
    assert report["prediction_drift_value"] == pytest.approx(0.04, abs=1e-3)
    assert report["prediction_drift_flagged"] is True
    assert report["drift_detected"] is True


def test_check_drift_missing_baseline(tmp_path):
    missing_path = tmp_path / "nonexistent_baseline.json"
    _, exit_code = check_drift(baseline_path=missing_path)
    assert exit_code == 2
