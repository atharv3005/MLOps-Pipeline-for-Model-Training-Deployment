import json
import pytest
from pathlib import Path

from retrain import run_retrain_pipeline
from utils import BEST_METRICS_PATH, METRICS_PATH, RETRAIN_REPORT_PATH, load_json, save_json


def test_retrain_pipeline_pass_and_promotion(tmp_path):
    report, exit_code = run_retrain_pipeline()
    assert exit_code == 0
    assert report["status"] == "PROMOTED"
    assert report["validation_passed"] is True
    assert report["candidate_accuracy"] >= report["min_accuracy_threshold"]
    assert RETRAIN_REPORT_PATH.exists()

    retrain_data = load_json(RETRAIN_REPORT_PATH)
    assert retrain_data["status"] == "PROMOTED"


def test_retrain_pipeline_rejection_guardrail(monkeypatch):
    # Simulate a champion model with artificially high accuracy (e.g., 0.9999)
    # The new candidate model (~0.86) should fail the gate and be REJECTED
    original_metrics = load_json(BEST_METRICS_PATH) if BEST_METRICS_PATH.exists() else {"accuracy": 0.86}
    
    try:
        save_json({"accuracy": 0.9999}, BEST_METRICS_PATH)
        report, exit_code = run_retrain_pipeline()
        
        assert exit_code == 1
        assert report["status"] == "REJECTED"
        assert report["validation_passed"] is False
        assert report["action"] == "retain_current_production_model"
        assert "rejection_reason" in report
    finally:
        # Restore original champion metrics
        save_json(original_metrics, BEST_METRICS_PATH)
