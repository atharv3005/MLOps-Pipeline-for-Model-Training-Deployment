"""
Automated Retraining and Model Revalidation Module:
Triggered by scheduled drift detection (or CI/CD workflow) when prediction drift > 3%.

Workflow:
1. Snapshot current production champion artifacts and baseline metrics.
2. Re-prepare data and train new candidate model.
3. Evaluate candidate model on hold-out test split.
4. Gate: Compare candidate metrics against:
   a. Minimum accuracy threshold (e.g. 0.70)
   b. Current production champion accuracy (Acc_cand >= Acc_champ)
5. Promotion / Rollback Safety:
   - PASS: Promote candidate to production artifacts (models/model.pkl, models/vectorizer.pkl),
           update models/best_metrics.json, update MLflow Model Registry, generate retrain_report.json.
   - FAIL: Preserve active production model, reject candidate, generate retrain_report.json,
           and exit with code 1 (preventing deployment of degraded model).
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_prep import prepare_data
from train import train_model
from utils import (
    MODELS_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    BEST_METRICS_PATH,
    RETRAIN_REPORT_PATH,
    load_json,
    save_json,
    load_params,
)


def run_retrain_pipeline(
    params_path: str = "params.yaml",
) -> Tuple[Dict[str, Any], int]:
    """
    Executes end-to-end retraining, revalidation gating, and atomic promotion or rollback.
    """
    params = load_params(params_path)
    min_accuracy_threshold = float(params["train"].get("min_accuracy_threshold", 0.70))
    registered_model_name = params["train"].get("registered_model_name", "sentiment-classifier")

    # 1. Retrieve current champion metrics
    champion_accuracy = 0.0
    if BEST_METRICS_PATH.exists():
        champion_accuracy = float(load_json(BEST_METRICS_PATH).get("accuracy", 0.0))
    elif METRICS_PATH.exists():
        champion_accuracy = float(load_json(METRICS_PATH).get("accuracy", 0.0))

    print(f"Current Production Champion Accuracy: {champion_accuracy:.4f}")
    print(f"Minimum Deployment Threshold:         {min_accuracy_threshold:.4f}")

    # 2. Backup current production artifacts if present
    temp_backup_dir = Path(tempfile.mkdtemp(prefix="champion_backup_"))
    has_champion = MODEL_PATH.exists() and VECTORIZER_PATH.exists()
    if has_champion:
        shutil.copy(MODEL_PATH, temp_backup_dir / "model.pkl")
        shutil.copy(VECTORIZER_PATH, temp_backup_dir / "vectorizer.pkl")

    try:
        # 3. Re-run data preparation
        print("Re-running data preparation stage...")
        prepare_data()

        # 4. Train candidate model
        print("Training candidate model with MLflow experiment tracking...")
        clf, vectorizer, candidate_metrics, run_id = train_model(save_local=False)
        cand_acc = candidate_metrics["accuracy"]

        print(f"Candidate Model Accuracy: {cand_acc:.4f} (Run ID: {run_id})")

        # 5. Validation Gate Evaluation
        passes_min_threshold = cand_acc >= min_accuracy_threshold
        beats_champion = cand_acc >= champion_accuracy
        validation_passed = bool(passes_min_threshold and beats_champion)

        report = {
            "timestamp": time.time(),
            "trigger": "scheduled_drift_or_manual",
            "run_id": run_id,
            "min_accuracy_threshold": min_accuracy_threshold,
            "champion_accuracy": champion_accuracy,
            "candidate_accuracy": cand_acc,
            "candidate_metrics": candidate_metrics,
            "passes_min_threshold": passes_min_threshold,
            "beats_champion": beats_champion,
            "validation_passed": validation_passed,
        }

        if validation_passed:
            print(f"\n[PROMOTION] Candidate passed all validation gates ({cand_acc:.4f} >= {champion_accuracy:.4f}).")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, MODEL_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)
            save_json(candidate_metrics, METRICS_PATH)
            save_json(candidate_metrics, BEST_METRICS_PATH)

            # MLflow Model Registry update
            try:
                client = mlflow.tracking.MlflowClient()
                versions = client.search_model_versions(f"name='{registered_model_name}'")
                if versions:
                    print(f"Setting MLflow model version {latest_version} as champion / Production...")
                    try:
                        client.set_registered_model_alias(
                            name=registered_model_name,
                            alias="champion",
                            version=str(latest_version),
                        )
                    except Exception:
                        pass
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            client.transition_model_version_stage(
                                name=registered_model_name,
                                version=latest_version,
                                stage="Production",
                                archive_existing_versions=True,
                            )
                    except Exception:
                        pass
            except Exception as e:
                print(f"MLflow client registry transition notice: {e}")

            report["status"] = "PROMOTED"
            report["action"] = "redeploy_docker_container"
            save_json(report, RETRAIN_REPORT_PATH)
            print(f"Retrain report written to {RETRAIN_REPORT_PATH}")
            return report, 0
        else:
            print(f"\n[REJECTED] Candidate failed validation gate. Preserving current production model.")
            if has_champion:
                shutil.copy(temp_backup_dir / "model.pkl", MODEL_PATH)
                shutil.copy(temp_backup_dir / "vectorizer.pkl", VECTORIZER_PATH)

            report["status"] = "REJECTED"
            report["action"] = "retain_current_production_model"
            report["rejection_reason"] = (
                f"Candidate accuracy ({cand_acc:.4f}) failed criteria "
                f"(min threshold: {min_accuracy_threshold:.4f}, champion: {champion_accuracy:.4f})"
            )
            save_json(report, RETRAIN_REPORT_PATH)
            print(f"Retrain report written to {RETRAIN_REPORT_PATH}")
            return report, 1

    finally:
        shutil.rmtree(temp_backup_dir, ignore_errors=True)


def main() -> None:
    _, exit_code = run_retrain_pipeline()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

