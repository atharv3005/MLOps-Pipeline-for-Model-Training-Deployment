"""
Stage 2 of the DVC Pipeline:
Trains TF-IDF vectorizer + Logistic Regression sentiment classifier,
logs run hyperparameters, metrics, git/dataset metadata, and model artifacts to MLflow,
registers the candidate model into the MLflow Model Registry,
and exports local artifacts (models/model.pkl, models/vectorizer.pkl, models/metrics.json).

Validation Gate: Exits with non-zero status if test accuracy falls below min_accuracy_threshold.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    BASELINE_STATS_PATH,
    MODELS_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    load_params,
    load_json,
    save_json,
)


def train_model(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    params_path: str = "params.yaml",
    save_local: bool = True,
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, float], str]:
    """
    Trains vectorizer and classifier, evaluates metrics, logs to MLflow, and saves artifacts.
    """
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {train_path} or {test_path}. Run data_prep.py first."
        )

    all_params = load_params(params_path)
    p = all_params["train"]
    experiment_name = p.get("experiment_name", "sentiment-classifier")
    registered_model_name = p.get("registered_model_name", "sentiment-classifier")
    min_accuracy_threshold = float(p.get("min_accuracy_threshold", 0.70))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    vectorizer = TfidfVectorizer(
        max_features=int(p["max_features"]),
        ngram_range=(int(p["ngram_range_min"]), int(p["ngram_range_max"])),
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    clf = LogisticRegression(
        C=float(p["C"]),
        max_iter=int(p["max_iter"]),
        random_state=int(p["random_state"]),
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) == 2 else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, pos_label="positive")),
        "precision": float(precision_score(y_test, preds, pos_label="positive")),
        "recall": float(recall_score(y_test, preds, pos_label="positive")),
    }
    if probs is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test == "positive", probs))

    # Dataset version tag
    dataset_hash = "unknown"
    if BASELINE_STATS_PATH.exists():
        baseline_data = load_json(BASELINE_STATS_PATH)
        dataset_hash = baseline_data.get("dataset_hash", "unknown")

    # MLflow Tracking & Registry
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                "model_type": "LogisticRegression",
                "feature_extractor": "TfidfVectorizer",
                "dataset_hash": dataset_hash,
                "pipeline_stage": "train",
            }
        )
        mlflow.log_params(
            {
                "max_features": p["max_features"],
                "ngram_range_min": p["ngram_range_min"],
                "ngram_range_max": p["ngram_range_max"],
                "C": p["C"],
                "max_iter": p["max_iter"],
                "random_state": p["random_state"],
                "min_accuracy_threshold": min_accuracy_threshold,
            }
        )
        mlflow.log_metrics(metrics)

        try:
            # Register in MLflow Model Registry
            mlflow.sklearn.log_model(
                clf,
                name="model",
                registered_model_name=registered_model_name,
            )
        except Exception as e:
            # Fallback for local testing or MLmodel logging
            print(f"MLflow model registration note: {e}")
            mlflow.sklearn.log_model(clf, artifact_path="model")

        if save_local:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, MODEL_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)
            save_json(metrics, METRICS_PATH)

    print(f"MLflow Run ID: {run_id}")
    print(f"Model Evaluation Metrics: {json.dumps(metrics, indent=2)}")

    return clf, vectorizer, metrics, run_id


def main() -> None:
    p = load_params()["train"]
    min_accuracy_threshold = float(p.get("min_accuracy_threshold", 0.70))

    _, _, metrics, run_id = train_model()

    if metrics["accuracy"] < min_accuracy_threshold:
        print(
            f"VALIDATION FAILED: accuracy {metrics['accuracy']:.4f} is below "
            f"minimum threshold {min_accuracy_threshold:.4f}"
        )
        sys.exit(1)

    print(f"VALIDATION PASSED: accuracy {metrics['accuracy']:.4f} >= threshold {min_accuracy_threshold:.4f}")


if __name__ == "__main__":
    main()

