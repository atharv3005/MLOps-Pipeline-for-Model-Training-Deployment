import json
import pytest
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import MODEL_PATH, VECTORIZER_PATH, METRICS_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from train import train_model


@pytest.mark.skipif(
    not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists(),
    reason="Processed data files not found. Run data_prep.py first.",
)
def test_train_model_outputs_valid_artifacts():
    clf, vectorizer, metrics, run_id = train_model(save_local=True)
    
    assert isinstance(clf, LogisticRegression)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(run_id, str) and len(run_id) > 0
    
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0

    assert MODEL_PATH.exists()
    assert VECTORIZER_PATH.exists()
    assert METRICS_PATH.exists()

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["accuracy"] == metrics["accuracy"]


def test_train_accuracy_threshold_gate(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    params_file = tmp_path / "params.yaml"

    train_data = {
        "text": ["great positive review"] * 20 + ["bad negative review"] * 20,
        "label": ["positive"] * 20 + ["negative"] * 20,
    }
    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(train_data).to_csv(test_csv, index=False)

    params_content = """
train:
  max_features: 100
  ngram_range_min: 1
  ngram_range_max: 1
  C: 1.0
  max_iter: 100
  random_state: 42
  min_accuracy_threshold: 0.999
  experiment_name: test-sentiment
  registered_model_name: test-sentiment
"""
    params_file.write_text(params_content.strip(), encoding="utf-8")

    _, _, metrics, _ = train_model(
        train_path=train_csv,
        test_path=test_csv,
        params_path=str(params_file.resolve()),
        save_local=False,
    )
    assert metrics["accuracy"] >= 0.70
