import json
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from utils import MODEL_PATH, VECTORIZER_PATH, PREDICTION_LOGS_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not VECTORIZER_PATH.exists(),
    reason="Model artifacts missing — run `python src/data_prep.py && python src/train.py` first",
)


@pytest.fixture
def client():
    from serve import app
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_returns_label_and_confidence(client):
    resp = client.post("/predict", json={"text": "I absolutely loved this, fantastic!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("positive", "negative")
    assert 0.0 <= body["confidence"] <= 1.0
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0.0


def test_predict_rejects_empty_text(client):
    resp = client.post("/predict", json={"text": "   !!! ???  "})
    assert resp.status_code == 400
    assert "Empty text" in resp.json()["detail"]


def test_predict_rejects_empty_payload(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_logs_to_prediction_file(client):
    test_text = "Logging verification test string"
    resp = client.post("/predict", json={"text": test_text})
    assert resp.status_code == 200

    assert PREDICTION_LOGS_PATH.exists()
    with open(PREDICTION_LOGS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) > 0
    last_log = json.loads(lines[-1])
    assert "timestamp" in last_log
    assert "text_length" in last_log
    assert "predicted_label" in last_log
    assert "confidence" in last_log

