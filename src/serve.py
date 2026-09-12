"""
Production Model Serving Layer:
FastAPI service exposing /health and /predict endpoints.
Loads validated champion model and vectorizer artifacts.
Logs every inference request (timestamp, text length, predicted label, confidence)
to data/prediction_logs.jsonl for continuous drift monitoring.
"""
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils import MODEL_PATH, VECTORIZER_PATH, PREDICTION_LOGS_PATH, clean_text

model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading and unloading model artifacts."""
    global model, vectorizer
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError(
            f"Model artifacts not found ({MODEL_PATH} or {VECTORIZER_PATH}). "
            "Run `python src/data_prep.py && python src/train.py` or `dvc repro` first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded production model and vectorizer from {MODEL_PATH}")
    yield


app = FastAPI(
    title="Sentiment Analysis Inference API",
    description="Production MLOps sentiment classification service with automated drift logging",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Review text to classify", min_length=1)


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment class ('positive' or 'negative')")
    confidence: float = Field(..., description="Prediction probability score (0.0 to 1.0)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


@app.get("/")
def root():
    return {
        "service": "Sentiment Analysis API",
        "status": "running",
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/health")
def health():
    is_ready = bool(model is not None and vectorizer is not None)
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    start_time = time.perf_counter()
    cleaned = clean_text(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty text after normalization.")

    X = vectorizer.transform([cleaned])
    label = str(model.predict(X)[0])
    proba = float(model.predict_proba(X).max())
    latency_ms = float(round((time.perf_counter() - start_time) * 1000.0, 3))

    # Append to asynchronous / persistent prediction log file for drift checks
    PREDICTION_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LOGS_PATH, "a", encoding="utf-8") as f:
        log_entry = {
            "timestamp": time.time(),
            "text_length": len(cleaned.split()),
            "predicted_label": label,
            "confidence": proba,
            "latency_ms": latency_ms,
        }
        f.write(json.dumps(log_entry) + "\n")

    return PredictResponse(
        label=label,
        confidence=proba,
        latency_ms=latency_ms,
    )