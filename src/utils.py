import json
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "reviews.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"
BASELINE_STATS_PATH = PROCESSED_DIR / "baseline_stats.json"
PREDICTION_LOGS_PATH = DATA_DIR / "prediction_logs.jsonl"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
BEST_METRICS_PATH = MODELS_DIR / "best_metrics.json"
DRIFT_REPORT_PATH = MODELS_DIR / "drift_report.json"
RETRAIN_REPORT_PATH = MODELS_DIR / "retrain_report.json"


def load_params(path: Any = "params.yaml") -> Dict[str, Any]:
    """Loads and returns parameters from params.yaml or specified path."""
    p = Path(path)
    param_file = p if p.is_absolute() else ROOT / p
    if not param_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {param_file}")
    with open(param_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: Optional[str]) -> str:
    """
    Standard text normalization pipeline applied consistently across training,
    evaluation, drift testing, and inference to eliminate train/serve skew.
    - Converts to lowercase string
    - Strips non-alphanumeric characters (preserves spaces)
    - Collapses multiple whitespace characters to single spaces
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(path: Path) -> Dict[str, Any]:
    """Helper to safely read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Helper to safely write JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

