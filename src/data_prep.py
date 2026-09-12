"""
Stage 1 of the DVC Pipeline:
Loads raw labeled review data, executes text normalization, partitions into
stratified train/test splits, and computes ground-truth baseline statistics
(positive/negative class balance, text length distributions, vocabulary size)
stored in data/processed/baseline_stats.json for production drift monitoring.
"""
import hashlib
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    RAW_DATA_PATH,
    PROCESSED_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    BASELINE_STATS_PATH,
    clean_text,
    load_params,
    save_json,
)


def compute_file_hash(path: Path) -> str:
    """Computes SHA256 checksum for raw data tracking."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def prepare_data(
    raw_path: Path = RAW_DATA_PATH,
    output_dir: Path = PROCESSED_DIR,
    params_path: str = "params.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw review data, splits into train/test, and records training baseline stats.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    params = load_params(params_path)["data_prep"]
    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))

    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()

    if len(df) == 0:
        raise ValueError("No valid rows remaining after text cleaning.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    lengths = df["text"].str.split().apply(len)
    class_counts = df["label"].value_counts(normalize=True).to_dict()
    data_hash = compute_file_hash(raw_path)

    baseline_stats = {
        "dataset_hash": data_hash,
        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "text_length_mean": float(lengths.mean()),
        "text_length_std": float(lengths.std()),
        "text_length_p50": float(lengths.median()),
        "text_length_p90": float(lengths.quantile(0.90)),
        "text_lengths_sample": lengths.tolist(),  # Used for KS-test comparison
        "class_distribution": class_counts,
        "positive_rate": float(class_counts.get("positive", 0.5)),
        "negative_rate": float(class_counts.get("negative", 0.5)),
        "vocab_size_approx": int(df["text"].str.split().explode().nunique()),
    }

    save_json(baseline_stats, output_dir / "baseline_stats.json")
    return train_df, test_df, baseline_stats


def main() -> None:
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    train_df, test_df, stats = prepare_data()
    print(f"Data preparation complete: {len(train_df)} train rows, {len(test_df)} test rows.")
    print(f"Class balance: {stats['class_distribution']}")
    print(f"Baseline statistics saved to {BASELINE_STATS_PATH}")


if __name__ == "__main__":
    main()

