import tempfile
from pathlib import Path
import pandas as pd
import pytest

from utils import clean_text
from data_prep import compute_file_hash, prepare_data


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("This Movie WAS Great!!!") == "this movie was great"


def test_clean_text_collapses_whitespace():
    assert clean_text("too   many    spaces") == "too many spaces"


def test_clean_text_handles_numbers():
    assert clean_text("Rated 10/10 would watch again") == "rated 10 10 would watch again"


def test_clean_text_handles_none_and_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   !!! ???  ") == ""


def test_compute_file_hash(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello world", encoding="utf-8")
    hash_val = compute_file_hash(sample_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA256 hex length


def test_prepare_data_workflow(tmp_path):
    raw_csv = tmp_path / "raw_reviews.csv"
    out_dir = tmp_path / "processed"
    
    # Create sample raw dataset with 20 balanced rows
    data = {
        "text": [f"This is good review number {i}" for i in range(10)] + [f"This is bad review number {i}" for i in range(10)],
        "label": ["positive"] * 10 + ["negative"] * 10,
    }
    pd.DataFrame(data).to_csv(raw_csv, index=False)

    train_df, test_df, baseline_stats = prepare_data(
        raw_path=raw_csv,
        output_dir=out_dir,
    )

    assert len(train_df) == 16
    assert len(test_df) == 4
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert (out_dir / "baseline_stats.json").exists()

    assert baseline_stats["n_samples"] == 20
    assert baseline_stats["positive_rate"] == 0.5
    assert baseline_stats["negative_rate"] == 0.5
    assert "text_lengths_sample" in baseline_stats
    assert len(baseline_stats["text_lengths_sample"]) == 20

