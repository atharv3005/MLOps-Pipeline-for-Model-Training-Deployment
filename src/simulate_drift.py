"""
Simulate Prediction Traffic Utility:
Generates synthetic production inference logs in data/prediction_logs.jsonl
to reproducibly test and demonstrate the drift detection and auto-retraining workflows.

Scenarios:
1. normal: In-distribution traffic (|Delta P| <= 1.5% <= 3% threshold, KS p-value > 0.05).
2. drift: Out-of-distribution traffic (|Delta P| >= 25% > 3% threshold).
3. boundary: Controlled boundary traffic (|Delta P| = 4.0% > 3% threshold).

Usage:
    python src/simulate_drift.py --scenario normal --n 50
    python src/simulate_drift.py --scenario drift --n 50
    python src/simulate_drift.py --scenario boundary --n 50
"""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from utils import PREDICTION_LOGS_PATH, BASELINE_STATS_PATH, TEST_DATA_PATH, load_json


def generate_traffic(
    scenario: str = "normal",
    n: int = 50,
    output_path: Path = PREDICTION_LOGS_PATH,
    clear_existing: bool = True,
) -> None:
    """Generates synthetic prediction traffic adhering to the target scenario."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if clear_existing else "a"

    baseline_pos = 0.5
    baseline_lengths = []
    if BASELINE_STATS_PATH.exists():
        baseline_stats = load_json(BASELINE_STATS_PATH)
        baseline_pos = float(baseline_stats.get("positive_rate", 0.5))
        baseline_lengths = baseline_stats.get("text_lengths_sample", [])

    test_pos_texts = []
    test_neg_texts = []
    if TEST_DATA_PATH.exists():
        test_df = pd.read_csv(TEST_DATA_PATH)
        test_pos_texts = test_df[test_df["label"] == "positive"]["text"].tolist()
        test_neg_texts = test_df[test_df["label"] == "negative"]["text"].tolist()

    if not test_pos_texts:
        test_pos_texts = ["fantastic movie great direction wonderfully written " * 20]
    if not test_neg_texts:
        test_neg_texts = ["terrible film awful plot horrible acting waste of time " * 20]

    records = []
    current_time = time.time() - (n * 60)

    if scenario == "normal":
        # Balanced 50/50 distribution + sampled from test set -> Delta P = 0.0% <= 3%, KS p-val > 0.05
        n_pos = n // 2
        n_neg = n - n_pos
        labels = (["positive"] * n_pos) + (["negative"] * n_neg)
        random.seed(42)
        random.shuffle(labels)

        for i, label in enumerate(labels):
            if label == "positive":
                text = random.choice(test_pos_texts)
            else:
                text = random.choice(test_neg_texts)
            conf = random.uniform(0.75, 0.96)
            records.append({
                "timestamp": current_time + (i * 60),
                "text_length": len(text.split()),
                "predicted_label": label,
                "confidence": float(round(conf, 4)),
            })

    elif scenario == "drift":
        # Skewed 80% positive distribution -> Delta P = |0.80 - 0.50| = 30% > 3%
        n_pos = int(n * 0.80)
        n_neg = n - n_pos
        labels = (["positive"] * n_pos) + (["negative"] * n_neg)
        random.shuffle(labels)

        for i, label in enumerate(labels):
            if label == "positive":
                text = random.choice(test_pos_texts)
            else:
                text = random.choice(test_neg_texts)
            conf = random.uniform(0.80, 0.98)
            records.append({
                "timestamp": current_time + (i * 60),
                "text_length": len(text.split()),
                "predicted_label": label,
                "confidence": float(round(conf, 4)),
            })

    elif scenario == "boundary":
        # Controlled shift: exactly 54% positive vs 50% baseline -> Delta P = 4.0% > 3.0% threshold
        n_pos = int(n * (baseline_pos + 0.04))
        n_neg = n - n_pos
        labels = (["positive"] * n_pos) + (["negative"] * n_neg)
        random.shuffle(labels)

        for i, label in enumerate(labels):
            if label == "positive":
                text = random.choice(test_pos_texts)
            else:
                text = random.choice(test_neg_texts)
            conf = random.uniform(0.75, 0.95)
            records.append({
                "timestamp": current_time + (i * 60),
                "text_length": len(text.split()),
                "predicted_label": label,
                "confidence": float(round(conf, 4)),
            })
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Choose 'normal', 'drift', or 'boundary'.")

    with open(output_path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    pos_count = sum(1 for r in records if r["predicted_label"] == "positive")
    pos_rate = pos_count / len(records)
    delta_p = abs(pos_rate - baseline_pos)
    print(f"Generated {len(records)} simulated prediction logs (Scenario: '{scenario}')")
    print(f"  Positive Rate: {pos_rate:.1%} | Baseline: {baseline_pos:.1%} | Delta P: {delta_p:.1%}")
    print(f"  Logs written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate production inference logs for drift testing.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["normal", "drift", "boundary"],
        default="normal",
        help="Simulation scenario: 'normal' (<3% drift), 'drift' (>3% drift), 'boundary' (~4% drift).",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of prediction logs to generate.")
    parser.add_argument("--append", action="store_true", help="Append to existing logs instead of overwriting.")
    args = parser.parse_args()

    generate_traffic(scenario=args.scenario, n=args.n, clear_existing=not args.append)


if __name__ == "__main__":
    main()
