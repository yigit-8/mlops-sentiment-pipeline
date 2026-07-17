"""
Data drift detection using Evidently.

Compares current predictions in the SQLite database against the reference
dataset saved by train.py and prints a drift report.

Usage:
    python src/drift.py
    python src/drift.py --min-samples 20
"""

import argparse
import json
import os
import sqlite3

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.report import Report

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reference.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drift_report.html")


def load_reference() -> pd.DataFrame:
    with open(REFERENCE_PATH) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_current(limit: int = 200) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["text", "label"])
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT text, label FROM predictions ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["text", "label"])


def run_drift_report(min_samples: int = 5) -> dict:
    if not os.path.exists(REFERENCE_PATH):
        print("No reference data found. Run src/train.py first.")
        return {"drift_detected": False, "reason": "no_reference_data"}

    reference = load_reference()
    current = load_current()

    if len(current) < min_samples:
        print(f"Not enough current data ({len(current)} rows, need {min_samples}).")
        return {
            "drift_detected": False,
            "reason": "insufficient_data",
            "count": len(current),
        }

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"Drift report saved to: {REPORT_PATH}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    summary = {
        "drift_detected": drift_detected,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "report_path": REPORT_PATH,
    }
    print(f"Drift detected: {drift_detected}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()
    run_drift_report(min_samples=args.min_samples)
