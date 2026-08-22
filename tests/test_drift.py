"""Tests for src/drift.py.

Every test redirects the module's reference / database / report paths at
``tmp_path`` so nothing is written into the repository.
"""

import json
import sqlite3

import pytest

from src import drift

POSITIVE_PHRASES = [
    "Absolutely love this, it works perfectly.",
    "Great value for the price, highly recommend.",
    "Outstanding performance and very easy to use.",
    "Exceeded all my expectations, truly amazing.",
    "Fantastic service and friendly staff.",
]

NEGATIVE_PHRASES = [
    "Terrible quality, it broke after one use.",
    "The worst experience I have ever had.",
    "Very disappointed, it did not meet expectations.",
    "Not worth the money, a complete waste.",
    "Horrible customer support, will not buy again.",
]


def _rows(phrases, label, n=60):
    return [{"text": phrases[i % len(phrases)], "label": label} for i in range(n)]


def _write_reference(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_current(db_path, rows):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE predictions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT,
                label     TEXT,
                score     REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
        conn.executemany(
            "INSERT INTO predictions (text, label, score) VALUES (?, ?, ?)",
            [(r["text"], r["label"], 0.99) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    db = tmp_path / "predictions.db"
    report = tmp_path / "reports" / "drift_report.html"
    monkeypatch.setattr(drift, "REFERENCE_PATH", str(reference))
    monkeypatch.setattr(drift, "DB_PATH", str(db))
    monkeypatch.setattr(drift, "REPORT_PATH", str(report))
    return reference, db, report


def test_reports_no_reference_data(paths):
    result = drift.run_drift_report()
    assert result == {"drift_detected": False, "reason": "no_reference_data"}


def test_reports_insufficient_current_data(paths):
    reference, db, _ = paths
    _write_reference(reference, _rows(POSITIVE_PHRASES, "POSITIVE"))
    _write_current(db, _rows(POSITIVE_PHRASES, "POSITIVE", n=2))

    result = drift.run_drift_report(min_samples=5)
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_data"
    assert result["count"] == 2


def test_drift_detected_when_current_data_shifts(paths):
    reference, db, report = paths
    reference_rows = _rows(POSITIVE_PHRASES, "POSITIVE")
    # Deliberately shifted: a different vocabulary and the opposite label.
    shifted_rows = _rows(NEGATIVE_PHRASES, "NEGATIVE")
    _write_reference(reference, reference_rows)
    _write_current(db, shifted_rows)

    result = drift.run_drift_report(min_samples=5)

    assert result["drift_detected"] is True
    assert result["reference_rows"] == len(reference_rows)
    assert result["current_rows"] == len(shifted_rows)
    assert report.exists(), "the HTML report should be written next to the results"


def test_no_drift_when_current_matches_reference(paths):
    reference, db, _ = paths
    rows = _rows(POSITIVE_PHRASES, "POSITIVE") + _rows(NEGATIVE_PHRASES, "NEGATIVE")
    _write_reference(reference, rows)
    # An unshifted copy of the very same distribution.
    _write_current(db, list(rows))

    result = drift.run_drift_report(min_samples=5)

    assert result["drift_detected"] is False
    assert result["current_rows"] == len(rows)


def test_load_current_returns_empty_frame_without_database(paths):
    frame = drift.load_current()
    assert frame.empty
    assert list(frame.columns) == ["text", "label"]
