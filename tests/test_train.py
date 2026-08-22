"""Tests for src/train.py.

The transformers pipeline and the MLflow client are the expensive, non-deterministic
parts, so both are replaced with stand-ins. What is exercised here is the code this
repository owns: metric computation and artifact writing. All output goes to
``tmp_path``.
"""

import json
from types import SimpleNamespace

import pytest

from src import train


class FakeClassifier:
    """Stand-in for pipeline("sentiment-analysis", ...).

    Returns a canned label per text and records how it was called, so a test can
    assert that evaluate() batches all texts into a single call.
    """

    def __init__(self, predictions: dict, score: float = 0.9):
        self._predictions = predictions
        self._score = score
        self.calls: list = []

    def __call__(self, texts):
        self.calls.append(texts)
        return [{"label": self._predictions[t], "score": self._score} for t in texts]


SAMPLES = [
    {"text": "great product", "label": "POSITIVE"},
    {"text": "awful product", "label": "NEGATIVE"},
    {"text": "love it", "label": "POSITIVE"},
    {"text": "hate it", "label": "NEGATIVE"},
]


def test_evaluate_scores_perfect_predictions():
    classifier = FakeClassifier({s["text"]: s["label"] for s in SAMPLES}, score=0.95)

    metrics = train.evaluate(classifier, SAMPLES)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["avg_confidence"] == pytest.approx(0.95)
    assert metrics["avg_latency_ms"] >= 0.0


def test_evaluate_scores_a_wrong_prediction():
    predictions = {s["text"]: s["label"] for s in SAMPLES}
    predictions["love it"] = "NEGATIVE"  # one false negative
    classifier = FakeClassifier(predictions, score=0.8)

    metrics = train.evaluate(classifier, SAMPLES)

    assert metrics["accuracy"] == pytest.approx(0.75)
    # precision 1.0, recall 0.5 for the POSITIVE class
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["avg_confidence"] == pytest.approx(0.8)


def test_evaluate_sends_all_texts_in_one_batch():
    classifier = FakeClassifier({s["text"]: s["label"] for s in SAMPLES})

    train.evaluate(classifier, SAMPLES)

    assert len(classifier.calls) == 1
    assert classifier.calls[0] == [s["text"] for s in SAMPLES]


def test_save_reference_data_writes_json(tmp_path, monkeypatch):
    target = tmp_path / "data" / "reference.json"  # parent does not exist yet
    monkeypatch.setattr(train, "REFERENCE_DATA_PATH", str(target))

    train.save_reference_data(SAMPLES)

    assert json.loads(target.read_text(encoding="utf-8")) == SAMPLES


def test_save_model_info_writes_json(tmp_path, monkeypatch):
    target = tmp_path / "model_info.json"
    monkeypatch.setattr(train, "MODEL_REGISTRY_PATH", str(target))
    metrics = {"accuracy": 1.0, "f1": 1.0}

    train.save_model_info("some-model", "run-123", metrics)

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "model_name": "some-model",
        "run_id": "run-123",
        "metrics": metrics,
    }


def test_train_logs_params_and_metrics_and_writes_artifacts(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    registry = tmp_path / "model_info.json"
    monkeypatch.setattr(train, "REFERENCE_DATA_PATH", str(reference))
    monkeypatch.setattr(train, "MODEL_REGISTRY_PATH", str(registry))

    labels = {s["text"]: s["label"] for s in train.EVAL_SAMPLES}
    monkeypatch.setattr(
        train, "pipeline", lambda task, model: FakeClassifier(labels, score=0.99)
    )

    logged_params: dict = {}
    logged_metrics: dict = {}

    class FakeRun:
        info = SimpleNamespace(run_id="run-abc")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(train.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda: FakeRun())
    monkeypatch.setattr(
        train.mlflow, "log_param", lambda k, v: logged_params.update({k: v})
    )
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda m: logged_metrics.update(m))

    train.train("some-model")

    assert logged_params["model_name"] == "some-model"
    assert logged_params["eval_samples"] == len(train.EVAL_SAMPLES)
    assert logged_metrics["accuracy"] == 1.0
    assert set(logged_metrics) == {"accuracy", "f1", "avg_confidence", "avg_latency_ms"}

    assert json.loads(reference.read_text(encoding="utf-8")) == train.EVAL_SAMPLES
    info = json.loads(registry.read_text(encoding="utf-8"))
    assert info["model_name"] == "some-model"
    assert info["run_id"] == "run-abc"
