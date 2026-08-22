"""
Train a sentiment classifier and log everything to MLflow.

Usage:
    python src/train.py
    python src/train.py --model distilbert-base-uncased-finetuned-sst-2-english
"""

import argparse
import json
import os
import time

import mlflow
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

REFERENCE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "reference.json"
)
MODEL_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "model_info.json"
)

EVAL_SAMPLES = [
    {"text": "I love this product, it works perfectly!", "label": "POSITIVE"},
    {"text": "This is the worst experience I have ever had.", "label": "NEGATIVE"},
    {"text": "Absolutely fantastic service and friendly staff.", "label": "POSITIVE"},
    {"text": "Terrible quality, broke after one use.", "label": "NEGATIVE"},
    {"text": "Great value for the price, highly recommend.", "label": "POSITIVE"},
    {"text": "Very disappointed, did not meet expectations.", "label": "NEGATIVE"},
    {"text": "Outstanding performance and easy to use.", "label": "POSITIVE"},
    {"text": "Not worth the money, complete waste.", "label": "NEGATIVE"},
    {"text": "Exceeded all my expectations, truly amazing.", "label": "POSITIVE"},
    {"text": "Horrible customer support, will not buy again.", "label": "NEGATIVE"},
]


def evaluate(classifier, samples: list[dict]) -> dict:
    texts = [s["text"] for s in samples]
    true_labels = [s["label"] for s in samples]

    start = time.perf_counter()
    preds = classifier(texts)
    latency_ms = (time.perf_counter() - start) / len(texts) * 1000

    pred_labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]

    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels, pos_label="POSITIVE"),
        "avg_confidence": sum(scores) / len(scores),
        "avg_latency_ms": latency_ms,
    }


def save_reference_data(samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(REFERENCE_DATA_PATH), exist_ok=True)
    with open(REFERENCE_DATA_PATH, "w") as f:
        json.dump(samples, f, indent=2)


def save_model_info(model_name: str, run_id: str, metrics: dict) -> None:
    info = {"model_name": model_name, "run_id": run_id, "metrics": metrics}
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(info, f, indent=2)


def train(model_name: str) -> None:
    mlflow.set_experiment("sentiment-pipeline")

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("eval_samples", len(EVAL_SAMPLES))

        print(f"Loading model: {model_name}")
        classifier = pipeline("sentiment-analysis", model=model_name)

        print("Evaluating...")
        metrics = evaluate(classifier, EVAL_SAMPLES)

        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        save_reference_data(EVAL_SAMPLES)
        save_model_info(model_name, run.info.run_id, metrics)

        print(f"Run ID: {run.info.run_id}")
        print("Training complete. Reference data and model info saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HuggingFace model name",
    )
    args = parser.parse_args()
    train(args.model)
