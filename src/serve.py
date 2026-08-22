"""
FastAPI serving layer.

Loads the model recorded in data/model_info.json (written by train.py).
Falls back to the default DistilBERT SST-2 model if no model info exists.

Endpoints:
    GET  /          health check
    POST /analyze   run sentiment analysis
    GET  /logs      recent predictions
    GET  /stats     label distribution
    GET  /health    readiness probe (model loaded?)
"""

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_INFO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "model_info.json"
)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Longest input accepted by /analyze. The model truncates at 512 tokens anyway;
# this bound stops a pathological payload from reaching it at all.
MAX_TEXT_LENGTH = 5000

classifier = None


def load_model() -> None:
    global classifier
    model_name = DEFAULT_MODEL
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            info = json.load(f)
        model_name = info.get("model_name", DEFAULT_MODEL)
    logger.info("Loading model: %s", model_name)
    classifier = pipeline("sentiment-analysis", model=model_name)
    logger.info("Model ready.")


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT,
            label     TEXT,
            score     REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(text: str, label: str, score: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO predictions (text, label, score) VALUES (?, ?, ?)",
            (text, label, score),
        )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="MLOps Sentiment Pipeline",
    description="Sentiment analysis API with MLflow tracking and drift detection.",
    version="1.0.0",
    lifespan=lifespan,
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class PredictionResponse(BaseModel):
    label: str
    score: float


@app.get("/")
def root():
    return {"message": "MLOps Sentiment Pipeline is running. Visit /docs for the API."}


@app.get("/health")
def health():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok", "model_loaded": True}


@app.post("/analyze", response_model=PredictionResponse)
def analyze(request: TextRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        result = classifier(request.text)[0]
    except Exception:
        logger.exception("Inference failed for a %d-character input", len(request.text))
        raise HTTPException(
            status_code=400,
            detail="Could not analyze the submitted text. Please try a shorter or simpler input.",
        )
    log_prediction(request.text, result["label"], result["score"])
    return PredictionResponse(label=result["label"], score=result["score"])


@app.get("/logs")
def get_logs(limit: int = Query(default=20, ge=1, le=200)):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT text, label, score, timestamp FROM predictions ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"text": r[0], "label": r[1], "score": r[2], "timestamp": r[3]} for r in rows
    ]


@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT label, COUNT(*) as count FROM predictions GROUP BY label"
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}
