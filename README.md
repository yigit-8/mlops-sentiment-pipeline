# MLOps Sentiment Pipeline

![CI](https://github.com/yigit-8/mlops-sentiment-pipeline/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A production-style MLOps pipeline for sentiment analysis, covering experiment tracking, model serving, data drift detection, and CI/CD automation.

## Architecture

```
train.py    ->  MLflow (experiment tracking)
                logs params, metrics, run ID

serve.py    ->  FastAPI (model serving)
                /analyze  /logs  /stats  /health

drift.py    ->  Evidently (data drift detection)
                compares live predictions vs reference data

docker-compose.yml  ->  API + MLflow UI (two containers)
.github/workflows/  ->  GitHub Actions CI (test + Docker build)
```

## Results

The pipeline serves `distilbert-base-uncased-finetuned-sst-2-english` rather than training its own classifier, so the number worth reporting is how that model behaves on a held-out set it was not tuned against. Measured on the **SST-2 validation split (872 sentences)**:

| Metric | Value |
|---|---|
| Accuracy | 0.911 |
| F1 | 0.914 |
| Mean confidence | 0.983 |
| Accuracy, positive / negative | 0.930 / 0.890 |

Serving latency, single requests on CPU (no GPU, no batching):

| | Latency |
|---|---|
| p50 | 37 ms |
| p95 | 64 ms |
| p99 | 80 ms |

A note on the ten sentences in `EVAL_SAMPLES`: they score a perfect 1.00, and that is exactly why they are a smoke test and not a benchmark. They are unambiguously polarised, so they catch a broken model load or a mislabelled output in CI without pretending to measure quality — the 0.911 above is the honest figure.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Model | HuggingFace Transformers (DistilBERT SST-2) |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |

## Quick Start

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Train and log to MLflow**

```bash
python src/train.py
mlflow ui
```

Open http://localhost:5000 to browse experiments.

**Serve the API**

```bash
uvicorn src.serve:app --reload
```

Open http://localhost:8000/docs for the interactive API docs.

**Check for data drift**

```bash
python src/drift.py
```

Generates `data/drift_report.html`.

**Run everything with Docker Compose**

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health message |
| GET | `/health` | Readiness probe |
| POST | `/analyze` | Run sentiment analysis |
| GET | `/logs` | Recent predictions |
| GET | `/stats` | Label distribution |

**Example request:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This pipeline is working perfectly!"}'
```

```json
{"label": "POSITIVE", "score": 0.9998}
```

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD

Every push to `main` triggers two jobs. The first runs the full pytest suite. If that passes, the second builds the Docker image and smoke-tests the `/health` endpoint.

## Pipeline Flow

```
1. Train      python src/train.py
              evaluates the model on a reference set
              logs params and metrics to MLflow
              saves reference data for drift detection

2. Serve      uvicorn src.serve:app
              loads the model recorded by train.py
              stores every prediction in SQLite

3. Monitor    python src/drift.py
              compares live predictions against the reference set
              generates an HTML drift report
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

