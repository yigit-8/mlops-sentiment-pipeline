import pytest
from fastapi.testclient import TestClient

from src.serve import app


@pytest.fixture(scope="module")
def client():
    # As a context manager so the lifespan handler runs: it creates the SQLite
    # database and loads the model, which every endpoint below depends on.
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_analyze_positive(client):
    response = client.post("/analyze", json={"text": "I love this, it works great!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert 0.0 < data["score"] <= 1.0


def test_analyze_negative(client):
    response = client.post("/analyze", json={"text": "This is terrible and I hate it."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert 0.0 < data["score"] <= 1.0


def test_analyze_empty_text(client):
    # Empty text is rejected by the TextRequest min_length constraint, so this
    # must be a validation error - not "either outcome is fine".
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_oversized_text(client):
    response = client.post("/analyze", json={"text": "a" * 5001})
    assert response.status_code == 422


def test_logs_limit_is_bounded(client):
    assert client.get("/logs", params={"limit": 0}).status_code == 422
    assert client.get("/logs", params={"limit": 201}).status_code == 422
    assert client.get("/logs", params={"limit": 200}).status_code == 200


def test_logs_returns_list(client):
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_returns_dict(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
