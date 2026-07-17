import pytest
from fastapi.testclient import TestClient

from src.serve import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_analyze_positive():
    response = client.post("/analyze", json={"text": "I love this, it works great!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert 0.0 < data["score"] <= 1.0


def test_analyze_negative():
    response = client.post("/analyze", json={"text": "This is terrible and I hate it."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert 0.0 < data["score"] <= 1.0


def test_analyze_empty_text():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code in (200, 422)


def test_logs_returns_list():
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_returns_dict():
    response = client.get("/stats")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
