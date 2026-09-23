"""Tests for Explainable AI (XAI) Endpoints and Dashboard.

@req SRS-XAI-001
@req SRS-XAI-002
"""
import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_explain_last_endpoint_returns_200(client):
    response = client.get("/api/explain/last")
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "waterfall" in data
    assert "feature_names" in data["explanation"]
    assert len(data["explanation"]["feature_names"]) == 9


def test_explain_last_has_waterfall_bars(client):
    response = client.get("/api/explain/last")
    assert response.status_code == 200
    data = response.json()
    bars = data["waterfall"]["bars"]
    assert len(bars) == 9
    assert "feature" in bars[0]
    assert "contribution" in bars[0]
    assert "direction" in bars[0]


def test_explain_global_endpoint_returns_ranking(client):
    response = client.get("/api/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "ranking" in data
    assert "feature_names" in data
    assert len(data["ranking"]) == 9


def test_explain_event_returns_counterfactual(client):
    response = client.get("/api/explain/EVT-TEST-123")
    assert response.status_code == 200
    data = response.json()
    assert data["event_id"] == "EVT-TEST-123"
    assert "explanation" in data
    assert "waterfall" in data
    assert "counterfactual" in data
    assert "changes" in data["counterfactual"]


def test_model_card_api_returns_markdown(client):
    response = client.get("/api/model-card")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "Model Card: CSI & mmWave Fall Detection Classifier" in response.text


def test_explain_html_served(client):
    response = client.get("/explain")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Explainable AI (XAI) & Transparency HUD" in response.text
