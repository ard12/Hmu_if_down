"""Tests for Federated Learning REST endpoints and HUD (hub/dashboard/app.py)."""

import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app, broadcaster
from hub.dp_trainer import DPModel
from hub.federated_server import FederatedAggregationServer
from hub.personalization_layer import PersonalizationLayer


@pytest.fixture
def client():
    # Fresh server instance for clean test state
    base_model = DPModel(weights=[0.0, 0.0, 0.0, 0.0], bias=0.0)
    broadcaster.federated_server = FederatedAggregationServer(global_model=base_model, min_participants=2)
    broadcaster.personalization_layer = PersonalizationLayer(global_model=base_model)
    with TestClient(app) as test_client:
        yield test_client


def test_get_federated_page(client):
    """GET /federated should serve HTML HUD."""
    response = client.get("/federated")
    assert response.status_code == 200
    assert "Federated Learning &amp; Differential Privacy HUD" in response.text or "Federated Learning & Differential Privacy HUD" in response.text
    assert "Differential Privacy (DP-SGD)" in response.text


def test_get_federated_status(client):
    """GET /api/federated/status returns round status and privacy budget."""
    response = client.get("/api/federated/status")
    assert response.status_code == 200
    data = response.json()
    assert data["round"] == 1
    assert data["participants"] == 0
    assert data["min_required"] == 2
    assert "total_epsilon" in data
    assert "history" in data
    assert isinstance(data["history"], list)


def test_get_federated_weights(client):
    """GET /api/federated/weights returns global model parameters."""
    response = client.get("/api/federated/weights")
    assert response.status_code == 200
    data = response.json()
    assert "round" in data
    assert "weights" in data
    assert "bias" in data
    assert "timestamp" in data
    assert len(data["weights"]) == 4


def test_post_federated_gradients_single_hub(client):
    """POST /api/federated/gradients receives gradients from first hub."""
    payload = {
        "hub_id": "hub-alpha",
        "gradients": [[0.1, -0.2, 0.05, 0.0], [0.01]],
        "n_samples": 50,
        "epsilon": 0.85,
    }
    response = client.post("/api/federated/gradients", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RECEIVED"
    assert data["round"] == 1
    assert data["participants"] == 1
    assert data["min_required"] == 2


def test_post_federated_gradients_aggregation_trigger(client):
    """POST /api/federated/gradients triggers aggregation when min_participants reached."""
    # Hub 1
    payload1 = {
        "hub_id": "hub-1",
        "gradients": [[0.2, -0.1, 0.0, 0.1], [0.02]],
        "n_samples": 40,
        "epsilon": 0.7,
    }
    r1 = client.post("/api/federated/gradients", json=payload1)
    assert r1.status_code == 200
    assert r1.json()["status"] == "RECEIVED"

    # Hub 2 triggers aggregation
    payload2 = {
        "hub_id": "hub-2",
        "gradients": [[-0.1, 0.3, 0.1, -0.2], [-0.01]],
        "n_samples": 60,
        "epsilon": 0.9,
    }
    r2 = client.post("/api/federated/gradients", json=payload2)
    assert r2.status_code == 200
    data = r2.json()
    assert data["status"] == "AGGREGATED"
    assert data["round"] == 1
    assert "weights" in data


def test_post_federated_gradients_validation_error(client):
    """POST /api/federated/gradients with missing fields returns 400."""
    response = client.post("/api/federated/gradients", json={"hub_id": "hub-x"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_post_federated_personalize_success(client):
    """POST /api/federated/personalize fits adaptation head and returns metrics."""
    features = [
        [0.5, -0.2, 0.1, 0.8],
        [-0.4, 0.1, 0.9, -0.1],
        [0.8, -0.6, 0.3, 0.7],
        [-0.1, 0.4, -0.5, -0.2],
    ] * 5  # 20 samples
    labels = [1, 0, 1, 0] * 5

    payload = {
        "features": features,
        "labels": labels,
        "epochs": 5,
        "hidden_dim": 8,
    }
    response = client.post("/api/federated/personalize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "TRAINED"
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert "f1" in data["metrics"]
    assert data["hidden_dim"] == 8
    assert data["is_fitted"] is True


def test_post_federated_personalize_validation_error(client):
    """POST /api/federated/personalize with mismatched lengths returns 400."""
    payload = {
        "features": [[1.0, 2.0, 3.0, 4.0]],
        "labels": [1, 0],
    }
    response = client.post("/api/federated/personalize", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()
