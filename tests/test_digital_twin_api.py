"""Tests for Digital Twin HUD & Simulation REST/WebSocket Endpoints.

@req SRS-SIM-001
@req SRS-MO-001
"""
import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app, broadcaster


@pytest.fixture
def client():
    return TestClient(app)


def test_simulation_state_endpoint_returns_200(client):
    response = client.get("/api/simulation/state")
    assert response.status_code == 200
    data = response.json()
    assert "occupants" in data
    assert "active_count" in data
    assert "frame_count" in data
    assert "fallen_count" in data


def test_simulation_state_has_occupants_key(client):
    response = client.get("/api/simulation/state")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["occupants"], list)


def test_inject_endpoint_creates_track(client):
    payload = {
        "centroid": [2.5, 3.0, 1.6],
        "area_m2": 0.6,
        "peak_velocity_mps": 0.2,
    }
    response = client.post("/api/simulation/inject", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "injected"
    assert data["active_count"] >= 1
    assert len(data["active_tracks"]) >= 1


def test_inject_invalid_payload_returns_422(client):
    # Missing centroid
    res1 = client.post("/api/simulation/inject", json={"area_m2": 0.5})
    assert res1.status_code == 422

    # Centroid has only 2 coordinates
    res2 = client.post("/api/simulation/inject", json={"centroid": [1.0, 2.0]})
    assert res2.status_code == 422

    # Non-numeric coordinate
    res3 = client.post("/api/simulation/inject", json={"centroid": ["invalid", 2.0, 3.0]})
    assert res3.status_code == 422


def test_digital_twin_html_served(client):
    response = client.get("/digital-twin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "3D Digital Twin HUD" in response.text
