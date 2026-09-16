"""Unit tests for the Real-Time Telemetry Web Dashboard."""

import asyncio
import json
import pytest
from starlette.testclient import TestClient

from hub.dashboard.app import app, broadcaster


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_index_route(client):
    """GET / should serve HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Multi-Modal Fall Detection HUD" in response.text
    assert "Radar Centroid Height" in response.text


def test_dashboard_status_api(client):
    """GET /api/status should return system health and node map."""
    broadcaster.latest_state = "Normal"
    broadcaster.latest_height = 1.65
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "target_height_m" in data
    assert "nodes" in data
    assert set(data["nodes"].keys()) == {"1", "2", "3"}


def test_dashboard_incidents_api(client):
    """GET /api/incidents should return recent incidents array."""
    response = client.get("/api/incidents")
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert isinstance(data["incidents"], list)


def test_dashboard_websocket_telemetry(client):
    """WebSocket /ws/telemetry connects, receives init packet, and handles ping/pong."""
    broadcaster.latest_state = "Normal"
    broadcaster.latest_height = 1.70

    with client.websocket_connect("/ws/telemetry") as websocket:
        init_raw = websocket.receive_text()
        init_data = json.loads(init_raw)
        assert init_data["type"] == "init"
        assert init_data["state"] == "Normal"
        assert init_data["height"] == 1.70

        # Send ping
        websocket.send_text("ping")
        resp_raw = websocket.receive_text()
        resp_data = json.loads(resp_raw)
        assert resp_data["type"] == "pong"


def test_dashboard_calibrate_api(client):
    """POST /api/calibrate should trigger baseline reset and return status."""
    response = client.post("/api/calibrate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "motionless_variance_threshold" in data


def test_dashboard_thresholds_api(client):
    """GET and POST /api/thresholds should read and update active thresholds."""
    # Read initial
    get_res = client.get("/api/thresholds")
    assert get_res.status_code == 200
    assert "thresholds" in get_res.json()

    # Update
    post_res = client.post("/api/thresholds", json={
        "motionless_variance_threshold": 0.095,
        "enable_radar_veto": True,
    })
    assert post_res.status_code == 200
    updated = post_res.json()["thresholds"]
    assert updated["motionless_variance_threshold"] == 0.095
    assert updated["enable_radar_veto"] is True


def test_dashboard_datasets_api(client):
    """GET /api/datasets should return recorded session archives list."""
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert "total_count" in data
    assert isinstance(data["datasets"], list)
