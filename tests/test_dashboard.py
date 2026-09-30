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


def test_dashboard_phase11_endpoints(client):
    """Test Phase 11 REST endpoints for labeling, drift status, and retraining."""
    # 1. POST /api/labels/{event_id}
    res_label = client.post("/api/labels/evt_test_1", json={"confirmed": True, "labeller": "nurse_42"})
    assert res_label.status_code == 200
    assert res_label.json()["status"] == "ok"
    assert res_label.json()["confirmed"] is True

    # 2. GET /api/labels/stats
    res_stats = client.get("/api/labels/stats")
    assert res_stats.status_code == 200
    assert "total" in res_stats.json()
    assert "positives" in res_stats.json()

    # 3. GET /api/drift/status
    res_drift = client.get("/api/drift/status")
    assert res_drift.status_code == 200
    assert "psi" in res_drift.json()
    assert "status" in res_drift.json()

    # 4. GET /api/retrain/status
    res_retrain_status = client.get("/api/retrain/status")
    assert res_retrain_status.status_code == 200
    assert "status" in res_retrain_status.json()

    # 5. POST /api/retrain/trigger
    res_retrain_trigger = client.post("/api/retrain/trigger")
    assert res_retrain_trigger.status_code == 200
    assert "status" in res_retrain_trigger.json()


def test_dashboard_security_headers(client):
    """Verify HTTP security headers are injected into HTTP responses (OWASP A05:2021)."""
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_dataset_path_traversal_protection(client):
    """Verify dataset endpoint strictly blocks path traversal attacks (CWE-22)."""
    # 1. Directory traversal sequence returns 400
    res_traversal = client.get("/api/datasets/..%2F..%2Fetc%2Fpasswd.json")
    assert res_traversal.status_code in (400, 404)

    # 2. Windows-style traversal returns 400
    res_win = client.get("/api/datasets/..%5C..%5Csecret.json")
    assert res_win.status_code in (400, 404)

    # 3. Non-existent legitimate file safely returns 404
    res_missing = client.get("/api/datasets/nonexistent_dataset.json")
    assert res_missing.status_code == 404


def test_firmware_path_traversal_protection(client):
    """Verify firmware download endpoint strictly blocks path traversal (CWE-22)."""
    res = client.get("/firmware/..%2F..%2Fsecret.bin")
    assert res.status_code in (400, 404)


@pytest.mark.parametrize(
    "route,expected_token",
    [
        ("/", "Multi-Modal Fall Detection HUD"),
        ("/caregiver", "Caregiver Alert Triage"),
        ("/family", "Family Care Portal"),
        ("/fleet", "Fleet Command Orchestrator"),
        ("/automations", "Environmental Safety & Smart Automations"),
        ("/digital-twin", "3D Digital Twin HUD"),
        ("/mobility", "Mobility & Fall Risk HUD"),
        ("/pose", "3D Video-Free Pose & Biomechanics HUD"),
        ("/mesh", "ESP-MESH & Room Handoff"),
        ("/analytics", "Population Health Analytics"),
        ("/explain", "Explainable AI (XAI)"),
        ("/acceleration", "Edge AI Acceleration"),
    ],
)
def test_all_ui_portal_pages_render(client, route, expected_token):
    """Verify that all 12 UI dashboards and clinical portals render successfully (UI/UX QA)."""
    res = client.get(route)
    assert res.status_code == 200, f"Route {route} failed to load: {res.status_code}"
    assert expected_token in res.text, f"Route {route} missing expected content token: {expected_token}"



