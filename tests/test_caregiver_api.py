"""Tests for Caregiver REST API and Mobile Authentication.

@req SRS-UX-002
"""
import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app, _get_notification_escalator


@pytest.fixture
def client():
    return TestClient(app)


def test_register_caregiver_returns_201(client):
    payload = {
        "caregiver_id": "nurse_sarah",
        "name": "Sarah Connor, RN",
        "phone": "+1-555-0199",
        "email": "sarah.connor@hospital.org",
        "role": "primary",
        "password": "SecurePassword123!",
    }
    response = client.post("/api/caregiver/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "created"
    assert data["caregiver"]["caregiver_id"] == "nurse_sarah"
    assert data["caregiver"]["name"] == "Sarah Connor, RN"


def test_login_returns_jwt_token(client):
    # Register first
    client.post("/api/caregiver/register", json={
        "caregiver_id": "dr_john",
        "name": "Dr. John Watson",
        "phone": "+1-555-0188",
        "email": "watson@clinic.org",
        "role": "secondary",
        "password": "WatsonPassword456!",
    })

    # Login
    response = client.post("/api/caregiver/login", json={
        "caregiver_id": "dr_john",
        "password": "WatsonPassword456!",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600
    assert data["caregiver_id"] == "dr_john"


def test_login_invalid_password_returns_401(client):
    response = client.post("/api/caregiver/login", json={
        "caregiver_id": "dr_john",
        "password": "WrongPassword!",
    })
    assert response.status_code == 401


def test_alerts_endpoint_requires_auth(client):
    # Without Authorization header -> 401
    response = client.get("/api/caregiver/alerts")
    assert response.status_code == 401

    # With invalid Bearer token -> 401
    bad_resp = client.get(
        "/api/caregiver/alerts",
        headers={"Authorization": "Bearer invalid_token_12345"},
    )
    assert bad_resp.status_code == 401


def test_ack_alert_returns_200(client):
    # Setup user and escalation event
    client.post("/api/caregiver/register", json={
        "caregiver_id": "nurse_betty",
        "name": "Betty White",
        "phone": "+1-555-0177",
        "email": "betty@care.org",
        "password": "BettyPassword789!",
    })
    login_resp = client.post("/api/caregiver/login", json={
        "caregiver_id": "nurse_betty",
        "password": "BettyPassword789!",
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create event in escalator
    escalator = _get_notification_escalator()
    escalator.start_escalation(event_id="EVT-ACK-TEST", severity="high", room_id=3)

    # Acknowledge via API
    ack_resp = client.post("/api/caregiver/alerts/EVT-ACK-TEST/ack", headers=headers)
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    assert ack_data["status"] == "acknowledged"
    assert ack_data["caregiver_id"] == "nurse_betty"
    assert ack_data["event"]["state"] == "acknowledged"


def test_ack_unknown_event_returns_404(client):
    login_resp = client.post("/api/caregiver/login", json={
        "caregiver_id": "dr_john",
        "password": "WatsonPassword456!",
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/api/caregiver/alerts/UNKNOWN-EVENT-999/ack", headers=headers)
    assert resp.status_code == 404


def test_timeline_returns_escalation_history(client):
    escalator = _get_notification_escalator()
    escalator.start_escalation(event_id="EVT-TIME-TEST", severity="critical", room_id=1)
    escalator.advance_time(time_after := 1000.0)

    resp = client.get("/api/caregiver/alerts/EVT-TIME-TEST/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_id"] == "EVT-TIME-TEST"
    assert "timeline" in data
    assert len(data["timeline"]) >= 1
