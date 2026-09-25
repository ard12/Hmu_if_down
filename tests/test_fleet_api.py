import pytest
from fastapi.testclient import TestClient
from hub.dashboard.app import app

# Create a TestClient using the main FastAPI app
client = TestClient(app)

def test_provision_endpoint():
    response = client.post("/api/fleet/provision", json={
        "facility_id": "FAC-TEST",
        "room_id": "ROOM-99",
        "firmware_ver": "1.0.0",
        "model_ver": "4.7.0"
    })
    assert response.status_code == 200
    data = response.json()
    assert "device_id" in data
    assert data["status"] == "provisioned"

def test_heartbeat_endpoint():
    # Provision first to get a valid device ID
    prov_resp = client.post("/api/fleet/provision", json={
        "facility_id": "FAC-TEST",
        "room_id": "ROOM-99",
        "firmware_ver": "1.0.0",
        "model_ver": "4.7.0"
    })
    device_id = prov_resp.json()["device_id"]
    
    # Send heartbeat
    hb_resp = client.post("/api/fleet/heartbeat", json={
        "device_id": device_id,
        "model_ver": "4.7.1"
    })
    assert hb_resp.status_code == 200
    assert hb_resp.json()["status"] == "ok"

def test_heartbeat_invalid_device():
    hb_resp = client.post("/api/fleet/heartbeat", json={
        "device_id": "invalid_hub",
    })
    assert hb_resp.status_code == 404

def test_list_devices_endpoint():
    response = client.get("/api/fleet/devices")
    assert response.status_code == 200
    data = response.json()
    assert "devices" in data
    assert "total" in data

def test_ota_deploy_endpoint():
    mock_model_hex = b"dummy_model_data".hex()
    response = client.post("/api/fleet/ota/deploy", json={
        "version": "4.8.0",
        "model_data_hex": mock_model_hex,
        "target_facility": "FAC-TEST"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deployed"
    assert "deployment" in data
    assert data["deployment"]["version"] == "4.8.0"

def test_ota_deploy_invalid_hex():
    response = client.post("/api/fleet/ota/deploy", json={
        "version": "4.8.0",
        "model_data_hex": "not_a_hex_string",
        "target_facility": "FAC-TEST"
    })
    assert response.status_code == 400
