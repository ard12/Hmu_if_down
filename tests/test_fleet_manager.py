import pytest
import time
from pathlib import Path
from hub.fleet_manager import FleetManager

@pytest.fixture
def temp_fleet_db(tmp_path):
    db_path = tmp_path / "test_fleet.db"
    manager = FleetManager(db_path=db_path)
    return manager

def test_provision_device(temp_fleet_db):
    """
    Covers: SRS-FLT-001
    Verifies that the system can provision a new edge hub and map it to a facility/room.
    """
    device_id = temp_fleet_db.provision_device("FAC-1", "ROOM-101", "v1.0", "v4.7.0")
    assert device_id.startswith("hub_")
    
    status = temp_fleet_db.get_fleet_status()
    assert len(status) == 1
    assert status[0]["device_id"] == device_id
    assert status[0]["facility_id"] == "FAC-1"
    assert status[0]["status"] == "online"

def test_heartbeat_updates_timestamp(temp_fleet_db):
    """
    Covers: SRS-FLT-002
    Verifies that edge hubs can update their heartbeat and version status.
    """
    device_id = temp_fleet_db.provision_device("FAC-1", "ROOM-102", "v1.0", "v4.7.0")
    
    status_before = temp_fleet_db.get_fleet_status()[0]
    time.sleep(0.1)
    
    # Send heartbeat with a version update
    success = temp_fleet_db.record_heartbeat(device_id, model_ver="v4.7.1")
    assert success is True
    
    status_after = temp_fleet_db.get_fleet_status()[0]
    assert status_after["last_seen_at"] > status_before["last_seen_at"]
    assert status_after["model_ver"] == "v4.7.1"

def test_stale_device_marked_offline(temp_fleet_db):
    """
    Covers: HAZ-032
    Verifies that offline devices are detected and marked appropriately.
    """
    # Hack the DB to make the device look old
    device_id = temp_fleet_db.provision_device("FAC-2", "ROOM-201", "v1.0", "v4.7.0")
    
    with temp_fleet_db._connect() as conn:
        conn.execute(
            "UPDATE fleet_devices SET last_seen_at = ? WHERE device_id = ?",
            (time.time() - 300, device_id) # 5 minutes ago
        )
        conn.commit()
        
    status = temp_fleet_db.get_fleet_status()
    assert len(status) == 1
    assert status[0]["status"] == "offline"

def test_heartbeat_unknown_device(temp_fleet_db):
    success = temp_fleet_db.record_heartbeat("unknown_hub_123")
    assert success is False
