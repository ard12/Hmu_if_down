"""Tests for Multi-Facility Cloud Gateway & Encrypted Offline Sync (Milestone 10.3)."""

from pathlib import Path
import pytest
from hub.cloud_sync import CloudSyncGateway


def test_enqueue_record_persists_to_db(tmp_path: Path):
    db_path = tmp_path / "sync.db"
    gw = CloudSyncGateway(db_path=db_path, facility_id="HOSP-99", building_id="TOWER-B")

    rec_id = gw.enqueue_record("FALL_ALERT", {"confidence": 0.98, "room": 102}, room_id=102)
    assert rec_id >= 1

    status = gw.get_status()
    assert status["facility_id"] == "HOSP-99"
    assert status["building_id"] == "TOWER-B"
    assert status["pending_records"] == 1
    assert status["total_synced"] == 0


def test_flush_queue_success_marks_synced(tmp_path: Path):
    db_path = tmp_path / "sync.db"
    received = []

    def mock_transport(packet):
        received.append(packet)
        return True

    gw = CloudSyncGateway(
        db_path=db_path,
        facility_id="CARE-01",
        transport_hook=mock_transport,
    )

    gw.enqueue_record("TELEMETRY", {"metric": 42}, room_id=1)
    gw.enqueue_record("FALL_ALERT", {"fall": True}, room_id=1)

    synced, failed = gw.flush_queue()
    assert synced == 2
    assert failed == 0
    assert len(received) == 2
    assert received[0]["payload"]["metric"] == 42
    assert received[1]["payload_type"] == "FALL_ALERT"

    status = gw.get_status()
    assert status["pending_records"] == 0
    assert status["total_synced"] == 2
    assert status["last_sync_utc"] is not None


def test_flush_queue_failure_increments_retry(tmp_path: Path):
    db_path = tmp_path / "sync.db"

    def failing_transport(packet):
        return False  # Network timeout / failure

    gw = CloudSyncGateway(
        db_path=db_path,
        facility_id="CARE-01",
        transport_hook=failing_transport,
    )

    gw.enqueue_record("AUDIT_EVENT", {"event": "TEST"}, room_id=5)

    synced, failed = gw.flush_queue()
    assert synced == 0
    assert failed == 1

    status = gw.get_status()
    assert status["pending_records"] == 1
    assert status["sync_failures"] == 1


def test_facility_hierarchy_metadata_in_packet(tmp_path: Path):
    db_path = tmp_path / "sync.db"
    captured_packet = {}

    def capture_hook(packet):
        captured_packet.update(packet)
        return True

    gw = CloudSyncGateway(
        db_path=db_path,
        facility_id="FAC-NORTH",
        building_id="BLD-3",
        wing_id="WING-ICU",
        transport_hook=capture_hook,
    )

    gw.enqueue_record("VITAL_SIGNS", {"bpm": 18.0}, room_id=305)
    gw.flush_queue()

    assert captured_packet["facility_id"] == "FAC-NORTH"
    assert captured_packet["building_id"] == "BLD-3"
    assert captured_packet["wing_id"] == "WING-ICU"
    assert captured_packet["room_id"] == 305
    assert captured_packet["payload_type"] == "VITAL_SIGNS"


def test_background_sync_lifecycle(tmp_path: Path):
    db_path = tmp_path / "sync.db"
    gw = CloudSyncGateway(db_path=db_path)
    gw.start_background_sync(interval_sec=0.1)
    assert gw._running is True
    gw.stop_background_sync()
    assert gw._running is False
