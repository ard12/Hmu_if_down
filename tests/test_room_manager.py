"""Tests for RoomManager multi-room mesh routing (Milestone 6.1)."""

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.room_manager import RoomManager, RoomContext, HANDOVER_TIMEOUT_SEC


# ---------------------------------------------------------------------------
# RoomContext unit tests
# ---------------------------------------------------------------------------

def test_room_context_is_inactive_before_first_packet():
    """A freshly created RoomContext has no packets and should be inactive."""
    ctx = RoomContext(room_id=1)
    assert not ctx.is_active
    assert ctx.last_seen == 0.0


def test_room_context_is_active_after_packet():
    """After setting last_csi_time, the context should report is_active=True."""
    ctx = RoomContext(room_id=2)
    ctx.last_csi_time = time.time()
    assert ctx.is_active


def test_room_context_goes_inactive_after_timeout():
    """Context with a packet timestamp older than HANDOVER_TIMEOUT_SEC is inactive."""
    ctx = RoomContext(room_id=3)
    ctx.last_csi_time = time.time() - HANDOVER_TIMEOUT_SEC - 1.0
    assert not ctx.is_active


def test_room_context_to_dict_schema():
    """to_dict() must contain the expected keys."""
    ctx = RoomContext(room_id=4)
    d = ctx.to_dict()
    for key in ("room_id", "is_active", "latest_state", "last_seen", "alert_count", "seconds_since_packet"):
        assert key in d, f"Missing key: {key}"
    assert d["room_id"] == 4


# ---------------------------------------------------------------------------
# RoomManager factory tests
# ---------------------------------------------------------------------------

def test_room_manager_creates_new_context():
    """get_or_create_room() must return a RoomContext for a new room_id."""
    mgr = RoomManager()
    ctx = mgr.get_or_create_room(room_id=10)
    assert isinstance(ctx, RoomContext)
    assert ctx.room_id == 10


def test_room_manager_returns_same_context_on_second_call():
    """Calling get_or_create_room() twice with the same ID returns the same object."""
    mgr = RoomManager()
    ctx1 = mgr.get_or_create_room(room_id=11)
    ctx2 = mgr.get_or_create_room(room_id=11)
    assert ctx1 is ctx2


def test_room_manager_isolates_different_rooms():
    """Two different room IDs must produce independent RoomContext objects."""
    mgr = RoomManager()
    ctx_a = mgr.get_or_create_room(room_id=20)
    ctx_b = mgr.get_or_create_room(room_id=21)
    assert ctx_a is not ctx_b
    assert ctx_a.room_id != ctx_b.room_id


def test_room_manager_get_room_returns_none_for_unknown():
    """get_room() must return None for a room_id that was never created."""
    mgr = RoomManager()
    assert mgr.get_room(room_id=99) is None


def test_room_manager_remove_room():
    """remove_room() must delete the context; subsequent get_room() returns None."""
    mgr = RoomManager()
    mgr.get_or_create_room(room_id=30)
    removed = mgr.remove_room(room_id=30)
    assert removed is True
    assert mgr.get_room(room_id=30) is None


def test_room_manager_remove_nonexistent_returns_false():
    """remove_room() for an unknown room_id must return False (not raise)."""
    mgr = RoomManager()
    assert mgr.remove_room(room_id=999) is False


def test_get_active_rooms_returns_list():
    """get_active_rooms() must return a list of dicts."""
    mgr = RoomManager()
    mgr.get_or_create_room(room_id=40)
    mgr.get_or_create_room(room_id=41)
    rooms = mgr.get_active_rooms()
    assert isinstance(rooms, list)
    assert len(rooms) == 2
    assert all(isinstance(r, dict) for r in rooms)


def test_room_manager_on_csi_packet_updates_timestamp():
    """on_csi_packet() must update the room's last_csi_time."""
    mgr = RoomManager()

    class _FakeCsiFeat:
        moving_variance = 0.05

    before = time.time()
    mgr.on_csi_packet(room_id=50, features=_FakeCsiFeat())
    ctx = mgr.get_room(room_id=50)
    assert ctx is not None
    assert ctx.last_csi_time >= before


def test_room_manager_on_radar_packet_updates_timestamp():
    """on_radar_packet() must update the room's last_radar_time."""
    mgr = RoomManager()
    before = time.time()
    mgr.on_radar_packet(room_id=60, telemetry=object())
    ctx = mgr.get_room(room_id=60)
    assert ctx is not None
    assert ctx.last_radar_time >= before


# ---------------------------------------------------------------------------
# REST endpoint tests (via TestClient)
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from hub.dashboard.app import app, broadcaster

client = TestClient(app)


def test_api_rooms_no_manager_returns_empty():
    """GET /api/rooms without a RoomManager returns total_rooms=0."""
    broadcaster.room_manager = None
    resp = client.get("/api/rooms")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rooms"] == 0


def test_api_rooms_with_manager_returns_room_list():
    """GET /api/rooms with a RoomManager lists registered rooms."""
    mgr = RoomManager()
    mgr.get_or_create_room(room_id=1)
    mgr.get_or_create_room(room_id=2)
    broadcaster.room_manager = mgr
    try:
        resp = client.get("/api/rooms")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rooms"] == 2
        room_ids = [r["room_id"] for r in data["rooms"]]
        assert set(room_ids) == {1, 2}
    finally:
        broadcaster.room_manager = None


def test_api_room_calibrate_404_for_missing_room():
    """POST /api/rooms/99/calibrate must return 404 when room 99 doesn't exist."""
    mgr = RoomManager()
    broadcaster.room_manager = mgr
    try:
        resp = client.post("/api/rooms/99/calibrate")
        assert resp.status_code == 404
    finally:
        broadcaster.room_manager = None
