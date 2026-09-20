"""Tests for Mesh Topology and Position REST API Endpoints (Milestone 13.4)."""

import time
from fastapi.testclient import TestClient
import pytest
from hub.dashboard.app import app, broadcaster
from hub.room_handoff import RoomHandoffManager
from hub.triangulation import TriangulationEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_api_mesh_topology_returns_200(client):
    """GET /api/mesh/topology returns 200 with mesh_enabled, root_node, and nodes."""
    res = client.get("/api/mesh/topology")
    assert res.status_code == 200
    data = res.json()
    assert data["mesh_enabled"] is True
    assert data["root_node"] == 1
    assert "nodes" in data
    assert isinstance(data["nodes"], list)


def test_api_position_current_returns_position_when_available(client):
    """GET /api/position/current returns position when engine has measurements."""
    engine = TriangulationEngine({
        "1": (0.0, 0.0),
        "2": (5.0, 0.0),
        "3": (2.5, 5.0),
    })
    now = time.time()
    engine.add_measurement("1", phase_slope=3.0, timestamp=now)
    engine.add_measurement("2", phase_slope=3.5, timestamp=now)
    engine.add_measurement("3", phase_slope=2.5, timestamp=now)

    broadcaster.triangulation_engine = engine

    res = client.get("/api/position/current")
    assert res.status_code == 200
    pos = res.json()
    assert "x" in pos
    assert "y" in pos
    assert "confidence" in pos
    assert pos["n_nodes_used"] == 3


def test_api_position_current_returns_404_when_no_measurements(client):
    """GET /api/position/current returns 404 when engine has no fresh measurements."""
    engine = TriangulationEngine({
        "1": (0.0, 0.0),
        "2": (5.0, 0.0),
        "3": (2.5, 5.0),
    })
    broadcaster.triangulation_engine = engine

    res = client.get("/api/position/current")
    assert res.status_code == 404
    assert "error" in res.json()


def test_api_position_history_returns_history_list(client):
    """GET /api/position/history returns history list."""
    engine = TriangulationEngine({
        "1": (0.0, 0.0),
        "2": (5.0, 0.0),
        "3": (2.5, 5.0),
    })
    now = time.time()
    engine.add_measurement("1", phase_slope=3.0, timestamp=now)
    engine.add_measurement("2", phase_slope=3.5, timestamp=now)
    engine.add_measurement("3", phase_slope=2.5, timestamp=now)
    engine.estimate_position(current_time=now)

    broadcaster.triangulation_engine = engine

    res = client.get("/api/position/history")
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
    assert len(data["history"]) >= 1


def test_api_mesh_topology_reports_handoff_rooms(client):
    """GET /api/mesh/topology reports active handoff rooms when manager is active."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_101": ["room_102"]})
    mgr.set_subject_room("default", "room_101")
    broadcaster.handoff_manager = mgr

    res = client.get("/api/mesh/topology")
    assert res.status_code == 200
    data = res.json()
    assert data["active_handoff_rooms"] == ["room_101"]


def test_api_position_current_with_floor_plan_returns_room_id(client):
    """GET /api/position/current with floor plan returns mapped room_id."""
    engine = TriangulationEngine({
        "1": (0.0, 0.0),
        "2": (10.0, 0.0),
        "3": (0.0, 10.0),
    })
    engine.set_floor_plan({
        "kitchen": [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)],
        "lounge": [(6.0, 0.0), (12.0, 0.0), (12.0, 6.0), (6.0, 6.0)],
    })
    now = time.time()
    # Position at (2.0, 2.0) in kitchen
    engine.add_measurement("1", phase_slope=2.8, timestamp=now)
    engine.add_measurement("2", phase_slope=8.2, timestamp=now)
    engine.add_measurement("3", phase_slope=8.2, timestamp=now)

    broadcaster.triangulation_engine = engine

    res = client.get("/api/position/current")
    assert res.status_code == 200
    pos = res.json()
    assert pos["room_id"] == "kitchen"
