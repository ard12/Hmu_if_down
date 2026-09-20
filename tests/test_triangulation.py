"""Tests for 2D CSI Phase-Differential Triangulation Engine (Milestone 13.3)."""

import math
import time
import pytest
from hub.triangulation import PositionUncertainError, TriangulationEngine


def test_known_geometry_produces_correct_position():
    """Known node geometry produces correct position within 0.5m of ground truth."""
    # Nodes placed at corners of a 10x10m area
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (10.0, 0.0),
        "node_2": (0.0, 10.0),
    })

    # True subject position: (3.0, 4.0)
    target_x, target_y = 3.0, 4.0
    d0 = math.hypot(target_x - 0.0, target_y - 0.0)    # 5.0
    d1 = math.hypot(target_x - 10.0, target_y - 0.0)   # sqrt(49 + 16) = sqrt(65)
    d2 = math.hypot(target_x - 0.0, target_y - 10.0)   # sqrt(9 + 36) = sqrt(45)

    now = 1000.0
    engine.add_measurement("node_0", phase_slope=d0, timestamp=now, rssi=-45.0)
    engine.add_measurement("node_1", phase_slope=d1, timestamp=now, rssi=-50.0)
    engine.add_measurement("node_2", phase_slope=d2, timestamp=now, rssi=-48.0)

    pos = engine.estimate_position(current_time=now)
    assert abs(pos["x"] - target_x) < 0.5
    assert abs(pos["y"] - target_y) < 0.5
    assert pos["confidence"] > 0.5
    assert pos["n_nodes_used"] == 3


def test_less_than_three_nodes_raises_uncertain_error():
    """PositionUncertainError raised when < 3 nodes reporting."""
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (10.0, 0.0),
        "node_2": (0.0, 10.0),
    })

    now = 2000.0
    engine.add_measurement("node_0", phase_slope=5.0, timestamp=now)
    engine.add_measurement("node_1", phase_slope=8.0, timestamp=now)

    with pytest.raises(PositionUncertainError) as exc_info:
        engine.estimate_position(current_time=now)
    assert "only 2 fresh" in str(exc_info.value)


def test_stale_measurements_excluded_from_wls():
    """Stale measurements (> 2s old) are excluded from WLS."""
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (10.0, 0.0),
        "node_2": (0.0, 10.0),
    })

    now = 3000.0
    # node_0 and node_1 fresh, node_2 stale (3.0s ago)
    engine.add_measurement("node_0", phase_slope=5.0, timestamp=now - 0.5)
    engine.add_measurement("node_1", phase_slope=8.0, timestamp=now - 0.8)
    engine.add_measurement("node_2", phase_slope=6.0, timestamp=now - 3.0)

    with pytest.raises(PositionUncertainError):
        engine.estimate_position(current_time=now)


def test_floor_plan_assigns_correct_room():
    """set_floor_plan + point-in-polygon correctly assigns room label."""
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (10.0, 0.0),
        "node_2": (0.0, 10.0),
    })
    engine.set_floor_plan({
        "bedroom": [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)],
        "living_room": [(5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 10.0)],
    })

    # Subject in bedroom at (2.0, 3.0)
    now = 4000.0
    d0 = math.hypot(2.0 - 0.0, 3.0 - 0.0)
    d1 = math.hypot(2.0 - 10.0, 3.0 - 0.0)
    d2 = math.hypot(2.0 - 0.0, 3.0 - 10.0)

    engine.add_measurement("node_0", phase_slope=d0, timestamp=now)
    engine.add_measurement("node_1", phase_slope=d1, timestamp=now)
    engine.add_measurement("node_2", phase_slope=d2, timestamp=now)

    pos = engine.estimate_position(current_time=now)
    assert pos["room_id"] == "bedroom"


def test_collinear_nodes_have_zero_confidence():
    """Confidence = 0 for degenerate geometry (all nodes collinear)."""
    # 3 nodes along x-axis
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (5.0, 0.0),
        "node_2": (10.0, 0.0),
    })

    now = 5000.0
    engine.add_measurement("node_0", phase_slope=3.0, timestamp=now)
    engine.add_measurement("node_1", phase_slope=2.0, timestamp=now)
    engine.add_measurement("node_2", phase_slope=7.0, timestamp=now)

    pos = engine.estimate_position(current_time=now)
    assert pos["confidence"] == 0.0


def test_boundary_point_assigned_to_nearest_room():
    """Boundary case: subject exactly on polygon edge assigned to nearest room."""
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (10.0, 0.0),
        "node_2": (0.0, 10.0),
    })
    engine.set_floor_plan({
        "room_A": [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
        "room_B": [(5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0)],
    })

    # Subject exactly on boundary x = 5.0, y = 2.5
    now = 6000.0
    d0 = math.hypot(5.0 - 0.0, 2.5 - 0.0)
    d1 = math.hypot(5.0 - 10.0, 2.5 - 0.0)
    d2 = math.hypot(5.0 - 0.0, 2.5 - 10.0)

    engine.add_measurement("node_0", phase_slope=d0, timestamp=now)
    engine.add_measurement("node_1", phase_slope=d1, timestamp=now)
    engine.add_measurement("node_2", phase_slope=d2, timestamp=now)

    pos = engine.estimate_position(current_time=now)
    assert pos["room_id"] in ("room_A", "room_B")


def test_rssi_weighting_favors_stronger_signals():
    """Higher RSSI nodes receive higher weight in WLS estimation."""
    engine = TriangulationEngine({
        "node_0": (0.0, 0.0),
        "node_1": (10.0, 0.0),
        "node_2": (0.0, 10.0),
    })

    now = 7000.0
    # True target at (2, 2)
    d0 = math.hypot(2.0 - 0.0, 2.0 - 0.0)
    d1 = math.hypot(2.0 - 10.0, 2.0 - 0.0)
    d2 = math.hypot(2.0 - 0.0, 2.0 - 10.0)

    # Node 0 has very strong RSSI (-35 dBm), Node 1 has very weak (-85 dBm)
    engine.add_measurement("node_0", phase_slope=d0, timestamp=now, rssi=-35.0)
    engine.add_measurement("node_1", phase_slope=d1 + 1.0, timestamp=now, rssi=-85.0)  # slightly noisy
    engine.add_measurement("node_2", phase_slope=d2, timestamp=now, rssi=-40.0)

    pos = engine.estimate_position(current_time=now)
    # Estimate should still be close to (2, 2) despite node 1 noise
    assert abs(pos["x"] - 2.0) < 1.0
    assert abs(pos["y"] - 2.0) < 1.0
