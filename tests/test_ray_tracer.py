"""Tests for 3D Ray-Tracing RF Multipath Room Simulator.

@req SRS-SIM-001
"""
import numpy as np
import pytest
from hub.simulation.ray_tracer import (
    HumanTarget,
    Obstacle,
    RoomGeometry,
    RoomRayTracer,
    SensorPlacement,
    Vec3,
    WallMaterial,
)


@pytest.fixture
def sample_room():
    return RoomGeometry(length_m=6.0, width_m=5.0, height_m=3.0)


@pytest.fixture
def sample_sensors():
    return [
        SensorPlacement(position=Vec3(0.5, 0.5, 2.0), node_id=1),
        SensorPlacement(position=Vec3(5.5, 4.5, 2.0), node_id=2),
        SensorPlacement(position=Vec3(0.5, 4.5, 2.0), node_id=3),
    ]


def test_empty_room_produces_valid_csi(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors[:2], max_reflections=1)
    results = tracer.trace()
    assert len(results) == 1
    csi = results[0]
    assert csi.tx_node_id == 1
    assert csi.rx_node_id == 2
    assert len(csi.subcarrier_amplitudes) == 64
    assert len(csi.subcarrier_phases) == 64
    assert -90.0 <= csi.rssi_dbm <= -10.0
    assert csi.path_count > 0


def test_subcarrier_count_matches_config(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors[:2], n_subcarriers=32)
    results = tracer.trace()
    assert len(results) == 1
    assert len(results[0].subcarrier_amplitudes) == 32
    assert len(results[0].subcarrier_phases) == 32


def test_path_loss_increases_with_distance(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors[:2])
    pl_1m = tracer._compute_path_loss(1.0)
    pl_5m = tracer._compute_path_loss(5.0)
    pl_10m = tracer._compute_path_loss(10.0)
    assert pl_1m < pl_5m < pl_10m


def test_path_loss_zero_distance(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors[:2])
    assert tracer._compute_path_loss(0.0) == 0.0
    assert tracer._compute_path_loss(-2.0) == 0.0


def test_human_target_attenuates_signal(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors[:2], max_reflections=0)
    empty_results = tracer.trace()

    # Place human in direct line between sensor 1 (0.5, 0.5) and sensor 2 (5.5, 4.5)
    target = HumanTarget(position=Vec3(3.0, 2.5, 1.0), height_m=1.7, radius_m=0.3, is_fallen=False)
    occupied_results = tracer.trace(targets=[target])

    assert np.mean(occupied_results[0].subcarrier_amplitudes) <= np.mean(
        empty_results[0].subcarrier_amplitudes
    )


def test_fallen_target_changes_csi_pattern(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors[:2], max_reflections=1)
    standing = HumanTarget(position=Vec3(3.0, 2.5, 1.0), height_m=1.7, radius_m=0.3, is_fallen=False)
    fallen = HumanTarget(position=Vec3(3.0, 2.5, 0.2), height_m=1.7, radius_m=0.3, is_fallen=True)

    res_standing = tracer.trace(targets=[standing])
    res_fallen = tracer.trace(targets=[fallen])

    # Amplitudes or phases should differ
    amp_diff = np.abs(
        res_standing[0].subcarrier_amplitudes - res_fallen[0].subcarrier_amplitudes
    )
    assert np.any(amp_diff > 0.0)


def test_multiple_sensor_pairs(sample_room, sample_sensors):
    tracer = RoomRayTracer(sample_room, sample_sensors)
    results = tracer.trace()
    # 3 sensors: pairs (1,2), (1,3), (2,3) -> 3 pairs
    assert len(results) == 3
    pairs = {(r.tx_node_id, r.rx_node_id) for r in results}
    assert pairs == {(1, 2), (1, 3), (2, 3)}


def test_optimal_placement_returns_requested_count(sample_room):
    tracer = RoomRayTracer(sample_room, [])
    placements = tracer.optimal_sensor_placement(n_sensors=4)
    assert len(placements) == 4
    for p in placements:
        assert isinstance(p, Vec3)
        assert 0.0 <= p.x <= sample_room.length_m
        assert 0.0 <= p.y <= sample_room.width_m
        assert p.z == 2.0


def test_image_source_count_increases_with_reflections(sample_room, sample_sensors):
    tracer1 = RoomRayTracer(sample_room, sample_sensors[:2], max_reflections=1)
    tracer3 = RoomRayTracer(sample_room, sample_sensors[:2], max_reflections=3)
    res1 = tracer1.trace()
    res3 = tracer3.trace()
    assert res3[0].path_count > res1[0].path_count


def test_vec3_to_and_from_array():
    v = Vec3(1.2, 3.4, 5.6)
    arr = v.to_array()
    assert np.allclose(arr, [1.2, 3.4, 5.6])
    v2 = Vec3.from_array([7.8, 9.0, 1.1])
    assert v2.x == 7.8 and v2.y == 9.0 and v2.z == 1.1


def test_obstacles_affect_signal(sample_room, sample_sensors):
    obstacle = Obstacle(position=Vec3(3.0, 2.5, 1.0), size=Vec3(1.0, 1.0, 1.0), attenuation_db=10.0)
    tracer_with_obs = RoomRayTracer(sample_room, sample_sensors[:2], obstacles=[obstacle], max_reflections=1)
    tracer_without = RoomRayTracer(sample_room, sample_sensors[:2], obstacles=[], max_reflections=1)

    res_obs = tracer_with_obs.trace()
    res_clean = tracer_without.trace()
    assert np.mean(res_obs[0].subcarrier_amplitudes) <= np.mean(res_clean[0].subcarrier_amplitudes)
