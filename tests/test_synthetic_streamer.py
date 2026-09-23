"""Tests for Synthetic CSI & Radar Streamer.

@req SRS-SIM-002
"""
import pytest
from hub.packet_parser import parse_v2_header
from hub.simulation.synthetic_streamer import (
    FallScenario,
    ScenarioPhase,
    SyntheticStreamer,
    TrajectoryKeyframe,
)


def test_forward_trip_scenario_has_keyframes():
    scenario = SyntheticStreamer.generate_forward_trip(10.0)
    assert len(scenario.keyframes) >= 5
    assert scenario.name == "forward_trip"
    assert scenario.fall_type == "forward_trip"


def test_syncope_scenario_ends_in_lying():
    scenario = SyntheticStreamer.generate_syncope_drop(10.0)
    assert scenario.keyframes[-1].phase == ScenarioPhase.LYING
    assert scenario.keyframes[-1].position_z < 0.5


def test_interpolate_at_exact_keyframe():
    streamer = SyntheticStreamer()
    scenario = SyntheticStreamer.generate_forward_trip(10.0)
    kf = streamer.interpolate_keyframes(scenario, 0.0)
    assert kf.time_s == 0.0
    assert kf.position_x == 2.0
    assert kf.position_z == 1.7
    assert kf.phase == ScenarioPhase.STANDING


def test_interpolate_between_keyframes():
    streamer = SyntheticStreamer()
    scenario = SyntheticStreamer.generate_forward_trip(10.0)
    # Between 4.0s (3.0, 2.0, 1.7) and 5.0s (3.5, 2.0, 1.5), at 4.5s:
    # x = 3.25, z = 1.6
    kf = streamer.interpolate_keyframes(scenario, 4.5)
    assert abs(kf.position_x - 3.25) < 1e-4
    assert abs(kf.position_z - 1.6) < 1e-4


def test_build_csi_packet_magic_bytes():
    streamer = SyntheticStreamer()
    kf = TrajectoryKeyframe(1.0, 2.0, 2.0, 1.7, 0.0, ScenarioPhase.STANDING)
    pkt = streamer.build_csi_packet(kf, node_id=1, room_id=2, seq=10)
    assert pkt[:4] == b"CSIF"


def test_build_csi_packet_length():
    streamer = SyntheticStreamer()
    kf = TrajectoryKeyframe(1.0, 2.0, 2.0, 1.7, 0.0, ScenarioPhase.STANDING)
    pkt = streamer.build_csi_packet(kf, node_id=1, room_id=2, seq=10)
    # 18 bytes header + 64 bytes subcarrier amplitudes
    assert len(pkt) == 18 + 64


def test_scenario_count_increments():
    streamer = SyntheticStreamer(packet_rate_hz=20)
    assert streamer.scenarios_played == 0
    scenario = SyntheticStreamer.generate_forward_trip(duration_s=0.5)
    packets = streamer.stream_scenario(scenario, send_udp=False)
    assert packets == 10
    assert streamer.scenarios_played == 1


def test_packet_compatible_with_hub_packet_parser():
    streamer = SyntheticStreamer()
    kf = TrajectoryKeyframe(2.5, 2.0, 2.0, 1.7, 1.2, ScenarioPhase.WALKING)
    pkt = streamer.build_csi_packet(kf, node_id=3, room_id=4, seq=42)

    parsed = parse_v2_header(pkt)
    assert parsed is not None
    assert parsed["magic"] == b"CSIF"
    assert parsed["node_id"] == 3
    assert parsed["room_id"] == 4
    assert parsed["subcarrier_count"] == 64
    assert parsed["seq_num"] == 42
    assert parsed["payload_offset"] == 18
