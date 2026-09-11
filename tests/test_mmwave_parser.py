"""Unit tests for mmWave Radar packet decoding and dual fusion."""

import pytest
from hub.mmwave_pipeline.radar_receiver import RadarReceiver, RadarFallState, RadarPosture, RadarTelemetry
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.csi_pipeline.multi_link_fusion import CSIFallState


def test_radar_json_datagram():
    receiver = RadarReceiver()
    json_str = '{"fall": 2, "posture": 3, "height": 0.28, "dwell": 6}'
    telemetry = receiver.parse_json_datagram(json_str)

    assert telemetry is not None
    assert telemetry.fall_state == RadarFallState.CONFIRMED
    assert telemetry.posture == RadarPosture.LYING
    assert pytest.approx(telemetry.target_height_m, abs=0.01) == 0.28
    assert telemetry.dwell_time_sec == 6


def test_radar_binary_frame_valid():
    receiver = RadarReceiver()

    # Frame structure:
    # Header: 0x53 0x59
    # Ctrl: 0x02 (Fall), Cmd: 0x01 (Status), Len: 0x0001, Data: 0x02 (Confirmed fall)
    frame_body = bytearray([0x53, 0x59, 0x02, 0x01, 0x00, 0x01, 0x02])
    checksum = sum(frame_body) & 0xFF
    full_frame = bytes(frame_body + bytearray([checksum]))

    telemetry = receiver.parse_binary_frame(full_frame)
    assert telemetry is not None
    assert telemetry.fall_state == RadarFallState.CONFIRMED


def test_radar_binary_frame_bad_checksum():
    receiver = RadarReceiver()
    # Frame with wrong checksum
    bad_frame = bytes([0x53, 0x59, 0x02, 0x01, 0x00, 0x01, 0x02, 0x00])
    telemetry = receiver.parse_binary_frame(bad_frame)
    assert telemetry is None


def test_dual_fusion_consensus():
    engine = DualFusionEngine(mode=OperatingMode.FUSION)

    # Radar reports floor posture and height
    radar_t = RadarTelemetry(
        fall_state=RadarFallState.CONFIRMED,
        posture=RadarPosture.LYING,
        target_height_m=0.20,
        dwell_time_sec=4,
    )
    engine.update_radar(radar_t)

    # In fusion mode, confirmed radar fall immediately triggers consensus confirmed fall
    assert engine.unified_state == UnifiedFallState.CONFIRMED
