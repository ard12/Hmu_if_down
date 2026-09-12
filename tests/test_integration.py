"""Integration tests covering multi-modal fall detection pipeline gaps.

Tests include:
- DualFusionEngine in CSI_ONLY, RADAR_ONLY, and FUSION operating modes
- Cross-verification between Wi-Fi CSI and mmWave Radar modalities
- False positive escalation behavior in FUSION mode
- NodeBuffer sequence gap detection and linear interpolation
- NodeBuffer heartbeat and liveness tracking
- AlertDispatcher rate-limiting cooldown and CSV logging
- MultiLinkFusionEngine state recovery following confirmed fall
"""

import csv
from pathlib import Path
import sys
import time
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.alert_dispatcher import AlertDispatcher
from hub.csi_pipeline.multi_link_fusion import CSIFallState, MultiLinkFusionEngine
from hub.csi_pipeline.pca_features import CSIDynamicFeatures
from hub.csi_pipeline.preprocessor import CSIPacket
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.mmwave_pipeline.radar_receiver import RadarFallState, RadarPosture, RadarTelemetry
from hub.server import NodeBuffer


@pytest.fixture
def alert_dispatcher(tmp_path):
    """Provide an isolated AlertDispatcher with audio disabled for integration tests."""
    return AlertDispatcher(enable_sound=False, log_dir=str(tmp_path))


def test_fusion_csi_only_mode(alert_dispatcher):
    """DualFusionEngine in CSI_ONLY mode confirms fall on coincident bursts + quiescence."""
    engine = DualFusionEngine(mode=OperatingMode.CSI_ONLY, alert_dispatcher=alert_dispatcher)
    t0 = 1000.0

    burst_n1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.2,
        energy_surge_ratio=4.0,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    burst_n2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.0,
        energy_surge_ratio=3.8,
        moving_variance=1.2,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    quiescence_feature = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.05,
        energy_surge_ratio=1.0,
        moving_variance=0.03,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )

    # 1. Single burst -> NORMAL
    s1 = engine.update_csi(burst_n1, current_time=t0)
    assert s1 == UnifiedFallState.NORMAL

    # 2. Coincident burst within window -> SUSPECTED
    s2 = engine.update_csi(burst_n2, current_time=t0 + 0.1)
    assert s2 == UnifiedFallState.SUSPECTED

    # 3. Post-impact stillness sustained for > 4.0s -> CONFIRMED
    s3 = engine.update_csi(quiescence_feature, current_time=t0 + 4.5)
    assert s3 == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED


def test_fusion_radar_only_mode(alert_dispatcher):
    """DualFusionEngine in RADAR_ONLY mode confirms fall on verified radar telemetry."""
    engine = DualFusionEngine(mode=OperatingMode.RADAR_ONLY, alert_dispatcher=alert_dispatcher)
    telemetry = RadarTelemetry(
        fall_state=RadarFallState.CONFIRMED,
        posture=RadarPosture.LYING,
        target_height_m=0.20,
        dwell_time_sec=5,
    )
    result_state = engine.update_radar(telemetry)
    assert result_state == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED


def test_fusion_mode_cross_verification(alert_dispatcher):
    """In FUSION mode, radar fall confirmation combined with CSI suspected fall triggers CONFIRMED."""
    engine = DualFusionEngine(mode=OperatingMode.FUSION, alert_dispatcher=alert_dispatcher)
    t0 = 1000.0

    burst_n1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.2,
        energy_surge_ratio=4.0,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    burst_n2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.0,
        energy_surge_ratio=3.8,
        moving_variance=1.2,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )

    # Trigger CSI suspected state
    engine.update_csi(burst_n1, current_time=t0)
    s_csi = engine.update_csi(burst_n2, current_time=t0 + 0.1)
    assert engine.last_csi_state == CSIFallState.SUSPECTED_FALL
    assert s_csi == UnifiedFallState.SUSPECTED

    # Radar reports floor posture and height
    radar_fall = RadarTelemetry(
        fall_state=RadarFallState.CONFIRMED,
        posture=RadarPosture.LYING,
        target_height_m=0.20,
        dwell_time_sec=3,
    )
    result_state = engine.update_radar(radar_fall)
    assert result_state == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED


def test_fusion_false_positive_rejection(alert_dispatcher):
    """CSI detects velocity burst + quiescence but radar shows standing at 1.6m.

    In FUSION mode, verify the system still escalates to CONFIRMED because
    either modality confirming currently triggers the CONFIRMED state.
    """
    engine = DualFusionEngine(mode=OperatingMode.FUSION, alert_dispatcher=alert_dispatcher)
    t0 = 1000.0

    # Radar initially reports standing posture at normal standing height
    standing_radar = RadarTelemetry(
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.STANDING,
        target_height_m=1.6,
        dwell_time_sec=0,
    )
    engine.update_radar(standing_radar)
    assert engine.unified_state == UnifiedFallState.NORMAL

    # CSI witnesses coincident velocity bursts followed by post-impact stillness
    burst_n1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.2,
        energy_surge_ratio=4.0,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    burst_n2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.0,
        energy_surge_ratio=3.8,
        moving_variance=1.2,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    quiescence_feature = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.05,
        energy_surge_ratio=1.0,
        moving_variance=0.03,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )

    engine.update_csi(burst_n1, current_time=t0)
    engine.update_csi(burst_n2, current_time=t0 + 0.1)
    engine.update_csi(quiescence_feature, current_time=t0 + 4.5)

    # In current fusion logic, CSI confirmation alone escalates to CONFIRMED
    assert engine.last_csi_state == CSIFallState.CONFIRMED_FALL
    assert engine.unified_state == UnifiedFallState.CONFIRMED

    # Re-updating radar with standing telemetry still yields CONFIRMED due to consensus OR rule
    result_state = engine.update_radar(standing_radar)
    assert result_state == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED


def test_node_buffer_sequence_gap_interpolation():
    """NodeBuffer detects sequence gap of 2 packets and generates linear interpolations."""
    buffer = NodeBuffer(window_size=100)

    phases = np.zeros(2)
    p0 = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=100, seq_num=0, amplitudes=np.array([10.0, 20.0]), phases=phases)
    p1 = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=110, seq_num=1, amplitudes=np.array([10.0, 20.0]), phases=phases)
    p2 = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=120, seq_num=2, amplitudes=np.array([10.0, 20.0]), phases=phases)
    p5 = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=150, seq_num=5, amplitudes=np.array([40.0, 50.0]), phases=phases)

    int0 = buffer.push(p0)
    int1 = buffer.push(p1)
    int2 = buffer.push(p2)
    assert int0 == 0 and int1 == 0 and int2 == 0

    # Sequence jumps from 2 to 5 -> gap of 2 frames (seq 3 and 4)
    int5 = buffer.push(p5)
    assert int5 == 2

    # Buffer should hold 6 frames: 3 initial real + 2 interpolated + 1 latest real
    assert len(buffer.amplitudes) == 6
    assert buffer.total_gaps == 2
    assert buffer.total_received == 4

    # Verify linear interpolation between p2 (10, 20) and p5 (40, 50)
    np.testing.assert_allclose(buffer.amplitudes[3], np.array([20.0, 30.0]))
    np.testing.assert_allclose(buffer.amplitudes[4], np.array([30.0, 40.0]))
    np.testing.assert_allclose(buffer.amplitudes[5], np.array([40.0, 50.0]))


def test_node_buffer_out_of_order_and_reboot():
    """NodeBuffer correctly drops late out-of-order packets and resets on tracker reboot."""
    buffer = NodeBuffer(window_size=100)
    phases = np.zeros(2)

    # Receive packets 10, 11, 12
    for s in [10, 11, 12]:
        p = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=100 + s * 10, seq_num=s, amplitudes=np.array([float(s), float(s)]), phases=phases)
        assert buffer.push(p) == 0

    assert buffer.last_seq == 12
    assert len(buffer.amplitudes) == 3

    # Late out-of-order packet with seq_num=11 arrives late
    p_late = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=125, seq_num=11, amplitudes=np.array([11.0, 11.0]), phases=phases)
    assert buffer.push(p_late) == 0
    # Should not corrupt last_seq or append duplicates
    assert buffer.last_seq == 12
    assert len(buffer.amplitudes) == 3

    # Now simulate node reboot: seq_num drops from 12 to 0 (after having run a long time, e.g. seq=1500)
    buffer.last_seq = 1500
    p_reboot = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=10, seq_num=0, amplitudes=np.array([0.0, 0.0]), phases=phases)
    buffer.push(p_reboot)
    assert buffer.last_seq == 0


def test_node_buffer_is_alive(monkeypatch):
    """NodeBuffer tracks liveness based on timeout threshold from last received packet."""
    buffer = NodeBuffer()
    assert buffer.is_alive is False

    fake_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: fake_time)

    p0 = CSIPacket(node_id=1, rssi=-50, subcarrier_count=2, timestamp_ms=100, seq_num=0, amplitudes=np.array([10.0, 20.0]), phases=np.zeros(2))
    buffer.push(p0)

    # Immediately after packet reception, node is alive
    assert buffer.is_alive is True

    # Advance time by 15s (greater than NODE_TIMEOUT_SEC = 10.0s)
    fake_time += 15.0
    assert buffer.is_alive is False


def test_alert_dispatcher_cooldown(tmp_path):
    """AlertDispatcher enforces cooldown window and rate-limits incident logging to CSV."""
    dispatcher = AlertDispatcher(cooldown_sec=2.0, enable_sound=False, log_dir=str(tmp_path))

    # Trigger alarm twice within 1 second (< cooldown_sec of 2.0s)
    dispatcher.trigger_alarm("fusion", "FALL_CONFIRMED", "First alarm trigger")
    dispatcher.trigger_alarm("fusion", "FALL_CONFIRMED", "Second alarm trigger within cooldown")

    assert dispatcher.csv_path.exists()
    with open(dispatcher.csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # First row is header
    assert rows[0] == ["Timestamp", "Modality", "Event", "Details"]

    # Only one alarm event should be logged due to cooldown rate limiting
    data_rows = rows[1:]
    assert len(data_rows) == 1
    assert data_rows[0][1] == "fusion"
    assert data_rows[0][2] == "FALL_CONFIRMED"
    assert data_rows[0][3] == "First alarm trigger"


def test_multi_link_recovery():
    """MultiLinkFusionEngine transitions from CONFIRMED_FALL to RECOVERED on post-fall motion."""
    engine = MultiLinkFusionEngine(
        coincidence_window_sec=0.4,
        min_coincident_links=2,
        post_fall_quiescence_sec=4.0,
        motionless_variance_threshold=0.08,
    )
    t0 = 1000.0

    # 1. Coincident bursts from 2 links -> SUSPECTED_FALL
    f_burst1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.0,
        energy_surge_ratio=4.0,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    f_burst2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=1.9,
        energy_surge_ratio=3.5,
        moving_variance=1.2,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    engine.register_feature(f_burst1, current_time=t0)
    engine.register_feature(f_burst2, current_time=t0 + 0.1)
    assert engine.state == CSIFallState.SUSPECTED_FALL

    # 2. Stillness for > 4.0 seconds -> CONFIRMED_FALL
    f_still = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.05,
        energy_surge_ratio=1.0,
        moving_variance=0.02,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )
    engine.register_feature(f_still, current_time=t0 + 4.5)
    assert engine.state == CSIFallState.CONFIRMED_FALL

    # 3. High velocity (>1.2 m/s) and high variance (> 3 * motionless_var_thresh = 0.24) -> RECOVERED
    f_recovery = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=1.8,
        energy_surge_ratio=3.0,
        moving_variance=0.35,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    state = engine.register_feature(f_recovery, current_time=t0 + 6.0)
    assert state == CSIFallState.RECOVERED
    assert engine.state == CSIFallState.RECOVERED
    assert engine.suspected_timestamp is None
