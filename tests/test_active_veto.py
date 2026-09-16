"""Unit tests for Active Radar Veto and Kinematic Slump Detection in DualFusionEngine."""

import numpy as np
import pytest

from hub.alert_dispatcher import AlertDispatcher
from hub.csi_pipeline.pca_features import CSIDynamicFeatures
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.mmwave_pipeline.radar_receiver import RadarFallState, RadarPosture, RadarTelemetry


@pytest.fixture
def alert_dispatcher(tmp_path):
    return AlertDispatcher(enable_sound=False, log_dir=str(tmp_path))


def test_active_radar_veto_suppresses_csi_false_alarm(alert_dispatcher):
    """Verify radar standing posture suppresses CSI fall false alarms when veto is enabled."""
    engine = DualFusionEngine(
        mode=OperatingMode.FUSION,
        alert_dispatcher=alert_dispatcher,
        enable_radar_veto=True,
    )
    t0 = 1000.0

    # Radar tracks occupant as actively standing upright
    standing_radar = RadarTelemetry(
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.STANDING,
        target_height_m=1.70,
        dwell_time_sec=0,
    )
    engine.update_radar(standing_radar, current_time=t0)
    assert engine.unified_state == UnifiedFallState.NORMAL

    # Spurious CSI disturbance (e.g. dropping a heavy object or slamming a door)
    burst1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.4,
        energy_surge_ratio=4.5,
        moving_variance=1.8,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    burst2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.1,
        energy_surge_ratio=4.2,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    stillness = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.02,
        energy_surge_ratio=1.0,
        moving_variance=0.02,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )

    engine.update_csi(burst1, current_time=t0)
    engine.update_csi(burst2, current_time=t0 + 0.1)
    engine.update_csi(stillness, current_time=t0 + 4.5)

    # Without veto, CSI would confirm fall. With Active Radar Veto, it is suppressed to SUSPECTED
    assert engine.unified_state == UnifiedFallState.SUSPECTED
    assert engine.veto_count >= 1
    assert "Radar verifies upright standing" in engine.last_veto_reason


def test_radar_veto_disabled_preserves_legacy_behavior(alert_dispatcher):
    """Verify legacy consensus (OR logic) operates when veto is disabled."""
    engine = DualFusionEngine(
        mode=OperatingMode.FUSION,
        alert_dispatcher=alert_dispatcher,
        enable_radar_veto=False,
    )
    t0 = 1000.0

    standing_radar = RadarTelemetry(
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.STANDING,
        target_height_m=1.65,
        dwell_time_sec=0,
    )
    engine.update_radar(standing_radar, current_time=t0)

    burst1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.3,
        energy_surge_ratio=4.0,
        moving_variance=1.6,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    burst2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.0,
        energy_surge_ratio=3.8,
        moving_variance=1.4,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    stillness = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.02,
        energy_surge_ratio=1.0,
        moving_variance=0.02,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )

    engine.update_csi(burst1, current_time=t0)
    engine.update_csi(burst2, current_time=t0 + 0.1)
    engine.update_csi(stillness, current_time=t0 + 4.5)

    # Without veto enabled, CSI alone escalates to CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED


def test_radar_floor_height_allows_confirmation(alert_dispatcher):
    """When occupant is on floor, active veto does not trigger and fall confirms."""
    engine = DualFusionEngine(
        mode=OperatingMode.FUSION,
        alert_dispatcher=alert_dispatcher,
        enable_radar_veto=True,
    )
    t0 = 1000.0

    floor_radar = RadarTelemetry(
        fall_state=RadarFallState.CONFIRMED,
        posture=RadarPosture.LYING,
        target_height_m=0.22,
        dwell_time_sec=4,
    )
    state = engine.update_radar(floor_radar, current_time=t0)
    assert state == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED


def test_kinematic_slump_detection_without_csi_burst(alert_dispatcher):
    """Detect slow sliding fall from chair (Z=0.85m -> 0.25m) without high Doppler burst."""
    engine = DualFusionEngine(
        mode=OperatingMode.FUSION,
        alert_dispatcher=alert_dispatcher,
        enable_slump_detection=True,
    )
    t0 = 2000.0

    # 1. Occupant initially sitting in chair at 0.85m
    chair_telemetry = RadarTelemetry(
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.SITTING,
        target_height_m=0.85,
        dwell_time_sec=0,
    )
    engine.update_radar(chair_telemetry, current_time=t0)
    assert engine.unified_state == UnifiedFallState.NORMAL

    # 2. Midway slump at t0 + 1.2s
    mid_telemetry = RadarTelemetry(
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.SITTING,
        target_height_m=0.55,
        dwell_time_sec=0,
    )
    engine.update_radar(mid_telemetry, current_time=t0 + 1.2)

    # 3. Slump to floor at t0 + 2.5s (lying down, dwell 2s)
    floor_telemetry = RadarTelemetry(
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.LYING,
        target_height_m=0.25,
        dwell_time_sec=2,
    )
    state = engine.update_radar(floor_telemetry, current_time=t0 + 2.5)

    assert engine.slump_detected is True
    assert state == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED
