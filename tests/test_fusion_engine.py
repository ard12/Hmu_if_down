"""
Comprehensive Unit Tests for Dual-Sensor Fusion Engine (Phase 29, F-16).
Tests OperatingMode, Active Radar Veto, Slump Detection, and Consensus Evaluation.

@covers SRS-001, SRS-008, SRS-009, SRS-019
"""

import time
import pytest
from unittest.mock import MagicMock

from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.csi_pipeline.multi_link_fusion import CSIFallState
from hub.csi_pipeline.pca_features import CSIDynamicFeatures
from hub.mmwave_pipeline.radar_receiver import RadarFallState, RadarPosture, RadarTelemetry


@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock()
    return dispatcher


@pytest.fixture
def fusion_engine(mock_dispatcher):
    return DualFusionEngine(
        mode=OperatingMode.FUSION,
        alert_dispatcher=mock_dispatcher,
        enable_radar_veto=True,
        enable_slump_detection=True,
    )


def test_fusion_engine_initialization(fusion_engine):
    """Verifies default engine state and parameters."""
    assert fusion_engine.mode == OperatingMode.FUSION
    assert fusion_engine.unified_state == UnifiedFallState.NORMAL
    assert fusion_engine.enable_radar_veto is True
    assert fusion_engine.enable_slump_detection is True


def test_csi_only_mode_transition(mock_dispatcher):
    """Verifies operating in CSI_ONLY mode transitions on high ML probability."""
    engine = DualFusionEngine(mode=OperatingMode.CSI_ONLY, alert_dispatcher=mock_dispatcher)
    import numpy as np
    features = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.5,
        energy_surge_ratio=5.0,
        moving_variance=2.0,
        is_velocity_burst=True,
        pc1_signal=np.array([1.0, 2.0]),
        ml_fall_probability=0.96,
    )

    state = engine.update_csi(features, ml_prob=0.96)
    assert state == UnifiedFallState.CONFIRMED
    assert engine.high_confidence is True
    mock_dispatcher.trigger_alarm.assert_called_once()


def test_radar_only_mode_transition(mock_dispatcher):
    """Verifies operating in RADAR_ONLY mode transitions on radar confirmation."""
    engine = DualFusionEngine(mode=OperatingMode.RADAR_ONLY, alert_dispatcher=mock_dispatcher)
    telemetry = RadarTelemetry(
        target_height_m=0.25,
        fall_state=RadarFallState.CONFIRMED,
        posture=RadarPosture.LYING,
        dwell_time_sec=3,
        is_clutter=False,
    )

    state = engine.update_radar(telemetry)
    assert state == UnifiedFallState.CONFIRMED
    mock_dispatcher.trigger_alarm.assert_called_once()


def test_active_radar_veto_suppresses_false_alarm(fusion_engine):
    """
    Verifies that when mmWave radar detects an active upright standing posture (z > 1.1m),
    a suspected/confirmed CSI fall is vetoed and suppressed to SUSPECTED.
    """
    now = time.time()
    # 1. Radar observes upright person standing
    telemetry = RadarTelemetry(
        target_height_m=1.75,
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.STANDING,
        dwell_time_sec=10,
        is_clutter=False,
    )
    fusion_engine.update_radar(telemetry, current_time=now)

    # 2. CSI registers a false alarm burst with ml_prob >= 0.80
    import numpy as np
    features = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.5,
        energy_surge_ratio=5.0,
        moving_variance=2.0,
        is_velocity_burst=True,
        pc1_signal=np.array([1.0, 2.0]),
        ml_fall_probability=0.85,
    )
    state = fusion_engine.update_csi(features, current_time=now, ml_prob=0.85)

    # Active veto should prevent CONFIRMED state
    assert state == UnifiedFallState.SUSPECTED
    assert fusion_engine.veto_count >= 1
    assert "upright standing" in fusion_engine.last_veto_reason


def test_slump_detection_gradual_descent(fusion_engine):
    """
    Verifies gradual slump detection over 1.0 - 4.5s window from upright to floor.
    """
    t0 = 100.0
    # Standing/chair position at t0
    standing = RadarTelemetry(
        target_height_m=1.20,
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.STANDING,
        dwell_time_sec=5,
        is_clutter=False,
    )
    fusion_engine.update_radar(standing, current_time=t0)

    # 2.5s later: slumped to floor
    t1 = t0 + 2.5
    slumped = RadarTelemetry(
        target_height_m=0.30,
        fall_state=RadarFallState.NONE,
        posture=RadarPosture.LYING,
        dwell_time_sec=3,
        is_clutter=False,
    )
    state = fusion_engine.update_radar(slumped, current_time=t1)

    assert fusion_engine.slump_detected is True
    assert state == UnifiedFallState.CONFIRMED


def test_radar_point_cloud_joint_estimation(mock_dispatcher):
    """
    Verifies point cloud processing and joint angle estimation when skeleton fitter is wired.
    """
    mock_fitter = MagicMock()
    mock_fitter.fit.return_value = {
        "valid": True,
        "torso_top": [0.0, 0.0, 0.3],
        "torso_bottom": [0.0, 0.0, 0.2],
    }

    mock_angles = MagicMock()
    mock_angles.classify_posture.return_value = "FALLEN"
    mock_angles.trunk_inclination.return_value = 85.0

    engine = DualFusionEngine(
        mode=OperatingMode.FUSION,
        alert_dispatcher=mock_dispatcher,
        skeleton_fitter=mock_fitter,
        joint_angle_estimator=mock_angles,
    )

    state = engine.update_radar_point_cloud([[0.1, 0.2, 0.3]], current_time=time.time())
    assert engine.last_posture == "FALLEN"
    assert engine.last_joint_angles["trunk_inclination_deg"] == 85.0
    assert engine.last_ml_prob >= 0.2
