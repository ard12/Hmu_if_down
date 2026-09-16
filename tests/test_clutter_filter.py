"""Unit tests for pet and ground clutter disambiguation in CSI and mmWave radar."""

import numpy as np
import pytest

from hub.csi_pipeline.pca_features import CSIDynamicFeatures
from hub.csi_pipeline.multi_link_fusion import MultiLinkFusionEngine, CSIFallState
from hub.mmwave_pipeline.radar_receiver import RadarReceiver, RadarFallState, RadarPosture, RadarTelemetry
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState


def test_csi_elevation_perturbation_ratio_calculation():
    engine = MultiLinkFusionEngine(epr_threshold=0.15)

    # Low elevation pet: Link 3 high variance, Links 1 & 2 low
    pet_vars = {1: 0.02, 2: 0.03, 3: 0.80}
    epr_pet = engine.compute_elevation_perturbation_ratio(pet_vars)
    # (0.02 + 0.03) / (2 * 0.80) = 0.05 / 1.60 = 0.03125
    assert pytest.approx(epr_pet, abs=0.001) == 0.03125
    assert engine.is_ground_clutter(pet_vars) is True

    # Human fall: all links exhibit strong perturbations
    fall_vars = {1: 1.20, 2: 0.95, 3: 1.40}
    epr_fall = engine.compute_elevation_perturbation_ratio(fall_vars)
    # (1.20 + 0.95) / (2 * 1.40) = 2.15 / 2.80 ~ 0.7678
    assert pytest.approx(epr_fall, abs=0.01) == 0.7678
    assert engine.is_ground_clutter(fall_vars) is False


def test_csi_pet_ground_clutter_suppresses_fall_alert():
    engine = MultiLinkFusionEngine(
        coincidence_window_sec=0.4,
        min_coincident_links=2,
        enable_pet_filter=True,
        epr_threshold=0.15,
        motionless_variance_threshold=0.08,
    )

    t0 = 100.0

    # Baseline quiet state on Links 1 and 2
    f_quiet1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.05,
        energy_surge_ratio=1.0,
        moving_variance=0.02,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )
    f_quiet2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=0.05,
        energy_surge_ratio=1.0,
        moving_variance=0.02,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )
    engine.register_feature(f_quiet1, current_time=t0)
    engine.register_feature(f_quiet2, current_time=t0)

    # Pet (cat or robot vacuum) runs across floor, perturbing Link 3
    f_pet_burst = CSIDynamicFeatures(
        node_id=3,
        dominant_velocity_mps=2.4,
        energy_surge_ratio=5.2,
        moving_variance=0.90,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    state = engine.register_feature(f_pet_burst, current_time=t0 + 0.05)

    # Must remain NORMAL and record suppression
    assert state == CSIFallState.NORMAL
    assert engine.clutter_suppressed_count >= 1
    assert len(engine.recent_bursts[3]) == 0  # Burst was filtered out


def test_csi_human_fall_transitions_normally():
    engine = MultiLinkFusionEngine(
        coincidence_window_sec=0.4,
        min_coincident_links=2,
        post_fall_quiescence_sec=1.5,
        enable_pet_filter=True,
        epr_threshold=0.15,
        motionless_variance_threshold=0.08,
    )

    t0 = 200.0

    # Human fall perturbs Links 1 and 2
    f_fall_n1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.1,
        energy_surge_ratio=4.0,
        moving_variance=1.1,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    f_fall_n2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=1.9,
        energy_surge_ratio=3.8,
        moving_variance=1.0,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )

    s1 = engine.register_feature(f_fall_n1, current_time=t0)
    assert s1 == CSIFallState.NORMAL

    s2 = engine.register_feature(f_fall_n2, current_time=t0 + 0.1)
    # Both elevated links triggered -> not ground clutter!
    assert s2 == CSIFallState.SUSPECTED_FALL

    # Post-impact stillness
    f_still = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.02,
        energy_surge_ratio=1.0,
        moving_variance=0.03,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )
    s3 = engine.register_feature(f_still, current_time=t0 + 1.8)
    assert s3 == CSIFallState.CONFIRMED_FALL


def test_radar_cluster_area_rejects_pet():
    receiver = RadarReceiver(min_cluster_area_m2=0.15, enable_cluster_filter=True)
    engine = DualFusionEngine(mode=OperatingMode.FUSION)

    # Synthetic pet datagram: fall reported by radar, but cluster area = 0.06 m^2 (< 0.15 m^2)
    pet_json = '{"fall": 2, "posture": 3, "height": 0.20, "dwell": 5, "cluster_area": 0.06}'
    telemetry = receiver.parse_json_datagram(pet_json)

    assert telemetry is not None
    assert telemetry.is_clutter is True
    assert telemetry.fall_state == RadarFallState.NONE
    assert telemetry.raw_valid is False

    # Feeding clutter telemetry to fusion engine must NOT confirm fall
    unified = engine.update_radar(telemetry)
    assert unified == UnifiedFallState.NORMAL


def test_radar_cluster_area_accepts_human_fall():
    receiver = RadarReceiver(min_cluster_area_m2=0.15, enable_cluster_filter=True)
    engine = DualFusionEngine(mode=OperatingMode.FUSION)

    # Human fall datagram: cluster area = 0.42 m^2 (>= 0.15 m^2)
    human_json = '{"fall": 2, "posture": 3, "height": 0.22, "dwell": 5, "cluster_area": 0.42}'
    telemetry = receiver.parse_json_datagram(human_json)

    assert telemetry is not None
    assert telemetry.is_clutter is False
    assert telemetry.fall_state == RadarFallState.CONFIRMED
    assert pytest.approx(telemetry.cluster_area_m2, abs=0.01) == 0.42

    unified = engine.update_radar(telemetry)
    assert unified == UnifiedFallState.CONFIRMED


def test_radar_binary_frame_cluster_area_parsing():
    receiver = RadarReceiver(min_cluster_area_m2=0.15, enable_cluster_filter=True)

    # Binary frame for cluster area: Ctrl 0x03, Cmd 0x03, Len 0x0002
    # Area = 600 cm^2 = 0.06 m^2 (< 0.15 m^2 -> clutter)
    # Area bytes: 600 = 0x0258
    frame_body = bytearray([0x53, 0x59, 0x03, 0x03, 0x00, 0x02, 0x02, 0x58])
    checksum = sum(frame_body) & 0xFF
    full_frame = bytes(frame_body + bytearray([checksum]))

    telemetry = receiver.parse_binary_frame(full_frame)
    assert telemetry is not None
    assert pytest.approx(telemetry.cluster_area_m2, abs=0.001) == 0.06
    assert telemetry.is_clutter is True
