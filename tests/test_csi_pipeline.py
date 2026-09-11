"""Unit tests for Wi-Fi CSI signal processing and multi-link fusion."""

import struct
import numpy as np
import pytest

from hub.csi_pipeline.preprocessor import CSIPreprocessor, CSIPacket
from hub.csi_pipeline.pca_features import CSIPCAExtractor, CSIDynamicFeatures
from hub.csi_pipeline.multi_link_fusion import MultiLinkFusionEngine, CSIFallState


def test_csi_packet_parsing():
    preprocessor = CSIPreprocessor()

    # Construct synthetic UDP packet:
    # Header: "CSIF", node_id=2, rssi=-45, subcarrier_count=4, timestamp=123456, seq=101
    subcarrier_count = 4
    header = struct.pack("<4sBbHII", b"CSIF", 2, -45, subcarrier_count, 123456, 101)

    # 4 subcarriers = 8 bytes (I, Q pairs)
    # [ (10, 20), (30, 40), (50, 60), (70, 80) ]
    iq_bytes = bytes([10, 20, 30, 40, 50, 60, 70, 80])
    raw_packet = header + iq_bytes

    parsed = preprocessor.parse_packet(raw_packet)
    assert parsed is not None
    assert parsed.node_id == 2
    assert parsed.rssi == -45
    assert parsed.subcarrier_count == 4
    assert parsed.timestamp_ms == 123456
    assert parsed.seq_num == 101
    assert len(parsed.amplitudes) == 4
    assert pytest.approx(parsed.amplitudes[0], abs=0.1) == np.hypot(10, 20)


def test_corrupt_csi_packet_rejected():
    preprocessor = CSIPreprocessor()
    # Bad magic
    header = struct.pack("<4sBbHII", b"BADM", 1, -50, 4, 100, 1)
    assert preprocessor.parse_packet(header + bytes(8)) is None


def test_pca_and_velocity_estimation():
    extractor = CSIPCAExtractor(sampling_rate_hz=100.0, velocity_threshold_mps=1.5)

    # Simulate 50 time samples across 32 subcarriers with 25 Hz Doppler sine wave
    t = np.linspace(0, 0.5, 50)
    sine = np.sin(2 * np.pi * 25.0 * t)[:, None]
    matrix = np.ones((50, 32)) * 10.0 + sine * 20.0

    features = extractor.extract(node_id=1, filtered_data=matrix)
    assert features.node_id == 1
    # At 25 Hz Doppler and 2.437 GHz (lambda ~ 0.123m), v = 0.123 * 25 / 2 ~ 1.53 m/s
    assert features.dominant_velocity_mps > 1.2
    assert features.is_velocity_burst is True


def test_multi_link_coincidence_and_quiescence():
    engine = MultiLinkFusionEngine(
        coincidence_window_sec=0.4,
        min_coincident_links=2,
        post_fall_quiescence_sec=2.0,
        motionless_variance_threshold=0.1,
    )

    t0 = 100.0

    # 1. Single link burst: should NOT trigger suspected fall
    f_burst_n1 = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.2,
        energy_surge_ratio=4.5,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    s1 = engine.register_feature(f_burst_n1, current_time=t0)
    assert s1 == CSIFallState.NORMAL

    # 2. Second link burst within coincidence window (0.1s later): SHOULD trigger suspected fall!
    f_burst_n2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.0,
        energy_surge_ratio=4.0,
        moving_variance=1.2,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    s2 = engine.register_feature(f_burst_n2, current_time=t0 + 0.1)
    assert s2 == CSIFallState.SUSPECTED_FALL

    # 3. Post-impact: Stillness for > 2.0 seconds -> CONFIRMED FALL
    f_still = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.05,
        energy_surge_ratio=1.0,
        moving_variance=0.03,  # Below threshold
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )
    s3 = engine.register_feature(f_still, current_time=t0 + 2.2)
    assert s3 == CSIFallState.CONFIRMED_FALL
