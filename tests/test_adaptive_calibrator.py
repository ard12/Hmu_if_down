"""Unit tests for carrier frequency scaling and background adaptive noise calibrator."""

import pytest

from hub.adaptive_calibrator import AdaptiveCalibrator
from hub.csi_pipeline.multi_link_fusion import MultiLinkFusionEngine
from hub.csi_pipeline.pca_features import CSIPCAExtractor


def test_carrier_frequency_scaling():
    """Verify carrier frequency configuration dynamically adjusts Doppler wavelength."""
    extractor = CSIPCAExtractor(carrier_freq_hz=2.437e9)
    assert pytest.approx(extractor.wavelength, abs=0.005) == 0.123

    # Switch to 5 GHz Wi-Fi Channel 36
    extractor.set_carrier_frequency(36)
    assert pytest.approx(extractor.wavelength, abs=0.005) == 0.0545

    # Switch back to 2.4 GHz Channel 1
    extractor.set_carrier_frequency(1)
    assert pytest.approx(extractor.wavelength, abs=0.005) == 0.123

    # Set explicit frequency in Hz (e.g. 5.8 GHz)
    extractor.set_carrier_frequency(5.8e9)
    assert pytest.approx(extractor.wavelength, abs=0.002) == 0.0517


def test_adaptive_calibrator_baseline_drift():
    """Verify background calibrator smoothly adapts threshold under steady environmental drift."""
    engine = MultiLinkFusionEngine(motionless_variance_threshold=0.08)
    calibrator = AdaptiveCalibrator(
        csi_engine=engine,
        alpha=0.1,  # Fast adaptation for unit test
        initial_baseline=0.02,
        threshold_multiplier=2.5,
        min_threshold=0.04,
    )

    # Initially: baseline = 0.02, threshold = max(0.04, 0.02 * 2.5) = 0.05
    assert calibrator.current_threshold == 0.05
    assert engine.motionless_var_thresh == 0.05

    # Feed 20 quiescent samples with elevated noise floor (0.04)
    for i in range(20):
        t = calibrator.update(variance=0.04, is_quiescent=True, current_time=100.0 + i)

    # Baseline should have adapted upwards towards ~0.04
    status = calibrator.get_status()
    assert status["baseline_variance"] > 0.035
    assert status["current_threshold"] >= 0.09
    assert engine.motionless_var_thresh == status["current_threshold"]
    assert status["samples_collected"] == 20


def test_adaptive_calibrator_rejects_motion_outliers():
    """Verify large kinetic motion bursts are rejected from quiescent baseline adaptation."""
    calibrator = AdaptiveCalibrator(
        initial_baseline=0.02,
        quiescent_max_variance=0.10,
    )
    init_baseline = calibrator.baseline_variance

    # 1. Non-quiescent sample should be rejected
    calibrator.update(variance=0.03, is_quiescent=False)
    assert calibrator.baseline_variance == init_baseline
    assert calibrator.samples_collected == 0

    # 2. Large kinetic burst (> 0.10) should be rejected even if is_quiescent flag is True
    calibrator.update(variance=1.85, is_quiescent=True)
    assert calibrator.baseline_variance == init_baseline
    assert calibrator.samples_collected == 0


def test_adaptive_calibrator_bounds_clamping():
    """Verify adapted thresholds clamp within safe min/max limits."""
    calibrator = AdaptiveCalibrator(
        alpha=0.5,
        initial_baseline=0.01,
        threshold_multiplier=2.5,
        min_threshold=0.05,
        max_threshold=0.15,
    )

    # Very small variance (0.001 * 2.5 = 0.0025) -> should clamp to min_threshold 0.05
    calibrator.update(variance=0.001, is_quiescent=True)
    assert calibrator.current_threshold == 0.05

    # Large allowed variance (0.09 * 2.5 = 0.225) -> should clamp to max_threshold 0.15
    for _ in range(10):
        calibrator.update(variance=0.09, is_quiescent=True)
    assert calibrator.current_threshold == 0.15
