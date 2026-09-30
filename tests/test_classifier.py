"""Unit tests for the Probabilistic Machine Learning Classifier."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from hub.csi_pipeline.classifier import CSIFeatureExtractor, FallClassifier, MLFeatures


def test_feature_extractor_from_window():
    """Verify feature extractor produces complete 9-dimensional kinematic vector."""
    extractor = CSIFeatureExtractor(sampling_rate_hz=100.0)

    # 100 samples (1 second) of synthetic 30 Hz kinetic oscillation
    t = np.linspace(0, 1.0, 100)
    burst = np.sin(2 * np.pi * 30.0 * t)[:, None]
    window = np.ones((100, 64)) * 25.0 + burst * 20.0

    features = extractor.extract_from_window(window, baseline_energy=10.0)

    assert isinstance(features, MLFeatures)
    assert features.dominant_velocity > 1.5  # 30 Hz at 0.123m ~ 1.84 m/s
    assert features.subband_energy_25_40hz > 0.0
    assert features.high_low_ratio > 0.0

    arr = features.to_array()
    assert len(arr) == 9
    assert np.all(np.isfinite(arr))


def test_classifier_predict_proba_discrimination():
    """Verify classifier discriminates between normal walking and true fall bursts."""
    clf = FallClassifier()

    # Synthetic normal walking feature vector
    normal_feat = MLFeatures(
        subband_energy_0_5hz=15.0,
        subband_energy_5_15hz=8.0,
        subband_energy_15_25hz=0.5,
        subband_energy_25_40hz=0.1,
        high_low_ratio=0.03,
        dominant_velocity=0.6,
        energy_surge=1.5,
        temporal_variance=0.15,
        spectral_entropy=3.2,
    )
    p_normal = clf.predict_proba(normal_feat)
    assert p_normal < 0.4, f"Normal activity probability too high: {p_normal}"

    # Synthetic fall burst feature vector
    fall_feat = MLFeatures(
        subband_energy_0_5hz=3.0,
        subband_energy_5_15hz=6.0,
        subband_energy_15_25hz=18.0,
        subband_energy_25_40hz=35.0,
        high_low_ratio=5.8,
        dominant_velocity=2.4,
        energy_surge=6.5,
        temporal_variance=2.1,
        spectral_entropy=4.9,
    )
    p_fall = clf.predict_proba(fall_feat)
    assert p_fall > 0.7, f"Fall probability too low: {p_fall}"


def test_classifier_save_and_load():
    """Verify model persistence and re-loading parity."""
    clf = FallClassifier()
    test_feat = np.array([5.0, 5.0, 20.0, 30.0, 5.0, 2.2, 5.0, 1.8, 4.5], dtype=np.float32)
    p_orig = clf.predict_proba(test_feat)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "test_model.pkl"
        clf.save(model_file)
        assert model_file.exists()
        assert model_file.with_suffix(".pkl.sha256").exists()

        loaded_clf = FallClassifier(model_path=model_file)
        p_loaded = loaded_clf.predict_proba(test_feat)

        np.testing.assert_almost_equal(p_orig, p_loaded, decimal=4)

        # Tampering with model file must cause load() to fail cryptographic integrity
        model_file.write_bytes(model_file.read_bytes() + b"\x00corrupt")
        with pytest.raises(ValueError, match="Cryptographic integrity check failed"):
            loaded_clf.load(model_file)


def test_hybrid_fusion_with_classifier():
    """Verify DualFusionEngine escalates to confirmed fall when ML classifier score is high."""
    from hub.alert_dispatcher import AlertDispatcher
    from hub.csi_pipeline.pca_features import CSIDynamicFeatures
    from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState

    alert = AlertDispatcher(enable_sound=False)
    engine = DualFusionEngine(mode=OperatingMode.CSI_ONLY, alert_dispatcher=alert)

    # Suspected fall burst
    burst = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.4,
        energy_surge_ratio=4.5,
        moving_variance=1.8,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    # First link burst -> NORMAL
    engine.update_csi(burst, current_time=100.0, ml_prob=0.2)
    # Second link burst -> SUSPECTED
    burst2 = CSIDynamicFeatures(
        node_id=2,
        dominant_velocity_mps=2.1,
        energy_surge_ratio=4.0,
        moving_variance=1.5,
        is_velocity_burst=True,
        pc1_signal=np.array([]),
    )
    # When ml_prob is 0.92, it immediately confirms fall
    state = engine.update_csi(burst2, current_time=100.1, ml_prob=0.92)
    assert state == UnifiedFallState.CONFIRMED
    assert engine.unified_state == UnifiedFallState.CONFIRMED
