"""Unit tests for Bayesian fall probability calibration and graduated severity (Milestone 8.2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.csi_pipeline.classifier import FallClassifier, MLFeatures
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.csi_pipeline.multi_link_fusion import CSIFallState
from hub.csi_pipeline.pca_features import CSIDynamicFeatures
from hub.train import generate_synthetic_features, train_and_export


def test_calibrated_classifier_probability_in_range():
    """Verify calibrated classifier outputs probabilities strictly within [0.0, 1.0]."""
    clf = FallClassifier()
    assert clf.is_calibrated is True

    # Test on random feature arrays
    rng = np.random.default_rng(42)
    for _ in range(20):
        feat = rng.uniform(0.0, 10.0, 9)
        prob = clf.predict_proba(feat)
        assert 0.0 <= prob <= 1.0


def test_calibrated_fall_probability_high_for_fall_features():
    """Verify high-energy descent features yield calibrated P(fall) >= 0.80."""
    clf = FallClassifier()
    fall_feat = MLFeatures(
        subband_energy_0_5hz=3.0,
        subband_energy_5_15hz=10.0,
        subband_energy_15_25hz=25.0,
        subband_energy_25_40hz=35.0,
        high_low_ratio=4.6,
        dominant_velocity=2.6,
        energy_surge=8.5,
        temporal_variance=2.2,
        spectral_entropy=5.0,
    )
    prob = clf.predict_proba(fall_feat)
    assert prob >= 0.80


def test_calibrated_adl_probability_low_for_quiet_features():
    """Verify quiescent ADL features yield calibrated P(fall) <= 0.20."""
    clf = FallClassifier()
    adl_feat = MLFeatures(
        subband_energy_0_5hz=18.0,
        subband_energy_5_15hz=5.0,
        subband_energy_15_25hz=0.3,
        subband_energy_25_40hz=0.05,
        high_low_ratio=0.015,
        dominant_velocity=0.35,
        energy_surge=1.05,
        temporal_variance=0.08,
        spectral_entropy=2.8,
    )
    prob = clf.predict_proba(adl_feat)
    assert prob <= 0.20


def test_graduated_severity_suspected_threshold():
    """Verify probability between p_suspected (0.55) and p_confirmed (0.80) maps to SUSPECTED."""
    engine = DualFusionEngine(mode=OperatingMode.CSI_ONLY)
    dummy_feat = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=0.5,
        energy_surge_ratio=1.5,
        moving_variance=0.05,
        is_velocity_burst=False,
        pc1_signal=np.array([]),
    )

    state = engine.update_csi(dummy_feat, ml_prob=0.60)
    assert state == UnifiedFallState.SUSPECTED
    assert engine.high_confidence is False


def test_graduated_severity_confirmed_threshold():
    """Verify probability >= p_confirmed (0.80) in combination with suspected state maps to CONFIRMED."""
    engine = DualFusionEngine(mode=OperatingMode.CSI_ONLY)
    engine.last_csi_state = CSIFallState.SUSPECTED_FALL
    engine.last_ml_prob = 0.85

    state = engine._evaluate_consensus()
    assert state == UnifiedFallState.CONFIRMED

    # When ml_prob >= p_high_confidence (0.95), high_confidence is flagged True
    engine.last_ml_prob = 0.98
    state_high = engine._evaluate_consensus()
    assert state_high == UnifiedFallState.CONFIRMED
    assert engine.high_confidence is True


def test_brier_score_in_evaluation_report(tmp_path):
    """Verify train_and_export() includes brier_score and calibrated=True in report."""
    X, y = generate_synthetic_features(n_samples=150, random_state=42)
    model_file = tmp_path / "models" / "fall_classifier.pkl"
    report_file = tmp_path / "models" / "evaluation_report.json"

    report = train_and_export(
        X,
        y,
        output_model_path=model_file,
        report_file_path=report_file,
        n_splits=3,
    )

    assert "brier_score" in report
    assert 0.0 <= report["brier_score"] <= 0.25  # Lower is better (0 is perfect)
    assert report.get("calibrated") is True
    assert (tmp_path / "models" / "fall_classifier_calibrated.pkl").exists()
