"""Tests for Population Stability Index (PSI) and KL divergence drift detection."""

import numpy as np
import pytest

from hub.drift_detector import DriftDetector


def test_psi_zero_for_identical_distributions():
    """PSI is 0 (or near-zero) for identical distributions."""
    detector = DriftDetector()
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    detector.fit_reference(probs)

    for p in probs:
        detector.record_prediction(p)

    psi = detector.compute_psi()
    assert psi < 1e-4


def test_psi_exceeds_retrain_threshold_on_shifted_mean():
    """PSI > RETRAIN_THRESHOLD when live distribution shifted by 30% mean."""
    detector = DriftDetector()
    np.random.seed(42)

    # Reference distribution centered at 0.25 (e.g. low fall risk baseline)
    ref_probs = np.clip(np.random.normal(loc=0.25, scale=0.08, size=500), 0.0, 1.0)
    detector.fit_reference(ref_probs)

    # Live distribution shifted by 30% (centered at 0.55)
    live_probs = np.clip(np.random.normal(loc=0.55, scale=0.08, size=500), 0.0, 1.0)
    for p in live_probs:
        detector.record_prediction(p)

    psi = detector.compute_psi()
    assert psi > detector.PSI_RETRAIN_THRESHOLD


def test_kl_divergence_for_gaussian_shift():
    """KL divergence is strictly positive and proportional for shifted distributions."""
    detector = DriftDetector()
    np.random.seed(42)

    ref = np.clip(np.random.normal(loc=0.3, scale=0.1, size=500), 0.0, 1.0)
    detector.fit_reference(ref)

    # Identical live: KL should be 0
    for p in ref:
        detector.record_prediction(p)
    assert detector.compute_kl() < 1e-4

    # Shifted live
    detector.fit_reference(ref)
    shifted = np.clip(np.random.normal(loc=0.6, scale=0.1, size=500), 0.0, 1.0)
    for p in shifted:
        detector.record_prediction(p)

    kl = detector.compute_kl()
    assert kl > detector.KL_WARN_THRESHOLD


def test_drift_status_states():
    """drift_status() returns 'OK' / 'WARNING' / 'RETRAIN_REQUIRED' appropriately."""
    detector = DriftDetector()
    np.random.seed(42)

    # 1. Baseline OK
    baseline = np.random.uniform(0.1, 0.4, size=300)
    detector.fit_reference(baseline)
    for p in np.random.uniform(0.1, 0.4, size=300):
        detector.record_prediction(p)

    status_ok = detector.drift_status()
    assert status_ok["status"] == "OK"

    # 2. Large shift -> RETRAIN_REQUIRED
    detector.fit_reference(baseline)
    shifted_large = np.random.uniform(0.7, 0.95, size=300)
    for p in shifted_large:
        detector.record_prediction(p)

    status_retrain = detector.drift_status()
    assert status_retrain["status"] == "RETRAIN_REQUIRED"
    assert status_retrain["psi"] > detector.PSI_RETRAIN_THRESHOLD


def test_reset_reference():
    """reset_reference() resets live buffer and resets PSI to 0."""
    detector = DriftDetector()
    np.random.seed(42)

    ref = np.random.uniform(0.1, 0.3, size=200)
    detector.fit_reference(ref)

    # Record shifted predictions
    for p in np.random.uniform(0.7, 0.9, size=200):
        detector.record_prediction(p)

    assert detector.compute_psi() > detector.PSI_RETRAIN_THRESHOLD

    # Reset reference to new distribution
    detector.reset_reference()
    assert detector.compute_psi() == 0.0

    status = detector.drift_status()
    assert status["live_n"] == 0
    assert status["status"] == "OK"


def test_record_prediction_handles_empty_reference():
    """record_prediction handles empty reference gracefully without raising exceptions."""
    detector = DriftDetector()
    # No reference fitted yet
    for p in [0.1, 0.2, 0.3]:
        detector.record_prediction(p)

    assert detector.compute_psi() == 0.0
    assert detector.compute_kl() == 0.0
    status = detector.drift_status()
    assert status["status"] == "OK"
    assert status["live_n"] == 3


def test_degenerate_distribution():
    """Edge case: all predictions identical (degenerate distribution)."""
    detector = DriftDetector()
    ref = np.array([0.5] * 100)
    detector.fit_reference(ref)

    for _ in range(100):
        detector.record_prediction(0.5)

    psi = detector.compute_psi()
    kl = detector.compute_kl()
    assert psi == 0.0
    assert kl == 0.0
    assert not np.isnan(psi)
    assert not np.isnan(kl)


def test_record_prediction_clips_bounds():
    """record_prediction clips probabilities to [0.0, 1.0]."""
    detector = DriftDetector()
    detector.record_prediction(-0.5)
    detector.record_prediction(1.5)

    assert detector._live_probs[0] == 0.0
    assert detector._live_probs[1] == 1.0

