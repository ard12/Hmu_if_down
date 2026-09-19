"""Tests for Post-Fall Vital Signs & Respiration Estimator (Milestone 10.1)."""

import numpy as np
import pytest

from hub.vital_signs import VitalSignsEstimator, VitalSignsResult


def test_vital_signs_detects_normal_breathing():
    estimator = VitalSignsEstimator()
    # 16 bpm = 0.267 Hz
    phase = estimator.generate_synthetic_vital_signal(
        duration_sec=15.0,
        sample_rate_hz=20.0,
        breathing_rate_bpm=16.0,
        displacement_mm=1.5,
        noise_std=0.02,
    )
    result = estimator.process_radar_quiescence(phase, sample_rate_hz=20.0)

    assert result.vital_detected is True
    assert 14.0 <= result.breathing_rate_bpm <= 18.0
    assert result.status == "NORMAL_BREATHING"
    assert result.confidence >= 0.5
    assert result.chest_displacement_mm >= 0.5


def test_vital_signs_detects_tachypnea():
    estimator = VitalSignsEstimator()
    # 28 bpm = 0.467 Hz (tachypnea: > 24 bpm)
    phase = estimator.generate_synthetic_vital_signal(
        duration_sec=15.0,
        sample_rate_hz=20.0,
        breathing_rate_bpm=28.0,
        displacement_mm=1.0,
        noise_std=0.02,
    )
    result = estimator.process_radar_quiescence(phase, sample_rate_hz=20.0)

    assert result.vital_detected is True
    assert 26.0 <= result.breathing_rate_bpm <= 30.0
    assert result.status == "TACHYPNEA"


def test_vital_signs_inanimate_signal_rejected():
    estimator = VitalSignsEstimator()
    # Pure flat noise without periodic respiration (e.g. dropped book or pillow)
    rng = np.random.default_rng(42)
    inanimate_phase = rng.normal(0, 0.01, size=300)
    result = estimator.process_radar_quiescence(inanimate_phase, sample_rate_hz=20.0)

    assert result.vital_detected is False
    assert result.breathing_rate_bpm == 0.0
    assert result.status == "APNEA_OR_INANIMATE"


def test_vital_signs_csi_quiescence():
    estimator = VitalSignsEstimator()
    # Synthesize CSI amplitude oscillation
    t = np.linspace(0, 10.0, 1000, endpoint=False)
    # 15 bpm = 0.25 Hz
    csi_amp = 50.0 + 3.0 * np.sin(2.0 * np.pi * 0.25 * t) + np.random.normal(0, 0.2, 1000)
    result = estimator.process_csi_quiescence(csi_amp, sample_rate_hz=100.0)

    assert result.vital_detected is True
    assert 13.0 <= result.breathing_rate_bpm <= 17.0
    assert result.status == "NORMAL_BREATHING"


def test_vital_signs_insufficient_data():
    estimator = VitalSignsEstimator()
    short_signal = np.array([1.0, 2.0, 3.0])
    result = estimator.process_radar_quiescence(short_signal, sample_rate_hz=20.0)
    assert result.vital_detected is False
    assert result.status == "INSUFFICIENT_DATA"


def test_vital_signs_to_dict():
    res = VitalSignsResult(
        breathing_rate_bpm=16.234,
        chest_displacement_mm=1.456,
        vital_detected=True,
        status="NORMAL_BREATHING",
        confidence=0.8876,
    )
    d = res.to_dict()
    assert d["breathing_rate_bpm"] == 16.2
    assert d["chest_displacement_mm"] == 1.46
    assert d["vital_detected"] is True
    assert d["status"] == "NORMAL_BREATHING"
    assert d["confidence"] == 0.888
