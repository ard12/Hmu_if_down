"""Tests for Pre-Fall Behavioural Anomaly Detector (Milestone 14.2)."""

import pytest
from hub.prefail_detector import PreFallAnomalyDetector


def test_score_zero_for_normal_gait_no_context():
    """Score = 0 and risk_level = NORMAL for normal walking with no context."""
    detector = PreFallAnomalyDetector()
    gait = {
        "gait_class": "NORMAL",
        "velocity_envelope_peak": 0.8,
        "cadence_hz": 1.8,
    }

    res = detector.update(gait, timestamp=1000.0)
    assert res["risk_score"] == 0
    assert res["risk_level"] == "NORMAL"
    assert len(res["contributing_factors"]) == 0


def test_score_gte_40_for_sustained_shuffle():
    """Score >= 40 for sustained SHUFFLE gait for >= 10s."""
    detector = PreFallAnomalyDetector()
    base_t = 2000.0
    gait = {"gait_class": "SHUFFLE", "velocity_envelope_peak": 0.3}

    # Feed 11 seconds of shuffle gait
    res = None
    for i in range(12):
        res = detector.update(gait, timestamp=base_t + i * 1.0)

    assert res["risk_score"] >= 40
    assert "Sustained shuffle gait (>=10s)" in res["contributing_factors"]


def test_score_gte_70_triggers_watch_level():
    """Score >= 70 triggers WATCH risk level."""
    detector = PreFallAnomalyDetector()
    base_t = 3000.0
    # Sustained shuffle (+40), velocity drop (+30) -> 70
    gait = {
        "gait_class": "SHUFFLE",
        "velocity_envelope_peak": 0.1,  # < 50% baseline
    }

    res = None
    for i in range(16):
        res = detector.update(gait, timestamp=base_t + i * 1.0)

    assert res["risk_score"] >= 70
    assert res["risk_level"] in ("WATCH", "IMMEDIATE_INTERVENTION")


def test_score_gte_90_triggers_immediate_intervention():
    """Score >= 90 triggers IMMEDIATE_INTERVENTION."""
    detector = PreFallAnomalyDetector()
    base_t = 4000.0
    # Sustained shuffle (+40), velocity drop (+30), reversals (+20), Morse >= 45 (+10) -> 100
    gait = {
        "gait_class": "SHUFFLE",
        "velocity_envelope_peak": 0.1,
        "reversals": 3,
    }
    pt_ctx = {"morse_fall_scale": 55}

    res = None
    for i in range(16):
        res = detector.update(gait, patient_context=pt_ctx, timestamp=base_t + i * 1.0)

    assert res["risk_score"] >= 90
    assert res["risk_level"] == "IMMEDIATE_INTERVENTION"


def test_morse_fall_scale_adds_ten_points():
    """Morse Fall Scale >= 45 adds 10 points to risk score."""
    detector = PreFallAnomalyDetector()
    gait = {"gait_class": "NORMAL", "velocity_envelope_peak": 0.8}
    pt_ctx = {"morse_fall_scale": 50}

    res = detector.update(gait, patient_context=pt_ctx, timestamp=5000.0)
    assert res["risk_score"] == 10
    assert "High Morse Fall Scale (>=45)" in res["contributing_factors"]


def test_contributing_factors_non_empty_when_score_positive():
    """contributing_factors list is non-empty when risk_score > 0."""
    detector = PreFallAnomalyDetector()
    detector.record_reversal(timestamp=6000.0)
    detector.record_reversal(timestamp=6005.0)
    detector.record_reversal(timestamp=6010.0)

    gait = {"gait_class": "NORMAL", "velocity_envelope_peak": 0.8}
    res = detector.update(gait, timestamp=6015.0)

    assert res["risk_score"] == 20
    assert len(res["contributing_factors"]) >= 1


def test_get_history_returns_capped_snapshots():
    """get_history returns at most 60 snapshots."""
    detector = PreFallAnomalyDetector()
    gait = {"gait_class": "NORMAL", "velocity_envelope_peak": 0.8}

    # Feed 150 snapshots spaced by 2.1s
    for i in range(150):
        detector.update(gait, timestamp=7000.0 + i * 2.1)

    history = detector.get_history()
    assert len(history) <= 60
    assert len(history) > 10


def test_score_resets_toward_zero_when_gait_recovers():
    """Score resets toward 0 when gait returns to NORMAL and velocity recovers."""
    detector = PreFallAnomalyDetector()
    base_t = 8000.0

    # Build up shuffle score
    for i in range(12):
        detector.update({"gait_class": "SHUFFLE", "velocity_envelope_peak": 0.3}, timestamp=base_t + i * 1.0)

    res_shuffle = detector.update({"gait_class": "SHUFFLE", "velocity_envelope_peak": 0.3}, timestamp=base_t + 12.0)
    assert res_shuffle["risk_score"] >= 40

    # Return to normal walking
    res_normal = detector.update({"gait_class": "NORMAL", "velocity_envelope_peak": 0.9}, timestamp=base_t + 13.0)
    assert res_normal["risk_score"] == 0
    assert res_normal["risk_level"] == "NORMAL"
