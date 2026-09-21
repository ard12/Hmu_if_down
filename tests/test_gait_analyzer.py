"""Tests for Gait Cadence Extractor from CSI micro-Doppler (Milestone 14.1)."""

import numpy as np
import pytest
from hub.gait_analyzer import GaitCadenceAnalyzer


def test_synthetic_normal_walk_cadence():
    """Synthetic 1.8 Hz sinusoidal Doppler classifies as NORMAL gait."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    # Generate 4 seconds of 1.8 Hz sinusoidal signal
    t = np.linspace(0, 4.0, 400, endpoint=False)
    signal = 2.0 * np.sin(2.0 * np.pi * 1.8 * t)

    analyzer.update(signal, timestamp=1004.0)
    res = analyzer.analyze(current_time=1004.0)

    assert res["gait_class"] == "NORMAL"
    assert abs(res["cadence_hz"] - 1.8) < 0.25
    assert res["confidence"] == 1.0


def test_synthetic_shuffle_walk_cadence():
    """Synthetic 0.7 Hz sinusoidal Doppler classifies as SHUFFLE gait."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    t = np.linspace(0, 4.0, 400, endpoint=False)
    signal = 1.5 * np.sin(2.0 * np.pi * 0.7 * t)

    analyzer.update(signal, timestamp=2004.0)
    res = analyzer.analyze(current_time=2004.0)

    assert res["gait_class"] == "SHUFFLE"
    assert abs(res["cadence_hz"] - 0.7) < 0.2
    assert res["confidence"] == 1.0


def test_dc_signal_classifies_as_stationary():
    """DC signal (constant amplitude) classifies as STATIONARY."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    signal = np.ones(350) * 5.0

    analyzer.update(signal, timestamp=3004.0)
    res = analyzer.analyze(current_time=3004.0)

    assert res["gait_class"] == "STATIONARY"
    assert res["cadence_hz"] == 0.0


def test_stride_regularity_near_one_for_periodic_signal():
    """Stride regularity is high (>0.85) for perfectly periodic signal."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    t = np.linspace(0, 4.0, 400, endpoint=False)
    signal = 2.0 * np.sin(2.0 * np.pi * 2.0 * t)

    analyzer.update(signal, timestamp=4004.0)
    res = analyzer.analyze(current_time=4004.0)

    assert res["stride_regularity"] > 0.85


def test_stride_regularity_low_for_white_noise():
    """Stride regularity is low (< 0.5) for random noise."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    np.random.seed(42)
    noise = np.random.normal(0, 1, 400)

    analyzer.update(noise, timestamp=5004.0)
    res = analyzer.analyze(current_time=5004.0)

    assert res["stride_regularity"] < 0.5


def test_confidence_degrades_when_window_sparse():
    """Confidence degrades (< 0.8) when analysis window contains < 80% valid samples."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    # Window is 3.0s (300 samples). Feed only 100 samples (33%).
    t = np.linspace(0, 1.0, 100, endpoint=False)
    signal = np.sin(2.0 * np.pi * 1.8 * t)

    analyzer.update(signal, timestamp=6001.0)
    res = analyzer.analyze(current_time=6003.0)

    assert res["confidence"] < 0.8


def test_stft_hop_time_shifted_not_duplicated():
    """STFT analysis shifted by hop produces time-consistent cadence."""
    analyzer = GaitCadenceAnalyzer(fs=100.0)
    t = np.linspace(0, 5.0, 500, endpoint=False)
    signal = np.sin(2.0 * np.pi * 1.5 * t)

    analyzer.update(signal, timestamp=7005.0)

    res1 = analyzer.analyze(current_time=7003.0)
    res2 = analyzer.analyze(current_time=7003.5)

    assert res1["gait_class"] == "NORMAL"
    assert res2["gait_class"] == "NORMAL"
    assert abs(res1["cadence_hz"] - res2["cadence_hz"]) < 0.2
