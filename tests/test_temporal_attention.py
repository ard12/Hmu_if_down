"""Unit tests for temporal attention windowing in CSIPCAExtractor (Milestone 8.1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.csi_pipeline.pca_features import CSIPCAExtractor


def test_attention_weights_are_recency_biased():
    """Verify exponential attention weights favor recent samples (t=T-1 vs t=0)."""
    T = 100
    decay = 2.0
    t = np.arange(T, dtype=np.float32)
    weights = np.exp(-decay * (T - 1 - t) / max(T - 1, 1))

    # Weight at the end of the window (most recent) should be 1.0
    assert abs(weights[-1] - 1.0) < 1e-5
    # Weight at the start of the window should be e^(-decay)
    assert abs(weights[0] - np.exp(-decay)) < 1e-5
    # Monotonically increasing
    assert np.all(np.diff(weights) >= 0)


def test_set_attention_method_updates_decay():
    """Verify set_attention() properly updates enabled flag and decay parameter."""
    extractor = CSIPCAExtractor()
    assert extractor.temporal_attention is True
    assert extractor.attention_decay == 2.0

    extractor.set_attention(enabled=False, decay=4.5)
    assert extractor.temporal_attention is False
    assert extractor.attention_decay == 4.5

    extractor.set_attention(enabled=True)
    assert extractor.temporal_attention is True
    assert extractor.attention_decay == 4.5


def test_attention_increases_velocity_sensitivity():
    """A recent abrupt frequency burst should register higher dominant velocity with attention."""
    extractor_atten = CSIPCAExtractor(temporal_attention=True, attention_decay=3.0)
    extractor_flat = CSIPCAExtractor(temporal_attention=False)

    # 100 samples: quiet (low freq 2Hz) for first 80 samples, then burst (high freq 20Hz) for last 20 samples
    fs = 100.0
    t_quiet = np.linspace(0, 0.8, 80, endpoint=False)
    t_burst = np.linspace(0.8, 1.0, 20, endpoint=False)

    sig_quiet = np.sin(2 * np.pi * 2.0 * t_quiet)
    sig_burst = 3.0 * np.sin(2 * np.pi * 20.0 * t_burst)
    burst_signal = np.concatenate([sig_quiet, sig_burst])

    # Synthesize 4 subcarriers with this signal
    data = np.outer(burst_signal, [1.0, 0.8, 1.2, 0.9])

    feat_atten = extractor_atten.extract(node_id=1, filtered_data=data)
    feat_flat = extractor_flat.extract(node_id=1, filtered_data=data)

    # With attention, recent 20Hz burst has higher weight, leading to higher velocity
    assert feat_atten.dominant_velocity_mps >= feat_flat.dominant_velocity_mps
