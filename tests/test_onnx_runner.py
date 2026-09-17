"""Tests for ONNXFallClassifier (Milestone 6.2).

These tests verify the ONNX runner in its pickle-fallback mode (since onnxruntime
is an optional dependency). When skl2onnx and onnxruntime are installed, the same
tests exercise the full ONNX inference path.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.onnx_runner import ONNXFallClassifier, FEATURE_DIM

PICKLE_PATH = PROJECT_ROOT / "models" / "fall_classifier.pkl"
_PICKLE_AVAILABLE = PICKLE_PATH.exists()


@pytest.fixture
def clf():
    """Classifier backed by the existing pickle model."""
    if not _PICKLE_AVAILABLE:
        pytest.skip("fall_classifier.pkl not found — run hub/train.py first")
    return ONNXFallClassifier(fallback_pickle=PICKLE_PATH)


def test_classifier_is_ready(clf):
    """Classifier must report is_ready=True when at least one backend is loaded."""
    assert clf.is_ready


def test_predict_returns_binary_label_and_confidence(clf):
    """predict() must return (int, float) where label ∈ {0,1} and confidence ∈ [0,1]."""
    features = [0.1] * FEATURE_DIM
    label, confidence = clf.predict(features)
    assert label in (0, 1)
    assert 0.0 <= confidence <= 1.0


def test_predict_proba_sums_to_one(clf):
    """predict_proba() must return a 2-element list summing to ≈1.0."""
    features = [0.5] * FEATURE_DIM
    proba = clf.predict_proba(features)
    assert len(proba) == 2
    assert abs(sum(proba) - 1.0) < 1e-5


def test_predict_fall_features_returns_fall_label(clf):
    """High-velocity, high-surge features should be classified as fall (label=1)."""
    # Physically extreme fall features: max energy in all sub-bands, very high velocity
    fall_features = [
        5.0,   # sub_band_0_energy (high)
        5.0,   # sub_band_1_energy (high)
        5.0,   # sub_band_2_energy (high)
        5.0,   # sub_band_3_energy (high)
        3.5,   # high_low_ratio (far above ADL baseline)
        3.5,   # dominant_velocity_mps (above fall threshold)
        5.0,   # energy_surge_ratio
        0.15,  # moving_variance (high)
        1.5,   # spectral_entropy (high disorder)
    ]
    label, confidence = clf.predict(fall_features)
    assert label == 1, f"Expected fall label=1 but got {label} (conf={confidence:.3f})"


def test_predict_adl_features_returns_adl_label(clf):
    """Near-zero feature vectors (person sitting still) should be classified as ADL."""
    adl_features = [0.002] * FEATURE_DIM
    label, confidence = clf.predict(adl_features)
    assert label == 0, f"Expected ADL label=0 but got {label} (conf={confidence:.3f})"


def test_benchmark_inference_returns_latency_dict(clf):
    """benchmark_inference() must return a dict with expected latency keys."""
    report = clf.benchmark_inference(n=50)
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "n_calls", "backend"):
        assert key in report, f"Missing key: {key}"
    assert report["n_calls"] == 50
    assert report["mean_ms"] > 0


def test_missing_model_raises_runtime_error():
    """ONNXFallClassifier with no model files must raise RuntimeError on predict()."""
    clf_empty = ONNXFallClassifier(
        model_path=Path("/nonexistent/model.onnx"),
        fallback_pickle=Path("/nonexistent/model.pkl"),
    )
    assert not clf_empty.is_ready
    with pytest.raises(RuntimeError):
        clf_empty.predict([0.0] * FEATURE_DIM)
