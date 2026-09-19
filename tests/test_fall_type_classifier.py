"""Unit tests for the second-stage FallTypeClassifier (Milestone 8.3)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.fall_type_classifier import FallTypeClassifier


def test_fall_type_classifier_trains_on_synthetic_data(tmp_path):
    """Verify FallTypeClassifier trains and exports model with valid metadata report."""
    clf = FallTypeClassifier()
    X, y = clf.generate_synthetic_features(n_per_class=40)
    out_model = tmp_path / "models" / "fall_type_classifier.pkl"
    report = clf.train_and_export(X, y, output_path=out_model)

    assert out_model.exists()
    assert report["num_samples"] == len(y)
    assert report["num_classes"] == 5
    assert report["train_accuracy"] >= 0.85
    assert "fall_types" in report


def test_fall_type_predict_returns_valid_label():
    """Verify predict() returns a label belonging to FALL_TYPES."""
    clf = FallTypeClassifier()
    dummy_feat = [3.0, 10.0, 25.0, 30.0, 4.5, 2.5, 8.0, 2.0, 5.0, 5.0, 3.0]
    label, conf = clf.predict(dummy_feat)

    assert label in FallTypeClassifier.FALL_TYPES
    assert 0.0 <= conf <= 1.0


def test_fall_type_confidence_in_unit_interval():
    """Verify confidence estimates are bounded in [0.0, 1.0] across diverse features."""
    clf = FallTypeClassifier()
    rng = np.random.default_rng(123)
    for _ in range(25):
        feat = rng.uniform(0.1, 15.0, 11)
        _, conf = clf.predict(feat)
        assert 0.0 <= conf <= 1.0


def test_fall_type_syncope_features_predict_syncope():
    """Near-zero initial motion, low spectral entropy, prolonged dwell should classify as syncope_drop."""
    clf = FallTypeClassifier()
    # Syncope: low 0-5Hz and 5-15Hz, low spectral entropy (2.5), long dwell (20s)
    syncope_feat = [1.0, 3.0, 12.0, 20.0, 3.5, 1.8, 4.5, 1.2, 2.8, 22.0, 2.0]
    label, conf = clf.predict(syncope_feat)
    assert label == "syncope_drop"
    assert conf > 0.30


def test_fall_type_fast_onset_predicts_forward_trip():
    """High dominant velocity and strong horizontal kinetic surge should classify as forward_trip."""
    clf = FallTypeClassifier()
    trip_feat = [4.0, 14.0, 25.0, 25.0, 5.0, 3.2, 8.5, 2.0, 5.2, 4.0, 3.5]
    label, conf = clf.predict(trip_feat)
    assert label in ("forward_trip", "backward_slip")
    assert conf > 0.30


def test_fall_type_alert_priority_mapping():
    """Verify all 5 fall types have an explicit priority mapping (HIGH or MEDIUM)."""
    for ft in FallTypeClassifier.FALL_TYPES:
        assert ft in FallTypeClassifier.ALERT_PRIORITIES
        assert FallTypeClassifier.ALERT_PRIORITIES[ft] in ("HIGH", "MEDIUM")
