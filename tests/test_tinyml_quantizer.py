"""Unit tests for TinyMLQuantizer (Milestone 19.2)."""

import sys
from pathlib import Path
import pytest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.tinyml_quantizer import TinyMLQuantizer, QuantisationReport


def test_quantise_dequantise_round_trip():
    q = TinyMLQuantizer(bit_depth=8, feature_range=(-10.0, 10.0))
    test_val = 3.25
    quant_val = q.quantise_value(test_val)
    dequant_val = q.dequantise_value(quant_val)
    tolerance = 20.0 / 255.0  # 1 LSB
    assert abs(test_val - dequant_val) <= tolerance + 1e-4


def test_invalid_bit_depth_raises():
    with pytest.raises(ValueError, match="Unsupported bit depth"):
        TinyMLQuantizer(bit_depth=7)


def test_invalid_feature_range_raises():
    with pytest.raises(ValueError, match="Invalid feature range"):
        TinyMLQuantizer(feature_range=(10.0, -10.0))


def test_quantise_clamps_out_of_range():
    q = TinyMLQuantizer(bit_depth=8, feature_range=(-5.0, 5.0))
    # Below min
    q_low = q.quantise_value(-100.0)
    assert q_low == 0
    # Above max
    q_high = q.quantise_value(100.0)
    assert q_high == 255


def test_scale_and_zero_point_properties():
    q = TinyMLQuantizer(bit_depth=8, feature_range=(0.0, 255.0))
    assert abs(q.scale - 1.0) < 1e-3
    assert q.zero_point == 0


def test_extract_thresholds_with_hgbc():
    X = np.random.uniform(-5.0, 5.0, size=(100, 9))
    y = (X[:, 0] + X[:, 5] > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=5, random_state=42)
    clf.fit(X, y)

    quantizer = TinyMLQuantizer(bit_depth=8, feature_range=(-6.0, 6.0))
    thresholds = quantizer.extract_thresholds(clf)
    assert len(thresholds) > 0
    first = thresholds[0]
    assert "tree_idx" in first
    assert "feature_idx" in first
    assert "threshold" in first
    assert "threshold_q" in first
    assert 0 <= first["threshold_q"] <= 255


def test_evaluate_quantisation_report():
    X = np.random.uniform(-5.0, 5.0, size=(120, 9))
    y = (X[:, 0] > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=5, random_state=42)
    clf.fit(X, y)

    quantizer = TinyMLQuantizer(bit_depth=8, feature_range=(-6.0, 6.0))
    quantizer.extract_thresholds(clf)
    report = quantizer.evaluate_quantisation(clf, X, y)

    assert isinstance(report, QuantisationReport)
    assert report.bit_depth == 8
    assert report.n_test_samples == 120
    assert 0.0 <= report.original_accuracy <= 1.0
    assert 0.0 <= report.quantised_accuracy <= 1.0
    assert report.max_abs_error >= 0.0
    assert report.model_size_bytes > 0


def test_export_c_header_generates_file(tmp_path):
    quantizer = TinyMLQuantizer(bit_depth=8, feature_range=(-5.0, 5.0))
    quantizer._thresholds = [
        {"tree_idx": 0, "feature_idx": 1, "threshold": 2.5, "threshold_q": 191},
        {"tree_idx": 0, "feature_idx": 3, "threshold": -1.0, "threshold_q": 102},
    ]

    hdr_file = tmp_path / "edge_infer.h"
    content = quantizer.export_c_header(hdr_file, model_name="fall_edge")

    assert hdr_file.exists()
    assert "#ifndef FALL_EDGE_TINYML_H" in content
    assert "TINYML_SCALE" in content
    assert "tinyml_quantise_feature" in content
    assert "fall_edge_thresholds[TINYML_N_THRESHOLDS]" in content
