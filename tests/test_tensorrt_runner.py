"""Unit tests for TensorRTRunner (Milestone 19.1)."""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.tensorrt_runner import (
    InferenceLatencyStats,
    InferenceProvider,
    InferenceResult,
    TensorRTRunner,
    FEATURE_DIM,
)


class MockSklearnModel:
    def predict(self, X):
        return np.ones(len(X), dtype=np.int64)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def test_tensorrt_runner_falls_back_to_sklearn_when_no_onnx():
    runner = TensorRTRunner(onnx_model_path=None, precision="fp16")
    assert runner.provider == InferenceProvider.SKLEARN_CPU
    assert not runner.is_tensorrt


def test_precision_validation():
    with pytest.raises(ValueError, match="Unsupported precision"):
        TensorRTRunner(precision="fp64")


def test_predict_returns_inference_result_dataclass():
    mock_model = MockSklearnModel()
    runner = TensorRTRunner(fallback_model=mock_model)
    res = runner.predict([0.5] * FEATURE_DIM)
    assert isinstance(res, InferenceResult)
    assert res.provider == InferenceProvider.SKLEARN_CPU
    assert res.latency_ms >= 0.0
    assert len(res.predictions) == 1
    assert res.predictions[0] == 1


def test_latency_stats_initially_empty():
    runner = TensorRTRunner()
    stats = runner.get_latency_stats()
    assert isinstance(stats, InferenceLatencyStats)
    assert stats.total_inferences == 0
    assert stats.mean_ms == 0.0


def test_latency_stats_after_predictions():
    mock_model = MockSklearnModel()
    runner = TensorRTRunner(fallback_model=mock_model)
    for _ in range(10):
        runner.predict([0.1] * FEATURE_DIM)
    stats = runner.get_latency_stats()
    assert stats.total_inferences == 10
    assert stats.mean_ms >= 0.0
    assert stats.p50_ms >= 0.0
    assert stats.p95_ms >= stats.p50_ms
    assert stats.p99_ms >= stats.p95_ms


def test_calibration_dataset_shape():
    runner = TensorRTRunner()
    ds = runner.generate_calibration_dataset(n_samples=50, n_features=9)
    assert ds.shape == (50, 9)
    assert ds.dtype == np.float32


def test_provider_enum_values():
    providers = list(InferenceProvider)
    assert len(providers) == 4
    str_values = [p.value for p in providers]
    assert len(set(str_values)) == 4
    assert "tensorrt_fp16" in str_values
    assert "tensorrt_int8" in str_values
    assert "onnx_cpu" in str_values
    assert "sklearn_cpu" in str_values


def test_export_tensorrt_engine_graceful():
    runner = TensorRTRunner()
    result = runner.export_tensorrt_engine(Path("dummy_engine.plan"))
    assert isinstance(result, bool)
