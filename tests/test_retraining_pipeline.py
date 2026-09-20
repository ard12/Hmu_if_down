"""Tests for the automatic retraining pipeline, safety floors, and rollback."""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock

from hub.drift_detector import DriftDetector
from hub.model_registry import ModelRegistry
from hub.retraining_pipeline import RetrainingPipeline
from hub.training_buffer import TrainingBuffer


@pytest.fixture
def temp_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_registry.db")
        yield ModelRegistry(db_path=db_path)


@pytest.fixture
def populated_buffer():
    buf = TrainingBuffer(max_size=200)
    np.random.seed(42)
    # 50 falls (class 1) with large positive values
    for _ in range(50):
        buf.append(features=np.random.normal(loc=5.0, scale=0.5, size=4), label=1)
    # 50 non-falls (class 0) with negative values
    for _ in range(50):
        buf.append(features=np.random.normal(loc=-5.0, scale=0.5, size=4), label=0)
    return buf


def test_retraining_no_drift(temp_registry, populated_buffer):
    """run() returns 'NO_DRIFT' when PSI < threshold."""
    detector = DriftDetector()
    detector.fit_reference(np.array([0.2] * 50))
    # Record identical predictions so no drift occurs
    for _ in range(50):
        detector.record_prediction(0.2)

    pipeline = RetrainingPipeline(registry=temp_registry, buffer=populated_buffer, detector=detector)
    res = pipeline.run()
    assert res["status"] == "NO_DRIFT"
    assert res["new_version_id"] is None


def test_retraining_insufficient_data(temp_registry):
    """InsufficientDataError from buffer is caught and returns 'RETRAIN_FAILED'."""
    detector = DriftDetector()
    # Force drift status to RETRAIN_REQUIRED
    detector.drift_status = MagicMock(return_value={"status": "RETRAIN_REQUIRED", "psi": 0.5})

    empty_buf = TrainingBuffer(max_size=50)
    pipeline = RetrainingPipeline(registry=temp_registry, buffer=empty_buf, detector=detector)
    res = pipeline.run()

    assert res["status"] == "RETRAIN_FAILED"
    assert "Insufficient" in res["reason"] or "empty" in res["reason"]


def test_retraining_success(temp_registry, populated_buffer):
    """run() returns 'RETRAINED' and sets new active model when drift detected and quality good."""
    detector = DriftDetector()
    detector.drift_status = MagicMock(return_value={"status": "RETRAIN_REQUIRED", "psi": 0.35})
    audit_mock = MagicMock()

    pipeline = RetrainingPipeline(
        registry=temp_registry,
        buffer=populated_buffer,
        detector=detector,
        audit_log=audit_mock,
    )

    res = pipeline.run()

    assert res["status"] == "RETRAINED"
    assert res["new_version_id"] is not None
    assert res["sensitivity"] >= pipeline.SENSITIVITY_FLOOR
    assert res["specificity"] >= pipeline.SPECIFICITY_FLOOR

    # Verify model is active in registry
    active_clf = temp_registry.get_active()
    assert active_clf is not None
    assert temp_registry.get_active_version_id() == res["new_version_id"]

    # Verify audit log was notified
    audit_mock.append.assert_called_once()
    event_type, payload = audit_mock.append.call_args[0]
    assert event_type == "RETRAIN_SUCCESS"
    assert payload["version_id"] == res["new_version_id"]


def test_retraining_rollback_on_low_sensitivity(temp_registry, populated_buffer):
    """run() returns 'ROLLED_BACK' when retrained model sensitivity < 98.5%."""
    from sklearn.dummy import DummyClassifier

    # Seed an initial active baseline model
    dummy_baseline = DummyClassifier(strategy="constant", constant=1)
    dummy_baseline.fit([[0]], [1])
    v_orig = temp_registry.save_model(dummy_baseline, metadata={"active": True})
    assert temp_registry.get_active_version_id() == v_orig

    detector = DriftDetector()
    detector.drift_status = MagicMock(return_value={"status": "RETRAIN_REQUIRED", "psi": 0.4})
    audit_mock = MagicMock()

    pipeline = RetrainingPipeline(
        registry=temp_registry,
        buffer=populated_buffer,
        detector=detector,
        audit_log=audit_mock,
    )

    # Mock _do_retrain to return a bad model that always predicts class 0
    bad_model = MagicMock()
    bad_model.predict_proba.return_value = np.zeros((20, 2))  # 0 probability for fall
    bad_model.predict.return_value = np.zeros(20)
    pipeline._do_retrain = MagicMock(return_value=bad_model)

    res = pipeline.run()

    assert res["status"] == "ROLLED_BACK"
    assert res["new_version_id"] is None
    assert res["sensitivity"] < pipeline.SENSITIVITY_FLOOR

    # Original model must remain active
    assert temp_registry.get_active_version_id() == v_orig

    # Verify audit log recorded rollback
    audit_mock.append.assert_called_once()
    event_type, payload = audit_mock.append.call_args[0]
    assert event_type == "RETRAIN_ROLLED_BACK"
    assert "Sensitivity" in payload["reason"]


def test_registry_get_active_updates_after_retrain(temp_registry, populated_buffer):
    """Registry get_active() returns new model version after successful retrain."""
    detector = DriftDetector()
    detector.drift_status = MagicMock(return_value={"status": "RETRAIN_REQUIRED", "psi": 0.4})

    pipeline = RetrainingPipeline(registry=temp_registry, buffer=populated_buffer, detector=detector)
    res = pipeline.run()

    assert res["status"] == "RETRAINED"
    active_id = temp_registry.get_active_version_id()
    assert active_id == res["new_version_id"]


def test_registry_preserves_original_model_after_rollback(temp_registry, populated_buffer):
    """Registry get_active() returns original model after rollback."""
    from sklearn.dummy import DummyClassifier

    dummy_baseline = DummyClassifier(strategy="constant", constant=0)
    dummy_baseline.fit([[0]], [0])
    orig_id = temp_registry.save_model(dummy_baseline, metadata={"active": True})

    detector = DriftDetector()
    detector.drift_status = MagicMock(return_value={"status": "RETRAIN_REQUIRED", "psi": 0.4})

    pipeline = RetrainingPipeline(registry=temp_registry, buffer=populated_buffer, detector=detector)

    # Force poor sensitivity
    bad_model = MagicMock()
    bad_model.predict_proba.return_value = np.array([[0.9, 0.1]] * 20)
    pipeline._do_retrain = MagicMock(return_value=bad_model)

    res = pipeline.run()
    assert res["status"] == "ROLLED_BACK"
    assert temp_registry.get_active_version_id() == orig_id
    assert isinstance(temp_registry.get_active(), DummyClassifier)
