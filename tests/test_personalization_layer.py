"""Tests for Per-Site Model Personalization Head (hub/personalization_layer.py)."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from hub.dp_trainer import DPModel
from hub.personalization_layer import PersonalizationLayer


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)
    return X, y


def test_personalization_layer_init():
    base_model = DPModel(weights=np.zeros(4), bias=0.0)
    layer = PersonalizationLayer(global_model=base_model, hidden_dim=8, learning_rate=0.05)

    assert layer.hidden_dim == 8
    assert layer.learning_rate == 0.05
    assert layer.global_model is base_model
    assert not layer._is_fitted


def test_personalization_fit_freezes_base_model(sample_data):
    X, y = sample_data
    base_weights = np.array([0.2, -0.4, 0.6, -0.8])
    base_bias = 0.15
    base_model = DPModel(weights=base_weights.copy(), bias=base_bias)

    layer = PersonalizationLayer(global_model=base_model, hidden_dim=16, learning_rate=0.02)
    layer.fit(X, y, epochs=10, batch_size=16)

    # Verify base model was strictly frozen
    np.testing.assert_array_equal(base_model.weights, base_weights)
    assert base_model.bias == base_bias
    assert layer._is_fitted


def test_predict_proba_and_predict_shapes(sample_data):
    X, y = sample_data
    base_model = DPModel(weights=np.zeros(4), bias=0.0)
    layer = PersonalizationLayer(global_model=base_model, hidden_dim=12)
    layer.fit(X, y, epochs=5, batch_size=16)

    probs = layer.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0, atol=1e-6)

    preds = layer.predict(X, threshold=0.5)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_personalization_eval_metrics(sample_data):
    X, y = sample_data
    base_model = DPModel(weights=np.zeros(4), bias=0.0)
    layer = PersonalizationLayer(global_model=base_model, hidden_dim=16)
    layer.fit(X, y, epochs=15, batch_size=16)

    metrics = layer.evaluate(X, y)
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["accuracy"] > 0.6


def test_personalization_save_and_load(sample_data):
    X, y = sample_data
    base_model = DPModel(weights=np.zeros(4), bias=0.0)
    layer = PersonalizationLayer(global_model=base_model, hidden_dim=8)
    layer.fit(X, y, epochs=10, batch_size=16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "personalization_head.json"
        layer.save(save_path)
        assert save_path.exists()

        loaded_layer = PersonalizationLayer.from_file(save_path, global_model=base_model)
        assert loaded_layer._is_fitted
        assert loaded_layer.hidden_dim == layer.hidden_dim
        np.testing.assert_array_almost_equal(loaded_layer.w1, layer.w1)
        np.testing.assert_array_almost_equal(loaded_layer.b1, layer.b1)
        np.testing.assert_array_almost_equal(loaded_layer.w2, layer.w2)
        assert loaded_layer.b2 == pytest.approx(layer.b2)

        orig_probs = layer.predict_proba(X)
        loaded_probs = loaded_layer.predict_proba(X)
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-6)


def test_personalization_adapts_to_shifted_distribution():
    # Base model trained on distribution A (e.g., feature 0 dominant)
    rng = np.random.RandomState(123)
    base_model = DPModel(weights=np.array([2.0, 0.0, 0.0, 0.0]), bias=0.0)

    # Local facility distribution B: feature 1 and 2 indicate fall instead
    X_local = rng.normal(size=(80, 4))
    y_local = (X_local[:, 1] + X_local[:, 2] > 0.5).astype(int)

    layer = PersonalizationLayer(global_model=base_model, hidden_dim=16, learning_rate=0.03)
    layer.fit(X_local, y_local, epochs=25, batch_size=16)

    metrics = layer.evaluate(X_local, y_local)
    assert metrics["accuracy"] >= 0.70
