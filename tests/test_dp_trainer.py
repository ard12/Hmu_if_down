"""Tests for hub.dp_trainer (Milestone 16.1)."""

import numpy as np
import pytest
from hub.dp_trainer import DPSGDTrainer, DPModel


def test_clipped_gradients_have_norm_le_max_grad_norm():
    trainer = DPSGDTrainer(max_grad_norm=1.0)
    # Gradients with large norms (e.g. 5.0, 10.0)
    large_grads = [
        np.array([3.0, 4.0]),  # norm = 5.0
        np.array([6.0, 8.0]),  # norm = 10.0
        np.array([0.2, 0.3]),  # norm ~ 0.36 <= 1.0
    ]
    clipped = trainer.clip_gradients(large_grads)

    for g in clipped:
        norm = np.linalg.norm(g)
        assert norm <= 1.0 + 1e-6
    # Check that smaller gradient is unchanged
    assert np.allclose(clipped[2], large_grads[2])


def test_noised_gradients_have_different_values_from_unnoised():
    trainer = DPSGDTrainer(noise_multiplier=1.1, max_grad_norm=1.0)
    grads = [np.array([0.5, 0.5]), np.array([0.2, 0.2])]
    noised_avg = trainer.add_noise(grads, n_samples=2)
    unnoised_avg = np.mean(grads, axis=0)

    assert not np.allclose(noised_avg, unnoised_avg)


def test_noise_scale_proportional_to_max_grad_norm_and_multiplier():
    trainer_low = DPSGDTrainer(noise_multiplier=0.5, max_grad_norm=1.0, seed=1)
    trainer_high = DPSGDTrainer(noise_multiplier=2.5, max_grad_norm=1.0, seed=1)

    grads = [np.zeros(10) for _ in range(100)]
    samples_low = [trainer_low.add_noise(grads, n_samples=100) for _ in range(50)]
    samples_high = [trainer_high.add_noise(grads, n_samples=100) for _ in range(50)]

    std_low = np.std(samples_low)
    std_high = np.std(samples_high)

    assert std_high > std_low * 2.0


def test_compute_epsilon_increases_with_more_training_steps():
    trainer = DPSGDTrainer()
    eps_20 = trainer.compute_epsilon(steps=20, batch_size=16, n=500)
    eps_100 = trainer.compute_epsilon(steps=100, batch_size=16, n=500)

    assert eps_100 > eps_20


def test_compute_epsilon_decreases_with_larger_batch_size():
    trainer = DPSGDTrainer()
    eps_small_b = trainer.compute_epsilon(steps=50, batch_size=16, n=500)
    eps_large_b = trainer.compute_epsilon(steps=50, batch_size=64, n=500)

    assert eps_large_b < eps_small_b


def test_train_dp_returns_epsilon_in_reasonable_budget():
    trainer = DPSGDTrainer()
    rng = np.random.RandomState(42)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)

    model, eps, delta = trainer.train_dp(None, X, y, n_epochs=5, batch_size=16)

    assert eps > 0.0
    assert eps <= 10.0
    assert delta == 1e-5


def test_model_trained_with_dp_achieves_high_accuracy_on_separable_data():
    trainer = DPSGDTrainer(noise_multiplier=0.3, max_grad_norm=2.0, seed=42)
    rng = np.random.RandomState(42)

    # Clearly separable synthetic dataset
    X_pos = rng.normal(loc=2.5, scale=0.5, size=(100, 3))
    X_neg = rng.normal(loc=-2.5, scale=0.5, size=(100, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 100 + [0] * 100)

    model, eps, delta = trainer.train_dp(
        None, X, y, n_epochs=15, batch_size=20, learning_rate=0.2
    )

    accuracy = model.score(X, y)
    assert accuracy >= 0.85
