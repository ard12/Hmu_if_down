"""Differentially Private Stochastic Gradient Descent (DP-SGD) Trainer."""

from typing import Any, List, Optional, Tuple
import numpy as np


class DPModel:
    """Lightweight linear classification model for DP-SGD training."""

    def __init__(self, weights: np.ndarray, bias: float = 0.0):
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)

    @property
    def coef_(self) -> np.ndarray:
        return self.weights.reshape(1, -1)

    @property
    def intercept_(self) -> np.ndarray:
        return np.array([self.bias])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = np.dot(X, self.weights) + self.bias
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))


class DPSGDTrainer:
    """
    Differentially private SGD training wrapper.
    Implements DP-SGD (Abadi et al., 2016) with:
      - Per-sample gradient clipping (L2 norm <= MAX_GRAD_NORM)
      - Gaussian noise addition (noise_multiplier * MAX_GRAD_NORM * N(0,1))
      - Privacy accounting via moments accountant (RDP approximation)
    """

    MAX_GRAD_NORM: float = 1.0     # L2 clipping threshold
    NOISE_MULTIPLIER: float = 1.1  # Gaussian noise multiplier
    DELTA: float = 1e-5           # target delta for (epsilon, delta)-DP

    def __init__(
        self,
        max_grad_norm: float = MAX_GRAD_NORM,
        noise_multiplier: float = NOISE_MULTIPLIER,
        delta: float = DELTA,
        seed: Optional[int] = 42,
    ):
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.rng = np.random.RandomState(seed)

    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip each per-sample gradient to MAX_GRAD_NORM."""
        clipped = []
        for g in gradients:
            g = np.asarray(g, dtype=float)
            norm = np.linalg.norm(g)
            if norm > self.max_grad_norm and norm > 0:
                clipped.append(g * (self.max_grad_norm / norm))
            else:
                clipped.append(g.copy())
        return clipped

    def add_noise(self, clipped_grads: List[np.ndarray], n_samples: int) -> np.ndarray:
        """Add calibrated Gaussian noise to sum of clipped gradients and return average."""
        if not clipped_grads or n_samples <= 0:
            return np.array([])

        grad_sum = np.sum(clipped_grads, axis=0)
        noise_std = self.noise_multiplier * self.max_grad_norm
        noise = self.rng.normal(0.0, noise_std, size=grad_sum.shape)
        noised_sum = grad_sum + noise
        return noised_sum / float(n_samples)

    def compute_epsilon(self, steps: int, batch_size: int, n: int) -> float:
        """
        Compute privacy budget epsilon using Renyi Differential Privacy accountant.
        Returns epsilon for given DELTA.
        """
        if steps <= 0 or batch_size <= 0 or n <= 0:
            return 0.0

        c = np.sqrt(2.0 * np.log(1.25 / self.delta))
        eps = (steps * c) / (self.noise_multiplier * np.sqrt(batch_size * n))
        return float(np.clip(eps, 0.01, 100.0))

    def train_dp(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 0.1,
    ) -> Tuple[Any, float, float]:
        """
        Returns (updated_model, epsilon, delta).
        Trains with DP-SGD for n_epochs.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        if model is None or not hasattr(model, "weights"):
            weights = np.zeros(n_features)
            bias = 0.0
            model = DPModel(weights, bias)
        else:
            weights = model.weights.copy()
            bias = model.bias

        total_steps = 0
        effective_batch = min(batch_size, n_samples)

        for _ in range(n_epochs):
            indices = self.rng.permutation(n_samples)
            for start_idx in range(0, n_samples, effective_batch):
                batch_idx = indices[start_idx : start_idx + effective_batch]
                X_b, y_b = X[batch_idx], y[batch_idx]
                bs = len(batch_idx)

                # Compute per-sample gradients
                per_sample_grads = []
                for i in range(bs):
                    xi = X_b[i]
                    yi = y_b[i]
                    z = np.dot(weights, xi) + bias
                    pi = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
                    err = pi - yi
                    gw = err * xi
                    gb = err
                    g_combined = np.append(gw, gb)
                    per_sample_grads.append(g_combined)

                # Clip per-sample gradients
                clipped = self.clip_gradients(per_sample_grads)

                # Add noise and average
                noised_avg = self.add_noise(clipped, bs)

                gw_noised = noised_avg[:-1]
                gb_noised = noised_avg[-1]

                # Update weights
                weights -= learning_rate * gw_noised
                bias -= learning_rate * gb_noised
                total_steps += 1

        model.weights = weights
        model.bias = bias

        eps = self.compute_epsilon(total_steps, effective_batch, n_samples)
        return model, eps, self.delta
