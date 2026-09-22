"""Per-Site Model Personalization Head for local facility adaptation.

Trains a 2-layer MLP adaptation head on top of frozen global model representations
to customize fall detection to specific facility/patient distributions without
modifying the federated base weights.
"""

from typing import Any, Dict, Optional, Tuple, Union
import json
from pathlib import Path
import numpy as np


class PersonalizationLayer:
    """Per-site personalization head that learns local adaptations on top of a frozen global model."""

    def __init__(
        self,
        global_model: Optional[Any] = None,
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        random_state: Optional[int] = 42,
    ):
        self.global_model = global_model
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(random_state)

        self.w1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.w2: Optional[np.ndarray] = None
        self.b2: Optional[float] = None
        self._is_fitted = False

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features by concatenating raw X with frozen global model predictions."""
        X = np.asarray(X, dtype=float)
        if self.global_model is not None and hasattr(self.global_model, "predict_proba"):
            global_probs = self.global_model.predict_proba(X)
            if global_probs.ndim == 2:
                global_score = global_probs[:, 1:2]
            else:
                global_score = global_probs.reshape(-1, 1)
            return np.hstack([X, global_score])
        return X

    def _init_weights(self, in_dim: int) -> None:
        """Initialize weights using He/Kaiming normal initialization."""
        self.w1 = self.rng.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.w2 = self.rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        batch_size: int = 16,
    ) -> "PersonalizationLayer":
        """Fit personalization head on local site data using Adam optimizer, keeping base model frozen."""
        H0 = self._extract_features(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n_samples, in_dim = H0.shape

        if self.w1 is None or self.w1.shape[0] != in_dim:
            self._init_weights(in_dim)

        # Adam optimizer state
        mw1, vw1 = np.zeros_like(self.w1), np.zeros_like(self.w1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mw2, vw2 = np.zeros_like(self.w2), np.zeros_like(self.w2)
        mb2, vb2 = 0.0, 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                t += 1
                batch_idx = indices[start_idx : start_idx + batch_size]
                h0_b = H0[batch_idx]
                y_b = y[batch_idx]
                b_size = len(batch_idx)

                # Forward pass
                # Layer 1: Linear + ReLU
                a1 = np.dot(h0_b, self.w1) + self.b1
                h1 = np.maximum(0, a1)

                # Layer 2: Linear + Sigmoid
                a2 = np.dot(h1, self.w2) + self.b2
                p = 1.0 / (1.0 + np.exp(-np.clip(a2, -20.0, 20.0)))

                # Backward pass
                da2 = (p - y_b) / b_size
                dw2 = np.dot(h1.T, da2)
                db2 = float(np.sum(da2))

                dh1 = np.dot(da2, self.w2.T)
                da1 = dh1 * (a1 > 0)
                dw1 = np.dot(h0_b.T, da1)
                db1 = np.sum(da1, axis=0)

                # Adam updates
                mw1 = beta1 * mw1 + (1 - beta1) * dw1
                vw1 = beta2 * vw1 + (1 - beta2) * (dw1 ** 2)
                m_hat_w1 = mw1 / (1 - beta1 ** t)
                v_hat_w1 = vw1 / (1 - beta2 ** t)
                self.w1 -= self.learning_rate * m_hat_w1 / (np.sqrt(v_hat_w1) + eps)

                mb1 = beta1 * mb1 + (1 - beta1) * db1
                vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
                m_hat_b1 = mb1 / (1 - beta1 ** t)
                v_hat_b1 = vb1 / (1 - beta2 ** t)
                self.b1 -= self.learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + eps)

                mw2 = beta1 * mw2 + (1 - beta1) * dw2
                vw2 = beta2 * vw2 + (1 - beta2) * (dw2 ** 2)
                m_hat_w2 = mw2 / (1 - beta1 ** t)
                v_hat_w2 = vw2 / (1 - beta2 ** t)
                self.w2 -= self.learning_rate * m_hat_w2 / (np.sqrt(v_hat_w2) + eps)

                mb2 = beta1 * mb2 + (1 - beta1) * db2
                vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
                m_hat_b2 = mb2 / (1 - beta1 ** t)
                v_hat_b2 = vb2 / (1 - beta2 ** t)
                self.b2 -= self.learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + eps)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities [P(normal), P(fall)]."""
        H0 = self._extract_features(X)
        if not self._is_fitted or self.w1 is None:
            # Fallback to global model if available
            if self.global_model is not None and hasattr(self.global_model, "predict_proba"):
                return self.global_model.predict_proba(X)
            # Default prior
            p = np.full((len(H0), 1), 0.5)
            return np.column_stack([1.0 - p, p])

        a1 = np.dot(H0, self.w1) + self.b1
        h1 = np.maximum(0, a1)
        a2 = np.dot(h1, self.w2) + self.b2
        p = 1.0 / (1.0 + np.exp(-np.clip(a2, -20.0, 20.0)))
        p = p.flatten()
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels (0: normal, 1: fall)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Compute evaluation metrics on validation/test set."""
        y = np.asarray(y, dtype=int).flatten()
        y_pred = self.predict(X, threshold=threshold)

        tp = int(np.sum((y == 1) & (y_pred == 1)))
        fp = int(np.sum((y == 0) & (y_pred == 1)))
        fn = int(np.sum((y == 1) & (y_pred == 0)))
        tn = int(np.sum((y == 0) & (y_pred == 0)))

        acc = (tp + tn) / max(len(y), 1)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save personalization head parameters to a JSON file."""
        data = {
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "is_fitted": self._is_fitted,
            "w1": self.w1.tolist() if self.w1 is not None else None,
            "b1": self.b1.tolist() if self.b1 is not None else None,
            "w2": self.w2.tolist() if self.w2 is not None else None,
            "b2": float(self.b2) if self.b2 is not None else None,
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Union[str, Path]) -> "PersonalizationLayer":
        """Load personalization head parameters from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.hidden_dim = data["hidden_dim"]
        self.learning_rate = data.get("learning_rate", 0.01)
        self._is_fitted = data["is_fitted"]
        self.w1 = np.array(data["w1"], dtype=float) if data["w1"] is not None else None
        self.b1 = np.array(data["b1"], dtype=float) if data["b1"] is not None else None
        self.w2 = np.array(data["w2"], dtype=float) if data["w2"] is not None else None
        self.b2 = float(data["b2"]) if data["b2"] is not None else None
        return self

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        global_model: Optional[Any] = None,
    ) -> "PersonalizationLayer":
        """Instantiate and load a PersonalizationLayer from file."""
        layer = cls(global_model=global_model)
        layer.load(filepath)
        return layer
