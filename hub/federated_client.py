"""Federated Learning Hub Client for local DP-SGD training and weight synchronization."""

from typing import Any, Dict, Optional
import numpy as np
from .dp_trainer import DPModel, DPSGDTrainer


class FederatedHubClient:
    """Client side: trains locally with DP-SGD, posts gradients, downloads global weights."""

    def __init__(
        self,
        server_url: str,
        hub_id: str,
        registry: Optional[Any] = None,
        dp_trainer: Optional[DPSGDTrainer] = None,
        server: Optional[Any] = None,
    ):
        self.server_url = server_url
        self.hub_id = hub_id
        self.registry = registry
        self.dp_trainer = dp_trainer or DPSGDTrainer()
        self.server = server
        self.local_model = DPModel(weights=np.zeros(4), bias=0.0)

    async def participate_in_round(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        1. Train local model with DP-SGD.
        2. Compute gradient diff vs. last global weights.
        3. POST /federated/gradients (or direct server.receive_gradients).
        4. If aggregation ready: GET /federated/weights, update local registry.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        if len(self.local_model.weights) != n_features:
            self.local_model = DPModel(weights=np.zeros(n_features), bias=0.0)

        initial_weights = self.local_model.weights.copy()
        initial_bias = self.local_model.bias

        # 1. Train locally with DP-SGD
        trained_model, eps, delta = self.dp_trainer.train_dp(
            self.local_model, X, y, n_epochs=5, batch_size=16
        )

        # 2. Gradient representation as update step: (initial - updated)
        grad_w = initial_weights - trained_model.weights
        grad_b = np.array([initial_bias - trained_model.bias])
        gradients = [grad_w, grad_b]

        # 3. Submit gradients
        if self.server is not None:
            self.server.receive_gradients(
                hub_id=self.hub_id,
                gradients=gradients,
                n_samples=n_samples,
                epsilon=eps,
            )

            # Check if server has enough participants to aggregate
            status = self.server.get_round_status()
            if status["participants"] >= status["min_required"]:
                broadcast_data = self.server.broadcast_global_weights()
                new_w = np.array(broadcast_data["weights"], dtype=float)
                new_b = float(broadcast_data["bias"])
                self.local_model.weights = new_w
                self.local_model.bias = new_b

                if self.registry is not None and hasattr(self.registry, "register_model"):
                    try:
                        self.registry.register_model(
                            model=self.local_model,
                            metrics={"accuracy": float(self.local_model.score(X, y)), "epsilon": eps},
                            hyperparameters={"round": broadcast_data["round"]},
                        )
                    except Exception:
                        pass

                return {
                    "status": "AGGREGATED",
                    "round": broadcast_data["round"],
                    "epsilon": eps,
                    "weights": broadcast_data["weights"],
                    "bias": broadcast_data["bias"],
                }

        return {
            "status": "SUBMITTED",
            "hub_id": self.hub_id,
            "epsilon": eps,
            "n_samples": n_samples,
        }
