from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np


class InsufficientParticipantsError(Exception):
    """Raised when fewer than min_participants have submitted gradients."""
    pass


class FederatedAggregationServer:
    """
    Lightweight FedAvg server.
    Hubs POST gradient payloads; server averages and returns new global weights.
    """

    MIN_PARTICIPANTS: int = 3  # minimum hubs required before aggregation

    def __init__(self, global_model: Any, min_participants: int = MIN_PARTICIPANTS, learning_rate: float = 0.1):
        self.global_model = global_model
        self.min_participants = int(min_participants)
        self.learning_rate = float(learning_rate)
        self.submissions: Dict[str, Dict[str, Any]] = {}
        self.round_num: int = 1
        self.aggregated: bool = False
        self._last_aggregated_grads: Optional[List[np.ndarray]] = None
        self.round_history: List[Dict[str, Any]] = []
        self.total_epsilon: float = 0.0

    def receive_gradients(
        self,
        hub_id: str,
        gradients: List[Any],
        n_samples: int,
        epsilon: float,
    ):
        """Accept gradient payload from one hub. Weighted by n_samples."""
        self.submissions[hub_id] = {
            "gradients": [np.asarray(g, dtype=float) for g in gradients],
            "n_samples": int(n_samples),
            "epsilon": float(epsilon),
        }
        self.aggregated = False

    def aggregate(self) -> List[np.ndarray]:
        """
        FedAvg: weighted average of gradients.
        Raises InsufficientParticipantsError if < MIN_PARTICIPANTS have submitted.
        """
        if len(self.submissions) < self.min_participants:
            raise InsufficientParticipantsError(
                f"Requires at least {self.min_participants} participants, got {len(self.submissions)}"
            )

        total_samples = sum(sub["n_samples"] for sub in self.submissions.values())
        if total_samples <= 0:
            raise ValueError("Total samples across all submissions must be > 0")

        first_sub = next(iter(self.submissions.values()))
        n_grads = len(first_sub["gradients"])
        aggregated_grads: List[np.ndarray] = []

        for g_idx in range(n_grads):
            weighted_g = np.zeros_like(first_sub["gradients"][g_idx], dtype=float)
            for sub in self.submissions.values():
                weight = sub["n_samples"] / float(total_samples)
                weighted_g += weight * sub["gradients"][g_idx]
            aggregated_grads.append(weighted_g)

        self.aggregated = True
        self._last_aggregated_grads = aggregated_grads
        return aggregated_grads

    def broadcast_global_weights(self) -> Dict[str, Any]:
        """
        Apply aggregated gradients to global model.
        Returns serializable weight dict for hubs to download.
        """
        if not self.aggregated or self._last_aggregated_grads is None:
            aggregated_grads = self.aggregate()
        else:
            aggregated_grads = self._last_aggregated_grads

        # Apply gradients to global model
        if hasattr(self.global_model, "weights"):
            gw = aggregated_grads[0]
            self.global_model.weights -= self.learning_rate * gw
            if len(aggregated_grads) > 1 and hasattr(self.global_model, "bias"):
                gb = aggregated_grads[1]
                bias_scalar = float(gb) if np.ndim(gb) == 0 else float(gb[0])
                self.global_model.bias -= self.learning_rate * bias_scalar

        weights_list = self.global_model.weights.tolist() if hasattr(self.global_model, "weights") else []
        bias_val = float(self.global_model.bias) if hasattr(self.global_model, "bias") else 0.0

        current_round = self.round_num
        round_eps = max((sub["epsilon"] for sub in self.submissions.values()), default=0.0)
        self.total_epsilon += round_eps
        self.round_history.append({
            "round": current_round,
            "participants": len(self.submissions),
            "epsilon": round_eps,
            "total_epsilon": self.total_epsilon,
            "timestamp": datetime.now().isoformat(),
        })

        self.round_num += 1
        self.submissions.clear()
        self.aggregated = False
        self._last_aggregated_grads = None

        return {
            "round": current_round,
            "weights": weights_list,
            "bias": bias_val,
            "timestamp": datetime.now().isoformat(),
        }

    def get_round_status(self) -> Dict[str, Any]:
        """
        Returns:
          {
            "round": int,
            "participants": int,
            "min_required": int,
            "aggregated": bool,
            "total_epsilon": float
          }
        """
        return {
            "round": self.round_num,
            "participants": len(self.submissions),
            "min_required": self.min_participants,
            "aggregated": self.aggregated,
            "total_epsilon": self.total_epsilon,
        }
