"""Monitors classifier output distribution for concept drift using PSI and KL divergence."""

import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class DriftDetector:
    """Monitors classifier output distribution for concept drift.

    Uses Population Stability Index (PSI) and Kullback-Leibler (KL) divergence
    over calibrated fall probability distributions to trigger automatic retraining.
    """

    PSI_WARN_THRESHOLD = 0.10
    PSI_RETRAIN_THRESHOLD = 0.20
    KL_WARN_THRESHOLD = 0.05
    REFERENCE_WINDOW = 500

    def __init__(self, reference_window: int = REFERENCE_WINDOW):
        self.reference_window = reference_window
        self._reference_probs: Optional[np.ndarray] = None
        self._live_probs: deque = deque(maxlen=reference_window)
        self._lock = threading.Lock()

    def record_prediction(self, prob: float):
        """Ingest live classifier output probability (0.0 - 1.0)."""
        val = max(0.0, min(1.0, float(prob)))
        with self._lock:
            self._live_probs.append(val)
            if self._reference_probs is None and len(self._live_probs) >= self.reference_window:
                self._reference_probs = np.array(self._live_probs, dtype=float)

    def fit_reference(self, probs: np.ndarray):
        """Establish baseline distribution from predictions."""
        arr = np.clip(np.asarray(probs, dtype=float).flatten(), 0.0, 1.0)
        with self._lock:
            self._reference_probs = arr
            self._live_probs.clear()

    def _get_distributions(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Compute smoothed decile probabilities for reference and live data."""
        if self._reference_probs is None or len(self._reference_probs) == 0 or len(self._live_probs) == 0:
            return None

        bins = np.linspace(0.0, 1.0, 11)
        ref_counts, _ = np.histogram(self._reference_probs, bins=bins)
        live_counts, _ = np.histogram(list(self._live_probs), bins=bins)

        # Check exact match
        if np.array_equal(ref_counts, live_counts):
            q = ref_counts / np.sum(ref_counts) if np.sum(ref_counts) > 0 else np.ones(10) / 10.0
            return q, q

        eps = 1e-6
        q = (ref_counts + eps) / (np.sum(ref_counts) + eps * len(ref_counts))
        p = (live_counts + eps) / (np.sum(live_counts) + eps * len(live_counts))
        return p, q

    def compute_psi(self) -> float:
        """Return Population Stability Index (PSI) vs. reference distribution."""
        with self._lock:
            dists = self._get_distributions()
            if dists is None:
                return 0.0

            p, q = dists
            if np.array_equal(p, q):
                return 0.0

            psi = np.sum((p - q) * np.log(p / q))
            return float(max(0.0, psi))

    def compute_kl(self) -> float:
        """Return Kullback-Leibler divergence (live || reference)."""
        with self._lock:
            dists = self._get_distributions()
            if dists is None:
                return 0.0

            p, q = dists
            if np.array_equal(p, q):
                return 0.0

            kl = np.sum(p * np.log(p / q))
            return float(max(0.0, kl))

    def drift_status(self) -> Dict[str, Any]:
        """Evaluate current concept drift status."""
        psi = self.compute_psi()
        kl = self.compute_kl()

        if psi >= self.PSI_RETRAIN_THRESHOLD:
            status = "RETRAIN_REQUIRED"
        elif psi >= self.PSI_WARN_THRESHOLD or kl >= self.KL_WARN_THRESHOLD:
            status = "WARNING"
        else:
            status = "OK"

        with self._lock:
            ref_n = len(self._reference_probs) if self._reference_probs is not None else 0
            live_n = len(self._live_probs)

        return {
            "psi": float(psi),
            "kl": float(kl),
            "status": status,
            "reference_n": ref_n,
            "live_n": live_n,
        }

    def reset_reference(self):
        """Reset reference to current live distribution and clear live buffer."""
        with self._lock:
            if len(self._live_probs) > 0:
                self._reference_probs = np.array(self._live_probs, dtype=float)
            self._live_probs.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
