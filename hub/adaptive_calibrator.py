"""Continuous background adaptive noise calibrator for dynamic environmental drift tracking.

Continuously updates ambient room baseline variance using an Exponential Moving Average (EMA)
during verified quiescent periods, adapting CSI stillness thresholds to seasonal temperature,
humidity, and RF multipath shifts without requiring manual recalibration.
"""

from collections import deque
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple


class AdaptiveCalibrator:
    """Monitors quiescent variance and dynamically updates CSI stillness thresholds."""

    def __init__(
        self,
        csi_engine=None,
        alpha: float = 0.02,
        initial_baseline: float = 0.03,
        threshold_multiplier: float = 2.5,
        min_threshold: float = 0.04,
        max_threshold: float = 0.20,
        quiescent_max_variance: float = 0.10,
    ):
        """
        Args:
            csi_engine: Optional reference to MultiLinkFusionEngine to auto-update.
            alpha: EMA smoothing factor (0.01 - 0.05 for slow, stable drift tracking).
            initial_baseline: Initial quiescent noise floor variance.
            threshold_multiplier: Multiplier applied to baseline variance to derive stillness threshold.
            min_threshold: Absolute lower bound on adapted threshold.
            max_threshold: Absolute upper bound on adapted threshold.
            quiescent_max_variance: Maximum variance sample admitted into baseline adaptation.
        """
        self.csi_engine = csi_engine
        self.alpha = alpha
        self.baseline_variance = initial_baseline
        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.quiescent_max_var = quiescent_max_variance

        self.current_threshold = max(
            min_threshold,
            min(initial_baseline * threshold_multiplier, max_threshold),
        )
        if self.csi_engine and hasattr(self.csi_engine, "motionless_var_thresh"):
            self.csi_engine.motionless_var_thresh = self.current_threshold

        self.samples_collected = 0
        self.last_update_time = 0.0
        self._lock = threading.Lock()
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=100)  # (timestamp, baseline, threshold)

    def update(
        self,
        variance: float,
        is_quiescent: bool = True,
        current_time: Optional[float] = None,
    ) -> float:
        """Feed a variance sample into the adaptive tracker.

        Args:
            variance: Current moving variance measured from CSI window.
            is_quiescent: True if radar or state machine confirms no active kinetic motion.
            current_time: Optional timestamp (epoch seconds).

        Returns:
            float: Updated stillness variance threshold.
        """
        now = current_time if current_time is not None else time.time()

        with self._lock:
            # Only admit samples during verified quiescent periods below outlier ceiling
            if is_quiescent and (variance <= self.quiescent_max_var):
                self.baseline_variance = (1.0 - self.alpha) * self.baseline_variance + self.alpha * variance
                self.samples_collected += 1
                self.last_update_time = now

                # Compute new threshold
                new_threshold = max(
                    self.min_threshold,
                    min(self.baseline_variance * self.threshold_multiplier, self.max_threshold),
                )
                self.current_threshold = round(new_threshold, 4)

                # Auto-update csi_engine if attached
                if self.csi_engine and hasattr(self.csi_engine, "motionless_var_thresh"):
                    self.csi_engine.motionless_var_thresh = self.current_threshold

                self.history.append((now, round(self.baseline_variance, 5), self.current_threshold))

            return self.current_threshold

    def get_status(self) -> Dict[str, float]:
        """Return diagnostic metrics of current adaptation state."""
        with self._lock:
            return {
                "baseline_variance": round(self.baseline_variance, 5),
                "current_threshold": round(self.current_threshold, 4),
                "samples_collected": self.samples_collected,
                "last_update_time": self.last_update_time,
            }

    def reset(self, baseline: Optional[float] = None):
        """Reset adaptation state."""
        with self._lock:
            if baseline is not None:
                self.baseline_variance = baseline
            self.current_threshold = max(
                self.min_threshold,
                min(self.baseline_variance * self.threshold_multiplier, self.max_threshold),
            )
            self.samples_collected = 0
            self.history.clear()
            if self.csi_engine and hasattr(self.csi_engine, "motionless_var_thresh"):
                self.csi_engine.motionless_var_thresh = self.current_threshold
