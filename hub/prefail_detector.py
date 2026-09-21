"""
Pre-Fall Behavioural Anomaly Detector.
Uses rolling gait analysis to score pre-fall risk over a 120-second window.
Triggers proactive WATCH and IMMEDIATE_INTERVENTION alerts when risk thresholds are exceeded.
"""

from collections import deque
import time
from typing import Deque, Dict, List, Optional


class PreFallAnomalyDetector:
    """
    Evaluates rolling gait deterioration, velocity drops, and hesitation patterns.

    Risk score components:
      - Gait deterioration: SHUFFLE for >= 10s -> +40 points
      - Velocity drop: envelope_peak < 50% baseline for >= 15s -> +30 points
      - Hesitation pattern: >= 3 direction reversals/min -> +20 points
      - High Morse Fall Scale (>= 45) from patient context -> +10 points
      Total >= 70 -> WATCH alert; >= 90 -> IMMEDIATE_INTERVENTION alert
    """

    WATCH_THRESHOLD = 70
    INTERVENTION_THRESHOLD = 90
    ASSESSMENT_WINDOW_S = 120.0

    def __init__(self):
        # Stores (timestamp: float, gait_dict: dict, patient_context: Optional[dict])
        self._history: Deque[dict] = deque(maxlen=200)
        # Snapshot history for dashboard (up to 60 snapshots)
        self._snapshots: Deque[dict] = deque(maxlen=60)
        self._last_snapshot_time = 0.0

        # Dynamic state tracking
        self.baseline_velocity = 0.8  # initial expected walking velocity in m/s
        self._shuffle_start_time: Optional[float] = None
        self._shuffle_duration = 0.0
        self._vel_drop_start_time: Optional[float] = None
        self._vel_drop_duration = 0.0
        self._reversals: Deque[float] = deque(maxlen=20)

    def record_reversal(self, timestamp: Optional[float] = None):
        """Record a micro-hesitation or directional reversal event."""
        ts = timestamp if timestamp is not None else time.time()
        self._reversals.append(ts)

    def update(
        self,
        gait: dict,
        patient_context: Optional[dict] = None,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Evaluate new gait frame and return risk evaluation dictionary.

        Returns:
          {
            "risk_score": int,
            "risk_level": "NORMAL" | "ELEVATED" | "WATCH" | "IMMEDIATE_INTERVENTION",
            "contributing_factors": list[str],
            "seconds_until_escalation": Optional[float],
          }
        """
        ts = timestamp if timestamp is not None else time.time()
        gait_class = gait.get("gait_class", "NORMAL")
        vel_peak = float(gait.get("velocity_envelope_peak", 0.0))
        reversals_in_gait = int(gait.get("reversals", 0))

        if reversals_in_gait > 0:
            for _ in range(reversals_in_gait):
                self._reversals.append(ts)

        # Clean up events older than 60s
        cutoff_reversals = ts - 60.0
        while self._reversals and self._reversals[0] < cutoff_reversals:
            self._reversals.popleft()

        # Update baseline velocity during NORMAL gait
        if gait_class == "NORMAL" and vel_peak > 0.3:
            self.baseline_velocity = 0.95 * self.baseline_velocity + 0.05 * vel_peak

        # 1. Shuffle gait duration tracking
        if gait_class == "SHUFFLE":
            if self._shuffle_start_time is None:
                self._shuffle_start_time = ts
            self._shuffle_duration = ts - self._shuffle_start_time
        else:
            if gait_class == "NORMAL":
                self._shuffle_start_time = None
                self._shuffle_duration = 0.0

        # 2. Velocity drop tracking
        vel_threshold = 0.5 * max(0.2, self.baseline_velocity)
        if vel_peak < vel_threshold and gait_class != "STATIONARY":
            if self._vel_drop_start_time is None:
                self._vel_drop_start_time = ts
            self._vel_drop_duration = ts - self._vel_drop_start_time
        else:
            if vel_peak >= vel_threshold:
                self._vel_drop_start_time = None
                self._vel_drop_duration = 0.0

        # 3. Compute score and contributing factors
        risk_score = 0
        contributing_factors: List[str] = []

        if self._shuffle_duration >= 10.0:
            risk_score += 40
            contributing_factors.append("Sustained shuffle gait (>=10s)")

        if self._vel_drop_duration >= 15.0:
            risk_score += 30
            contributing_factors.append("Velocity envelope drop (>50% reduction for >=15s)")

        if len(self._reversals) >= 3:
            risk_score += 20
            contributing_factors.append("Hesitation gait pattern (>=3 direction reversals/min)")

        if patient_context is not None:
            morse = int(patient_context.get("morse_fall_scale", 0))
            if morse >= 45:
                risk_score += 10
                contributing_factors.append("High Morse Fall Scale (>=45)")

        risk_score = min(100, max(0, risk_score))

        # Classify risk level
        if risk_score >= self.INTERVENTION_THRESHOLD:
            risk_level = "IMMEDIATE_INTERVENTION"
            seconds_until_escalation = 0.0
        elif risk_score >= self.WATCH_THRESHOLD:
            risk_level = "WATCH"
            # Estimated seconds until reaching intervention threshold
            seconds_until_escalation = max(5.0, round(30.0 - self._shuffle_duration, 1))
        elif risk_score >= 30:
            risk_level = "ELEVATED"
            seconds_until_escalation = None
        else:
            risk_level = "NORMAL"
            seconds_until_escalation = None

        result = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_factors": contributing_factors,
            "seconds_until_escalation": seconds_until_escalation,
            "timestamp": ts,
        }

        # Store snapshot every >= 2s
        if ts - self._last_snapshot_time >= 2.0 or not self._snapshots:
            self._snapshots.append({
                "risk_score": risk_score,
                "risk_level": risk_level,
                "timestamp": ts,
            })
            self._last_snapshot_time = ts

        return result

    def get_history(self) -> List[dict]:
        """Return last 60 risk score snapshots."""
        return list(self._snapshots)
