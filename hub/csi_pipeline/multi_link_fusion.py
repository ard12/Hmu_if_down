"""Multi-link spatial coincidence engine and post-fall inactivity verification."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import time
from typing import Deque, Dict, List, Optional, Tuple

from .pca_features import CSIDynamicFeatures


class CSIFallState(Enum):
    NORMAL = "Normal"
    SUSPECTED_FALL = "Fall Suspected"
    CONFIRMED_FALL = "FALL DETECTED"
    RECOVERED = "Recovered"


@dataclass
class LinkEvent:
    node_id: int
    timestamp: float
    velocity_mps: float
    energy_surge: float


class MultiLinkFusionEngine:
    """Combines multi-bistatic RF links to detect coincident fall kinematics."""

    def __init__(
        self,
        coincidence_window_sec: float = 0.4,
        min_coincident_links: int = 2,
        post_fall_quiescence_sec: float = 4.0,
        motionless_variance_threshold: float = 0.08,
    ):
        self.coincidence_window = coincidence_window_sec
        self.min_links = min_coincident_links
        self.quiescence_duration = post_fall_quiescence_sec
        self.motionless_var_thresh = motionless_variance_threshold

        # Recent velocity burst events per node
        self.recent_bursts: Dict[int, Deque[LinkEvent]] = {
            1: deque(maxlen=10),
            2: deque(maxlen=10),
            3: deque(maxlen=10),
        }

        # State machine
        self.state = CSIFallState.NORMAL
        self.suspected_timestamp: Optional[float] = None
        self.coincident_nodes: List[int] = []

    def register_feature(self, features: CSIDynamicFeatures, current_time: Optional[float] = None) -> CSIFallState:
        """Evaluate incoming feature from one of the tracker nodes and update state."""
        now = current_time if current_time is not None else time.time()
        node_id = features.node_id

        if node_id not in self.recent_bursts:
            self.recent_bursts[node_id] = deque(maxlen=10)

        # Record velocity burst if detected
        if features.is_velocity_burst:
            self.recent_bursts[node_id].append(
                LinkEvent(
                    node_id=node_id,
                    timestamp=now,
                    velocity_mps=features.dominant_velocity_mps,
                    energy_surge=features.energy_surge_ratio,
                )
            )

        # 1. State: NORMAL -> Check for Coincident Bursts
        if self.state in (CSIFallState.NORMAL, CSIFallState.RECOVERED):
            coincident = self._check_coincidence(now)
            if len(coincident) >= self.min_links:
                self.state = CSIFallState.SUSPECTED_FALL
                self.suspected_timestamp = now
                self.coincident_nodes = coincident

        # 2. State: SUSPECTED_FALL -> Monitor Post-Impact Quiescence
        elif self.state == CSIFallState.SUSPECTED_FALL:
            elapsed = now - (self.suspected_timestamp or now)

            # If sudden vigorous motion resumes before quiescence time, person stood back up
            if features.is_velocity_burst and elapsed > 0.8:
                self.state = CSIFallState.RECOVERED
                self.suspected_timestamp = None
            elif elapsed >= self.quiescence_duration:
                # Stillness sustained over the threshold window -> Fall Confirmed!
                if features.moving_variance <= self.motionless_var_thresh:
                    self.state = CSIFallState.CONFIRMED_FALL

        # 3. State: CONFIRMED_FALL -> Check for eventual recovery
        elif self.state == CSIFallState.CONFIRMED_FALL:
            if features.dominant_velocity_mps > 1.2 and features.moving_variance > self.motionless_var_thresh * 3:
                self.state = CSIFallState.RECOVERED
                self.suspected_timestamp = None

        return self.state

    def _check_coincidence(self, current_time: float) -> List[int]:
        """Find how many distinct nodes logged a burst within the coincidence window."""
        coincident_nodes = []
        for nid, events in self.recent_bursts.items():
            for ev in reversed(events):
                if current_time - ev.timestamp <= self.coincidence_window:
                    coincident_nodes.append(nid)
                    break  # Count each node at most once
        return coincident_nodes

    def reset(self):
        """Reset internal state machine."""
        self.state = CSIFallState.NORMAL
        self.suspected_timestamp = None
        self.coincident_nodes.clear()
        for q in self.recent_bursts.values():
            q.clear()
