"""Ring-buffer of (feature_vector, label) for incremental retraining."""

import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class InsufficientDataError(Exception):
    """Raised when training buffer does not contain enough positive labels."""
    pass


class TrainingBuffer:
    """Ring-buffer of (feature_vector, label) for incremental retraining.

    Maintains a rolling window of confirmed and candidate fall events
    with thread-safe access and ground-truth nursing annotation support.
    """

    MAX_BUFFER_SIZE = 2000

    def __init__(self, max_size: int = MAX_BUFFER_SIZE):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._event_labels: Dict[str, bool] = {}

    def append(
        self,
        features: np.ndarray,
        label: int,
        timestamp: Optional[float] = None,
        room_id: str = "room_default",
        event_id: Optional[str] = None,
    ):
        """Append feature vector and label to the buffer."""
        ts = float(timestamp) if timestamp is not None else time.time()
        feat = np.asarray(features, dtype=float).flatten()
        lbl = int(label)

        with self._lock:
            if event_id and event_id in self._event_labels:
                lbl = 1 if self._event_labels[event_id] else 0

            record = {
                "features": feat,
                "label": lbl,
                "timestamp": ts,
                "room_id": str(room_id),
                "event_id": str(event_id) if event_id else None,
                "confirmed": bool(lbl == 1),
            }
            self._buffer.append(record)

    def label_event(self, event_id: str, confirmed: bool):
        """Link stored feature to ground-truth label from nursing staff."""
        with self._lock:
            self._event_labels[event_id] = bool(confirmed)
            for item in self._buffer:
                if item.get("event_id") == event_id:
                    item["confirmed"] = bool(confirmed)
                    item["label"] = 1 if confirmed else 0

    def get_batch(self, min_positives: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for model fitting.

        Raises InsufficientDataError if positive samples < min_positives.
        """
        with self._lock:
            if not self._buffer:
                raise InsufficientDataError("Training buffer is empty")

            positives = sum(1 for r in self._buffer if r["label"] == 1)
            if positives < min_positives:
                raise InsufficientDataError(
                    f"Insufficient positive labels: {positives} < {min_positives}"
                )

            X_list = [r["features"] for r in self._buffer]
            y_list = [r["label"] for r in self._buffer]

            return np.array(X_list, dtype=float), np.array(y_list, dtype=int)

    def stats(self) -> Dict[str, Any]:
        """Return current buffer statistics."""
        with self._lock:
            total = len(self._buffer)
            if total == 0:
                return {
                    "total": 0,
                    "positives": 0,
                    "negatives": 0,
                    "oldest_ts": 0.0,
                    "newest_ts": 0.0,
                }

            positives = sum(1 for r in self._buffer if r["label"] == 1)
            negatives = total - positives
            oldest_ts = self._buffer[0]["timestamp"]
            newest_ts = self._buffer[-1]["timestamp"]

            return {
                "total": total,
                "positives": positives,
                "negatives": negatives,
                "oldest_ts": oldest_ts,
                "newest_ts": newest_ts,
            }

    def __getstate__(self):
        """Allow serialization via pickle without threading.Lock."""
        state = self.__dict__.copy()
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state):
        """Restore serialized state and recreate threading.Lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
