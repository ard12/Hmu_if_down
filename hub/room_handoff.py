"""
Room Boundary Handoff State Machine.
Tracks subject movement across room boundaries to eliminate transition blind spots.
"""

from collections import deque
from enum import Enum, auto
import logging
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


class HandoffState(Enum):
    IDLE = auto()       # Subject firmly in one room
    DETECTING = auto()  # RSSI drop suggests movement toward boundary
    DUAL = auto()       # Actively monitoring source + destination rooms
    CONFIRMED = auto()  # Subject confirmed in new room
    ABORT = auto()      # Subject returned to source room


class RoomHandoffManager:
    """
    Tracks per-subject movement across room boundaries.
    Coordinates with RoomManager to temporarily dual-monitor adjacent rooms.
    """

    RSSI_DROP_THRESHOLD_DB = -10.0  # dB drop triggers DETECTING
    DUAL_MONITOR_DURATION_S = 8.0   # Seconds to dual-monitor during transition
    HYSTERESIS_FRAMES = 5           # Consecutive frames before state transition

    def __init__(self, adjacent_rooms: Optional[Dict[str, List[str]]] = None):
        self.adjacent_rooms = adjacent_rooms or {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[str, str, str, float], None]] = []

        # Per-subject tracking data: subject_id -> tracking dict
        # {
        #   "state": HandoffState,
        #   "current_room": str,
        #   "target_room": Optional[str],
        #   "baseline_rssi": Dict[str, float],
        #   "drop_count": int,
        #   "dual_start_time": float,
        #   "history": Deque[Tuple[str, float, float]]  # (room_id, rssi, timestamp)
        # }
        self._subjects: Dict[str, dict] = {}
        self._event_listeners: List[Callable[[str, dict], None]] = []

    def on_handoff_completed(self, callback: Callable[[str, str, str, float], None]):
        """Register callback(from_room, to_room, subject_id, timestamp)."""
        with self._lock:
            self._callbacks.append(callback)

    def add_event_listener(self, listener: Callable[[str, dict], None]):
        """Register generic event listener(event_type, payload)."""
        with self._lock:
            self._event_listeners.append(listener)

    def _emit_event(self, event_type: str, payload: dict):
        for listener in self._event_listeners:
            try:
                listener(event_type, payload)
            except Exception as e:
                logger.warning(f"Error in handoff event listener: {e}")

    def _get_subject_record(self, subject_id: str, initial_room: str) -> dict:
        if subject_id not in self._subjects:
            self._subjects[subject_id] = {
                "state": HandoffState.IDLE,
                "current_room": initial_room,
                "target_room": None,
                "baseline_rssi": {},
                "drop_count": 0,
                "dual_start_time": 0.0,
                "history": deque(maxlen=50),
            }
        return self._subjects[subject_id]

    def update(
        self,
        room_id: str,
        node_id: str,
        rssi: float,
        timestamp: Optional[float] = None,
        subject_id: str = "default",
    ) -> HandoffState:
        """
        Feed per-frame RSSI and return current HandoffState.
        Emits 'handoff_started', 'handoff_completed', or 'handoff_aborted' events.
        """
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            record = self._get_subject_record(subject_id, room_id)
            current_room = record["current_room"]
            state = record["state"]
            record["history"].append((room_id, rssi, ts))

            # Maintain EMA baseline for the current room
            baselines = record["baseline_rssi"]
            if room_id not in baselines:
                baselines[room_id] = rssi
            else:
                baselines[room_id] = 0.9 * baselines[room_id] + 0.1 * rssi

            baseline = baselines.get(current_room, rssi)

            if state == HandoffState.IDLE:
                if room_id == current_room:
                    # Check for RSSI drop below threshold
                    if rssi <= baseline + self.RSSI_DROP_THRESHOLD_DB:
                        record["drop_count"] += 1
                        if record["drop_count"] >= self.HYSTERESIS_FRAMES:
                            record["state"] = HandoffState.DETECTING
                            self._emit_event("handoff_started", {
                                "subject_id": subject_id,
                                "from_room": current_room,
                                "timestamp": ts,
                            })
                    else:
                        record["drop_count"] = max(0, record["drop_count"] - 1)
                elif room_id != current_room:
                    # Packet received from a different room while in IDLE
                    # If strong, could trigger detecting
                    if rssi > baseline:
                        record["drop_count"] += 1
                        if record["drop_count"] >= self.HYSTERESIS_FRAMES:
                            record["target_room"] = room_id
                            record["state"] = HandoffState.DETECTING
                            self._emit_event("handoff_started", {
                                "subject_id": subject_id,
                                "from_room": current_room,
                                "target_room": room_id,
                                "timestamp": ts,
                            })

            elif state == HandoffState.DETECTING:
                # In DETECTING: look for target room signal or transition to DUAL
                if room_id != current_room:
                    record["target_room"] = room_id
                    record["state"] = HandoffState.DUAL
                    record["dual_start_time"] = ts
                else:
                    # If still only receiving from current room, check if drop continues
                    if rssi <= baseline + self.RSSI_DROP_THRESHOLD_DB:
                        # Transition to DUAL to search for adjacent rooms
                        record["state"] = HandoffState.DUAL
                        record["dual_start_time"] = ts
                        # Pick adjacent room if known
                        adjs = self.adjacent_rooms.get(current_room, [])
                        if adjs and not record["target_room"]:
                            record["target_room"] = adjs[0]
                    else:
                        # Recovered in current room
                        record["state"] = HandoffState.IDLE
                        record["drop_count"] = 0

            elif state == HandoffState.DUAL:
                target_room = record["target_room"]
                dual_elapsed = ts - record["dual_start_time"]

                if room_id == target_room and rssi > baseline:
                    # Target room signal has recovered / is stronger
                    record["state"] = HandoffState.CONFIRMED
                    from_room = current_room
                    to_room = target_room
                    record["current_room"] = to_room
                    record["target_room"] = None
                    record["drop_count"] = 0

                    self._emit_event("handoff_completed", {
                        "subject_id": subject_id,
                        "from_room": from_room,
                        "to_room": to_room,
                        "timestamp": ts,
                    })
                    for cb in self._callbacks:
                        try:
                            cb(from_room, to_room, subject_id, ts)
                        except Exception as e:
                            logger.error(f"Error in handoff callback: {e}")

                elif room_id == current_room and rssi > baseline - 3.0:
                    # Source room RSSI recovered: subject turned back
                    record["state"] = HandoffState.ABORT
                    record["target_room"] = None
                    record["drop_count"] = 0
                    self._emit_event("handoff_aborted", {
                        "subject_id": subject_id,
                        "from_room": current_room,
                        "timestamp": ts,
                    })

                elif dual_elapsed > self.DUAL_MONITOR_DURATION_S:
                    # Dual monitor timeout reached: confirm if target had any packets, else abort
                    if target_room:
                        record["state"] = HandoffState.CONFIRMED
                        from_room = current_room
                        to_room = target_room
                        record["current_room"] = to_room
                        record["target_room"] = None
                        record["drop_count"] = 0

                        self._emit_event("handoff_completed", {
                            "subject_id": subject_id,
                            "from_room": from_room,
                            "to_room": to_room,
                            "timestamp": ts,
                        })
                        for cb in self._callbacks:
                            try:
                                cb(from_room, to_room, subject_id, ts)
                            except Exception as e:
                                logger.error(f"Error in handoff callback: {e}")
                    else:
                        record["state"] = HandoffState.ABORT
                        record["drop_count"] = 0
                        self._emit_event("handoff_aborted", {
                            "subject_id": subject_id,
                            "from_room": current_room,
                            "timestamp": ts,
                        })

            elif state in (HandoffState.CONFIRMED, HandoffState.ABORT):
                # Settle back to IDLE
                record["state"] = HandoffState.IDLE
                record["drop_count"] = 0

            return record["state"]

    def get_active_rooms(self, subject_id: str = "default") -> List[str]:
        """
        Return list of rooms currently monitored for subject.
        Returns 2 rooms during DUAL state, 1 during IDLE/CONFIRMED/ABORT.
        """
        with self._lock:
            record = self._subjects.get(subject_id)
            if not record:
                return ["default_room"]

            state = record["state"]
            current_room = record["current_room"]
            target_room = record.get("target_room")

            if state == HandoffState.DUAL and target_room and target_room != current_room:
                return [current_room, target_room]

            return [current_room]

    def set_subject_room(self, subject_id: str, room_id: str):
        """Explicitly set subject's current room."""
        with self._lock:
            record = self._get_subject_record(subject_id, room_id)
            record["current_room"] = room_id
            record["state"] = HandoffState.IDLE
            record["target_room"] = None
            record["drop_count"] = 0
