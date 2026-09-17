"""Multi-Room Spatial Mesh — Room Manager (Milestone 6.1).

Maintains one independent processing context per room (identified by an integer
room_id byte embedded in every UDP packet). Each context owns its own:
  - MultiLinkFusionEngine  (CSI coincidence/quiescence state machine)
  - DualFusionEngine       (CSI + radar consensus)
  - AdaptiveCalibrator     (background EMA noise tracking)

Thread-safety: get_or_create_room() uses a reentrant lock; individual engine
operations are protected by each engine's own internal locks.

Usage:
    manager = RoomManager()
    ctx = manager.get_or_create_room(room_id=1)
    state = manager.on_csi_packet(room_id=1, features=csi_features)
    state = manager.on_radar_packet(room_id=1, telemetry=radar_tel)
    active = manager.get_active_rooms()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("room_manager")

# Lazy imports to avoid hard dependency at module load time
_ENGINES_AVAILABLE = False
try:
    from hub.csi_pipeline.multi_link_fusion import MultiLinkFusionEngine
    from hub.fusion_engine import DualFusionEngine
    from hub.adaptive_calibrator import AdaptiveCalibrator
    _ENGINES_AVAILABLE = True
except ImportError as _import_err:
    logger.warning("Engine imports unavailable: %s", _import_err)

# Seconds of no packets before a room is considered INACTIVE
HANDOVER_TIMEOUT_SEC: float = 30.0


@dataclass
class RoomContext:
    """Per-room state container."""

    room_id: int
    csi_engine: Any = None    # MultiLinkFusionEngine
    fusion_engine: Any = None  # DualFusionEngine
    calibrator: Any = None     # AdaptiveCalibrator
    last_csi_time: float = 0.0
    last_radar_time: float = 0.0
    last_alert_time: float = 0.0
    alert_count: int = 0
    latest_state: str = "NORMAL"
    _extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_seen(self) -> float:
        return max(self.last_csi_time, self.last_radar_time)

    @property
    def is_active(self) -> bool:
        """True if any packet received within the handover timeout window."""
        elapsed = time.time() - self.last_seen
        return self.last_seen > 0.0 and elapsed < HANDOVER_TIMEOUT_SEC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "is_active": self.is_active,
            "latest_state": self.latest_state,
            "last_seen": self.last_seen,
            "alert_count": self.alert_count,
            "seconds_since_packet": round(time.time() - self.last_seen, 1) if self.last_seen > 0 else None,
        }


class RoomManager:
    """Thread-safe factory and dispatcher for per-room processing contexts.

    Args:
        handover_timeout_sec: Seconds of silence before a room goes INACTIVE.
    """

    def __init__(self, handover_timeout_sec: float = HANDOVER_TIMEOUT_SEC):
        self.handover_timeout_sec = handover_timeout_sec
        self._rooms: Dict[int, RoomContext] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Room lifecycle
    # ------------------------------------------------------------------

    def get_or_create_room(self, room_id: int) -> RoomContext:
        """Return existing RoomContext or create a fresh one with new engines."""
        with self._lock:
            if room_id in self._rooms:
                return self._rooms[room_id]

            ctx = RoomContext(room_id=room_id)
            if _ENGINES_AVAILABLE:
                ctx.csi_engine = MultiLinkFusionEngine()
                ctx.fusion_engine = DualFusionEngine(csi_engine=ctx.csi_engine)
                ctx.calibrator = AdaptiveCalibrator(csi_engine=ctx.csi_engine)

            self._rooms[room_id] = ctx
            logger.info("Created new RoomContext for room_id=%d", room_id)
            return ctx

    def get_room(self, room_id: int) -> Optional[RoomContext]:
        """Return an existing RoomContext, or None if not yet created."""
        with self._lock:
            return self._rooms.get(room_id)

    def all_rooms(self) -> List[RoomContext]:
        """Return a snapshot list of all registered RoomContexts."""
        with self._lock:
            return list(self._rooms.values())

    def get_active_rooms(self) -> List[Dict[str, Any]]:
        """Return dict summaries for all rooms that have received packets recently."""
        with self._lock:
            return [ctx.to_dict() for ctx in self._rooms.values()]

    def remove_room(self, room_id: int) -> bool:
        """Deregister a room (e.g., when a node cluster is taken offline)."""
        with self._lock:
            if room_id in self._rooms:
                del self._rooms[room_id]
                logger.info("Removed RoomContext for room_id=%d", room_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Packet dispatch
    # ------------------------------------------------------------------

    def on_csi_packet(self, room_id: int, features: Any) -> str:
        """Route a CSI feature update to the correct room engine.

        Args:
            room_id: Integer room identifier from packet header.
            features: CSIDynamicFeatures object.

        Returns:
            Latest UnifiedFallState string for the room.
        """
        ctx = self.get_or_create_room(room_id)
        ctx.last_csi_time = time.time()

        if ctx.fusion_engine is None:
            # No engine available (import failed or unit test stub)
            return ctx.latest_state

        try:
            state = ctx.fusion_engine.update_csi(features)
            ctx.latest_state = str(state)
            # Propagate to calibrator
            if ctx.calibrator is not None:
                ctx.calibrator.update(features.moving_variance)
        except Exception as exc:
            logger.error("CSI update error in room %d: %s", room_id, exc)

        return ctx.latest_state

    def on_radar_packet(self, room_id: int, telemetry: Any) -> str:
        """Route a radar telemetry update to the correct room engine.

        Args:
            room_id: Integer room identifier.
            telemetry: RadarTelemetry object.

        Returns:
            Latest UnifiedFallState string for the room.
        """
        ctx = self.get_or_create_room(room_id)
        ctx.last_radar_time = time.time()

        if ctx.fusion_engine is None:
            return ctx.latest_state

        try:
            state = ctx.fusion_engine.update_radar(telemetry)
            ctx.latest_state = str(state)
        except Exception as exc:
            logger.error("Radar update error in room %d: %s", room_id, exc)

        return ctx.latest_state
