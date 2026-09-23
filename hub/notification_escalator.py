"""
Progressive Notification Escalation Engine.

Manages tiered alert escalation to mitigate caregiver alarm fatigue:
  TIER 1 (0s)   -> Silent push notification to primary caregiver
  TIER 2 (30s)  -> Audible alert + vibration to primary caregiver
  TIER 3 (60s)  -> Push notification to ALL registered caregivers
  TIER 4 (120s) -> Auto-dial emergency contact phone number (Twilio/webhook)
  TIER 5 (180s) -> Dispatch to facility nursing station / 911 webhook

Halts when acknowledged; de-escalates upon detected patient recovery.

@req SRS-UX-001
@req SRS-UX-002
"""
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EscalationTier(Enum):
    TIER_1_SILENT_PUSH = 1
    TIER_2_AUDIBLE_ALERT = 2
    TIER_3_BROADCAST_ALL = 3
    TIER_4_AUTODIAL_CONTACT = 4
    TIER_5_EMERGENCY_DISPATCH = 5


class EscalationState(Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class EscalationPolicy:
    """Escalation timeouts and thresholds."""
    tiers: List[EscalationTier] = field(
        default_factory=lambda: [
            EscalationTier.TIER_1_SILENT_PUSH,
            EscalationTier.TIER_2_AUDIBLE_ALERT,
            EscalationTier.TIER_3_BROADCAST_ALL,
            EscalationTier.TIER_4_AUTODIAL_CONTACT,
            EscalationTier.TIER_5_EMERGENCY_DISPATCH,
        ]
    )
    timeout_per_tier_s: List[float] = field(
        default_factory=lambda: [30.0, 30.0, 60.0, 60.0]
    )
    auto_dial_enabled: bool = False
    emergency_webhook_url: Optional[str] = None


@dataclass
class EscalationEvent:
    event_id: str
    severity: str
    room_id: int
    start_time: float
    current_tier: EscalationTier
    state: EscalationState = EscalationState.PENDING
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    last_tier_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "severity": self.severity,
            "room_id": self.room_id,
            "start_time": self.start_time,
            "current_tier": self.current_tier.name,
            "tier_level": self.current_tier.value,
            "state": self.state.value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "response_time_s": (
                self.acknowledged_at - self.start_time if self.acknowledged_at else None
            ),
            "history": self.history,
        }


class NotificationEscalator:
    """State engine managing progressive caregiver notification tiers."""

    def __init__(
        self,
        policy: Optional[EscalationPolicy] = None,
        alert_dispatcher: Optional[Any] = None,
    ):
        self.policy = policy or EscalationPolicy()
        self.alert_dispatcher = alert_dispatcher
        self._events: Dict[str, EscalationEvent] = {}

    def start_escalation(
        self,
        event_id: str,
        severity: str = "critical",
        room_id: int = 1,
        timestamp: Optional[float] = None,
    ) -> EscalationEvent:
        """Start a new escalation process for a fall event."""
        now = time.time() if timestamp is None else timestamp
        initial_tier = self.policy.tiers[0]
        event = EscalationEvent(
            event_id=event_id,
            severity=severity,
            room_id=room_id,
            start_time=now,
            current_tier=initial_tier,
            last_tier_time=now,
            state=EscalationState.PENDING,
            history=[{"tier": initial_tier.name, "timestamp": now, "action": "initiated"}],
        )
        self._events[event_id] = event
        self._dispatch_tier_action(event, initial_tier)
        return event

    def advance_time(self, current_time: float) -> List[EscalationEvent]:
        """Check all active events and advance tiers if timeout exceeded."""
        advanced_events = []
        for event in self._events.values():
            if event.state not in (EscalationState.PENDING, EscalationState.ESCALATED):
                continue

            current_tier_idx = self.policy.tiers.index(event.current_tier)
            if current_tier_idx < len(self.policy.timeout_per_tier_s):
                timeout = self.policy.timeout_per_tier_s[current_tier_idx]
                if current_time - event.last_tier_time >= timeout:
                    next_tier_idx = current_tier_idx + 1
                    if next_tier_idx < len(self.policy.tiers):
                        event.current_tier = self.policy.tiers[next_tier_idx]
                        event.state = EscalationState.ESCALATED
                        event.last_tier_time = current_time
                        event.history.append({
                            "tier": event.current_tier.name,
                            "timestamp": current_time,
                            "action": "escalated",
                        })
                        self._dispatch_tier_action(event, event.current_tier)
                        advanced_events.append(event)
                    else:
                        event.state = EscalationState.EXPIRED
                        event.history.append({
                            "tier": event.current_tier.name,
                            "timestamp": current_time,
                            "action": "expired_unacknowledged",
                        })
                        advanced_events.append(event)
            else:
                # Max tier exceeded
                if current_time - event.last_tier_time >= 60.0:
                    event.state = EscalationState.EXPIRED
                    advanced_events.append(event)

        return advanced_events

    def acknowledge(
        self, event_id: str, caregiver_id: str, timestamp: Optional[float] = None
    ) -> bool:
        """Acknowledge alert by a caregiver, halting further escalation."""
        if event_id not in self._events:
            return False
        event = self._events[event_id]
        if event.state in (EscalationState.RESOLVED, EscalationState.EXPIRED):
            return False

        now = time.time() if timestamp is None else timestamp
        event.state = EscalationState.ACKNOWLEDGED
        event.acknowledged_by = caregiver_id
        event.acknowledged_at = now
        event.history.append({
            "tier": event.current_tier.name,
            "timestamp": now,
            "action": f"acknowledged by {caregiver_id}",
        })
        return True

    def resolve_recovery(self, event_id: str, timestamp: Optional[float] = None) -> bool:
        """Resolve alert when patient recovery is confirmed."""
        if event_id not in self._events:
            return False
        event = self._events[event_id]
        now = time.time() if timestamp is None else timestamp
        event.state = EscalationState.RESOLVED
        event.history.append({
            "tier": event.current_tier.name,
            "timestamp": now,
            "action": "resolved_recovery",
        })
        return True

    def get_event(self, event_id: str) -> Optional[EscalationEvent]:
        return self._events.get(event_id)

    def get_active_escalations(self) -> List[Dict[str, Any]]:
        return [
            e.to_dict()
            for e in self._events.values()
            if e.state in (EscalationState.PENDING, EscalationState.ESCALATED)
        ]

    def get_all_events(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events.values()]

    def get_response_metrics(self) -> Dict[str, Any]:
        """Aggregate caregiver response metrics."""
        total = len(self._events)
        if total == 0:
            return {
                "total_events": 0,
                "acknowledged_count": 0,
                "mean_response_time_s": 0.0,
                "expired_count": 0,
                "resolved_count": 0,
                "pending_count": 0,
            }

        ack_times = [
            (e.acknowledged_at - e.start_time)
            for e in self._events.values()
            if e.acknowledged_at is not None
        ]
        mean_time = float(sum(ack_times) / len(ack_times)) if ack_times else 0.0

        return {
            "total_events": total,
            "acknowledged_count": len(ack_times),
            "mean_response_time_s": round(mean_time, 2),
            "expired_count": sum(1 for e in self._events.values() if e.state == EscalationState.EXPIRED),
            "resolved_count": sum(1 for e in self._events.values() if e.state == EscalationState.RESOLVED),
            "pending_count": sum(
                1 for e in self._events.values() if e.state in (EscalationState.PENDING, EscalationState.ESCALATED)
            ),
        }

    def _dispatch_tier_action(self, event: EscalationEvent, tier: EscalationTier) -> None:
        """Route escalation action to alert dispatcher if available."""
        if self.alert_dispatcher is not None:
            try:
                self.alert_dispatcher.dispatch(
                    modality="escalation",
                    event_type=f"fall_{tier.name.lower()}",
                    message=f"Escalation {tier.name} for Event {event.event_id} in Room {event.room_id}",
                    severity=event.severity,
                )
            except Exception as e:
                logger.warning(f"Error dispatching escalation alert: {e}")
