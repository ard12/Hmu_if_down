"""Tests for Progressive Notification Escalation Engine.

@req SRS-UX-001
@req SRS-UX-002
"""
import pytest
from hub.notification_escalator import (
    EscalationEvent,
    EscalationPolicy,
    EscalationState,
    EscalationTier,
    NotificationEscalator,
)


def test_initial_tier_is_silent_push():
    escalator = NotificationEscalator()
    event = escalator.start_escalation(event_id="EVT-001", severity="critical", room_id=1, timestamp=100.0)
    assert event.current_tier == EscalationTier.TIER_1_SILENT_PUSH
    assert event.state == EscalationState.PENDING
    assert event.start_time == 100.0


def test_escalation_advances_after_timeout():
    escalator = NotificationEscalator()
    escalator.start_escalation(event_id="EVT-002", timestamp=100.0)

    # 10s later -> still Tier 1
    adv = escalator.advance_time(110.0)
    assert len(adv) == 0
    evt = escalator.get_event("EVT-002")
    assert evt.current_tier == EscalationTier.TIER_1_SILENT_PUSH

    # 35s later -> advances to Tier 2 (30s timeout)
    adv = escalator.advance_time(135.0)
    assert len(adv) == 1
    assert evt.current_tier == EscalationTier.TIER_2_AUDIBLE_ALERT
    assert evt.state == EscalationState.ESCALATED


def test_acknowledge_halts_escalation():
    escalator = NotificationEscalator()
    escalator.start_escalation(event_id="EVT-003", timestamp=100.0)

    # Caregiver acknowledges at 115.0s
    success = escalator.acknowledge(event_id="EVT-003", caregiver_id="nurse_smith", timestamp=115.0)
    assert success is True
    evt = escalator.get_event("EVT-003")
    assert evt.state == EscalationState.ACKNOWLEDGED
    assert evt.acknowledged_by == "nurse_smith"
    assert evt.acknowledged_at == 115.0

    # Advancing time to 500s does NOT advance tier
    adv = escalator.advance_time(500.0)
    assert len(adv) == 0
    assert evt.current_tier == EscalationTier.TIER_1_SILENT_PUSH


def test_recovery_resolves_escalation():
    escalator = NotificationEscalator()
    escalator.start_escalation(event_id="EVT-004", timestamp=100.0)
    res = escalator.resolve_recovery(event_id="EVT-004", timestamp=120.0)
    assert res is True
    evt = escalator.get_event("EVT-004")
    assert evt.state == EscalationState.RESOLVED


def test_response_metrics_track_mean_time():
    escalator = NotificationEscalator()
    escalator.start_escalation("EVT-A", timestamp=100.0)
    escalator.acknowledge("EVT-A", "caregiver_1", timestamp=110.0)  # 10s

    escalator.start_escalation("EVT-B", timestamp=200.0)
    escalator.acknowledge("EVT-B", "caregiver_2", timestamp=230.0)  # 30s

    metrics = escalator.get_response_metrics()
    assert metrics["total_events"] == 2
    assert metrics["acknowledged_count"] == 2
    assert metrics["mean_response_time_s"] == 20.0  # (10 + 30)/2


def test_custom_policy_timeouts():
    custom_policy = EscalationPolicy(
        tiers=[EscalationTier.TIER_1_SILENT_PUSH, EscalationTier.TIER_5_EMERGENCY_DISPATCH],
        timeout_per_tier_s=[5.0],
    )
    escalator = NotificationEscalator(policy=custom_policy)
    escalator.start_escalation("EVT-CUSTOM", timestamp=10.0)

    escalator.advance_time(16.0)
    evt = escalator.get_event("EVT-CUSTOM")
    assert evt.current_tier == EscalationTier.TIER_5_EMERGENCY_DISPATCH


def test_max_tier_reached_marks_expired():
    custom_policy = EscalationPolicy(
        tiers=[EscalationTier.TIER_1_SILENT_PUSH, EscalationTier.TIER_2_AUDIBLE_ALERT],
        timeout_per_tier_s=[10.0],
    )
    escalator = NotificationEscalator(policy=custom_policy)
    escalator.start_escalation("EVT-EXP", timestamp=0.0)

    # 15s -> advances to TIER_2
    escalator.advance_time(15.0)
    assert escalator.get_event("EVT-EXP").current_tier == EscalationTier.TIER_2_AUDIBLE_ALERT

    # 80s -> exceeds final tier limit -> EXPIRED
    escalator.advance_time(85.0)
    assert escalator.get_event("EVT-EXP").state == EscalationState.EXPIRED


def test_ack_unknown_event_returns_false():
    escalator = NotificationEscalator()
    assert escalator.acknowledge("NONEXISTENT", "caregiver_1") is False
