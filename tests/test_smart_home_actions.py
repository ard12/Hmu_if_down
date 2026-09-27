"""
Unit and integration tests for Smart Home Automations & Environmental Safety (Phase 25).
"""

import pytest
from fastapi.testclient import TestClient
from hub.smart_home_actions import (
    SmartHomeActionEngine,
    AutomationRule,
    AutomationActionType,
    ActionStatus,
)
from hub.dashboard.app import app


@pytest.fixture
def engine():
    return SmartHomeActionEngine(max_retries=2)


def test_emergency_chain_execution(engine):
    """
    Covers: SRS-AUT-001
    Verifies that lighting, unlocking, vacuum pause, and thermostat actions fire upon a fall event.
    """
    results = engine.trigger_emergency_chain(event_id="evt_fall_001", room_id="room_east")
    assert len(results) >= 4

    action_names = [r["action"] for r in results]
    assert "LIGHTS_ON" in action_names
    assert "UNLOCK_DOORS" in action_names
    assert "PAUSE_VACUUM" in action_names
    assert "ADJUST_THERMOSTAT" in action_names

    # Ensure all default enabled rules succeeded
    for res in results:
        assert res["status"] == ActionStatus.SUCCESS.value


def test_dry_run_safety_mode(engine):
    """
    Covers: SRS-AUT-002
    Verifies dry-run execution records actions without physically commanding peripherals.
    """
    results = engine.trigger_emergency_chain(event_id="evt_test_dry", dry_run=True)
    for res in results:
        assert res["status"] == ActionStatus.DRY_RUN.value

    history = engine.get_history(limit=10)
    assert any(h["status"] == "DRY_RUN" for h in history)


def test_action_retry_and_resilience(engine):
    """
    Covers: HAZ-034
    Verifies retry handling when a smart lock peripheral is intermittently unresponsive.
    """
    call_counts = {"count": 0}

    def flaky_executor(record):
        call_counts["count"] += 1
        # Succeed only on the 2nd attempt
        if call_counts["count"] < 2:
            raise ConnectionError("Zigbee mesh timeout")
        return True

    engine.set_custom_executor(flaky_executor)
    # Target single action
    record = engine._execute_single_action(
        AutomationActionType.UNLOCK_DOORS,
        "lock.back_door",
        {},
        dry_run=False,
    )
    assert record.status == ActionStatus.SUCCESS
    assert record.retries == 1  # Succeeded on retry 1


def test_rule_toggle_and_skip(engine):
    """
    Covers: SRS-AUT-002
    Verifies disabling a rule skips its execution during emergency chain.
    """
    engine.set_rule_enabled("rule_pause_vacuum", False)
    results = engine.trigger_emergency_chain(event_id="evt_skip_test")

    vacuum_result = next(r for r in results if r["action"] == "PAUSE_VACUUM")
    assert vacuum_result["status"] == ActionStatus.SKIPPED.value


def test_smart_home_api_endpoints():
    """
    Covers: SRS-AUT-001, SRS-AUT-002
    Verifies REST API endpoints for trigger, rules query, and history inspection.
    """
    client = TestClient(app)

    # 1. Trigger chain via API
    resp = client.post(
        "/api/automations/trigger",
        json={"event_id": "api_test_01", "dry_run": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"
    assert data["dry_run"] is True

    # 2. Get rules
    resp_rules = client.get("/api/automations/rules")
    assert resp_rules.status_code == 200
    assert len(resp_rules.json()["rules"]) > 0

    # 3. Toggle rule
    resp_toggle = client.post("/api/automations/rules/rule_lights_emergency/toggle?enabled=false")
    assert resp_toggle.status_code == 200
    assert resp_toggle.json()["enabled"] is False

    # 4. Get history
    resp_hist = client.get("/api/automations/history")
    assert resp_hist.status_code == 200
    assert "history" in resp_hist.json()
