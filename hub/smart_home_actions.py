"""
Smart Home Automation & Environmental Safety Engine (v4.9.0).
Coordinates physical environment response upon confirmed fall events:
illumination, smart lock release for first responders, robot vacuum halts,
and ambient temperature stabilization to prevent hypothermia.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("smart_home_actions")


class AutomationActionType(str, Enum):
    LIGHTS_ON = "LIGHTS_ON"
    UNLOCK_DOORS = "UNLOCK_DOORS"
    PAUSE_VACUUM = "PAUSE_VACUUM"
    ADJUST_THERMOSTAT = "ADJUST_THERMOSTAT"
    SOUND_LOCAL_CHIME = "SOUND_LOCAL_CHIME"


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    DRY_RUN = "DRY_RUN"


@dataclass
class ActionRecord:
    action_type: AutomationActionType
    target_entity: str
    status: ActionStatus
    timestamp: float
    retries: int = 0
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationRule:
    rule_id: str
    action_type: AutomationActionType
    target_entity: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class SmartHomeActionEngine:
    """Dispatches safety automations to smart home peripherals."""

    DEFAULT_RULES = [
        AutomationRule(
            rule_id="rule_lights_emergency",
            action_type=AutomationActionType.LIGHTS_ON,
            target_entity="light.patient_room_all",
            enabled=True,
            parameters={"brightness": 255, "transition_sec": 1.0},
        ),
        AutomationRule(
            rule_id="rule_unlock_entrance",
            action_type=AutomationActionType.UNLOCK_DOORS,
            target_entity="lock.front_door",
            enabled=True,
            parameters={"auto_relock_min": 60},
        ),
        AutomationRule(
            rule_id="rule_pause_vacuum",
            action_type=AutomationActionType.PAUSE_VACUUM,
            target_entity="vacuum.floor_cleaner",
            enabled=True,
            parameters={},
        ),
        AutomationRule(
            rule_id="rule_thermostat_comfort",
            action_type=AutomationActionType.ADJUST_THERMOSTAT,
            target_entity="climate.home_thermostat",
            enabled=True,
            parameters={"target_temperature_c": 22.5},
        ),
    ]

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.rules: Dict[str, AutomationRule] = {
            r.rule_id: r for r in self.DEFAULT_RULES
        }
        self.action_history: List[ActionRecord] = []
        self._custom_executor: Optional[Callable[[ActionRecord], bool]] = None

    def set_custom_executor(self, executor: Callable[[ActionRecord], bool]) -> None:
        """Allow custom MQTT or Home Assistant client executor to be injected."""
        self._custom_executor = executor

    def add_rule(self, rule: AutomationRule) -> None:
        """Add or update an automation rule."""
        self.rules[rule.rule_id] = rule

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """Toggle an automation rule on or off."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = enabled
            return True
        return False

    def _execute_single_action(
        self, action_type: AutomationActionType, target: str, params: Dict[str, Any], dry_run: bool
    ) -> ActionRecord:
        """Execute a single peripheral automation with retries."""
        now = time.time()
        record = ActionRecord(
            action_type=action_type,
            target_entity=target,
            status=ActionStatus.DRY_RUN if dry_run else ActionStatus.PENDING,
            timestamp=now,
            parameters=params,
        )

        if dry_run:
            logger.info(f"[DRY_RUN] Dispatched {action_type.value} to {target} (params={params})")
            self.action_history.append(record)
            return record

        for attempt in range(self.max_retries + 1):
            record.retries = attempt
            try:
                if self._custom_executor:
                    success = self._custom_executor(record)
                else:
                    # Built-in reliable simulation executor
                    success = True

                if success:
                    record.status = ActionStatus.SUCCESS
                    logger.info(f"Automation executed: {action_type.value} -> {target}")
                    break
                else:
                    raise RuntimeError("Custom executor returned False")
            except Exception as e:
                record.error_message = str(e)
                logger.warning(
                    f"Automation attempt {attempt + 1}/{self.max_retries + 1} failed for {target}: {e}"
                )
                if attempt == self.max_retries:
                    record.status = ActionStatus.FAILED

        self.action_history.append(record)
        return record

    def trigger_emergency_chain(
        self, event_id: str, room_id: str = "room_default", dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute the full environmental safety chain for a confirmed fall event.
        Returns list of executed action summaries.
        """
        results = []
        for rule in self.rules.values():
            if not rule.enabled:
                rec = ActionRecord(
                    action_type=rule.action_type,
                    target_entity=rule.target_entity,
                    status=ActionStatus.SKIPPED,
                    timestamp=time.time(),
                    parameters=rule.parameters,
                )
                self.action_history.append(rec)
                results.append(
                    {
                        "rule_id": rule.rule_id,
                        "action": rule.action_type.value,
                        "entity": rule.target_entity,
                        "status": ActionStatus.SKIPPED.value,
                    }
                )
                continue

            rec = self._execute_single_action(
                rule.action_type, rule.target_entity, rule.parameters, dry_run=dry_run
            )
            results.append(
                {
                    "rule_id": rule.rule_id,
                    "action": rule.action_type.value,
                    "entity": rule.target_entity,
                    "status": rec.status.value,
                    "retries": rec.retries,
                    "error": rec.error_message,
                }
            )

        logger.info(
            f"Emergency automation chain completed for event={event_id}, room={room_id} ({len(results)} actions)"
        )
        return results

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent action dispatch logs."""
        return [
            {
                "action": r.action_type.value,
                "target": r.target_entity,
                "status": r.status.value,
                "timestamp": r.timestamp,
                "retries": r.retries,
                "error": r.error_message,
                "params": r.parameters,
            }
            for r in self.action_history[-limit:]
        ]


# Default singleton instance
smart_home_engine = SmartHomeActionEngine()
