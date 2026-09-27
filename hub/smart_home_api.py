"""
Smart Home Automation REST API — Phase 25 (v4.9.0).
Provides endpoints to configure automation rules, manually test emergency
chains (dry-run mode), and inspect execution logs.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from hub.smart_home_actions import (
    smart_home_engine,
    AutomationRule,
    AutomationActionType,
)

router = APIRouter(prefix="/api/automations", tags=["automations"])


class TriggerChainRequest(BaseModel):
    event_id: str
    room_id: Optional[str] = "room_default"
    dry_run: Optional[bool] = False


class RuleUpdateRequest(BaseModel):
    rule_id: str
    action_type: str
    target_entity: str
    enabled: Optional[bool] = True
    parameters: Optional[Dict[str, Any]] = None


@router.post("/trigger")
def trigger_automation_chain(req: TriggerChainRequest):
    """Trigger the environmental safety chain for a fall event."""
    results = smart_home_engine.trigger_emergency_chain(
        event_id=req.event_id,
        room_id=req.room_id or "room_default",
        dry_run=bool(req.dry_run),
    )
    return {
        "status": "COMPLETED",
        "dry_run": req.dry_run,
        "event_id": req.event_id,
        "actions_executed": len(results),
        "results": results,
    }


@router.get("/rules")
def get_rules():
    """Retrieve all configured smart home emergency rules."""
    return {
        "rules": [
            {
                "rule_id": r.rule_id,
                "action_type": r.action_type.value,
                "target_entity": r.target_entity,
                "enabled": r.enabled,
                "parameters": r.parameters,
            }
            for r in smart_home_engine.rules.values()
        ]
    }


@router.post("/rules/{rule_id}/toggle")
def toggle_rule(rule_id: str, enabled: bool):
    """Enable or disable a specific automation rule."""
    success = smart_home_engine.set_rule_enabled(rule_id, enabled)
    if not success:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return {"status": "UPDATED", "rule_id": rule_id, "enabled": enabled}


@router.get("/history")
def get_automation_history(limit: Optional[int] = 50):
    """Retrieve execution log history of peripheral smart actions."""
    return {"history": smart_home_engine.get_history(limit=limit or 50)}
