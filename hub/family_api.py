"""
Family Portal REST API — Phase 27 (v5.1.0).
Provides non-clinical, family-friendly activity digests, mobility trend charts,
and proactive early warning alerts regarding loved ones' physical stability.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import datetime

from hub.longitudinal_tracker import (
    longitudinal_tracker,
    DailyMobilityRecord,
)

router = APIRouter(prefix="/api/family", tags=["family"])


class DailyRecordInput(BaseModel):
    patient_id: str
    record_date: Optional[str] = None
    cadence_spm: float
    active_minutes: float
    shuffle_index: float
    frax_score: float
    falls_count: Optional[int] = 0


@router.post("/record")
def record_daily_metrics(req: DailyRecordInput):
    """Log or sync daily mobility metrics for a patient."""
    date_str = req.record_date or datetime.date.today().isoformat()
    record = DailyMobilityRecord(
        patient_id=req.patient_id,
        record_date=date_str,
        cadence_spm=req.cadence_spm,
        active_minutes=req.active_minutes,
        shuffle_index=req.shuffle_index,
        frax_score=req.frax_score,
        falls_count=req.falls_count or 0,
    )
    longitudinal_tracker.record_day(record)
    return {"status": "RECORDED", "patient_id": req.patient_id, "date": date_str}


@router.get("/summary")
def get_family_summary(patient_id: str = "PAT_DEFAULT"):
    """Retrieve family-friendly day-at-a-glance mobility digest."""
    trend = longitudinal_tracker.evaluate_mobility_trend(patient_id)
    history = longitudinal_tracker.get_patient_history(patient_id, days=7)
    latest = history[-1] if history else None

    return {
        "patient_id": patient_id,
        "mobility_status": trend.get("status", "STEADY"),
        "mobility_label": trend.get("status_label", "Stable & Active"),
        "today_active_minutes": latest["active_minutes"] if latest else 0,
        "cadence_steps_per_min": latest["cadence_spm"] if latest else 0,
        "wow_change_pct": trend.get("wow_cadence_change_pct", 0.0),
        "advisories": trend.get("advisories", []),
        "recent_days": history,
    }


@router.get("/trends")
def get_trends(patient_id: str = "PAT_DEFAULT", days: int = 30):
    """Retrieve multi-week trend trajectory data for charting."""
    history = longitudinal_tracker.get_patient_history(patient_id, days=days)
    trend = longitudinal_tracker.evaluate_mobility_trend(patient_id)
    return {
        "patient_id": patient_id,
        "days": days,
        "data": history,
        "trend_summary": trend,
    }
