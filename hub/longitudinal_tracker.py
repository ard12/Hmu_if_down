"""
Longitudinal Health Intelligence Engine (v5.1.0).
Tracks multi-day and multi-week mobility trends: cadence, active minutes,
shuffle index, and FRAX risk trajectories. Emits early warning advisories
when week-over-week physical mobility degrades.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import datetime
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("longitudinal_tracker")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_mobility (
    patient_id     TEXT NOT NULL,
    record_date    TEXT NOT NULL,
    cadence_spm    REAL NOT NULL,
    active_minutes REAL NOT NULL,
    shuffle_index  REAL NOT NULL,
    frax_score     REAL NOT NULL,
    falls_count    INTEGER NOT NULL,
    PRIMARY KEY(patient_id, record_date)
);
"""


@dataclass
class DailyMobilityRecord:
    patient_id: str
    record_date: str  # YYYY-MM-DD
    cadence_spm: float
    active_minutes: float
    shuffle_index: float
    frax_score: float
    falls_count: int = 0


@dataclass
class MobilityAdvisory:
    patient_id: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    reason: str
    metric_name: str
    percentage_change: float
    timestamp: float = field(default_factory=time.time)


class LongitudinalTracker:
    """Computes rolling statistical trajectories and early mobility degradation advisories."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            self.db_path = (
                Path(__file__).resolve().parent.parent / "audits" / "longitudinal.db"
            )
        else:
            self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def record_day(self, record: DailyMobilityRecord) -> None:
        """Insert or replace daily aggregated mobility metrics for a patient."""
        safe_cadence = max(0.0, float(record.cadence_spm))
        safe_active = max(0.0, float(record.active_minutes))
        safe_shuffle = max(0.0, float(record.shuffle_index))
        safe_frax = max(0.0, float(record.frax_score))
        safe_falls = max(0, int(record.falls_count))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_mobility
                (patient_id, record_date, cadence_spm, active_minutes, shuffle_index, frax_score, falls_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.patient_id,
                    record.record_date,
                    safe_cadence,
                    safe_active,
                    safe_shuffle,
                    safe_frax,
                    safe_falls,
                ),
            )
            conn.commit()

    def get_patient_history(
        self, patient_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Retrieve historical daily records ordered chronologically."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM daily_mobility
                WHERE patient_id = ?
                ORDER BY record_date ASC
                LIMIT ?
                """,
                (patient_id, max(1, int(days))),
            ).fetchall()
            return [dict(r) for r in rows]

    def evaluate_mobility_trend(
        self, patient_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate week-over-week cadence degradation and generate family advisory status.
        """
        history = self.get_patient_history(patient_id, days=14)
        if len(history) < 2:
            return {
                "patient_id": patient_id,
                "status": "INSUFFICIENT_DATA",
                "status_label": "Baseline Establishing",
                "wow_cadence_change_pct": 0.0,
                "advisories": [],
                "days_tracked": len(history),
            }

        # Divide into previous week vs current week
        midpoint = max(1, len(history) // 2)
        prev_week = history[:midpoint]
        curr_week = history[midpoint:]

        if not prev_week or not curr_week:
            return {
                "patient_id": patient_id,
                "status": "INSUFFICIENT_DATA",
                "status_label": "Baseline Establishing",
                "wow_cadence_change_pct": 0.0,
                "advisories": [],
                "days_tracked": len(history),
            }

        prev_avg_cadence = sum(r["cadence_spm"] for r in prev_week) / len(prev_week)
        curr_avg_cadence = sum(r["cadence_spm"] for r in curr_week) / len(curr_week)

        if prev_avg_cadence > 0:
            pct_change = ((curr_avg_cadence - prev_avg_cadence) / prev_avg_cadence) * 100.0
        else:
            pct_change = 0.0

        advisories: List[Dict[str, Any]] = []
        if pct_change <= -15.0:
            status = "DECLINING"
            status_label = "Noticeable Mobility Decline"
            advisories.append(
                {
                    "severity": "WARNING",
                    "reason": f"Gait cadence declined by {abs(pct_change):.1f}% week-over-week.",
                    "metric_name": "cadence_spm",
                    "percentage_change": round(pct_change, 1),
                }
            )
        elif pct_change <= -5.0:
            status = "CAUTION"
            status_label = "Mild Activity Slowdown"
        else:
            status = "STEADY"
            status_label = "Stable & Active"

        latest = history[-1]
        return {
            "patient_id": patient_id,
            "status": status,
            "status_label": status_label,
            "current_cadence_spm": round(curr_avg_cadence, 1),
            "previous_cadence_spm": round(prev_avg_cadence, 1),
            "wow_cadence_change_pct": round(pct_change, 1),
            "latest_active_minutes": latest["active_minutes"],
            "latest_frax_score": latest["frax_score"],
            "advisories": advisories,
            "days_tracked": len(history),
        }


# Default singleton instance
longitudinal_tracker = LongitudinalTracker()
