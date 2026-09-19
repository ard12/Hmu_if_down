"""Population Health Analytics and Trend Engine for Fall Audit Logs."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.audit_log import AuditLog


class FallAnalytics:
    """Aggregates and computes clinical telemetry and fall risk statistics from AuditLog."""

    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse ISO timestamp into UTC datetime."""
        try:
            # Handle ISO string with trailing Z or timezone offset
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    def fall_frequency(self, room_id: Optional[int] = None, days: int = 30) -> Dict[str, Any]:
        """Compute fall occurrences, daily averages, and per-room breakdown."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        events = self.audit_log.query(event_type="FALL_CONFIRMED", room_id=room_id, limit=10000)

        # Filter by date
        valid_events = []
        room_counter: Counter = Counter()
        for ev in events:
            dt = self._parse_timestamp(ev["timestamp_utc"])
            if dt and dt >= cutoff:
                valid_events.append(ev)
                room_counter[ev.get("room_id", 0)] += 1

        total_falls = len(valid_events)
        daily_avg = total_falls / max(days, 1)

        return {
            "total_falls": total_falls,
            "days_analyzed": days,
            "daily_average": round(daily_avg, 3),
            "by_room": dict(room_counter),
        }

    def hourly_distribution(self, room_id: Optional[int] = None, days: int = 30) -> List[Dict[str, int]]:
        """Return 24-bucket histogram of falls by hour of day (0-23)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        events = self.audit_log.query(event_type="FALL_CONFIRMED", room_id=room_id, limit=10000)

        hour_counts = [0] * 24
        for ev in events:
            dt = self._parse_timestamp(ev["timestamp_utc"])
            if dt and dt >= cutoff:
                hour_counts[dt.hour] += 1

        return [{"hour": h, "count": hour_counts[h]} for h in range(24)]

    def high_risk_windows(self, threshold: int = 2, window_days: int = 7) -> List[Dict[str, Any]]:
        """Identify hours of the day where fall frequency meets or exceeds threshold."""
        hourly = self.hourly_distribution(days=window_days)
        risk_hours = []
        for item in hourly:
            if item["count"] >= threshold:
                risk_hours.append({
                    "hour": item["hour"],
                    "fall_count": item["count"],
                    "risk_level": "CRITICAL" if item["count"] >= threshold * 2 else "HIGH",
                })
        return risk_hours

    def fall_type_distribution(self, room_id: Optional[int] = None) -> Dict[str, int]:
        """Extract and aggregate fall-type phenotypes from audit JSON payloads."""
        events = self.audit_log.query(event_type="FALL_CONFIRMED", room_id=room_id, limit=10000)
        type_counts: Counter = Counter()

        for ev in events:
            payload = ev.get("payload", {})
            ft = payload.get("fall_type") or payload.get("details", "")
            found_type = "unclassified"
            for candidate in ("forward_trip", "backward_slip", "lateral_collapse", "slow_slump", "syncope_drop"):
                if candidate in str(ft).lower():
                    found_type = candidate
                    break
            type_counts[found_type] += 1

        return dict(type_counts)

    def mean_time_between_falls(self, room_id: Optional[int] = None) -> Optional[float]:
        """Calculate mean time between falls (MTBF) in seconds."""
        events = self.audit_log.query(event_type="FALL_CONFIRMED", room_id=room_id, limit=10000)
        if len(events) < 2:
            return None

        # Sort chronologically by timestamp
        timestamps = []
        for ev in events:
            dt = self._parse_timestamp(ev["timestamp_utc"])
            if dt:
                timestamps.append(dt.timestamp())

        timestamps.sort()
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        if not intervals:
            return None
        return round(float(sum(intervals) / len(intervals)), 2)

    def cancellation_rate(self) -> float:
        """Calculate alert cancellation rate as a false-positive proxy."""
        confirmed = self.audit_log.query(event_type="FALL_CONFIRMED", limit=10000)
        cancelled = self.audit_log.query(event_type="FALL_CANCELLED", limit=10000)

        n_conf = len(confirmed)
        n_canc = len(cancelled)
        if n_conf == 0:
            return 0.0
        return round(float(n_canc / n_conf), 4)

    def full_report(self, room_id: Optional[int] = None, days: int = 30) -> Dict[str, Any]:
        """Assemble comprehensive population health and fall trend analytics."""
        freq = self.fall_frequency(room_id=room_id, days=days)
        return {
            "summary": freq,
            "hourly_distribution": self.hourly_distribution(room_id=room_id, days=days),
            "high_risk_windows": self.high_risk_windows(),
            "fall_types": self.fall_type_distribution(room_id=room_id),
            "mtbf_seconds": self.mean_time_between_falls(room_id=room_id),
            "cancellation_rate": self.cancellation_rate(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
