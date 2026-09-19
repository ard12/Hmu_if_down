"""Unit tests for the FallAnalytics population health module and REST endpoints (Milestone 8.4)."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.analytics import FallAnalytics
from hub.audit_log import AuditLog
from hub.dashboard.app import app, broadcaster


@pytest.fixture
def temp_audit_log(tmp_path):
    db_path = tmp_path / "test_audit.db"
    return AuditLog(db_path=db_path)


def test_fall_frequency_returns_zero_for_empty_log(temp_audit_log):
    """Verify fall_frequency returns 0 for an empty audit log."""
    analytics = FallAnalytics(audit_log=temp_audit_log)
    freq = analytics.fall_frequency(days=30)
    assert freq["total_falls"] == 0
    assert freq["daily_average"] == 0.0
    assert freq["by_room"] == {}


def test_fall_frequency_counts_correctly(temp_audit_log):
    """Verify fall_frequency counts multiple confirmed falls and breaks down by room."""
    analytics = FallAnalytics(audit_log=temp_audit_log)
    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={"details": "CSI burst"})
    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={"details": "Radar floor height"})
    temp_audit_log.append("FALL_CONFIRMED", room_id=2, payload={"details": "Slump detected"})

    freq = analytics.fall_frequency(days=30)
    assert freq["total_falls"] == 3
    assert freq["by_room"][1] == 2
    assert freq["by_room"][2] == 1


def test_hourly_distribution_has_24_buckets(temp_audit_log):
    """Verify hourly_distribution returns a 24-element list covering hours 0 to 23."""
    analytics = FallAnalytics(audit_log=temp_audit_log)
    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={"details": "test"})

    dist = analytics.hourly_distribution(days=30)
    assert len(dist) == 24
    assert [d["hour"] for d in dist] == list(range(24))
    total_in_dist = sum(d["count"] for d in dist)
    assert total_in_dist == 1


def test_cancellation_rate_formula(temp_audit_log):
    """Verify cancellation_rate = cancelled / confirmed."""
    analytics = FallAnalytics(audit_log=temp_audit_log)
    assert analytics.cancellation_rate() == 0.0

    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={})
    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={})
    temp_audit_log.append("FALL_CANCELLED", room_id=1, payload={})

    rate = analytics.cancellation_rate()
    # 1 cancelled / 2 confirmed = 0.5
    assert rate == 0.5


def test_mean_time_between_falls_single_fall(temp_audit_log):
    """Verify MTBF returns None when fewer than 2 falls exist."""
    analytics = FallAnalytics(audit_log=temp_audit_log)
    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={})
    assert analytics.mean_time_between_falls() is None


def test_mean_time_between_falls_two_falls(temp_audit_log):
    """Verify MTBF calculates interval between multiple falls."""
    analytics = FallAnalytics(audit_log=temp_audit_log)
    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(hours=2)).isoformat()
    t2 = now.isoformat()

    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={}, timestamp_utc=t1)
    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={}, timestamp_utc=t2)

    mtbf = analytics.mean_time_between_falls()
    assert mtbf is not None
    # ~2 hours = 7200 seconds
    assert 7100 <= mtbf <= 7300


def test_api_analytics_summary_endpoint(temp_audit_log):
    """Verify GET /api/analytics/summary returns 200 with full analytics report."""
    client = TestClient(app)
    broadcaster.audit_log = temp_audit_log
    broadcaster.analytics = None  # Force fresh analytics instance

    temp_audit_log.append("FALL_CONFIRMED", room_id=1, payload={"fall_type": "forward_trip"})

    try:
        resp = client.get("/api/analytics/summary?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert data["summary"]["total_falls"] == 1
        assert "hourly_distribution" in data
        assert "fall_types" in data
        assert data["fall_types"].get("forward_trip") == 1
    finally:
        broadcaster.audit_log = None
        broadcaster.analytics = None
