"""
Unit and integration tests for Longitudinal Health Intelligence & Family Dashboard (Phase 27).
"""

import pytest
from fastapi.testclient import TestClient
from hub.longitudinal_tracker import (
    LongitudinalTracker,
    DailyMobilityRecord,
)
from hub.dashboard.app import app


@pytest.fixture
def tracker(tmp_path):
    db_file = tmp_path / "test_longitudinal.db"
    return LongitudinalTracker(db_path=db_file)


def test_daily_mobility_recording(tracker):
    """
    Covers: SRS-LNG-001
    Verifies daily recording of cadence, active minutes, and frax scores.
    """
    rec = DailyMobilityRecord(
        patient_id="PAT_007",
        record_date="2026-09-01",
        cadence_spm=88.5,
        active_minutes=145.0,
        shuffle_index=0.12,
        frax_score=14.2,
        falls_count=0,
    )
    tracker.record_day(rec)

    hist = tracker.get_patient_history("PAT_007")
    assert len(hist) == 1
    assert hist[0]["cadence_spm"] == 88.5
    assert hist[0]["record_date"] == "2026-09-01"


def test_week_over_week_mobility_decline_detection(tracker):
    """
    Covers: SRS-LNG-001, HAZ-036
    Verifies alert generation when week-over-week cadence drops >15%.
    """
    # Week 1: Healthy baseline cadence (~90 spm)
    for i in range(1, 8):
        tracker.record_day(
            DailyMobilityRecord(
                patient_id="PAT_008",
                record_date=f"2026-09-{i:02d}",
                cadence_spm=90.0,
                active_minutes=120.0,
                shuffle_index=0.10,
                frax_score=12.0,
            )
        )

    # Week 2: Frailty / sedation decline (~70 spm, a ~22% decline)
    for i in range(8, 15):
        tracker.record_day(
            DailyMobilityRecord(
                patient_id="PAT_008",
                record_date=f"2026-09-{i:02d}",
                cadence_spm=70.0,
                active_minutes=75.0,
                shuffle_index=0.35,
                frax_score=18.5,
            )
        )

    trend = tracker.evaluate_mobility_trend("PAT_008")
    assert trend["status"] == "DECLINING"
    assert trend["wow_cadence_change_pct"] <= -15.0
    assert len(trend["advisories"]) >= 1
    assert "declined" in trend["advisories"][0]["reason"].lower()


def test_stable_mobility_status(tracker):
    """
    Covers: SRS-LNG-001
    Verifies steady status when mobility metrics remain consistent.
    """
    for i in range(1, 15):
        tracker.record_day(
            DailyMobilityRecord(
                patient_id="PAT_009",
                record_date=f"2026-09-{i:02d}",
                cadence_spm=85.0 + (i % 2),
                active_minutes=100.0,
                shuffle_index=0.15,
                frax_score=11.0,
            )
        )

    trend = tracker.evaluate_mobility_trend("PAT_009")
    assert trend["status"] == "STEADY"
    assert len(trend["advisories"]) == 0


def test_family_api_endpoints():
    """
    Covers: SRS-LNG-002
    Verifies Family Portal REST endpoints for summary, trend, and record sync.
    """
    client = TestClient(app)

    # 1. Sync daily record
    resp_sync = client.post(
        "/api/family/record",
        json={
            "patient_id": "PAT_FAMILY_TEST",
            "record_date": "2026-09-20",
            "cadence_spm": 82.0,
            "active_minutes": 110.0,
            "shuffle_index": 0.14,
            "frax_score": 10.5,
        },
    )
    assert resp_sync.status_code == 200
    assert resp_sync.json()["status"] == "RECORDED"

    # 2. Get family summary
    resp_sum = client.get("/api/family/summary?patient_id=PAT_FAMILY_TEST")
    assert resp_sum.status_code == 200
    data = resp_sum.json()
    assert "mobility_status" in data
    assert "today_active_minutes" in data

    # 3. Get trends
    resp_trends = client.get("/api/family/trends?patient_id=PAT_FAMILY_TEST&days=7")
    assert resp_trends.status_code == 200
    assert "trend_summary" in resp_trends.json()
