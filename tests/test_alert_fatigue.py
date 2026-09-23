"""Tests for Notification Fatigue Analytics and Caregiver UI.

@req SRS-UX-003
"""
from pathlib import Path
import pytest
from starlette.testclient import TestClient
from hub.analytics import FallAnalytics
from hub.audit_log import AuditLog
from hub.dashboard.app import app


@pytest.fixture
def analytics(tmp_path):
    db_path = tmp_path / "test_fatigue_audit.db"
    log = AuditLog(db_path=db_path)
    return FallAnalytics(log)


@pytest.fixture
def client():
    return TestClient(app)


def test_fatigue_score_zero_when_all_acknowledged(analytics):
    events = [
        {"event_id": "1", "state": "acknowledged", "acknowledged_at": 105.0, "start_time": 100.0, "response_time_s": 5.0},
        {"event_id": "2", "state": "acknowledged", "acknowledged_at": 210.0, "start_time": 200.0, "response_time_s": 10.0},
    ]
    score = analytics.alert_fatigue_score(events=events)
    assert score == 0.0


def test_fatigue_score_high_when_all_ignored(analytics):
    events = [
        {"event_id": "1", "state": "expired", "acknowledged_at": None, "start_time": 100.0},
        {"event_id": "2", "state": "pending", "acknowledged_at": None, "start_time": 200.0},
    ]
    score = analytics.alert_fatigue_score(events=events)
    assert score == 1.0


def test_response_distribution_has_percentile_keys(analytics):
    events = [
        {"event_id": str(i), "response_time_s": float(i * 10)}
        for i in range(1, 11)
    ]
    dist = analytics.response_time_distribution(events=events)
    assert "p25_s" in dist
    assert "p50_s" in dist
    assert "p75_s" in dist
    assert "p95_s" in dist
    assert "mean_s" in dist
    assert dist["sample_count"] == 10
    assert dist["p50_s"] > 0


def test_time_of_day_fatigue_has_24_buckets(analytics):
    events = [
        {"event_id": "1", "state": "expired", "start_time": 1700000000.0, "acknowledged_at": None}
    ]
    buckets = analytics.time_of_day_fatigue(days=3650, events=events)
    assert len(buckets) == 24
    for b in buckets:
        assert "hour" in b
        assert "unacknowledged_count" in b


def test_api_fatigue_endpoint_returns_200(client):
    response = client.get("/api/caregiver/fatigue")
    assert response.status_code == 200
    data = response.json()
    assert "fatigue_score" in data
    assert "response_time_distribution" in data
    assert "time_of_day_fatigue" in data
    assert len(data["time_of_day_fatigue"]) == 24


def test_caregiver_html_served(client):
    response = client.get("/caregiver")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Caregiver Alert Triage" in response.text
