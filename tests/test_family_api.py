"""
Unit Tests for Family Portal REST API (Phase 29, F-16).
Tests logging daily metrics, retrieving family summary digests, and multi-week trend trajectories.

@covers SRS-LNG-001, SRS-LNG-002
"""

import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_family_record_and_summary_lifecycle(client):
    """Verifies posting daily mobility record and reading back family summary."""
    # 1. Post daily record
    payload = {
        "patient_id": "PAT_FAMILY_TEST",
        "record_date": "2026-09-28",
        "cadence_spm": 92.5,
        "active_minutes": 45.0,
        "shuffle_index": 0.12,
        "frax_score": 5.4,
        "falls_count": 0,
    }
    resp = client.post("/api/family/record", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "RECORDED"
    assert data["patient_id"] == "PAT_FAMILY_TEST"

    # 2. Get summary
    resp_sum = client.get("/api/family/summary?patient_id=PAT_FAMILY_TEST")
    assert resp_sum.status_code == 200
    sum_data = resp_sum.json()
    assert sum_data["patient_id"] == "PAT_FAMILY_TEST"
    assert "mobility_status" in sum_data
    assert sum_data["today_active_minutes"] == 45.0
    assert sum_data["cadence_steps_per_min"] == 92.5


def test_family_trends_endpoint(client):
    """Verifies trends query returns historical data slice."""
    resp = client.get("/api/family/trends?patient_id=PAT_FAMILY_TEST&days=14")
    assert resp.status_code == 200
    trends = resp.json()
    assert trends["patient_id"] == "PAT_FAMILY_TEST"
    assert trends["days"] == 14
    assert isinstance(trends["data"], list)
    assert "trend_summary" in trends
