"""Tests for Kubernetes readiness and liveness self-tests (Milestone 17.4)."""

import time
import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app, broadcaster


@pytest.fixture
def client():
    broadcaster.is_ready = True
    with TestClient(app) as test_client:
        yield test_client
    broadcaster.is_ready = True


def test_health_returns_200_when_ready(client):
    """GET /health returns 200 with status ready during normal operation."""
    broadcaster.is_ready = True
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}


def test_health_returns_503_when_not_ready(client):
    """GET /health returns 503 during startup or unready state."""
    broadcaster.is_ready = False
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json() == {"status": "not_ready"}


def test_diagnostics_health_returns_200_normally(client):
    """GET /api/diagnostics/health returns 200 and healthy status."""
    res = client.get("/api/diagnostics/health")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "healthy"
    assert "self_test" in data


def test_diagnostics_health_returns_503_on_self_test_fail(client, monkeypatch):
    """GET /api/diagnostics/health returns 503 when self-test fails, pulling pod from rotation."""
    class BrokenWatcher:
        def run_self_test(self):
            return {
                "self_test_passed": False,
                "checks": {"critical_sensor": False},
                "iec_60601_compliance": "FAIL",
            }
        def evaluate_health(self):
            return {"status": "FAULT"}

    monkeypatch.setattr(broadcaster, "diagnostics_watcher", BrokenWatcher())
    res = client.get("/api/diagnostics/health")
    assert res.status_code == 503
    data = res.json()
    assert data.get("iec_60601_compliance") == "FAIL"


def test_k8s_probe_response_time_under_200ms(client):
    """Probes must respond in under 200ms to avoid Kubernetes timeout."""
    start = time.perf_counter()
    res1 = client.get("/health")
    elapsed1 = (time.perf_counter() - start) * 1000.0
    assert res1.status_code == 200
    assert elapsed1 < 200.0, f"Health probe took {elapsed1:.1f}ms > 200ms"

    start = time.perf_counter()
    res2 = client.get("/api/diagnostics/health")
    elapsed2 = (time.perf_counter() - start) * 1000.0
    assert res2.status_code == 200
    assert elapsed2 < 200.0, f"Diagnostics probe took {elapsed2:.1f}ms > 200ms"


def test_k8s_probe_responses_are_valid_json(client):
    """All probe endpoints must return valid JSON with application/json header."""
    res = client.get("/health")
    assert "application/json" in res.headers.get("content-type", "")
    assert isinstance(res.json(), dict)

    res2 = client.get("/api/diagnostics/health")
    assert "application/json" in res2.headers.get("content-type", "")
    assert isinstance(res2.json(), dict)
