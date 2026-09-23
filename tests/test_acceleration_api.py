"""Tests for Edge AI Acceleration and Latency REST API (Milestone 19.4)."""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.dashboard.app import app, broadcaster
from hub.tensorrt_runner import TensorRTRunner
from hub.offload_manager import OffloadManager


@pytest.fixture
def client():
    broadcaster.tensorrt_runner = TensorRTRunner()
    broadcaster.offload_manager = OffloadManager()
    with TestClient(app) as test_client:
        yield test_client


def test_acceleration_page_returns_200(client):
    response = client.get("/acceleration")
    assert response.status_code == 200
    assert "Edge AI Acceleration & Latency HUD" in response.text
    assert "Active Execution Provider" in response.text


def test_acceleration_static_file_served(client):
    response = client.get("/static/acceleration.html")
    assert response.status_code == 200
    assert "Edge AI Acceleration" in response.text


def test_acceleration_status_endpoint_returns_200(client):
    response = client.get("/api/acceleration/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "active_provider" in data
    assert "is_tensorrt" in data
    assert "provider_health" in data
    assert "latency_stats" in data
    assert "p50_ms" in data["latency_stats"]
    assert "total_inferences" in data["latency_stats"]


def test_acceleration_benchmark_endpoint(client):
    response = client.post("/api/acceleration/benchmark", json={"n": 15})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["n_runs"] == 15
    assert "stats" in data
    assert data["stats"]["total_inferences"] >= 15
    assert data["stats"]["mean_ms"] >= 0.0


def test_acceleration_sla_endpoint(client):
    # Run benchmark first to populate latencies
    client.post("/api/acceleration/benchmark", json={"n": 10})
    response = client.get("/api/acceleration/sla")
    assert response.status_code == 200
    data = response.json()
    assert "sla_target_ms" in data
    assert data["sla_target_ms"] == 50.0
    assert "compliance_percentage" in data
    assert 0.0 <= data["compliance_percentage"] <= 100.0
    assert data["total_evaluated"] >= 10
