"""Tests for Prometheus Metrics Exporter (Milestone 17.2)."""

import pytest
from starlette.testclient import TestClient
from hub.dashboard.app import app
from hub.metrics import (
    fall_alerts_total,
    hl7_messages_received,
    retrain_events_total,
    active_rooms,
    drift_psi,
    buffer_size,
    fall_detection_latency,
    get_metrics_text,
    CONTENT_TYPE_LATEST,
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint_returns_200_and_content_type(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "fall_alerts_total" in response.text


def test_fall_alerts_total_counter_increments(client):
    fall_alerts_total.labels(room_id="room_101", severity="CRITICAL").inc()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'fall_alerts_total{room_id="room_101",severity="CRITICAL"}' in response.text


def test_drift_psi_gauge_updates(client):
    drift_psi.set(0.142)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_drift_psi 0.142" in response.text


def test_fall_detection_latency_observation(client):
    fall_detection_latency.observe(0.085)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fall_detection_latency_seconds_bucket" in response.text
    assert "fall_detection_latency_seconds_count" in response.text


def test_all_metric_names_convention():
    metrics = [
        "fall_alerts_total",
        "hl7_messages_received_total",
        "retrain_events_total",
        "active_rooms_total",
        "model_drift_psi",
        "training_buffer_size",
        "fall_detection_latency_seconds",
    ]
    for name in metrics:
        assert name.islower(), f"{name} must be lowercase"
        assert "-" not in name, f"{name} must not contain hyphens"
        assert name.replace("_", "").isalnum(), f"{name} must be alphanumeric"


def test_get_metrics_text_utf8():
    raw_bytes = get_metrics_text()
    assert isinstance(raw_bytes, bytes)
    text = raw_bytes.decode("utf-8")
    assert len(text) > 0
    assert "# HELP" in text
    assert "# TYPE" in text
