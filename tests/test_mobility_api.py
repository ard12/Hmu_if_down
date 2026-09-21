"""Tests for Mobility & Pre-Fall Risk REST API Endpoints (Milestone 14.4)."""

import time
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest
from hub.dashboard.app import app, broadcaster
from hub.gait_analyzer import GaitCadenceAnalyzer
from hub.patient_context import PatientContextStore
from hub.prefail_detector import PreFallAnomalyDetector


@pytest.fixture
def client():
    return TestClient(app)


def test_api_mobility_risk_returns_200(client):
    """GET /api/mobility/risk returns 200 with gait and pre_fall sections."""
    res = client.get("/api/mobility/risk")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "gait" in data
    assert "pre_fall" in data
    assert "gait_class" in data["gait"]
    assert "risk_score" in data["pre_fall"]


def test_api_mobility_risk_reflects_gait_analyzer_state(client):
    """GET /api/mobility/risk reflects injected gait analyzer state."""
    analyzer = GaitCadenceAnalyzer()
    analyzer.analyze = MagicMock(return_value={
        "cadence_hz": 2.1,
        "gait_class": "NORMAL",
        "stride_regularity": 0.92,
        "velocity_envelope_peak": 1.1,
        "confidence": 0.95,
    })
    broadcaster.gait_analyzer = analyzer

    res = client.get("/api/mobility/risk")
    assert res.status_code == 200
    data = res.json()
    assert data["gait"]["cadence_hz"] == 2.1
    assert data["gait"]["gait_class"] == "NORMAL"


def test_api_prefail_status_returns_200(client):
    """GET /api/prefail/status returns risk_score and risk_level."""
    detector = PreFallAnomalyDetector()
    broadcaster.prefail_detector = detector

    res = client.get("/api/prefail/status")
    assert res.status_code == 200
    data = res.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert "contributing_factors" in data


def test_api_prefail_status_with_patient_context(client):
    """GET /api/prefail/status?room_id=ROOM_101 includes patient context."""
    store = PatientContextStore()
    store.upsert("ROOM_101", {"patient_id": "P1", "morse_fall_scale": 60})
    broadcaster.patient_context_store = store

    detector = PreFallAnomalyDetector()
    broadcaster.prefail_detector = detector

    res = client.get("/api/prefail/status?room_id=ROOM_101")
    assert res.status_code == 200
    data = res.json()
    assert data["risk_score"] >= 10
    assert any("Morse" in factor for factor in data["contributing_factors"])


def test_api_prefail_history_returns_history_list(client):
    """GET /api/prefail/history returns list of snapshots."""
    detector = PreFallAnomalyDetector()
    detector.update({"gait_class": "NORMAL", "velocity_envelope_peak": 0.8}, timestamp=time.time())
    broadcaster.prefail_detector = detector

    res = client.get("/api/prefail/history")
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
    assert len(data["history"]) >= 1


def test_api_frax_unoccupied_room_returns_baseline(client):
    """GET /api/frax/ROOM_UNOCCUPIED returns baseline score for unoccupied room."""
    res = client.get("/api/frax/ROOM_UNOCCUPIED")
    assert res.status_code == 200
    data = res.json()
    assert data["room_id"] == "ROOM_UNOCCUPIED"
    assert data["patient_id"] == "UNOCCUPIED"
    assert "ten_year_fall_risk" in data
    assert data["risk_category"] in ("LOW", "MODERATE", "HIGH")


def test_api_frax_with_patient_context_calculates_high_risk(client):
    """GET /api/frax/ROOM_ELDERLY calculates high risk from patient context."""
    store = PatientContextStore()
    store.upsert("ROOM_ELDERLY", {
        "patient_id": "PAT_ELDERLY_01",
        "dob": "19400101",
        "gender": "F",
        "bmi": 18.2,
        "prior_fall": True,
        "morse_fall_scale": 65,
        "medications": ["warfarin", "oxycodone", "metoprolol"],
    })
    broadcaster.patient_context_store = store

    res = client.get("/api/frax/ROOM_ELDERLY")
    assert res.status_code == 200
    data = res.json()
    assert data["patient_id"] == "PAT_ELDERLY_01"
    assert data["ten_year_fall_risk"] > 0.50
    assert data["risk_category"] == "HIGH"
    assert len(data["primary_risk_factors"]) >= 2


def test_api_frax_returns_valid_probability_and_factors(client):
    """GET /api/frax/ROOM_STD returns valid probability in [0, 1] and risk factors."""
    res = client.get("/api/frax/ROOM_STD")
    assert res.status_code == 200
    data = res.json()
    prob = data["ten_year_fall_risk"]
    assert 0.0 <= prob <= 1.0
    assert isinstance(data["primary_risk_factors"], list)
