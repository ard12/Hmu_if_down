"""Tests for GET /api/version and GET /firmware/<filename> endpoints (Milestone 6.5)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from hub.dashboard.app import app

client = TestClient(app)


def test_version_endpoint_returns_version_string():
    """GET /api/version should return a version string and phase number."""
    resp = client.get("/api/version")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert "phase" in data
    assert data["service"] == "falldetect-hub"
    # Version must be a non-empty string (e.g. "3.0.0")
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


def test_version_endpoint_phase_is_integer():
    """Phase field must be a non-negative integer."""
    resp = client.get("/api/version")
    data = resp.json()
    assert isinstance(data["phase"], int)
    assert data["phase"] >= 0


def test_firmware_endpoint_rejects_directory_traversal():
    """GET /firmware/../secrets.txt must be rejected with 400."""
    resp = client.get("/firmware/..%2Fsecrets.txt")
    # FastAPI URL decodes the path; the handler should return 400 or 404
    assert resp.status_code in (400, 404)


def test_firmware_endpoint_returns_404_for_missing_file():
    """GET /firmware/nonexistent.bin should return 404."""
    resp = client.get("/firmware/nonexistent.bin")
    assert resp.status_code == 404
    assert "not found" in resp.json().get("error", "").lower()
