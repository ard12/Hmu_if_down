"""Tests for AuditLog, FHIRExporter, and audit REST endpoints (Milestone 6.3)."""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.audit_log import AuditLog, EVENT_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_audit(tmp_path):
    """AuditLog backed by a temporary SQLite database."""
    return AuditLog(db_path=tmp_path / "test_audit.db")


# ---------------------------------------------------------------------------
# AuditLog unit tests
# ---------------------------------------------------------------------------

def test_audit_log_append_creates_row(tmp_audit):
    """append() must create exactly one row in the database."""
    row_id = tmp_audit.append("FALL_CONFIRMED", {"modality": "fusion", "height": 0.32})
    assert isinstance(row_id, int)
    assert row_id >= 1
    assert tmp_audit.count() == 1


def test_audit_log_multiple_appends(tmp_audit):
    """Sequential appends produce incrementing row IDs."""
    id1 = tmp_audit.append("SYSTEM_START", {})
    id2 = tmp_audit.append("FALL_CONFIRMED", {"modality": "csi"})
    id3 = tmp_audit.append("CALIBRATE", {"room_id": 1})
    assert id2 > id1
    assert id3 > id2
    assert tmp_audit.count() == 3


def test_audit_log_invalid_event_type_raises(tmp_audit):
    """append() must raise ValueError for unknown event_type."""
    with pytest.raises(ValueError, match="Unknown event_type"):
        tmp_audit.append("NOT_A_REAL_EVENT", {})


def test_audit_log_query_by_event_type(tmp_audit):
    """query() filtered by event_type returns only matching rows."""
    tmp_audit.append("FALL_CONFIRMED", {"modality": "csi"})
    tmp_audit.append("CALIBRATE", {})
    tmp_audit.append("FALL_CONFIRMED", {"modality": "radar"})

    falls = tmp_audit.query(event_type="FALL_CONFIRMED")
    assert len(falls) == 2
    assert all(r["event_type"] == "FALL_CONFIRMED" for r in falls)


def test_audit_log_query_by_room_id(tmp_audit):
    """query() filtered by room_id returns only that room's events."""
    tmp_audit.append("FALL_CONFIRMED", {}, room_id=1)
    tmp_audit.append("FALL_CONFIRMED", {}, room_id=2)
    tmp_audit.append("FALL_CONFIRMED", {}, room_id=1)

    room1_events = tmp_audit.query(room_id=1)
    assert len(room1_events) == 2
    assert all(r["room_id"] == 1 for r in room1_events)


def test_audit_log_verify_chain_intact(tmp_audit):
    """verify_chain() returns (True, None) for an untampered log."""
    tmp_audit.append("SYSTEM_START", {})
    tmp_audit.append("FALL_CONFIRMED", {"modality": "fusion"})
    tmp_audit.append("SYSTEM_STOP", {})

    intact, broken_id = tmp_audit.verify_chain()
    assert intact is True
    assert broken_id is None


def test_audit_log_verify_chain_detects_tampering(tmp_audit):
    """verify_chain() returns (False, row_id) when a hash is corrupted."""
    tmp_audit.append("SYSTEM_START", {})
    row_id = tmp_audit.append("FALL_CONFIRMED", {"height": 0.30})
    tmp_audit.append("CALIBRATE", {})

    # Directly corrupt the hash of row 2 in SQLite
    conn = sqlite3.connect(str(tmp_audit.db_path))
    conn.execute(
        "UPDATE audit_events SET sha256_hash = 'deadbeef' WHERE id = ?", (row_id,)
    )
    conn.commit()
    conn.close()

    intact, broken_id = tmp_audit.verify_chain()
    assert intact is False
    assert broken_id == row_id


def test_audit_log_empty_chain_is_intact(tmp_audit):
    """An empty audit log has an intact (vacuously true) chain."""
    intact, broken_id = tmp_audit.verify_chain()
    assert intact is True
    assert broken_id is None


# ---------------------------------------------------------------------------
# FHIRExporter tests
# ---------------------------------------------------------------------------

def test_fhir_export_produces_bundle(tmp_audit, tmp_path):
    """FHIRExporter must produce a FHIR R4 Bundle JSON file."""
    from hub.fhir_export import FHIRExporter

    tmp_audit.append("FALL_CONFIRMED", {"modality": "fusion", "height": 0.31}, room_id=1)
    tmp_audit.append("FALL_CONFIRMED", {"modality": "radar", "height": 0.28}, room_id=2)

    exporter = FHIRExporter(audit_log=tmp_audit, output_dir=tmp_path)
    out_path = exporter.export()

    assert out_path.exists()
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["resourceType"] == "Bundle"
    assert bundle["total"] == 2
    assert len(bundle["entry"]) == 2


def test_fhir_export_observation_has_loinc_code(tmp_audit, tmp_path):
    """Each Observation in the FHIR bundle must carry LOINC code 55122-0."""
    from hub.fhir_export import FHIRExporter

    tmp_audit.append("FALL_CONFIRMED", {"modality": "csi"})
    exporter = FHIRExporter(audit_log=tmp_audit, output_dir=tmp_path)
    bundle = exporter.export_as_dict()

    obs = bundle["entry"][0]["resource"]
    codes = [c["code"] for c in obs["code"]["coding"]]
    assert "55122-0" in codes


def test_fhir_export_empty_log_produces_empty_bundle(tmp_audit, tmp_path):
    """Export with no FALL_CONFIRMED events produces a bundle with total=0."""
    from hub.fhir_export import FHIRExporter

    tmp_audit.append("SYSTEM_START", {})
    exporter = FHIRExporter(audit_log=tmp_audit, output_dir=tmp_path)
    bundle = exporter.export_as_dict()
    assert bundle["total"] == 0
    assert bundle["entry"] == []


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from hub.dashboard.app import app, broadcaster

client = TestClient(app)


def test_api_audit_no_log_returns_empty():
    """GET /api/audit without AuditLog returns empty events list."""
    broadcaster.audit_log = None
    resp = client.get("/api/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0


def test_api_audit_verify_no_log_returns_503():
    """GET /api/audit/verify without AuditLog returns 503."""
    broadcaster.audit_log = None
    resp = client.get("/api/audit/verify")
    assert resp.status_code == 503


def test_api_audit_verify_intact_chain(tmp_path):
    """GET /api/audit/verify returns intact=True for untampered log."""
    audit = AuditLog(db_path=tmp_path / "audit_test.db")
    audit.append("SYSTEM_START", {})
    audit.append("FALL_CONFIRMED", {"modality": "fusion"})

    broadcaster.audit_log = audit
    try:
        resp = client.get("/api/audit/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert data["intact"] is True
        assert data["total_events"] == 2
    finally:
        broadcaster.audit_log = None
