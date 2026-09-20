"""Tests for PatientContextStore, high-risk medication filtering, and alert enrichment."""

import os
import tempfile
import threading
import time
import pytest

from hub.patient_context import PatientContextStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_patient_context.db")
        yield PatientContextStore(db_path=db_path)


def test_upsert_get_round_trip(temp_store):
    """upsert / get round-trip returns identical patient record."""
    patient = {
        "patient_id": "P1001",
        "dob": "19600101",
        "gender": "F",
        "admit_dt": "20260920080000",
        "morse_fall_scale": 50,
        "medications": ["Metoprolol", "Aspirin"],
    }
    temp_store.upsert("room_101", patient)

    retrieved = temp_store.get("room_101")
    assert retrieved is not None
    assert retrieved["patient_id"] == "P1001"
    assert retrieved["gender"] == "F"
    assert retrieved["morse_fall_scale"] == 50
    assert retrieved["medications"] == ["Metoprolol", "Aspirin"]


def test_clear_removes_record(temp_store):
    """clear removes record (get returns None)."""
    patient = {"patient_id": "P1002", "gender": "M"}
    temp_store.upsert("room_102", patient)
    assert temp_store.get("room_102") is not None

    temp_store.clear("room_102")
    assert temp_store.get("room_102") is None


def test_enrich_alert_injects_patient_fields(temp_store):
    """enrich_alert injects patient_id, age, gender, and risk flags."""
    patient = {
        "patient_id": "P1003",
        "dob": "19500510",
        "gender": "M",
        "morse_fall_scale": 60,
        "medications": ["Warfarin sodium", "Lisinopril"],
    }
    temp_store.upsert("room_103", patient)

    base_alert = {"state": "FALL_DETECTED", "severity": "HIGH"}
    enriched = temp_store.enrich_alert(base_alert, "room_103")

    assert enriched["patient_id"] == "P1003"
    assert enriched["gender"] == "M"
    assert enriched["morse_fall_scale"] == 60
    assert enriched["high_risk"] is True
    assert "Warfarin sodium" in enriched["high_risk_meds"]
    assert isinstance(enriched["age"], int)
    assert enriched["age"] >= 70


def test_high_risk_medication_filtering(temp_store):
    """HIGH_RISK_MEDICATIONS filter tags anticoagulants/sedatives but not antibiotics."""
    patient = {
        "patient_id": "P1004",
        "medications": [
            "Amoxicillin 500mg",
            "Warfarin 5mg",
            "Paracetamol",
            "Oxycodone HCl",
            "Atorvastatin",
        ],
    }
    temp_store.upsert("room_104", patient)

    enriched = temp_store.enrich_alert({}, "room_104")
    hrm = enriched["high_risk_meds"]

    assert "Warfarin 5mg" in hrm
    assert "Oxycodone HCl" in hrm
    assert "Amoxicillin 500mg" not in hrm
    assert "Paracetamol" not in hrm


def test_morse_fall_scale_threshold_flag(temp_store):
    """Morse Fall Scale > 45 flagged as high_risk: True, <= 45 as False."""
    # Patient 1: Low risk (MFS = 30)
    temp_store.upsert("room_low", {"patient_id": "P_LOW", "morse_fall_scale": 30})
    enriched_low = temp_store.enrich_alert({}, "room_low")
    assert enriched_low["high_risk"] is False

    # Patient 2: High risk (MFS = 55)
    temp_store.upsert("room_high", {"patient_id": "P_HIGH", "morse_fall_scale": 55})
    enriched_high = temp_store.enrich_alert({}, "room_high")
    assert enriched_high["high_risk"] is True


def test_concurrent_upsert_and_get(temp_store):
    """Concurrent upsert and get from two threads is safe."""
    def writer():
        for i in range(50):
            temp_store.upsert(f"room_{i % 5}", {"patient_id": f"PAT_{i}", "morse_fall_scale": i})
            time.sleep(0.001)

    def reader():
        for i in range(50):
            temp_store.get(f"room_{i % 5}")
            time.sleep(0.001)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert temp_store.get("room_0") is not None


def test_sqlite_persistence_across_instances():
    """Records survive PatientContextStore reconstruction from same db_path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "persist.db")

        store1 = PatientContextStore(db_path=db_path)
        store1.upsert("room_persist", {"patient_id": "P_SAVED", "morse_fall_scale": 70})

        # Re-create from same db
        store2 = PatientContextStore(db_path=db_path)
        retrieved = store2.get("room_persist")
        assert retrieved is not None
        assert retrieved["patient_id"] == "P_SAVED"
        assert retrieved["morse_fall_scale"] == 70
