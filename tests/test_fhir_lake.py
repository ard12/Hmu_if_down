"""Tests for AES-256 Fernet encrypted FHIR R4 Data Lake."""

import os
import sqlite3
import tempfile
import threading
import time
import pytest
from cryptography.fernet import Fernet, InvalidToken

from hub.fhir_lake import FHIRDataLake


@pytest.fixture
def temp_lake():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_fhir.db")
        key = Fernet.generate_key()
        yield FHIRDataLake(db_path=db_path, key=key), db_path, key


def test_write_read_round_trip(temp_lake):
    """write_resource + read_resource round-trip produces identical dict."""
    lake, _, _ = temp_lake
    patient = {
        "resourceType": "Patient",
        "name": [{"family": "Smith", "given": ["Alice"]}],
        "gender": "female",
        "birthDate": "1948-11-20",
    }
    res_id = lake.write_resource("Patient", patient)

    retrieved = lake.read_resource(res_id)
    assert retrieved["resourceType"] == "Patient"
    assert retrieved["id"] == res_id
    assert retrieved["name"][0]["family"] == "Smith"
    assert retrieved["gender"] == "female"


def test_raw_sqlite_row_is_encrypted(temp_lake):
    """Raw SQLite row is not plaintext (encryption verified)."""
    lake, db_path, _ = temp_lake
    observation = {
        "resourceType": "Observation",
        "status": "final",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "217082002", "display": "Fall"}]},
        "subject": {"reference": "Patient/PAT999"},
    }
    res_id = lake.write_resource("Observation", observation)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT resource_json_enc FROM fhir_resources WHERE id = ?", (res_id,))
    raw_enc = cursor.fetchone()[0]
    conn.close()

    # Raw bytes must NOT contain plaintext strings
    assert b"217082002" not in raw_enc
    assert b"Observation" not in raw_enc
    assert b"PAT999" not in raw_enc


def test_search_by_patient_id(temp_lake):
    """search by patient_id returns only matching records."""
    lake, _, _ = temp_lake

    obs1 = {
        "resourceType": "Observation",
        "status": "final",
        "subject": {"reference": "Patient/P1"},
        "note": [{"text": "Fall link 1"}],
    }
    obs2 = {
        "resourceType": "Observation",
        "status": "final",
        "subject": {"reference": "Patient/P2"},
        "note": [{"text": "Fall link 2"}],
    }
    lake.write_resource("Observation", obs1)
    lake.write_resource("Observation", obs2)

    results_p1 = lake.search("Observation", patient_id="P1")
    assert len(results_p1) == 1
    assert results_p1[0]["note"][0]["text"] == "Fall link 1"

    results_p2 = lake.search("Observation", patient_id="P2")
    assert len(results_p2) == 1
    assert results_p2[0]["note"][0]["text"] == "Fall link 2"


def test_export_bundle(temp_lake):
    """export_bundle produces valid FHIR R4 Bundle with resourceType: 'Bundle'."""
    lake, _, _ = temp_lake

    patient_res = {
        "resourceType": "Patient",
        "id": "PAT_BUNDLE_1",
        "gender": "male",
    }
    obs_res = {
        "resourceType": "Observation",
        "status": "final",
        "subject": {"reference": "Patient/PAT_BUNDLE_1"},
    }
    lake.write_resource("Patient", patient_res)
    lake.write_resource("Observation", obs_res)

    bundle = lake.export_bundle("PAT_BUNDLE_1")

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] == 2
    assert len(bundle["entry"]) == 2
    types = {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert types == {"Patient", "Observation"}


def test_rotate_key(temp_lake):
    """rotate_key re-encrypts records with new key, making old key invalid."""
    lake, db_path, old_key = temp_lake
    res_id = lake.write_resource("Patient", {"resourceType": "Patient", "gender": "other"})

    new_key = Fernet.generate_key()
    lake.rotate_key(new_key)

    # Lake now reads under new key
    retrieved = lake.read_resource(res_id)
    assert retrieved["gender"] == "other"

    # Old key cannot decrypt
    old_lake = FHIRDataLake(db_path=db_path, key=old_key)
    with pytest.raises(InvalidToken):
        old_lake.read_resource(res_id)


def test_wrong_key_raises_invalid_token(temp_lake):
    """Attempting to read with incorrect key raises InvalidToken."""
    lake, db_path, _ = temp_lake
    res_id = lake.write_resource("Patient", {"resourceType": "Patient", "gender": "unknown"})

    wrong_key = Fernet.generate_key()
    wrong_lake = FHIRDataLake(db_path=db_path, key=wrong_key)

    with pytest.raises(InvalidToken):
        wrong_lake.read_resource(res_id)


def test_unknown_resource_type_raises_value_error(temp_lake):
    """write_resource for unknown resource type raises ValueError."""
    lake, _, _ = temp_lake
    with pytest.raises(ValueError):
        lake.write_resource("InvalidResourceType", {"resourceType": "InvalidResourceType"})


def test_concurrent_writes(temp_lake):
    """Thread-safe concurrent writes to same patient."""
    lake, _, _ = temp_lake

    def writer(thread_id):
        for i in range(15):
            lake.write_resource(
                "Observation",
                {
                    "resourceType": "Observation",
                    "subject": {"reference": "Patient/P_CONCURRENT"},
                    "note": [{"text": f"Event {thread_id}_{i}"}],
                },
            )

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results = lake.search("Observation", patient_id="P_CONCURRENT")
    assert len(results) == 60
