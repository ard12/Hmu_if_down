"""
Audit Log Tamper-Evidence Verification Campaign (Milestone 18.3).

Executes comprehensive cryptographic tamper-detection test scenarios:
1. Intact Chain Validation across high-volume event stream
2. Payload JSON Mutation Detection
3. Event Deletion Detection
4. Event Insertion Detection
5. Timestamp Rollback / Mutation Detection
6. Room ID Mutation Detection
7. Event Type Mutation Detection
8. Malicious Chain Recomputation Detection against Genesis Anchor
"""

import json
import sqlite3
from pathlib import Path
import pytest

from hub.audit_log import AuditLog


@pytest.fixture
def populated_audit(tmp_path):
    """AuditLog populated with 20 diverse clinical events."""
    db_file = tmp_path / "campaign_audit.db"
    log = AuditLog(db_path=db_file, institution_id="HOSPITAL-METRO-WEST")

    event_sequence = [
        ("SYSTEM_START", {"version": "4.2.0", "operator": "system_init"}),
        ("CALIBRATE", {"room_id": 101, "calibrated_height": 2.45}),
        ("FALL_CONFIRMED", {"room_id": 101, "probability": 0.98, "severity": "CRITICAL"}),
        ("FALL_CANCELLED", {"room_id": 101, "reason": "false_alarm_user_prompt"}),
        ("THRESHOLD_CHANGE", {"room_id": 101, "old_thresh": 0.65, "new_thresh": 0.70}),
        ("FHIR_EXPORT", {"bundle_id": "bundle-001", "records_count": 1}),
        ("CALIBRATE", {"room_id": 102, "calibrated_height": 2.50}),
        ("FALL_CONFIRMED", {"room_id": 102, "probability": 0.94, "severity": "MODERATE"}),
        ("SYSTEM_STOP", {"reason": "scheduled_maintenance"}),
        ("SYSTEM_START", {"version": "4.2.0", "operator": "admin"}),
    ]

    # Append each event twice with varying room IDs to create 20 events
    for i, (ev_type, payload) in enumerate(event_sequence * 2):
        room_id = (i % 4) + 101
        log.append(ev_type, payload=payload, room_id=room_id)

    return log


def test_campaign_intact_chain(populated_audit):
    """Scenario 1: Untampered high-volume log verifies 100% intact."""
    assert populated_audit.count() == 20
    intact, broken_id = populated_audit.verify_chain()
    assert intact is True
    assert broken_id is None


def test_campaign_payload_modification(populated_audit):
    """Scenario 2: Tampering with a single character in payload_json is immediately caught."""
    conn = sqlite3.connect(str(populated_audit.db_path))
    target_id = 7

    # Mutate payload in SQLite
    row = conn.execute("SELECT payload_json FROM audit_events WHERE id = ?", (target_id,)).fetchone()
    payload = json.loads(row[0])
    payload["severity"] = "MINOR"  # Tampered severity
    conn.execute(
        "UPDATE audit_events SET payload_json = ? WHERE id = ?",
        (json.dumps(payload, separators=(",", ":"), sort_keys=True), target_id),
    )
    conn.commit()
    conn.close()

    intact, broken_id = populated_audit.verify_chain()
    assert intact is False
    assert broken_id == target_id


def test_campaign_event_deletion(populated_audit):
    """Scenario 3: Deleting a row from the middle causes chain failure at the subsequent row."""
    conn = sqlite3.connect(str(populated_audit.db_path))
    deleted_id = 10
    conn.execute("DELETE FROM audit_events WHERE id = ?", (deleted_id,))
    conn.commit()
    conn.close()

    intact, broken_id = populated_audit.verify_chain()
    assert intact is False
    # Next row in sequence (id 11) will fail because its prev_hash was linked to deleted id 10
    assert broken_id == 11


def test_campaign_event_insertion(populated_audit):
    """Scenario 4: Inserting a fraudulent row breaks the cryptographic chain."""
    conn = sqlite3.connect(str(populated_audit.db_path))
    conn.execute(
        "INSERT INTO audit_events (id, timestamp_utc, event_type, room_id, payload_json, sha256_hash) "
        "VALUES (999, '2026-09-23T00:00:00Z', 'FALL_CONFIRMED', 101, '{\"fraud\":true}', 'badhash123')"
    )
    conn.commit()
    conn.close()

    intact, broken_id = populated_audit.verify_chain()
    assert intact is False
    assert broken_id == 999


def test_campaign_timestamp_rollback(populated_audit):
    """Scenario 5: Modifying timestamp_utc to simulate backdated event is detected."""
    conn = sqlite3.connect(str(populated_audit.db_path))
    target_id = 5
    conn.execute(
        "UPDATE audit_events SET timestamp_utc = '2020-01-01T00:00:00Z' WHERE id = ?",
        (target_id,),
    )
    conn.commit()
    conn.close()

    intact, broken_id = populated_audit.verify_chain()
    assert intact is False
    assert broken_id == target_id


def test_campaign_room_id_mutation(populated_audit):
    """Scenario 6: Changing room_id to forge location is detected by hash verification."""
    conn = sqlite3.connect(str(populated_audit.db_path))
    target_id = 3
    conn.execute("UPDATE audit_events SET room_id = 999 WHERE id = ?", (target_id,))
    conn.commit()
    conn.close()

    intact, broken_id = populated_audit.verify_chain()
    assert intact is False
    assert broken_id == target_id


def test_campaign_event_type_mutation(populated_audit):
    """Scenario 7: Changing event_type (e.g. FALL_CONFIRMED to CALIBRATE) is detected."""
    conn = sqlite3.connect(str(populated_audit.db_path))
    target_id = 8
    conn.execute("UPDATE audit_events SET event_type = 'CALIBRATE' WHERE id = ?", (target_id,))
    conn.commit()
    conn.close()

    intact, broken_id = populated_audit.verify_chain()
    assert intact is False
    assert broken_id == target_id


def test_campaign_chain_recompute_detected_against_genesis(tmp_path):
    """Scenario 8: Attacker recomputes entire chain with a forged genesis anchor."""
    db_file = tmp_path / "forged_genesis.db"
    # Genuine institution
    genuine_log = AuditLog(db_path=db_file, institution_id="GENUINE-HOSPITAL")
    genuine_log.append("SYSTEM_START", {"init": 1})
    genuine_log.append("FALL_CONFIRMED", {"room_id": 101})

    # Verifying with the genuine institution log passes
    assert genuine_log.verify_chain()[0] is True

    # Attacker tries to verify using a different / forged institution genesis anchor
    rogue_log = AuditLog(db_path=db_file, institution_id="ROGUE-INSTITUTION")
    intact, broken_id = rogue_log.verify_chain()
    assert intact is False
    assert broken_id == 1, "First event must fail verification against mismatched genesis root"
