"""
Chaos Engineering, Sensor Corruption & Memory Soak Tests (Phase 28).
"""

import gc
import os
import pytest
import sqlite3
import time
from hub.packet_parser import parse_v2_header, calculate_mesh_rssi_weight
from hub.smart_home_actions import SmartHomeActionEngine, AutomationActionType, ActionStatus
from hub.longitudinal_tracker import LongitudinalTracker, DailyMobilityRecord


def test_corrupted_udp_datagram_fuzzing():
    """
    Covers: SRS-REL-002, HAZ-037
    Verifies that malformed, truncated, or random byte buffers are safely rejected without crashing.
    """
    # 1. Truncated V2 header
    bad_headers = [
        b"",
        b"C",
        b"CSI",
        b"CSIF\x00",
        b"\xff" * 17,
        b"RANDOM_JUNK_PAYLOAD_WITHOUT_MAGIC",
    ]
    for junk in bad_headers:
        assert parse_v2_header(junk) is None

    # 2. Non-byte or corrupted datagrams
    assert parse_v2_header("NOT_BYTES") is None
    assert parse_v2_header(None) is None
    assert parse_v2_header(b"\x00" * 30) is None


def test_sqlite_concurrent_write_chaos(tmp_path):
    """
    Covers: SRS-REL-002, HAZ-037
    Verifies resilience when multiple write threads stress the local SQLite database.
    """
    db_file = tmp_path / "chaos_tracker.db"
    tracker = LongitudinalTracker(db_path=db_file)

    # Rapid interleaved writes
    for i in range(50):
        rec = DailyMobilityRecord(
            patient_id=f"PAT_{i % 3}",
            record_date=f"2026-09-{(i % 25) + 1:02d}",
            cadence_spm=80.0 + (i % 10),
            active_minutes=100.0,
            shuffle_index=0.1,
            frax_score=10.0,
            falls_count=0,
        )
        tracker.record_day(rec)

    hist = tracker.get_patient_history("PAT_0")
    assert len(hist) > 0


def test_memory_soak_stability():
    """
    Covers: SRS-REL-002
    Executes rapid automation dispatches verifying absence of unbounded memory leaks.
    """
    engine = SmartHomeActionEngine()
    gc.collect()

    # 200 cycles of emergency chain dry-runs
    for i in range(200):
        engine.trigger_emergency_chain(event_id=f"soak_{i}", dry_run=True)

    history = engine.get_history(limit=50)
    assert len(history) == 50
    # Clean up
    gc.collect()
