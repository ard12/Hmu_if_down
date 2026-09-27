"""
Extreme Depth Codebase Audit & Edge Case Stress Verification Script.
Exhaustively exercises boundary conditions, thread contention, and sanitization
across Phases 20 through 28 modules.
"""

import concurrent.futures
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hub.voice_responder import VoiceResponder, VoiceState
from hub.smart_home_actions import SmartHomeActionEngine, AutomationRule, AutomationActionType, ActionStatus
from hub.billing_coder import BillingCoder
from hub.incident_report_generator import IncidentReportGenerator, IncidentReportData
from hub.longitudinal_tracker import LongitudinalTracker, DailyMobilityRecord
from hub.startup_self_test import StartupSelfTest
from hub.diagnostics import WatchdogHeartbeat
from hub.fleet_manager import FleetManager
from hub.ota_updater import OTAManager


def audit_voice_responder():
    print("[1/7] Auditing Voice Responder boundary cases...")
    vr = VoiceResponder(room_id="audit_room", confidence_threshold=0.6)

    # Edge cases in transcripts
    assert vr.process_transcript("") is None
    assert vr.process_transcript(None) is None
    assert vr.process_transcript("   ") is None
    assert vr.process_transcript("!@#$%^&*()") is None
    assert vr.process_transcript("🚀🚨🏥") is None

    # Multi-word distress with noise
    evt = vr.process_transcript("Um hello, please help me right now")
    assert evt is not None
    assert evt.event_type == "DISTRESS_KEYWORD"

    # Cancellation
    cancel_evt = vr.process_transcript("Stop! I am fine, false alarm!")
    assert cancel_evt is not None
    assert cancel_evt.event_type == "CANCEL_KEYWORD"
    assert vr.state == VoiceState.ALERT_SUPPRESSED

    # Audio energy bounds & NaN/Inf sanitization
    assert vr.calculate_audio_energy([]) == -100.0
    assert vr.calculate_audio_energy([float("nan"), float("inf")]) == -100.0
    assert vr.calculate_audio_energy([0.0, 0.0, 0.0]) == -100.0
    db = vr.calculate_audio_energy([0.5, -0.5, 0.5, -0.5])
    assert -10.0 <= db <= 0.0

    # Rapid intercom toggling
    for _ in range(10):
        vr.open_intercom()
        assert vr.intercom_channel_open is True
        vr.close_intercom()
        assert vr.intercom_channel_open is False

    status = vr.get_status()
    assert isinstance(status, dict)
    print("      -> Voice Responder: PASSED")


def audit_smart_home_actions():
    print("[2/7] Auditing Smart Home Actions under concurrency...")
    engine = SmartHomeActionEngine(max_retries=-1)  # Test negative retries clamp
    assert engine.max_retries == 0

    engine.max_retries = 2

    # Concurrent emergency chain dispatches
    def worker(worker_id):
        return engine.trigger_emergency_chain(event_id=f"concurrent_evt_{worker_id}", dry_run=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            results = f.result()
            assert len(results) >= 4

    history = engine.get_history(limit=100)
    assert len(history) >= 20
    print("      -> Smart Home Actions: PASSED")


def audit_billing_coder():
    print("[3/7] Auditing Medical Billing Coder edge cases...")
    coder = BillingCoder()

    # None, empty, and irregular strings
    assert coder.map_biomechanics_to_etiology(None) == "UNSPECIFIED"
    assert coder.map_biomechanics_to_etiology("") == "UNSPECIFIED"
    assert coder.map_biomechanics_to_etiology("   ") == "UNSPECIFIED"
    assert coder.map_biomechanics_to_etiology("SLIP_ON_WET_FLOOR") == "SLIP_TRIP"
    assert coder.map_biomechanics_to_etiology("SYNCOPE_EPISODE") == "SYNCOPE"

    rec = coder.generate_billing_recommendation(
        biomechanics_label="chair_fall",
        has_prior_falls=1,
        gait_cadence_abnormal=0,
    )
    assert rec.primary_icd10.code == "W07.XXXA"
    assert any(c.code == "Z91.81" for c in rec.secondary_icd10)
    print("      -> Medical Billing Coder: PASSED")


def audit_incident_report_generator():
    print("[4/7] Auditing Incident Report Generator edge cases...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = IncidentReportGenerator(output_dir=Path(tmp_dir))

        # Test extreme kinematic bounds & negative/zero timestamps
        data = IncidentReportData(
            event_id="extreme_001",
            patient_id="PAT_EDGE_01",
            room_id="room_iso_9",
            timestamp=-500.0,  # Negative timestamp guard test
            biomechanics_class="unspecified",
            peak_velocity_mps=-15.0,
            impact_altitude_m=-0.5,
            torso_angle_deg=180.0,
            shap_top_features=[],  # Empty features
            respiration_bpm=0.0,
            escalation_tier_reached=5,
            caregiver_response_time_sec=None,  # Unacknowledged
            automations_triggered=[],
            voice_transcript=None,
        )

        md = gen.generate_markdown_report(data)
        assert "IR-extreme_001" in md
        assert "Pending / Auto-Dispatched" in md

        fhir = gen.generate_fhir_diagnostic_report(data)
        assert fhir["resourceType"] == "DiagnosticReport"
        assert fhir["id"] == "incident-report-extreme_001"
    print("      -> Incident Report Generator: PASSED")


def audit_longitudinal_tracker():
    print("[5/7] Auditing Longitudinal Tracker concurrency & math...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "longitudinal_audit.db"
        tracker = LongitudinalTracker(db_path=db_path)

        # Ingestion of negative values (should be clamped >= 0)
        rec = DailyMobilityRecord(
            patient_id="PAT_TEST_EDGE",
            record_date="2026-09-01",
            cadence_spm=-10.0,
            active_minutes=-50.0,
            shuffle_index=-0.2,
            frax_score=-5.0,
            falls_count=-1,
        )
        tracker.record_day(rec)
        hist = tracker.get_patient_history("PAT_TEST_EDGE")
        assert len(hist) == 1
        assert hist[0]["cadence_spm"] == 0.0
        assert hist[0]["active_minutes"] == 0.0

        # Concurrent read/write stress
        def writer(day_idx):
            tracker.record_day(
                DailyMobilityRecord(
                    patient_id=f"CONCURRENT_{day_idx % 2}",
                    record_date=f"2026-09-{(day_idx % 25) + 1:02d}",
                    cadence_spm=80.0,
                    active_minutes=90.0,
                    shuffle_index=0.1,
                    frax_score=10.0,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(writer, i) for i in range(30)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        trend = tracker.evaluate_mobility_trend("CONCURRENT_0")
        assert "status" in trend
    print("      -> Longitudinal Tracker: PASSED")


def audit_startup_self_test_and_watchdog():
    print("[6/7] Auditing Startup Self-Test & Watchdog...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tester = StartupSelfTest(project_root=Path(tmp_dir))
        report = tester.run_preflight()
        assert report.passed is True

    # Watchdog NTP time shift simulation
    wd = WatchdogHeartbeat(stall_timeout_sec=5.0)
    wd.last_heartbeat = time.time() + 100.0  # Clock skewed into future
    health = wd.check_health()
    assert health["healthy"] is True  # Should not crash or report negative elapsed stall
    print("      -> Startup Self-Test & Watchdog: PASSED")


def audit_fleet_and_ota():
    print("[7/7] Auditing Fleet Manager & OTA Updater...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "fleet_audit.db"
        fm = FleetManager(db_path=db_path)

        dev_id = fm.provision_device("FAC_A", "ROOM_01", "v1.0", "v4.8.0")
        assert dev_id.startswith("hub_")
        assert fm.record_heartbeat(dev_id, model_ver="v4.8.1") is True

    ota = OTAManager()
    # Test None model bytes
    try:
        ota.generate_update_payload("5.0.0", None)
        assert False, "Should have raised ValueError on None model_bytes"
    except ValueError:
        pass

    # Valid payload
    payload = ota.generate_update_payload("5.0.0", b"test_model_binary")
    assert ota.verify_and_apply_update(payload) is True

    # Malformed data_hex
    bad_payload = dict(payload)
    bad_payload["data_hex"] = "INVALID_HEX_STRING_ZZZ"
    assert ota.verify_and_apply_update(bad_payload) is False
    print("      -> Fleet & OTA: PASSED")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("STARTING EXTREME CODEBASE AUDIT")
    print("=" * 60)

    audit_voice_responder()
    audit_smart_home_actions()
    audit_billing_coder()
    audit_incident_report_generator()
    audit_longitudinal_tracker()
    audit_startup_self_test_and_watchdog()
    audit_fleet_and_ota()

    t_elapsed = time.time() - t0
    print("=" * 60)
    print(f"ALL 7/7 EXTREME AUDIT MODULES COMPLETED SUCCESSFULLY in {t_elapsed:.2f}s!")
    print("=" * 60)
