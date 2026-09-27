"""
Unit tests for Post-Fall Clinical Incident Report Generator (Phase 26).
"""

import time
import pytest
from pathlib import Path
from hub.incident_report_generator import (
    IncidentReportGenerator,
    IncidentReportData,
)


@pytest.fixture
def generator(tmp_path):
    return IncidentReportGenerator(output_dir=tmp_path)


@pytest.fixture
def sample_incident_data():
    return IncidentReportData(
        event_id="evt_test_7788",
        patient_id="PAT_4412",
        room_id="room_suite_b",
        timestamp=time.time(),
        biomechanics_class="slip_trip",
        peak_velocity_mps=-2.15,
        impact_altitude_m=0.28,
        torso_angle_deg=78.5,
        shap_top_features=[
            {"name": "vertical_velocity", "value": -2.15, "importance": 0.42},
            {"name": "floor_proximity", "value": 0.28, "importance": 0.38},
            {"name": "torso_angular_rate", "value": 145.0, "importance": 0.12},
        ],
        respiration_bpm=18.5,
        escalation_tier_reached=3,
        caregiver_response_time_sec=42.0,
        automations_triggered=["LIGHTS_ON (light.room_suite_b)", "UNLOCK_DOORS (lock.front_door)"],
        voice_transcript="help me please",
    )


def test_markdown_report_generation(generator, sample_incident_data):
    """
    Covers: SRS-CLN-001
    Verifies creation of human-readable, physician-grade Markdown incident report.
    """
    md = generator.generate_markdown_report(sample_incident_data)
    assert "# CLINICAL INCIDENT REPORT: ACUTE FALL EVENT" in md
    assert "evt_test_7788" in md
    assert "PAT_4412" in md
    assert "room_suite_b" in md
    assert "vertical_velocity" in md
    assert "W01.0XXA" in md
    assert "Attending Physician" in md


def test_fhir_diagnostic_report_generation(generator, sample_incident_data):
    """
    Covers: SRS-CLN-001, HAZ-035
    Verifies generation of HL7 FHIR R4 DiagnosticReport conforming to schema.
    """
    fhir = generator.generate_fhir_diagnostic_report(sample_incident_data)
    assert fhir["resourceType"] == "DiagnosticReport"
    assert fhir["id"] == "incident-report-evt_test_7788"
    assert fhir["subject"]["reference"] == "Patient/PAT_4412"
    assert "conclusion" in fhir
    assert fhir["code"]["coding"][0]["code"] == "55122-0"
