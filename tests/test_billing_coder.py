"""
Unit tests for Medical Billing & Clinical Coding Engine (Phase 26).
"""

import pytest
from hub.billing_coder import BillingCoder, FallEtiology


@pytest.fixture
def coder():
    return BillingCoder()


def test_etiology_mapping_rules(coder):
    """
    Covers: SRS-CLN-002
    Verifies normalized biomechanics labels map to standard etiology classifications.
    """
    assert coder.map_biomechanics_to_etiology("forward_slip_trip") == "SLIP_TRIP"
    assert coder.map_biomechanics_to_etiology("bed_roll_fall") == "BED_FALL"
    assert coder.map_biomechanics_to_etiology("chair_collapse") == "CHAIR_FALL"
    assert coder.map_biomechanics_to_etiology("syncope_drop") == "SYNCOPE"
    assert coder.map_biomechanics_to_etiology("unidentified_motion") == "UNSPECIFIED"


def test_billing_recommendation_generation(coder):
    """
    Covers: SRS-CLN-002
    Verifies generation of primary ICD-10 code, secondary codes, and CPT RPM codes.
    """
    rec = coder.generate_billing_recommendation(
        biomechanics_label="slip_trip",
        has_prior_falls=True,
        gait_cadence_abnormal=True,
    )

    assert rec.primary_icd10.code == "W01.0XXA"
    assert rec.primary_icd10.system == "ICD-10-CM"

    # Secondary diagnoses
    sec_codes = [c.code for c in rec.secondary_icd10]
    assert "Z91.81" in sec_codes  # History of falling
    assert "R29.6" in sec_codes   # Tendency to fall
    assert "R26.2" in sec_codes   # Difficulty walking

    # CPT codes
    cpt_codes = [c.code for c in rec.cpt_codes]
    assert "99453" in cpt_codes
    assert "99454" in cpt_codes
    assert "99457" in cpt_codes


def test_deterministic_unspecified_fallback(coder):
    """
    Covers: HAZ-035
    Verifies that unknown or corrupt biomechanics label safely falls back to W19.XXXA.
    """
    rec = coder.generate_billing_recommendation(
        biomechanics_label="completely_unknown_movement_xyz"
    )
    assert rec.primary_icd10.code == "W19.XXXA"
    assert "unspecified" in rec.primary_icd10.description.lower()
