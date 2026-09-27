"""
Medical Billing & Clinical Coding Engine (v5.0.0).
Maps detected fall biomechanics, gait anomalies, and remote telemetry events
to standard ICD-10-CM diagnostic codes and CPT Remote Patient Monitoring (RPM) billing codes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FallEtiology(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    SLIP_TRIP = "SLIP_TRIP"
    BED_FALL = "BED_FALL"
    CHAIR_FALL = "CHAIR_FALL"
    SYNCOPE = "SYNCOPE"
    SLUMP_COLLAPSE = "SLUMP_COLLAPSE"


@dataclass
class ClinicalCodeEntry:
    code: str
    system: str  # "ICD-10-CM" or "CPT"
    description: str
    reimbursement_category: str
    confidence: float = 1.0


@dataclass
class BillingRecommendation:
    primary_icd10: ClinicalCodeEntry
    secondary_icd10: List[ClinicalCodeEntry]
    cpt_codes: List[ClinicalCodeEntry]
    notes: str


class BillingCoder:
    """Clinical ICD-10 and CPT coding assistant for remote fall telemetry."""

    # Primary ICD-10-CM Fall codes (Initial encounter - suffix 'A')
    ICD10_MAP: Dict[str, ClinicalCodeEntry] = {
        "SLIP_TRIP": ClinicalCodeEntry(
            code="W01.0XXA",
            system="ICD-10-CM",
            description="Fall on same level from slipping, tripping, and stumbling without falling onto stairs or steps, initial encounter",
            reimbursement_category="Accidental Fall",
            confidence=0.95,
        ),
        "BED_FALL": ClinicalCodeEntry(
            code="W06.XXXA",
            system="ICD-10-CM",
            description="Fall from bed, initial encounter",
            reimbursement_category="Accidental Fall",
            confidence=0.95,
        ),
        "CHAIR_FALL": ClinicalCodeEntry(
            code="W07.XXXA",
            system="ICD-10-CM",
            description="Fall from chair, initial encounter",
            reimbursement_category="Accidental Fall",
            confidence=0.95,
        ),
        "SYNCOPE": ClinicalCodeEntry(
            code="R55",
            system="ICD-10-CM",
            description="Syncope and collapse",
            reimbursement_category="Symptom/Etiology",
            confidence=0.90,
        ),
        "UNSPECIFIED": ClinicalCodeEntry(
            code="W19.XXXA",
            system="ICD-10-CM",
            description="Unspecified fall, initial encounter",
            reimbursement_category="Accidental Fall",
            confidence=0.85,
        ),
    }

    # Secondary comorbidity / mobility findings
    SECONDARY_CODES: Dict[str, ClinicalCodeEntry] = {
        "HISTORY_OF_FALLS": ClinicalCodeEntry(
            code="Z91.81",
            system="ICD-10-CM",
            description="History of falling",
            reimbursement_category="Risk Factor",
        ),
        "TENDENCY_TO_FALL": ClinicalCodeEntry(
            code="R29.6",
            system="ICD-10-CM",
            description="Repeated falls / Tendency to fall",
            reimbursement_category="Mobility Disorder",
        ),
        "GAIT_DIFFICULTY": ClinicalCodeEntry(
            code="R26.2",
            system="ICD-10-CM",
            description="Difficulty in walking, not elsewhere classified",
            reimbursement_category="Mobility Disorder",
        ),
    }

    # CPT RPM Remote Physiological Monitoring codes
    CPT_RPM_CODES = [
        ClinicalCodeEntry(
            code="99453",
            system="CPT",
            description="Remote monitoring of physiologic parameter(s); initial set-up and patient education on use of equipment",
            reimbursement_category="RPM Setup",
        ),
        ClinicalCodeEntry(
            code="99454",
            system="CPT",
            description="Remote monitoring of physiologic parameter(s); initial collection, transmission, and report per 30 days",
            reimbursement_category="RPM Transmission",
        ),
        ClinicalCodeEntry(
            code="99457",
            system="CPT",
            description="Remote physiologic monitoring treatment management services, clinical staff/physician time, first 20 minutes",
            reimbursement_category="Clinical Management",
        ),
    ]

    @classmethod
    def map_biomechanics_to_etiology(cls, biomechanics_label: str) -> str:
        """Normalize biomechanics classifier label to a standard etiology key."""
        label = (biomechanics_label or "").lower()
        if "slip" in label or "trip" in label or "forward" in label:
            return "SLIP_TRIP"
        elif "bed" in label:
            return "BED_FALL"
        elif "chair" in label or "sit" in label:
            return "CHAIR_FALL"
        elif "syncope" in label or "faint" in label:
            return "SYNCOPE"
        return "UNSPECIFIED"

    def generate_billing_recommendation(
        self,
        biomechanics_label: str = "unspecified",
        has_prior_falls: bool = False,
        gait_cadence_abnormal: bool = False,
    ) -> BillingRecommendation:
        """Derive ICD-10 and CPT coding set from patient telemetry observations."""
        etiology_key = self.map_biomechanics_to_etiology(biomechanics_label)
        primary = self.ICD10_MAP.get(etiology_key, self.ICD10_MAP["UNSPECIFIED"])

        secondary: List[ClinicalCodeEntry] = []
        if has_prior_falls:
            secondary.append(self.SECONDARY_CODES["TENDENCY_TO_FALL"])
            secondary.append(self.SECONDARY_CODES["HISTORY_OF_FALLS"])
        if gait_cadence_abnormal:
            secondary.append(self.SECONDARY_CODES["GAIT_DIFFICULTY"])

        notes = (
            f"Derived from automated multi-modal sensor telemetry (biomechanics={biomechanics_label}). "
            f"Requires clinical physician review prior to claim filing."
        )

        return BillingRecommendation(
            primary_icd10=primary,
            secondary_icd10=secondary,
            cpt_codes=list(self.CPT_RPM_CODES),
            notes=notes,
        )


billing_coder = BillingCoder()
