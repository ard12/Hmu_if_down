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
    pers_codes: List[ClinicalCodeEntry] = field(default_factory=list)
    rtm_codes: List[ClinicalCodeEntry] = field(default_factory=list)
    compliance_disclaimer: str = (
        "CLINICAL & REGULATORY COMPLIANCE NOTICE: Codes are automated suggestions derived from "
        "ambient sensor telemetry. CPT RPM codes (99453-99457) require qualified physiologic monitoring. "
        "For passive fall detection without vital sign parameters, HCPCS PERS (S5160/S5161) or RTM "
        "(98975/98977) codes should be evaluated. All claims require attending clinician attestation."
    )


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

    # HCPCS Monitored Personal Emergency Response System (PERS) codes
    HCPCS_PERS_CODES = [
        ClinicalCodeEntry(
            code="S5160",
            system="HCPCS",
            description="Emergency response system; installation and testing",
            reimbursement_category="PERS Installation",
        ),
        ClinicalCodeEntry(
            code="S5161",
            system="HCPCS",
            description="Emergency response system; service fee, per month (excludes installation and testing)",
            reimbursement_category="PERS Monthly Monitoring",
        ),
    ]

    # CPT Remote Therapeutic Monitoring (RTM) codes
    CPT_RTM_CODES = [
        ClinicalCodeEntry(
            code="98975",
            system="CPT",
            description="Remote therapeutic monitoring (e.g., musculoskeletal system status); initial set-up and patient education",
            reimbursement_category="RTM Setup",
        ),
        ClinicalCodeEntry(
            code="98977",
            system="CPT",
            description="Remote therapeutic monitoring; device(s) supply with scheduled transmission to monitor musculoskeletal system, each 30 days",
            reimbursement_category="RTM Transmission",
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
            pers_codes=list(self.HCPCS_PERS_CODES),
            rtm_codes=list(self.CPT_RTM_CODES),
        )


billing_coder = BillingCoder()
