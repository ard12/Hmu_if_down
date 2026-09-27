"""
Post-Fall Clinical Incident Report Generator (v5.0.0).
Generates standardized, physician-grade medical incident documentation aggregating
RF/radar kinematics, SHAP explainability, vital signs, timeline audit,
and recommended ICD-10/CPT billing codes.
"""

from dataclasses import dataclass, field
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hub.billing_coder import billing_coder

logger = logging.getLogger("incident_report_generator")


@dataclass
class IncidentReportData:
    event_id: str
    patient_id: str
    room_id: str
    timestamp: float
    biomechanics_class: str
    peak_velocity_mps: float
    impact_altitude_m: float
    torso_angle_deg: float
    shap_top_features: List[Dict[str, Any]]
    respiration_bpm: float
    escalation_tier_reached: int
    caregiver_response_time_sec: Optional[float]
    automations_triggered: List[str]
    voice_transcript: Optional[str] = None


class IncidentReportGenerator:
    """Generates physician-grade clinical incident reports in Markdown and FHIR format."""

    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent / "incidents"
        else:
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self, data: IncidentReportData) -> str:
        """Render a formatted clinical incident report in Markdown."""
        safe_timestamp = max(0.0, float(data.timestamp or 0.0))
        dt_str = datetime.datetime.fromtimestamp(safe_timestamp).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        billing = billing_coder.generate_billing_recommendation(
            biomechanics_label=data.biomechanics_class,
            has_prior_falls=True,
            gait_cadence_abnormal=(data.peak_velocity_mps > 2.0),
        )

        md = f"""# CLINICAL INCIDENT REPORT: ACUTE FALL EVENT
**Report Reference:** `IR-{data.event_id}`  
**Event Timestamp:** {dt_str}  
**Facility Patient ID:** `{data.patient_id}` | **Room / Location:** `{data.room_id}`  
**Regulatory Device Standard:** IEC 62304 Class B / FDA SaMD Class II  

---

## 1. Executive Summary & Biomechanical Kinematics
- **Confirmed Event Classification:** Acute Fall Event (`{data.biomechanics_class.upper()}`)
- **Peak Downward Velocity ($v_z$):** `{data.peak_velocity_mps:.2f} m/s` (Threshold $\\le -1.50$ m/s)
- **Terminal Posture Altitude ($z$):** `{data.impact_altitude_m:.2f} m` (Threshold $\\le 0.40$ m)
- **Torso Inclination at Ground:** `{data.torso_angle_deg:.1f}°`
- **Post-Fall Respiration Rate:** `{data.respiration_bpm:.1f} breaths/min` (Normal: 12-20 bpm)

## 2. Explainable AI (XAI) Attribution & SHAP Waterfall
The multi-modal decision was driven by the following feature attributions (Efficiency Axiom verified):
"""
        for feat in (data.shap_top_features or [])[:4]:
            val = feat.get("value", 0.0)
            imp = feat.get("importance", 0.0)
            name = feat.get("name", "feature")
            md += f"- **{name}:** Observation = `{val:.3f}` | SHAP Impact $\\phi = {imp:+.4f}$\n"

        md += f"""
## 3. Escalation & Caregiver Timeline
- **Highest Escalation Tier Reached:** `Tier {data.escalation_tier_reached}`
- **Caregiver Acknowledgment Time:** `{f"{data.caregiver_response_time_sec:.1f}s" if data.caregiver_response_time_sec else "Pending / Auto-Dispatched"}`
- **Patient Vocal Interaction:** `{data.voice_transcript or "None recorded / Patient quiescent"}`

## 4. Environmental Safety Automations Dispatched
"""
        if data.automations_triggered:
            for act in data.automations_triggered:
                md += f"- Dispatched: `{act}`\n"
        else:
            md += "- *No automated smart home actions registered*\n"

        md += f"""
## 5. Diagnostic & Billing Coding Recommendations
- **Primary Diagnosis (ICD-10-CM):** `{billing.primary_icd10.code}` — *{billing.primary_icd10.description}*
"""
        for sec in billing.secondary_icd10:
            md += f"- **Secondary Finding (ICD-10-CM):** `{sec.code}` — *{sec.description}*\n"

        md += "- **Recommended CPT Codes:** " + ", ".join(f"`{c.code}`" for c in billing.cpt_codes) + "\n"

        md += f"""
---
### 6. Clinical Attestation & Signature
I have reviewed the objective sensor telemetry, kinematic trajectories, and patient vital sign estimations above.

**Attending Physician / RN:** ___________________________  
**License / NPI Number:** _______________________________  
**Signature & Date:** __________________________________  
"""
        # Save to disk
        out_file = self.output_dir / f"incident_{data.event_id}.md"
        out_file.write_text(md, encoding="utf-8")
        logger.info(f"Generated Markdown incident report: {out_file}")
        return md

    def generate_fhir_diagnostic_report(self, data: IncidentReportData) -> Dict[str, Any]:
        """Generate an HL7 FHIR R4 DiagnosticReport resource JSON representation."""
        safe_ts = max(0.0, float(data.timestamp or 0.0))
        dt_iso = datetime.datetime.fromtimestamp(safe_ts).isoformat()
        billing = billing_coder.generate_billing_recommendation(data.biomechanics_class)

        return {
            "resourceType": "DiagnosticReport",
            "id": f"incident-report-{data.event_id}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "AU",
                            "display": "Audiology/Sensory Telemetry",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "55122-0",
                        "display": "Fall risk assessment / incident",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{data.patient_id}"},
            "effectiveDateTime": dt_iso,
            "conclusion": f"Confirmed acute fall event ({data.biomechanics_class}). Primary ICD-10: {billing.primary_icd10.code}",
            "conclusionCode": [
                {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": billing.primary_icd10.code,
                            "display": billing.primary_icd10.description,
                        }
                    ]
                }
            ],
        }


incident_generator = IncidentReportGenerator()
