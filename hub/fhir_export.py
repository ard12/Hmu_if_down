"""FHIR R4 Observation Export for Clinical Audit Events (Milestone 6.3).

Maps FALL_CONFIRMED audit events to FHIR R4 Observation resources
(LOINC 55122-0 — "Fall risk factor") and exports them as a JSON Bundle
for clinical record integration.

Reference: https://www.hl7.org/fhir/R4/observation.html

Usage:
    from hub.audit_log import AuditLog
    from hub.fhir_export import FHIRExporter
    audit = AuditLog()
    exporter = FHIRExporter(audit_log=audit)
    bundle_path = exporter.export(start_utc="2026-09-01T00:00:00Z", end_utc="2026-09-30T23:59:59Z")
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("fhir_export")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FHIR R4 LOINC code for fall-related observation
_FALL_LOINC_CODE = "55122-0"
_FALL_LOINC_DISPLAY = "Fall risk factors"
_FHIR_VERSION = "4.0.1"


def _make_observation(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single FALL_CONFIRMED audit row to a FHIR R4 Observation resource."""
    obs_id = str(uuid.uuid4())
    payload = event.get("payload", {})
    room_id = event.get("room_id")
    timestamp = event.get("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "survey",
                        "display": "Survey",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": _FALL_LOINC_CODE,
                    "display": _FALL_LOINC_DISPLAY,
                }
            ],
            "text": "Fall Detected",
        },
        "effectiveDateTime": timestamp,
        "valueString": f"Fall confirmed by {payload.get('modality', 'unknown')} modality",
        "component": [],
    }

    # Room location component
    if room_id is not None:
        obs["component"].append({
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "74160-7", "display": "Location"}],
                "text": "Room ID",
            },
            "valueInteger": room_id,
        })

    # Target height component
    if "height" in payload:
        obs["component"].append({
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "8302-2", "display": "Body height"}],
                "text": "Detected floor height (m)",
            },
            "valueQuantity": {
                "value": payload["height"],
                "unit": "m",
                "system": "http://unitsofmeasure.org",
                "code": "m",
            },
        })

    # Audit id as identifier extension
    obs["extension"] = [
        {
            "url": "https://github.com/ard12/Hmu_if_down/audit-log-id",
            "valueInteger": event.get("id"),
        }
    ]

    return obs


class FHIRExporter:
    """Exports FALL_CONFIRMED audit events to a FHIR R4 Observation Bundle.

    Args:
        audit_log: An AuditLog instance.
        output_dir: Directory where exported bundles are saved. Defaults to audits/.
    """

    def __init__(self, audit_log, output_dir: Optional[Path] = None):
        self.audit_log = audit_log
        self.output_dir = output_dir or (PROJECT_ROOT / "audits")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        start_utc: Optional[str] = None,
        end_utc: Optional[str] = None,
        room_id: Optional[int] = None,
    ) -> Path:
        """Export FALL_CONFIRMED events as a FHIR R4 Bundle JSON file.

        Args:
            start_utc: ISO 8601 lower bound.
            end_utc: ISO 8601 upper bound.
            room_id: Optional room filter.

        Returns:
            Path to the saved FHIR bundle file.
        """
        events = self.audit_log.query(
            event_type="FALL_CONFIRMED",
            room_id=room_id,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=10000,
        )

        entries: List[Dict[str, Any]] = []
        for event in events:
            obs = _make_observation(event)
            entries.append({
                "fullUrl": f"urn:uuid:{obs['id']}",
                "resource": obs,
            })

        bundle: Dict[str, Any] = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "meta": {"versionId": "1", "lastUpdated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "type": "collection",
            "total": len(entries),
            "entry": entries,
        }

        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = self.output_dir / f"fhir_export_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

        logger.info("FHIR bundle exported: %d observations → %s", len(entries), out_path)
        return out_path

    def export_as_dict(
        self,
        start_utc: Optional[str] = None,
        end_utc: Optional[str] = None,
        room_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return the FHIR bundle as a Python dict (without writing to disk)."""
        events = self.audit_log.query(
            event_type="FALL_CONFIRMED",
            room_id=room_id,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=10000,
        )

        entries = [{"fullUrl": f"urn:uuid:{(obs := _make_observation(e))['id']}", "resource": obs} for e in events]

        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "collection",
            "total": len(entries),
            "entry": entries,
        }
