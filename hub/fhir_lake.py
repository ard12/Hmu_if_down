"""AES-256 (via Fernet) encrypted at-rest FHIR R4 data lake."""

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken


class FHIRDataLake:
    """AES-256 (via Fernet) encrypted at-rest FHIR R4 resource store.

    Backend: SQLite table `fhir_resources (id, resource_type, resource_json_enc, created_at, patient_id)`.
    Supports Patient, Observation (fall events), MedicationStatement, RiskAssessment, Bundle.
    """

    SUPPORTED_RESOURCES = {
        "Patient",
        "Observation",
        "MedicationStatement",
        "RiskAssessment",
        "Bundle",
    }

    def __init__(self, db_path: str, key: bytes):
        """key must be 32-byte URL-safe base64 (Fernet.generate_key())."""
        self.db_path = db_path
        self.key = key
        self.fernet = Fernet(key)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fhir_resources (
                        id TEXT PRIMARY KEY,
                        resource_type TEXT NOT NULL,
                        resource_json_enc BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        patient_id TEXT
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fhir_patient ON fhir_resources(patient_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fhir_type ON fhir_resources(resource_type)"
                )
                conn.commit()
            finally:
                conn.close()

    def write_resource(self, resource_type: str, fhir_dict: Dict[str, Any]) -> str:
        """Encrypts JSON, stores in SQLite, returns FHIR resource ID (UUID)."""
        if resource_type not in self.SUPPORTED_RESOURCES:
            raise ValueError(f"Unsupported FHIR resource type: {resource_type}")

        dict_type = fhir_dict.get("resourceType")
        if dict_type and dict_type != resource_type:
            raise ValueError(
                f"Resource type mismatch: expected {resource_type}, got {dict_type}"
            )

        resource_id = str(fhir_dict.get("id") or uuid.uuid4())
        fhir_dict["id"] = resource_id
        fhir_dict["resourceType"] = resource_type

        patient_id = ""
        if resource_type == "Patient":
            patient_id = resource_id
        elif "subject" in fhir_dict and isinstance(fhir_dict["subject"], dict):
            ref = str(fhir_dict["subject"].get("reference", ""))
            patient_id = ref.replace("Patient/", "").strip()
        elif "patient_id" in fhir_dict:
            patient_id = str(fhir_dict["patient_id"])

        json_bytes = json.dumps(fhir_dict).encode("utf-8")
        encrypted_bytes = self.fernet.encrypt(json_bytes)
        now = time.time()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO fhir_resources (
                        id, resource_type, resource_json_enc, created_at, patient_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (resource_id, resource_type, encrypted_bytes, now, patient_id),
                )
                conn.commit()
            finally:
                conn.close()

        return resource_id

    def read_resource(self, resource_id: str) -> Dict[str, Any]:
        """Decrypts and returns FHIR resource dict."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT resource_json_enc FROM fhir_resources WHERE id = ?",
                    (resource_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"FHIR resource not found: {resource_id}")
                encrypted_bytes = row[0]
            finally:
                conn.close()

        decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
        return json.loads(decrypted_bytes.decode("utf-8"))

    def search(self, resource_type: str, **params) -> List[Dict[str, Any]]:
        """Decrypts all matching records. Supports patient_id and date filtering."""
        patient_id = params.get("patient_id")
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        query = "SELECT resource_json_enc FROM fhir_resources WHERE resource_type = ?"
        sql_params: List[Any] = [resource_type]

        if patient_id:
            query += " AND patient_id = ?"
            sql_params.append(str(patient_id))
        if start_date is not None:
            query += " AND created_at >= ?"
            sql_params.append(float(start_date))
        if end_date is not None:
            query += " AND created_at <= ?"
            sql_params.append(float(end_date))

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(query, sql_params)
                rows = cursor.fetchall()
            finally:
                conn.close()

        results: List[Dict[str, Any]] = []
        for (enc_bytes,) in rows:
            dec_bytes = self.fernet.decrypt(enc_bytes)
            results.append(json.loads(dec_bytes.decode("utf-8")))

        return results

    def export_bundle(self, patient_id: str) -> Dict[str, Any]:
        """Returns FHIR R4 Bundle JSON with all resources for a patient."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT resource_json_enc FROM fhir_resources
                    WHERE patient_id = ? OR (resource_type = 'Patient' AND id = ?)
                    ORDER BY created_at ASC
                    """,
                    (str(patient_id), str(patient_id)),
                )
                rows = cursor.fetchall()
            finally:
                conn.close()

        entries: List[Dict[str, Any]] = []
        for (enc_bytes,) in rows:
            dec_bytes = self.fernet.decrypt(enc_bytes)
            res_dict = json.loads(dec_bytes.decode("utf-8"))
            entries.append({"resource": res_dict})

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "total": len(entries),
            "entry": entries,
        }

    def rotate_key(self, new_key: bytes):
        """Re-encrypts all records with new_key in an atomic transaction."""
        new_fernet = Fernet(new_key)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, resource_json_enc FROM fhir_resources")
                rows = cursor.fetchall()

                updated_records: List[tuple] = []
                for res_id, old_enc in rows:
                    decrypted = self.fernet.decrypt(old_enc)
                    new_enc = new_fernet.encrypt(decrypted)
                    updated_records.append((new_enc, res_id))

                cursor.executemany(
                    "UPDATE fhir_resources SET resource_json_enc = ? WHERE id = ?",
                    updated_records,
                )
                conn.commit()
            finally:
                conn.close()

        self.fernet = new_fernet
        self.key = new_key
