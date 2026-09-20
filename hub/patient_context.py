"""Thread-safe patient context store indexed by room_id with alert enrichment."""

import datetime
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional


class PatientContextStore:
    """In-memory + SQLite-backed patient context indexed by room_id.

    Thread-safe. Provides patient context enrichment for fall alerts.
    """

    HIGH_RISK_MEDICATIONS = [
        "warfarin",
        "apixaban",
        "rivaroxaban",
        "clopidogrel",
        "furosemide",
        "hydrochlorothiazide",
        "metoprolol",
        "oxycodone",
        "morphine",
        "lorazepam",
        "diazepam",
    ]

    def __init__(self, db_path: str = "models/patient_context.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = {}
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        self._load_cache()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patient_records (
                        room_id TEXT PRIMARY KEY,
                        patient_id TEXT,
                        dob TEXT,
                        gender TEXT,
                        admit_dt TEXT,
                        morse_fall_scale INTEGER,
                        medications_json TEXT,
                        updated_at REAL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def _load_cache(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT room_id, patient_id, dob, gender, admit_dt,
                           morse_fall_scale, medications_json
                    FROM patient_records
                    """
                )
                rows = cursor.fetchall()
                for r in rows:
                    meds = json.loads(r[6]) if r[6] else []
                    self._cache[r[0]] = {
                        "room_id": r[0],
                        "patient_id": r[1],
                        "dob": r[2],
                        "gender": r[3],
                        "admit_dt": r[4],
                        "morse_fall_scale": r[5],
                        "medications": meds,
                    }
            finally:
                conn.close()

    def upsert(self, room_id: str, patient: Dict[str, Any]):
        """Write/overwrite patient record for a room."""
        r_id = str(room_id)
        p_id = str(patient.get("patient_id", ""))
        dob = str(patient.get("dob", ""))
        gender = str(patient.get("gender", "U"))
        admit_dt = str(patient.get("admit_dt", ""))
        mfs = patient.get("morse_fall_scale")
        mfs_int = int(mfs) if mfs is not None else None
        meds = list(patient.get("medications", []))
        now = time.time()

        record = {
            "room_id": r_id,
            "patient_id": p_id,
            "dob": dob,
            "gender": gender,
            "admit_dt": admit_dt,
            "morse_fall_scale": mfs_int,
            "medications": meds,
        }

        with self._lock:
            self._cache[r_id] = record
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO patient_records (
                        room_id, patient_id, dob, gender, admit_dt,
                        morse_fall_scale, medications_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (r_id, p_id, dob, gender, admit_dt, mfs_int, json.dumps(meds), now),
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Return current patient dict for room, or None if unoccupied."""
        with self._lock:
            rec = self._cache.get(str(room_id))
            return dict(rec) if rec else None

    def clear(self, room_id: str):
        """Clear patient context upon discharge."""
        r_id = str(room_id)
        with self._lock:
            self._cache.pop(r_id, None)
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM patient_records WHERE room_id = ?", (r_id,))
                conn.commit()
            finally:
                conn.close()

    def _compute_age(self, dob_str: str) -> Optional[int]:
        """Compute age in years from YYYYMMDD or YYYY-MM-DD."""
        if not dob_str:
            return None
        cleaned = dob_str.replace("-", "")[:8]
        if len(cleaned) < 8 or not cleaned.isdigit():
            return None
        try:
            birth_year = int(cleaned[:4])
            birth_month = int(cleaned[4:6])
            birth_day = int(cleaned[6:8])
            today = datetime.date.today()
            age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
            return age
        except Exception:
            return None

    def enrich_alert(self, alert: Dict[str, Any], room_id: str) -> Dict[str, Any]:
        """Enrich a fall alert dict with patient demographics, risk score, and medications."""
        enriched = dict(alert)
        patient = self.get(room_id)
        if not patient:
            return enriched

        p_id = patient.get("patient_id", "")
        mfs = patient.get("morse_fall_scale")
        dob = patient.get("dob", "")
        gender = patient.get("gender", "U")
        meds = patient.get("medications", [])

        # Filter high-risk medications
        high_risk_meds = []
        for m in meds:
            m_lower = m.lower()
            for hrm in self.HIGH_RISK_MEDICATIONS:
                if hrm in m_lower:
                    high_risk_meds.append(m)
                    break

        age = self._compute_age(dob)
        is_high_risk = bool(mfs is not None and mfs > 45)

        enriched["patient_id"] = p_id
        enriched["morse_fall_scale"] = mfs
        enriched["high_risk_meds"] = high_risk_meds
        enriched["age"] = age
        enriched["gender"] = gender
        enriched["high_risk"] = is_high_risk

        return enriched
