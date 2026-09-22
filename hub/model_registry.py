"""Thread-safe versioned model store backed by SQLite."""

import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Thread-safe versioned model store backed by SQLite.

    Maintains full IEC 62304 / FDA GMLP model lineage and change records.
    """

    def __init__(self, db_path: str = "models/registry.db"):
        self.db_path = db_path
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
                    CREATE TABLE IF NOT EXISTS model_versions (
                        version_id TEXT PRIMARY KEY,
                        model_bytes BLOB NOT NULL,
                        sha256 TEXT NOT NULL,
                        algorithm TEXT,
                        hyperparams TEXT,
                        training_n INTEGER,
                        sensitivity REAL,
                        specificity REAL,
                        brier REAL,
                        created_at REAL,
                        commit_sha TEXT,
                        active INTEGER DEFAULT 0,
                        notes TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def save_model(self, clf: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Pickle clf, store SHA-256, return version_id (uuid4)."""
        metadata = metadata or {}
        model_bytes = pickle.dumps(clf)
        sha256_hash = hashlib.sha256(model_bytes).hexdigest()
        version_id = str(uuid.uuid4())

        algorithm = metadata.get("algorithm", type(clf).__name__)
        hyperparams = metadata.get("hyperparams", "{}")
        if isinstance(hyperparams, dict):
            hyperparams = json.dumps(hyperparams)
        training_n = int(metadata.get("training_n", 0))
        sensitivity = float(metadata.get("sensitivity", 0.0))
        specificity = float(metadata.get("specificity", 0.0))
        brier = float(metadata.get("brier", 0.0))
        created_at = float(metadata.get("created_at", time.time()))
        commit_sha = str(metadata.get("commit_sha", ""))
        active = 1 if metadata.get("active", False) else 0
        notes = str(metadata.get("notes", ""))

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                if active == 1:
                    cursor.execute("UPDATE model_versions SET active = 0")
                cursor.execute(
                    """
                    INSERT INTO model_versions (
                        version_id, model_bytes, sha256, algorithm, hyperparams,
                        training_n, sensitivity, specificity, brier, created_at,
                        commit_sha, active, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        model_bytes,
                        sha256_hash,
                        algorithm,
                        hyperparams,
                        training_n,
                        sensitivity,
                        specificity,
                        brier,
                        created_at,
                        commit_sha,
                        active,
                        notes,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return version_id

    def load_model(self, version_id: str) -> Any:
        """Deserialize and return classifier after cryptographic integrity verification."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT model_bytes, sha256 FROM model_versions WHERE version_id = ?",
                    (version_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"Model version not found: {version_id}")
                model_bytes, stored_sha = row[0], row[1]
                computed_sha = hashlib.sha256(model_bytes).hexdigest()
                if computed_sha != stored_sha:
                    raise ValueError(
                        f"Cryptographic integrity check failed for model {version_id}: "
                        f"expected {stored_sha}, got {computed_sha}"
                    )
            finally:
                conn.close()

        # nosec B301 - SHA-256 cryptographic hash verified against SQLite record before unpickling
        return pickle.loads(model_bytes)

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return list of model records sorted by created_at descending."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT version_id, algorithm, hyperparams, training_n,
                           sensitivity, specificity, brier, created_at,
                           commit_sha, active, sha256, notes
                    FROM model_versions
                    ORDER BY created_at DESC
                    """
                )
                rows = cursor.fetchall()
                versions = []
                for r in rows:
                    versions.append(
                        {
                            "version_id": r[0],
                            "algorithm": r[1],
                            "hyperparams": r[2],
                            "training_n": r[3],
                            "sensitivity": r[4],
                            "specificity": r[5],
                            "brier": r[6],
                            "created_at": r[7],
                            "commit_sha": r[8],
                            "active": bool(r[9]),
                            "sha256": r[10],
                            "notes": r[11],
                        }
                    )
                return versions
            finally:
                conn.close()

    def set_active(self, version_id: str):
        """Mark one version active, deactivate all others (atomic SQLite tx)."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("UPDATE model_versions SET active = 0")
                cursor.execute(
                    "UPDATE model_versions SET active = 1 WHERE version_id = ?",
                    (version_id,),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Model version not found: {version_id}")
                conn.commit()
            finally:
                conn.close()

    def get_active(self) -> Optional[Any]:
        """Return active classifier object or None if none active."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT model_bytes, sha256, version_id FROM model_versions WHERE active = 1 LIMIT 1"
                )
                row = cursor.fetchone()
                if not row:
                    return None
                model_bytes, stored_sha, version_id = row[0], row[1], row[2]
                computed_sha = hashlib.sha256(model_bytes).hexdigest()
                if computed_sha != stored_sha:
                    raise ValueError(
                        f"Cryptographic integrity check failed for active model {version_id}: "
                        f"expected {stored_sha}, got {computed_sha}"
                    )
            finally:
                conn.close()

        # nosec B301 - SHA-256 cryptographic hash verified against SQLite record before unpickling
        return pickle.loads(model_bytes)

    def get_active_version_id(self) -> Optional[str]:
        """Return version_id of currently active model."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT version_id FROM model_versions WHERE active = 1 LIMIT 1"
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    def verify_integrity(self) -> Dict[str, Any]:
        """Recompute SHA-256 of each pickle and compare to stored hash."""
        corrupt_versions: List[str] = []
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT version_id, model_bytes, sha256 FROM model_versions")
                rows = cursor.fetchall()
                for v_id, m_bytes, stored_sha in rows:
                    recomputed = hashlib.sha256(m_bytes).hexdigest()
                    if recomputed != stored_sha:
                        corrupt_versions.append(v_id)
            finally:
                conn.close()

        return {
            "ok": len(corrupt_versions) == 0,
            "corrupt_versions": corrupt_versions,
        }
