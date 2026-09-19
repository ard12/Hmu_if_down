"""Clinical Audit Log — Immutable Hash-Chain Event Store (Milestone 6.3).

Provides a tamper-evident, SHA-256 hash-chained event log backed by SQLite.
Each row links to the previous via its hash, making retrospective tampering
detectable by re-deriving the chain.

Designed to meet FDA SaMD audit trail requirements (21 CFR Part 11 / IEC 62304).

Usage:
    log = AuditLog(db_path=Path("audits/audit.db"))
    log.append("FALL_CONFIRMED", {"room_id": 1, "modality": "fusion", "height": 0.32})
    log.append("CALIBRATE", {"room_id": 1, "triggered_by": "rest_api"})
    ok, broken_id = log.verify_chain()   # True, None  or  False, 7
    rows = log.query(event_type="FALL_CONFIRMED", room_id=1)
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("audit_log")

# Valid event types
EVENT_TYPES = frozenset({
    "FALL_CONFIRMED",
    "FALL_CANCELLED",
    "CALIBRATE",
    "THRESHOLD_CHANGE",
    "SYSTEM_START",
    "SYSTEM_STOP",
    "FHIR_EXPORT",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc  TEXT    NOT NULL,
    event_type     TEXT    NOT NULL,
    room_id        INTEGER,
    payload_json   TEXT    NOT NULL DEFAULT '{}',
    sha256_hash    TEXT    NOT NULL
);
"""


class AuditLog:
    """SQLite-backed, SHA-256 hash-chained clinical audit log.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            db_path = project_root / "audits" / "audit.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    @staticmethod
    def _compute_hash(prev_hash: str, timestamp_utc: str, payload_json: str) -> str:
        """SHA-256 of concatenated prev_hash + timestamp + payload."""
        raw = f"{prev_hash}{timestamp_utc}{payload_json}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _get_last_hash(self, conn: sqlite3.Connection) -> str:
        """Return the hash of the most recent row (or 'GENESIS' if empty)."""
        row = conn.execute(
            "SELECT sha256_hash FROM audit_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["sha256_hash"] if row else "GENESIS"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        room_id: Optional[int] = None,
        timestamp_utc: Optional[str] = None,
    ) -> int:
        """Append a new event to the hash-chained audit log.

        Args:
            event_type: One of the EVENT_TYPES constants.
            payload: Arbitrary JSON-serialisable metadata dict.
            room_id: Optional room identifier for multi-room deployments.
            timestamp_utc: Optional ISO-8601 UTC timestamp (defaults to current time).

        Returns:
            The inserted row id.

        Raises:
            ValueError: If event_type is not in EVENT_TYPES.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {event_type!r}. Must be one of {EVENT_TYPES}")

        payload = payload or {}
        if timestamp_utc is None:
            timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)

        with self._lock:
            with self._connect() as conn:
                prev_hash = self._get_last_hash(conn)
                sha256_hash = self._compute_hash(prev_hash, timestamp_utc, payload_json)
                cursor = conn.execute(
                    "INSERT INTO audit_events (timestamp_utc, event_type, room_id, payload_json, sha256_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (timestamp_utc, event_type, room_id, payload_json, sha256_hash),
                )
                conn.commit()
                row_id = cursor.lastrowid

        logger.info("Audit event #%d: %s room=%s", row_id, event_type, room_id)
        return row_id

    def query(
        self,
        event_type: Optional[str] = None,
        room_id: Optional[int] = None,
        start_utc: Optional[str] = None,
        end_utc: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query audit events with optional filters.

        Args:
            event_type: Filter by event type.
            room_id: Filter by room ID.
            start_utc: ISO 8601 timestamp lower bound (inclusive).
            end_utc: ISO 8601 timestamp upper bound (inclusive).
            limit: Maximum rows to return (default 100).
            offset: Pagination offset (default 0).

        Returns:
            List of row dicts with keys: id, timestamp_utc, event_type, room_id, payload, sha256_hash.
        """
        conditions = []
        params: List[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if room_id is not None:
            conditions.append("room_id = ?")
            params.append(room_id)
        if start_utc:
            conditions.append("timestamp_utc >= ?")
            params.append(start_utc)
        if end_utc:
            conditions.append("timestamp_utc <= ?")
            params.append(end_utc)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM audit_events {where} ORDER BY id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        result = []
        for row in rows:
            result.append({
                "id": row["id"],
                "timestamp_utc": row["timestamp_utc"],
                "event_type": row["event_type"],
                "room_id": row["room_id"],
                "payload": json.loads(row["payload_json"]),
                "sha256_hash": row["sha256_hash"],
            })
        return result

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Re-derive the entire hash chain and check for tampering.

        Returns:
            (intact: bool, first_broken_id: Optional[int])
            intact=True means the full chain is valid.
            If intact=False, first_broken_id is the id of the first tampered row.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp_utc, payload_json, sha256_hash FROM audit_events ORDER BY id ASC"
            ).fetchall()

        prev_hash = "GENESIS"
        for row in rows:
            expected = self._compute_hash(prev_hash, row["timestamp_utc"], row["payload_json"])
            if expected != row["sha256_hash"]:
                logger.warning("Audit chain broken at id=%d", row["id"])
                return False, row["id"]
            prev_hash = row["sha256_hash"]

        return True, None

    def count(self) -> int:
        """Return total number of audit events."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
