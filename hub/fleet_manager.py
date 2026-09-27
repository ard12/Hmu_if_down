"""
Fleet Management Module — Handles Edge Hub provisioning and heartbeat monitoring.
"""
from contextlib import contextmanager
import sqlite3
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

logger = logging.getLogger("fleet_manager")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fleet_devices (
    device_id       TEXT PRIMARY KEY,
    facility_id     TEXT NOT NULL,
    room_id         TEXT NOT NULL,
    firmware_ver    TEXT NOT NULL,
    model_ver       TEXT NOT NULL,
    status          TEXT NOT NULL,
    last_seen_at    REAL NOT NULL,
    registered_at   REAL NOT NULL
);
"""

class FleetManager:
    """Manages the registry of Edge Hubs across multiple facilities."""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "audits" / "fleet.db"
        else:
            self.db_path = db_path
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def provision_device(self, facility_id: str, room_id: str, firmware_ver: str, model_ver: str) -> str:
        """Securely provision a new Edge Hub and return its device_id."""
        device_id = f"hub_{uuid4().hex[:12]}"
        now = time.time()
        
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fleet_devices 
                (device_id, facility_id, room_id, firmware_ver, model_ver, status, last_seen_at, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (device_id, facility_id, room_id, firmware_ver, model_ver, "online", now, now)
            )
            conn.commit()
        
        logger.info(f"Provisioned new device {device_id} in {facility_id}/{room_id}")
        return device_id

    def record_heartbeat(self, device_id: str, firmware_ver: Optional[str] = None, model_ver: Optional[str] = None) -> bool:
        """Update the last_seen_at timestamp and optionally versions for a device."""
        now = time.time()
        
        with self._connect() as conn:
            # Check if device exists
            row = conn.execute("SELECT * FROM fleet_devices WHERE device_id = ?", (device_id,)).fetchone()
            if not row:
                return False
                
            update_sql = "UPDATE fleet_devices SET last_seen_at = ?, status = 'online'"
            params = [now]
            
            if firmware_ver:
                update_sql += ", firmware_ver = ?"
                params.append(firmware_ver)
            if model_ver:
                update_sql += ", model_ver = ?"
                params.append(model_ver)
                
            update_sql += " WHERE device_id = ?"
            params.append(device_id)
            
            conn.execute(update_sql, params)
            conn.commit()
            
        return True

    def get_fleet_status(self, facility_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve the status matrix of all devices, evaluating staleness."""
        now = time.time()
        STALE_THRESHOLD_S = 120.0  # 2 minutes without heartbeat -> offline
        
        with self._connect() as conn:
            if facility_id:
                rows = conn.execute("SELECT * FROM fleet_devices WHERE facility_id = ?", (facility_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM fleet_devices").fetchall()
                
            devices = []
            for r in rows:
                device = dict(r)
                if now - device["last_seen_at"] > STALE_THRESHOLD_S:
                    device["status"] = "offline"
                    # We don't write this back to DB here to avoid locking on reads,
                    # just report it dynamically. A background worker could update DB.
                devices.append(device)
                
            return devices

fleet_manager = FleetManager()
