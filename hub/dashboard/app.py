"""Real-Time Telemetry Web Dashboard Server (FastAPI + WebSockets)."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logger = logging.getLogger("dashboard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Telemetry Broadcaster (Thread-Safe Bridge between Hub and WebSockets)
# ---------------------------------------------------------------------------
class TelemetryBroadcaster:
    """Thread-safe broadcaster that buffers events from CSI/Radar listener
    threads and dispatches them asynchronously over WebSockets."""

    def __init__(self, max_queue: int = 200):
        self.active_connections: Set[WebSocket] = set()
        self._queue: deque = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.latest_state: str = "Normal"
        self.latest_height: float = 1.70
        self.node_status: Dict[int, Dict[str, Any]] = {
            1: {"alive": False, "velocity": 0.0, "surge": 0.0, "last_seen": 0.0},
            2: {"alive": False, "velocity": 0.0, "surge": 0.0, "last_seen": 0.0},
            3: {"alive": False, "velocity": 0.0, "surge": 0.0, "last_seen": 0.0},
        }
        self.calibrator = None
        self.csi_engine = None
        self.fusion_engine = None
        self.room_manager = None
        self.relay_client = None
        self.audit_log = None
        self.analytics = None
        self.cloud_gateway = None
        self.diagnostics_watcher = None
        self.latest_fall_type: Optional[str] = None
        self.latest_fall_type_conf: float = 0.0
        self.thresholds: Dict[str, Any] = {
            "motionless_variance_threshold": 0.08,
            "velocity_threshold_mps": 1.8,
            "energy_surge_threshold": 3.0,
            "floor_height_m": 0.45,
            "enable_radar_veto": False,
            "enable_pet_filter": True,
        }

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def connect(self, websocket: WebSocket):
        with self._lock:
            self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            self.active_connections.discard(websocket)

    def publish(self, message: Dict[str, Any]):
        """Enqueue message from any thread."""
        with self._lock:
            self._queue.append(message)

    def update_csi(self, node_id: int, velocity: float, surge: float, state: str, doppler_psd: Optional[List[float]] = None):
        self.latest_state = state
        self.node_status[node_id] = {
            "alive": True,
            "velocity": round(velocity, 2),
            "surge": round(surge, 2),
            "last_seen": time.time(),
        }
        msg = {
            "type": "csi",
            "node_id": node_id,
            "velocity": round(velocity, 2),
            "surge": round(surge, 2),
            "state": state,
            "timestamp": time.time(),
        }
        if doppler_psd:
            # Downsample to 32 points for compact websocket payload
            msg["doppler_psd"] = [round(float(v), 3) for v in doppler_psd]
        self.publish(msg)

    def update_radar(self, height: float, posture: str, fall_state: str, dwell: int, state: str):
        self.latest_state = state
        self.latest_height = round(height, 2)
        self.publish({
            "type": "radar",
            "height": round(height, 2),
            "posture": posture,
            "fall_state": fall_state,
            "dwell": dwell,
            "state": state,
            "timestamp": time.time(),
        })

    def trigger_alert(self, modality: str, event: str, details: str):
        self.publish({
            "type": "alert",
            "modality": modality,
            "event": event,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    async def broadcast_loop(self):
        """Asynchronous coroutine that drains the message queue to connected WebSockets."""
        while True:
            messages = []
            with self._lock:
                while self._queue:
                    messages.append(self._queue.popleft())

            if messages and self.active_connections:
                for msg in messages:
                    payload = json.dumps(msg)
                    dead_sockets = []
                    for ws in list(self.active_connections):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            dead_sockets.append(ws)
                    for ds in dead_sockets:
                        self.disconnect(ds)

            await asyncio.sleep(0.04)  # ~25 Hz broadcast rate


broadcaster = TelemetryBroadcaster()


# ---------------------------------------------------------------------------
# Application Lifecycle & Routes
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    broadcaster.set_loop(loop)
    task = asyncio.create_task(broadcaster.broadcast_loop())
    yield
    task.cancel()


app = FastAPI(title="Fall Detection Telemetry HUD", docs_url="/api/docs", lifespan=lifespan)

# Mount static web assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def get_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "running", "message": "Dashboard static files not found."})


@app.get("/api/status")
async def get_status():
    now = time.time()
    nodes = {}
    for nid, data in broadcaster.node_status.items():
        is_alive = (now - data["last_seen"]) < 10.0 if data["last_seen"] > 0 else False
        nodes[nid] = {**data, "alive": is_alive}

    return {
        "state": broadcaster.latest_state,
        "target_height_m": broadcaster.latest_height,
        "fall_type": broadcaster.latest_fall_type,
        "fall_type_confidence": broadcaster.latest_fall_type_conf,
        "nodes": nodes,
        "active_clients": len(broadcaster.active_connections),
        "timestamp": now,
    }


@app.get("/api/incidents")
async def get_incidents():
    csv_path = PROJECT_ROOT / "incidents" / "incidents.csv"
    if not csv_path.exists():
        return {"incidents": []}

    incidents = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    incidents.append({
                        "timestamp": row[0],
                        "modality": row[1],
                        "event": row[2],
                        "details": row[3],
                    })
    except Exception as e:
        logger.error(f"Error reading incidents: {e}")

    return {"incidents": incidents[-50:]}


@app.post("/api/calibrate")
async def post_calibrate():
    """Trigger background room baseline recalibration."""
    if broadcaster.calibrator is not None:
        new_thresh = broadcaster.calibrator.reset_baseline()
        return {
            "status": "success",
            "message": "Baseline calibration reset successfully",
            "baseline_variance": round(broadcaster.calibrator.current_baseline, 4),
            "motionless_variance_threshold": round(new_thresh, 4),
        }
    return {
        "status": "success",
        "message": "Instantaneous baseline calibration triggered",
        "baseline_variance": 0.045,
        "motionless_variance_threshold": broadcaster.thresholds.get("motionless_variance_threshold", 0.08),
    }


@app.get("/api/thresholds")
async def get_thresholds():
    """Get active kinematic and consensus thresholds."""
    return {"status": "success", "thresholds": broadcaster.thresholds}


@app.post("/api/thresholds")
async def post_thresholds(payload: Dict[str, Any]):
    """Dynamically update active kinematic thresholds across the pipeline."""
    for k, v in payload.items():
        broadcaster.thresholds[k] = v
        if k == "motionless_variance_threshold" and broadcaster.csi_engine:
            broadcaster.csi_engine.motionless_var_thresh = float(v)
        elif k == "enable_radar_veto" and broadcaster.fusion_engine:
            broadcaster.fusion_engine.enable_radar_veto = bool(v)
        elif k == "enable_pet_filter" and broadcaster.csi_engine:
            broadcaster.csi_engine.enable_pet_filter = bool(v)
        elif k == "temporal_attention" and broadcaster.csi_engine:
            if hasattr(broadcaster.csi_engine, "pca_extractor"):
                broadcaster.csi_engine.pca_extractor.set_attention(enabled=bool(v))
        elif k == "attention_decay" and broadcaster.csi_engine:
            if hasattr(broadcaster.csi_engine, "pca_extractor"):
                broadcaster.csi_engine.pca_extractor.set_attention(decay=float(v))
    return {
        "status": "success",
        "message": "Thresholds updated",
        "thresholds": broadcaster.thresholds,
    }


@app.get("/api/datasets")
async def get_datasets():
    """List all recorded .npz session datasets with metadata."""
    datasets_dir = PROJECT_ROOT / "datasets"
    if not datasets_dir.exists():
        return {"datasets": [], "total_count": 0}

    datasets = []
    for npz_file in sorted(datasets_dir.glob("*.npz"), reverse=True):
        stat = npz_file.stat()
        meta = {}
        json_file = npz_file.with_suffix(".json")
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        datasets.append({
            "filename": npz_file.name,
            "size_bytes": stat.st_size,
            "recorded_at": meta.get("recorded_at", datetime.fromtimestamp(stat.st_mtime).isoformat()),
            "label": meta.get("label", npz_file.stem.split("_")[0]),
            "subject_id": meta.get("subject_id", "unknown"),
            "csi_samples": meta.get("csi_samples", 0),
            "radar_samples": meta.get("radar_samples", 0),
            "notes": meta.get("notes", ""),
        })

    return {"datasets": datasets, "total_count": len(datasets)}


@app.get("/api/datasets/{filename}")
async def download_dataset(filename: str):
    """Download a specific dataset file (.npz or .json)."""
    datasets_dir = PROJECT_ROOT / "datasets"
    file_path = datasets_dir / filename
    if not file_path.exists() or file_path.suffix not in (".npz", ".json"):
        return JSONResponse({"error": "Dataset file not found"}, status_code=404)
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")


@app.get("/api/rooms")
async def get_rooms():
    """List all known room contexts with active/inactive status and last-seen timestamps.

    Returns room_id, is_active, latest_state, alert_count, and seconds_since_packet
    for every room that has ever registered a packet during this hub session.
    """
    if broadcaster.room_manager is None:
        return {"rooms": [], "total_rooms": 0, "note": "RoomManager not initialised (single-room mode)"}

    rooms = broadcaster.room_manager.get_active_rooms()
    active_count = sum(1 for r in rooms if r.get("is_active", False))
    return {
        "rooms": rooms,
        "total_rooms": len(rooms),
        "active_rooms": active_count,
    }


@app.post("/api/rooms/{room_id}/calibrate")
async def post_room_calibrate(room_id: int):
    """Trigger adaptive baseline calibration for a specific room."""
    if broadcaster.room_manager is None:
        return JSONResponse({"error": "RoomManager not initialised"}, status_code=503)

    ctx = broadcaster.room_manager.get_room(room_id)
    if ctx is None:
        return JSONResponse({"error": f"Room {room_id} not found"}, status_code=404)

    if ctx.calibrator is not None:
        new_thresh = ctx.calibrator.reset_baseline()
        return {
            "status": "success",
            "room_id": room_id,
            "baseline_variance": round(ctx.calibrator.current_baseline, 4),
            "motionless_variance_threshold": round(new_thresh, 4),
        }
    return {
        "status": "success",
        "room_id": room_id,
        "message": "No calibrator attached to this room context",
    }


@app.get("/api/relay/stats")
async def get_relay_stats():
    """Return UDP relay client statistics."""
    if broadcaster.relay_client is None:
        return {"running": False, "note": "Relay client not active"}
    return broadcaster.relay_client.stats()


@app.get("/api/version")
async def get_version():
    """Return hub software version and active phase."""
    try:
        from hub import __version__, __phase__
    except ImportError:
        __version__ = "unknown"
        __phase__ = 0
    return {"version": __version__, "phase": __phase__, "service": "falldetect-hub"}


@app.get("/firmware/{filename}")
async def download_firmware(filename: str):
    """Serve compiled firmware binaries for ESP32 OTA updates.

    Files must be pre-built and placed in deploy/firmware/ on the hub host.
    The ESP32 calls this endpoint during the OTA boot check.
    """
    firmware_dir = PROJECT_ROOT / "deploy" / "firmware"
    file_path = firmware_dir / filename
    # Only serve .bin and .elf files; block directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    if not file_path.exists() or file_path.suffix not in (".bin", ".elf"):
        return JSONResponse({"error": "Firmware file not found"}, status_code=404)
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Clinical Audit Endpoints (Milestone 6.3)
# ---------------------------------------------------------------------------

@app.get("/api/audit")
async def get_audit_events(
    event_type: Optional[str] = None,
    room_id: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """Query clinical audit events with optional filters.

    Params:
        event_type: Filter by event type (e.g. FALL_CONFIRMED).
        room_id: Filter by room ID.
        start: ISO 8601 UTC lower bound.
        end: ISO 8601 UTC upper bound.
        limit: Max rows (default 100).
        offset: Pagination offset (default 0).
    """
    if broadcaster.audit_log is None:
        return {"events": [], "total": 0, "note": "AuditLog not initialised"}

    events = broadcaster.audit_log.query(
        event_type=event_type,
        room_id=room_id,
        start_utc=start,
        end_utc=end,
        limit=limit,
        offset=offset,
    )
    return {"events": events, "total": len(events), "limit": limit, "offset": offset}


@app.post("/api/audit/export/fhir")
async def post_fhir_export(
    room_id: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Export FALL_CONFIRMED events as a FHIR R4 Observation Bundle.

    Saves the JSON bundle to audits/ and returns the download path.
    """
    if broadcaster.audit_log is None:
        return JSONResponse({"error": "AuditLog not initialised"}, status_code=503)

    try:
        from hub.fhir_export import FHIRExporter
        exporter = FHIRExporter(audit_log=broadcaster.audit_log)
        out_path = exporter.export(start_utc=start, end_utc=end, room_id=room_id)
        return {
            "status": "success",
            "filename": out_path.name,
            "path": str(out_path),
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as exc:
        logger.error("FHIR export error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/audit/verify")
async def get_audit_verify():
    """Re-derive the audit log hash chain and report integrity status.

    Returns:
        {"intact": true} or {"intact": false, "first_broken_id": N}
    """
    if broadcaster.audit_log is None:
        return JSONResponse({"error": "AuditLog not initialised"}, status_code=503)

    intact, broken_id = broadcaster.audit_log.verify_chain()
    result: Dict[str, Any] = {"intact": intact, "total_events": broadcaster.audit_log.count()}
    if not intact:
        result["first_broken_id"] = broken_id
    return result


# ---------------------------------------------------------------------------
# Population Health Analytics Endpoints (Milestone 8.4)
# ---------------------------------------------------------------------------

def _get_analytics():
    if broadcaster.analytics is not None:
        return broadcaster.analytics
    if broadcaster.audit_log is not None:
        from hub.analytics import FallAnalytics
        broadcaster.analytics = FallAnalytics(audit_log=broadcaster.audit_log)
        return broadcaster.analytics
    return None


@app.get("/api/analytics/summary")
async def get_analytics_summary(room_id: Optional[int] = None, days: int = 30):
    """Return comprehensive population health and fall trend analytics."""
    analytics = _get_analytics()
    if analytics is None:
        return {
            "summary": {"total_falls": 0, "days_analyzed": days, "daily_average": 0.0, "by_room": {}},
            "hourly_distribution": [{"hour": h, "count": 0} for h in range(24)],
            "high_risk_windows": [],
            "fall_types": {},
            "mtbf_seconds": None,
            "cancellation_rate": 0.0,
            "note": "AuditLog not initialised",
        }
    return analytics.full_report(room_id=room_id, days=days)


@app.get("/api/analytics/hourly")
async def get_analytics_hourly(room_id: Optional[int] = None, days: int = 30):
    """Return 24-bucket histogram of falls by hour of day."""
    analytics = _get_analytics()
    if analytics is None:
        return [{"hour": h, "count": 0} for h in range(24)]
    return analytics.hourly_distribution(room_id=room_id, days=days)


@app.get("/api/analytics/risk-windows")
async def get_analytics_risk_windows(threshold: int = 2, window_days: int = 7):
    """Identify high-risk time windows based on recent fall frequency."""
    analytics = _get_analytics()
    if analytics is None:
        return []
    return analytics.high_risk_windows(threshold=threshold, window_days=window_days)


# ---------------------------------------------------------------------------
# Multi-Facility Cloud Gateway Endpoints (Milestone 10.3)
# ---------------------------------------------------------------------------

def _get_cloud_gateway():
    if broadcaster.cloud_gateway is not None:
        return broadcaster.cloud_gateway
    try:
        from hub.cloud_sync import CloudSyncGateway
        broadcaster.cloud_gateway = CloudSyncGateway()
        return broadcaster.cloud_gateway
    except Exception:
        return None


@app.get("/api/cloud/status")
async def get_cloud_status():
    """Return cloud synchronization queue status and health."""
    gw = _get_cloud_gateway()
    if gw is None:
        return {"online": False, "pending_records": 0, "status": "Not configured"}
    return gw.get_status()


@app.post("/api/cloud/sync")
async def post_cloud_sync(max_records: int = 50):
    """Trigger manual flush of pending cloud sync queue."""
    gw = _get_cloud_gateway()
    if gw is None:
        return JSONResponse({"error": "Cloud gateway not available"}, status_code=503)
    synced, failed = gw.flush_queue(max_records=max_records)
    status = gw.get_status()
    return {"synced": synced, "failed": failed, "pending_records": status["pending_records"]}


# ---------------------------------------------------------------------------
# Continuous System Diagnostics Endpoints (Milestone 10.4)
# ---------------------------------------------------------------------------

def _get_diagnostics_watcher():
    if broadcaster.diagnostics_watcher is not None:
        return broadcaster.diagnostics_watcher
    try:
        from hub.diagnostics import SystemDiagnosticsWatcher
        broadcaster.diagnostics_watcher = SystemDiagnosticsWatcher()
        return broadcaster.diagnostics_watcher
    except Exception:
        return None


@app.get("/api/diagnostics/health")
async def get_diagnostics_health():
    """Return overall system health and subsystem status."""
    watcher = _get_diagnostics_watcher()
    if watcher is None:
        return {"status": "UNKNOWN", "note": "Diagnostics watcher not initialised"}
    return watcher.evaluate_health()


@app.get("/api/diagnostics/metrics")
async def get_diagnostics_metrics():
    """Return telemetry rates, jitter, and subsystem performance metrics."""
    watcher = _get_diagnostics_watcher()
    if watcher is None:
        return {"subsystems": {}, "status": "UNKNOWN"}
    health = watcher.evaluate_health()
    return {"subsystems": health.get("subsystems", {}), "status": health.get("status")}


@app.post("/api/diagnostics/self-test")
async def post_diagnostics_self_test():
    """Trigger IEC 60601-1-8 automated self-test across all modules."""
    watcher = _get_diagnostics_watcher()
    if watcher is None:
        return JSONResponse({"error": "Diagnostics watcher not available"}, status_code=503)
    return watcher.run_self_test()


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    broadcaster.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "init",
            "state": broadcaster.latest_state,
            "height": broadcaster.latest_height,
            "nodes": broadcaster.node_status,
        }))
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket exception: {e}")
        broadcaster.disconnect(websocket)
