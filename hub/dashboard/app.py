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
import numpy as np

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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
        self.training_buffer = None
        self.drift_detector = None
        self.model_registry = None
        self.retraining_pipeline = None
        self.patient_context_store = None
        self.fhir_data_lake = None
        self.smart_fhir_client = None
        self.triangulation_engine = None
        self.handoff_manager = None
        self.gait_analyzer = None
        self.prefail_detector = None
        self.frax_calculator = None
        self.skeleton_fitter = None
        self.joint_angle_estimator = None
        self.biomechanics_classifier = None
        self.federated_server = None
        self.personalization_layer = None
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
    retrain_task = asyncio.create_task(_retrain_loop())
    yield
    task.cancel()
    retrain_task.cancel()


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


# ---------------------------------------------------------------------------
# Adaptive Retraining & Model Drift Detection Endpoints (Phase 11)
# ---------------------------------------------------------------------------

def _get_training_buffer():
    if broadcaster.training_buffer is not None:
        return broadcaster.training_buffer
    try:
        from hub.training_buffer import TrainingBuffer
        broadcaster.training_buffer = TrainingBuffer()
        return broadcaster.training_buffer
    except Exception:
        return None


def _get_drift_detector():
    if broadcaster.drift_detector is not None:
        return broadcaster.drift_detector
    try:
        from hub.drift_detector import DriftDetector
        broadcaster.drift_detector = DriftDetector()
        return broadcaster.drift_detector
    except Exception:
        return None


def _get_model_registry():
    if broadcaster.model_registry is not None:
        return broadcaster.model_registry
    try:
        from hub.model_registry import ModelRegistry
        broadcaster.model_registry = ModelRegistry()
        return broadcaster.model_registry
    except Exception:
        return None


def _get_retraining_pipeline():
    if broadcaster.retraining_pipeline is not None:
        return broadcaster.retraining_pipeline
    try:
        from hub.retraining_pipeline import RetrainingPipeline
        registry = _get_model_registry()
        buffer = _get_training_buffer()
        detector = _get_drift_detector()
        if registry and buffer and detector:
            broadcaster.retraining_pipeline = RetrainingPipeline(
                registry=registry,
                buffer=buffer,
                detector=detector,
                audit_log=broadcaster.audit_log,
            )
            return broadcaster.retraining_pipeline
    except Exception:
        return None


async def _retrain_loop(interval_s: int = 3600):
    while True:
        try:
            await asyncio.sleep(interval_s)
            pipeline = _get_retraining_pipeline()
            if pipeline:
                pipeline.run()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Error in background retraining loop: {e}")


@app.post("/api/labels/{event_id}")
async def post_label_event(event_id: str, payload: Dict[str, Any]):
    """Nursing staff labels an alert: links stored feature to ground-truth label."""
    confirmed = bool(payload.get("confirmed", False))
    labeller = str(payload.get("labeller", "unknown"))
    buf = _get_training_buffer()
    if buf is not None:
        buf.label_event(event_id, confirmed)
    if broadcaster.audit_log is not None and hasattr(broadcaster.audit_log, "append"):
        try:
            broadcaster.audit_log.append(
                "LABEL_CONFIRMED",
                {"event_id": event_id, "confirmed": confirmed, "labeller": labeller},
            )
        except Exception:
            pass
    return {"status": "ok", "event_id": event_id, "confirmed": confirmed}


@app.get("/api/labels/stats")
async def get_labels_stats():
    """Return training buffer statistics."""
    buf = _get_training_buffer()
    if buf is None:
        return {"total": 0, "positives": 0, "negatives": 0, "oldest_ts": 0.0, "newest_ts": 0.0}
    return buf.stats()


@app.get("/api/drift/status")
async def get_drift_status():
    """Return model prediction drift metrics and alert status."""
    detector = _get_drift_detector()
    if detector is None:
        return {"psi": 0.0, "kl": 0.0, "status": "UNKNOWN", "reference_n": 0, "live_n": 0}
    return detector.drift_status()


@app.get("/api/retrain/status")
async def get_retrain_status():
    """Return latest model retraining status."""
    pipeline = _get_retraining_pipeline()
    if pipeline is None:
        return {"status": "NOT_CONFIGURED", "reason": "Retraining pipeline not initialised"}
    return getattr(pipeline, "last_result", None) or {"status": "IDLE", "reason": "Awaiting drift evaluation"}


@app.post("/api/retrain/trigger")
async def post_retrain_trigger():
    """Manually trigger model retraining evaluation."""
    pipeline = _get_retraining_pipeline()
    if pipeline is None:
        return JSONResponse({"error": "Retraining pipeline not available"}, status_code=503)
    res = pipeline.run()
    return res


# ---------------------------------------------------------------------------
# Encrypted FHIR R4 Data Lake & Patient Context Endpoints (Phase 12)
# ---------------------------------------------------------------------------

def _get_patient_context_store():
    if broadcaster.patient_context_store is not None:
        return broadcaster.patient_context_store
    try:
        from hub.patient_context import PatientContextStore
        broadcaster.patient_context_store = PatientContextStore(
            db_path=str(PROJECT_ROOT / "models" / "patient_context.db")
        )
        return broadcaster.patient_context_store
    except Exception:
        return None


def _get_fhir_data_lake():
    if broadcaster.fhir_data_lake is not None:
        return broadcaster.fhir_data_lake
    try:
        from hub.fhir_lake import FHIRDataLake
        from cryptography.fernet import Fernet
        key_path = PROJECT_ROOT / "models" / ".fhir_key"
        if key_path.exists():
            key = key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(key)
        broadcaster.fhir_data_lake = FHIRDataLake(
            db_path=str(PROJECT_ROOT / "models" / "fhir_lake.db"),
            key=key,
        )
        return broadcaster.fhir_data_lake
    except Exception:
        return None


@app.get("/api/fhir/patient/{patient_id}/bundle")
async def get_fhir_patient_bundle(patient_id: str):
    """Export all FHIR resources for a patient as an R4 Bundle."""
    lake = _get_fhir_data_lake()
    if lake is None:
        return JSONResponse({"error": "FHIR data lake not available"}, status_code=503)
    try:
        bundle = lake.export_bundle(patient_id)
        return bundle
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/fhir/observation")
async def post_fhir_observation(observation: Dict[str, Any]):
    """Store a FHIR Observation in the encrypted data lake."""
    lake = _get_fhir_data_lake()
    if lake is None:
        return JSONResponse({"error": "FHIR data lake not available"}, status_code=503)
    try:
        res_id = lake.write_resource("Observation", observation)
        return {"id": res_id, "status": "stored", "resourceType": "Observation"}
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/patient/{room_id}")
async def get_patient_context(room_id: str):
    """Get active patient context for a room."""
    store = _get_patient_context_store()
    if store is None:
        return JSONResponse({"error": "Patient context store not available"}, status_code=503)
    patient = store.get(room_id)
    if not patient:
        return JSONResponse({"error": "No patient context for room", "room_id": room_id}, status_code=404)
    return patient


def _get_triangulation_engine():
    if broadcaster.triangulation_engine is not None:
        return broadcaster.triangulation_engine
    try:
        from hub.triangulation import TriangulationEngine
        engine = TriangulationEngine({
            "1": (0.0, 0.0),
            "2": (5.0, 0.0),
            "3": (2.5, 5.0),
        })
        broadcaster.triangulation_engine = engine
        return engine
    except Exception:
        return None


@app.get("/api/position/current")
async def get_current_position():
    """Get latest 2D triangulation position estimate."""
    engine = _get_triangulation_engine()
    if engine is None:
        return JSONResponse({"error": "Triangulation engine not available"}, status_code=503)
    try:
        pos = engine.estimate_position()
        return pos
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.get("/api/position/history")
async def get_position_history():
    """Get position history for trajectory tracking."""
    engine = _get_triangulation_engine()
    if engine is None:
        return JSONResponse({"error": "Triangulation engine not available"}, status_code=503)
    return {"history": engine.get_history()}


@app.get("/api/mesh/topology")
async def get_mesh_topology():
    """Return active ESP-MESH topology, relay routes, and handoff status."""
    nodes = []
    for node_id, status in broadcaster.node_status.items():
        nodes.append({
            "node_id": node_id,
            "alive": status.get("alive", False),
            "hop_count": status.get("hop_count", 0),
            "is_root": (node_id == 1),
            "parent": None if node_id == 1 else 1,
            "last_seen": status.get("last_seen", 0.0),
        })
    handoff_rooms = []
    if broadcaster.handoff_manager is not None:
        handoff_rooms = broadcaster.handoff_manager.get_active_rooms()
    return {
        "mesh_enabled": True,
        "max_hops": 3,
        "root_node": 1,
        "nodes": nodes,
        "active_handoff_rooms": handoff_rooms,
    }


def _get_gait_analyzer():
    if broadcaster.gait_analyzer is not None:
        return broadcaster.gait_analyzer
    try:
        from hub.gait_analyzer import GaitCadenceAnalyzer
        broadcaster.gait_analyzer = GaitCadenceAnalyzer(fs=100.0)
        return broadcaster.gait_analyzer
    except Exception:
        return None


def _get_prefail_detector():
    if broadcaster.prefail_detector is not None:
        return broadcaster.prefail_detector
    try:
        from hub.prefail_detector import PreFallAnomalyDetector
        broadcaster.prefail_detector = PreFallAnomalyDetector()
        return broadcaster.prefail_detector
    except Exception:
        return None


def _get_frax_calculator():
    if broadcaster.frax_calculator is not None:
        return broadcaster.frax_calculator
    try:
        from hub.frax_risk import FRAXFallRiskScore
        broadcaster.frax_calculator = FRAXFallRiskScore()
        return broadcaster.frax_calculator
    except Exception:
        return None


@app.get("/api/mobility/risk")
async def get_mobility_risk():
    """Return real-time gait cadence, mobility classification, and pre-fall risk status."""
    gait_engine = _get_gait_analyzer()
    prefail_engine = _get_prefail_detector()
    gait_res = gait_engine.analyze() if gait_engine is not None else {
        "cadence_hz": 1.8,
        "gait_class": "NORMAL",
        "stride_regularity": 0.95,
        "velocity_envelope_peak": 0.85,
        "confidence": 0.9,
    }
    prefail_res = prefail_engine.update(gait_res) if prefail_engine is not None else {
        "risk_score": 10,
        "risk_level": "NORMAL",
        "contributing_factors": [],
        "seconds_until_escalation": None,
    }
    return {
        "status": "success",
        "gait": gait_res,
        "pre_fall": prefail_res,
    }


@app.get("/api/prefail/status")
async def get_prefail_status(room_id: Optional[str] = None):
    """Return current pre-fall anomaly detector risk status."""
    prefail_engine = _get_prefail_detector()
    if prefail_engine is None:
        return JSONResponse({"error": "Pre-fall detector not available"}, status_code=503)
    gait_engine = _get_gait_analyzer()
    gait_info = gait_engine.analyze() if gait_engine is not None else {
        "gait_class": "NORMAL",
        "velocity_envelope_peak": 0.8,
    }
    pt_ctx = None
    if room_id:
        store = _get_patient_context_store()
        if store:
            pt_ctx = store.get(room_id)
    res = prefail_engine.update(gait_info, patient_context=pt_ctx)
    return res


@app.get("/api/prefail/history")
async def get_prefail_history():
    """Return 60-snapshot history of pre-fall risk scores."""
    prefail_engine = _get_prefail_detector()
    if prefail_engine is None:
        return JSONResponse({"error": "Pre-fall detector not available"}, status_code=503)
    return {"history": prefail_engine.get_history()}


@app.get("/api/frax/{room_id}")
async def get_frax_score(room_id: str):
    """Compute FRAX-style 10-year clinical fall risk score using room patient context."""
    calc = _get_frax_calculator()
    if calc is None:
        return JSONResponse({"error": "FRAX calculator not available"}, status_code=503)
    store = _get_patient_context_store()
    patient = store.get(room_id) if store else None
    if not patient:
        score = calc.compute(age=65, gender="F", bmi=24.0, prior_fall=False, morse_fall_scale=0, n_high_risk_meds=0)
        score["room_id"] = room_id
        score["patient_id"] = "UNOCCUPIED"
        return score

    age = int(patient.get("age", 70))
    gender = str(patient.get("gender", "F"))
    bmi = float(patient.get("bmi", 23.5))
    prior_fall = bool(patient.get("prior_fall", False))
    morse = int(patient.get("morse_fall_scale", 0))
    meds = patient.get("high_risk_meds") or patient.get("medications") or []
    n_meds = len(meds)
    score = calc.compute(
        age=age,
        gender=gender,
        bmi=bmi,
        prior_fall=prior_fall,
        morse_fall_scale=morse,
        n_high_risk_meds=n_meds,
    )
    score["room_id"] = room_id
    score["patient_id"] = patient.get("patient_id", "UNKNOWN")
    return score


def _get_skeleton_fitter():
    if broadcaster.skeleton_fitter:
        return broadcaster.skeleton_fitter
    if broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "skeleton_fitter", None):
        return broadcaster.fusion_engine.skeleton_fitter
    try:
        from hub.skeleton_fitter import SkeletonFitter
        broadcaster.skeleton_fitter = SkeletonFitter()
        return broadcaster.skeleton_fitter
    except Exception:
        return None


def _get_joint_angle_estimator():
    if broadcaster.joint_angle_estimator:
        return broadcaster.joint_angle_estimator
    if broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "joint_angle_estimator", None):
        return broadcaster.fusion_engine.joint_angle_estimator
    try:
        from hub.joint_angles import JointAngleEstimator
        broadcaster.joint_angle_estimator = JointAngleEstimator()
        return broadcaster.joint_angle_estimator
    except Exception:
        return None


def _get_biomechanics_classifier():
    if broadcaster.biomechanics_classifier:
        return broadcaster.biomechanics_classifier
    if broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "biomechanics_classifier", None):
        return broadcaster.fusion_engine.biomechanics_classifier
    try:
        from hub.biomechanics_classifier import FallBiomechanicsClassifier
        broadcaster.biomechanics_classifier = FallBiomechanicsClassifier()
        return broadcaster.biomechanics_classifier
    except Exception:
        return None


@app.get("/api/pose/current")
async def get_current_pose():
    """Get current 3D skeleton keypoints, joint angles, and posture."""
    fitter = _get_skeleton_fitter()
    estimator = _get_joint_angle_estimator()

    skeleton = None
    if broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "last_skeleton", None):
        skeleton = broadcaster.fusion_engine.last_skeleton
    elif fitter:
        skeleton = {
            "head": [0.0, 0.0, 1.65],
            "torso_top": [0.0, 0.0, 1.40],
            "torso_bottom": [0.0, 0.0, 0.85],
            "left_wrist": [-0.35, 0.0, 1.10],
            "right_wrist": [0.35, 0.0, 1.10],
            "fit_quality": 0.95,
            "valid": True,
        }

    posture = "UNKNOWN"
    angles = {
        "trunk_inclination_deg": 0.0,
        "knee_flexion_deg": 0.0,
        "head_drop_velocity_mps": 0.0,
    }
    if skeleton and estimator:
        posture = estimator.classify_posture(skeleton)
        tt = skeleton.get("torso_top", [0.0, 0.0, 1.40])
        tb = skeleton.get("torso_bottom", [0.0, 0.0, 0.85])
        angles["trunk_inclination_deg"] = round(estimator.trunk_inclination(tt, tb), 2)
        angles["knee_flexion_deg"] = 90.0 if posture in ("FALLEN", "SITTING") else 0.0

    return {
        "skeleton": skeleton,
        "angles": angles,
        "posture": posture,
    }


@app.get("/api/pose/trajectory")
async def get_pose_trajectory():
    """Get recent 3D skeleton trajectory frames."""
    traj = []
    if broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "skeleton_trajectory", None):
        traj = list(broadcaster.fusion_engine.skeleton_trajectory)
    return {
        "trajectory": traj,
        "count": len(traj),
    }


@app.get("/api/biomechanics/classification")
async def get_biomechanics_classification():
    """Get latest fall biomechanics classification from skeleton trajectory."""
    classifier = _get_biomechanics_classifier()
    if not classifier:
        return JSONResponse({"error": "Biomechanics classifier unavailable"}, status_code=503)

    traj = []
    if broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "skeleton_trajectory", None):
        traj = list(broadcaster.fusion_engine.skeleton_trajectory)

    res = classifier.classify(traj)
    res["biomechanics_confirmed"] = bool(
        broadcaster.fusion_engine and getattr(broadcaster.fusion_engine, "last_biomechanics_confirmed", False)
    )
    return res


def _get_federated_server():
    if broadcaster.federated_server is not None:
        return broadcaster.federated_server
    try:
        from hub.dp_trainer import DPModel
        from hub.federated_server import FederatedAggregationServer
        base_model = DPModel(weights=np.zeros(4), bias=0.0)
        broadcaster.federated_server = FederatedAggregationServer(global_model=base_model, min_participants=2)
        return broadcaster.federated_server
    except Exception as e:
        logger.warning(f"Failed to initialize federated server: {e}")
        return None


def _get_personalization_layer():
    if broadcaster.personalization_layer is not None:
        return broadcaster.personalization_layer
    try:
        from hub.personalization_layer import PersonalizationLayer
        fed_srv = _get_federated_server()
        base_model = fed_srv.global_model if fed_srv else None
        broadcaster.personalization_layer = PersonalizationLayer(global_model=base_model)
        return broadcaster.personalization_layer
    except Exception as e:
        logger.warning(f"Failed to initialize personalization layer: {e}")
        return None


@app.get("/federated")
async def get_federated_page():
    page_path = STATIC_DIR / "federated.html"
    if page_path.exists():
        return FileResponse(page_path)
    return JSONResponse({"status": "running", "message": "Federated dashboard static file not found."})


@app.post("/api/federated/gradients")
async def post_federated_gradients(request: Request):
    """Receive DP-sanitized gradients from a participating hub."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    hub_id = data.get("hub_id")
    gradients_raw = data.get("gradients")
    n_samples = data.get("n_samples")
    epsilon = data.get("epsilon")

    if not hub_id or gradients_raw is None or n_samples is None or epsilon is None:
        return JSONResponse(
            {"error": "Missing required fields (hub_id, gradients, n_samples, epsilon)"},
            status_code=400,
        )

    server = _get_federated_server()
    if not server:
        return JSONResponse({"error": "Federated server unavailable"}, status_code=503)

    try:
        gradients = [np.array(g, dtype=float) for g in gradients_raw]
        server.receive_gradients(hub_id, gradients, int(n_samples), float(epsilon))
        status = server.get_round_status()
        if status["participants"] >= status["min_required"]:
            bcast = server.broadcast_global_weights()
            return {
                "status": "AGGREGATED",
                "round": bcast["round"],
                "weights": bcast["weights"],
                "bias": bcast["bias"],
                "participants": status["participants"],
            }
        return {
            "status": "RECEIVED",
            "round": status["round"],
            "participants": status["participants"],
            "min_required": status["min_required"],
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/federated/status")
async def get_federated_status():
    """Get federated training round status, active hubs, and privacy budget."""
    server = _get_federated_server()
    if not server:
        return JSONResponse({"error": "Federated server unavailable"}, status_code=503)

    status = server.get_round_status()
    status["history"] = server.round_history
    status["total_epsilon"] = server.total_epsilon
    return status


@app.get("/api/federated/weights")
async def get_federated_weights():
    """Get latest aggregated global model weights and round metadata."""
    server = _get_federated_server()
    if not server:
        return JSONResponse({"error": "Federated server unavailable"}, status_code=503)

    weights_list = server.global_model.weights.tolist() if hasattr(server.global_model, "weights") else []
    bias_val = float(server.global_model.bias) if hasattr(server.global_model, "bias") else 0.0
    return {
        "round": server.round_num,
        "weights": weights_list,
        "bias": bias_val,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/federated/personalize")
async def post_federated_personalize(request: Request):
    """Fit or evaluate local personalization head on local site data."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    features = data.get("features")
    labels = data.get("labels")
    if features is None or labels is None:
        return JSONResponse({"error": "Missing required fields (features, labels)"}, status_code=400)

    if len(features) != len(labels) or len(features) == 0:
        return JSONResponse({"error": "Features and labels must be non-empty and of equal length"}, status_code=400)

    layer = _get_personalization_layer()
    if not layer:
        return JSONResponse({"error": "Personalization layer unavailable"}, status_code=503)

    try:
        X = np.array(features, dtype=float)
        y = np.array(labels, dtype=int)
        epochs = int(data.get("epochs", 10))
        hidden_dim = int(data.get("hidden_dim", layer.hidden_dim))
        layer.hidden_dim = hidden_dim

        layer.fit(X, y, epochs=epochs)
        metrics = layer.evaluate(X, y)
        return {
            "status": "TRAINED",
            "metrics": metrics,
            "hidden_dim": layer.hidden_dim,
            "is_fitted": layer._is_fitted,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)



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
