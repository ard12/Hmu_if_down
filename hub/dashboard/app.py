"""Real-Time Telemetry Web Dashboard Server (FastAPI + WebSockets)."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import csv
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
