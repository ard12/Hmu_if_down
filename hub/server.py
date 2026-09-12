"""Central Hub Server for Wi-Fi CSI, mmWave Radar, and Dual-Sensor Fusion."""

import argparse
from collections import deque
import json
from pathlib import Path
import socket
import sys
import threading
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from hub.alert_dispatcher import AlertDispatcher
from hub.csi_pipeline.multi_link_fusion import MultiLinkFusionEngine
from hub.csi_pipeline.pca_features import CSIPCAExtractor
from hub.csi_pipeline.preprocessor import CSIPreprocessor
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.mmwave_pipeline.radar_receiver import RadarFallState, RadarPosture, RadarReceiver, RadarTelemetry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_SIZE = 100          # Rolling buffer depth (100 samples @ 100Hz = 1s)
EXTRACTION_INTERVAL = 10   # Run PCA/velocity every N packets (~100ms)
NODE_TIMEOUT_SEC = 10.0    # Mark node as dead if no heartbeat for this long


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Fall Detection Processing Hub")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["csi", "radar", "fusion"],
        default="csi",
        help="Operating mode: 'csi' (Plan 1), 'radar' (Plan 2), or 'fusion' (Dual Modality)",
    )
    parser.add_argument(
        "--csi-port",
        type=int,
        default=5555,
        help="UDP listening port for Tracker nodes CSI stream (default: 5555)",
    )
    parser.add_argument(
        "--radar-port",
        type=int,
        default=5556,
        help="UDP listening port for mmWave Radar gateway (default: 5556)",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audible alarm beeper",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run simulated live multi-link CSI and radar replay demo",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-Node Rolling Buffer with Sequence Tracking
# ---------------------------------------------------------------------------
class NodeBuffer:
    """Maintains a rolling window of CSI amplitudes for one tracker node,
    with sequence number gap detection and linear interpolation."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.amplitudes: deque = deque(maxlen=window_size)
        self.last_seq: int = -1
        self.last_seen: float = 0.0
        self.total_received: int = 0
        self.total_gaps: int = 0

    def push(self, packet) -> int:
        """Add a parsed CSI packet. Returns the number of interpolated frames inserted."""
        self.total_received += 1
        self.last_seen = time.time()
        interpolated = 0

        # Sequence gap detection and interpolation
        if self.last_seq >= 0 and packet.seq_num > self.last_seq + 1:
            gap = min(packet.seq_num - self.last_seq - 1, 5)  # Cap interpolation at 5 frames
            self.total_gaps += gap
            if len(self.amplitudes) > 0:
                last_amp = self.amplitudes[-1]
                for i in range(1, gap + 1):
                    # Linear interpolation between last known and current
                    alpha = i / (gap + 1)
                    interp = last_amp * (1 - alpha) + packet.amplitudes * alpha
                    self.amplitudes.append(interp)
                    interpolated += 1

        self.last_seq = packet.seq_num
        self.amplitudes.append(packet.amplitudes)
        return interpolated

    def get_window(self) -> np.ndarray:
        """Return the current rolling window as a (T, N_subcarrier) matrix."""
        if len(self.amplitudes) < 2:
            return np.empty((0, 0))
        return np.vstack(list(self.amplitudes))

    @property
    def is_alive(self) -> bool:
        return (time.time() - self.last_seen) < NODE_TIMEOUT_SEC if self.last_seen > 0 else False


# ---------------------------------------------------------------------------
# CSI Listener Thread
# ---------------------------------------------------------------------------
def csi_listener_thread(
    sock: socket.socket,
    preprocessor: CSIPreprocessor,
    pca_extractor: CSIPCAExtractor,
    fusion_engine: DualFusionEngine,
    node_buffers: dict,
    stop_event: threading.Event,
):
    """Receives UDP CSI packets, maintains per-node rolling buffers,
    and periodically runs PCA + velocity extraction → fusion engine."""

    packet_count = 0

    while not stop_event.is_set():
        try:
            sock.settimeout(1.0)
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break

        # Check for heartbeat JSON packets from tracker nodes
        if data.startswith(b"{"):
            try:
                hb = json.loads(data.decode("utf-8", errors="ignore"))
                if "heartbeat" in hb:
                    node_id = hb["heartbeat"]
                    if node_id in node_buffers:
                        node_buffers[node_id].last_seen = time.time()
                    continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        packet = preprocessor.parse_packet(data)
        if packet is None:
            continue

        node_id = packet.node_id
        if node_id not in node_buffers:
            node_buffers[node_id] = NodeBuffer()

        buf = node_buffers[node_id]
        interpolated = buf.push(packet)
        packet_count += 1

        # Run feature extraction every EXTRACTION_INTERVAL packets per node
        if buf.total_received % EXTRACTION_INTERVAL == 0:
            window = buf.get_window()
            if window.shape[0] >= 32:
                filtered = preprocessor.filter_stream(window)
                features = pca_extractor.extract(node_id=node_id, filtered_data=filtered)
                state = fusion_engine.update_csi(features)

                # Print live status
                status_sym = "🟢" if state == UnifiedFallState.NORMAL else (
                    "🟡" if state == UnifiedFallState.SUSPECTED else "🔴"
                )
                alive_nodes = [nid for nid, nb in node_buffers.items() if nb.is_alive]
                print(f"\r[CSI] Node {node_id} | v={features.dominant_velocity_mps:.2f} m/s | "
                      f"surge={features.energy_surge_ratio:.1f} | "
                      f"{status_sym} {state.value:<15} | "
                      f"Active nodes: {alive_nodes} | "
                      f"Gaps: {buf.total_gaps}",
                      end="", flush=True)


# ---------------------------------------------------------------------------
# Radar Listener Thread
# ---------------------------------------------------------------------------
def radar_listener_thread(
    sock: socket.socket,
    radar_receiver: RadarReceiver,
    fusion_engine: DualFusionEngine,
    stop_event: threading.Event,
):
    """Receives UDP radar datagrams (JSON or binary) and feeds the fusion engine."""

    while not stop_event.is_set():
        try:
            sock.settimeout(1.0)
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break

        # Try JSON first (from ESP32 gateway), then binary
        try:
            payload = data.decode("utf-8", errors="ignore").strip()
            telemetry = radar_receiver.parse_json_datagram(payload)
        except Exception:
            telemetry = None

        if telemetry is None:
            telemetry = radar_receiver.parse_binary_frame(data)

        if telemetry is not None:
            state = fusion_engine.update_radar(telemetry)
            status_sym = "🟢" if state == UnifiedFallState.NORMAL else (
                "🟡" if state == UnifiedFallState.SUSPECTED else "🔴"
            )
            print(f"\n[RADAR] Height={telemetry.target_height_m:.2f}m | "
                  f"Posture={telemetry.posture.value} | "
                  f"Fall={telemetry.fall_state.value} | "
                  f"{status_sym} {state.value}",
                  flush=True)


# ---------------------------------------------------------------------------
# Node Health Monitor
# ---------------------------------------------------------------------------
def node_health_monitor(
    node_buffers: dict,
    fusion_engine: DualFusionEngine,
    stop_event: threading.Event,
):
    """Periodically checks which nodes are alive and adjusts fusion parameters."""
    while not stop_event.is_set():
        time.sleep(5.0)
        alive = [nid for nid, nb in node_buffers.items() if nb.is_alive]
        dead = [nid for nid, nb in node_buffers.items() if not nb.is_alive and nb.total_received > 0]

        if dead:
            print(f"\n[HEALTH] ⚠️  Dead nodes: {dead} | Alive: {alive}", flush=True)

        # Graceful degradation: reduce min_coincident_links if nodes drop
        csi_engine = fusion_engine.csi_engine
        if len(alive) <= 1 and csi_engine.min_links > 1:
            csi_engine.min_links = 1
            print(f"\n[HEALTH] ⚠️  Only {len(alive)} node(s) alive — "
                  f"reduced coincidence threshold to 1 link", flush=True)
        elif len(alive) >= 2 and csi_engine.min_links < 2:
            csi_engine.min_links = 2


# ---------------------------------------------------------------------------
# Demo Mode
# ---------------------------------------------------------------------------
def run_demo(fusion_engine: DualFusionEngine):
    """Simulates real-time CSI packets from 3 links and mmWave radar data."""
    print("\n" + "=" * 65)
    print("      STARTING LIVE MULTI-MODAL FALL DETECTION DEMO")
    print("=" * 65)
    print("Simulating 4-Node Wi-Fi CSI (1 AP + 3 Trackers) and 60GHz mmWave Radar...\n")

    preprocessor = CSIPreprocessor()
    pca_extractor = CSIPCAExtractor(velocity_threshold_mps=1.8)

    # Simulation timeline:
    # 0s - 3s: Normal standing / walking (moderate movement)
    # 3.2s: Sudden Fall event (simultaneous high Doppler on Links 1 & 2)
    # 3.5s - 8s: Post-fall stillness on the floor (low variance, target height = 0.22m)

    sim_start = time.time()

    for step in range(80):
        now = time.time()
        elapsed = now - sim_start

        # Determine simulated state
        is_falling = (3.0 <= elapsed <= 3.6)
        is_lying_on_floor = (elapsed > 3.6)

        print(f"\r[T={elapsed:4.1f}s] ", end="")

        for node_id in (1, 2, 3):
            # Generate 64 subcarriers of simulated CSI amplitudes (100 samples = 1s @ 100Hz)
            if is_falling and node_id in (1, 2):
                # Fall: Fast downward Doppler oscillation at ~32 Hz → v = λ*32/2 ≈ 1.97 m/s
                t_arr = np.linspace(0, 1.0, 100)
                fast_wave = np.sin(2 * np.pi * 32.0 * t_arr)[:, None]
                sim_amplitudes = np.ones((100, 64)) * 30.0 + fast_wave * 50.0 + np.random.normal(0, 2.0, (100, 64))
            elif is_lying_on_floor:
                # Floor stillness: very low variance
                sim_amplitudes = np.ones((100, 64)) * 15.0 + np.random.normal(0, 0.05, (100, 64))
            else:
                # Normal walking: low frequency (3.5 Hz) moderate oscillation
                t_arr = np.linspace(0, 1.0, 100)
                walk_wave = np.sin(2 * np.pi * 3.5 * t_arr)[:, None]
                sim_amplitudes = np.ones((100, 64)) * 25.0 + walk_wave * 8.0 + np.random.normal(0, 1.0, (100, 64))

            filtered = preprocessor.filter_stream(sim_amplitudes)
            features = pca_extractor.extract(node_id=node_id, filtered_data=filtered)
            state = fusion_engine.update_csi(features, current_time=now)

        # Simultaneously update mmWave radar simulation
        if is_lying_on_floor:
            radar_telemetry = RadarTelemetry(
                fall_state=RadarFallState.CONFIRMED if elapsed > 4.5 else RadarFallState.SUSPECTED,
                posture=RadarPosture.LYING,
                target_height_m=0.22,
                dwell_time_sec=int(elapsed - 3.6),
            )
        else:
            radar_telemetry = RadarTelemetry(
                fall_state=RadarFallState.NONE,
                posture=RadarPosture.STANDING,
                target_height_m=1.65,
                dwell_time_sec=0,
            )

        fusion_state = fusion_engine.update_radar(radar_telemetry)

        # Print real-time status HUD
        status_color = "\033[92m" if fusion_state == UnifiedFallState.NORMAL else (
            "\033[93m" if fusion_state == UnifiedFallState.SUSPECTED else "\033[91m"
        )
        print(f"Status: {status_color}{fusion_state.value:<15}\033[0m | "
              f"Radar Alt: {radar_telemetry.target_height_m:.2f}m | "
              f"Post-Fall Dwell: {radar_telemetry.dwell_time_sec}s", end="", flush=True)

        time.sleep(0.1)

    print("\n\n[Demo Completed] Successfully demonstrated multi-link coincidence and floor altitude confirmation.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    mode = OperatingMode(args.mode)
    alert = AlertDispatcher(enable_sound=not args.no_sound)
    fusion_engine = DualFusionEngine(mode=mode, alert_dispatcher=alert)

    if args.demo:
        run_demo(fusion_engine)
        return

    print("=" * 60)
    print(f"  FALL DETECTION HUB RUNNING IN [{mode.value.upper()}] MODE")
    print("=" * 60)

    stop_event = threading.Event()
    threads = []
    sockets = []

    # Shared state
    preprocessor = CSIPreprocessor()
    pca_extractor = CSIPCAExtractor()
    radar_receiver = RadarReceiver()
    node_buffers = {1: NodeBuffer(), 2: NodeBuffer(), 3: NodeBuffer()}

    # CSI Listener
    if mode in (OperatingMode.CSI_ONLY, OperatingMode.FUSION):
        csi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csi_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        csi_sock.bind(("0.0.0.0", args.csi_port))
        sockets.append(csi_sock)
        print(f"  [CSI]   Listening on UDP port {args.csi_port}")

        t = threading.Thread(
            target=csi_listener_thread,
            args=(csi_sock, preprocessor, pca_extractor, fusion_engine, node_buffers, stop_event),
            daemon=True,
        )
        threads.append(t)

    # Radar Listener
    if mode in (OperatingMode.RADAR_ONLY, OperatingMode.FUSION):
        radar_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        radar_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        radar_sock.bind(("0.0.0.0", args.radar_port))
        sockets.append(radar_sock)
        print(f"  [RADAR] Listening on UDP port {args.radar_port}")

        t = threading.Thread(
            target=radar_listener_thread,
            args=(radar_sock, radar_receiver, fusion_engine, stop_event),
            daemon=True,
        )
        threads.append(t)

    # Node health monitor
    health_t = threading.Thread(
        target=node_health_monitor,
        args=(node_buffers, fusion_engine, stop_event),
        daemon=True,
    )
    threads.append(health_t)

    # Start all threads
    for t in threads:
        t.start()

    print(f"\nPress Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Shutting down] Server stopped by user.")
    finally:
        stop_event.set()
        for s in sockets:
            try:
                s.close()
            except Exception:
                pass
        for t in threads:
            t.join(timeout=2.0)


if __name__ == "__main__":
    main()

