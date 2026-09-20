"""Central Hub Server for Wi-Fi CSI, mmWave Radar, and Dual-Sensor Fusion."""

import argparse
from collections import deque
import json
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from hub.alert_dispatcher import AlertDispatcher
from hub.csi_pipeline.classifier import FallClassifier
from hub.csi_pipeline.multi_link_fusion import MultiLinkFusionEngine
from hub.csi_pipeline.pca_features import CSIPCAExtractor
from hub.csi_pipeline.preprocessor import CSIPreprocessor
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.ha_discovery import HomeAssistantMQTTDiscoveryManager
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
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch real-time web telemetry dashboard HUD",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8000,
        help="Listening port for web dashboard HUD (default: 8000)",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Replay recorded multimodal session (.npz) through detection pipeline",
    )
    parser.add_argument(
        "--mqtt-broker",
        type=str,
        default=None,
        help="MQTT Broker host for alerting and Home Assistant integration",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT Broker port (default: 1883)",
    )
    parser.add_argument(
        "--ha-discovery",
        action="store_true",
        help="Announce Home Assistant MQTT Auto-Discovery entities on startup",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=None,
        help="Path to trained FallClassifier model (.pkl)",
    )
    parser.add_argument(
        "--multi-room",
        action="store_true",
        help="Enable multi-room spatial mesh mode using RoomManager",
    )
    parser.add_argument(
        "--relay-port",
        type=int,
        default=None,
        help="Listening port for secondary hub UDP relay client (e.g. 9200)",
    )
    parser.add_argument(
        "--fhir-base-url",
        type=str,
        default=None,
        help="SMART-on-FHIR server base URL",
    )
    parser.add_argument(
        "--fhir-client-id",
        type=str,
        default=None,
        help="SMART-on-FHIR OAuth2 client ID",
    )
    parser.add_argument(
        "--fhir-client-secret",
        type=str,
        default=None,
        help="SMART-on-FHIR OAuth2 client secret",
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
        if self.last_seq >= 0:
            if packet.seq_num > self.last_seq + 1:
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
            elif packet.seq_num == self.last_seq + 1:
                self.last_seq = packet.seq_num
            else:  # packet.seq_num <= self.last_seq
                if self.last_seq - packet.seq_num > 1000:
                    # Tracker node rebooted: reset sequence baseline
                    self.last_seq = packet.seq_num
                else:
                    # Late out-of-order or duplicate packet: drop to preserve chronological order
                    return 0
        else:
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
    classifier: Optional[FallClassifier] = None,
    ha_manager: Optional[HomeAssistantMQTTDiscoveryManager] = None,
    room_manager: Optional[Any] = None,
):
    """Receives UDP CSI packets, maintains per-node rolling buffers,
    and periodically runs PCA + velocity extraction → fusion engine / room manager."""

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
        room_id = getattr(packet, "room_id", 0)
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

                ml_prob = 0.0
                if classifier:
                    ml_feat = classifier.extractor.extract_from_window(filtered)
                    ml_prob = classifier.predict_proba(ml_feat)

                if room_manager is not None:
                    state_val = room_manager.on_csi_packet(room_id, features, ml_prob=ml_prob)
                    try:
                        state = UnifiedFallState(state_val)
                    except (ValueError, TypeError):
                        state = UnifiedFallState.NORMAL
                else:
                    state = fusion_engine.update_csi(features, ml_prob=ml_prob)

                # Broadcast to Web HUD
                try:
                    from hub.dashboard.app import broadcaster
                    psd = features.doppler_psd.tolist() if hasattr(features, "doppler_psd") and features.doppler_psd is not None else None
                    broadcaster.update_csi(
                        node_id=node_id,
                        velocity=features.dominant_velocity_mps,
                        surge=features.energy_surge_ratio,
                        state=state.value,
                        doppler_psd=psd,
                    )
                except Exception:
                    pass

                # Publish to Home Assistant MQTT
                if ha_manager:
                    try:
                        ha_manager.publish_csi(
                            node_id=node_id,
                            velocity=features.dominant_velocity_mps,
                            surge=features.energy_surge_ratio,
                            state=state.value,
                        )
                        ha_manager.publish_ml_probability(ml_prob)
                        ha_manager.publish_state(state.value)
                    except Exception:
                        pass

                # Print live status
                status_sym = "[OK]" if state == UnifiedFallState.NORMAL else (
                    "[SUSPECT]" if state == UnifiedFallState.SUSPECTED else "[FALL]"
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
    ha_manager: Optional[HomeAssistantMQTTDiscoveryManager] = None,
    room_manager: Optional[Any] = None,
):
    """Receives UDP radar datagrams (JSON or binary) and feeds the fusion engine / room manager."""

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
            if room_manager is not None:
                state_val = room_manager.on_radar_packet(0, telemetry)
                try:
                    state = UnifiedFallState(state_val)
                except (ValueError, TypeError):
                    state = UnifiedFallState.NORMAL
            else:
                state = fusion_engine.update_radar(telemetry)

            # Broadcast to Web HUD
            try:
                from hub.dashboard.app import broadcaster
                broadcaster.update_radar(
                    height=telemetry.target_height_m,
                    posture=telemetry.posture.value,
                    fall_state=telemetry.fall_state.value,
                    dwell=telemetry.dwell_time_sec,
                    state=state.value,
                )
            except Exception:
                pass

            # Publish to Home Assistant MQTT
            if ha_manager:
                try:
                    ha_manager.publish_radar(
                        height=telemetry.target_height_m,
                        posture=telemetry.posture.value,
                        fall_state=telemetry.fall_state.value,
                        dwell=telemetry.dwell_time_sec,
                    )
                    ha_manager.publish_state(state.value)
                except Exception:
                    pass

            status_sym = "[OK]" if state == UnifiedFallState.NORMAL else (
                "[SUSPECT]" if state == UnifiedFallState.SUSPECTED else "[FALL]"
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
def run_demo(
    fusion_engine: DualFusionEngine,
    is_web: bool = False,
    classifier: Optional[FallClassifier] = None,
    ha_manager: Optional[HomeAssistantMQTTDiscoveryManager] = None,
):
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

            ml_prob = 0.0
            if classifier:
                ml_feat = classifier.extractor.extract_from_window(filtered)
                ml_prob = classifier.predict_proba(ml_feat)

            state = fusion_engine.update_csi(features, current_time=now, ml_prob=ml_prob)

            if ha_manager:
                try:
                    ha_manager.publish_csi(
                        node_id=node_id,
                        velocity=features.dominant_velocity_mps,
                        surge=features.energy_surge_ratio,
                        state=state.value,
                    )
                    ha_manager.publish_ml_probability(ml_prob)
                except Exception:
                    pass

            if is_web:
                try:
                    from hub.dashboard.app import broadcaster
                    psd = features.doppler_psd.tolist() if hasattr(features, "doppler_psd") and features.doppler_psd is not None else None
                    broadcaster.update_csi(
                        node_id=node_id,
                        velocity=features.dominant_velocity_mps,
                        surge=features.energy_surge_ratio,
                        state=state.value,
                        doppler_psd=psd,
                    )
                except Exception:
                    pass

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

        if is_web:
            try:
                from hub.dashboard.app import broadcaster
                broadcaster.update_radar(
                    height=radar_telemetry.target_height_m,
                    posture=radar_telemetry.posture.value,
                    fall_state=radar_telemetry.fall_state.value,
                    dwell=radar_telemetry.dwell_time_sec,
                    state=fusion_state.value,
                )
            except Exception:
                pass

        if ha_manager:
            try:
                ha_manager.publish_radar(
                    height=radar_telemetry.target_height_m,
                    posture=radar_telemetry.posture.value,
                    fall_state=radar_telemetry.fall_state.value,
                    dwell=radar_telemetry.dwell_time_sec,
                )
                ha_manager.publish_state(fusion_state.value)
            except Exception:
                pass

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
# Replay Mode (Offline Multimodal Dataset Playback)
# ---------------------------------------------------------------------------
def run_replay(
    npz_path: Path,
    fusion_engine: DualFusionEngine,
    is_web: bool = False,
    classifier: Optional[FallClassifier] = None,
    ha_manager: Optional[HomeAssistantMQTTDiscoveryManager] = None,
):
    """Replays recorded CSI and Radar telemetry through the detection pipeline."""
    if not npz_path.exists():
        print(f"[ERROR] Dataset file not found: {npz_path}", file=sys.stderr)
        return

    print("\n" + "=" * 65)
    print(f"      REPLAYING MULTIMODAL DATASET: {npz_path.name}")
    print("=" * 65)

    with np.load(npz_path) as data:
        csi_node_ids = data.get("csi_node_ids", np.array([]))
        csi_timestamps = data.get("csi_timestamps", np.array([]))
        csi_amplitudes = data.get("csi_amplitudes", np.empty((0, 0)))
        radar_timestamps = data.get("radar_timestamps", np.array([]))
        radar_heights = data.get("radar_heights", np.array([]))
        radar_postures = data.get("radar_postures", np.array([]))
        radar_fall_states = data.get("radar_fall_states", np.array([]))
        radar_dwells = data.get("radar_dwells", np.array([]))

    preprocessor = CSIPreprocessor()
    pca_extractor = CSIPCAExtractor()
    node_buffers = {1: NodeBuffer(), 2: NodeBuffer(), 3: NodeBuffer()}

    events = []
    for i in range(len(csi_timestamps)):
        events.append((csi_timestamps[i], "csi", i))
    for j in range(len(radar_timestamps)):
        events.append((radar_timestamps[j], "radar", j))
    events.sort(key=lambda x: x[0])

    print(f"Total events: {len(events)} ({len(csi_timestamps)} CSI frames, {len(radar_timestamps)} Radar frames)\n")

    fall_detected_time = None
    from hub.csi_pipeline.preprocessor import CSIPacket

    for idx, (t_rel, ev_type, ev_idx) in enumerate(events):
        if ev_type == "csi":
            nid = int(csi_node_ids[ev_idx])
            amps = csi_amplitudes[ev_idx]
            if nid in node_buffers:
                buf = node_buffers[nid]
                pkt = CSIPacket(
                    node_id=nid,
                    rssi=-50,
                    subcarrier_count=len(amps),
                    timestamp_ms=int(t_rel * 1000),
                    seq_num=ev_idx,
                    amplitudes=amps,
                    phases=np.zeros_like(amps),
                )
                buf.push(pkt)
                if buf.total_received % 10 == 0:
                    window = buf.get_window()
                    if window.shape[0] >= 32:
                        filtered = preprocessor.filter_stream(window)
                        features = pca_extractor.extract(node_id=nid, filtered_data=filtered)

                        ml_prob = 0.0
                        if classifier:
                            ml_feat = classifier.extractor.extract_from_window(filtered)
                            ml_prob = classifier.predict_proba(ml_feat)

                        st = fusion_engine.update_csi(features, current_time=time.time(), ml_prob=ml_prob)

                        if ha_manager:
                            try:
                                ha_manager.publish_csi(
                                    node_id=nid,
                                    velocity=features.dominant_velocity_mps,
                                    surge=features.energy_surge_ratio,
                                    state=st.value,
                                )
                                ha_manager.publish_ml_probability(ml_prob)
                                ha_manager.publish_state(st.value)
                            except Exception:
                                pass

                        if is_web:
                            try:
                                from hub.dashboard.app import broadcaster
                                psd = features.doppler_psd.tolist() if hasattr(features, "doppler_psd") and features.doppler_psd is not None else None
                                broadcaster.update_csi(
                                    node_id=nid,
                                    velocity=features.dominant_velocity_mps,
                                    surge=features.energy_surge_ratio,
                                    state=st.value,
                                    doppler_psd=psd,
                                )
                            except Exception:
                                pass
        elif ev_type == "radar":
            h = float(radar_heights[ev_idx])
            posture_val = radar_postures[ev_idx]
            fall_st_val = radar_fall_states[ev_idx]
            dwell = int(radar_dwells[ev_idx])
            telemetry = RadarTelemetry(
                fall_state=fall_st_val,
                posture=posture_val,
                target_height_m=h,
                dwell_time_sec=dwell,
            )
            st = fusion_engine.update_radar(telemetry)
            if st == UnifiedFallState.CONFIRMED and fall_detected_time is None:
                fall_detected_time = t_rel

            if ha_manager:
                try:
                    ha_manager.publish_radar(
                        height=h,
                        posture=str(posture_val),
                        fall_state=str(fall_st_val),
                        dwell=dwell,
                    )
                    ha_manager.publish_state(st.value)
                except Exception:
                    pass

            if is_web:
                try:
                    from hub.dashboard.app import broadcaster
                    broadcaster.update_radar(
                        height=h,
                        posture=str(posture_val),
                        fall_state=str(fall_st_val),
                        dwell=dwell,
                        state=st.value,
                    )
                except Exception:
                    pass

        if idx % 20 == 0:
            status_sym = "[OK]" if fusion_engine.unified_state == UnifiedFallState.NORMAL else (
                "[SUSPECT]" if fusion_engine.unified_state == UnifiedFallState.SUSPECTED else "[FALL]"
            )
            print(f"\r[Replay: {t_rel:5.2f}s] {status_sym} State: {fusion_engine.unified_state.value:<15}", end="", flush=True)

    print(f"\n\n[Replay Completed]")
    if fall_detected_time is not None:
        print(f"  Result: [!] FALL DETECTED at T={fall_detected_time:.2f}s")
    else:
        print(f"  Result: [OK] No fall detected (Normal activity)")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    mode = OperatingMode(args.mode)
    alert = AlertDispatcher(
        enable_sound=not args.no_sound,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
    )
    fusion_engine = DualFusionEngine(mode=mode, alert_dispatcher=alert)
    classifier = FallClassifier(model_path=args.ml_model)

    ha_manager = None
    if args.mqtt_broker and alert._mqtt_client:
        ha_manager = HomeAssistantMQTTDiscoveryManager(
            mqtt_client=alert._mqtt_client,
            base_topic="fall_detection",
        )
        if args.ha_discovery:
            count = ha_manager.announce_discovery()
            print(f"  [HA]    Announced {count} Home Assistant MQTT discovery entities")

    room_manager = None
    if args.multi_room:
        from hub.room_manager import RoomManager
        room_manager = RoomManager()
        print("  [MESH]  Multi-room spatial mesh enabled")

    if args.web:
        from hub.dashboard.app import broadcaster as web_broadcaster
        web_broadcaster.csi_engine = fusion_engine.csi_engine
        web_broadcaster.fusion_engine = fusion_engine
        if room_manager:
            web_broadcaster.room_manager = room_manager

        def _run_web(port: int):
            import uvicorn
            from hub.dashboard.app import app as web_app
            uvicorn.run(web_app, host="0.0.0.0", port=port, log_level="warning")

        web_thread = threading.Thread(target=_run_web, args=(args.web_port,), daemon=True)
        web_thread.start()
        print(f"  [WEB]   Dashboard HUD available at http://localhost:{args.web_port}")

    if getattr(args, "fhir_base_url", None) and getattr(args, "fhir_client_id", None) and getattr(args, "fhir_client_secret", None):
        try:
            from hub.smart_fhir_client import SMARTFHIRClient
            smart_client = SMARTFHIRClient(
                fhir_base_url=args.fhir_base_url,
                client_id=args.fhir_client_id,
                client_secret=args.fhir_client_secret,
            )
            print(f"  [FHIR]  SMART-on-FHIR client initialized for {args.fhir_base_url}")
            if args.web:
                from hub.dashboard.app import broadcaster as web_broadcaster
                web_broadcaster.smart_fhir_client = smart_client
        except Exception as e:
            print(f"  [FHIR]  Failed to initialize SMART-on-FHIR client: {e}")

    if args.demo:
        run_demo(fusion_engine, is_web=args.web, classifier=classifier, ha_manager=ha_manager)
        return

    if args.replay:
        run_replay(args.replay, fusion_engine, is_web=args.web, classifier=classifier, ha_manager=ha_manager)
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

    # Optional UDP relay client for secondary clusters
    relay_client = None
    if args.relay_port:
        from hub.relay_client import RelayClient
        node_buffers_by_room = {}

        def _handle_raw_csi(r_id: int, payload: bytes):
            pkt = preprocessor.parse_packet(payload)
            if pkt:
                pkt.room_id = r_id
                if r_id not in node_buffers_by_room:
                    node_buffers_by_room[r_id] = {}
                if pkt.node_id not in node_buffers_by_room[r_id]:
                    node_buffers_by_room[r_id][pkt.node_id] = NodeBuffer()
                nb = node_buffers_by_room[r_id][pkt.node_id]
                nb.push(pkt)
                if nb.total_received % EXTRACTION_INTERVAL == 0:
                    win = nb.get_window()
                    if win.shape[0] >= 32:
                        flt = preprocessor.filter_stream(win)
                        feat = pca_extractor.extract(node_id=pkt.node_id, filtered_data=flt)
                        if room_manager:
                            room_manager.on_csi_packet(r_id, feat)

        def _handle_raw_radar(r_id: int, payload: bytes):
            try:
                pl = payload.decode("utf-8", errors="ignore").strip()
                tel = radar_receiver.parse_json_datagram(pl)
            except Exception:
                tel = None
            if tel is None:
                tel = radar_receiver.parse_binary_frame(payload)
            if tel is not None and room_manager:
                room_manager.on_radar_packet(r_id, tel)

        relay_client = RelayClient(
            room_manager=room_manager,
            relay_port=args.relay_port,
            raw_csi_handler=_handle_raw_csi,
            raw_radar_handler=_handle_raw_radar,
        )
        relay_client.start()
        print(f"  [RELAY] UDP relay listening on port {args.relay_port}")
        if args.web:
            web_broadcaster.relay_client = relay_client

    # CSI Listener
    if mode in (OperatingMode.CSI_ONLY, OperatingMode.FUSION):
        csi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csi_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        csi_sock.bind(("0.0.0.0", args.csi_port))
        sockets.append(csi_sock)
        print(f"  [CSI]   Listening on UDP port {args.csi_port}")

        t = threading.Thread(
            target=csi_listener_thread,
            args=(csi_sock, preprocessor, pca_extractor, fusion_engine, node_buffers, stop_event, classifier, ha_manager, room_manager),
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
            args=(radar_sock, radar_receiver, fusion_engine, stop_event, ha_manager, room_manager),
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
        if relay_client:
            relay_client.stop()
        for s in sockets:
            try:
                s.close()
            except Exception:
                pass
        for t in threads:
            t.join(timeout=2.0)


if __name__ == "__main__":
    main()

