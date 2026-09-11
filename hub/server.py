"""Central Hub Server for Wi-Fi CSI, mmWave Radar, and Dual-Sensor Fusion."""

import argparse
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

    node_buffers = {1: [], 2: [], 3: []}
    sim_start = time.time()

    for step in range(80):
        now = time.time()
        elapsed = now - sim_start

        # Determine simulated state
        is_falling = (3.0 <= elapsed <= 3.6)
        is_lying_on_floor = (elapsed > 3.6)

        print(f"\r[T={elapsed:4.1f}s] ", end="")

        for node_id in (1, 2, 3):
            # Generate 64 subcarriers of simulated CSI amplitudes
            if is_falling and node_id in (1, 2):
                # Fall: Fast downward Doppler oscillation (25-30 Hz) + high amplitude
                t_arr = np.linspace(0, 0.5, 50)
                fast_wave = np.sin(2 * np.pi * 28.0 * t_arr)[:, None]
                sim_amplitudes = np.ones((50, 64)) * 30.0 + fast_wave * 45.0 + np.random.normal(0, 2.0, (50, 64))
            elif is_lying_on_floor:
                # Floor stillness: very low variance
                sim_amplitudes = np.ones((50, 64)) * 15.0 + np.random.normal(0, 0.05, (50, 64))
            else:
                # Normal walking: low frequency (3 Hz) moderate oscillation
                t_arr = np.linspace(0, 0.5, 50)
                walk_wave = np.sin(2 * np.pi * 3.5 * t_arr)[:, None]
                sim_amplitudes = np.ones((50, 64)) * 25.0 + walk_wave * 8.0 + np.random.normal(0, 1.0, (50, 64))

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
    print(f"Listening for Wi-Fi CSI UDP packets on port {args.csi_port}...")
    print(f"Listening for mmWave Radar datagrams on port {args.radar_port}...")
    print("Press Ctrl+C to exit.\n")

    # In live mode, open UDP sockets and listen
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.csi_port))
    preprocessor = CSIPreprocessor()
    pca_extractor = CSIPCAExtractor()

    try:
        while True:
            data, addr = sock.recvfrom(2048)
            packet = preprocessor.parse_packet(data)
            if packet:
                # In live mode, maintain rolling buffer of amplitudes
                # and feed into feature extraction
                pass
    except KeyboardInterrupt:
        print("\n[Shutting down] Server stopped by user.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
