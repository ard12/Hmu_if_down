"""Room calibration tool for multi-node Wi-Fi CSI baseline noise profiling.

Listens for UDP CSI packets from ESP32 tracker nodes (or generates synthetic
baseline data) to measure background RF noise floor, per-subcarrier variance,
RSSI, and packet rates. Computes recommended thresholds for motionless detection
and energy surges, saving results to config/calibration.yaml.
"""

import argparse
from datetime import datetime
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from hub.csi_pipeline.preprocessor import CSIPacket, CSIPreprocessor


# ---------------------------------------------------------------------------
# YAML FlowList Formatter (for compact subcarrier arrays in calibration.yaml)
# ---------------------------------------------------------------------------
class FlowList(list):
    """List subclass that serializes as inline flow style [x, y, ...] in YAML."""
    pass


def _flow_list_representer(dumper: yaml.Dumper, data: FlowList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, _flow_list_representer)
try:
    yaml.representer.SafeRepresenter.add_representer(FlowList, _flow_list_representer)
except Exception:
    pass


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Room Calibration Tool for Multi-Node Wi-Fi CSI Fall Detection"
    )
    parser.add_argument(
        "--csi-port",
        type=int,
        default=5555,
        help="UDP port to listen for tracker node CSI packets (default: 5555)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Calibration duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate synthetic baseline CSI data (no human movement) for testing without hardware",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "config" / "calibration.yaml",
        help="Target YAML calibration file (default: config/calibration.yaml)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic Baseline Data Generator
# ---------------------------------------------------------------------------
def generate_synthetic_baseline(
    duration: float,
    preprocessor: CSIPreprocessor,
    nodes: Tuple[int, ...] = (1, 2, 3),
    sampling_rate_hz: float = 100.0,
    subcarrier_count: int = 64,
    noise_std: float = 0.05,
    seed: Optional[int] = 42,
) -> Tuple[Dict[int, List[CSIPacket]], float]:
    """Generate synthetic baseline CSI packets (no human movement) for test/validation.

    Constructs raw binary packets in the exact ESP32 UDP format ("CSIF" header +
    signed 8-bit I/Q payload) and unpacks them using CSIPreprocessor.
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = max(1, int(round(duration * sampling_rate_hz)))
    collected: Dict[int, List[CSIPacket]] = {nid: [] for nid in nodes}

    # Distinct baseline multipath profile and RSSI per node
    base_rssi = {1: -45, 2: -50, 3: -55}

    for nid in nodes:
        # Static baseline amplitudes (22 - 32) and phases (-pi to +pi)
        base_amps = np.random.uniform(22.0, 32.0, subcarrier_count)
        base_phases = np.random.uniform(-np.pi, np.pi, subcarrier_count)
        base_i = base_amps * np.cos(base_phases)
        base_q = base_amps * np.sin(base_phases)
        rssi_val = base_rssi.get(nid, -48)

        for t in range(num_samples):
            # Add small thermal/ambient noise to simulate motionless room baseline
            noise_i = np.random.normal(0.0, noise_std, subcarrier_count)
            noise_q = np.random.normal(0.0, noise_std, subcarrier_count)

            raw_i = np.clip(np.round(base_i + noise_i), -128, 127).astype(np.int8)
            raw_q = np.clip(np.round(base_q + noise_q), -128, 127).astype(np.int8)

            # Interleave into (I, Q) pairs
            iq_buf = np.empty(subcarrier_count * 2, dtype=np.int8)
            iq_buf[0::2] = raw_i
            iq_buf[1::2] = raw_q

            # Timestamp in milliseconds, seq_num incrementing
            ts_ms = int(t * (1000.0 / sampling_rate_hz))
            seq_num = t
            # Slight RSSI jitter (+/- 1 dBm)
            rssi = int(rssi_val + np.random.choice([-1, 0, 1]))

            header = struct.pack(
                CSIPreprocessor.HEADER_FORMAT,
                b"CSIF",
                nid,
                rssi,
                subcarrier_count,
                ts_ms,
                seq_num,
            )
            raw_packet = header + iq_buf.tobytes()

            # Parse with the preprocessor
            packet = preprocessor.parse_packet(raw_packet)
            if packet is not None:
                collected[nid].append(packet)

    return collected, duration


# ---------------------------------------------------------------------------
# Live UDP CSI Collection
# ---------------------------------------------------------------------------
def collect_live_csi(
    port: int,
    duration: float,
    preprocessor: CSIPreprocessor,
    expected_nodes: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[Dict[int, List[CSIPacket]], float]:
    """Listen on UDP port for incoming CSI packets over the specified duration."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(("0.0.0.0", port))
    except OSError as e:
        print(f"\n[ERROR] Failed to bind to UDP port {port}: {e}", file=sys.stderr)
        return {nid: [] for nid in expected_nodes}, 0.0

    sock.settimeout(0.5)
    collected: Dict[int, List[CSIPacket]] = {nid: [] for nid in expected_nodes}

    print(f"Listening on UDP 0.0.0.0:{port} for {duration:.1f}s...")
    print("Ensure the room is vacant and motionless for accurate noise profiling.\n")

    start_time = time.time()
    last_print = 0.0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                pass
            except OSError:
                break
            else:
                # Ignore JSON heartbeats or corrupted fragments
                if data.startswith(b"{"):
                    continue

                packet = preprocessor.parse_packet(data)
                if packet is not None:
                    if packet.node_id not in collected:
                        collected[packet.node_id] = []
                    collected[packet.node_id].append(packet)

            # Update progress line periodically
            now = time.time()
            if now - last_print >= 0.2:
                last_print = now
                progress_parts = [
                    f"Node {nid}: {len(pkts)} pkts"
                    for nid, pkts in sorted(collected.items())
                ]
                status = " | ".join(progress_parts)
                print(
                    f"\r[Calibrating: {min(elapsed, duration):4.1f}s / {duration:4.1f}s] {status}",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n\n[INFO] Calibration interrupted by user early.")
    finally:
        sock.close()

    actual_duration = max(time.time() - start_time, 0.001)
    print()  # New line after progress
    return collected, actual_duration


# ---------------------------------------------------------------------------
# Compute Calibration Metrics & Thresholds
# ---------------------------------------------------------------------------
def compute_calibration(
    collected: Dict[int, List[CSIPacket]],
    duration: float,
    expected_nodes: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Compute per-node metrics and auto-tune fall detection thresholds.

    Per node:
      - mean_amplitudes: baseline noise floor per subcarrier
      - std_amplitudes: standard deviation per subcarrier
      - mean_rssi: average received signal strength (dBm)
      - packet_rate_hz: received packets per second

    Thresholds:
      - motionless_variance_threshold = 2.0 * mean_variance_across_nodes
      - energy_surge_ratio = 3.0 (default unless environment is very noisy)
    """
    node_stats: Dict[int, Dict[str, Any]] = {}
    node_variances: List[float] = []

    for nid in expected_nodes:
        packets = collected.get(nid, [])
        count = len(packets)

        if count == 0:
            node_stats[nid] = {
                "packet_count": 0,
                "packet_rate_hz": 0.0,
                "mean_rssi_dbm": None,
                "subcarrier_count": 0,
                "mean_variance": 0.0,
                "mean_amplitudes": [],
                "std_amplitudes": [],
                "status": "NO_DATA",
            }
            continue

        # Stack amplitude vectors into (N, subcarrier_count)
        amp_matrix = np.vstack([p.amplitudes for p in packets])
        subcarrier_count = amp_matrix.shape[1]

        mean_amps = np.mean(amp_matrix, axis=0)
        std_amps = np.std(amp_matrix, axis=0)
        rssi_vals = [p.rssi for p in packets]

        # Mean temporal variance across all subcarriers for this node
        var_per_subcarrier = std_amps ** 2
        mean_var = float(np.mean(var_per_subcarrier))
        node_variances.append(mean_var)

        node_stats[nid] = {
            "packet_count": count,
            "packet_rate_hz": float(round(count / max(duration, 0.001), 2)),
            "mean_rssi_dbm": float(round(float(np.mean(rssi_vals)), 1)),
            "subcarrier_count": int(subcarrier_count),
            "mean_variance": float(round(mean_var, 6)),
            "mean_amplitudes": FlowList([float(round(v, 3)) for v in mean_amps]),
            "std_amplitudes": FlowList([float(round(v, 4)) for v in std_amps]),
            "status": "OK",
        }

    # Threshold auto-computation
    if len(node_variances) > 0:
        mean_var_across_nodes = float(np.mean(node_variances))
    else:
        # Fallback default if no packets were captured
        mean_var_across_nodes = 0.04

    # motionless_variance_threshold = 2.0 * mean_variance_across_nodes
    motionless_thresh = float(round(2.0 * mean_var_across_nodes, 4))

    # energy_surge_ratio: default 3.0 unless environment is very noisy (variance > 0.25)
    is_very_noisy = mean_var_across_nodes > 0.25
    if is_very_noisy:
        # Elevated ambient noise floor; elevate surge ratio to avoid false triggers
        energy_surge = float(round(min(5.0, 3.0 + (mean_var_across_nodes - 0.25) * 2.0), 2))
    else:
        energy_surge = 3.0

    thresholds = {
        "motionless_variance_threshold": motionless_thresh,
        "energy_surge_ratio": energy_surge,
        "mean_variance_across_nodes": float(round(mean_var_across_nodes, 6)),
        "is_noisy_environment": bool(is_very_noisy),
    }

    return node_stats, thresholds


# ---------------------------------------------------------------------------
# Output File Writer
# ---------------------------------------------------------------------------
def write_calibration_yaml(
    output_path: Path,
    duration: float,
    mode: str,
    node_stats: Dict[int, Dict[str, Any]],
    thresholds: Dict[str, Any],
    csi_port: int,
):
    """Write formatted calibration results to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "calibration_metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": float(round(duration, 2)),
            "mode": mode,
            "csi_port": int(csi_port),
        },
        "recommended_thresholds": thresholds,
        "nodes": node_stats,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Stdout Summary Printer
# ---------------------------------------------------------------------------
def print_summary(
    mode: str,
    duration: float,
    node_stats: Dict[int, Dict[str, Any]],
    thresholds: Dict[str, Any],
    output_path: Path,
):
    """Display an informative terminal summary of the calibration."""
    sep = "=" * 78
    line = "-" * 78

    print("\n" + sep)
    print("                    CSI ROOM CALIBRATION SUMMARY")
    print(sep)
    print(f"  Mode:                {mode.upper()}")
    print(f"  Duration:            {duration:.1f} s")
    print(f"  Output Path:         {output_path.resolve()}")
    print(line)
    print(
        f"  {'Node':<6} {'Status':<9} {'Packets':<9} {'Rate (Hz)':<11} "
        f"{'Mean RSSI':<11} {'Subcarriers':<13} {'Mean Var':<10}"
    )
    print(line)

    for nid, stats in sorted(node_stats.items()):
        if stats["status"] == "OK":
            rssi_str = f"{stats['mean_rssi_dbm']:.1f} dBm"
            var_str = f"{stats['mean_variance']:.4f}"
            print(
                f"  {nid:<6} {'OK':<9} {stats['packet_count']:<9} "
                f"{stats['packet_rate_hz']:<11.1f} {rssi_str:<11} "
                f"{stats['subcarrier_count']:<13} {var_str:<10}"
            )
        else:
            print(
                f"  {nid:<6} {'NO DATA':<9} {'0':<9} {'0.0':<11} "
                f"{'N/A':<11} {'0':<13} {'N/A':<10}"
            )

    print(line)
    print("  Recommended Detection Thresholds:")
    print(
        f"    * motionless_variance_threshold: {thresholds['motionless_variance_threshold']:.4f} "
        f"(2.0 * mean_variance: {thresholds['mean_variance_across_nodes']:.4f})"
    )
    noise_tag = "[WARN] Elevated ambient noise" if thresholds["is_noisy_environment"] else "[OK] Baseline quiet"
    print(
        f"    * energy_surge_ratio:            {thresholds['energy_surge_ratio']:.2f} "
        f"({noise_tag})"
    )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def run_calibration(
    port: int = 5555,
    duration: float = 30.0,
    simulate: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Execute complete room calibration pipeline."""
    if output_path is None:
        output_path = PROJECT_ROOT / "config" / "calibration.yaml"

    preprocessor = CSIPreprocessor()

    if simulate:
        print(f"\n[SIMULATE] Generating {duration:.1f}s of synthetic baseline data (nodes 1, 2, 3)...")
        collected, actual_duration = generate_synthetic_baseline(
            duration=duration,
            preprocessor=preprocessor,
        )
        mode = "simulate"
    else:
        collected, actual_duration = collect_live_csi(
            port=port,
            duration=duration,
            preprocessor=preprocessor,
        )
        mode = "live_udp"

    node_stats, thresholds = compute_calibration(
        collected=collected,
        duration=actual_duration,
    )

    write_calibration_yaml(
        output_path=output_path,
        duration=actual_duration,
        mode=mode,
        node_stats=node_stats,
        thresholds=thresholds,
        csi_port=port,
    )

    print_summary(
        mode=mode,
        duration=actual_duration,
        node_stats=node_stats,
        thresholds=thresholds,
        output_path=output_path,
    )

    return node_stats, thresholds


def main():
    args = parse_args()
    run_calibration(
        port=args.csi_port,
        duration=args.duration,
        simulate=args.simulate,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
