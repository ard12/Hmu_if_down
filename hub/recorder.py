"""Ground-Truth Multimodal Dataset Recorder.

Listens for synchronized Wi-Fi CSI packets and mmWave radar telemetry during
human trial sessions, saving timestamped, labeled recordings to compressed
.npz datasets with JSON metadata for model training and benchmark replay.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from hub.csi_pipeline.preprocessor import CSIPacket, CSIPreprocessor
from hub.mmwave_pipeline.radar_receiver import RadarReceiver, RadarTelemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Fall Detection Dataset Recorder")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Ground-truth activity label (e.g. fall_forward, fall_backward, sit_chair, walk_normal, drop_object)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subject_01",
        help="Subject identifier or anonymized trial code (default: subject_01)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Session recording duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets",
        help="Directory to save recorded session files (default: datasets/)",
    )
    parser.add_argument(
        "--csi-port",
        type=int,
        default=5555,
        help="UDP listening port for CSI stream (default: 5555)",
    )
    parser.add_argument(
        "--radar-port",
        type=int,
        default=5556,
        help="UDP listening port for mmWave Radar stream (default: 5556)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional qualitative notes about the trial (e.g. carpet floor, loose clothing)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate synthetic session data for offline verification and testing",
    )
    return parser.parse_args()


class SessionRecorder:
    """Manages recording buffers and synchronizes multimodal frames."""

    def __init__(self, label: str, subject_id: str, notes: str = ""):
        self.label = label
        self.subject_id = subject_id
        self.notes = notes
        self.session_id = f"{label}_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # CSI buffers
        self.csi_node_ids: List[int] = []
        self.csi_timestamps: List[float] = []
        self.csi_seq_nums: List[int] = []
        self.csi_rssi: List[int] = []
        self.csi_amplitudes: List[np.ndarray] = []

        # Radar buffers
        self.radar_timestamps: List[float] = []
        self.radar_heights: List[float] = []
        self.radar_postures: List[int] = []
        self.radar_fall_states: List[int] = []
        self.radar_dwells: List[int] = []

    def add_csi_packet(self, packet: CSIPacket, record_time: float):
        self.csi_node_ids.append(packet.node_id)
        self.csi_timestamps.append(record_time)
        self.csi_seq_nums.append(packet.seq_num)
        self.csi_rssi.append(packet.rssi)
        self.csi_amplitudes.append(packet.amplitudes)

    def add_radar_telemetry(self, telemetry: RadarTelemetry, record_time: float):
        self.radar_timestamps.append(record_time)
        self.radar_heights.append(telemetry.target_height_m)
        self.radar_postures.append(telemetry.posture.value if hasattr(telemetry.posture, "value") else 0)
        self.radar_fall_states.append(telemetry.fall_state.value if hasattr(telemetry.fall_state, "value") else 0)
        self.radar_dwells.append(telemetry.dwell_time_sec)

    def save(self, output_dir: Path) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / f"{self.session_id}.npz"
        json_path = output_dir / f"{self.session_id}.json"

        # Convert CSI amplitudes to 2D matrix
        if self.csi_amplitudes:
            # Handle potential variable subcarrier length by trimming to minimum
            min_sub = min(len(a) for a in self.csi_amplitudes)
            amp_matrix = np.vstack([a[:min_sub] for a in self.csi_amplitudes])
        else:
            amp_matrix = np.empty((0, 0), dtype=np.float32)

        # Save compressed NPZ
        np.savez_compressed(
            npz_path,
            csi_node_ids=np.array(self.csi_node_ids, dtype=np.uint8),
            csi_timestamps=np.array(self.csi_timestamps, dtype=np.float64),
            csi_seq_nums=np.array(self.csi_seq_nums, dtype=np.uint32),
            csi_rssi=np.array(self.csi_rssi, dtype=np.int8),
            csi_amplitudes=amp_matrix.astype(np.float32),
            radar_timestamps=np.array(self.radar_timestamps, dtype=np.float64),
            radar_heights=np.array(self.radar_heights, dtype=np.float32),
            radar_postures=np.array(self.radar_postures),
            radar_fall_states=np.array(self.radar_fall_states),
            radar_dwells=np.array(self.radar_dwells, dtype=np.int32),
        )

        # Count per-node packets
        node_counts = {}
        for nid in self.csi_node_ids:
            node_counts[nid] = node_counts.get(nid, 0) + 1

        # Save JSON metadata
        metadata = {
            "session_id": self.session_id,
            "recorded_at": datetime.now().isoformat(),
            "label": self.label,
            "subject_id": self.subject_id,
            "notes": self.notes,
            "csi_samples": len(self.csi_timestamps),
            "radar_samples": len(self.radar_timestamps),
            "node_distribution": node_counts,
            "files": {
                "dataset_npz": npz_path.name,
                "metadata_json": json_path.name,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return npz_path, json_path


def record_live_session(
    label: str,
    subject_id: str,
    duration: float,
    csi_port: int = 5555,
    radar_port: int = 5556,
    notes: str = "",
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Capture live UDP streams over the given duration."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "datasets"

    recorder = SessionRecorder(label=label, subject_id=subject_id, notes=notes)
    preprocessor = CSIPreprocessor()
    radar_receiver = RadarReceiver()

    # Open sockets
    csi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    csi_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    csi_sock.bind(("0.0.0.0", csi_port))
    csi_sock.settimeout(0.05)

    radar_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    radar_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    radar_sock.bind(("0.0.0.0", radar_port))
    radar_sock.settimeout(0.05)

    print(f"\n=======================================================")
    print(f"  RECORDING SESSION: {recorder.session_id}")
    print(f"  Label: {label} | Subject: {subject_id} | Duration: {duration:.1f}s")
    print(f"=======================================================\n")

    start_time = time.time()
    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= duration:
                break

            # Read CSI
            try:
                data, _ = csi_sock.recvfrom(2048)
                if not data.startswith(b"{"):
                    pkt = preprocessor.parse_packet(data)
                    if pkt is not None:
                        recorder.add_csi_packet(pkt, elapsed)
            except socket.timeout:
                pass

            # Read Radar
            try:
                rdata, _ = radar_sock.recvfrom(2048)
                payload = rdata.decode("utf-8", errors="ignore").strip()
                telemetry = radar_receiver.parse_json_datagram(payload)
                if telemetry is None:
                    telemetry = radar_receiver.parse_binary_frame(rdata)
                if telemetry is not None:
                    recorder.add_radar_telemetry(telemetry, elapsed)
            except socket.timeout:
                pass

            # Update progress
            pct = min(100, int((elapsed / duration) * 100))
            print(
                f"\r[Recording: {elapsed:4.1f}s / {duration:4.1f}s ({pct}%)] "
                f"CSI Frames: {len(recorder.csi_timestamps)} | "
                f"Radar Frames: {len(recorder.radar_timestamps)}",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Recording ended early by user.")
    finally:
        csi_sock.close()
        radar_sock.close()

    print("\n\nWriting dataset to disk...")
    npz_path, json_path = recorder.save(output_dir)
    print(f"Saved dataset: {npz_path}")
    print(f"Saved metadata: {json_path}\n")
    return npz_path, json_path


def record_simulated_session(
    label: str,
    subject_id: str,
    duration: float = 10.0,
    notes: str = "synthetic test session",
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Generate synthetic multimodal session data for automated tests & validation."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "datasets"

    recorder = SessionRecorder(label=label, subject_id=subject_id, notes=notes)
    num_samples = int(duration * 100)  # 100 Hz

    for t in range(num_samples):
        timestamp = t * 0.01

        # Simulate 3 tracker nodes
        for nid in (1, 2, 3):
            if "fall" in label.lower() and (3.0 <= timestamp <= 3.5):
                amps = np.ones(64) * 30.0 + np.sin(np.linspace(0, 10, 64)) * 25.0
            else:
                amps = np.ones(64) * 20.0 + np.random.normal(0, 0.5, 64)

            pkt = CSIPacket(
                node_id=nid,
                rssi=-48,
                subcarrier_count=64,
                timestamp_ms=int(timestamp * 1000),
                seq_num=t,
                amplitudes=amps,
                phases=np.zeros(64),
            )
            recorder.add_csi_packet(pkt, timestamp)

        # Simulate radar at 10 Hz
        if t % 10 == 0:
            h = 0.22 if ("fall" in label.lower() and timestamp > 3.5) else 1.65
            posture = "Lying" if h < 0.35 else "Standing"
            fall_st = "Confirmed" if (h < 0.35 and timestamp > 4.5) else "None"
            telemetry = RadarTelemetry(
                fall_state=fall_st,
                posture=posture,
                target_height_m=h,
                dwell_time_sec=int(max(0, timestamp - 3.5)) if h < 0.35 else 0,
            )
            recorder.add_radar_telemetry(telemetry, timestamp)

    npz_path, json_path = recorder.save(output_dir)
    return npz_path, json_path


def main():
    args = parse_args()
    if args.simulate:
        record_simulated_session(
            label=args.label,
            subject_id=args.subject,
            duration=args.duration,
            notes=args.notes,
            output_dir=args.output_dir,
        )
    else:
        record_live_session(
            label=args.label,
            subject_id=args.subject,
            duration=args.duration,
            csi_port=args.csi_port,
            radar_port=args.radar_port,
            notes=args.notes,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
