"""
Synthetic CSI & Radar UDP streamer for end-to-end testing.

Generates time-series trajectories of human movement (standing -> walking ->
falling -> lying) and streams the corresponding synthetic CSI packets and
radar telemetry frames to the hub on configurable ports.

@req SRS-SIM-002
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional
import logging
import socket
import struct
import threading
import time
import numpy as np

logger = logging.getLogger(__name__)


class ScenarioPhase(Enum):
    STANDING = "standing"
    WALKING = "walking"
    PRE_FALL = "pre_fall"
    FALLING = "falling"
    LYING = "lying"
    RECOVERY = "recovery"


@dataclass
class TrajectoryKeyframe:
    time_s: float
    position_x: float
    position_y: float
    position_z: float  # Head height in metres
    velocity_mps: float
    phase: ScenarioPhase


@dataclass
class FallScenario:
    """Pre-defined fall trajectory for simulation."""
    name: str
    keyframes: List[TrajectoryKeyframe]
    fall_type: str = "forward_trip"
    duration_s: float = 10.0


class SyntheticStreamer:
    """Streams simulated fall scenarios as UDP packets."""

    MAGIC = b"CSIF"

    def __init__(
        self,
        target_host: str = "127.0.0.1",
        csi_port: int = 5555,
        radar_port: int = 5556,
        packet_rate_hz: int = 100,
    ):
        self.target_host = target_host
        self.csi_port = csi_port
        self.radar_port = radar_port
        self.packet_rate_hz = packet_rate_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scenarios_played = 0

    @staticmethod
    def generate_forward_trip(duration_s: float = 10.0) -> FallScenario:
        """Generate a forward trip fall scenario."""
        return FallScenario(
            name="forward_trip",
            keyframes=[
                TrajectoryKeyframe(0.0, 2.0, 2.0, 1.7, 0.0, ScenarioPhase.STANDING),
                TrajectoryKeyframe(2.0, 2.0, 2.0, 1.7, 0.0, ScenarioPhase.STANDING),
                TrajectoryKeyframe(4.0, 3.0, 2.0, 1.7, 1.0, ScenarioPhase.WALKING),
                TrajectoryKeyframe(5.0, 3.5, 2.0, 1.5, 1.5, ScenarioPhase.PRE_FALL),
                TrajectoryKeyframe(5.5, 4.0, 2.0, 0.8, 3.0, ScenarioPhase.FALLING),
                TrajectoryKeyframe(6.0, 4.3, 2.0, 0.3, 0.5, ScenarioPhase.LYING),
                TrajectoryKeyframe(duration_s, 4.3, 2.0, 0.3, 0.0, ScenarioPhase.LYING),
            ],
            fall_type="forward_trip",
            duration_s=duration_s,
        )

    @staticmethod
    def generate_syncope_drop(duration_s: float = 10.0) -> FallScenario:
        """Generate a syncope (fainting) fall scenario."""
        return FallScenario(
            name="syncope_drop",
            keyframes=[
                TrajectoryKeyframe(0.0, 2.0, 2.0, 1.7, 0.0, ScenarioPhase.STANDING),
                TrajectoryKeyframe(3.0, 2.0, 2.0, 1.7, 0.0, ScenarioPhase.STANDING),
                TrajectoryKeyframe(3.2, 2.0, 2.0, 1.5, 0.1, ScenarioPhase.PRE_FALL),
                TrajectoryKeyframe(3.8, 2.0, 2.0, 0.3, 4.0, ScenarioPhase.FALLING),
                TrajectoryKeyframe(4.2, 2.1, 2.0, 0.2, 0.2, ScenarioPhase.LYING),
                TrajectoryKeyframe(duration_s, 2.1, 2.0, 0.2, 0.0, ScenarioPhase.LYING),
            ],
            fall_type="syncope_drop",
            duration_s=duration_s,
        )

    def interpolate_keyframes(
        self, scenario: FallScenario, t: float
    ) -> TrajectoryKeyframe:
        """Linearly interpolate between keyframes at time t."""
        kfs = scenario.keyframes
        if not kfs:
            return TrajectoryKeyframe(t, 0.0, 0.0, 1.7, 0.0, ScenarioPhase.STANDING)
        if t <= kfs[0].time_s:
            return kfs[0]
        if t >= kfs[-1].time_s:
            return kfs[-1]

        for i in range(len(kfs) - 1):
            if kfs[i].time_s <= t <= kfs[i + 1].time_s:
                delta = kfs[i + 1].time_s - kfs[i].time_s
                alpha = 0.0 if delta <= 0 else (t - kfs[i].time_s) / delta
                return TrajectoryKeyframe(
                    time_s=t,
                    position_x=float(kfs[i].position_x + alpha * (kfs[i + 1].position_x - kfs[i].position_x)),
                    position_y=float(kfs[i].position_y + alpha * (kfs[i + 1].position_y - kfs[i].position_y)),
                    position_z=float(kfs[i].position_z + alpha * (kfs[i + 1].position_z - kfs[i].position_z)),
                    velocity_mps=float(kfs[i].velocity_mps + alpha * (kfs[i + 1].velocity_mps - kfs[i].velocity_mps)),
                    phase=kfs[i + 1].phase if alpha > 0.5 else kfs[i].phase,
                )
        return kfs[-1]

    def build_csi_packet(
        self, keyframe: TrajectoryKeyframe, node_id: int = 1, room_id: int = 1, seq: int = 0
    ) -> bytes:
        """Build a synthetic CSI UDP packet from a trajectory keyframe."""
        n_subcarriers = 64
        rng = np.random.default_rng(seq)
        base_amplitude = max(-128.0, min(127.0, 30.0 - keyframe.velocity_mps * 5.0))
        noise = rng.standard_normal(n_subcarriers) * 3.0
        amplitudes = np.clip(base_amplitude + noise, -128, 127).astype(np.int8)

        rssi = int(np.clip(-40 - keyframe.velocity_mps * 3, -128, 127))
        ts_ms = int(keyframe.time_s * 1000) & 0xFFFFFFFF

        # Standard 18-byte V2 header:
        # magic: 4s (b"CSIF")
        # node_id: B (uint8)
        # room_id: B (uint8)
        # rssi: b (int8)
        # pad: x (1 byte)
        # subcarrier_count: H (uint16)
        # timestamp_ms: I (uint32)
        # seq_num: I (uint32)
        header = struct.pack(
            "<4sBBbxHII",
            self.MAGIC,
            node_id & 0xFF,
            room_id & 0xFF,
            rssi,
            n_subcarriers,
            ts_ms,
            seq & 0xFFFFFFFF,
        )
        return header + amplitudes.tobytes()

    def stream_scenario(self, scenario: FallScenario, send_udp: bool = True) -> int:
        """Stream a single scenario. Returns packet count."""
        sock = None
        if send_udp:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        dt = 1.0 / max(1, self.packet_rate_hz)
        n_packets = max(1, int(scenario.duration_s * self.packet_rate_hz))

        try:
            for seq in range(n_packets):
                t = seq * dt
                kf = self.interpolate_keyframes(scenario, t)
                pkt = self.build_csi_packet(kf, seq=seq)
                if sock is not None:
                    try:
                        sock.sendto(pkt, (self.target_host, self.csi_port))
                    except Exception:
                        pass
        finally:
            if sock is not None:
                sock.close()

        self._scenarios_played += 1
        return n_packets

    @property
    def scenarios_played(self) -> int:
        return self._scenarios_played
