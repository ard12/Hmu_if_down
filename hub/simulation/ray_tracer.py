"""
3D ray-tracing RF multipath room simulator.

Models Wi-Fi 2.4 GHz / 5 GHz signal propagation through rectangular
rooms using geometric optics (image source method) with configurable:
  - Room dimensions (L × W × H)
  - Wall material attenuation coefficients
  - Furniture as rectangular occluders
  - Human body as ellipsoidal reflector/absorber
  - TX/RX antenna positions (matching ESP32 placements)

Outputs synthetic CSI subcarrier amplitude/phase matrix.

@req SRS-SIM-001
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "Vec3":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class WallMaterial:
    name: str
    attenuation_db: float  # Per-wall attenuation at 2.4 GHz
    reflection_coeff: float  # 0.0 - 1.0


@dataclass
class RoomGeometry:
    """Rectangular room definition."""
    length_m: float
    width_m: float
    height_m: float
    wall_material: WallMaterial = field(
        default_factory=lambda: WallMaterial("drywall", 3.0, 0.6)
    )


@dataclass
class Obstacle:
    """Rectangular occluder (furniture, equipment)."""
    position: Vec3
    size: Vec3
    attenuation_db: float = 6.0


@dataclass
class HumanTarget:
    """Ellipsoidal human body model for CSI simulation."""
    position: Vec3
    height_m: float = 1.7
    radius_m: float = 0.25
    is_fallen: bool = False
    velocity_mps: float = 0.0


@dataclass
class SensorPlacement:
    position: Vec3
    node_id: int
    antenna_gain_dbi: float = 2.0


@dataclass
class SimulatedCSI:
    """Output of the ray tracer — synthetic CSI packet."""
    subcarrier_amplitudes: np.ndarray  # (n_subcarriers,) float32
    subcarrier_phases: np.ndarray  # (n_subcarriers,) float32
    rssi_dbm: float
    path_count: int
    tx_node_id: int
    rx_node_id: int
    timestamp_ms: float


class RoomRayTracer:
    """Image-source-method ray tracer for Wi-Fi CSI simulation."""

    SPEED_OF_LIGHT = 3e8
    FREQ_HZ = 2.4e9
    N_SUBCARRIERS = 64

    def __init__(
        self,
        room: RoomGeometry,
        sensors: List[SensorPlacement],
        obstacles: Optional[List[Obstacle]] = None,
        max_reflections: int = 3,
        n_subcarriers: int = 64,
    ):
        self.room = room
        self.sensors = sensors
        self.obstacles = obstacles or []
        self.max_reflections = max_reflections
        self.n_subcarriers = n_subcarriers
        self._wavelength = self.SPEED_OF_LIGHT / self.FREQ_HZ

    def _compute_path_loss(self, distance_m: float) -> float:
        """Free-space path loss in dB."""
        if distance_m <= 0.0:
            return 0.0
        return float(20.0 * np.log10(distance_m) + 20.0 * np.log10(self.FREQ_HZ) - 147.55)

    def _image_sources(self, tx: Vec3, order: int) -> List[Tuple[Vec3, float, int]]:
        """Generate image source positions for up to `order` reflections."""
        sources: List[Tuple[Vec3, float, int]] = [(tx, 0.0, 0)]  # (pos, total_atten, ref_count)
        if order == 0:
            return sources
        L, W, H = self.room.length_m, self.room.width_m, self.room.height_m
        atten = self.room.wall_material.attenuation_db

        for ref_order in range(1, order + 1):
            for dx in range(-ref_order, ref_order + 1):
                for dy in range(-ref_order, ref_order + 1):
                    if abs(dx) + abs(dy) != ref_order:
                        continue
                    img_x = dx * L + (tx.x if dx % 2 == 0 else L - tx.x)
                    img_y = dy * W + (tx.y if dy % 2 == 0 else W - tx.y)
                    total_atten = atten * ref_order
                    sources.append((
                        Vec3(img_x, img_y, tx.z),
                        total_atten,
                        ref_order,
                    ))
        return sources

    def trace(
        self, targets: Optional[List[HumanTarget]] = None, timestamp_ms: float = 0.0
    ) -> List[SimulatedCSI]:
        """
        Run ray trace for all TX-RX sensor pairs.

        Returns one SimulatedCSI per unique (TX, RX) sensor pair.
        """
        if targets is None:
            targets = []

        results = []
        for i, tx in enumerate(self.sensors):
            for j, rx in enumerate(self.sensors):
                if i >= j:
                    continue

                image_sources = self._image_sources(tx.position, self.max_reflections)

                amplitudes = np.zeros(self.n_subcarriers, dtype=np.float32)
                phases = np.zeros(self.n_subcarriers, dtype=np.float32)

                for src_pos, atten_db, ref_count in image_sources:
                    p1 = src_pos.to_array()
                    p2 = rx.position.to_array()
                    ray_vec = p2 - p1
                    ray_len_sq = float(np.dot(ray_vec, ray_vec))
                    d = float(np.sqrt(max(1e-6, ray_len_sq)))
                    if d < 0.01:
                        d = 0.01
                    path_loss = self._compute_path_loss(d) + atten_db

                    # Target body interaction along ray segment
                    body_atten = 0.0
                    for target in targets:
                        t_pos = target.position.to_array()
                        if target.is_fallen:
                            effective_pos = np.array([t_pos[0], t_pos[1], min(0.3, t_pos[2])])
                            effective_radius = target.radius_m * 1.5
                            atten_scale = 3.0
                        else:
                            effective_pos = np.array([t_pos[0], t_pos[1], min(target.height_m * 0.7, t_pos[2])])
                            effective_radius = target.radius_m * 3.5
                            atten_scale = 12.0

                        if ray_len_sq > 1e-6:
                            u = float(np.clip(np.dot(effective_pos - p1, ray_vec) / ray_len_sq, 0.0, 1.0))
                            proj = p1 + u * ray_vec
                            t_dist = float(np.linalg.norm(effective_pos - proj))
                        else:
                            t_dist = float(np.linalg.norm(effective_pos - p1))

                        if t_dist < effective_radius:
                            body_atten += atten_scale * (1.0 - t_dist / effective_radius)

                    # Obstacle interaction along ray
                    for obs in self.obstacles:
                        obs_pos = obs.position.to_array()
                        if ray_len_sq > 1e-6:
                            u = float(np.clip(np.dot(obs_pos - p1, ray_vec) / ray_len_sq, 0.0, 1.0))
                            proj = p1 + u * ray_vec
                            obs_dist = float(np.linalg.norm(obs_pos - proj))
                        else:
                            obs_dist = float(np.linalg.norm(obs_pos - p1))
                        if obs_dist < max(obs.size.x, obs.size.y):
                            body_atten += obs.attenuation_db * 0.5

                    total_loss = path_loss + body_atten
                    amplitude = float(10.0 ** (-total_loss / 20.0))

                    for k in range(self.n_subcarriers):
                        subcarrier_freq = self.FREQ_HZ + k * 312.5e3
                        phase = float((2 * np.pi * d * subcarrier_freq / self.SPEED_OF_LIGHT) % (2 * np.pi))
                        amplitudes[k] += amplitude * np.cos(phase)
                        phases[k] += phase

                rssi = float(-30.0 - 10.0 * np.log10(max(1e-10, float(np.mean(amplitudes ** 2)))))

                results.append(SimulatedCSI(
                    subcarrier_amplitudes=np.abs(amplitudes),
                    subcarrier_phases=phases % (2 * np.pi),
                    rssi_dbm=float(np.clip(rssi, -90, -10)),
                    path_count=len(image_sources),
                    tx_node_id=tx.node_id,
                    rx_node_id=rx.node_id,
                    timestamp_ms=timestamp_ms,
                ))
        return results

    def optimal_sensor_placement(
        self, n_sensors: int = 3, resolution: float = 0.5
    ) -> List[Vec3]:
        """Find sensor positions that maximise room coverage."""
        L, W = self.room.length_m, self.room.width_m
        candidates = []
        # Place along perimeter walls at height 2.0m
        for x in np.arange(0, L, resolution):
            candidates.append(Vec3(float(x), 0.0, 2.0))
            candidates.append(Vec3(float(x), float(W), 2.0))
        for y in np.arange(0, W, resolution):
            candidates.append(Vec3(0.0, float(y), 2.0))
            candidates.append(Vec3(float(L), float(y), 2.0))

        if len(candidates) <= n_sensors:
            return candidates[:n_sensors]

        # Greedy furthest-point sampling for maximum spread
        selected = [candidates[0]]
        for _ in range(n_sensors - 1):
            best_dist = -1.0
            best_candidate = candidates[0]
            for c in candidates:
                if any(np.allclose(c.to_array(), s.to_array()) for s in selected):
                    continue
                min_dist = min(
                    float(np.linalg.norm(c.to_array() - s.to_array())) for s in selected
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_candidate = c
            selected.append(best_candidate)
        return selected
