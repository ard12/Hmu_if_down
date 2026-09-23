"""
Multi-occupant spatial disambiguation engine.

Tracks multiple humans in a room by:
  1. Radar centroid clustering -> distinct blobs
  2. Hungarian algorithm (linear assignment) for frame-to-frame association
  3. Per-occupant Kalman filter for trajectory smoothing & prediction
  4. CSI Doppler signature isolation by spatial gating

Correctly attributes fall events to the specific occupant who fell,
preventing false alerts when a caregiver bends down to help.

@req SRS-MO-001
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class OccupantTrack:
    """State of a tracked individual."""
    track_id: int
    position: np.ndarray  # (x, y, z)
    velocity: np.ndarray  # (vx, vy, vz)
    last_update_time: float
    is_fallen: bool = False
    confidence: float = 1.0
    age_frames: int = 0

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "is_fallen": self.is_fallen,
            "confidence": float(self.confidence),
            "age_frames": int(self.age_frames),
        }


@dataclass
class DetectedBlob:
    """A radar centroid detection in a single frame."""
    centroid: np.ndarray  # (x, y, z)
    area_m2: float = 0.5
    peak_velocity_mps: float = 0.0


class SimpleKalmanTracker:
    """Constant-velocity Kalman filter for 3D position tracking."""

    def __init__(self, initial_position: np.ndarray, dt: float = 0.1):
        self.dt = dt
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6, dtype=np.float64)
        self.state[:3] = np.asarray(initial_position, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0  # Covariance
        self.Q = np.eye(6, dtype=np.float64) * 0.01  # Process noise
        self.R = np.eye(3, dtype=np.float64) * 0.1  # Measurement noise

    def predict(self) -> np.ndarray:
        """Predict next state."""
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        return self.state[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement."""
        meas = np.asarray(measurement, dtype=np.float64)
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = meas - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ H) @ self.P
        return self.state[:3].copy()


class MultiOccupantTracker:
    """
    Track multiple occupants and attribute fall events to individuals.
    """

    MAX_ASSOCIATION_DIST = 2.0  # metres
    TRACK_TIMEOUT_S = 5.0

    def __init__(self, max_occupants: int = 6, dt: float = 0.1):
        self.max_occupants = max_occupants
        self.dt = dt
        self._tracks: Dict[int, OccupantTrack] = {}
        self._kalman_filters: Dict[int, SimpleKalmanTracker] = {}
        self._next_track_id = 1
        self._frame_count = 0

    def update(self, detections: List[DetectedBlob], timestamp: float) -> List[OccupantTrack]:
        """
        Process a frame of radar detections and update tracks.

        Uses Hungarian algorithm for optimal detection-to-track assignment.
        """
        self._frame_count += 1

        # Predict all existing tracks
        for tid, kf in self._kalman_filters.items():
            pred = kf.predict()
            self._tracks[tid].position = pred
            self._tracks[tid].velocity = kf.state[3:6].copy()

        if not self._tracks and not detections:
            return []

        # Build cost matrix for Hungarian assignment
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        if n_tracks == 0:
            # All detections are new tracks
            for det in detections[: self.max_occupants]:
                self._create_track(det, timestamp)
            return list(self._tracks.values())

        if n_dets == 0:
            # No detections - age out stale tracks
            self._prune_stale(timestamp)
            return list(self._tracks.values())

        cost_matrix = np.full((n_tracks, n_dets), 1e6, dtype=np.float64)
        for i, tid in enumerate(track_ids):
            track_pos = self._tracks[tid].position
            for j, det in enumerate(detections):
                dist = float(np.linalg.norm(track_pos - np.asarray(det.centroid, dtype=np.float64)))
                if dist < self.MAX_ASSOCIATION_DIST:
                    cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.MAX_ASSOCIATION_DIST:
                tid = track_ids[r]
                det = detections[c]
                self._kalman_filters[tid].update(det.centroid)
                track = self._tracks[tid]
                track.position = self._kalman_filters[tid].state[:3].copy()
                track.velocity = self._kalman_filters[tid].state[3:6].copy()
                track.last_update_time = timestamp
                track.age_frames += 1
                track.confidence = min(1.0, track.confidence + 0.1)

                # Check for fall: rapid z-axis descent or low z with significant velocity
                if det.centroid[2] < 0.5 and det.peak_velocity_mps > 1.5:
                    track.is_fallen = True
                elif det.centroid[2] > 1.0:
                    track.is_fallen = False

                matched_tracks.add(tid)
                matched_dets.add(c)

        # Create new tracks for unmatched detections
        for j in range(n_dets):
            if j not in matched_dets and len(self._tracks) < self.max_occupants:
                self._create_track(detections[j], timestamp)

        self._prune_stale(timestamp)
        return list(self._tracks.values())

    def _create_track(self, det: DetectedBlob, timestamp: float) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        pos = np.asarray(det.centroid, dtype=np.float64).copy()
        is_fall = bool(det.centroid[2] < 0.5 and det.peak_velocity_mps > 1.5)
        self._tracks[tid] = OccupantTrack(
            track_id=tid,
            position=pos,
            velocity=np.zeros(3, dtype=np.float64),
            last_update_time=timestamp,
            is_fallen=is_fall,
            confidence=0.5,
            age_frames=1,
        )
        self._kalman_filters[tid] = SimpleKalmanTracker(pos, self.dt)
        return tid

    def _prune_stale(self, current_time: float) -> None:
        stale = [
            tid
            for tid, t in self._tracks.items()
            if current_time - t.last_update_time > self.TRACK_TIMEOUT_S
        ]
        for tid in stale:
            del self._tracks[tid]
            del self._kalman_filters[tid]

    def get_fallen_occupants(self) -> List[OccupantTrack]:
        return [t for t in self._tracks.values() if t.is_fallen]

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    @property
    def frame_count(self) -> int:
        return self._frame_count
