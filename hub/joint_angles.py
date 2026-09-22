"""Biomechanically relevant joint angle estimator and posture classifier."""

from typing import Any, Dict, List, Sequence
import numpy as np


class JointAngleEstimator:
    """
    Computes biomechanically relevant joint angles from skeleton keypoints.
    """

    def trunk_inclination(self, torso_top: Sequence[float], torso_bottom: Sequence[float]) -> float:
        """
        Returns trunk inclination from vertical (degrees).
        0° = fully upright; 90° = horizontal (fallen).
        """
        v = np.array(torso_top, dtype=float) - np.array(torso_bottom, dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return 0.0

        # Cosine with the upward vertical unit vector [0, 0, 1]
        cos_theta = np.clip(v[2] / norm, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        return float(np.degrees(angle_rad))

    def knee_flexion(
        self,
        hip: Sequence[float],
        knee: Sequence[float],
        ankle: Sequence[float],
    ) -> float:
        """
        Returns knee flexion angle (degrees).
        0° = fully extended; 90° = sitting; 130°+ = kneeling.
        """
        v_thigh = np.array(hip, dtype=float) - np.array(knee, dtype=float)
        v_shank = np.array(ankle, dtype=float) - np.array(knee, dtype=float)

        norm_thigh = np.linalg.norm(v_thigh)
        norm_shank = np.linalg.norm(v_shank)
        if norm_thigh < 1e-6 or norm_shank < 1e-6:
            return 0.0

        cos_phi = np.clip(np.dot(v_thigh, v_shank) / (norm_thigh * norm_shank), -1.0, 1.0)
        phi_deg = np.degrees(np.arccos(cos_phi))
        # Flexion is deviation from full 180° extension
        flexion = 180.0 - phi_deg
        return float(np.clip(flexion, 0.0, 180.0))

    def head_drop_velocity(self, head_positions: Sequence[Sequence[float]], dt: float) -> float:
        """
        Computes vertical velocity of head centroid (m/s).
        Negative = falling downward. Threshold < -0.8 m/s indicates fall onset.
        """
        if len(head_positions) < 2 or dt <= 0.0:
            return 0.0

        total_time = (len(head_positions) - 1) * dt
        delta_z = head_positions[-1][2] - head_positions[0][2]
        return float(delta_z / total_time)

    def classify_posture(self, skeleton: Dict[str, Any]) -> str:
        """
        Returns: "STANDING" | "SITTING" | "BENDING" | "FALLEN" | "UNKNOWN"
        Based on trunk_inclination and max_z of skeleton.
        """
        if not skeleton.get("valid", True):
            return "UNKNOWN"

        torso_top = skeleton.get("torso_top")
        torso_bottom = skeleton.get("torso_bottom")
        if torso_top is None or torso_bottom is None:
            return "UNKNOWN"

        inclination = self.trunk_inclination(torso_top, torso_bottom)
        head = skeleton.get("head", [0.0, 0.0, 0.0])
        max_z = max(head[2], torso_top[2])

        if inclination > 60.0 or max_z <= 0.50:
            return "FALLEN"
        elif inclination > 35.0 and max_z > 0.80:
            return "BENDING"
        elif torso_bottom[2] < 0.65 or max_z < 1.30:
            return "SITTING"
        else:
            return "STANDING"
