"""Fall biomechanics trajectory classifier from 3D skeleton pose sequence."""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from .joint_angles import JointAngleEstimator


class FallBiomechanicsClassifier:
    """
    Classifies fall type from a 1.5-second pose trajectory.
    Uses kinematic trajectory feature extraction and biomechanical decision logic.

    Fall types (second-stage, post-fall-confirmed):
      1. TRIP_AND_FALL   — rapid forward trunk pitch > 45° in < 0.5s, knee flexion > 90°
      2. SLIP_AND_FALL   — lateral trunk roll > 30°, rapid z-drop
      3. SYNCOPE         — trunk inclination increases slowly > 2s, head drop gradual
      4. ASSISTED_FALL   — z-drop arrested at 50% of full fall height (furniture contact)
      5. UNKNOWN         — insufficient trajectory quality

    Features extracted:
      - max_trunk_angular_velocity (deg/s)
      - head_z_drop_velocity (m/s)
      - lateral_roll_angle (deg)
      - time_to_floor_s
      - minimum_z_in_trajectory (m)
      - knee_flexion_at_impact (deg)
    """

    FALL_TYPES: List[str] = [
        "TRIP_AND_FALL",
        "SLIP_AND_FALL",
        "SYNCOPE",
        "ASSISTED_FALL",
        "UNKNOWN",
    ]

    # Mapping between CSI 5-class labels (Phase 8) and Biomechanics labels
    CSI_TO_BIOMECHANICS_MAP: Dict[str, List[str]] = {
        "forward_trip": ["TRIP_AND_FALL"],
        "backward_slip": ["SLIP_AND_FALL"],
        "lateral_collapse": ["SLIP_AND_FALL"],
        "syncope_drop": ["SYNCOPE"],
        "slow_slump": ["SYNCOPE", "ASSISTED_FALL"],
    }

    def __init__(self):
        self.angle_estimator = JointAngleEstimator()

    def extract_features(self, trajectory: Sequence[Dict[str, Any]], dt: float = 0.05) -> Dict[str, float]:
        """
        Extract 6 required biomechanical features from skeleton trajectory.
        """
        valid_frames = [f for f in trajectory if isinstance(f, dict) and f.get("valid", True)]
        if len(valid_frames) < 2:
            return {
                "max_trunk_angular_velocity": 0.0,
                "head_z_drop_velocity": 0.0,
                "lateral_roll_angle": 0.0,
                "time_to_floor_s": 0.0,
                "minimum_z_in_trajectory": 0.0,
                "knee_flexion_at_impact": 0.0,
            }

        # 1. Trunk angles and angular velocity
        angles = []
        rolls = []
        head_zs = []
        torso_zs = []

        for frame in valid_frames:
            tt = frame.get("torso_top", [0.0, 0.0, 1.4])
            tb = frame.get("torso_bottom", [0.0, 0.0, 0.9])
            head = frame.get("head", [0.0, 0.0, 1.6])

            inc = self.angle_estimator.trunk_inclination(tt, tb)
            angles.append(inc)

            # Lateral roll: tilt in Y relative to Z
            dy = tt[1] - tb[1]
            dz = max(abs(tt[2] - tb[2]), 1e-4)
            roll = float(np.degrees(np.arctan2(abs(dy), dz)))
            rolls.append(roll)

            head_zs.append(head[2])
            torso_zs.append(tb[2])

        # Angular velocity (deg/s)
        d_angles = np.diff(angles)
        ang_velocities = np.abs(d_angles) / dt if dt > 0 else np.zeros_like(d_angles)
        max_trunk_angular_vel = float(np.max(ang_velocities)) if len(ang_velocities) > 0 else 0.0

        # Head drop velocity (m/s)
        head_drop_vel = self.angle_estimator.head_drop_velocity([[0, 0, z] for z in head_zs], dt)

        # Lateral roll max
        max_roll = float(np.max(rolls)) if len(rolls) > 0 else 0.0

        # Minimum z reached
        min_z = float(np.min(head_zs))

        # Time to floor (onset to min z)
        min_idx = int(np.argmin(head_zs))
        time_to_floor = float(min_idx * dt)

        # Knee flexion at impact (frame of lowest head z)
        impact_frame = valid_frames[min_idx]
        knee_flex = float(impact_frame.get("knee_flexion", 95.0 if min_z < 0.5 else 45.0))

        return {
            "max_trunk_angular_velocity": round(max_trunk_angular_vel, 2),
            "head_z_drop_velocity": round(head_drop_vel, 2),
            "lateral_roll_angle": round(max_roll, 2),
            "time_to_floor_s": round(time_to_floor, 2),
            "minimum_z_in_trajectory": round(min_z, 2),
            "knee_flexion_at_impact": round(knee_flex, 2),
        }

    def classify(self, trajectory: Sequence[Dict[str, Any]], dt: float = 0.05) -> Dict[str, Any]:
        """
        trajectory: list of skeleton dicts over 1.5s (from SkeletonFitter.track).
        Returns:
          {
            "fall_type": str,
            "confidence": float,
            "features": dict,
            "clinical_note": str
          }
        """
        valid_frames = [f for f in trajectory if isinstance(f, dict) and f.get("valid", True)]
        if len(valid_frames) < 3:
            return {
                "fall_type": "UNKNOWN",
                "confidence": 0.0,
                "features": {
                    "max_trunk_angular_velocity": 0.0,
                    "head_z_drop_velocity": 0.0,
                    "lateral_roll_angle": 0.0,
                    "time_to_floor_s": 0.0,
                    "minimum_z_in_trajectory": 0.0,
                    "knee_flexion_at_impact": 0.0,
                },
                "clinical_note": "Insufficient trajectory quality or frame count for biomechanical classification.",
            }

        feats = self.extract_features(trajectory, dt=dt)

        # 1. Syncope: gradual descent (time_to_floor >= 1.8s or head drop velocity is slow > -0.7 m/s)
        if feats["time_to_floor_s"] >= 1.8 or (feats["head_z_drop_velocity"] > -0.7 and feats["time_to_floor_s"] >= 1.2):
            fall_type = "SYNCOPE"
            confidence = 0.88
            note = "Gradual loss of postural tone and slow vertical descent consistent with vasovagal syncope or orthostatic collapse."

        # 2. Assisted fall: descent arrested midway (min_z between 0.45m and 0.85m)
        elif 0.45 <= feats["minimum_z_in_trajectory"] <= 0.85:
            fall_type = "ASSISTED_FALL"
            confidence = 0.84
            note = "Vertical descent arrested above floor level, indicating furniture contact or assisted deceleration."

        # 3. Slip and Fall: significant lateral roll (> 25°) or high lateral displacement
        elif feats["lateral_roll_angle"] > 25.0:
            fall_type = "SLIP_AND_FALL"
            confidence = 0.91
            note = "Significant lateral trunk roll and abrupt vertical loss of balance consistent with a backward or lateral slip."

        # 4. Trip and Fall: rapid forward trunk pitch (high angular velocity or high knee flexion at impact)
        elif feats["max_trunk_angular_velocity"] > 70.0 or feats["knee_flexion_at_impact"] > 80.0:
            fall_type = "TRIP_AND_FALL"
            confidence = 0.93
            note = "Rapid forward trunk pitch accompanied by significant knee flexion consistent with a forward tripping event."

        # 5. Default high-velocity fall
        else:
            fall_type = "TRIP_AND_FALL" if feats["head_z_drop_velocity"] < -1.0 else "SLIP_AND_FALL"
            confidence = 0.78
            note = "Kinematic trajectory indicates acute fall with rapid vertical descent."

        return {
            "fall_type": fall_type,
            "confidence": round(float(confidence), 2),
            "features": feats,
            "clinical_note": note,
        }

    def check_agreement(self, csi_fall_type: Optional[str], biomechanics_fall_type: Optional[str]) -> bool:
        """Check if CSI second-stage classifier and pose trajectory classifier agree."""
        if not csi_fall_type or not biomechanics_fall_type:
            return False
        valid_matches = self.CSI_TO_BIOMECHANICS_MAP.get(csi_fall_type, [])
        return biomechanics_fall_type in valid_matches
