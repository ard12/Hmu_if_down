"""Kinematic heuristic fall detection logic and state machine."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Deque, Dict, Optional, Tuple

from .pose_estimator import PoseLandmarks


class FallState(Enum):
    NORMAL = "Normal"
    FALL_SUSPECTED = "Fall Suspected"
    FALL_CONFIRMED = "FALL DETECTED"
    RECOVERED = "Recovered"


@dataclass
class FallMetrics:
    torso_angle: float          # Degrees from vertical [0.0 - 90.0]
    aspect_ratio: float         # Bounding box H / W
    vertical_velocity: float    # Downward centroid displacement
    is_inclined: bool           # Whether angle exceeds threshold
    is_horizontal: bool         # Whether aspect ratio is < threshold
    is_falling_velocity: bool   # Whether downward velocity was triggered
    state: FallState
    consecutive_fall_frames: int


class FallDetector:
    """Detects falls based on posture kinematics, aspect ratio, and vertical descent."""

    def __init__(
        self,
        torso_angle_threshold: float = 55.0,
        aspect_ratio_threshold: float = 0.85,
        fall_velocity_threshold: float = 0.04,
        confirmation_frames: int = 10,
        recovery_frames: int = 15,
        velocity_history_size: int = 6,
    ):
        self.torso_angle_threshold = torso_angle_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.fall_velocity_threshold = fall_velocity_threshold
        self.confirmation_frames = confirmation_frames
        self.recovery_frames = recovery_frames

        # State tracking
        self.current_state = FallState.NORMAL
        self.consecutive_fall_frames = 0
        self.consecutive_normal_frames = 0
        self.recent_centroids: Deque[float] = deque(maxlen=velocity_history_size)

    @staticmethod
    def calculate_torso_angle(landmarks: PoseLandmarks) -> Optional[float]:
        """Compute the torso inclination angle in degrees relative to the vertical axis.
        
        0 degrees = perfectly upright
        90 degrees = completely horizontal
        """
        l_shoulder = landmarks.get("left_shoulder")
        r_shoulder = landmarks.get("right_shoulder")
        l_hip = landmarks.get("left_hip")
        r_hip = landmarks.get("right_hip")

        if not (l_shoulder and r_shoulder and l_hip and r_hip):
            return None

        # Require reasonable visibility
        if min(l_shoulder.visibility, r_shoulder.visibility, l_hip.visibility, r_hip.visibility) < 0.35:
            return None

        # Midpoint of shoulders
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0

        # Midpoint of hips
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        # Vector from shoulder to hip (downward along torso)
        dx = mid_hip_x - mid_shoulder_x
        dy = mid_hip_y - mid_shoulder_y

        magnitude = math.hypot(dx, dy)
        if magnitude < 1e-6:
            return 0.0

        # Angle with downward vertical vector (0, 1)
        # cos(theta) = dot(v, [0, 1]) / |v| = dy / magnitude
        cos_theta = dy / magnitude
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        # We constrain the angle to [0, 90] representing deviation from vertical
        if angle_deg > 90.0:
            angle_deg = 180.0 - angle_deg

        return angle_deg

    def calculate_aspect_ratio(
        self, landmarks: PoseLandmarks, frame_width: int, frame_height: int
    ) -> Optional[float]:
        """Calculate bounding box Height / Width ratio."""
        bbox = landmarks.get_bbox(frame_width, frame_height)
        if not bbox:
            return None

        xmin, ymin, xmax, ymax = bbox
        width = max(1, xmax - xmin)
        height = max(1, ymax - ymin)

        return float(height) / float(width)

    def calculate_vertical_velocity(self, landmarks: PoseLandmarks) -> float:
        """Compute the vertical downward velocity of the hip centroid."""
        l_hip = landmarks.get("left_hip")
        r_hip = landmarks.get("right_hip")

        if not (l_hip and r_hip):
            return 0.0

        mid_hip_y = (l_hip.y + r_hip.y) / 2.0
        self.recent_centroids.append(mid_hip_y)

        if len(self.recent_centroids) < self.recent_centroids.maxlen:
            return 0.0

        # Displacement over window (positive = moving downward on screen)
        displacement = self.recent_centroids[-1] - self.recent_centroids[0]
        return max(0.0, displacement)

    def evaluate(
        self,
        landmarks: Optional[PoseLandmarks],
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> FallMetrics:
        """Evaluate landmarks and advance state machine."""
        if landmarks is None:
            return FallMetrics(
                torso_angle=0.0,
                aspect_ratio=1.0,
                vertical_velocity=0.0,
                is_inclined=False,
                is_horizontal=False,
                is_falling_velocity=False,
                state=self.current_state,
                consecutive_fall_frames=self.consecutive_fall_frames,
            )

        angle = self.calculate_torso_angle(landmarks) or 0.0
        ar = self.calculate_aspect_ratio(landmarks, frame_width, frame_height) or 1.0
        vel = self.calculate_vertical_velocity(landmarks)

        is_inclined = angle >= self.torso_angle_threshold
        is_horizontal = ar <= self.aspect_ratio_threshold
        is_falling_vel = vel >= self.fall_velocity_threshold

        # Posture matches a fallen/lying state if significantly inclined or horizontal
        posture_down = is_inclined or is_horizontal

        # State transition logic
        if posture_down:
            self.consecutive_fall_frames += 1
            self.consecutive_normal_frames = 0

            if self.consecutive_fall_frames >= self.confirmation_frames:
                self.current_state = FallState.FALL_CONFIRMED
            else:
                self.current_state = FallState.FALL_SUSPECTED
        else:
            self.consecutive_normal_frames += 1
            if self.consecutive_normal_frames >= self.recovery_frames:
                if self.current_state == FallState.FALL_CONFIRMED:
                    self.current_state = FallState.RECOVERED
                else:
                    self.current_state = FallState.NORMAL
                self.consecutive_fall_frames = 0

        return FallMetrics(
            torso_angle=angle,
            aspect_ratio=ar,
            vertical_velocity=vel,
            is_inclined=is_inclined,
            is_horizontal=is_horizontal,
            is_falling_velocity=is_falling_vel,
            state=self.current_state,
            consecutive_fall_frames=self.consecutive_fall_frames,
        )

    def reset(self):
        """Reset internal state machine counters."""
        self.current_state = FallState.NORMAL
        self.consecutive_fall_frames = 0
        self.consecutive_normal_frames = 0
        self.recent_centroids.clear()
