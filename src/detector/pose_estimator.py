"""Pose estimation module using MediaPipe Pose."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class LandmarkPoint:
    x: float  # Normalized [0.0, 1.0]
    y: float  # Normalized [0.0, 1.0]
    z: float  # Depth
    visibility: float  # Confidence [0.0, 1.0]


@dataclass
class PoseLandmarks:
    """Standardized landmarks container independent of underlying framework."""
    points: Dict[str, LandmarkPoint]
    raw_landmarks: Optional[Any] = None

    def get(self, name: str) -> Optional[LandmarkPoint]:
        return self.points.get(name)

    def get_bbox(self, frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate pixel bounding box (xmin, ymin, xmax, ymax)."""
        valid_points = [
            pt for pt in self.points.values()
            if pt.visibility > 0.4
        ]
        if not valid_points:
            return None

        xs = [pt.x * frame_width for pt in valid_points]
        ys = [pt.y * frame_height for pt in valid_points]

        xmin, xmax = max(0, int(min(xs))), min(frame_width, int(max(xs)))
        ymin, ymax = max(0, int(min(ys))), min(frame_height, int(max(ys)))

        return xmin, ymin, xmax, ymax


# Mapping common landmark names to MediaPipe indices
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class PoseEstimator:
    """Extracts human skeletal landmarks using MediaPipe Pose."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is required for PoseEstimator. Install via `pip install mediapipe`"
            )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            static_image_mode=False,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        """Process a BGR image frame and return extracted PoseLandmarks if detected."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        points: Dict[str, LandmarkPoint] = {}
        for idx, name in LANDMARK_NAMES.items():
            lm = results.pose_landmarks.landmark[idx]
            points[name] = LandmarkPoint(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )

        return PoseLandmarks(points=points, raw_landmarks=results.pose_landmarks)

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "pose") and self.pose:
            self.pose.close()
