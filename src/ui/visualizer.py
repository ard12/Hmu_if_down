"""Visualizer module for rendering skeletons, bounding boxes, and HUD metrics."""

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from ..detector.fall_logic import FallMetrics, FallState
from ..detector.pose_estimator import PoseLandmarks


class Visualizer:
    """Renders real-time HUD, bounding box, skeleton, and alarm banners on video frames."""

    # Colors (BGR)
    COLOR_GREEN = (0, 220, 0)
    COLOR_YELLOW = (0, 200, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 180, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (20, 20, 20)
    COLOR_GRAY = (80, 80, 80)

    def __init__(self, draw_skeleton: bool = True):
        self.draw_skeleton = draw_skeleton

    def _get_state_color(self, state: FallState) -> Tuple[int, int, int]:
        if state == FallState.FALL_CONFIRMED:
            return self.COLOR_RED
        elif state == FallState.FALL_SUSPECTED:
            return self.COLOR_YELLOW
        elif state == FallState.RECOVERED:
            return self.COLOR_BLUE
        return self.COLOR_GREEN

    def draw(
        self,
        frame: np.ndarray,
        landmarks: Optional[PoseLandmarks],
        metrics: FallMetrics,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Render HUD, skeleton, and alerts onto the frame."""
        output = frame.copy()
        h, w = output.shape[:2]
        state_color = self._get_state_color(metrics.state)

        # 1. Draw skeleton if landmarks available
        if self.draw_skeleton and landmarks and landmarks.raw_landmarks and MEDIAPIPE_AVAILABLE:
            mp_drawing.draw_landmarks(
                output,
                landmarks.raw_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=state_color, thickness=2, circle_radius=2),
            )

        # 2. Draw bounding box
        if landmarks:
            bbox = landmarks.get_bbox(w, h)
            if bbox:
                xmin, ymin, xmax, ymax = bbox
                thickness = 4 if metrics.state == FallState.FALL_CONFIRMED else 2
                cv2.rectangle(output, (xmin, ymin), (xmax, ymax), state_color, thickness)

                # Label on top of box
                label = f"{metrics.state.value}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    output,
                    (xmin, max(0, ymin - lh - 8)),
                    (xmin + lw + 8, ymin),
                    state_color,
                    -1,
                )
                cv2.putText(
                    output,
                    label,
                    (xmin + 4, max(lh + 2, ymin - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.COLOR_WHITE,
                    2,
                    cv2.LINE_AA,
                )

        # 3. Top Alert Banner
        if metrics.state == FallState.FALL_CONFIRMED:
            # Flashing red emergency banner across top
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), self.COLOR_RED, -1)
            cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
            text = "EMERGENCY: FALL DETECTED!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            cv2.putText(
                output,
                text,
                ((w - tw) // 2, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                self.COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )
        elif metrics.state == FallState.FALL_SUSPECTED:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w, 45), self.COLOR_YELLOW, -1)
            cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)
            text = f"WARNING: FALL SUSPECTED (Confirming: {metrics.consecutive_fall_frames})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(
                output,
                text,
                ((w - tw) // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                self.COLOR_BLACK,
                2,
                cv2.LINE_AA,
            )

        # 4. Metrics HUD (Semi-transparent card on top-left)
        hud_w, hud_h = 240, 120
        hud_x, hud_y = 15, 75 if metrics.state in (FallState.FALL_CONFIRMED, FallState.FALL_SUSPECTED) else 20
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            self.COLOR_BLACK,
            -1,
        )
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        cv2.rectangle(
            output,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            self.COLOR_GRAY,
            1,
        )

        # Text in HUD
        hud_lines = [
            (f"Status: {metrics.state.value}", state_color),
            (f"Torso Angle: {metrics.torso_angle:.1f} deg", self.COLOR_WHITE),
            (f"Aspect Ratio: {metrics.aspect_ratio:.2f}", self.COLOR_WHITE),
            (f"FPS: {fps:.1f}", (180, 180, 180)),
        ]

        text_y = hud_y + 25
        for line, col in hud_lines:
            cv2.putText(
                output,
                line,
                (hud_x + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col,
                1,
                cv2.LINE_AA,
            )
            text_y += 24

        return output
