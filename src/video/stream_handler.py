"""Video stream handler supporting webcams, video files, and RTSP streams."""

from typing import Optional, Tuple, Union
import cv2
import numpy as np


class StreamHandler:
    """Manages video capture, resolution scaling, and frame iteration."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ):
        self.source = source
        self.target_width = target_width
        self.target_height = target_height

        # Parse source if it's a numeric string (e.g. "0")
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Probe actual resolution
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and optionally resize next frame."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.target_width and self.target_height:
            frame = cv2.resize(
                frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA
            )

        return True, frame

    def release(self):
        """Release video capture device."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
