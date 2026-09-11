"""Alert management system for audio signals and incident logging."""

import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

from ..detector.fall_logic import FallMetrics, FallState
from ..utils import get_project_root

# Try importing Windows winsound
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AlertManager:
    """Manages audio alarms, event cooldowns, and incident disk logging."""

    def __init__(
        self,
        enable_sound: bool = True,
        beep_frequency: int = 1000,
        beep_duration_ms: int = 500,
        cooldown_seconds: float = 5.0,
        enable_logging: bool = True,
        log_dir: str = "incidents",
    ):
        self.enable_sound = enable_sound
        self.beep_frequency = beep_frequency
        self.beep_duration_ms = beep_duration_ms
        self.cooldown_seconds = cooldown_seconds
        self.enable_logging = enable_logging

        self.last_alert_time: float = 0.0
        self._is_beeping = False

        # Set up log directory and incident CSV
        self.log_dir = get_project_root() / log_dir
        if self.enable_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_file = self.log_dir / "incidents.csv"
            self._init_csv()

    def _init_csv(self):
        """Initialize CSV log with header if not exists."""
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "State",
                    "TorsoAngleDeg",
                    "AspectRatio",
                    "ConsecutiveFrames",
                ])

    def _play_beep_async(self):
        """Play sound in background thread so video stream does not stutter."""
        def _beep_worker():
            self._is_beeping = True
            try:
                if WINSOUND_AVAILABLE:
                    winsound.Beep(self.beep_frequency, self.beep_duration_ms)
                else:
                    # Terminal bell fallback for non-Windows platforms
                    sys.stdout.write("\a")
                    sys.stdout.flush()
            except Exception:
                pass
            finally:
                self._is_beeping = False

        thread = threading.Thread(target=_beep_worker, daemon=True)
        thread.start()

    def trigger(self, metrics: FallMetrics):
        """Evaluate fall metrics and sound alarm / log incident if condition met."""
        if metrics.state != FallState.FALL_CONFIRMED:
            return

        now = time.time()
        # Cooldown check
        if now - self.last_alert_time < self.cooldown_seconds:
            return

        self.last_alert_time = now

        # Sound alarm
        if self.enable_sound and not self._is_beeping:
            self._play_beep_async()

        # Log to CSV
        if self.enable_logging:
            self._log_incident(metrics)

    def _log_incident(self, metrics: FallMetrics):
        """Append incident row to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    metrics.state.value,
                    f"{metrics.torso_angle:.1f}",
                    f"{metrics.aspect_ratio:.2f}",
                    metrics.consecutive_fall_frames,
                ])
        except Exception as e:
            print(f"[AlertManager] Error logging incident: {e}", file=sys.stderr)
