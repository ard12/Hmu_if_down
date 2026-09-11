"""Unified alert dispatcher for sirens, logging, and smart home notifications."""

import csv
from datetime import datetime
from pathlib import Path
import sys
import threading
import time
from typing import Optional

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AlertDispatcher:
    """Dispatches audio alarms, handles rate-limiting cooldowns, and logs incidents."""

    def __init__(
        self,
        enable_sound: bool = True,
        beep_freq: int = 1200,
        beep_duration_ms: int = 600,
        cooldown_sec: float = 5.0,
        log_dir: str = "incidents",
    ):
        self.enable_sound = enable_sound
        self.beep_freq = beep_freq
        self.beep_duration_ms = beep_duration_ms
        self.cooldown_sec = cooldown_sec

        self.last_alert_time: float = 0.0
        self._beeping = False

        # Set up CSV incident logger
        self.log_dir = Path(__file__).resolve().parent.parent / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "incidents.csv"
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Modality", "Event", "Details"])

    def _beep_worker(self):
        self._beeping = True
        try:
            if WINSOUND_AVAILABLE:
                # 3 quick urgent alarm beeps
                for _ in range(3):
                    winsound.Beep(self.beep_freq, 150)
                    time.sleep(0.05)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            self._beeping = False

    def trigger_alarm(self, modality: str, event_name: str, details: str = ""):
        """Trigger siren and record incident if outside cooldown window."""
        now = time.time()
        if now - self.last_alert_time < self.cooldown_sec:
            return

        self.last_alert_time = now

        # Play sound in background thread
        if self.enable_sound and not self._beeping:
            threading.Thread(target=self._beep_worker, daemon=True).start()

        # Log incident to disk
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, modality, event_name, details])
            print(f"\n[ALERT] [{timestamp}] {modality.upper()} -> {event_name}: {details}")
        except Exception as e:
            print(f"[AlertDispatcher] Logging error: {e}", file=sys.stderr)
