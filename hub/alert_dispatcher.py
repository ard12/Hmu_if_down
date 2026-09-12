"""Unified alert dispatcher for sirens, logging, and smart home notifications."""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
from typing import Optional
import urllib.request

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
        mqtt_broker: Optional[str] = None,
        mqtt_port: int = 1883,
        mqtt_topic: str = "falldetect/alert",
        webhook_url: Optional[str] = None,
    ):
        self.enable_sound = enable_sound
        self.beep_freq = beep_freq
        self.beep_duration_ms = beep_duration_ms
        self.cooldown_sec = cooldown_sec

        self.last_alert_time: float = 0.0
        self._beeping = False
        self._lock = threading.Lock()

        # MQTT and webhook alerting
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self._mqtt_topic = mqtt_topic
        self.webhook_url = webhook_url
        self._webhook_url = webhook_url
        self._mqtt_client = None

        if mqtt_broker:
            try:
                import paho.mqtt.client as mqtt
                try:
                    if hasattr(mqtt, "CallbackAPIVersion"):
                        self._mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                    else:
                        self._mqtt_client = mqtt.Client()
                except Exception:
                    self._mqtt_client = mqtt.Client()
                self._mqtt_client.connect(mqtt_broker, mqtt_port)
            except ImportError:
                print("[AlertDispatcher] Warning: paho-mqtt is not installed. MQTT alerting disabled.")
                self._mqtt_client = None
            except Exception as e:
                print(f"[AlertDispatcher] Warning: Failed to connect to MQTT broker ({mqtt_broker}:{mqtt_port}): {e}")
                self._mqtt_client = None

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
        with self._lock:
            now = time.time()
            if now - self.last_alert_time < self.cooldown_sec:
                return

            self.last_alert_time = now

            # Play sound in background thread
            if self.enable_sound and not self._beeping:
                threading.Thread(target=self._beep_worker, daemon=True).start()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log incident to disk
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, modality, event_name, details])
                print(f"\n[ALERT] [{timestamp}] {modality.upper()} -> {event_name}: {details}")
            except Exception as e:
                print(f"[AlertDispatcher] Logging error: {e}", file=sys.stderr)

            # JSON payload for external alert dispatchers
            payload = {
                "timestamp": timestamp,
                "modality": modality,
                "event": event_name,
                "details": details,
            }
            payload_json = json.dumps(payload)

            # MQTT notification
            if self._mqtt_client is not None:
                try:
                    self._mqtt_client.publish(self._mqtt_topic, payload_json)
                except Exception as e:
                    print(f"[AlertDispatcher] MQTT publish error: {e}", file=sys.stderr)

            # Webhook notification
            if self._webhook_url is not None:
                try:
                    req = urllib.request.Request(
                        self._webhook_url,
                        data=payload_json.encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    resp = urllib.request.urlopen(req, timeout=5.0)
                    if hasattr(resp, "close"):
                        resp.close()
                except Exception as e:
                    print(f"[AlertDispatcher] Webhook error: {e}", file=sys.stderr)
