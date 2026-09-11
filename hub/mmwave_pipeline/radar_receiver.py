"""Radar packet receiver, binary protocol parser, and telemetry decoder."""

from dataclasses import dataclass
from enum import Enum
import json
from typing import Callable, Optional, Tuple


class RadarFallState(Enum):
    NONE = "Normal"
    SUSPECTED = "Fall Suspected"
    CONFIRMED = "FALL DETECTED"
    DWELLING = "Dwelling on Floor"


class RadarPosture(Enum):
    UNKNOWN = "Unknown"
    STANDING = "Standing"
    SITTING = "Sitting"
    LYING = "Lying Down"


@dataclass
class RadarTelemetry:
    fall_state: RadarFallState
    posture: RadarPosture
    target_height_m: float
    dwell_time_sec: int
    raw_valid: bool = True


class RadarReceiver:
    """Decodes binary radar packets or JSON UDP datagrams from ESP32 mmWave gateway."""

    HEADER = b"\x53\x59"

    def __init__(self, callback: Optional[Callable[[RadarTelemetry], None]] = None):
        self.callback = callback
        self.current_telemetry = RadarTelemetry(
            fall_state=RadarFallState.NONE,
            posture=RadarPosture.UNKNOWN,
            target_height_m=1.6,
            dwell_time_sec=0,
        )

    def parse_json_datagram(self, payload: str) -> Optional[RadarTelemetry]:
        """Decode JSON string forwarded by ESP32 gateway:
        {"fall": 2, "posture": 3, "height": 0.25, "dwell": 5}
        """
        try:
            data = json.loads(payload)
            fall_map = {
                0: RadarFallState.NONE,
                1: RadarFallState.SUSPECTED,
                2: RadarFallState.CONFIRMED,
                3: RadarFallState.DWELLING,
            }
            posture_map = {
                0: RadarPosture.UNKNOWN,
                1: RadarPosture.STANDING,
                2: RadarPosture.SITTING,
                3: RadarPosture.LYING,
            }

            telemetry = RadarTelemetry(
                fall_state=fall_map.get(data.get("fall", 0), RadarFallState.NONE),
                posture=posture_map.get(data.get("posture", 0), RadarPosture.UNKNOWN),
                target_height_m=float(data.get("height", 1.6)),
                dwell_time_sec=int(data.get("dwell", 0)),
            )
            self.current_telemetry = telemetry
            if self.callback:
                self.callback(telemetry)
            return telemetry
        except Exception:
            return None

    def parse_binary_frame(self, frame: bytes) -> Optional[RadarTelemetry]:
        """Parse raw binary packet conforming to 60GHz MR60FDA1 protocol."""
        if len(frame) < 7 or frame[:2] != self.HEADER:
            return None

        # Checksum check
        checksum = sum(frame[:-1]) & 0xFF
        if checksum != frame[-1]:
            return None

        ctrl = frame[2]
        cmd = frame[3]
        data_len = (frame[4] << 8) | frame[5]
        payload = frame[6 : 6 + data_len]

        updated = False

        # Control Word 0x02: Fall Detection
        if ctrl == 0x02:
            if cmd == 0x01 and len(payload) >= 1:
                val = payload[0]
                if val == 0:
                    self.current_telemetry.fall_state = RadarFallState.NONE
                elif val == 1:
                    self.current_telemetry.fall_state = RadarFallState.SUSPECTED
                elif val == 2:
                    self.current_telemetry.fall_state = RadarFallState.CONFIRMED
                elif val == 3:
                    self.current_telemetry.fall_state = RadarFallState.DWELLING
                updated = True
            elif cmd == 0x02 and len(payload) >= 2:
                self.current_telemetry.dwell_time_sec = (payload[0] << 8) | payload[1]
                updated = True

        # Control Word 0x03: Posture & Height
        elif ctrl == 0x03:
            if cmd == 0x01 and len(payload) >= 1:
                pval = payload[0]
                self.current_telemetry.posture = (
                    RadarPosture.STANDING if pval == 1 else
                    RadarPosture.SITTING if pval == 2 else
                    RadarPosture.LYING if pval == 3 else
                    RadarPosture.UNKNOWN
                )
                updated = True
            elif cmd == 0x02 and len(payload) >= 2:
                height_cm = (payload[0] << 8) | payload[1]
                self.current_telemetry.target_height_m = height_cm / 100.0
                updated = True

        if updated and self.callback:
            self.callback(self.current_telemetry)

        return self.current_telemetry
