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
    cluster_area_m2: float = 0.0
    is_clutter: bool = False


class RadarReceiver:
    """Decodes binary radar packets or JSON UDP datagrams from ESP32 mmWave gateway."""

    HEADER = b"\x53\x59"

    def __init__(
        self,
        callback: Optional[Callable[[RadarTelemetry], None]] = None,
        min_cluster_area_m2: float = 0.15,
        enable_cluster_filter: bool = True,
    ):
        self.callback = callback
        self.min_cluster_area_m2 = min_cluster_area_m2
        self.enable_cluster_filter = enable_cluster_filter
        self.current_telemetry = RadarTelemetry(
            fall_state=RadarFallState.NONE,
            posture=RadarPosture.UNKNOWN,
            target_height_m=1.6,
            dwell_time_sec=0,
            raw_valid=True,
            cluster_area_m2=0.0,
            is_clutter=False,
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

            cluster_area = float(data.get("cluster_area", data.get("area", 0.0)))
            fall_raw = fall_map.get(data.get("fall", 0), RadarFallState.NONE)
            posture_raw = posture_map.get(data.get("posture", 0), RadarPosture.UNKNOWN)
            height_raw = float(data.get("height", 1.6))
            dwell_raw = int(data.get("dwell", 0))

            is_clutter = False
            if self.enable_cluster_filter and 0.0 < cluster_area < self.min_cluster_area_m2:
                is_clutter = True
                fall_raw = RadarFallState.NONE
                posture_raw = RadarPosture.UNKNOWN

            telemetry = RadarTelemetry(
                fall_state=fall_raw,
                posture=posture_raw,
                target_height_m=height_raw,
                dwell_time_sec=dwell_raw,
                raw_valid=not is_clutter,
                cluster_area_m2=cluster_area,
                is_clutter=is_clutter,
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
                # Re-apply clutter suppression if cluster area was already flagged
                if self.enable_cluster_filter and self.current_telemetry.is_clutter:
                    self.current_telemetry.fall_state = RadarFallState.NONE
                updated = True
            elif cmd == 0x02 and len(payload) >= 2:
                self.current_telemetry.dwell_time_sec = (payload[0] << 8) | payload[1]
                updated = True

        # Control Word 0x03: Posture & Height & Cluster Area
        elif ctrl == 0x03:
            if cmd == 0x01 and len(payload) >= 1:
                pval = payload[0]
                self.current_telemetry.posture = (
                    RadarPosture.STANDING if pval == 1 else
                    RadarPosture.SITTING if pval == 2 else
                    RadarPosture.LYING if pval == 3 else
                    RadarPosture.UNKNOWN
                )
                if self.enable_cluster_filter and self.current_telemetry.is_clutter:
                    self.current_telemetry.posture = RadarPosture.UNKNOWN
                updated = True
            elif cmd == 0x02 and len(payload) >= 2:
                height_cm = (payload[0] << 8) | payload[1]
                self.current_telemetry.target_height_m = height_cm / 100.0
                updated = True
            elif cmd == 0x03 and len(payload) >= 2:
                area_cm2 = (payload[0] << 8) | payload[1]
                self.current_telemetry.cluster_area_m2 = area_cm2 / 10000.0
                if self.enable_cluster_filter and 0.0 < self.current_telemetry.cluster_area_m2 < self.min_cluster_area_m2:
                    self.current_telemetry.is_clutter = True
                    self.current_telemetry.fall_state = RadarFallState.NONE
                    self.current_telemetry.raw_valid = False
                else:
                    self.current_telemetry.is_clutter = False
                    self.current_telemetry.raw_valid = True
                updated = True

        if updated and self.callback:
            self.callback(self.current_telemetry)

        return self.current_telemetry
