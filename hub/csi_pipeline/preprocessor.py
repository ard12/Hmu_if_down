"""CSI raw packet unpacking, amplitude/phase extraction, and bandpass filtering."""

from dataclasses import dataclass
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class CSIPacket:
    node_id: int
    rssi: int
    subcarrier_count: int
    timestamp_ms: int
    seq_num: int
    amplitudes: np.ndarray  # Shape: (subcarrier_count,)
    phases: np.ndarray      # Shape: (subcarrier_count,)
    room_id: int = 0


class CSIPreprocessor:
    """Parses raw UDP byte streams, sanitizes phases, and applies bandpass filters."""

    # Packet Header Format V2 (18 bytes): 4s (magic), B (node_id), B (room_id), b (rssi), B (_pad), H (subcarrier_count), I (timestamp_ms), I (seq_num)
    HEADER_FORMAT_V2 = "<4sBBbBHII"
    HEADER_SIZE_V2 = struct.calcsize(HEADER_FORMAT_V2)

    # Legacy Packet Header Format (16 bytes): 4s (magic), B (node_id), b (rssi), H (subcarrier_count), I (timestamp_ms), I (seq_num)
    HEADER_FORMAT_LEGACY = "<4sBbHII"
    HEADER_SIZE_LEGACY = struct.calcsize(HEADER_FORMAT_LEGACY)

    HEADER_FORMAT = HEADER_FORMAT_LEGACY
    HEADER_SIZE = HEADER_SIZE_LEGACY

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 35.0,
        order: int = 4,
    ):
        self.fs = sampling_rate_hz
        self.lowcut = lowcut_hz
        self.highcut = highcut_hz
        self.order = order

        # Design digital Butterworth bandpass filter
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = min(self.highcut / nyquist, 0.99)
        self.sos = signal.butter(self.order, [low, high], btype="band", output="sos")

    def parse_packet(self, data: bytes) -> Optional[CSIPacket]:
        """Decode binary UDP CSI packet from an ESP32 tracker node.
        
        Supports both 18-byte header (with room_id) and 16-byte legacy header.
        """
        if len(data) < self.HEADER_SIZE_LEGACY:
            return None

        magic = data[:4]
        if magic != b"CSIF":
            return None

        # Determine if packet has 18-byte V2 header or 16-byte legacy header
        sub_legacy = struct.unpack_from("<H", data, 6)[0] if len(data) >= self.HEADER_SIZE_LEGACY else 0
        expected_legacy = self.HEADER_SIZE_LEGACY + sub_legacy * 2

        sub_v2 = struct.unpack_from("<H", data, 8)[0] if len(data) >= self.HEADER_SIZE_V2 else 0
        expected_v2 = self.HEADER_SIZE_V2 + sub_v2 * 2

        if len(data) == expected_v2 and sub_v2 > 0:
            is_v2 = True
        elif len(data) == expected_legacy and sub_legacy > 0:
            is_v2 = False
        elif (
            len(data) >= self.HEADER_SIZE_V2
            and struct.unpack_from("<B", data, 7)[0] == 0
            and struct.unpack_from("<b", data, 6)[0] < 0
            and sub_v2 > 0
            and len(data) >= expected_v2
        ):
            is_v2 = True
        else:
            is_v2 = False

        if is_v2:
            magic, node_id, room_id, rssi, _pad, subcarrier_count, ts_ms, seq = struct.unpack_from(
                self.HEADER_FORMAT_V2, data, 0
            )
            header_size = self.HEADER_SIZE_V2
        else:
            magic, node_id, rssi, subcarrier_count, ts_ms, seq = struct.unpack_from(
                self.HEADER_FORMAT_LEGACY, data, 0
            )
            room_id = 0
            header_size = self.HEADER_SIZE_LEGACY

        expected_payload_len = subcarrier_count * 2
        if subcarrier_count == 0 or len(data) < header_size + expected_payload_len:
            return None

        # Unpack raw I/Q signed 8-bit bytes
        iq_raw = np.frombuffer(
            data, dtype=np.int8, count=expected_payload_len, offset=header_size
        )
        i_vals = iq_raw[0::2].astype(np.float32)
        q_vals = iq_raw[1::2].astype(np.float32)

        # Amplitude and raw phase
        amplitudes = np.hypot(i_vals, q_vals)
        phases = np.arctan2(q_vals, i_vals)

        return CSIPacket(
            node_id=node_id,
            rssi=rssi,
            subcarrier_count=subcarrier_count,
            timestamp_ms=ts_ms,
            seq_num=seq,
            amplitudes=amplitudes,
            phases=phases,
            room_id=room_id,
        )

    def sanitize_phase(self, phases: np.ndarray) -> np.ndarray:
        """Linear phase unwrapping and removal of carrier/sampling frequency offsets."""
        unwrapped = np.unwrap(phases)
        k = np.arange(len(unwrapped))
        # Linear regression to remove slope and offset
        slope = (unwrapped[-1] - unwrapped[0]) / max(1, len(unwrapped) - 1)
        offset = np.mean(unwrapped)
        sanitized = unwrapped - (slope * k) - offset
        return sanitized

    def filter_stream(self, amplitude_window: np.ndarray) -> np.ndarray:
        """Apply zero-phase bandpass filter over temporal window of subcarrier amplitudes.
        
        Args:
            amplitude_window: Matrix of shape (T, num_subcarriers)
        Returns:
            Filtered matrix of shape (T, num_subcarriers)
        """
        # A 4th-order bandpass produces 4 SOS sections; sosfiltfilt default
        # padlen = 3 * max_section_order ≈ 27.  Guard against short windows.
        min_samples = 3 * (2 * self.order) + 1  # = 25 for order=4, add margin
        if amplitude_window.shape[0] <= max(min_samples, 30):
            return amplitude_window

        filtered = signal.sosfiltfilt(self.sos, amplitude_window, axis=0,
                                      padlen=min(amplitude_window.shape[0] - 1, 3 * (2 * self.order)))
        return filtered
