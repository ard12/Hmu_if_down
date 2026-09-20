"""
Hub-side packet parser for CSI and mmWave radar UDP datagrams.
Supports V1 (legacy 16B), V2 (18B standard), and V2 Mesh-Relayed extensions.
"""

import math
import struct
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


MINIMUM_V2_SIZE = 18
MAXIMUM_PACKET_SIZE = 65535
MAX_MESH_HOPS = 3


def calculate_mesh_rssi_weight(hop_count: int) -> float:
    """
    Calculate RSSI weighting factor for mesh-relayed packets.
    Weight decays exponentially with hop count: e^(-hop_count).
    0-hop: 1.0
    1-hop: ~0.368
    2-hop: ~0.135
    """
    return math.exp(-max(0, hop_count))


class MeshDeduplicator:
    """
    Deduplicates frames arriving over multiple mesh paths.
    Identifies frames by (original_node_id, seq_num).
    """

    def __init__(self, capacity: int = 2000, ttl_seconds: float = 5.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[Tuple[str, int], float] = OrderedDict()

    def is_duplicate(self, original_node_id: str, seq_num: int) -> bool:
        """
        Check if (original_node_id, seq_num) has already been processed.
        Returns True if duplicate, False if new (and records it).
        """
        key = (str(original_node_id), int(seq_num))
        now = time.time()

        # Purge expired entries if cache is large
        if len(self._cache) > self.capacity:
            cutoff = now - self.ttl_seconds
            keys_to_remove = [k for k, v in self._cache.items() if v < cutoff]
            for k in keys_to_remove:
                del self._cache[k]
            # If still exceeding capacity, pop oldest
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

        if key in self._cache:
            return True

        self._cache[key] = now
        return False

    def clear(self):
        """Clear deduplication cache."""
        self._cache.clear()


def parse_v2_header(data: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse V2 packet header with optional ESP-MESH relay fields.

    Standard V2 Header (18 bytes):
      - 0..3: magic (b'CSIF')
      - 4: node_id (uint8)
      - 5: room_id (uint8)
      - 6: rssi (int8)
      - 7: flags/pad (uint8)
      - 8: subcarrier_count LSB / byte 8
      - 9: hop_count (uint8): 0 for direct node, >0 for mesh relay
      - 10..15: original_node_id (6 bytes MAC) if mesh-relayed
      - 16..19: timestamp_ms (uint32)
      - 20..23: seq_num (uint32)

    Also supports backward-compatible 18-byte packets where:
      - 8..9: subcarrier_count (uint16)
      - 10..13: timestamp_ms (uint32)
      - 14..17: seq_num (uint32)
      (in which case byte 9 == 0 or subcarrier_count is unpacked).

    Returns:
      dict with parsed fields, or None if packet is malformed / TTL exceeded.
    """
    if not isinstance(data, (bytes, bytearray)):
        return None

    data_len = len(data)
    if data_len < MINIMUM_V2_SIZE or data_len > MAXIMUM_PACKET_SIZE:
        return None

    try:
        magic = bytes(data[:4])
        if magic != b"CSIF":
            return None

        node_id = data[4]
        room_id = data[5]
        rssi = struct.unpack("b", bytes([data[6]]))[0]

        # Byte 9 indicates mesh hop count
        hop_count = data[9] if data_len > 9 else 0

        # Reject frames that exceed maximum mesh TTL
        if hop_count > MAX_MESH_HOPS:
            return None

        if hop_count > 0:
            # Mesh-relayed packet
            mesh_relayed = True
            # Extract 6-byte MAC / original node id from bytes 10..15 if available
            if data_len >= 16:
                orig_bytes = bytes(data[10:16])
                original_node_id = ":".join(f"{b:02x}" for b in orig_bytes)
            else:
                original_node_id = str(node_id)

            # Subcarrier count and timestamps in mesh format or standard
            if data_len >= 26:
                # Extended 26-byte mesh header
                subcarrier_count = struct.unpack("<H", data[16:18])[0]
                timestamp_ms = struct.unpack("<I", data[18:22])[0]
                seq_num = struct.unpack("<I", data[22:26])[0]
                payload_offset = 26
            elif data_len >= 24:
                # 24-byte mesh header
                timestamp_ms = struct.unpack("<I", data[16:20])[0]
                seq_num = struct.unpack("<I", data[20:24])[0]
                subcarrier_count = data[8]
                payload_offset = 24
            else:
                # Standard 18-byte packed mesh frame
                timestamp_ms = struct.unpack("<I", data[10:14])[0]
                seq_num = struct.unpack("<I", data[14:18])[0]
                subcarrier_count = data[8]
                payload_offset = 18
        else:
            # Standard non-mesh direct frame (hop_count == 0)
            mesh_relayed = False
            original_node_id = str(node_id)
            subcarrier_count = struct.unpack("<H", data[8:10])[0]
            timestamp_ms = struct.unpack("<I", data[10:14])[0]
            seq_num = struct.unpack("<I", data[14:18])[0]
            payload_offset = 18

        return {
            "magic": magic,
            "node_id": node_id,
            "room_id": room_id,
            "rssi": rssi,
            "hop_count": hop_count,
            "mesh_relayed": mesh_relayed,
            "original_node_id": original_node_id,
            "subcarrier_count": subcarrier_count,
            "timestamp_ms": timestamp_ms,
            "seq_num": seq_num,
            "payload_offset": payload_offset,
        }

    except (struct.error, IndexError, ValueError):
        return None
