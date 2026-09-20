"""Tests for ESP-MESH multi-hop forwarding and packet parsing (Milestone 13.1)."""

import math
import struct
import pytest
from hub.packet_parser import (
    MeshDeduplicator,
    calculate_mesh_rssi_weight,
    parse_v2_header,
)


def _build_v2_packet(
    node_id: int = 1,
    room_id: int = 1,
    rssi: int = -55,
    hop_count: int = 0,
    orig_mac: bytes = b"\xaa\xbb\xcc\xdd\xee\xff",
    timestamp_ms: int = 123456,
    seq_num: int = 42,
    subcarriers: int = 64,
) -> bytes:
    """Helper to build standard or mesh V2 CSI packets."""
    magic = b"CSIF"
    pad = 0

    if hop_count > 0:
        # Extended 26-byte mesh header
        header = (
            magic
            + struct.pack("BBbB", node_id, room_id, rssi, pad)
            + struct.pack("B", subcarriers & 0xFF)  # byte 8
            + struct.pack("B", hop_count)            # byte 9
            + orig_mac                               # bytes 10..15
            + struct.pack("<H", subcarriers)         # bytes 16..17
            + struct.pack("<I", timestamp_ms)        # bytes 18..21
            + struct.pack("<I", seq_num)             # bytes 22..25
        )
    else:
        # Standard 18-byte V2 header
        header = struct.pack(
            "<4sBBbBHII",
            magic,
            node_id,
            room_id,
            rssi,
            pad,
            subcarriers,
            timestamp_ms,
            seq_num,
        )

    # Dummy payload of 2 bytes per subcarrier
    payload = b"\x00" * (subcarriers * 2)
    return header + payload


def test_parse_v2_header_extracts_mesh_fields():
    """parse_v2_header correctly extracts original_node_id and hop_count for mesh packets."""
    mac = b"\x12\x34\x56\x78\x9a\xbc"
    packet = _build_v2_packet(
        node_id=2,
        room_id=3,
        rssi=-60,
        hop_count=2,
        orig_mac=mac,
        timestamp_ms=9999,
        seq_num=101,
    )

    parsed = parse_v2_header(packet)
    assert parsed is not None
    assert parsed["mesh_relayed"] is True
    assert parsed["hop_count"] == 2
    assert parsed["original_node_id"] == "12:34:56:78:9a:bc"
    assert parsed["timestamp_ms"] == 9999
    assert parsed["seq_num"] == 101


def test_calculate_mesh_rssi_weights():
    """RSSI weight for 0-hop = 1.0, 1-hop ≈ 0.368, 2-hop ≈ 0.135."""
    w0 = calculate_mesh_rssi_weight(0)
    w1 = calculate_mesh_rssi_weight(1)
    w2 = calculate_mesh_rssi_weight(2)

    assert w0 == 1.0
    assert pytest.approx(w1, rel=1e-2) == 0.368
    assert pytest.approx(w2, rel=1e-2) == 0.135


def test_mesh_ttl_exceeded_rejected():
    """Frames with hop_count > 3 are rejected (TTL exceeded)."""
    packet = _build_v2_packet(hop_count=4)
    parsed = parse_v2_header(packet)
    assert parsed is None


def test_non_mesh_packet_has_mesh_relayed_false():
    """Non-mesh packets (hop_count = 0) have mesh_relayed=False."""
    packet = _build_v2_packet(node_id=1, hop_count=0)
    parsed = parse_v2_header(packet)

    assert parsed is not None
    assert parsed["mesh_relayed"] is False
    assert parsed["hop_count"] == 0
    assert parsed["original_node_id"] == "1"


def test_mesh_deduplicator_filters_duplicate_frames():
    """Aggregated frame from two mesh paths deduped by (original_node_id, seq_no)."""
    dedup = MeshDeduplicator()

    # First frame from Path A
    is_dup1 = dedup.is_duplicate("node_A", 100)
    assert is_dup1 is False

    # Second frame from Path B (same original node and seq_no)
    is_dup2 = dedup.is_duplicate("node_A", 100)
    assert is_dup2 is True

    # Next sequence number from same node
    is_dup3 = dedup.is_duplicate("node_A", 101)
    assert is_dup3 is False


def test_original_sender_timestamp_preserved():
    """Timestamp from original sender used, not relay timestamp."""
    orig_ts = 55443322
    packet = _build_v2_packet(hop_count=1, timestamp_ms=orig_ts)
    parsed = parse_v2_header(packet)

    assert parsed is not None
    assert parsed["timestamp_ms"] == orig_ts


def test_malformed_packets_safely_rejected():
    """Short packets, invalid magic, or oversized frames return None."""
    assert parse_v2_header(b"") is None
    assert parse_v2_header(b"SHORT") is None
    assert parse_v2_header(b"WIFI" + b"\x00" * 20) is None  # Wrong magic
    assert parse_v2_header(b"CSIF" + b"\x00" * 70000) is None  # Oversized
