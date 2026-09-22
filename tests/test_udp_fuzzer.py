"""
UDP Stream Fuzzer & Parser Hardening Tests (Milestone 18.4).

Uses Hypothesis property-based testing to verify that UDP packet parsing and
mesh deduplication are impervious to arbitrary, malformed, truncated, or
malicious payload streams.
"""

import struct
import pytest
from hypothesis import given, settings, strategies as st

from hub.packet_parser import (
    MAXIMUM_PACKET_SIZE,
    MINIMUM_V2_SIZE,
    MeshDeduplicator,
    parse_v2_header,
)


@given(st.binary(min_size=0, max_size=70000))
@settings(max_examples=150, deadline=None)
def test_fuzz_arbitrary_binary_stream_never_crashes(payload: bytes):
    """Fuzz parse_v2_header with arbitrary byte arrays; it must never raise an unhandled exception."""
    try:
        result = parse_v2_header(payload)
        assert result is None or isinstance(result, dict)
    except Exception as exc:
        pytest.fail(f"parse_v2_header raised unexpected exception on input {payload[:30]!r}: {exc}")


@given(
    st.binary(min_size=4, max_size=4).filter(lambda b: b != b"CSIF"),
    st.binary(min_size=14, max_size=100),
)
@settings(max_examples=50, deadline=None)
def test_fuzz_corrupted_magic_rejected(bad_magic: bytes, rest: bytes):
    """Any packet with magic != b'CSIF' is immediately rejected as None."""
    packet = bad_magic + rest
    assert parse_v2_header(packet) is None


@given(st.binary(min_size=0, max_size=MINIMUM_V2_SIZE - 1))
@settings(max_examples=30, deadline=None)
def test_fuzz_truncated_packets_rejected(short_data: bytes):
    """Any packet shorter than MINIMUM_V2_SIZE (18 bytes) returns None."""
    assert parse_v2_header(short_data) is None


@given(st.integers(min_value=MAXIMUM_PACKET_SIZE + 1, max_value=MAXIMUM_PACKET_SIZE + 5000))
@settings(max_examples=10, deadline=None)
def test_fuzz_oversized_packets_rejected(size: int):
    """Packets exceeding MAXIMUM_PACKET_SIZE (65535 bytes) return None."""
    oversized = b"CSIF" + b"\x00" * (size - 4)
    assert parse_v2_header(oversized) is None


@given(st.integers(min_value=0, max_value=255))
@settings(max_examples=50, deadline=None)
def test_fuzz_mesh_hop_count_boundaries(hop_count: int):
    """Packets with hop_count > 3 must return None; hop_count in [0, 3] are parsed."""
    # Construct a valid 26-byte extended mesh frame
    magic = b"CSIF"
    node_id = 1
    room_id = 1
    rssi = -50
    pad = 0
    subcarriers = 64
    orig_mac = b"\x11\x22\x33\x44\x55\x66"
    timestamp_ms = 1000
    seq_num = 1

    packet = (
        magic
        + struct.pack("BBbB", node_id, room_id, rssi, pad)
        + struct.pack("B", subcarriers & 0xFF)
        + struct.pack("B", hop_count)
        + orig_mac
        + struct.pack("<H", subcarriers)
        + struct.pack("<I", timestamp_ms)
        + struct.pack("<I", seq_num)
    )

    result = parse_v2_header(packet)
    if hop_count > 3:
        assert result is None, f"Hop count {hop_count} > 3 must be rejected"
    else:
        assert result is not None, f"Hop count {hop_count} <= 3 must be accepted"
        assert result["hop_count"] == hop_count


@given(st.integers(min_value=0, max_value=65535))
@settings(max_examples=50, deadline=None)
def test_fuzz_subcarrier_count_extremes(subcarrier_val: int):
    """Arbitrary uint16 subcarrier count in standard header parsed safely without overflow."""
    packet = struct.pack("<4sBBbBHII", b"CSIF", 1, 1, -50, 0, subcarrier_val, 100, 1)
    result = parse_v2_header(packet)
    assert result is not None
    assert result["subcarrier_count"] == subcarrier_val


@given(st.text(min_size=0, max_size=50), st.integers(min_value=-1000000, max_value=1000000))
@settings(max_examples=60, deadline=None)
def test_fuzz_mesh_deduplicator_with_random_inputs(node_id: str, seq_num: int):
    """MeshDeduplicator handles arbitrary string node IDs and sequence numbers safely."""
    dedup = MeshDeduplicator(capacity=100)
    assert dedup.is_duplicate(node_id, seq_num) is False
    assert dedup.is_duplicate(node_id, seq_num) is True


def test_fuzz_invalid_types_handled_gracefully():
    """Non-byte inputs (None, int, float, list, dict) return None without error."""
    assert parse_v2_header(None) is None  # type: ignore
    assert parse_v2_header(12345) is None  # type: ignore
    assert parse_v2_header([1, 2, 3]) is None  # type: ignore
    assert parse_v2_header({"magic": "CSIF"}) is None  # type: ignore
