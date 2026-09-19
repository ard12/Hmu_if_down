"""Unit tests for room-id routing and CSI packet header parsing (Milestone 7.3)."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.csi_pipeline.preprocessor import CSIPreprocessor, CSIPacket
from hub.room_manager import RoomManager
from hub.csi_pipeline.pca_features import CSIDynamicFeatures


def test_csi_packet_with_room_id_byte():
    """Verify CSIPreprocessor correctly parses 18-byte V2 header containing room_id."""
    preprocessor = CSIPreprocessor()

    # 18-byte header: magic(4s), node_id(B), room_id(B), rssi(b), _pad(B), subcarrier_count(H), ts_ms(I), seq(I)
    node_id = 2
    room_id = 7
    rssi = -45
    pad = 0
    subcarrier_count = 4
    ts_ms = 123456
    seq = 42

    header = struct.pack(
        "<4sBBbBHII",
        b"CSIF",
        node_id,
        room_id,
        rssi,
        pad,
        subcarrier_count,
        ts_ms,
        seq,
    )
    # 4 subcarriers = 8 bytes I/Q
    payload = struct.pack("8b", 10, 20, -10, -20, 5, 15, -5, -15)
    packet_bytes = header + payload

    pkt = preprocessor.parse_packet(packet_bytes)
    assert pkt is not None
    assert pkt.node_id == node_id
    assert pkt.room_id == room_id
    assert pkt.rssi == rssi
    assert pkt.subcarrier_count == subcarrier_count
    assert pkt.timestamp_ms == ts_ms
    assert pkt.seq_num == seq
    assert len(pkt.amplitudes) == 4
    assert len(pkt.phases) == 4


def test_csi_packet_legacy_16byte_header_defaults_room_id_zero():
    """Verify CSIPreprocessor accepts legacy 16-byte header and defaults room_id=0."""
    preprocessor = CSIPreprocessor()

    # 16-byte header: magic(4s), node_id(B), rssi(b), subcarrier_count(H), ts_ms(I), seq(I)
    node_id = 1
    rssi = -55
    subcarrier_count = 2
    ts_ms = 99999
    seq = 10

    header = struct.pack(
        "<4sBbHII",
        b"CSIF",
        node_id,
        rssi,
        subcarrier_count,
        ts_ms,
        seq,
    )
    # 2 subcarriers = 4 bytes I/Q
    payload = struct.pack("4b", 12, 16, -12, -16)
    packet_bytes = header + payload

    pkt = preprocessor.parse_packet(packet_bytes)
    assert pkt is not None
    assert pkt.node_id == node_id
    assert pkt.room_id == 0
    assert pkt.rssi == rssi
    assert pkt.subcarrier_count == subcarrier_count
    assert len(pkt.amplitudes) == 2


def test_server_routes_room1_and_room2_to_separate_contexts():
    """Verify RoomManager isolates packets from different room_ids."""
    mgr = RoomManager()

    class MockFeatures:
        def __init__(self, var=0.05, vel=0.1):
            self.moving_variance = var
            self.dominant_velocity_mps = vel
            self.energy_surge_ratio = 1.0
            self.node_id = 1

    feat_room1 = MockFeatures(var=0.02, vel=0.2)
    feat_room2 = MockFeatures(var=0.15, vel=1.9)

    st1 = mgr.on_csi_packet(room_id=1, features=feat_room1)
    st2 = mgr.on_csi_packet(room_id=2, features=feat_room2)

    ctx1 = mgr.get_room(1)
    ctx2 = mgr.get_room(2)

    assert ctx1 is not None
    assert ctx2 is not None
    assert ctx1 is not ctx2
    assert ctx1.room_id == 1
    assert ctx2.room_id == 2
    assert ctx1.last_csi_time > 0
    assert ctx2.last_csi_time > 0
