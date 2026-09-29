"""Unit tests for UDP RelayClient and dashboard relay stats API (Milestone 7.4)."""

import socket
import struct
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.relay_client import RelayClient, _RELAY_MAGIC
from hub.room_manager import RoomManager
from hub.dashboard.app import app, broadcaster


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_relay_client_start_stop():
    """Verify RelayClient starts background listener and stops cleanly."""
    mgr = RoomManager()
    port = get_free_port()
    relay = RelayClient(room_manager=mgr, relay_port=port, bind_host="127.0.0.1")

    relay.start()
    st = relay.stats()
    assert st["running"] is True
    assert st["port"] == port
    assert st["packets_received"] == 0

    relay.stop()
    st = relay.stats()
    assert st["running"] is False


def wait_until(predicate, timeout=1.0, step=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def test_relay_client_ignores_bad_magic():
    """Verify RelayClient rejects packets that do not start with _RELAY_MAGIC."""
    mgr = RoomManager()
    port = get_free_port()
    relay = RelayClient(room_manager=mgr, relay_port=port, bind_host="127.0.0.1")
    relay.start()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bad magic: 0xEE instead of 0xFD
        bad_packet = struct.pack("BBBB", 0xEE, 1, 0, 0) + b"hello"
        sock.sendto(bad_packet, ("127.0.0.1", port))
        sock.close()

        assert wait_until(lambda: relay.stats()["packets_received"] == 1)
        st = relay.stats()
        assert st["packets_received"] == 1
        assert st["packets_rejected"] == 1
    finally:
        relay.stop()


def test_relay_client_routes_json_to_radar_handler():
    """Verify RelayClient routes packets beginning with '{' to raw_radar_handler."""
    mgr = RoomManager()
    port = get_free_port()
    radar_received = []

    def radar_cb(room_id, payload):
        radar_received.append((room_id, payload))

    relay = RelayClient(
        room_manager=mgr,
        relay_port=port,
        bind_host="127.0.0.1",
        raw_radar_handler=radar_cb,
    )
    relay.start()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Magic 0xFD, room_id 5, reserved 0, 0
        header = struct.pack("BBBB", _RELAY_MAGIC, 5, 0, 0)
        payload = b'{"fall": 1, "height": 0.40}\n'
        sock.sendto(header + payload, ("127.0.0.1", port))
        sock.close()

        assert wait_until(lambda: len(radar_received) == 1)
        assert len(radar_received) == 1
        r_id, data = radar_received[0]
        assert r_id == 5
        assert data == payload
    finally:
        relay.stop()


def test_relay_client_routes_binary_to_csi_handler():
    """Verify RelayClient routes non-JSON packets to raw_csi_handler."""
    mgr = RoomManager()
    port = get_free_port()
    csi_received = []

    def csi_cb(room_id, payload):
        csi_received.append((room_id, payload))

    relay = RelayClient(
        room_manager=mgr,
        relay_port=port,
        bind_host="127.0.0.1",
        raw_csi_handler=csi_cb,
    )
    relay.start()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        header = struct.pack("BBBB", _RELAY_MAGIC, 3, 0, 0)
        payload = b"CSIF\x01\x03\x00\x00"
        sock.sendto(header + payload, ("127.0.0.1", port))
        sock.close()

        assert wait_until(lambda: len(csi_received) == 1)
        assert len(csi_received) == 1
        r_id, data = csi_received[0]
        assert r_id == 3
        assert data == payload
    finally:
        relay.stop()


def test_api_relay_stats_endpoint():
    """Verify GET /api/relay/stats returns 200 with stats or running=False."""
    client = TestClient(app)

    # When no relay client is configured
    broadcaster.relay_client = None
    resp = client.get("/api/relay/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False

    # When relay client is attached
    mgr = RoomManager()
    port = get_free_port()
    relay = RelayClient(room_manager=mgr, relay_port=port, bind_host="127.0.0.1")
    relay.start()
    broadcaster.relay_client = relay

    try:
        resp = client.get("/api/relay/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert data["port"] == port
    finally:
        relay.stop()
        broadcaster.relay_client = None
