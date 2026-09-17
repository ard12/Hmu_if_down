"""UDP Relay Client — Secondary Hub Packet Forwarder (Milestone 6.1).

Listens on a local relay port for room_id-tagged datagrams forwarded from
secondary ESP32 AP clusters and re-injects them into the primary hub's
RoomManager. Useful in multi-building or multi-wing deployments where a
dedicated Pi cannot run a full hub but can forward raw ESP32 UDP packets.

Wire format (relay header, 4 bytes prepended to the original ESP32 packet):
  Byte 0: Magic 0xFD
  Byte 1: room_id  (0x01–0xFE)
  Bytes 2-3: Reserved (0x00 0x00)
  Bytes 4+: Original ESP32 UDP payload

Usage:
    relay = RelayClient(room_manager=manager, relay_port=9200)
    relay.start()   # starts background thread
    relay.stop()
"""

import logging
import socket
import struct
import threading
from typing import Optional

logger = logging.getLogger("relay_client")

_RELAY_MAGIC = 0xFD
_HEADER_SIZE = 4  # magic(1) + room_id(1) + reserved(2)


class RelayClient:
    """Background UDP relay listener that injects forwarded packets into RoomManager.

    Args:
        room_manager: The primary hub's RoomManager instance.
        relay_port: UDP port to listen on (default 9200).
        bind_host: Interface to bind (default '' = all interfaces).
        raw_csi_handler: Optional callable(room_id, raw_bytes) for raw CSI frames.
        raw_radar_handler: Optional callable(room_id, raw_bytes) for raw radar frames.
    """

    def __init__(
        self,
        room_manager,  # RoomManager — avoid circular import by using duck typing
        relay_port: int = 9200,
        bind_host: str = "",
        raw_csi_handler=None,
        raw_radar_handler=None,
    ):
        self.room_manager = room_manager
        self.relay_port = relay_port
        self.bind_host = bind_host
        self._raw_csi_handler = raw_csi_handler
        self._raw_radar_handler = raw_radar_handler

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.packets_received: int = 0
        self.packets_rejected: int = 0

    def start(self) -> None:
        """Bind UDP socket and start background listener thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.bind_host, self.relay_port))
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="relay-listener")
        self._thread.start()
        logger.info("RelayClient listening on UDP :%d", self.relay_port)

    def stop(self) -> None:
        """Stop the listener thread and close the socket."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("RelayClient stopped.")

    def _listen_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            self.packets_received += 1

            if len(data) < _HEADER_SIZE:
                logger.debug("Relay: packet too short (%d bytes) from %s", len(data), addr)
                self.packets_rejected += 1
                continue

            magic, room_id, _r0, _r1 = struct.unpack_from("BBBB", data, 0)
            if magic != _RELAY_MAGIC:
                logger.debug("Relay: bad magic 0x%02X from %s", magic, addr)
                self.packets_rejected += 1
                continue

            payload = data[_HEADER_SIZE:]
            self._dispatch(room_id, payload, addr)

    def _dispatch(self, room_id: int, payload: bytes, addr) -> None:
        """Classify payload as CSI or radar and route to the RoomManager.

        Heuristic: radar JSON datagrams start with '{' (0x7B); everything else
        is treated as a raw CSI frame and forwarded to the optional raw handler.
        """
        if not payload:
            return

        if payload[0] == 0x7B:  # JSON → radar datagram
            if self._raw_radar_handler:
                self._raw_radar_handler(room_id, payload)
            else:
                logger.debug("Relay: radar packet from room %d (no handler)", room_id)
        else:
            if self._raw_csi_handler:
                self._raw_csi_handler(room_id, payload)
            else:
                logger.debug("Relay: CSI packet from room %d (no handler)", room_id)

    def stats(self) -> dict:
        """Return relay statistics."""
        return {
            "port": self.relay_port,
            "running": self._running,
            "packets_received": self.packets_received,
            "packets_rejected": self.packets_rejected,
        }
