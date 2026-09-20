"""Tests for Room Boundary Handoff State Machine (Milestone 13.2)."""

import threading
import time
from unittest.mock import MagicMock
import pytest
from hub.room_handoff import HandoffState, RoomHandoffManager


def test_hysteresis_single_frame_drop_does_not_trigger_detecting():
    """Single-frame drop does NOT trigger DETECTING state (hysteresis requires 5 frames)."""
    mgr = RoomHandoffManager()

    # Establish baseline in room_1 at -50 dBm
    for _ in range(5):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-50.0, timestamp=time.time())

    # Single frame drop by 15 dB (-65 dBm)
    state = mgr.update(room_id="room_1", node_id="node_1", rssi=-65.0, timestamp=time.time())
    assert state == HandoffState.IDLE


def test_state_transitions_idle_to_detecting_to_dual():
    """State transitions: IDLE -> DETECTING on RSSI drop for 5 frames -> DUAL."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_1": ["room_2"]})

    # Establish baseline in room_1
    base_time = 1000.0
    for i in range(10):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-45.0, timestamp=base_time + i * 0.1)

    # 5 consecutive dropped frames
    for i in range(5):
        state = mgr.update(room_id="room_1", node_id="node_1", rssi=-65.0, timestamp=base_time + 1.0 + i * 0.1)

    assert state == HandoffState.DETECTING

    # Transition to DUAL on candidate room frame or continued drop
    state = mgr.update(room_id="room_2", node_id="node_2", rssi=-55.0, timestamp=base_time + 2.0)
    assert state == HandoffState.DUAL


def test_dual_to_confirmed_when_new_room_rssi_recovers():
    """DUAL -> CONFIRMED when new room RSSI recovers."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_1": ["room_2"]})
    cb = MagicMock()
    mgr.on_handoff_completed(cb)

    base_time = 2000.0
    # Establish baseline
    for i in range(10):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-40.0, timestamp=base_time + i * 0.1)

    # Trigger DETECTING
    for i in range(5):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-60.0, timestamp=base_time + 1.0 + i * 0.1)

    # Trigger DUAL
    mgr.update(room_id="room_2", node_id="node_2", rssi=-55.0, timestamp=base_time + 1.6)

    # Strong signal in room_2 confirms handoff
    state = mgr.update(room_id="room_2", node_id="node_2", rssi=-42.0, timestamp=base_time + 2.0)
    assert state == HandoffState.CONFIRMED

    cb.assert_called_once()
    args = cb.call_args[0]
    assert args[0] == "room_1"  # from_room
    assert args[1] == "room_2"  # to_room


def test_dual_to_abort_when_subject_returns_to_source():
    """DUAL -> ABORT when subject returns to source room."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_1": ["room_2"]})

    base_time = 3000.0
    # Baseline
    for i in range(10):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-45.0, timestamp=base_time + i * 0.1)

    # Trigger DETECTING
    for i in range(5):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-65.0, timestamp=base_time + 1.0 + i * 0.1)

    # Trigger DUAL
    mgr.update(room_id="room_2", node_id="node_2", rssi=-60.0, timestamp=base_time + 1.6)

    # Signal in source room recovers
    state = mgr.update(room_id="room_1", node_id="node_1", rssi=-46.0, timestamp=base_time + 2.0)
    assert state == HandoffState.ABORT


def test_get_active_rooms_returns_two_during_dual():
    """get_active_rooms returns 2 rooms during DUAL, 1 during IDLE/CONFIRMED."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_1": ["room_2"]})

    base_time = 4000.0
    for i in range(10):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-45.0, timestamp=base_time + i * 0.1)

    # In IDLE: exactly 1 room
    assert mgr.get_active_rooms() == ["room_1"]

    # Trigger DETECTING and DUAL
    for i in range(5):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-65.0, timestamp=base_time + 1.0 + i * 0.1)
    mgr.update(room_id="room_2", node_id="node_2", rssi=-55.0, timestamp=base_time + 1.6)

    # In DUAL: 2 rooms
    rooms = mgr.get_active_rooms()
    assert len(rooms) == 2
    assert "room_1" in rooms and "room_2" in rooms


def test_dual_timeout_confirms_target_room():
    """When DUAL_MONITOR_DURATION_S expires, state transitions to CONFIRMED."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_1": ["room_2"]})

    base_time = 5000.0
    for i in range(10):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-45.0, timestamp=base_time + i * 0.1)

    for i in range(5):
        mgr.update(room_id="room_1", node_id="node_1", rssi=-65.0, timestamp=base_time + 1.0 + i * 0.1)
    mgr.update(room_id="room_2", node_id="node_2", rssi=-58.0, timestamp=base_time + 1.6)

    # Advance timestamp beyond 8 seconds
    state = mgr.update(room_id="room_2", node_id="node_2", rssi=-58.0, timestamp=base_time + 11.0)
    assert state == HandoffState.CONFIRMED
    assert mgr.get_active_rooms() == ["room_2"]


def test_concurrent_updates_thread_safe():
    """Concurrent updates from two rooms don't race-condition state machine."""
    mgr = RoomHandoffManager(adjacent_rooms={"room_1": ["room_2"]})

    def worker(room_id: str, rssi: float):
        for i in range(50):
            mgr.update(room_id=room_id, node_id="node_1", rssi=rssi, timestamp=time.time() + i * 0.01)

    t1 = threading.Thread(target=worker, args=("room_1", -50.0))
    t2 = threading.Thread(target=worker, args=("room_2", -52.0))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Must have a valid state and not crashed
    rooms = mgr.get_active_rooms()
    assert len(rooms) in (1, 2)
