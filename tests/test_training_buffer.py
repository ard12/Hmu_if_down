"""Tests for incremental training buffer and ground-truth event labeling."""

import pickle
import threading
import time
import numpy as np
import pytest

from hub.training_buffer import InsufficientDataError, TrainingBuffer


def test_buffer_ring_eviction():
    """Buffer respects MAX_BUFFER_SIZE (ring eviction)."""
    buf = TrainingBuffer(max_size=5)
    for i in range(10):
        buf.append(features=np.array([i, i * 2]), label=i % 2, timestamp=100.0 + i)

    stats = buf.stats()
    assert stats["total"] == 5
    # The oldest items 0..4 should have been evicted; items 5..9 remain
    assert stats["oldest_ts"] == 105.0
    assert stats["newest_ts"] == 109.0


def test_get_batch_insufficient_data():
    """get_batch raises InsufficientDataError when positives < threshold."""
    buf = TrainingBuffer(max_size=50)
    for i in range(15):
        buf.append(features=np.array([float(i)]), label=1 if i < 5 else 0)

    # We only have 5 positives, request 20
    with pytest.raises(InsufficientDataError) as exc_info:
        buf.get_batch(min_positives=20)
    assert "Insufficient positive labels" in str(exc_info.value)


def test_get_batch_success():
    """get_batch returns (X, y) when enough positives are present."""
    buf = TrainingBuffer(max_size=50)
    for i in range(30):
        buf.append(features=np.array([float(i), float(i * 2)]), label=1 if i < 25 else 0)

    X, y = buf.get_batch(min_positives=20)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (30, 2)
    assert y.shape == (30,)
    assert np.sum(y == 1) == 25


def test_label_event_flips_confirmed():
    """label_event correctly flips confirmed flag and label."""
    buf = TrainingBuffer(max_size=20)
    buf.append(features=np.array([1.0, 2.0]), label=0, event_id="evt_123")
    buf.append(features=np.array([3.0, 4.0]), label=0, event_id="evt_456")

    # Initially label=0
    assert buf.stats()["positives"] == 0

    # Staff confirms evt_123 was a fall
    buf.label_event("evt_123", confirmed=True)
    assert buf.stats()["positives"] == 1

    # Staff explicitly marks evt_123 as false alarm (unconfirmed)
    buf.label_event("evt_123", confirmed=False)
    assert buf.stats()["positives"] == 0


def test_stats_counts_accurate():
    """Stats counts are accurate after mixed appends."""
    buf = TrainingBuffer(max_size=100)
    assert buf.stats()["total"] == 0

    for i in range(10):
        buf.append(features=np.zeros(3), label=1 if i < 4 else 0, timestamp=10.0 + i)

    stats = buf.stats()
    assert stats["total"] == 10
    assert stats["positives"] == 4
    assert stats["negatives"] == 6
    assert stats["oldest_ts"] == 10.0
    assert stats["newest_ts"] == 19.0


def test_thread_safety_concurrent_append_and_label():
    """Thread-safety: concurrent append + label_event from multiple threads."""
    buf = TrainingBuffer(max_size=500)
    stop_flag = False

    def appender():
        for i in range(100):
            buf.append(
                features=np.random.randn(4),
                label=0,
                event_id=f"evt_{i}",
                timestamp=time.time(),
            )
            time.sleep(0.001)

    def labeller():
        for i in range(100):
            buf.label_event(f"evt_{i}", confirmed=(i % 2 == 0))
            time.sleep(0.001)

    t1 = threading.Thread(target=appender)
    t2 = threading.Thread(target=labeller)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    stats = buf.stats()
    assert stats["total"] == 100


def test_serialization_round_trip():
    """Serialization round-trip (pickle/unpickle) preserves data and functionality."""
    buf = TrainingBuffer(max_size=50)
    buf.append(features=np.array([1.5, 2.5]), label=1, timestamp=123.456, event_id="e1")
    buf.append(features=np.array([3.5, 4.5]), label=0, timestamp=123.789, event_id="e2")

    pickled = pickle.dumps(buf)
    unpickled: TrainingBuffer = pickle.loads(pickled)

    assert unpickled.stats()["total"] == 2
    assert unpickled.stats()["positives"] == 1
    # Test that unpickled object can still append and label
    unpickled.append(features=np.array([5.5, 6.5]), label=1)
    assert unpickled.stats()["total"] == 3
    unpickled.label_event("e2", confirmed=True)
    assert unpickled.stats()["positives"] == 3


def test_pre_labeled_event_applied_on_append():
    """Event labeled before arrival automatically adopts confirmed label upon append."""
    buf = TrainingBuffer(max_size=50)
    buf.label_event("future_evt_999", confirmed=True)
    buf.append(features=np.array([1.0, 2.0]), label=0, event_id="future_evt_999")

    assert buf.stats()["positives"] == 1

