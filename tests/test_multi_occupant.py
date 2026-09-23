"""Tests for Multi-Occupant Spatial Disambiguation Engine.

@req SRS-MO-001
"""
import numpy as np
import pytest
from hub.multi_occupant import (
    DetectedBlob,
    MultiOccupantTracker,
    OccupantTrack,
    SimpleKalmanTracker,
)


def test_single_detection_creates_track():
    tracker = MultiOccupantTracker()
    blob = DetectedBlob(centroid=np.array([2.0, 3.0, 1.5]), peak_velocity_mps=0.2)
    tracks = tracker.update([blob], timestamp=10.0)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert np.allclose(tracks[0].position, [2.0, 3.0, 1.5], atol=0.1)
    assert not tracks[0].is_fallen
    assert tracker.frame_count == 1


def test_two_occupants_tracked_separately():
    tracker = MultiOccupantTracker()
    blob1 = DetectedBlob(centroid=np.array([1.0, 1.0, 1.7]), peak_velocity_mps=0.1)
    blob2 = DetectedBlob(centroid=np.array([4.0, 4.0, 1.6]), peak_velocity_mps=0.1)
    tracks = tracker.update([blob1, blob2], timestamp=1.0)

    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2
    assert tracker.active_count == 2


def test_hungarian_assignment_correct_association():
    tracker = MultiOccupantTracker()
    blob1 = DetectedBlob(centroid=np.array([1.0, 1.0, 1.7]), peak_velocity_mps=0.1)
    blob2 = DetectedBlob(centroid=np.array([4.0, 4.0, 1.6]), peak_velocity_mps=0.1)
    tracker.update([blob1, blob2], timestamp=1.0)

    # Frame 2: minor movement
    next_blob1 = DetectedBlob(centroid=np.array([1.05, 1.02, 1.7]), peak_velocity_mps=0.1)
    next_blob2 = DetectedBlob(centroid=np.array([3.95, 4.01, 1.6]), peak_velocity_mps=0.1)
    tracks2 = tracker.update([next_blob2, next_blob1], timestamp=1.1)

    assert len(tracks2) == 2
    # Verify track 1 is associated with next_blob1 (near [1,1,1.7])
    t1 = next(t for t in tracks2 if t.track_id == 1)
    assert np.allclose(t1.position[:2], [1.05, 1.02], atol=0.2)

    # Verify track 2 is associated with next_blob2 (near [4,4,1.6])
    t2 = next(t for t in tracks2 if t.track_id == 2)
    assert np.allclose(t2.position[:2], [3.95, 4.01], atol=0.2)


def test_stale_track_pruned():
    tracker = MultiOccupantTracker()
    blob = DetectedBlob(centroid=np.array([2.0, 2.0, 1.5]), peak_velocity_mps=0.1)
    tracker.update([blob], timestamp=1.0)
    assert tracker.active_count == 1

    # Empty frames past timeout
    tracks = tracker.update([], timestamp=7.0)
    assert tracker.active_count == 0
    assert len(tracks) == 0


def test_fall_detection_on_z_descent():
    tracker = MultiOccupantTracker()
    blob = DetectedBlob(centroid=np.array([2.0, 2.0, 1.7]), peak_velocity_mps=0.1)
    tracker.update([blob], timestamp=1.0)

    # Sudden descent to ground level with high downward velocity
    fall_blob = DetectedBlob(centroid=np.array([2.1, 2.0, 0.3]), peak_velocity_mps=2.5)
    tracks = tracker.update([fall_blob], timestamp=1.1)

    assert len(tracks) == 1
    assert tracks[0].is_fallen is True
    assert len(tracker.get_fallen_occupants()) == 1


def test_recovery_resets_fall_flag():
    tracker = MultiOccupantTracker()
    blob = DetectedBlob(centroid=np.array([2.0, 2.0, 0.3]), peak_velocity_mps=2.5)
    tracker.update([blob], timestamp=1.0)
    assert tracker.get_fallen_occupants()[0].is_fallen is True

    # Stand back up
    stand_blob = DetectedBlob(centroid=np.array([2.0, 2.0, 1.6]), peak_velocity_mps=0.5)
    tracks = tracker.update([stand_blob], timestamp=2.0)
    assert tracks[0].is_fallen is False
    assert len(tracker.get_fallen_occupants()) == 0


def test_max_occupants_cap():
    tracker = MultiOccupantTracker(max_occupants=3)
    blobs = [
        DetectedBlob(centroid=np.array([float(i), float(i), 1.7]))
        for i in range(10)
    ]
    tracks = tracker.update(blobs, timestamp=1.0)
    assert len(tracks) == 3
    assert tracker.active_count == 3


def test_to_dict_schema():
    track = OccupantTrack(
        track_id=42,
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.1, 0.2, 0.3]),
        last_update_time=123.456,
        is_fallen=True,
        confidence=0.95,
        age_frames=12,
    )
    d = track.to_dict()
    assert d["track_id"] == 42
    assert d["position"] == [1.0, 2.0, 3.0]
    assert d["velocity"] == [0.1, 0.2, 0.3]
    assert d["is_fallen"] is True
    assert d["confidence"] == 0.95
    assert d["age_frames"] == 12


def test_kalman_tracker_predict_and_update():
    kf = SimpleKalmanTracker(initial_position=np.array([0.0, 0.0, 0.0]), dt=0.1)
    pred = kf.predict()
    assert np.allclose(pred, [0.0, 0.0, 0.0])

    updated = kf.update(np.array([1.0, 0.0, 0.0]))
    assert updated[0] > 0.0
