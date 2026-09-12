"""[LEGACY] Unit tests for CV-based fall detection kinematics and state machine.

These tests cover the older camera-based fall detection pipeline in src/detector/.
They are NOT part of the current Wi-Fi CSI + mmWave radar architecture in hub/.
Kept for reference — skip in CI unless the src/ pipeline is being actively developed.
"""

import pytest
from src.detector.fall_logic import FallDetector, FallState
from src.detector.pose_estimator import LandmarkPoint, PoseLandmarks

pytestmark = pytest.mark.skipif(
    True, reason="Legacy CV pipeline tests — not part of active RF/radar architecture"
)


def create_mock_pose(
    left_shoulder: tuple,
    right_shoulder: tuple,
    left_hip: tuple,
    right_hip: tuple,
    extra_points: dict = None,
) -> PoseLandmarks:
    """Helper to construct synthetic PoseLandmarks for testing."""
    points = {
        "left_shoulder": LandmarkPoint(x=left_shoulder[0], y=left_shoulder[1], z=0.0, visibility=0.9),
        "right_shoulder": LandmarkPoint(x=right_shoulder[0], y=right_shoulder[1], z=0.0, visibility=0.9),
        "left_hip": LandmarkPoint(x=left_hip[0], y=left_hip[1], z=0.0, visibility=0.9),
        "right_hip": LandmarkPoint(x=right_hip[0], y=right_hip[1], z=0.0, visibility=0.9),
    }
    if extra_points:
        for name, coords in extra_points.items():
            points[name] = LandmarkPoint(x=coords[0], y=coords[1], z=0.0, visibility=0.9)

    return PoseLandmarks(points=points)


def test_torso_angle_upright():
    # Perfectly vertical torso (shoulders at y=0.2, hips at y=0.6)
    pose = create_mock_pose(
        left_shoulder=(0.45, 0.2),
        right_shoulder=(0.55, 0.2),
        left_hip=(0.45, 0.6),
        right_hip=(0.55, 0.6),
    )
    angle = FallDetector.calculate_torso_angle(pose)
    assert angle is not None
    assert pytest.approx(angle, abs=1.0) == 0.0


def test_torso_angle_horizontal():
    # Completely horizontal torso (shoulders at x=0.2, hips at x=0.6, same y=0.5)
    pose = create_mock_pose(
        left_shoulder=(0.2, 0.48),
        right_shoulder=(0.2, 0.52),
        left_hip=(0.6, 0.48),
        right_hip=(0.6, 0.52),
    )
    angle = FallDetector.calculate_torso_angle(pose)
    assert angle is not None
    assert pytest.approx(angle, abs=1.0) == 90.0


def test_aspect_ratio_standing_vs_lying():
    detector = FallDetector()

    # Standing: tall box
    standing_pose = create_mock_pose(
        left_shoulder=(0.45, 0.2),
        right_shoulder=(0.55, 0.2),
        left_hip=(0.45, 0.6),
        right_hip=(0.55, 0.6),
        extra_points={
            "nose": (0.5, 0.1),
            "left_ankle": (0.45, 0.9),
            "right_ankle": (0.55, 0.9),
        },
    )
    ar_standing = detector.calculate_aspect_ratio(standing_pose, frame_width=1000, frame_height=1000)
    assert ar_standing is not None
    assert ar_standing > 1.5

    # Lying: wide box
    lying_pose = create_mock_pose(
        left_shoulder=(0.2, 0.8),
        right_shoulder=(0.2, 0.85),
        left_hip=(0.6, 0.8),
        right_hip=(0.6, 0.85),
        extra_points={
            "nose": (0.1, 0.8),
            "left_ankle": (0.9, 0.8),
            "right_ankle": (0.9, 0.85),
        },
    )
    ar_lying = detector.calculate_aspect_ratio(lying_pose, frame_width=1000, frame_height=1000)
    assert ar_lying is not None
    assert ar_lying < 0.5


def test_state_machine_fall_confirmation_and_recovery():
    detector = FallDetector(confirmation_frames=3, recovery_frames=3)

    standing_pose = create_mock_pose(
        left_shoulder=(0.45, 0.2),
        right_shoulder=(0.55, 0.2),
        left_hip=(0.45, 0.6),
        right_hip=(0.55, 0.6),
    )

    lying_pose = create_mock_pose(
        left_shoulder=(0.2, 0.8),
        right_shoulder=(0.2, 0.85),
        left_hip=(0.6, 0.8),
        right_hip=(0.6, 0.85),
    )

    # Frame 1: Standing
    m1 = detector.evaluate(standing_pose)
    assert m1.state == FallState.NORMAL

    # Frame 2: Lying down (frame 1 of fall) -> suspected
    m2 = detector.evaluate(lying_pose)
    assert m2.state == FallState.FALL_SUSPECTED

    # Frame 3: Lying down (frame 2 of fall) -> still suspected
    m3 = detector.evaluate(lying_pose)
    assert m3.state == FallState.FALL_SUSPECTED

    # Frame 4: Lying down (frame 3 of fall) -> confirmed!
    m4 = detector.evaluate(lying_pose)
    assert m4.state == FallState.FALL_CONFIRMED

    # Recovery frames
    detector.evaluate(standing_pose)
    detector.evaluate(standing_pose)
    m_rec = detector.evaluate(standing_pose)
    assert m_rec.state == FallState.RECOVERED
