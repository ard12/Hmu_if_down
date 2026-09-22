"""Tests for hub.joint_angles (Milestone 15.2)."""

import pytest
from hub.joint_angles import JointAngleEstimator


def test_upright_skeleton_trunk_inclination_approx_zero():
    estimator = JointAngleEstimator()
    torso_top = [0.0, 0.0, 1.40]
    torso_bottom = [0.0, 0.0, 0.80]
    angle = estimator.trunk_inclination(torso_top, torso_bottom)
    assert abs(angle - 0.0) < 1e-3


def test_horizontal_skeleton_trunk_inclination_approx_ninety():
    estimator = JointAngleEstimator()
    torso_top = [0.60, 0.0, 0.20]
    torso_bottom = [0.0, 0.0, 0.20]
    angle = estimator.trunk_inclination(torso_top, torso_bottom)
    assert abs(angle - 90.0) < 1e-3


def test_forty_five_degree_lean_skeleton_inclination():
    estimator = JointAngleEstimator()
    # 45° forward lean in X
    torso_bottom = [0.0, 0.0, 0.80]
    torso_top = [0.60, 0.0, 0.80 + 0.60]
    angle = estimator.trunk_inclination(torso_top, torso_bottom)
    assert abs(angle - 45.0) < 2.0


def test_extended_knee_flexion_approx_zero():
    estimator = JointAngleEstimator()
    hip = [0.0, 0.0, 0.90]
    knee = [0.0, 0.0, 0.50]
    ankle = [0.0, 0.0, 0.10]
    flexion = estimator.knee_flexion(hip, knee, ankle)
    assert abs(flexion - 0.0) < 1.0


def test_seated_knee_flexion_approx_ninety():
    estimator = JointAngleEstimator()
    # Thigh horizontal forward along Y, shank vertical downward along Z
    hip = [0.0, 0.40, 0.50]
    knee = [0.0, 0.0, 0.50]
    ankle = [0.0, 0.0, 0.10]
    flexion = estimator.knee_flexion(hip, knee, ankle)
    assert abs(flexion - 90.0) < 5.0


def test_head_drop_velocity_negative_for_simulated_fall():
    estimator = JointAngleEstimator()
    dt = 0.05
    # Simulated fall: head drops from 1.65m to 0.25m over 1.0s (20 frames)
    z_positions = [1.65 - (1.40 * (i / 20.0)) for i in range(21)]
    head_positions = [[0.0, 0.0, z] for z in z_positions]

    vel = estimator.head_drop_velocity(head_positions, dt)
    assert vel < -0.80  # Drop velocity should indicate rapid downward fall onset


def test_classify_posture_returns_fallen_for_trunk_inclination_greater_than_sixty():
    estimator = JointAngleEstimator()
    skeleton = {
        "valid": True,
        "head": [0.50, 0.0, 0.30],
        "torso_top": [0.50, 0.0, 0.25],
        "torso_bottom": [0.0, 0.0, 0.25],
    }
    # Horizontal trunk: inclination ~ 90° > 60°
    posture = estimator.classify_posture(skeleton)
    assert posture == "FALLEN"


def test_classify_posture_standing_bending_sitting_and_invalid():
    estimator = JointAngleEstimator()

    # Standing
    standing = {
        "valid": True,
        "head": [0.0, 0.0, 1.70],
        "torso_top": [0.0, 0.0, 1.45],
        "torso_bottom": [0.0, 0.0, 0.90],
    }
    assert estimator.classify_posture(standing) == "STANDING"

    # Bending: moderate inclination > 35° with elevated head
    bending = {
        "valid": True,
        "head": [0.35, 0.0, 1.15],
        "torso_top": [0.30, 0.0, 1.10],
        "torso_bottom": [0.0, 0.0, 0.80],
    }
    assert estimator.classify_posture(bending) == "BENDING"

    # Sitting: upright trunk but lower torso/max_z
    sitting = {
        "valid": True,
        "head": [0.0, 0.0, 1.15],
        "torso_top": [0.0, 0.0, 0.95],
        "torso_bottom": [0.0, 0.0, 0.50],
    }
    assert estimator.classify_posture(sitting) == "SITTING"

    # Invalid
    invalid = {"valid": False}
    assert estimator.classify_posture(invalid) == "UNKNOWN"
