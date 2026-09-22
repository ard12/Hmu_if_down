"""Tests for hub.skeleton_fitter (Milestone 15.1)."""

import numpy as np
import pytest
from hub.skeleton_fitter import SkeletonFitter


def generate_standing_point_cloud(height_m: float = 1.70, n_points: int = 50) -> np.ndarray:
    """Generate synthetic radar point cloud of a standing person."""
    rng = np.random.RandomState(123)
    # Head points: z in [0.90, 0.98] * height_m
    head_z = rng.uniform(0.90 * height_m, 0.98 * height_m, size=(8, 1))
    head_xy = rng.normal(0.0, 0.05, size=(8, 2))
    head_pts = np.hstack([head_xy, head_z])

    # Torso points: z in [0.50, 0.88] * height_m
    torso_z = rng.uniform(0.50 * height_m, 0.88 * height_m, size=(24, 1))
    torso_xy = rng.normal(0.0, 0.12, size=(24, 2))
    torso_pts = np.hstack([torso_xy, torso_z])

    # Legs points: z in [0.05, 0.48] * height_m
    legs_z = rng.uniform(0.05 * height_m, 0.48 * height_m, size=(18, 1))
    legs_xy = rng.normal(0.0, 0.08, size=(18, 2))
    legs_pts = np.hstack([legs_xy, legs_z])

    return np.vstack([head_pts, torso_pts, legs_pts])


def generate_fallen_point_cloud(length_m: float = 1.70, n_points: int = 50) -> np.ndarray:
    """Generate synthetic radar point cloud of a fallen person lying horizontally on floor."""
    rng = np.random.RandomState(456)
    # X axis is body length [0.0, length_m], Z is low elevation [0.05, 0.30]
    x = rng.uniform(0.0, length_m, size=(n_points, 1))
    y = rng.normal(0.0, 0.15, size=(n_points, 1))
    z = rng.uniform(0.05, 0.30, size=(n_points, 1))
    return np.hstack([x, y, z])


def test_standing_person_point_cloud_valid_fit_head_near_top():
    fitter = SkeletonFitter(subject_height_m=1.70)
    pc = generate_standing_point_cloud(height_m=1.70, n_points=50)
    result = fitter.fit(pc)

    assert result["valid"] is True
    assert result["fit_quality"] >= 0.4
    # Head z should be near the top (> 1.4m for 1.7m standing)
    assert result["head"][2] >= 1.40
    # Torso top should be higher than torso bottom
    assert result["torso_top"][2] > result["torso_bottom"][2]


def test_fallen_person_point_cloud_head_at_low_z():
    fitter = SkeletonFitter(subject_height_m=1.70)
    pc = generate_fallen_point_cloud(length_m=1.70, n_points=50)
    result = fitter.fit(pc)

    assert result["valid"] is True
    # For horizontal fallen subject, head z must be at floor level (< 0.40m)
    assert result["head"][2] <= 0.40
    assert result["torso_top"][2] <= 0.40
    assert result["torso_bottom"][2] <= 0.40


def test_fit_quality_high_for_clean_low_for_noise():
    fitter = SkeletonFitter(subject_height_m=1.70)
    clean_pc = generate_standing_point_cloud(height_m=1.70, n_points=50)
    clean_result = fitter.fit(clean_pc)

    # Random noise scattered widely across a 10m x 10m x 3m space
    rng = np.random.RandomState(999)
    noise_pc = rng.uniform(-5.0, 5.0, size=(60, 3))
    noise_result = fitter.fit(noise_pc)

    assert clean_result["fit_quality"] > noise_result["fit_quality"]
    assert noise_result["valid"] is False or noise_result["fit_quality"] < 0.4


def test_valid_false_when_point_count_less_than_min_points():
    fitter = SkeletonFitter(subject_height_m=1.70)
    # Only 5 points (< MIN_POINTS = 8)
    small_pc = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2], [0.0, 0.0, 1.4], [0.0, 0.0, 1.6], [0.0, 0.0, 1.7]])
    result = fitter.fit(small_pc)

    assert result["valid"] is False
    assert result["fit_quality"] == 0.0


def test_track_returns_same_length_list_as_input_frames():
    fitter = SkeletonFitter(subject_height_m=1.70)
    frames = [generate_standing_point_cloud(1.70) for _ in range(4)]
    tracked = fitter.track(frames)

    assert isinstance(tracked, list)
    assert len(tracked) == 4
    for item in tracked:
        assert "head" in item
        assert "torso_top" in item
        assert "valid" in item


def test_ransac_stable_across_multiple_runs_deterministic():
    fitter = SkeletonFitter(subject_height_m=1.70)
    pc = generate_standing_point_cloud(height_m=1.70, n_points=50)

    first_result = fitter.fit(pc)
    for _ in range(5):
        run_result = fitter.fit(pc)
        assert run_result["head"] == first_result["head"]
        assert run_result["torso_top"] == first_result["torso_top"]
        assert run_result["torso_bottom"] == first_result["torso_bottom"]
        assert run_result["fit_quality"] == first_result["fit_quality"]


def test_subject_height_scaling_proportionally_scales_segments():
    # Standing cloud for 1.5m subject vs 1.9m subject
    fitter_short = SkeletonFitter(subject_height_m=1.50)
    fitter_tall = SkeletonFitter(subject_height_m=1.90)

    pc_short = generate_standing_point_cloud(height_m=1.50, n_points=50)
    pc_tall = generate_standing_point_cloud(height_m=1.90, n_points=50)

    res_short = fitter_short.fit(pc_short)
    res_tall = fitter_tall.fit(pc_tall)

    # Segment length: torso_top to torso_bottom
    torso_len_short = res_short["torso_top"][2] - res_short["torso_bottom"][2]
    torso_len_tall = res_tall["torso_top"][2] - res_tall["torso_bottom"][2]

    assert torso_len_tall > torso_len_short
    # Ratio of torso lengths should roughly match 1.9 / 1.5 (~1.267)
    ratio = torso_len_tall / torso_len_short
    assert 1.10 <= ratio <= 1.40
