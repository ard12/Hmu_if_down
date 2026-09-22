"""Tests for 3D Pose and Biomechanics REST API (Milestone 15.4)."""

import pytest
from fastapi.testclient import TestClient
from hub.dashboard.app import app, broadcaster
from hub.fusion_engine import DualFusionEngine, OperatingMode
from hub.skeleton_fitter import SkeletonFitter
from hub.joint_angles import JointAngleEstimator
from hub.biomechanics_classifier import FallBiomechanicsClassifier


@pytest.fixture
def client():
    return TestClient(app)


def test_get_pose_current_returns_200(client):
    response = client.get("/api/pose/current")
    assert response.status_code == 200
    data = response.json()
    assert "skeleton" in data
    assert "angles" in data
    assert "posture" in data


def test_get_pose_current_has_skeleton_and_angles(client):
    response = client.get("/api/pose/current")
    assert response.status_code == 200
    data = response.json()
    skel = data["skeleton"]
    assert "head" in skel
    assert "torso_top" in skel
    assert "torso_bottom" in skel
    assert "valid" in skel

    angles = data["angles"]
    assert "trunk_inclination_deg" in angles
    assert "knee_flexion_deg" in angles
    assert "head_drop_velocity_mps" in angles


def test_get_pose_trajectory_returns_list(client):
    response = client.get("/api/pose/trajectory")
    assert response.status_code == 200
    data = response.json()
    assert "trajectory" in data
    assert "count" in data
    assert isinstance(data["trajectory"], list)


def test_get_biomechanics_classification_returns_200(client):
    response = client.get("/api/biomechanics/classification")
    assert response.status_code == 200
    data = response.json()
    assert "fall_type" in data
    assert "confidence" in data
    assert "features" in data
    assert "clinical_note" in data
    assert "biomechanics_confirmed" in data


def test_get_biomechanics_classification_has_features(client):
    response = client.get("/api/biomechanics/classification")
    assert response.status_code == 200
    feats = response.json()["features"]
    assert "max_trunk_angular_velocity" in feats
    assert "head_z_drop_velocity" in feats
    assert "lateral_roll_angle" in feats
    assert "time_to_floor_s" in feats
    assert "minimum_z_in_trajectory" in feats
    assert "knee_flexion_at_impact" in feats


def test_pose_html_static_file_accessible(client):
    response = client.get("/static/pose.html")
    assert response.status_code == 200
    assert "3D Video-Free Pose & Biomechanics HUD" in response.text
    assert "skeleton-svg" in response.text


def test_pose_endpoints_work_when_no_fusion_engine(client):
    old_engine = broadcaster.fusion_engine
    try:
        broadcaster.fusion_engine = None
        resp_pose = client.get("/api/pose/current")
        assert resp_pose.status_code == 200
        assert resp_pose.json()["skeleton"] is not None

        resp_traj = client.get("/api/pose/trajectory")
        assert resp_traj.status_code == 200
        assert resp_traj.json()["count"] == 0

        resp_bio = client.get("/api/biomechanics/classification")
        assert resp_bio.status_code == 200
        assert resp_bio.json()["fall_type"] == "UNKNOWN"
    finally:
        broadcaster.fusion_engine = old_engine
