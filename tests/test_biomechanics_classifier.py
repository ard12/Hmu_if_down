import numpy as np
import pytest
from hub.biomechanics_classifier import FallBiomechanicsClassifier
from hub.fusion_engine import DualFusionEngine, OperatingMode, UnifiedFallState
from hub.csi_pipeline.multi_link_fusion import CSIFallState
from hub.csi_pipeline.pca_features import CSIDynamicFeatures


def generate_trip_trajectory(n_frames: int = 20, dt: float = 0.05):
    """Rapid forward pitch > 45° in < 0.5s, knee flexion at impact > 80°."""
    frames = []
    for i in range(n_frames):
        t = i * dt
        if t < 0.3:
            # Standing
            frames.append({
                "valid": True,
                "head": [0.0, 0.0, 1.65],
                "torso_top": [0.0, 0.0, 1.40],
                "torso_bottom": [0.0, 0.0, 0.85],
                "knee_flexion": 10.0,
            })
        else:
            # Rapid forward pitch
            progress = min(1.0, (t - 0.3) / 0.4)
            pitch_x = 0.8 * progress
            z_drop = 1.65 - 1.35 * progress
            frames.append({
                "valid": True,
                "head": [pitch_x, 0.0, z_drop],
                "torso_top": [pitch_x * 0.8, 0.0, z_drop - 0.2],
                "torso_bottom": [0.0, 0.0, max(0.2, 0.85 - 0.65 * progress)],
                "knee_flexion": 10.0 + 85.0 * progress,
            })
    return frames


def generate_syncope_trajectory(n_frames: int = 50, dt: float = 0.05):
    """Slow gradual descent over 2.5s."""
    frames = []
    for i in range(n_frames):
        t = i * dt
        progress = min(1.0, t / 2.5)
        frames.append({
            "valid": True,
            "head": [0.0, 0.0, 1.65 - 1.40 * progress],
            "torso_top": [0.0, 0.0, 1.40 - 1.20 * progress],
            "torso_bottom": [0.0, 0.0, 0.85 - 0.65 * progress],
            "knee_flexion": 10.0 + 40.0 * progress,
        })
    return frames


def test_trip_trajectory_features_predicts_trip_and_fall():
    classifier = FallBiomechanicsClassifier()
    traj = generate_trip_trajectory()
    res = classifier.classify(traj)

    assert res["fall_type"] == "TRIP_AND_FALL"
    assert res["confidence"] >= 0.70
    assert "forward tripping" in res["clinical_note"].lower() or "forward" in res["clinical_note"].lower()


def test_syncope_trajectory_features_predicts_syncope():
    classifier = FallBiomechanicsClassifier()
    traj = generate_syncope_trajectory()
    res = classifier.classify(traj)

    assert res["fall_type"] == "SYNCOPE"
    assert res["confidence"] >= 0.70
    assert "syncope" in res["clinical_note"].lower()


def test_empty_trajectory_returns_unknown():
    classifier = FallBiomechanicsClassifier()
    res = classifier.classify([])

    assert res["fall_type"] == "UNKNOWN"
    assert res["confidence"] == 0.0
    assert len(res["clinical_note"]) > 0


def test_confidence_above_threshold_for_prototypical_trajectories():
    classifier = FallBiomechanicsClassifier()
    trip_res = classifier.classify(generate_trip_trajectory())
    syncope_res = classifier.classify(generate_syncope_trajectory())

    assert trip_res["confidence"] > 0.70
    assert syncope_res["confidence"] > 0.70


def test_clinical_note_is_non_empty_string():
    classifier = FallBiomechanicsClassifier()
    for traj in [generate_trip_trajectory(), generate_syncope_trajectory()]:
        res = classifier.classify(traj)
        assert isinstance(res["clinical_note"], str)
        assert len(res["clinical_note"]) > 10


def test_feature_dict_contains_all_six_required_keys():
    classifier = FallBiomechanicsClassifier()
    res = classifier.classify(generate_trip_trajectory())
    feats = res["features"]

    expected_keys = {
        "max_trunk_angular_velocity",
        "head_z_drop_velocity",
        "lateral_roll_angle",
        "time_to_floor_s",
        "minimum_z_in_trajectory",
        "knee_flexion_at_impact",
    }
    assert expected_keys.issubset(feats.keys())


def test_agreement_between_csi_and_biomechanics_sets_confirmed():
    classifier = FallBiomechanicsClassifier()
    assert classifier.check_agreement("forward_trip", "TRIP_AND_FALL") is True
    assert classifier.check_agreement("backward_slip", "SLIP_AND_FALL") is True
    assert classifier.check_agreement("syncope_drop", "SYNCOPE") is True
    assert classifier.check_agreement("forward_trip", "SYNCOPE") is False

    # Integration test with DualFusionEngine
    class MockFallTypeClassifier:
        def predict(self, feats):
            return "forward_trip", 0.95

    engine = DualFusionEngine(
        mode=OperatingMode.CSI_ONLY,
        fall_type_classifier=MockFallTypeClassifier(),
        biomechanics_classifier=classifier,
    )
    # Populate trajectory with trip
    for f in generate_trip_trajectory():
        engine.skeleton_trajectory.append(f)

    # Trigger fall confirmation
    feat = CSIDynamicFeatures(
        node_id=1,
        dominant_velocity_mps=2.5,
        energy_surge_ratio=8.0,
        moving_variance=0.01,
        is_velocity_burst=True,
        pc1_signal=np.ones(64),
        ml_fall_probability=0.98,
    )
    engine.update_csi(feat, ml_prob=0.98)

    assert engine.unified_state == UnifiedFallState.CONFIRMED
    assert engine.last_biomechanics_confirmed is True
    assert engine.last_biomechanics is not None
    assert engine.last_biomechanics["fall_type"] == "TRIP_AND_FALL"
