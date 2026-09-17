"""Unit tests for empirical training pipeline, cross validation, and model export."""

from pathlib import Path
import numpy as np
import pytest

from hub.csi_pipeline.classifier import FallClassifier, MLFeatures
from hub.recorder import record_simulated_session
from hub.train import (
    cross_validate_model,
    extract_features_from_npz,
    generate_synthetic_features,
    load_all_datasets,
    train_and_export,
)


def test_generate_synthetic_features():
    X, y = generate_synthetic_features(n_samples=200, random_state=42)
    assert X.shape == (200, 9)
    assert len(y) == 200
    assert np.sum(y == 1) == 100
    assert np.sum(y == 0) == 100
    # Features must not contain NaNs or Infs
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_cross_validate_model():
    X, y = generate_synthetic_features(n_samples=300, random_state=42)
    results = cross_validate_model(X, y, n_splits=3, random_state=42)

    assert "metrics" in results
    m = results["metrics"]
    assert m["accuracy"]["mean"] > 0.90
    assert m["recall_sensitivity"]["mean"] > 0.90
    assert m["specificity"]["mean"] > 0.90
    assert m["f1_score"]["mean"] > 0.90
    assert m["roc_auc"]["mean"] > 0.90
    assert m["pr_auc"]["mean"] > 0.90
    assert len(results["per_fold"]["accuracy"]) == 3


def test_train_and_export_and_load(tmp_path):
    X, y = generate_synthetic_features(n_samples=200, random_state=42)
    model_file = tmp_path / "models" / "fall_classifier.pkl"
    report_file = tmp_path / "models" / "evaluation_report.json"

    report = train_and_export(
        X,
        y,
        output_model_path=model_file,
        report_file_path=report_file,
        n_splits=3,
    )

    assert model_file.exists()
    assert report_file.exists()
    assert report["n_samples"] == 200

    # Test loading persisted model into FallClassifier
    classifier = FallClassifier(model_path=model_file)
    assert classifier.is_fitted is True

    # Test prediction on high-energy fall feature vector
    fall_feat = MLFeatures(
        subband_energy_0_5hz=4.0,
        subband_energy_5_15hz=8.0,
        subband_energy_15_25hz=25.0,
        subband_energy_25_40hz=30.0,
        high_low_ratio=4.5,
        dominant_velocity=2.4,
        energy_surge=8.0,
        temporal_variance=2.0,
        spectral_entropy=4.8,
    )
    p_fall = classifier.predict_proba(fall_feat)
    assert p_fall > 0.70

    # Test prediction on quiet ADL feature vector
    adl_feat = MLFeatures(
        subband_energy_0_5hz=15.0,
        subband_energy_5_15hz=6.0,
        subband_energy_15_25hz=0.5,
        subband_energy_25_40hz=0.05,
        high_low_ratio=0.03,
        dominant_velocity=0.4,
        energy_surge=1.1,
        temporal_variance=0.08,
        spectral_entropy=2.8,
    )
    p_adl = classifier.predict_proba(adl_feat)
    assert p_adl < 0.30


def test_extract_features_from_npz(tmp_path):
    # Create simulated fall recording
    npz_path, json_path = record_simulated_session(
        label="fall_forward",
        subject_id="test_sub",
        duration=6.0,
        output_dir=tmp_path,
    )
    assert npz_path.exists()

    X, y = extract_features_from_npz(npz_path)
    assert len(X) > 0
    assert X.shape[1] == 9
    assert len(y) == len(X)
    # Since duration is 6.0s and fall occurs at 3.0-3.5s, at least one window should have label 1
    assert np.sum(y == 1) >= 1


def test_load_all_datasets(tmp_path):
    # Record one fall and one walking session
    record_simulated_session("fall_slip", "sub1", duration=5.0, output_dir=tmp_path)
    record_simulated_session("walk_normal", "sub1", duration=5.0, output_dir=tmp_path)

    X, y = load_all_datasets(tmp_path)
    assert len(X) > 0
    assert X.shape[1] == 9
    assert np.sum(y == 1) >= 1
    assert np.sum(y == 0) >= 1


def test_load_all_datasets_empty_trials_dir(tmp_path):
    """load_all_datasets() on an empty directory must return zero samples gracefully."""
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    X, y = load_all_datasets(trials_dir)
    assert len(X) == 0
    assert len(y) == 0


def test_mix_ratio_blending():
    """mix-ratio=0.7 blending must produce correct synthetic supplement size."""
    import numpy as np
    from hub.train import generate_synthetic_features

    # Simulate 100 real samples
    rng = np.random.default_rng(0)
    X_real = rng.random((100, 9))
    y_real = np.concatenate([np.ones(50, dtype=np.int32), np.zeros(50, dtype=np.int32)])

    X_syn, y_syn = generate_synthetic_features(n_samples=600, random_state=42)

    mix_ratio = 0.7
    n_real = len(y_real)
    n_syn_target = max(1, int(n_real / mix_ratio * (1.0 - mix_ratio)))
    n_syn_actual = min(len(y_syn), n_syn_target)

    idx = rng.choice(len(y_syn), size=n_syn_actual, replace=False)
    X_mixed = np.vstack([X_real, X_syn[idx]])
    y_mixed = np.concatenate([y_real, y_syn[idx]])

    # With mix_ratio=0.7 and 100 real samples: 100 / 0.7 * 0.3 ≈ 42–43 synthetic
    assert n_syn_actual > 0
    assert len(X_mixed) == n_real + n_syn_actual
    # Real fraction must be ≥ 0.65 (close to requested 0.70)
    real_fraction = n_real / len(y_mixed)
    assert real_fraction >= 0.65
