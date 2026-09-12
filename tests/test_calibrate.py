"""Unit tests for the room calibration tool."""

from pathlib import Path
import tempfile
import yaml
import numpy as np
import pytest

from hub.csi_pipeline.preprocessor import CSIPreprocessor
from hub.calibrate import (
    generate_synthetic_baseline,
    compute_calibration,
    run_calibration,
)


def test_synthetic_baseline_generation():
    """Verify synthetic baseline produces valid CSI packets for all nodes."""
    preprocessor = CSIPreprocessor()
    duration = 1.0
    collected, actual_duration = generate_synthetic_baseline(
        duration=duration,
        preprocessor=preprocessor,
        nodes=(1, 2, 3),
        sampling_rate_hz=100.0,
        subcarrier_count=32,
    )

    assert actual_duration == 1.0
    assert set(collected.keys()) == {1, 2, 3}
    for nid in (1, 2, 3):
        assert len(collected[nid]) == 100
        first_pkt = collected[nid][0]
        assert first_pkt.node_id == nid
        assert first_pkt.subcarrier_count == 32
        assert len(first_pkt.amplitudes) == 32


def test_compute_calibration_metrics():
    """Verify computation of variance, RSSI, and recommended thresholds."""
    preprocessor = CSIPreprocessor()
    collected, duration = generate_synthetic_baseline(
        duration=2.0,
        preprocessor=preprocessor,
        nodes=(1, 2, 3),
        sampling_rate_hz=100.0,
    )

    node_stats, thresholds = compute_calibration(collected, duration)


    assert set(node_stats.keys()) == {1, 2, 3}
    for nid, stats in node_stats.items():
        assert stats["status"] == "OK"
        assert stats["packet_count"] == 200
        assert stats["packet_rate_hz"] == 100.0
        assert stats["subcarrier_count"] == 64
        assert stats["mean_variance"] > 0.0
        assert len(stats["mean_amplitudes"]) == 64
        assert len(stats["std_amplitudes"]) == 64


    # Thresholds
    assert "motionless_variance_threshold" in thresholds
    assert "energy_surge_ratio" in thresholds
    assert thresholds["motionless_variance_threshold"] > 0.0
    assert thresholds["energy_surge_ratio"] == 3.0
    assert thresholds["is_noisy_environment"] is False


def test_calibration_yaml_export():
    """Verify calibration YAML output format and readability."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "test_calibration.yaml"

        node_stats, thresholds = run_calibration(
            duration=1.0,
            simulate=True,
            output_path=yaml_path,
        )

        assert yaml_path.exists()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)


        assert "calibration_metadata" in data
        assert data["calibration_metadata"]["mode"] == "simulate"
        assert "recommended_thresholds" in data
        assert "nodes" in data
        assert set(data["nodes"].keys()) == {1, 2, 3}
