"""[LEGACY] Unit tests for configuration loading and validation.

These tests cover the older CV-based pipeline config in src/utils.
They are NOT part of the current Wi-Fi CSI + mmWave radar architecture.
Kept for reference — skip in CI unless the src/ pipeline is being actively developed.
"""

import pytest
from src.utils import load_config, get_project_root

pytestmark = pytest.mark.skipif(
    True, reason="Legacy CV pipeline tests — not part of active RF/radar architecture"
)


def test_load_default_config():
    config = load_config()
    assert isinstance(config, dict)
    assert "detector" in config
    assert "fall_heuristics" in config
    assert "alert" in config
    assert "video" in config


def test_fall_heuristics_config_ranges():
    config = load_config()
    heuristics = config["fall_heuristics"]

    assert 0.0 < heuristics["torso_angle_threshold"] < 90.0
    assert 0.0 < heuristics["aspect_ratio_threshold"] < 2.0
    assert heuristics["confirmation_frames"] >= 1
    assert heuristics["recovery_frames"] >= 1


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_file.yaml")
