"""Tests for ISO 14971 FMEA Risk Analysis (Milestone 9.2)."""

import subprocess
import sys
from pathlib import Path
import pytest
import yaml

from docs.validate_risk_analysis import load_risk_analysis, validate_hazards, REQUIRED_FIELDS


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_risk_yaml_parseable(project_root: Path):
    yaml_path = project_root / "docs" / "risk_analysis.yaml"
    assert yaml_path.exists(), "docs/risk_analysis.yaml must exist"
    hazards = load_risk_analysis(yaml_path)
    assert len(hazards) >= 20, f"Expected >=20 hazards, found {len(hazards)}"


def test_all_hazards_have_required_fields(project_root: Path):
    yaml_path = project_root / "docs" / "risk_analysis.yaml"
    hazards = load_risk_analysis(yaml_path)
    for haz_id, data in hazards.items():
        for field in REQUIRED_FIELDS:
            assert field in data, f"{haz_id} is missing required field '{field}'"


def test_all_residual_risks_marked_acceptable(project_root: Path):
    yaml_path = project_root / "docs" / "risk_analysis.yaml"
    hazards = load_risk_analysis(yaml_path)
    valid, errors = validate_hazards(hazards, fail_on_unacceptable=True)
    assert valid, f"Validation failed with errors: {errors}"


def test_risk_before_equals_severity_times_probability(project_root: Path):
    yaml_path = project_root / "docs" / "risk_analysis.yaml"
    hazards = load_risk_analysis(yaml_path)
    for haz_id, data in hazards.items():
        sev = data["severity"]
        prob = data["probability"]
        rb = data["risk_before"]
        assert rb == sev * prob, f"{haz_id}: risk_before ({rb}) != {sev} * {prob}"


def test_validate_script_exits_zero(project_root: Path):
    script_path = project_root / "docs" / "validate_risk_analysis.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--fail-on-unacceptable"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    assert proc.returncode == 0, f"Script failed with output: {proc.stderr}\n{proc.stdout}"
    assert "SUCCESS: Validated" in proc.stdout
