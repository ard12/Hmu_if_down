"""Tests for IEC 62304 Requirement Traceability (Milestone 9.1)."""

from pathlib import Path
import pytest
import yaml

from docs.generate_traceability import (
    load_requirements,
    scan_source_annotations,
    scan_test_annotations,
    merge_mappings,
    coverage_percentage,
    generate_matrix,
    BASELINE_IMPL_MAP,
    BASELINE_TEST_MAP,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_requirements_yaml_parseable(project_root: Path):
    req_path = project_root / "docs" / "requirements.yaml"
    assert req_path.exists(), "docs/requirements.yaml must exist"
    reqs = load_requirements(req_path)
    assert len(reqs) >= 30, f"Expected >=30 requirements, found {len(reqs)}"
    for req_id, data in reqs.items():
        assert "title" in data, f"{req_id} missing title"
        assert "criticality" in data, f"{req_id} missing criticality"
        assert data["criticality"] in ("Critical", "Essential", "Minor"), f"{req_id} invalid criticality"


def test_all_critical_requirements_have_tests(project_root: Path):
    req_path = project_root / "docs" / "requirements.yaml"
    reqs = load_requirements(req_path)
    test_map = merge_mappings(
        scan_test_annotations(project_root / "tests"), BASELINE_TEST_MAP
    )

    critical_missing = []
    for req_id, data in reqs.items():
        if data.get("criticality") == "Critical":
            if not test_map.get(req_id):
                critical_missing.append(req_id)

    assert not critical_missing, f"Critical requirements missing test coverage: {critical_missing}"


def test_traceability_matrix_generates_without_error(project_root: Path):
    req_path = project_root / "docs" / "requirements.yaml"
    reqs = load_requirements(req_path)
    src_map = merge_mappings(
        scan_source_annotations(project_root / "hub"), BASELINE_IMPL_MAP
    )
    test_map = merge_mappings(
        scan_test_annotations(project_root / "tests"), BASELINE_TEST_MAP
    )

    matrix_md = generate_matrix(reqs, src_map, test_map)
    assert matrix_md.startswith("# Software Requirement Traceability Matrix")
    assert "| `SRS-001` |" in matrix_md
    assert "## Coverage Summary by Criticality" in matrix_md


def test_coverage_percentage_above_threshold(project_root: Path):
    req_path = project_root / "docs" / "requirements.yaml"
    reqs = load_requirements(req_path)
    test_map = merge_mappings(
        scan_test_annotations(project_root / "tests"), BASELINE_TEST_MAP
    )
    cov = coverage_percentage(reqs, test_map)
    assert cov >= 80.0, f"Coverage {cov:.1f}% is below 80.0% threshold"
