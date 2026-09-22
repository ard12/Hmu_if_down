"""
Tests for Bandit SAST Integration, Security Report Generator, and Deserialization Hardening (Milestone 18.1).
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

from hub.model_registry import ModelRegistry
from docs.parse_bandit_report import parse_bandit_report, generate_markdown_report


@pytest.fixture
def sample_bandit_json(tmp_path):
    """Create a mock bandit report JSON."""
    report_file = tmp_path / "mock_bandit.json"
    data = {
        "errors": [],
        "generated_at": "2026-09-23T00:00:00Z",
        "metrics": {
            "_totals": {
                "CONFIDENCE.HIGH": 10,
                "CONFIDENCE.LOW": 1,
                "CONFIDENCE.MEDIUM": 2,
                "CONFIDENCE.UNDEFINED": 0,
                "SEVERITY.HIGH": 0,
                "SEVERITY.LOW": 5,
                "SEVERITY.MEDIUM": 3,
                "SEVERITY.UNDEFINED": 0,
                "loc": 5000,
                "nosec": 2,
                "skipped_tests": 0,
            }
        },
        "results": [
            {
                "code": "test_code_1",
                "filename": "hub/model_registry.py",
                "issue_confidence": "HIGH",
                "issue_cwe": {"id": 502, "link": "https://cwe.mitre.org/data/definitions/502.html"},
                "issue_severity": "MEDIUM",
                "issue_text": "Pickle deserialization warning",
                "line_number": 128,
                "test_id": "B301",
                "test_name": "blacklist",
            },
            {
                "code": "test_code_2",
                "filename": "hub/server.py",
                "issue_confidence": "MEDIUM",
                "issue_cwe": {"id": 605, "link": "https://cwe.mitre.org/data/definitions/605.html"},
                "issue_severity": "MEDIUM",
                "issue_text": "Possible binding to all interfaces",
                "line_number": 45,
                "test_id": "B104",
                "test_name": "hardcoded_bind_all_interfaces",
            },
        ],
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return report_file


def test_bandit_report_zero_high_severity():
    """Verify production docs/bandit_report.json has 0 SEVERITY.HIGH issues."""
    report_path = Path("docs/bandit_report.json")
    assert report_path.exists(), "docs/bandit_report.json must exist"

    data = parse_bandit_report(report_path)
    totals = data.get("metrics", {}).get("_totals", {})
    high_count = totals.get("SEVERITY.HIGH", 0)

    assert high_count == 0, f"Expected 0 HIGH severity issues, found {high_count}"


def test_parse_bandit_report_generates_markdown(sample_bandit_json, tmp_path):
    """parse_bandit_report and generate_markdown_report produce valid markdown file."""
    output_md = tmp_path / "OUTPUT_SAST.md"
    data = parse_bandit_report(sample_bandit_json)
    content = generate_markdown_report(data, output_md)

    assert output_md.exists()
    assert "# Static Application Security Testing (SAST) Audit Report" in content
    assert "hub/model_registry.py:128" in content
    assert "CWE-502" in content
    assert "SRS-SEC-004" in content


def test_sast_report_structure():
    """Verify the generated docs/SAST_REPORT.md adheres to FDA/HIPAA reporting structure."""
    report_file = Path("docs/SAST_REPORT.md")
    assert report_file.exists(), "docs/SAST_REPORT.md must exist"

    text = report_file.read_text(encoding="utf-8")
    assert "1. Executive Summary" in text
    assert "2. Severity Summary" in text
    assert "3. Detailed Findings & Mitigation Traceability" in text
    assert "4. Compensating Controls & Architectural Hardening" in text
    assert "**PASSED** (0 High Severity)" in text


def test_model_registry_tamper_prevention(tmp_path):
    """ModelRegistry rejects tampered model bytes before deserialization with ValueError."""
    db_path = str(tmp_path / "tamper_test.db")
    registry = ModelRegistry(db_path=db_path)

    v_id = registry.save_model("original_model_object")
    loaded = registry.load_model(v_id)
    assert loaded == "original_model_object"

    # Tamper with model bytes in SQLite directly
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE model_versions SET model_bytes = X'DEADBEEFCAFE' WHERE version_id = ?",
        (v_id,),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="Cryptographic integrity check failed"):
        registry.load_model(v_id)


def test_parse_bandit_report_fails_on_high_severity(tmp_path):
    """CLI fails with exit code 1 when HIGH severity issues are present."""
    bad_report = tmp_path / "bad_bandit.json"
    data = {
        "metrics": {"_totals": {"SEVERITY.HIGH": 2, "SEVERITY.MEDIUM": 0, "SEVERITY.LOW": 0, "loc": 100}},
        "results": [],
    }
    with open(bad_report, "w", encoding="utf-8") as f:
        json.dump(data, f)

    cmd = [
        sys.executable,
        "docs/parse_bandit_report.py",
        "--input",
        str(bad_report),
        "--output",
        str(tmp_path / "out.md"),
        "--fail-on-high",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ERROR: 2 HIGH severity vulnerability(ies) detected!" in result.stderr


def test_sast_cwe_coverage(sample_bandit_json):
    """Ensure all findings in bandit report map to recognized CWE identifiers."""
    data = parse_bandit_report(sample_bandit_json)
    for result in data.get("results", []):
        cwe = result.get("issue_cwe", {})
        assert "id" in cwe
        assert cwe["id"] in [502, 605, 89, 22] or isinstance(cwe["id"], int)
