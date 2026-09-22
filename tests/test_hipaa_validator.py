"""
Tests for HIPAA Technical Safeguards Compliance Validator (Milestone 18.2).
"""

from pathlib import Path
import pytest

from docs.hipaa_validator import SAFEGUARDS, evaluate_safeguard, validate_all_safeguards, generate_compliance_report
from hub.auth import create_break_glass_token, is_token_expired, verify_token


def test_hipaa_all_safeguards_evaluated():
    """Ensure exactly 9 HIPAA technical safeguard standards are defined and audited."""
    assert len(SAFEGUARDS) == 9
    ids = [s["id"] for s in SAFEGUARDS]
    assert "164.312(a)(1)" in ids
    assert "164.312(a)(2)(i)" in ids
    assert "164.312(a)(2)(ii)" in ids
    assert "164.312(a)(2)(iii)" in ids
    assert "164.312(b)" in ids
    assert "164.312(c)(1)" in ids
    assert "164.312(c)(2)" in ids
    assert "164.312(d)" in ids
    assert "164.312(e)(1)-(2)" in ids


def test_hipaa_validator_passes_current_codebase():
    """Verify that current codebase scores 100% PASS across all 9 safeguards."""
    project_root = Path(__file__).resolve().parent.parent
    results = validate_all_safeguards(project_root)

    for r in results:
        assert r["status"] == "PASS", f"Safeguard {r['id']} ({r['name']}) failed verification"


def test_hipaa_report_generation(tmp_path):
    """generate_compliance_report outputs formatted markdown report with 100% compliance."""
    project_root = Path(__file__).resolve().parent.parent
    results = validate_all_safeguards(project_root)
    out_file = tmp_path / "TEST_HIPAA_REPORT.md"

    content = generate_compliance_report(results, out_file)
    assert out_file.exists()
    assert "# HIPAA Technical Safeguards Compliance Assessment Report" in content
    assert "FULL COMPLIANCE (100%)" in content
    assert "164.312(a)(1)" in content
    assert "164.312(b)" in content
    assert "164.312(e)(1)-(2)" in content


def test_access_control_safeguard_detection():
    """Verify §164.312(a)(1) correctly evaluates hub/auth.py."""
    project_root = Path(__file__).resolve().parent.parent
    sg = next(s for s in SAFEGUARDS if s["id"] == "164.312(a)(1)")
    result = evaluate_safeguard(sg, project_root)

    assert result["status"] == "PASS"
    assert "hub/auth.py" in result["verified_files"]


def test_audit_controls_safeguard_detection():
    """Verify §164.312(b) correctly evaluates hub/audit_log.py."""
    project_root = Path(__file__).resolve().parent.parent
    sg = next(s for s in SAFEGUARDS if s["id"] == "164.312(b)")
    result = evaluate_safeguard(sg, project_root)

    assert result["status"] == "PASS"
    assert "hub/audit_log.py" in result["verified_files"]


def test_transmission_security_safeguard_detection():
    """Verify §164.312(e)(1)-(2) correctly evaluates cloud sync and ingress."""
    project_root = Path(__file__).resolve().parent.parent
    sg = next(s for s in SAFEGUARDS if s["id"] == "164.312(e)(1)-(2)")
    result = evaluate_safeguard(sg, project_root)

    assert result["status"] == "PASS"
    assert any("ingress.yaml" in f or "cloud_sync.py" in f for f in result["verified_files"])


def test_integrity_controls_safeguard_detection():
    """Verify §164.312(c)(1) and (c)(2) correctly detect SHA-256 hash chaining and model verification."""
    project_root = Path(__file__).resolve().parent.parent
    sg1 = next(s for s in SAFEGUARDS if s["id"] == "164.312(c)(1)")
    sg2 = next(s for s in SAFEGUARDS if s["id"] == "164.312(c)(2)")

    r1 = evaluate_safeguard(sg1, project_root)
    r2 = evaluate_safeguard(sg2, project_root)

    assert r1["status"] == "PASS"
    assert r2["status"] == "PASS"
    assert "hub/model_registry.py" in r1["verified_files"]
    assert "hub/audit_log.py" in r2["verified_files"]


def test_break_glass_token_functionality():
    """Verify emergency break-glass token creation and expiration check."""
    token = create_break_glass_token(user_id="dr_smith", reason="Cardiac arrest resuscitation in Room 104", ttl_seconds=10)
    assert token.startswith("break_glass_")

    meta = verify_token(token)
    assert meta is not None
    assert meta["role"] == "emergency"
    assert meta["sub"] == "dr_smith"
    assert meta["break_glass"] is True

    # Test expiration
    meta_expired = {"exp": 1000.0, "role": "admin"}
    assert is_token_expired(meta_expired) is True
