"""Tests for Clinical Performance Validation Report Generator (Milestone 9.4)."""

import json
from pathlib import Path
import pytest

from docs.generate_clinical_report import (
    load_evaluation_report,
    load_audit_stats,
    load_git_metadata,
    render_report,
)
from hub.audit_log import AuditLog


def test_load_evaluation_report_valid_json(tmp_path: Path):
    mock_eval = {
        "model_name": "TestModel",
        "sensitivity": 0.995,
        "specificity": 0.991,
        "accuracy": 0.993,
        "brier_score": 0.012,
    }
    report_file = tmp_path / "mock_eval.json"
    report_file.write_text(json.dumps(mock_eval), encoding="utf-8")

    loaded = load_evaluation_report(report_file)
    assert loaded["model_name"] == "TestModel"
    assert loaded["sensitivity"] == 0.995
    assert loaded["specificity"] == 0.991
    assert loaded["brier_score"] == 0.012


def test_render_report_contains_sensitivity(tmp_path: Path):
    eval_data = {
        "sensitivity": 0.992,
        "specificity": 0.989,
        "accuracy": 0.991,
        "roc_auc": 0.998,
        "pr_auc": 0.997,
        "brier_score": 0.015,
    }
    audit_data = {"total_events": 10, "fall_count": 2, "cancellation_count": 0, "cancellation_rate": 0.0}
    git_meta = {"version": "3.3.0", "commit": "abcdef1", "branch": "main"}

    report_md = render_report(eval_data, audit_data, git_meta)
    assert "Sensitivity" in report_md
    assert "99.20%" in report_md
    assert "Specificity" in report_md
    assert "98.90%" in report_md
    assert "Executive Summary" in report_md


def test_render_report_contains_version_string(tmp_path: Path):
    eval_data = {}
    audit_data = {}
    git_meta = {"version": "3.3.0", "commit": "1234567", "branch": "main"}

    report_md = render_report(eval_data, audit_data, git_meta)
    assert "v3.3.0" in report_md
    assert "1234567" in report_md


def test_generate_audit_stats_empty_db(tmp_path: Path):
    db_file = tmp_path / "empty_audit.db"
    log = AuditLog(db_path=db_file)
    # Database initialized with empty table
    stats = load_audit_stats(db_file)
    assert stats["total_events"] == 0
    assert stats["fall_count"] == 0
    assert stats["cancellation_rate"] == 0.0


def test_generate_audit_stats_with_events(tmp_path: Path):
    db_file = tmp_path / "populated_audit.db"
    log = AuditLog(db_path=db_file)

    for i in range(5):
        log.append("FALL_CONFIRMED", {"room_id": 1, "confidence": 0.92}, room_id=1)
    log.append("FALL_CANCELLED", {"room_id": 1, "reason": "user_cancelled"}, room_id=1)
    log.append("CALIBRATE", {"room_id": 1}, room_id=1)

    stats = load_audit_stats(db_file)
    assert stats["total_events"] == 7
    assert stats["fall_count"] == 5
    assert stats["cancellation_count"] == 1
    assert stats["calibration_count"] == 1
    assert stats["unique_rooms"] == 1
    assert abs(stats["cancellation_rate"] - 0.20) < 1e-4
