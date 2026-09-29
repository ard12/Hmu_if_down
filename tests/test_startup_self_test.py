"""
Unit tests for Startup Self-Test & Boot Preflight Harness (Phase 28).
"""

import pytest
from pathlib import Path
from hub.startup_self_test import StartupSelfTest, PreflightReport
from hub.diagnostics import WatchdogHeartbeat
import time


@pytest.fixture
def tester(tmp_path):
    return StartupSelfTest(project_root=tmp_path)


def test_full_startup_preflight_passes(tester):
    """
    Covers: SRS-REL-001
    Verifies that all preflight components (audit DB, models, space, permissions) pass.
    """
    report = tester.run_preflight()
    assert isinstance(report, PreflightReport)
    assert report.passed is True
    assert report.checks.get("audit_log_store") is True
    assert report.checks.get("directory_permissions") is True
    assert len(report.errors) == 0


def test_watchdog_heartbeat_and_stall_detection():
    """
    Covers: SRS-REL-001, HAZ-037
    Verifies watchdog detects stall when heartbeat is not refreshed within threshold.
    """
    watchdog = WatchdogHeartbeat(stall_timeout_sec=0.1)

    # Initial state should be healthy
    health1 = watchdog.check_health()
    assert health1["healthy"] is True
    assert health1["stall_count"] == 0

    # Simulate stall by decrementing last_heartbeat timestamp past stall_timeout_sec
    watchdog.last_heartbeat -= 0.5
    health2 = watchdog.check_health()
    assert health2["healthy"] is False
    assert health2["stall_count"] >= 1

    # Call beat() and verify recovery
    watchdog.beat()
    health3 = watchdog.check_health()
    assert health3["healthy"] is True
