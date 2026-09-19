"""Tests for Continuous System Diagnostics & IEC 60601-1-8 Watchdog (Milestone 10.4)."""

import time
import pytest
from hub.diagnostics import SystemDiagnosticsWatcher, SystemHealthStatus


def test_diagnostics_compute_rate_and_jitter():
    watcher = SystemDiagnosticsWatcher(window_sec=2.0)
    now = 100.0

    # 100 Hz steady stream: packets every 10ms
    for i in range(100):
        t = now - 1.0 + (i * 0.01)
        watcher.record_csi_packet(1, timestamp=t)

    rate, jitter = watcher.compute_rate_and_jitter(watcher.csi_timestamps[1], now=now)
    assert 95.0 <= rate <= 105.0
    assert jitter < 2.0  # Steady stream -> near-zero jitter


def test_diagnostics_healthy_stream_returns_ok():
    watcher = SystemDiagnosticsWatcher(min_csi_rate_hz=80.0, min_radar_rate_hz=10.0, window_sec=2.0)
    now = 200.0

    # Feed healthy CSI packets to all 3 nodes
    for nid in (1, 2, 3):
        for i in range(100):
            t = now - 1.0 + (i * 0.01)
            watcher.record_csi_packet(nid, timestamp=t)

    # Feed healthy Radar frames at 20 Hz
    for i in range(25):
        t = now - 1.0 + (i * 0.05)
        watcher.record_radar_frame(timestamp=t)

    health = watcher.evaluate_health(current_time=now)
    assert health["status"] == SystemHealthStatus.OK
    assert len(health["issues"]) == 0
    assert health["subsystems"]["radar"]["rate_hz"] >= 15.0


def test_diagnostics_degraded_when_packets_missing():
    watcher = SystemDiagnosticsWatcher(window_sec=2.0)
    now = 300.0

    # No packets sent at all
    health = watcher.evaluate_health(current_time=now)
    assert health["status"] in (SystemHealthStatus.DEGRADED, SystemHealthStatus.FAULT)
    assert len(health["issues"]) > 0


def test_diagnostics_high_jitter_detected():
    watcher = SystemDiagnosticsWatcher(max_jitter_ms=10.0, window_sec=2.0)
    now = 400.0

    # Bursty packets: highly irregular intervals (0.001s, then 0.05s, etc.)
    t = now - 1.0
    intervals = [0.001, 0.05, 0.002, 0.06, 0.001, 0.07] * 5
    for dt in intervals:
        t += dt
        watcher.record_csi_packet(1, timestamp=t)

    health = watcher.evaluate_health(current_time=now)
    jitter_issues = [i for i in health["issues"] if "High network jitter" in i]
    assert len(jitter_issues) > 0


def test_diagnostics_self_test_passes():
    watcher = SystemDiagnosticsWatcher()
    res = watcher.run_self_test()
    assert res["self_test_passed"] is True
    assert res["iec_60601_compliance"] == "PASS"
    assert res["checks"]["csi_pipeline_import"] is True
    assert res["checks"]["vital_signs_import"] is True
    assert res["checks"]["cloud_sync_import"] is True
