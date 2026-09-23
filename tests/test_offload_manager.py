"""Unit tests for OffloadManager (Milestone 19.3)."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hub.offload_manager import (
    OffloadDecision,
    OffloadManager,
    OffloadTarget,
    ProviderHealth,
)


def test_default_decision_selects_tensorrt_when_available():
    manager = OffloadManager(sla_target_ms=50.0)
    decision = manager.decide()
    assert isinstance(decision, OffloadDecision)
    assert decision.target == OffloadTarget.TENSORRT
    assert decision.sla_met is True
    assert "tensorrt" in decision.reason


def test_thermal_throttle_skips_tensorrt():
    manager = OffloadManager(thermal_threshold_c=85.0)
    manager.report_thermal(OffloadTarget.TENSORRT, 88.5)
    decision = manager.decide()
    assert decision.target == OffloadTarget.ONNX_CPU


def test_consecutive_failures_disable_provider():
    manager = OffloadManager()
    for _ in range(5):
        manager.report_inference(OffloadTarget.TENSORRT, latency_ms=10.0, success=False)
    health = manager.get_provider_health()
    assert health["tensorrt"]["available"] is False

    decision = manager.decide()
    assert decision.target != OffloadTarget.TENSORRT
    assert decision.target == OffloadTarget.ONNX_CPU


def test_high_jitter_skips_edge():
    manager = OffloadManager(jitter_threshold_ms=20.0)
    manager.register_provider(OffloadTarget.TENSORRT, available=False)
    manager.register_provider(OffloadTarget.ONNX_CPU, available=False)
    decision = manager.decide(network_jitter_ms=25.0)
    # Edge is skipped due to jitter > 20ms, falls back to SKLEARN
    assert decision.target == OffloadTarget.SKLEARN


def test_all_providers_unavailable_falls_back_to_sklearn():
    manager = OffloadManager()
    for target in (OffloadTarget.TENSORRT, OffloadTarget.ONNX_CPU, OffloadTarget.EDGE_ESP32, OffloadTarget.SKLEARN):
        manager.register_provider(target, available=False)
    decision = manager.decide()
    assert decision.target == OffloadTarget.SKLEARN
    assert decision.sla_met is False


def test_report_inference_resets_failure_count_on_success():
    manager = OffloadManager()
    manager.report_inference(OffloadTarget.TENSORRT, latency_ms=15.0, success=False)
    manager.report_inference(OffloadTarget.TENSORRT, latency_ms=15.0, success=False)
    assert manager._providers[OffloadTarget.TENSORRT].consecutive_failures == 2

    manager.report_inference(OffloadTarget.TENSORRT, latency_ms=12.0, success=True)
    assert manager._providers[OffloadTarget.TENSORRT].consecutive_failures == 0
    assert manager._providers[OffloadTarget.TENSORRT].total_inferences == 3


def test_provider_health_dict_keys():
    manager = OffloadManager()
    health = manager.get_provider_health()
    for t in ("tensorrt", "onnx_cpu", "edge_esp32", "sklearn"):
        assert t in health
        assert "available" in health[t]
        assert "thermal_throttled" in health[t]
        assert "consecutive_failures" in health[t]
        assert "last_latency_ms" in health[t]
        assert "total_inferences" in health[t]


def test_decision_count_increments():
    manager = OffloadManager()
    assert manager.decision_count == 0
    manager.decide()
    manager.decide()
    assert manager.decision_count == 2
