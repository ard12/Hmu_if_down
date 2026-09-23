"""Dynamic compute offloading engine.

Routes ML inference to the optimal available execution provider based on:
  - Observed inference latency vs SLA target (default: 50ms)
  - Thermal throttle state (GPU/NPU junction temperature)
  - Network jitter to edge nodes
  - Provider health (consecutive failure count)

@req SRS-PERF-003
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger("offload_manager")


class OffloadTarget(Enum):
    TENSORRT = "tensorrt"
    ONNX_CPU = "onnx_cpu"
    EDGE_ESP32 = "edge_esp32"
    SKLEARN = "sklearn"


@dataclass
class ProviderHealth:
    """Health tracking for a single inference provider."""
    target: OffloadTarget
    consecutive_failures: int = 0
    last_latency_ms: float = 0.0
    is_available: bool = True
    thermal_throttled: bool = False
    total_inferences: int = 0


@dataclass
class OffloadDecision:
    """Result of the offload routing decision."""
    target: OffloadTarget
    reason: str
    sla_met: bool
    fallback_chain: List[str] = field(default_factory=list)


class OffloadManager:
    """Adaptive inference offload coordinator."""

    MAX_CONSECUTIVE_FAILURES = 5
    DEFAULT_SLA_MS = 50.0

    def __init__(
        self,
        sla_target_ms: float = 50.0,
        thermal_threshold_c: float = 85.0,
        jitter_threshold_ms: float = 20.0,
    ):
        self.sla_target_ms = sla_target_ms
        self.thermal_threshold_c = thermal_threshold_c
        self.jitter_threshold_ms = jitter_threshold_ms
        self._providers: Dict[OffloadTarget, ProviderHealth] = {
            t: ProviderHealth(target=t) for t in OffloadTarget
        }
        self._decision_log: List[OffloadDecision] = []

    def register_provider(self, target: OffloadTarget, available: bool = True) -> None:
        """Register or update provider availability."""
        self._providers[target].is_available = available

    def report_thermal(self, target: OffloadTarget, temperature_c: float) -> None:
        """Report GPU/device junction temperature."""
        throttled = temperature_c >= self.thermal_threshold_c
        self._providers[target].thermal_throttled = throttled
        if throttled:
            logger.warning(
                "Provider %s throttled due to temperature: %.1fC >= %.1fC",
                target.value, temperature_c, self.thermal_threshold_c,
            )

    def report_inference(
        self, target: OffloadTarget, latency_ms: float, success: bool
    ) -> None:
        """Report inference completion for health tracking."""
        p = self._providers[target]
        p.last_latency_ms = latency_ms
        p.total_inferences += 1
        if success:
            p.consecutive_failures = 0
        else:
            p.consecutive_failures += 1
            if p.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                p.is_available = False
                logger.warning(
                    "Provider %s disabled after %d consecutive failures.",
                    target.value, p.consecutive_failures,
                )

    def decide(self, network_jitter_ms: float = 0.0) -> OffloadDecision:
        """Select the optimal inference provider."""
        preference_order = [
            OffloadTarget.TENSORRT,
            OffloadTarget.ONNX_CPU,
            OffloadTarget.EDGE_ESP32,
            OffloadTarget.SKLEARN,
        ]

        for target in preference_order:
            p = self._providers[target]
            if not p.is_available:
                continue
            if p.thermal_throttled:
                continue
            if target == OffloadTarget.EDGE_ESP32 and network_jitter_ms > self.jitter_threshold_ms:
                continue

            sla_met = p.last_latency_ms <= self.sla_target_ms or p.total_inferences == 0
            decision = OffloadDecision(
                target=target,
                reason=f"Selected {target.value} (latency={p.last_latency_ms:.1f}ms)",
                sla_met=sla_met,
                fallback_chain=[t.value for t in preference_order],
            )
            self._decision_log.append(decision)
            return decision

        # Ultimate fallback
        decision = OffloadDecision(
            target=OffloadTarget.SKLEARN,
            reason="All providers unavailable; sklearn fallback",
            sla_met=False,
            fallback_chain=["sklearn"],
        )
        self._decision_log.append(decision)
        return decision

    def get_provider_health(self) -> Dict[str, Dict]:
        """Return health status of all providers."""
        return {
            t.value: {
                "available": p.is_available,
                "thermal_throttled": p.thermal_throttled,
                "consecutive_failures": p.consecutive_failures,
                "last_latency_ms": p.last_latency_ms,
                "total_inferences": p.total_inferences,
            }
            for t, p in self._providers.items()
        }

    @property
    def decision_count(self) -> int:
        return len(self._decision_log)
