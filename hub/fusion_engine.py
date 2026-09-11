"""Dual-sensor fusion engine combining Wi-Fi CSI multipath and mmWave Radar."""

from enum import Enum
import time
from typing import Optional

from .alert_dispatcher import AlertDispatcher
from .csi_pipeline.multi_link_fusion import CSIFallState, MultiLinkFusionEngine
from .csi_pipeline.pca_features import CSIDynamicFeatures
from .mmwave_pipeline.radar_receiver import RadarFallState, RadarPosture, RadarTelemetry


class OperatingMode(Enum):
    CSI_ONLY = "csi"
    RADAR_ONLY = "radar"
    FUSION = "fusion"


class UnifiedFallState(Enum):
    NORMAL = "Normal"
    SUSPECTED = "Fall Suspected"
    CONFIRMED = "FALL DETECTED"
    RECOVERED = "Recovered"


class DualFusionEngine:
    """Manages multi-modal consensus and triggers alerts."""

    def __init__(
        self,
        mode: OperatingMode = OperatingMode.FUSION,
        alert_dispatcher: Optional[AlertDispatcher] = None,
        csi_engine: Optional[MultiLinkFusionEngine] = None,
    ):
        self.mode = mode
        self.alert = alert_dispatcher or AlertDispatcher()
        self.csi_engine = csi_engine or MultiLinkFusionEngine()

        self.last_radar: Optional[RadarTelemetry] = None
        self.last_csi_state: CSIFallState = CSIFallState.NORMAL
        self.unified_state: UnifiedFallState = UnifiedFallState.NORMAL

    def update_csi(self, features: CSIDynamicFeatures, current_time: Optional[float] = None) -> UnifiedFallState:
        """Process incoming CSI feature from one of the tracker nodes."""
        self.last_csi_state = self.csi_engine.register_feature(features, current_time)
        return self._evaluate_consensus()

    def update_radar(self, telemetry: RadarTelemetry) -> UnifiedFallState:
        """Process incoming radar telemetry from mmWave gateway."""
        self.last_radar = telemetry
        return self._evaluate_consensus()

    def _evaluate_consensus(self) -> UnifiedFallState:
        prev_state = self.unified_state

        if self.mode == OperatingMode.CSI_ONLY:
            if self.last_csi_state == CSIFallState.CONFIRMED_FALL:
                self.unified_state = UnifiedFallState.CONFIRMED
            elif self.last_csi_state == CSIFallState.SUSPECTED_FALL:
                self.unified_state = UnifiedFallState.SUSPECTED
            elif self.last_csi_state == CSIFallState.RECOVERED:
                self.unified_state = UnifiedFallState.RECOVERED
            else:
                self.unified_state = UnifiedFallState.NORMAL

        elif self.mode == OperatingMode.RADAR_ONLY:
            if not self.last_radar:
                self.unified_state = UnifiedFallState.NORMAL
            elif self.last_radar.fall_state in (RadarFallState.CONFIRMED, RadarFallState.DWELLING):
                self.unified_state = UnifiedFallState.CONFIRMED
            elif self.last_radar.fall_state == RadarFallState.SUSPECTED:
                self.unified_state = UnifiedFallState.SUSPECTED
            else:
                self.unified_state = UnifiedFallState.NORMAL

        elif self.mode == OperatingMode.FUSION:
            # Dual-modality cross-verification:
            # Condition 1: Radar directly confirms fall AND target is near floor
            radar_confirmed = (
                self.last_radar is not None
                and self.last_radar.fall_state in (RadarFallState.CONFIRMED, RadarFallState.DWELLING)
                and (self.last_radar.target_height_m < 0.45 or self.last_radar.posture == RadarPosture.LYING)
            )

            # Condition 2: CSI observes multi-link velocity burst & post-impact stillness
            csi_confirmed = (self.last_csi_state == CSIFallState.CONFIRMED_FALL)

            # Consensus Rule:
            # - If BOTH confirm -> Immediate Highest-Confidence Emergency
            # - If radar confirms floor height + CSI suspected -> Confirmed
            # - If CSI confirms and radar not reporting -> Fall suspected or confirmed
            if radar_confirmed and (csi_confirmed or self.last_csi_state == CSIFallState.SUSPECTED_FALL):
                self.unified_state = UnifiedFallState.CONFIRMED
            elif radar_confirmed or csi_confirmed:
                self.unified_state = UnifiedFallState.CONFIRMED
            elif self.last_csi_state == CSIFallState.SUSPECTED_FALL:
                self.unified_state = UnifiedFallState.SUSPECTED
            else:
                self.unified_state = UnifiedFallState.NORMAL

        # Trigger alert on transition to CONFIRMED
        if self.unified_state == UnifiedFallState.CONFIRMED and prev_state != UnifiedFallState.CONFIRMED:
            details = f"Mode={self.mode.value}, CSI={self.last_csi_state.value}"
            if self.last_radar:
                details += f", RadarHeight={self.last_radar.target_height_m:.2f}m"
            self.alert.trigger_alarm(self.mode.value, "FALL_CONFIRMED", details)

        return self.unified_state
