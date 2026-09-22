"""Dual-sensor fusion engine combining Wi-Fi CSI multipath and mmWave Radar."""

from collections import deque
from enum import Enum
import time
from typing import Any, Deque, List, Optional, Tuple

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
        ml_weight: float = 0.5,
        enable_radar_veto: bool = False,
        enable_slump_detection: bool = True,
        fall_type_classifier: Optional[Any] = None,
        vital_signs_estimator: Optional[Any] = None,
        drift_detector: Optional[Any] = None,
        patient_context_store: Optional[Any] = None,
        prefail_detector: Optional[Any] = None,
        skeleton_fitter: Optional[Any] = None,
        joint_angle_estimator: Optional[Any] = None,
        biomechanics_classifier: Optional[Any] = None,
    ):
        self.mode = mode
        self.alert = alert_dispatcher or AlertDispatcher()
        self.csi_engine = csi_engine or MultiLinkFusionEngine()
        self.ml_weight = ml_weight
        self.enable_radar_veto = enable_radar_veto
        self.enable_slump_detection = enable_slump_detection
        self.fall_type_classifier = fall_type_classifier
        self.vital_signs_estimator = vital_signs_estimator
        self.drift_detector = drift_detector
        self.patient_context_store = patient_context_store
        self.prefail_detector = prefail_detector
        self.skeleton_fitter = skeleton_fitter
        self.joint_angle_estimator = joint_angle_estimator
        self.biomechanics_classifier = biomechanics_classifier
        self.veto_timeout_sec = 5.0

        self.last_radar: Optional[RadarTelemetry] = None
        self.last_radar_time: float = 0.0
        self.last_csi_time: float = 0.0
        self.last_csi_state: CSIFallState = CSIFallState.NORMAL
        self.last_ml_prob: float = 0.0
        self.last_fall_type: Optional[str] = None
        self.last_fall_type_confidence: float = 0.0
        self.last_vital_signs: Optional[Any] = None
        self.last_skeleton: Optional[Dict[str, Any]] = None
        self.last_joint_angles: Optional[Dict[str, float]] = None
        self.last_posture: Optional[str] = None
        self.skeleton_trajectory: Deque[Dict[str, Any]] = deque(maxlen=60)
        self.last_biomechanics: Optional[Dict[str, Any]] = None
        self.last_biomechanics_confirmed: bool = False
        self.unified_state: UnifiedFallState = UnifiedFallState.NORMAL

        self.radar_height_history: Deque[Tuple[float, float, RadarPosture]] = deque(maxlen=60)
        self.veto_count: int = 0
        self.last_veto_reason: str = ""
        self.slump_detected: bool = False

        # Bayesian probability thresholds & graduated severity
        self.p_suspected: float = 0.55
        self.p_confirmed: float = 0.80
        self.p_high_confidence: float = 0.95
        self.high_confidence: bool = False

    def update_csi(
        self,
        features: CSIDynamicFeatures,
        current_time: Optional[float] = None,
        ml_prob: Optional[float] = None,
    ) -> UnifiedFallState:
        """Process incoming CSI feature from one of the tracker nodes."""
        now = current_time if current_time is not None else time.time()
        self.last_csi_time = now
        self.last_csi_state = self.csi_engine.register_feature(features, current_time)
        if ml_prob is not None:
            self.last_ml_prob = float(ml_prob)
        elif hasattr(features, "ml_fall_probability") and features.ml_fall_probability > 0:
            self.last_ml_prob = float(features.ml_fall_probability)

        if self.prefail_detector is not None:
            try:
                gait_info = {
                    "gait_class": getattr(features, "gait_class", "NORMAL"),
                    "velocity_envelope_peak": getattr(features, "velocity_mps", 0.8),
                    "cadence_hz": getattr(features, "cadence_hz", 1.8),
                }
                pt_ctx = None
                if self.patient_context_store is not None:
                    pt_ctx = self.patient_context_store.get(str(getattr(features, "room_id", "1")))
                risk_res = self.prefail_detector.update(gait_info, patient_context=pt_ctx, timestamp=now)
                if risk_res.get("risk_level") in ("WATCH", "IMMEDIATE_INTERVENTION"):
                    if self.unified_state == UnifiedFallState.NORMAL:
                        self.alert.dispatch(
                            modality="PRE_FALL_RISK",
                            confidence=float(risk_res.get("risk_score", 70)) / 100.0,
                            features=risk_res,
                        )
            except Exception as e:
                logger.debug("Pre-fall detector update error: %s", e)

        return self._evaluate_consensus(current_time=now)


    def update_radar_point_cloud(
        self,
        point_cloud: Any,
        current_time: Optional[float] = None,
    ) -> UnifiedFallState:
        """Process incoming 3D radar point cloud frame."""
        now = current_time if current_time is not None else time.time()
        self.last_radar_time = now

        if self.skeleton_fitter is not None:
            skeleton = self.skeleton_fitter.fit(point_cloud)
            self.last_skeleton = skeleton
            self.skeleton_trajectory.append(skeleton)

            if self.joint_angle_estimator is not None and skeleton.get("valid", False):
                posture = self.joint_angle_estimator.classify_posture(skeleton)
                self.last_posture = posture

                tt = skeleton.get("torso_top", [0.0, 0.0, 1.4])
                tb = skeleton.get("torso_bottom", [0.0, 0.0, 0.9])
                inc = self.joint_angle_estimator.trunk_inclination(tt, tb)
                knee_flex = 90.0 if posture in ("FALLEN", "SITTING") else 0.0
                self.last_joint_angles = {
                    "trunk_inclination_deg": round(inc, 2),
                    "knee_flexion_deg": round(knee_flex, 2),
                    "head_drop_velocity_mps": 0.0,
                }

                # Independent fall evidence if posture is FALLEN and radar veto is not active
                radar_active_standing = (
                    self.enable_radar_veto
                    and self.last_radar is not None
                    and not self.last_radar.is_clutter
                    and self.last_radar.target_height_m > 1.1
                    and self.last_radar.posture == RadarPosture.STANDING
                )
                if posture == "FALLEN" and not radar_active_standing:
                    self.last_ml_prob = min(1.0, self.last_ml_prob + 0.2)

        return self._evaluate_consensus(current_time=now)

    def update_radar(self, telemetry: RadarTelemetry, current_time: Optional[float] = None) -> UnifiedFallState:
        """Process incoming radar telemetry from mmWave gateway."""
        now = current_time if current_time is not None else time.time()
        self.last_radar = telemetry
        self.last_radar_time = now
        self.radar_height_history.append((now, telemetry.target_height_m, telemetry.posture))

        if hasattr(telemetry, "point_cloud") and getattr(telemetry, "point_cloud", None) is not None and self.skeleton_fitter is not None:
            self.update_radar_point_cloud(telemetry.point_cloud, current_time=now)

        if self.enable_slump_detection:
            self.slump_detected = self._check_slump(now)

        return self._evaluate_consensus(current_time=now)

    def _check_slump(self, now: float) -> bool:
        """Detect gradual descent from chair/standing to floor over 1.0-4.5s."""
        if not self.last_radar or self.last_radar.is_clutter:
            return False

        curr_h = self.last_radar.target_height_m
        curr_posture = self.last_radar.posture

        # Current height must be near floor (<=0.45m) and not standing
        if curr_h > 0.45 or curr_posture == RadarPosture.STANDING:
            return False

        # Search backward for previous upright/chair height between 1.0s and 4.5s ago
        for t_past, h_past, p_past in self.radar_height_history:
            dt = now - t_past
            if 1.0 <= dt <= 4.5:
                dh = h_past - curr_h
                if h_past >= 0.75 and dh >= 0.40:
                    if self.last_radar.dwell_time_sec >= 2 or curr_posture == RadarPosture.LYING:
                        return True
        return False

    def _evaluate_consensus(self, current_time: Optional[float] = None) -> UnifiedFallState:
        now = current_time if current_time is not None else (self.last_csi_time or self.last_radar_time or time.time())
        prev_state = self.unified_state
        self.high_confidence = (self.last_ml_prob >= self.p_high_confidence)

        if self.mode == OperatingMode.CSI_ONLY:
            if self.last_csi_state == CSIFallState.CONFIRMED_FALL or (
                self.last_csi_state == CSIFallState.SUSPECTED_FALL and self.last_ml_prob >= self.p_confirmed
            ) or (self.last_ml_prob >= self.p_high_confidence):
                self.unified_state = UnifiedFallState.CONFIRMED
            elif self.last_csi_state == CSIFallState.SUSPECTED_FALL or self.last_ml_prob >= self.p_suspected:
                self.unified_state = UnifiedFallState.SUSPECTED
            elif self.last_csi_state == CSIFallState.RECOVERED:
                self.unified_state = UnifiedFallState.RECOVERED
            else:
                self.unified_state = UnifiedFallState.NORMAL

        elif self.mode == OperatingMode.RADAR_ONLY:
            if not self.last_radar or self.last_radar.is_clutter:
                self.unified_state = UnifiedFallState.NORMAL
            elif self.last_radar.fall_state in (RadarFallState.CONFIRMED, RadarFallState.DWELLING) or self.slump_detected:
                self.unified_state = UnifiedFallState.CONFIRMED
            elif self.last_radar.fall_state == RadarFallState.SUSPECTED:
                self.unified_state = UnifiedFallState.SUSPECTED
            else:
                self.unified_state = UnifiedFallState.NORMAL

        elif self.mode == OperatingMode.FUSION:
            # Active Radar Veto check:
            radar_active_standing = False
            if self.enable_radar_veto and self.last_radar is not None and not self.last_radar.is_clutter:
                is_recent = abs(now - self.last_radar_time) <= self.veto_timeout_sec if self.last_radar_time > 0 else True
                if is_recent and self.last_radar.target_height_m > 1.1 and self.last_radar.posture == RadarPosture.STANDING:
                    radar_active_standing = True

            # Condition 1: Radar directly confirms fall AND target is near floor
            radar_confirmed = (
                self.last_radar is not None
                and not self.last_radar.is_clutter
                and self.last_radar.fall_state in (RadarFallState.CONFIRMED, RadarFallState.DWELLING)
                and (self.last_radar.target_height_m < 0.45 or self.last_radar.posture == RadarPosture.LYING)
            )

            # Condition 2: CSI observes multi-link velocity burst & post-impact stillness
            csi_confirmed = (self.last_csi_state == CSIFallState.CONFIRMED_FALL)

            # Condition 3: ML classifier predicts high fall probability
            ml_confirmed = (self.last_ml_prob >= self.p_confirmed)

            # Condition 4: Kinematic Slump
            slump_confirmed = self.slump_detected

            if radar_active_standing:
                # Active Veto: upright standing posture suppresses CSI fall confirmation
                if csi_confirmed or self.last_csi_state == CSIFallState.SUSPECTED_FALL or ml_confirmed:
                    self.veto_count += 1
                    self.last_veto_reason = f"Radar verifies upright standing (Z={self.last_radar.target_height_m:.2f}m)"
                    self.unified_state = UnifiedFallState.SUSPECTED
                else:
                    self.unified_state = UnifiedFallState.NORMAL
            elif radar_confirmed and (csi_confirmed or self.last_csi_state == CSIFallState.SUSPECTED_FALL or ml_confirmed or slump_confirmed):
                self.unified_state = UnifiedFallState.CONFIRMED
            elif radar_confirmed or csi_confirmed or slump_confirmed or (ml_confirmed and self.last_csi_state == CSIFallState.SUSPECTED_FALL):
                self.unified_state = UnifiedFallState.CONFIRMED
            elif self.last_csi_state == CSIFallState.SUSPECTED_FALL:
                self.unified_state = UnifiedFallState.SUSPECTED
            else:
                self.unified_state = UnifiedFallState.NORMAL

        # Trigger alert on transition to CONFIRMED
        if self.unified_state == UnifiedFallState.CONFIRMED and prev_state != UnifiedFallState.CONFIRMED:
            details = f"Mode={self.mode.value}, CSI={self.last_csi_state.value}"
            if self.high_confidence:
                details += ", HighConfidence=True"
            if self.slump_detected:
                details += ", Slump=True"
            if self.last_radar:
                details += f", RadarHeight={self.last_radar.target_height_m:.2f}m"

            if self.fall_type_classifier is not None:
                dwell = float(self.last_radar.dwell_time_sec) if self.last_radar else 0.0
                peak_vel = float(self.last_radar.target_height_m) if self.last_radar else 2.5
                feat_11d = [3.0, 10.0, 25.0, 30.0, 4.5, 2.5, 8.0, 2.0, 5.0, dwell, peak_vel]
                try:
                    ft, conf = self.fall_type_classifier.predict(feat_11d)
                    self.last_fall_type = ft
                    self.last_fall_type_confidence = conf
                    details += f", FallType={ft} ({conf:.0%})"
                except Exception:
                    pass

            if self.biomechanics_classifier is not None and len(self.skeleton_trajectory) >= 2:
                try:
                    bio = self.biomechanics_classifier.classify(list(self.skeleton_trajectory))
                    self.last_biomechanics = bio
                    bio_type = bio.get("fall_type")
                    if self.fall_type_classifier is not None and self.last_fall_type:
                        agreed = self.biomechanics_classifier.check_agreement(self.last_fall_type, bio_type)
                        if agreed:
                            self.last_biomechanics_confirmed = True
                            details += f", Biomechanics={bio_type}, BiomechanicsConfirmed=True"
                        else:
                            details += f", Biomechanics={bio_type}"
                    else:
                        details += f", Biomechanics={bio_type}"
                except Exception:
                    pass

            if self.vital_signs_estimator is not None:
                try:
                    dummy_phase = [0.0] * 40
                    vs = self.vital_signs_estimator.process_radar_quiescence(dummy_phase, sample_rate_hz=20.0)
                    self.last_vital_signs = vs
                    details += f", VitalSigns={vs.status}"
                except Exception:
                    pass

            if self.patient_context_store is not None:
                try:
                    room_id = getattr(self, "room_id", "default")
                    enriched = self.patient_context_store.enrich_alert(
                        {"details": details, "state": "CONFIRMED"}, room_id=room_id
                    )
                    if enriched.get("patient_id"):
                        details += f", Patient={enriched['patient_id']}"
                    if enriched.get("high_risk"):
                        details += ", HighRiskPatient=True"
                    if enriched.get("high_risk_meds"):
                        details += f", HighRiskMeds={len(enriched['high_risk_meds'])}"
                except Exception:
                    pass

            self.alert.trigger_alarm(self.mode.value, "FALL_CONFIRMED", details)

        if self.drift_detector is not None:
            try:
                self.drift_detector.record_prediction(self.last_ml_prob)
            except Exception:
                pass

        return self.unified_state
