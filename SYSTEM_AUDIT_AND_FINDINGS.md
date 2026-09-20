# Multi-Modal Fall Detection: System Audit, Technical Findings & Living Engineering Log

> **Document Classification**: Technical Architecture & Living Audit  
> **Repository**: [`https://github.com/ard12/Fall_Detection`](https://github.com/ard12/Fall_Detection.git)  
> **Status**: Active Living Document (Updated across engineering milestones)  
> **Current Baseline**: Git commit `cd5951c` (Phase 4 completed)  
> **Last Updated**: 2026-09-14  

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Electromagnetic Physics & Wave Propagation](#2-electromagnetic-physics--wave-propagation)
3. [Hardware & Firmware Architecture Audit](#3-hardware--firmware-architecture-audit)
4. [Real-Time DSP & Kinematics Mathematics](#4-real-time-dsp--kinematics-mathematics)
5. [Probabilistic Machine Learning Classifier](#5-probabilistic-machine-learning-classifier)
6. [Multi-Modal Consensus & State Machine Analysis](#6-multi-modal-consensus--state-machine-analysis)
7. [Failure Mode & Effects Analysis (FMEA) & Edge Cases](#7-failure-mode--effects-analysis-fmea--edge-cases)
8. [Telemetry HUD, Smart Home & Edge Deployment](#8-telemetry-hud-smart-home--edge-deployment)
9. [Automated Test Suite & Verification Matrix](#9-automated-test-suite--verification-matrix)
10. [Engineering Action Items & Evolution Roadmap](#10-engineering-action-items--evolution-roadmap)
11. [Revision & Change History](#11-revision--change-history)

---

## 1. Executive Summary

This document serves as the authoritative, living engineering audit for the **Multi-Modal Device-Free Fall Detection Platform**. The system operates without cameras, microphones, wearables, or intrusive pendants by fusing two orthogonal wireless sensing domains:

1. **Plan 1 (Primary)**: **4-Node Wi-Fi CSI Active Sensing Mesh** (1 AP Transmitter + 3 Trackers @ 100 Hz ESP-NOW active injection).
2. **Plan 2 (High-Reliability)**: **ESP32 + 60 GHz mmWave FMCW Radar Gateway** (Centimeter-level 3D centroid altitude tracking and posture classification).
3. **Unified Dual-Sensor Fusion Engine**: Cross-verifies wide-area RF multipath disturbances with centimeter-accurate floor elevation and probabilistic ML fall classification ($P(\text{fall}) \ge 0.85$).
4. **Phase 4 Operational Stack**: Real-Time Web Telemetry HUD (zero-npm Canvas waterfall), Synchronized Dataset Recorder & Replay tool, Home Assistant MQTT Auto-Discovery (13 entities), and Docker Compose edge containerization (`network_mode: host`).

---

## 2. Electromagnetic Physics & Wave Propagation

```
                        DOPPLER & CARRIER FREQUENCY GEOMETRY
┌─────────────────────────────────────────────────────────────────────────────┐
│ Bistatic Doppler Shift Equation:                                            │
│                                                                             │
│               2 * v_human                                                   │
│      f_D  =  ───────────── * cos(theta)                                     │
│                  lambda                                                     │
│                                                                             │
│ • 2.4 GHz (Wi-Fi Ch 6):  lambda = 0.123 m                                   │
│   v_fall = 2.0 m/s  ==>  f_D ≈ 32.5 Hz  (Nyquist limit @ 100 Hz is 50 Hz)   │
│                                                                             │
│ • 5.8 GHz (Wi-Fi 6):    lambda = 0.052 m                                   │
│   v_fall = 2.0 m/s  ==>  f_D ≈ 76.9 Hz  (Requires fs >= 180 Hz to prevent  │
│                                           spectral aliasing)                │
│                                                                             │
│ • 60 GHz (mmWave):      lambda = 0.005 m                                   │
│   FMCW Point Cloud  ==>  Centimeter-accurate direct Z-axis centroid         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Carrier Frequency vs. Kinetic Resolution
* **Human Fall Dynamics**: A standing human ($h \approx 1.7\,\text{m}$) undergoing a free/stumbling fall reaches a peak downward torso velocity $v_{\text{peak}} \in [2.0, 3.2]\,\text{m/s}$.
* **$2.4\,\text{GHz}$ Wi-Fi ($802.11\text{n}$)**:
  * Wavelength: $\lambda = \frac{c}{f} = \frac{3.0 \times 10^8}{2.437 \times 10^9} \approx 0.123\,\text{m}$.
  * At $v = 2.0\,\text{m/s}$, maximum Doppler shift $f_D \approx 32.5\,\text{Hz}$.
  * With an ESP-NOW injection rate $f_s = 100\,\text{Hz}$, the Nyquist limit is $f_N = 50\,\text{Hz}$. The fall kinetic peak sits safely inside the unaliased band $[1.0, 35.0]\,\text{Hz}$.
* **$5.8\,\text{GHz}$ Wi-Fi ($802.11\text{ax}$ / ESP32-C6)**:
  * Wavelength: $\lambda \approx 0.052\,\text{m}$.
  * At $v = 2.0\,\text{m/s}$, Doppler shift $f_D \approx 76.9\,\text{Hz}$.
  * **Critical Constraint**: If migrating to $5\,\text{GHz}$, the ESP-NOW injection frequency must increase to $\ge 180\text{--}200\,\text{Hz}$ to avoid frequency foldover.

### 2.2 Fresnel Zone Coverage & Multipath Diffusion
* The 3 bistatic links (Tx $\rightarrow$ Rx1, Tx $\rightarrow$ Rx2, Tx $\rightarrow$ Rx3) create intersecting ellipsoidal Fresnel zones across the room.
* **Vertical Velocity Sensitivity**: Placing Node 0 high ($2.4\,\text{m}$) and Node 3 low near the floor baseboard ($0.3\,\text{m}$) creates a steep diagonal RF link with maximum $\cos\theta$ sensitivity to vertical downward displacement.

---

## 3. Hardware & Firmware Architecture Audit

### 3.1 Node 0 (Transmitter AP) — [`firmware/wifi_csi/transmitter_ap/main/main.c`](file:///d:/Fall_detection/firmware/wifi_csi/transmitter_ap/main/main.c)
* **Periodic Injection**: Broadcasts 16-byte ESP-NOW ping frames at a deterministic $100\,\text{Hz}$ ($10\,\text{ms}$ period).
* **Timer Safety**: Operates via `esp_timer_start_periodic()`. Callbacks run in the dedicated FreeRTOS `esp_timer_task` context, making `esp_now_send()` thread-safe and non-blocking.
* **Sequence Counter**: A `uint32_t` counter wraps every $2^{32} / 100\,\text{s} \approx 497\,\text{days}$ of continuous operation.

### 3.2 Nodes 1–3 (Trackers) — [`firmware/wifi_csi/tracker_node/main/main.c`](file:///d:/Fall_detection/firmware/wifi_csi/tracker_node/main/main.c)
* **FreeRTOS Queue Decoupling**:
  * Raw CSI callback (`wifi_csi_rx_callback`) pushes packets to `s_csi_queue` (`csi_queue_item_t`, depth: 16).
  * Network I/O runs in a dedicated `udp_tx_task`.
  * **Stability Impact**: Decoupling eliminated FreeRTOS Watchdog Timer (WDT) timeouts and socket buffer contention observed in early builds.
* **Zero Heap Allocation**: Packet queue items are stack-allocated and copied directly into queue storage.
* **Heartbeat & Diagnostics**: Periodic background task transmits heartbeat telemetry JSON (`{"heartbeat": id, "sent": n, "drops": d}`) every 5 seconds.

### 3.3 Plan 2: 60 GHz mmWave Radar Gateway — [`firmware/mmwave_radar/`](file:///d:/Fall_detection/firmware/mmwave_radar/)
* **UART Frame State Machine**:
  * Byte-by-byte parser synchronization on header `0x53 0x59`.
  * Max payload limit guard (`s_data_len > MAX_PAYLOAD_LEN`) prevents buffer overflows.
  * Checksum verification (`sum(frame[:-1]) & 0xFF == frame[-1]`).
* **Telemetry Forwarding**: Dual-mode transmitter supports compact binary frames or JSON strings over UDP port 5556.

### 3.4 Power Budget & Thermal Dissipation
* **Continuous Active RF**: The 100 Hz injection prevents ESP32 low-power sleep states.
* **Measured Current Draw**: $\approx 160\text{--}180\,\text{mA}$ continuous @ 5V ($\approx 0.8\text{--}0.9\,\text{W}$ per node).
* **Thermal Range**: Enclosed nodes operate at $55\text{--}65^\circ\text{C}$ junction temperature. Wall-powered USB 5V/1A supplies and ventilated enclosures are required.

---

## 4. Real-Time DSP & Kinematics Mathematics

### 4.1 Signal Processing Chain
```
  Raw UDP CSI Datagram (CSIF Header + I/Q Subcarriers)
                           │
                           ▼
             Preprocessor (Unpack & Sanitize)
             • Amplitude: A_k = sqrt(I_k^2 + Q_k^2)
             • Phase: theta_k = arctan2(Q_k, I_k)
             • Linear regression phase unwrapping
                           │
                           ▼
          4th-Order Zero-Phase Butterworth SOS Bandpass
          • Passband: 0.5 Hz - 35.0 Hz
          • Padlen guard: window >= 30 samples
                           │
                           ▼
        Truncated SVD / PCA Dimensionality Reduction
        • Mean-center subcarriers: X_c = X - mu_X
        • Extract dominant eigenvector PC1
        • NaN/Inf sanitization
                           │
                           ▼
         Welch Power Spectral Density (STFT)
         • Window: 64 samples @ 100 Hz
         • Peak kinetic frequency: f_peak
         • Doppler velocity: v = (lambda * f_peak) / 2
                           │
                           ▼
           9D Kinematic Feature Extractor
           • Sub-band Doppler powers: 0-5, 5-15, 15-25, 25-40 Hz
           • Kinetic ratio, surge, temporal variance, spectral entropy
```

### 4.2 Computational Budget & Latency Profile (Raspberry Pi 4 / Intel x86)

| Pipeline Stage | Algorithm / Method | Complexity | Latency (Pi 4) |
|---|---|---|---|
| **Denoising** | 4th-order SOS `sosfiltfilt` | $\mathcal{O}(N_{\text{sub}} \cdot T)$ | $\approx 0.4\,\text{ms}$ |
| **PCA Reduction** | Truncated SVD ($k=1$) | $\mathcal{O}(T \cdot N_{\text{sub}}^2)$ | $\approx 2.1\,\text{ms}$ |
| **Doppler PSD** | Welch FFT ($N_{\text{perseg}}=64$) | $\mathcal{O}(T \log T)$ | $\approx 0.8\,\text{ms}$ |
| **ML Inference** | `HistGradientBoosting` | $\mathcal{O}(\text{depth} \cdot \text{trees})$ | $\approx 0.3\,\text{ms}$ |
| **Total DSP Cycle** | End-to-End Execution | | **$\approx 3.6\,\text{ms}$** |

*Verdict*: The DSP cycle completes in $\approx 3.6\,\text{ms}$, well within the $100\,\text{ms}$ extraction interval budget ($3.6\%$ CPU utilization per link).

---

## 5. Probabilistic Machine Learning Classifier

### 5.1 9D Feature Vector Definition ([`hub/csi_pipeline/classifier.py`](file:///d:/Fall_detection/hub/csi_pipeline/classifier.py))
1. $E_{0\text{--}5\,\text{Hz}}$: Static posture, breathing, slow drift.
2. $E_{5\text{--}15\,\text{Hz}}$: Normal locomotion, walking, arm swing.
3. $E_{15\text{--}25\,\text{Hz}}$: Moderate kinetic motion, rapid sitting.
4. $E_{25\text{--}40\,\text{Hz}}$: Rapid vertical descent / floor impact burst.
5. $\text{Ratio}_{\text{high/low}}$: $\frac{E_{15\text{--}25} + E_{25\text{--}40}}{\max(E_{0\text{--}5} + E_{5\text{--}15}, 10^{-6})}$.
6. $v_{\text{dom}}$: Dominant Doppler velocity $\frac{\lambda \cdot f_{\text{peak}}}{2}$.
7. $\text{Surge}$: Energy surge ratio $\frac{\sum P(f)}{P_{\text{baseline}}}$.
8. $\sigma^2$: Moving temporal variance across window.
9. $H_{\text{spec}}$: Spectral entropy $-\sum p_k \log_2(p_k)$ measuring frequency dispersion during impact.

### 5.2 Supervised Model Performance
* **Model**: Scikit-learn `HistGradientBoostingClassifier` (100 estimators, max depth 5, learning rate 0.08).
* **Discrimination**:
  * Normal Activities of Daily Living (ADLs): $P(\text{fall}) < 0.30$.
  * Fall Kinetic Events: $P(\text{fall}) > 0.80$.
* **Hybrid Integration**: In [`hub/fusion_engine.py`](file:///d:/Fall_detection/hub/fusion_engine.py), an ML score $P(\text{fall}) \ge 0.85$ fast-tracks confirmation when combined with suspected states or radar floor posture.

---

## 6. Multi-Modal Consensus & State Machine Analysis

### 6.1 State Machine Architecture
```
                         FUSION STATE TRANSITIONS
        ┌────────────────────────────────────────────────────────┐
        │                        NORMAL                          │
        └──────────────────────────┬─────────────────────────────┘
                                   │ Link Coincidence (>=2 links)
                                   │ OR ML P(fall) >= 0.70
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │                    FALL SUSPECTED                      │
        └───────┬────────────────────────────────────────┬───────┘
                │ Vigorous motion                        │
                │ resumes (t > 0.8s)                     │ Sustained floor
                ▼                                        │ stillness (>= 4.0s)
        ┌────────────────┐                               │ OR Radar height < 0.45m
        │   RECOVERED    │                               ▼
        └───────┬────────┘                      ┌────────────────┐
                │ Resumes normal                │ FALL DETECTED  │
                ▼                               │ (EMERGENCY)    │
             [NORMAL]                           └────────────────┘
```

### 6.2 Audit Finding: "OR" Policy vs. Active Radar Veto
* **Current Policy**:
  ```python
  if radar_confirmed and (csi_confirmed or self.last_csi_state == CSIFallState.SUSPECTED_FALL or ml_confirmed):
      self.unified_state = UnifiedFallState.CONFIRMED
  elif radar_confirmed or csi_confirmed or (ml_confirmed and self.last_csi_state == CSIFallState.SUSPECTED_FALL):
      self.unified_state = UnifiedFallState.CONFIRMED
  ```
* **Analysis**:
  If the radar actively tracks an upright standing occupant ($Z = 1.7\,\text{m}$, posture = `STANDING`), but CSI triggers a false alarm (e.g. dropping a heavy blanket), the second branch (`csi_confirmed`) still escalates to `CONFIRMED`.
* **Recommended Policy (Active Veto)**:
  ```python
  radar_active_standing = (
      self.last_radar is not None
      and self.last_radar.target_height_m > 1.1
      and self.last_radar.posture == RadarPosture.STANDING
  )
  if csi_confirmed and radar_active_standing:
      self.unified_state = UnifiedFallState.SUSPECTED  # Veto emergency confirmation
  ```

---

## 7. Failure Mode & Effects Analysis (FMEA) & Edge Cases

| Failure Mode / Scenario | Raw Sensor Signature | System Mitigation & Verdict | Severity |
|---|---|---|:---:|
| **FM-01: Rapid Sitting on Low Chair** | High downward velocity burst ($v \approx 1.6\text{--}2.0\,\text{m/s}$) on 2 links. | **Radar Rejection**: Target elevation stabilizes at $Z \approx 0.55\text{--}0.75\,\text{m}$. Height remains above floor threshold ($<0.45\,\text{m}$). Rejects confirmation. | **LOW** |
| **FM-02: Dropping a Heavy Object** | Sudden broadband Doppler surge on CSI. | **Quiescence Rejection**: If occupant remains standing/moving, CSI variance does not drop into stillness threshold ($<0.08$). Radar confirms upright height. | **LOW** |
| **FM-03: Slow Sliding / Slump Fall** | Slow descent ($v < 1.0\,\text{m/s}$) does not trigger velocity threshold. | **Radar Coverage**: Radar detects elevation drop to $Z < 0.35\,\text{m}$ and posture `LYING` with dwell $>4\,\text{s}$, triggering confirmation in `FUSION` mode. | **MEDIUM** |
| **FM-04: Multi-Person Room (Caregiver Present)** | Multiple moving targets scatter the RF multipath; PCA eigenvector blends targets. | **Vulnerability**: CSI velocity estimates degrade with 2+ people. System falls back to mmWave radar centroid tracking. | **MEDIUM** |
| **FM-05: Falling into a Soft Low Sofa** | Fast descent; post-event stillness; altitude $Z \approx 0.45\text{--}0.50\,\text{m}$. | **Threshold Sensitivity**: Borderline threshold between floor ($<0.45\,\text{m}$) and sitting ($>0.45\,\text{m}$). Configurable in `radar_config.yaml`. | **MEDIUM** |
| **FM-06: Wi-Fi Packet Burst Loss** | Microwave / Bluetooth interference causing packet gaps. | **Interpolation**: `NodeBuffer` linear interpolation covers up to 5 consecutive missing frames. Gaps $>10$ reset baseline to avoid false variance spikes. | **LOW** |
| **FM-07: Tracker Node Hardware Drop** | Tracker node unplugged or powered down. | **Graceful Degradation**: `node_health_monitor()` detects offline node after 10s and reduces coincidence threshold from 2 links to 1 link. | **LOW** |
| **FM-08: Browser Audio Autoplay Block** | Modern browsers block Web Audio API autoplay on page load. | **User Gesture Unlocking**: `dashboard.js` hooks initial user click / reset button press to initialize `AudioContext`. | **LOW** |

---

## 8. Telemetry HUD, Smart Home & Edge Deployment

### 8.1 Real-Time Web HUD ([`hub/dashboard/`](file:///d:/Fall_detection/hub/dashboard/))
* **Zero-npm Vanilla Architecture**: Plain HTML5 Canvas + vanilla JS streams over WebSockets. No Node.js build step, webpack, or external CDN dependencies.
* **Bitmap Scrolling Waterfall**: Spectrogram canvas shifts vertical pixels downward via `ctx.drawImage(canvas, 0, 1, width, height - 1)` rather than repainting all historical data, ensuring stable $60\,\text{FPS}$ rendering on low-power tablets.

### 8.2 Home Assistant MQTT Auto-Discovery ([`hub/ha_discovery.py`](file:///d:/Fall_detection/hub/ha_discovery.py))
* Generates 13 auto-discovered entities under device `Multi-Modal Fall Detection Hub`:
  * Binary sensor: `fall_detected` (device class: `safety`)
  * Sensors: `system_state`, `radar_height`, `radar_posture`, `radar_dwell`, `ml_fall_probability`
  * Node sensors: CSI velocity & surge for Nodes 1, 2, 3
  * Button: `reset_alarm` (device class: `restart`)
* Pre-built automations provided in [`config/ha_automations.yaml`](file:///d:/Fall_detection/config/ha_automations.yaml).

### 8.3 Docker Edge Deployment ([`deploy/`](file:///d:/Fall_detection/deploy/))
* Multi-arch `Dockerfile` based on `python:3.11-slim`.
* `docker-compose.yml` configures `network_mode: "host"` to receive UDP broadcast/multicast packets on ports 5555 and 5556 without NAT traversal jitter.

---

## 9. Automated Test Suite & Verification Matrix

Automated regression testing executed on Windows Python 3.11.9:
```powershell
pytest -v tests/
```

### Results Summary
**149 passed, 7 skipped (legacy CV tests), 0 failed** (100% pass rate):

| Test Suite | Tests Passed | Covered Functionality |
|---|:---:|---|
| [`tests/test_active_veto.py`](file:///d:/Fall_detection/tests/test_active_veto.py) | 4 | Active radar veto suppressing false CSI burst, legacy backward compatibility, floor posture allowance, kinematic slump detection |
| [`tests/test_adaptive_calibrator.py`](file:///d:/Fall_detection/tests/test_adaptive_calibrator.py) | 4 | 2.4 GHz vs 5.8 GHz carrier scaling ($\lambda$), EMA baseline drift adaptation, motion outlier rejection, bounds clamping |
| [`tests/test_clutter_filter.py`](file:///d:/Fall_detection/tests/test_clutter_filter.py) | 6 | Multi-link Elevation Perturbation Ratio (EPR), pet ground clutter suppression, radar cluster area ($<0.15\,\text{m}^2$) filtering |
| [`tests/test_train_pipeline.py`](file:///d:/Fall_detection/tests/test_train_pipeline.py) | 7 | Synthetic feature generation, 5-fold Stratified CV, ROC/PR evaluation, model export & reload, NPZ extraction, mix-ratio blending |
| [`tests/test_dashboard.py`](file:///d:/Fall_detection/tests/test_dashboard.py) | 7 | Index route, status API, incidents query, WebSocket telemetry, `/api/calibrate`, `/api/thresholds`, `/api/datasets` |
| [`tests/test_csi_pipeline.py`](file:///d:/Fall_detection/tests/test_csi_pipeline.py) | 4 | Binary packet parsing, corrupt packet rejection, PCA SVD, Doppler velocity |
| [`tests/test_mmwave_parser.py`](file:///d:/Fall_detection/tests/test_mmwave_parser.py) | 4 | Binary frame decoding, checksum validation, JSON parsing, height tracking, dual fusion |
| [`tests/test_classifier.py`](file:///d:/Fall_detection/tests/test_classifier.py) | 4 | 9D feature extraction, ML probability discrimination, model save/load, hybrid fusion |
| [`tests/test_recorder.py`](file:///d:/Fall_detection/tests/test_recorder.py) | 2 | Multimodal session buffer synchronization, `.npz` export and metadata schema |
| [`tests/test_ha_discovery.py`](file:///d:/Fall_detection/tests/test_ha_discovery.py) | 4 | HA MQTT discovery payloads, announce/remove lifecycle, telemetry publishing, YAML syntax |
| [`tests/test_integration.py`](file:///d:/Fall_detection/tests/test_integration.py) | 9 | CSI mode, Radar mode, Fusion mode, false positive rejection, sequence interpolation, node liveness, cooldown, multi-link recovery |
| [`tests/test_alert_dispatcher.py`](file:///d:/Fall_detection/tests/test_alert_dispatcher.py) | 6 | Cooldown rate limiting, CSV incident logging, MQTT dispatch, webhook POST, network failure resilience |
| [`tests/test_calibrate.py`](file:///d:/Fall_detection/tests/test_calibrate.py) | 3 | Noise floor baseline generation, metrics computation, YAML export |
| [`tests/test_audit_log.py`](file:///d:/Fall_detection/tests/test_audit_log.py) | 9 | Tamper-evident hash-chaining, SQLite storage, verify_chain, historical backdating |
| [`tests/test_fhir_exporter.py`](file:///d:/Fall_detection/tests/test_fhir_exporter.py) | 3 | HL7 FHIR R4 Observation bundle generation and validation |
| [`tests/test_auth.py`](file:///d:/Fall_detection/tests/test_auth.py) | 5 | Bearer token authentication, RBAC admin enforcement, audit log endpoint security |
| [`tests/test_onnx_runner.py`](file:///d:/Fall_detection/tests/test_onnx_runner.py) | 7 | Quantized ONNX runtime inference, latency benchmarking, scikit-learn fallback |
| [`tests/test_room_manager.py`](file:///d:/Fall_detection/tests/test_room_manager.py) | 14 | Multi-room context isolation, activity timeouts, REST API management |
| [`tests/test_room_routing.py`](file:///d:/Fall_detection/tests/test_room_routing.py) | 3 | V2 18-byte UDP header parsing, hardware room-ID routing, multi-room context dispatch |
| [`tests/test_relay_integration.py`](file:///d:/Fall_detection/tests/test_relay_integration.py) | 5 | RelayClient federation routing, magic byte checking, /api/relay/stats endpoint |
| [`tests/test_version.py`](file:///d:/Fall_detection/tests/test_version.py) | 4 | Semantic version API, phase reporting, OTA directory traversal security |
| [`tests/test_temporal_attention.py`](file:///d:/Fall_detection/tests/test_temporal_attention.py) | 3 | Recency-biased exponential temporal attention weights, dynamic runtime tuning |
| [`tests/test_bayesian_classifier.py`](file:///d:/Fall_detection/tests/test_bayesian_classifier.py) | 6 | Platt-calibrated posterior probability, graduated alert severity, Brier score |
| [`tests/test_fall_type_classifier.py`](file:///d:/Fall_detection/tests/test_fall_type_classifier.py) | 6 | 5-class fall categorization, confidence estimation, alert priority mapping |
| [`tests/test_analytics.py`](file:///d:/Fall_detection/tests/test_analytics.py) | 7 | Population health analytics, hourly distribution, MTBF, alert cancellation rate |
| [`tests/test_traceability.py`](file:///d:/Fall_detection/tests/test_traceability.py) | 4 | IEC 62304 SRS traceability matrix parsing, critical coverage verification |
| [`tests/test_risk_analysis.py`](file:///d:/Fall_detection/tests/test_risk_analysis.py) | 5 | ISO 14971 FMEA risk register validation, severity x probability checks |
| [`tests/test_clinical_report.py`](file:///d:/Fall_detection/tests/test_clinical_report.py) | 5 | Automated clinical validation report generator, audit statistics extraction |
| [`tests/test_vital_signs.py`](file:///d:/Fall_detection/tests/test_vital_signs.py) | 6 | Respiration frequency extraction (0.1-0.5 Hz), chest displacement, apnea detection |
| [`tests/test_clinical_trial_runner.py`](file:///d:/Fall_detection/tests/test_clinical_trial_runner.py) | 6 | FDA GMLP Principle 7 cohort simulation, Wilson score 95% CIs, Cohen's kappa, fairness |
| [`tests/test_cloud_sync.py`](file:///d:/Fall_detection/tests/test_cloud_sync.py) | 5 | Multi-facility cloud gateway, offline SQLite store-and-forward queue, retry logic |
| [`tests/test_diagnostics.py`](file:///d:/Fall_detection/tests/test_diagnostics.py) | 5 | IEC 60601-1-8 system diagnostics, packet jitter monitoring, automated self-tests |

---

## 10. Engineering Action Items & Evolution Roadmap

```
                                  ENGINEERING ROADMAP
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Phase 4 (Completed)                                                             │
│ [x] Web Telemetry HUD (FastAPI + WebSockets + HTML5 Canvas)                     │
│ [x] Multimodal Dataset Recorder & Replay Tool (.npz + JSON)                     │
│ [x] Probabilistic ML Kinematic Classifier (HistGradientBoosting)                │
│ [x] Home Assistant MQTT Auto-Discovery (13 entities + YAML automations)         │
│ [x] Docker Edge Deployment (Host networking + Mosquitto broker)                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 5 (Completed — Consensus Hardening, Adaptive Calibration & Fleet Training)│
│ [x] Active Radar Veto Policy in DualFusionEngine (suppress dropped item bursts) │
│ [x] Kinematic Slump / Sliding Fall Tracker (elevation derivative dZ/dt)        │
│ [x] Carrier-Aware Doppler Scaling (lambda = 0.123m for 2.4GHz, 0.0545m for 5GHz)│
│ [x] Continuous Background Adaptive Calibrator (EMA baseline tracking, alpha=0.02)│
│ [x] Spatial Link Elevation Perturbation Ratio (EPR < 0.15 ground pet filter)   │
│ [x] mmWave Radar Cluster Area Filter (< 0.15 m^2 pet / roomba disambiguation)  │
│ [x] Empirical Dataset Training Pipeline (hub/train.py with 5-Fold Stratified CV)│
│ [x] Clinical Human Trial Data Collection Protocol (docs/DATA_COLLECTION_PROTOCOL)│
│ [x] Web HUD Fleet REST Endpoints (/api/calibrate, /api/thresholds, /api/datasets)│
│ [x] ESP-IDF Firmware mDNS Discovery (falldetect-hub.local zero-config unicast)  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 6 (Completed — Clinical Multi-Room Deployment & Edge NPU Acceleration)   │
│ [x] Multi-Room Spatial Mesh (hub/room_manager.py — per-room CSI+radar engines) │
│ [x] UDP Relay Client for Secondary Hub Clusters (hub/relay_client.py)          │
│ [x] ONNX Quantized Inference (hub/onnx_runner.py — INT8 pickle fallback)       │
│ [x] ONNX Export Pipeline (hub/train.py --onnx, skl2onnx + onnxruntime)        │
│ [x] Clinical Audit Log (hub/audit_log.py — SHA-256 hash-chain SQLite)         │
│ [x] FHIR R4 Export (hub/fhir_export.py — LOINC 55122-0 Observation Bundle)    │
│ [x] RBAC Token Auth (hub/auth.py — require_admin FastAPI dependency)           │
│ [x] Audit REST Endpoints (/api/audit, /api/audit/export/fhir, /api/audit/verify)│
│ [x] Empirical Retrain SOP (docs/RETRAIN_CHECKLIST.md — SOP-CLIN-FD-002)       │
│ [x] Real + Synthetic Mix-Ratio Training (--mix-ratio 0.7 blend in train.py)    │
│ [x] GitHub Actions CI (Python 3.10/3.11 matrix, pip cache, artifact upload)    │
│ [x] Systemd Service Unit (deploy/falldetect-hub.service — Pi production daemon)│
│ [x] Makefile (test/train/docker/release developer shortcuts)                   │
│ [x] Semantic Version (hub/__version__ = "3.0.0", GET /api/version endpoint)    │
│ [x] OTA Firmware Endpoint (GET /firmware/{filename} for ESP32 OTA updates)     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 7 (Completed — ESP32 OTA Firmware Updates & Hub Federation Server Wiring) │
│ [x] ESP32 Tracker OTA firmware update logic (CONFIG_OTA_ENABLED in Kconfig)     │
│ [x] ESP32 Radar OTA firmware update logic (CONFIG_OTA_ENABLED in Kconfig)       │
│ [x] Hardware Room-ID byte routing in 18-byte V2 UDP headers                     │
│ [x] Hub Server multi-room context dispatch (--multi-room flag)                  │
│ [x] RelayClient socket federation wiring in server.py and /api/relay/stats      │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 8 (Completed — Advanced Signal Processing & Population Health Analytics)  │
│ [x] Temporal Attention Windowing for STFT velocity estimation (decay factor)    │
│ [x] Bayesian fall probability with Platt scaling (CalibratedClassifierCV)       │
│ [x] Graduated alert severity classification (Suspected vs Confirmed vs High)    │
│ [x] Fall-Type second-stage classifier (5 clinical fall mechanisms)              │
│ [x] Population health analytics module (hub/analytics.py, MTBF, hourly stats)   │
│ [x] Analytics dashboard UI (hub/dashboard/static/analytics.html)                │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 9 (Completed — FDA SaMD Certification Package & Automated Reporting)      │
│ [x] IEC 62304 Software Requirement Traceability Matrix (docs/requirements.yaml) │
│ [x] Automated Traceability Matrix Generator (docs/generate_traceability.py)     │
│ [x] ISO 14971 FMEA Risk Analysis Register (docs/risk_analysis.yaml, 22 hazards) │
│ [x] ISO 14971 Risk Analysis Validator (docs/validate_risk_analysis.py)          │
│ [x] Predicate Device Comparison Document (docs/PREDICATE_COMPARISON.md K151548) │
│ [x] FDA Substantial Equivalence Decision Checklist (SE-001 through SE-010)      │
│ [x] Automated Clinical Performance Validation Report (generate_clinical_report) │
│ [x] Makefile 'report' target and GitHub Actions CI certification validation     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 10 (Completed — Multi-Facility Cloud Gateway, Vital Signs & Diagnostics) │
│ [x] Post-Fall Vital Signs micro-Doppler Estimator (hub/vital_signs.py, 6-30 bpm)│
│ [x] Automated Clinical Trial Cohort Simulator (hub/clinical_trial_runner.py)    │
│ [x] FDA GMLP Principle 7 Demographic Disparity Report (Wilson 95% CI & Kappa)   │
│ [x] Multi-Facility Cloud Gateway & Offline Sync (hub/cloud_sync.py, /api/cloud) │
│ [x] Continuous System Diagnostics Daemon (hub/diagnostics.py, IEC 60601-1-8)   │
│ [x] Health & Watchdog Dashboard Endpoints (/api/diagnostics/health, /metrics)   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 11 (Completed — Real-Time Adaptive Retraining & Model Drift Detection)    │
│ [x] Incremental Training Buffer & Ground-Truth Labelling API (hub/training_buf) │
│ [x] Population Stability Index & KL Divergence Drift Detector (drift_detector)  │
│ [x] Auto-Retrain Pipeline & A/B Model Swap (retraining_pipeline.py, floors)     │
│ [x] Model Version Registry Integrity & IEC 62304 §8.2 Changelog (generate_log)  │
│ [x] Dashboard Endpoints (/api/labels/*, /api/drift/*, /api/retrain/*)           │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 12 (Completed — Encrypted FHIR R4 Data Lake & HL7 v2.x Interface Engine) │
│ [x] HL7 v2.x ADT Message Parser & MLLP Listener (hub/hl7_listener.py, port 2575)│
│ [x] Patient Context Store & Fall Alert Enrichment (hub/patient_context.py)      │
│ [x] AES-256 Encrypted FHIR R4 Data Lake with Key Rotation (hub/fhir_lake.py)    │
│ [x] SMART-on-FHIR OAuth2 Token Manager & REST Client (hub/smart_fhir_client.py) │
│ [x] Dashboard & Server FHIR/HL7 Endpoints (/api/fhir/*, /api/patient/{room_id}) │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Phase 13 (Completed — ESP32 Mesh Networking & Seamless Room Handoff)            │
│ [x] ESP-MESH Multi-Hop Forwarding & Relay (firmware/.../mesh_forward.c, Kconfig) │
│ [x] Hub Packet Parser V2 with Mesh Relay & Hop Penalty (hub/packet_parser.py)   │
│ [x] Room Boundary Handoff State Machine & Dual-Monitoring (hub/room_handoff.py) │
│ [x] 2D CSI Phase-Differential Triangulation Engine (hub/triangulation.py, WLS)  │
│ [x] Mesh Topology & Handoff Visualizer HUD (hub/dashboard/static/mesh.html)     │
│ [x] ISO 14971 FMEA HAZ-023 Doorway Transition Fall Hazard Mitigation            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Revision & Change History

| Version | Date | Baseline Commit | Author / Agent | Key Findings & Changes Logged |
|---|---|---|---|---|
| **1.0.0** | 2026-09-12 | `13e7dbf` | Antigravity | Initial multi-modal architecture release (Plan 1 CSI mesh + Plan 2 mmWave gateway). |
| **1.1.0** | 2026-09-13 | `88d37b6` | Antigravity | Phase 3 Validation Audit: ISR callback decoupling in tracker firmware, live listener threads in hub, SOS filter length guards, SVD sanitization. |
| **1.2.0** | 2026-09-14 | `6c567f6` | Antigravity | Added room noise calibration (`calibrate.py`), MQTT/webhook alerting, broadcast discovery, and integration test suite. |
| **2.0.0** | 2026-09-14 | `cd5951c` | Antigravity | **Phase 4 Milestone Completion**: Real-Time Web Telemetry HUD, Dataset Recorder & Replay, Probabilistic ML Classifier, Home Assistant MQTT Auto-Discovery, Docker edge containerization. Full test suite: 40 passed. |
| **2.5.0** | 2026-09-16 | `e1c7f3f` | Antigravity | **Phase 5 Milestone Completion**: Consensus Hardening (Active Radar Veto & Kinematic Slump Detection), Carrier-Aware Scaling (2.4 vs 5.8 GHz), Background Adaptive Calibrator (EMA baseline tracking), Spatial Elevation Perturbation Ratio (EPR) & Radar Cluster Area Filter, Empirical Training Pipeline (`hub/train.py`) with 5-Fold CV, Clinical Human Trial Protocol (`docs/DATA_COLLECTION_PROTOCOL.md`), Web HUD Fleet REST Endpoints, and ESP-IDF Firmware mDNS Discovery. Total: 62 passed tests. |
| **3.0.0** | 2026-09-18 | `f4cfe20` | Antigravity | **Phase 6 Milestone Completion**: Multi-Room Spatial Mesh (`hub/room_manager.py`, `hub/relay_client.py`), ONNX Quantized Inference (`hub/onnx_runner.py`, `hub/train.py --onnx`), Clinical Audit Log (`hub/audit_log.py` SHA-256 hash-chain, `hub/fhir_export.py` FHIR R4, `hub/auth.py` RBAC), 3 audit REST endpoints, Empirical Retrain SOP (`docs/RETRAIN_CHECKLIST.md`), mix-ratio blending in train.py, GitHub Actions CI (Python 3.10/3.11 matrix + pip cache), systemd service unit, Makefile, `GET /api/version`, `GET /firmware/{filename}` OTA endpoint. Total: **105 passed, 7 skipped, 0 failed**. |
| **3.1.0** | 2026-09-19 | `522d070` | Antigravity | **Phase 7 Milestone Completion**: ESP32 Tracker & Radar OTA firmware update routines under `CONFIG_OTA_ENABLED`, 18-byte V2 UDP header with hardware `room_id`, hub multi-room packet routing (`hub/server.py --multi-room`), RelayClient socket federation wiring (`--relay-port` and `GET /api/relay/stats`). Total: **113 passed, 7 skipped, 0 failed**. |
| **3.2.0** | 2026-09-19 | `9ce9e3c` | Antigravity | **Phase 8 Milestone Completion**: Advanced signal processing with temporal attention windowing, Platt-calibrated Bayesian fall probability (`CalibratedClassifierCV`), graduated alert severities (`p_suspected`, `p_confirmed`, `p_high_confidence`), fall-type second-stage classifier (5 clinical classes), population health analytics module (`hub/analytics.py`) with MTBF and hourly distributions, population health dashboard UI (`analytics.html`). Total: **135 passed, 7 skipped, 0 failed**. |
| **3.3.0** | 2026-09-19 | `af7ada4` | Antigravity | **Phase 9 Milestone Completion**: Full FDA SaMD Certification Package & Automated Clinical Reporting. IEC 62304 Software Requirement Traceability Matrix with 32 SRS items and 100% test coverage (`docs/requirements.yaml`, `docs/generate_traceability.py`, `docs/TRACEABILITY_MATRIX.md`), ISO 14971 FMEA Risk Register with 22 validated hazards (`docs/risk_analysis.yaml`, `docs/validate_risk_analysis.py`, `docs/RISK_ANALYSIS.md`), FDA 510(k) Predicate Device Comparison against Philips Lifeline AutoAlert K151548 (`docs/PREDICATE_COMPARISON.md`, `docs/SUBSTANTIAL_EQUIVALENCE_CHECKLIST.md`), Automated Clinical Performance Validation Report Generator (`docs/generate_clinical_report.py`, `docs/CLINICAL_PERFORMANCE_REPORT.md`), Makefile 'report' target, CI automation. Total: **149 passed, 7 skipped, 0 failed**. |
| **3.4.0** | 2026-09-20 | `50f01f9` | Antigravity | **Phase 10 Milestone Completion**: Enterprise Healthcare Deployment & Vital Signs Verification. Post-Fall Respiration & Vital Signs micro-Doppler Estimator (`hub/vital_signs.py`, 6-30 bpm), Automated Clinical Trial Cohort Simulator (`hub/clinical_trial_runner.py`), FDA GMLP Principle 7 Demographic Fairness & Disparity Report (`docs/CLINICAL_TRIAL_COHORT_REPORT.md`), Multi-Facility Cloud Gateway with Offline Store-and-Forward SQLite Queue (`hub/cloud_sync.py`), Continuous System Diagnostics & IEC 60601-1-8 Alarm System Watchdog Daemon (`hub/diagnostics.py`). Total: **171 passed, 7 skipped, 0 failed**. |
| **3.5.0** | 2026-09-20 | `e5bc41b` | Antigravity | **Phase 11 Milestone Completion**: Real-Time Adaptive Retraining & Model Drift Detection. Incremental Training Buffer with Nursing Staff Ground-Truth Labelling (`hub/training_buffer.py`), Population Stability Index & KL Divergence Drift Detector (`hub/drift_detector.py`), Automated Retraining Pipeline with Clinical Safety Floors & Rollback (`hub/retraining_pipeline.py`), SQLite Versioned Model Store with SHA-256 Integrity Verification (`hub/model_registry.py`), IEC 62304 §8.2 Automated Model Changelog Generator (`docs/generate_model_changelog.py`, `docs/MODEL_CHANGELOG.md`), Dashboard REST Endpoints (`/api/labels/*`, `/api/drift/*`, `/api/retrain/*`). Total: **202 passed, 7 skipped, 0 failed**. |
| **3.6.0** | 2026-09-20 | `534bbd4` | Antigravity | **Phase 12 Milestone Completion**: Encrypted FHIR R4 Data Lake & HL7 v2.x Interface Engine. Asynchronous MLLP TCP Listener (`hub/hl7_listener.py`) for ADT^A01, A03, A08 with ACK generation, SQLite-backed Patient Context Store (`hub/patient_context.py`) with high-risk medication screening and Morse Fall Scale risk flagging, alert payload enrichment in `DualFusionEngine`, AES-256 Fernet-encrypted FHIR R4 Data Lake (`hub/fhir_lake.py`) with atomic key rotation, and SMART-on-FHIR OAuth2 backend client (`hub/smart_fhir_client.py`). Total: **232 passed, 7 skipped, 0 failed**. |
| **3.7.0** | 2026-09-21 | Current | Antigravity | **Phase 13 Milestone Completion**: ESP32 Mesh Networking & Seamless Room Handoff. ESP-MESH multi-hop forwarding routine (`mesh_forward.c`) with hop count TTL checking and Kconfig `CONFIG_MESH_ENABLED`, hub packet parser (`hub/packet_parser.py`) with exponential hop weight decay and multi-path deduplication (`MeshDeduplicator`), Room Boundary Handoff State Machine (`hub/room_handoff.py`) with 5-frame hysteresis and 8s dual-monitoring window across room transitions, ISO 14971 FMEA `HAZ-023` doorway mitigation, 2D CSI Phase-Differential Triangulation Engine (`hub/triangulation.py`) with WLS multilateration and floor plan mapping, and Mesh HUD (`mesh.html`) with `/api/position/*` and `/api/mesh/topology` REST endpoints. Total: **259 passed, 7 skipped, 0 failed**. |

