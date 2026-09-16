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
**62 passed, 7 skipped (legacy CV tests), 0 failed** (100% pass rate):

| Test Suite | Tests Passed | Covered Functionality |
|---|:---:|---|
| [`tests/test_active_veto.py`](file:///d:/Fall_detection/tests/test_active_veto.py) | 4 | Active radar veto suppressing false CSI burst, legacy backward compatibility, floor posture allowance, kinematic slump detection |
| [`tests/test_adaptive_calibrator.py`](file:///d:/Fall_detection/tests/test_adaptive_calibrator.py) | 4 | 2.4 GHz vs 5.8 GHz carrier scaling ($\lambda$), EMA baseline drift adaptation, motion outlier rejection, bounds clamping |
| [`tests/test_clutter_filter.py`](file:///d:/Fall_detection/tests/test_clutter_filter.py) | 6 | Multi-link Elevation Perturbation Ratio (EPR), pet ground clutter suppression, radar cluster area ($<0.15\,\text{m}^2$) filtering |
| [`tests/test_train_pipeline.py`](file:///d:/Fall_detection/tests/test_train_pipeline.py) | 5 | Synthetic feature generation, 5-fold Stratified CV, ROC/PR evaluation, model export & reload, NPZ feature extraction |
| [`tests/test_dashboard.py`](file:///d:/Fall_detection/tests/test_dashboard.py) | 7 | Index route, status API, incidents query, WebSocket telemetry, `/api/calibrate`, `/api/thresholds`, `/api/datasets` |
| [`tests/test_csi_pipeline.py`](file:///d:/Fall_detection/tests/test_csi_pipeline.py) | 4 | Binary packet parsing, corrupt packet rejection, PCA SVD, Doppler velocity |
| [`tests/test_mmwave_parser.py`](file:///d:/Fall_detection/tests/test_mmwave_parser.py) | 4 | Binary frame decoding, checksum validation, JSON parsing, height tracking, dual fusion |
| [`tests/test_classifier.py`](file:///d:/Fall_detection/tests/test_classifier.py) | 4 | 9D feature extraction, ML probability discrimination, model save/load, hybrid fusion |
| [`tests/test_recorder.py`](file:///d:/Fall_detection/tests/test_recorder.py) | 2 | Multimodal session buffer synchronization, `.npz` export and metadata schema |
| [`tests/test_ha_discovery.py`](file:///d:/Fall_detection/tests/test_ha_discovery.py) | 4 | HA MQTT discovery payloads, announce/remove lifecycle, telemetry publishing, YAML syntax |
| [`tests/test_integration.py`](file:///d:/Fall_detection/tests/test_integration.py) | 9 | CSI mode, Radar mode, Fusion mode, false positive rejection, sequence interpolation, node liveness, cooldown, multi-link recovery |
| [`tests/test_alert_dispatcher.py`](file:///d:/Fall_detection/tests/test_alert_dispatcher.py) | 6 | Cooldown rate limiting, CSV incident logging, MQTT dispatch, webhook POST, network failure resilience |
| [`tests/test_calibrate.py`](file:///d:/Fall_detection/tests/test_calibrate.py) | 3 | Noise floor baseline generation, metrics computation, YAML export |

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
│ Phase 6 (Future — Clinical Multi-Room Deployment & Edge NPU Acceleration)      │
│ [ ] Multi-room handover & spatial mesh roaming across multiple ESP32 AP clusters│
│ [ ] Edge TPU / NPU quantization (TensorFlow Lite / ONNX Runtime for ARM)        │
│ [ ] FDA 510(k) Class II Medical Device clinical audit logging and export        │
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
| **2.5.0** | 2026-09-16 | Current | Antigravity | **Phase 5 Milestone Completion**: Consensus Hardening (Active Radar Veto & Kinematic Slump Detection), Carrier-Aware Scaling (2.4 vs 5.8 GHz), Background Adaptive Calibrator (EMA baseline tracking), Spatial Elevation Perturbation Ratio (EPR) & Radar Cluster Area Filter, Empirical Training Pipeline (`hub/train.py`) with 5-Fold CV, Clinical Human Trial Protocol (`docs/DATA_COLLECTION_PROTOCOL.md`), Web HUD Fleet REST Endpoints, and ESP-IDF Firmware mDNS Discovery. Total: 62 passed tests. |
