# Multi-Modal Device-Free Fall Detection System

[![CI](https://github.com/ard12/Fall_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ard12/Fall_Detection/actions)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v5.0+-red.svg)](https://docs.espressif.com/projects/esp-idf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](deploy/Dockerfile)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-MQTT%20Discovery-41BDF5.svg)](config/ha_automations.yaml)
[![Audit Log](https://img.shields.io/badge/Audit%20Log-Living%20Document-orange.svg)](SYSTEM_AUDIT_AND_FINDINGS.md)

An intelligent, non-invasive, privacy-preserving fall detection platform engineered for elderly monitoring and healthcare facilities. Operates completely **device-free** (no wearables, pendants, or intrusive cameras) by combining two complementary wireless sensing paradigms:

1. **Plan 1 (Primary)**: **4-Node Wi-Fi CSI Active Sensing Mesh** (1 AP / Transmitter + 3 Receivers / Trackers using ESP-NOW @ 100 Hz).
2. **Plan 2 (High-Reliability)**: **ESP32 + 60 GHz mmWave FMCW Radar** (Direct 3D centroid altitude tracking, cluster area filtering, and posture classification).
3. **Dual-Sensor Fusion Engine**: Cross-verifies wide-area RF multipath disturbances with centimeter-accurate floor height detection, Active Radar Veto (to suppress dropped item bursts), and Kinematic Slump Detection for geriatric sliding falls.
4. **Pet & Ground Clutter Disambiguation**: Multi-link Elevation Perturbation Ratio (EPR) and radar cluster area filtering ($<0.15\,\text{m}^2$) to reject dogs, cats, and robot vacuums.
5. **Continuous Adaptive Calibration & 5 GHz Carrier Scaling**: Real-time EMA background variance tracking and carrier wavelength scaling ($\lambda = 0.123\,\text{m}$ for 2.4 GHz vs $0.0545\,\text{m}$ for 5.8 GHz).
6. **Empirical ML Training Pipeline**: End-to-end training and 5-fold cross-validation suite (`hub/train.py`) with ONNX FP32 + INT8 quantized export (`hub/onnx_runner.py`) and clinical trial protocol (`docs/DATA_COLLECTION_PROTOCOL.md`).
7. **Real-Time Web Telemetry HUD & REST Fleet API**: Zero-npm, canvas-rendered dashboard with live WebSocket spectra, altitude gauges, and REST management endpoints (`/api/calibrate`, `/api/thresholds`, `/api/datasets`, `/api/rooms`, `/api/version`).
8. **Smart Home & Zero-Config Edge Ready**: Native Home Assistant MQTT Auto-Discovery, ESP-IDF mDNS unicast discovery (`falldetect-hub.local`), Docker Compose deployment, and systemd service unit (`deploy/falldetect-hub.service`).
9. **Multi-Room Spatial Mesh**: `RoomManager` (`hub/room_manager.py`) maintains independent CSI + radar engines per room; `RelayClient` (`hub/relay_client.py`) forwards packets from secondary ESP32 clusters.
10. **Clinical Audit Logging (FDA SaMD-ready)**: SHA-256 hash-chained SQLite audit log (`hub/audit_log.py`), FHIR R4 Observation Bundle export (`hub/fhir_export.py`, LOINC 55122-0), RBAC token auth (`hub/auth.py`).
11. **Production Packaging & CI/CD**: GitHub Actions matrix CI (Python 3.10/3.11, pip cache, artifact upload), `Makefile` developer shortcuts (`make test`, `make train`, `make report`, `make release`), OTA firmware delivery (`GET /firmware/{filename}`).
12. **ESP32 OTA Firmware & Hub Federation (v3.1.0)**: Over-The-Air firmware updates for tracker and radar nodes (`CONFIG_OTA_ENABLED`), 18-byte V2 UDP headers with hardware room-ID routing, and RelayClient federation server wiring (`GET /api/relay/stats`).
13. **Advanced Signal Processing & Population Health (v3.2.0)**: Recency-weighted temporal attention windowing for STFT velocity estimation, Platt-calibrated Bayesian fall probabilities (`CalibratedClassifierCV`), graduated alert severities, 5-class second-stage fall-type classifier, and population health analytics (`/api/analytics/*`, `analytics.html`).
14. **FDA SaMD Class II Certification Package (v3.3.0)**: Complete IEC 62304 Software Requirement Traceability Matrix with 32 SRS items and 100% test coverage (`docs/TRACEABILITY_MATRIX.md`), ISO 14971 FMEA Risk Register with 22 hazards (`docs/RISK_ANALYSIS.md`), FDA 510(k) Predicate Comparison against Philips Lifeline AutoAlert K151548 (`docs/PREDICATE_COMPARISON.md`), and automated Clinical Performance Validation Report generator (`docs/CLINICAL_PERFORMANCE_REPORT.md`).
15. **Enterprise Healthcare Deployment & Vital Signs (v3.4.0)**: Post-Fall Respiration & Vital Signs micro-Doppler Estimator (`hub/vital_signs.py`, 6-30 bpm), Automated Clinical Trial Cohort Simulator (`hub/clinical_trial_runner.py`), FDA GMLP Principle 7 Demographic Fairness & Disparity Report (`docs/CLINICAL_TRIAL_COHORT_REPORT.md`), Multi-Facility Cloud Gateway with Offline Store-and-Forward SQLite Queue (`hub/cloud_sync.py`), Continuous System Diagnostics & IEC 60601-1-8 Alarm System Watchdog Daemon (`hub/diagnostics.py`).
16. **Detailed Audit & Technical Findings**: Consult [**`SYSTEM_AUDIT_AND_FINDINGS.md`**](SYSTEM_AUDIT_AND_FINDINGS.md) for the comprehensive living engineering log, physics calculations, FMEA edge cases, and architectural benchmarks.


---

## System Architecture

```
                                  ROOM DEPLOYMENT
   =============================================================================
   [Ceiling / High Wall @ 2.4m]             [Opposite Wall @ 1.2m]
      Node 0: ESP32 Transmitter (AP)           Node 1: Tracker (Rx1)
      (100 Hz ESP-NOW Ping Frames)              (CSI Subcarrier Extraction)
                |                                          |
                +-----------------+------------------------+
                                  |
               [Opposite Wall @ 1.2m]       [Floor Baseboard @ 0.3m]
                  Node 2: Tracker (Rx2)        Node 3: Tracker (Rx3)
               (CSI Subcarrier Extraction)  (CSI Vertical Velocity Link)
   =============================================================================
                                  | (UDP Streams)
                                  v
                    +---------------------------+
                    | Central Processing Hub    |
                    | (PC / Home Assistant / Pi)|
                    |                           |
                    | * Butterworth Bandpass    |
                    | * PCA Dimensionality Red. |
                    | * Doppler Velocity (STFT) |
                    | * 3-Link Coincidence Vote |
                    | * Gradient-Boosted ML Clf |
                    +-------------+-------------+
                                  |
                                  +<--------- [Plan 2: 60 GHz mmWave Radar]
                                  |           (Direct Floor Altitude <0.35m)
                                  v
                    +---------------------------+
                    | Dual-Modality Fusion      |
                    | * Consensus State Machine |
                    | * Inactivity Timer (>=4s) |
                    | * P(fall) Score Weighting |
                    +------+-------------+------+
                           |             |
            +--------------+             +--------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
| Real-Time Web HUD     |                   | Alert & Integrations  |
| * Doppler Waterfall   |                   | * Home Assistant MQTT |
| * Z-Axis Alt Chart    |                   | * Siren Beep / Audio  |
| * Node Mesh Badges    |                   | * CSV Incident Log    |
+-----------------------+                   +-----------------------+
```

---

## Hardware Bill of Materials (BOM)

### Plan 1: 4-Node Wi-Fi CSI System
- **4x ESP32 or ESP32-C6 Development Boards** (ESP32-C6 recommended for 802.11ax Wi-Fi 6 CSI).
- Micro-USB / USB-C cables and 5V USB power adapters.

### Plan 2: mmWave Radar System
- **1x ESP32 or ESP32-C6 Development Board**.
- **1x 60 GHz mmWave Fall Detection Radar Module** (e.g., Seeed Studio MR60FDA1 or HLK-LD6002 / LD2450).
- 4x Jumper wires:
  - `VCC` -> ESP32 `5V`
  - `GND` -> ESP32 `GND`
  - `TX`  -> ESP32 `GPIO16` (RXD)
  - `RX`  -> ESP32 `GPIO17` (TXD)

---

## Power Budget & Deployment Constraints

> [!IMPORTANT]
> **All ESP32 nodes must be wall-powered (USB 5V).** Battery operation is not viable.

The 100 Hz ESP-NOW active injection architecture prevents any ESP32 sleep modes. Each node will draw **~150–200 mA continuously** (Wi-Fi Tx + CSI Rx + UDP streaming). Use standard 5V/1A USB adapters for each node.

| Node | Role | Current Draw | Power Source |
|------|------|-------------|--------------|
| Node 0 (AP/Tx) | 100 Hz ESP-NOW broadcast | ~160 mA | USB 5V adapter |
| Nodes 1–3 (Rx) | CSI capture + UDP stream | ~180 mA | USB 5V adapter |
| mmWave Gateway | UART parse + UDP forward | ~120 mA | USB 5V adapter |
| 60 GHz Radar Module | FMCW sensing | ~100 mA | Powered via ESP32 5V pin |

**Multi-Person Limitation:** The PCA-based motion extraction targets the dominant eigenvector. With 2+ people in the room, the CSI subsystem may produce mixed velocity estimates. When the radar detects multiple targets, the system should rely on mmWave centroid altitude tracking.

**Network:** All firmware nodes use **UDP broadcast** (`255.255.255.255`) by default — no hardcoded hub IP required. The hub binds on `0.0.0.0` and receives packets from any node on the local subnet.

## Room Geometry & Node Placement

Because human falls are **vertical kinetic events** ($1.7\,\text{m} \rightarrow 0\,\text{m}$), 3D spatial positioning maximizes Doppler shift sensitivity:
- **Node 0 (AP / Tx)**: Mount on the ceiling or upper wall ($2.0 - 2.4\,\text{m}$).
- **Node 1 & Node 2 (Trackers / Rx1, Rx2)**: Mount at mid-height ($0.9 - 1.2\,\text{m}$) on opposite lateral walls.
- **Node 3 (Tracker / Rx3)**: Mount low near the floor ($0.2 - 0.4\,\text{m}$) to capture ground-level multipath changes.

---

## Firmware Setup & Flashing (ESP-IDF)

### 1. Flash Transmitter AP (Node 0)
```powershell
cd firmware/wifi_csi/transmitter_ap
idf.py set-target esp32c6   # or esp32
idf.py build
idf.py -p COM_PORT flash monitor
```

### 2. Flash Tracker Nodes (Nodes 1, 2, 3)
For each tracker, edit `CONFIG_TRACKER_NODE_ID` in [`firmware/wifi_csi/tracker_node/main/main.c`](firmware/wifi_csi/tracker_node/main/main.c) to `1`, `2`, or `3`, then flash:
```powershell
cd firmware/wifi_csi/tracker_node
idf.py set-target esp32c6
idf.py build
idf.py -p COM_PORT flash monitor
```

### 3. Flash mmWave Radar Gateway (Plan 2)
```powershell
cd firmware/mmwave_radar
idf.py set-target esp32c6
idf.py build
idf.py -p COM_PORT flash monitor
```

---

## Host Python Hub Quickstart

### 1. Setup Virtual Environment
```powershell
# Create environment with Python 3.11
py -3.11 -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Interactive Simulation Demo & Web HUD
Experience live multi-link CSI and mmWave radar detection with the interactive Web HUD without physical hardware:
```powershell
python hub/server.py --demo --web
```
Open **`http://localhost:8000`** in your browser to observe the live Doppler waterfall spectra, altitude gauge, and real-time state transitions.

### 3. Run Live Sensing Modes
```powershell
# Plan 1: 4-Node Wi-Fi CSI Mode (UDP port 5555)
python hub/server.py --mode csi --web

# Plan 2: mmWave Radar Mode (UDP port 5556)
python hub/server.py --mode radar --web

# Dual-Sensor Fusion Mode (Combines both modalities + ML)
python hub/server.py --mode fusion --web

# Enable Home Assistant MQTT Auto-Discovery
python hub/server.py --mode fusion --web --mqtt-broker 192.168.1.50 --ha-discovery
```

---

## Multimodal Dataset Recorder & Replay Tool

Record real-world Wi-Fi CSI matrices and mmWave radar telemetry into compressed `.npz` datasets with JSON metadata for model training and benchmark verification:

```powershell
# Record 30 seconds of live activity labeled as 'fall'
python -m hub.recorder --duration 30 --label fall --output datasets/fall_experiment_01.npz

# Record simulated data for testing without live nodes
python -m hub.recorder --simulate --duration 10 --label simulated_fall --output datasets/test.npz

# Replay recorded dataset through the detection engine and Web HUD
python hub/server.py --replay datasets/fall_experiment_01.npz --web
```

---

## Probabilistic Machine Learning Classifier

In addition to thresholded PCA Doppler analysis, the system includes a supervised `HistGradientBoostingClassifier` (`hub/csi_pipeline/classifier.py`):
- Extracts a 9-dimensional kinematic feature vector per window:
  - Doppler sub-band energies: `0-5 Hz`, `5-15 Hz`, `15-25 Hz`, `25-40 Hz`
  - High-to-low kinetic ratio
  - Dominant Doppler velocity
  - Total energy surge ratio
  - Temporal variance decay
  - Spectral entropy
- Estimates the posterior probability of a human fall $P(\text{fall}) \in [0.0, 1.0]$.
- Integrates into `DualFusionEngine` for hybrid confidence escalation.

---

## Home Assistant MQTT Integration

The system natively implements the Home Assistant MQTT Discovery protocol (`hub/ha_discovery.py`):

- **13 Auto-Discovered Entities**:
  - `binary_sensor.fall_detection_hub_fall_detected` (Safety device class)
  - `sensor.fall_detection_hub_system_state`
  - `sensor.fall_detection_hub_radar_height` (Centroid distance in meters)
  - `sensor.fall_detection_hub_radar_posture` (Standing / Sitting / Lying Down)
  - `sensor.fall_detection_hub_radar_dwell` (Seconds on floor)
  - `sensor.fall_detection_hub_ml_fall_probability` (0–100%)
  - `sensor.fall_detection_hub_csi_node_<1..3>_velocity`
  - `sensor.fall_detection_hub_csi_node_<1..3>_surge`
  - `button.fall_detection_hub_reset_alarm`
- Ready-to-use automations are provided in [`config/ha_automations.yaml`](config/ha_automations.yaml) for critical sirens, emergency light flashing, and smart speaker announcements.

---

## Docker Edge Deployment

Deploy the entire fall detection hub and an optional local Mosquitto MQTT broker on edge devices (Raspberry Pi 4/5, x86 mini PCs):

```powershell
cd deploy

# Start Hub and Mosquitto broker
docker compose up -d

# View live logs
docker compose logs -f hub
```

> [!TIP]
> The container uses `network_mode: "host"` so the hub can directly receive low-latency UDP broadcast/multicast packets on ports 5555 and 5556 without NAT overhead.

---

## Running Automated Tests

Run the comprehensive test suite covering signal processing, ML classification, web endpoints, and consensus logic:

```powershell
pytest -v tests/
```

Test coverage (202 passing tests, 7 skipped):
- `test_training_buffer.py`: Rolling ring-buffer for incremental retraining with ground-truth nursing event annotation.
- `test_drift_detector.py`: Population Stability Index (PSI) and KL divergence concept drift detection.
- `test_retraining_pipeline.py`: Automated model retraining, evaluation, clinical safety floors, and rollback.
- `test_model_registry.py`: SQLite versioned model store, SHA-256 cryptographic integrity verification, and IEC 62304 changelog.
- `test_vital_signs.py`: Post-fall respiration micro-Doppler estimator (6-30 bpm) and inanimate signal rejection.
- `test_clinical_trial_runner.py`: Clinical trial cohort simulation (4 demographics) and FDA GMLP fairness disparity verification.
- `test_cloud_sync.py`: Multi-facility cloud gateway with offline store-and-forward SQLite queue and retry policies.
- `test_diagnostics.py`: Continuous system diagnostics daemon, packet rate/jitter monitoring, and IEC 60601-1-8 self-test.
- `test_traceability.py`: IEC 62304 Software Requirement Traceability Matrix with 32 SRS items and 100% test coverage.
- `test_risk_analysis.py`: ISO 14971 FMEA Risk Register with 22 validated hazards.
- `test_clinical_report.py`: Automated clinical performance validation report generator.
- `test_active_veto.py`: Active radar veto, standing posture false alarm suppression, and kinematic slump detection.
- `test_adaptive_calibrator.py`: Carrier-aware Doppler scaling (2.4 GHz vs 5.8 GHz) and continuous EMA noise tracking.
- `test_clutter_filter.py`: Spatial Elevation Perturbation Ratio (EPR) and radar cluster area ($<0.15\,\text{m}^2$) pet filters.
- `test_train_pipeline.py`: Empirical dataset training, Stratified 5-Fold Cross-Validation, and ROC/PR metric evaluations.
- `test_dashboard.py`: FastAPI routes, WebSockets, and REST management endpoints (`/api/calibrate`, `/api/thresholds`, `/api/labels/*`, `/api/drift/*`, `/api/retrain/*`).
- `test_csi_pipeline.py`: Raw CSI packet decoding (`CSIF`), Butterworth filtering, PCA, and Doppler velocity.
- `test_mmwave_parser.py`: 60 GHz mmWave radar binary frame parser (`0x53 0x59`) with checksum validation and cluster parsing.
- `test_classifier.py`: 9D kinematic feature extraction, ML probability discrimination, and hybrid fusion escalation.
- `test_recorder.py`: Multimodal session buffer synchronization and `.npz` dataset replay.
- `test_ha_discovery.py`: Home Assistant MQTT discovery schemas, retained announcements, and automation YAML validation.
- `test_integration.py`: Multi-link consensus, false positive rejection, sequence gap interpolation, and node health monitoring.
- `test_alert_dispatcher.py`: Cooldown rate-limiting, CSV logging, MQTT alerts, and HTTP webhooks.
- `test_calibrate.py`: Ambient noise floor baseline calibration.

---

## Project Directory Structure

```
Fall_Detection/
├── config/
│   ├── csi_config.yaml           # Wi-Fi CSI thresholds & network settings
│   ├── radar_config.yaml         # mmWave radar parameters & height thresholds
│   ├── calibration.yaml          # Auto-generated room noise profile & thresholds
│   └── ha_automations.yaml       # Home Assistant automation templates
├── deploy/
│   ├── Dockerfile                # Multi-arch edge deployment container
│   ├── docker-compose.yml        # Hub + Mosquitto broker compose stack
│   └── mosquitto.conf            # Local MQTT broker configuration
├── docs/
│   └── DATA_COLLECTION_PROTOCOL.md # Clinical human trial data capture protocol
├── firmware/
│   ├── wifi_csi/
│   │   ├── transmitter_ap/       # Node 0 (AP): 100 Hz ESP-NOW active injector
│   │   │   ├── CMakeLists.txt
│   │   │   └── main/             # ESP-IDF component directory
│   │   └── tracker_node/         # Nodes 1, 2, 3: CSI receiver & UDP streamer (mDNS enabled)
│   │       ├── CMakeLists.txt
│   │       └── main/
│   └── mmwave_radar/             # Plan 2: ESP32 + 60GHz mmWave radar gateway (mDNS enabled)
│       ├── CMakeLists.txt
│       └── main/
├── hub/
│   ├── csi_pipeline/
│   │   ├── preprocessor.py       # Denoising, phase unwrapping & Butterworth filter
│   │   ├── pca_features.py       # PCA dimensionality & carrier-aware Doppler velocity STFT
│   │   ├── multi_link_fusion.py  # 3-Link coincidence voting, EPR pet filter, & state machine
│   │   └── classifier.py         # 9D kinematic feature extractor & ML classifier
│   ├── dashboard/
│   │   ├── app.py                # FastAPI + WebSockets broadcaster & REST fleet endpoints
│   │   └── static/               # Zero-npm canvas HUD (Doppler waterfall, Z-axis)
│   ├── mmwave_pipeline/
│   │   └── radar_receiver.py     # Binary & JSON protocol decoder & cluster area filter
│   ├── adaptive_calibrator.py    # Background EMA noise floor calibrator
│   ├── alert_dispatcher.py       # Sirens, CSV logger, MQTT & HTTP Webhooks
│   ├── calibrate.py              # Room noise floor calibration & threshold generator
│   ├── fusion_engine.py          # Dual-modality consensus, Active Radar Veto & Slump Tracker
│   ├── ha_discovery.py           # Home Assistant MQTT Auto-Discovery generator
│   ├── recorder.py               # Multimodal dataset recorder (.npz + JSON)
│   ├── server.py                 # Multi-threaded hub server, replay, & demo simulator
│   └── train.py                  # Empirical ML training, cross-validation & ROC evaluator
├── models/
│   ├── fall_classifier.pkl       # Calibrated production classifier model
│   └── evaluation_report.json    # 5-fold cross-validation metrics report
├── tests/                        # Full unit and integration test suite (62+ passing tests)
├── .gitignore
├── LICENSE                       # MIT License
├── README.md                     # Documentation
└── requirements.txt              # Dependencies
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
