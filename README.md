# Multi-Modal Device-Free Fall Detection System

[![CI](https://github.com/ard12/Fall_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ard12/Fall_Detection/actions)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v5.0+-red.svg)](https://docs.espressif.com/projects/esp-idf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, non-invasive, privacy-preserving fall detection platform engineered for elderly monitoring and healthcare facilities. Operates completely **device-free** (no wearables, pendants, or intrusive cameras) by combining two complementary wireless sensing paradigms:

1. **Plan 1 (Primary)**: **4-Node Wi-Fi CSI Active Sensing Mesh** (1 AP / Transmitter + 3 Receivers / Trackers using ESP-NOW @ 100 Hz).
2. **Plan 2 (High-Reliability)**: **ESP32 + 60 GHz mmWave FMCW Radar** (Direct 3D centroid altitude tracking and posture classification).
3. **Dual-Sensor Fusion Engine**: Cross-verifies wide-area RF multipath disturbances with centimeter-accurate floor height detection to achieve near-zero false alarms.

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
                    +-------------+-------------+
                                  |
                                  +<--------- [Plan 2: 60 GHz mmWave Radar]
                                  |           (Direct Floor Altitude <0.35m)
                                  v
                    +---------------------------+
                    | Dual-Modality Fusion      |
                    | * Consensus State Machine |
                    | * Inactivity Timer (>=4s) |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Alert & Siren Dispatcher  |
                    | * Audible Windows Beep    |
                    | * CSV Incident Logging    |
                    +---------------------------+
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
For each tracker, edit `CONFIG_TRACKER_NODE_ID` in [`firmware/wifi_csi/tracker_node/main.c`](firmware/wifi_csi/tracker_node/main.c) to `1`, `2`, or `3`, then flash:
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

### 2. Run Interactive Simulation Demo
Experience live multi-link CSI and mmWave radar detection without physical hardware:
```powershell
python hub/server.py --demo
```

### 3. Run Live Modes
```powershell
# Plan 1: 4-Node Wi-Fi CSI Mode (UDP port 5555)
python hub/server.py --mode csi

# Plan 2: mmWave Radar Mode (UDP port 5556)
python hub/server.py --mode radar

# Dual-Sensor Fusion Mode (Combines both modalities)
python hub/server.py --mode fusion
```

---

## Running Automated Tests

Run the comprehensive test suite covering both signal processing pipelines and consensus logic:

```powershell
pytest -v tests/
```

Test coverage includes:
- Raw CSI binary packet decoding (`CSIF` protocol) and corruption rejection.
- Zero-phase Butterworth filtering and subcarrier PCA decomposition.
- Doppler velocity estimation ($v = \frac{\lambda f_D}{2}$).
- 3-Link spatial coincidence window voting and floor quiescence verification.
- 60 GHz mmWave radar binary frame parser (`0x53 0x59`) with checksum validation.
- Dual-sensor cross-modal consensus engine.

---

## Project Directory Structure

```
Fall_Detection/
├── config/
│   ├── csi_config.yaml           # Wi-Fi CSI thresholds & network settings
│   └── radar_config.yaml         # mmWave radar parameters & height thresholds
├── firmware/
│   ├── wifi_csi/
│   │   ├── transmitter_ap/       # Node 0 (AP): 100 Hz ESP-NOW active injector
│   │   └── tracker_node/         # Nodes 1, 2, 3: CSI receiver & UDP streamer
│   └── mmwave_radar/             # Plan 2: ESP32 + 60GHz mmWave radar gateway
├── hub/
│   ├── csi_pipeline/
│   │   ├── preprocessor.py       # Denoising & Butterworth bandpass filter
│   │   ├── pca_features.py       # PCA dimensionality & Doppler velocity STFT
│   │   └── multi_link_fusion.py  # 3-Link coincidence voting & stillness state machine
│   ├── mmwave_pipeline/
│   │   └── radar_receiver.py     # Binary protocol decoder & height tracker
│   ├── alert_dispatcher.py       # Audio siren, cooldowns, and CSV logger
│   ├── fusion_engine.py          # Dual-modality consensus engine
│   └── server.py                 # Unified CLI hub and live demo simulator
├── tests/
│   ├── test_csi_pipeline.py      # Wi-Fi CSI signal tests
│   └── test_mmwave_parser.py     # Radar frame decoding tests
├── .gitignore
├── LICENSE                       # MIT License
├── README.md                     # Documentation
└── requirements.txt              # Dependencies
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
