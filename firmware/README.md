# Firmware — Embedded Edge Microcontroller Subsystems

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `firmware/` directory contains embedded C firmware targeting ESP-IDF (v5.x) on ESP32-S3 and standard ESP32 microcontrollers:

- **`wifi_csi/`**:
  - `injector_node/`: Fixed 100 Hz deterministic ESP-NOW frame injection with sequence counters and non-blocking FreeRTOS timer tasks.
  - `tracker_node/`: Receiver node capturing raw 802.11n Channel State Information (CSI) across 64 subcarriers (52 usable), queue decoupling, and low-latency UDP streaming to port 5555.
  - `mesh_forward/`: ESP-MESH multi-hop forwarding logic with TTL guards and hop weight metadata.
- **`mmwave_radar/`**:
  - 60 GHz FMCW radar UART gateway parsing frame headers, point clouds, target velocities, and broadcasting structured frames over UDP to port 5556.
