# Config — System Configuration & Sensor Profiles

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `config/` directory contains configuration profiles, sensor parameters, and smart home integration rules:

- **`config.yaml`**: Primary runtime parameters for the central hub daemon, including socket ports, fusion thresholds, and alert intervals.
- **`csi_config.yaml`**: Wi-Fi CSI processing settings, subcarrier groupings, STFT window parameters, and noise filtering gains.
- **`radar_config.yaml`**: 60 GHz mmWave FMCW radar thresholds, baud rate, UART interface paths, and elevation boundary filters.
- **`ha_automations.yaml`**: MQTT auto-discovery manifests and sensor integration templates for Home Assistant.
- **`api_tokens.yaml.template`**: Sanitized secret template for bearer tokens and external webhook credentials (actual secrets are excluded via `.gitignore`).
