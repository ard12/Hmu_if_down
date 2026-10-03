# Deploy — Production Edge & Container Deployment

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `deploy/` directory provides production deployment assets for bare-metal edge devices, localized Docker hosts, and clinical gateway nodes:

- **`Dockerfile`**: Multi-stage hardened container build running as non-root user `appuser:appuser`, embedding health checks and minimal attack surfaces.
- **`docker-compose.yml`**: Full-stack edge composition orchestrating the hub daemon, local Mosquitto MQTT broker, and persistent data volumes.
- **`falldetect-hub.service`**: Linux systemd unit file for headless appliances running 24/7 with automatic restart policies and resource limits.
- **`mosquitto.conf`**: Hardened MQTT broker configuration for local device telemetry and Home Assistant integration.
