# Helm — Cloud-Native Kubernetes Deployment Chart

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `helm/` directory provides enterprise-grade Helm v3 charts for deploying the system onto Kubernetes clusters in enterprise healthcare and hospital environments:

- **`fall-detection-hub/`**:
  - `Chart.yaml`: Helm chart metadata for `fall-detection-hub` (v4.1.0).
  - `values.yaml`: Configurable cluster parameters, replica management, ingress definitions, resource limits, and telemetry endpoints.
  - `templates/`: Manifests for Kubernetes Deployments, Services, Horizontal Pod Autoscalers (HPA), PersistentVolumeClaims (PVC), ConfigMaps, and Prometheus ServiceMonitors.
