# Hub — Central Processing & Analytics Engine

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `hub/` directory contains the core Python daemon, signal processing pipelines, machine learning inference engines, and clinical web applications:

- **`csi_pipeline/`**: Micro-Doppler STFT spectrogram generation, PCA noise suppression, and fall feature extraction from raw 100 Hz Wi-Fi Channel State Information (CSI).
- **`mmwave_pipeline/`**: High-frequency 60 GHz FMCW radar point cloud parsing, elevation tracking, centroid altitude regression, and posture analysis.
- **`fusion_engine.py`**: Dual-modality consensus engine with active radar veto, multi-hypothesis tracking, and slump progression detection.
- **`dashboard/`**: FastAPI application serving real-time WebSocket telemetry streams and 12 clinical & engineering web portals.
- **`multi_occupant.py`**: Hungarian assignment and Kalman filtering for multi-person spatial tracking and disambiguation.
- **`audit_log.py`**: IEC 62304 & HIPAA §164.312-compliant SHA-256 tamper-evident clinical audit logging and FHIR export.
