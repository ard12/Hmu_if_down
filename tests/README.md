# Tests — Automated Verification & Validation (V&V) Suite

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `tests/` directory contains the automated pytest verification test suite (570+ test cases) implementing IEC 62304 Section 5.7 (Software Integration and Integration Testing) and Section 5.8 (Software System Testing):

- **Unit Tests**: Verifies signal preprocessors, Butterworth SOS bandpass filters, PCA feature extractors, and radar point-cloud parsers.
- **Fusion Engine Tests**: Evaluates multi-modality Bayesian consensus, active radar veto suppression, slump tracking, and timeout recovery.
- **API & Portal Tests**: Tests FastAPI endpoints, WebSocket telemetry streaming, and state machine transitions across all 12 operational portals.
- **Security & Cryptographic Tests**: Asserts HMAC model integrity, SHA-256 audit log hash-chain verification under simulated tampering, and session expiration timeouts.
- **Fuzzing & Chaos Tests**: Hypothesis property-based UDP packet fuzzer (`test_udp_fuzzer.py`) and network degradation simulations.
