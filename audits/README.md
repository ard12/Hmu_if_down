# Audits — Clinical Audit Logging & FHIR Storage

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

FHIR R4 exports and SQLite audit databases are stored in this directory:
- Created automatically by `AuditLog` on first use.
- Runtime database files (`audit.db`, `fleet.db`, `longitudinal.db`) and FHIR export snapshots (`fhir_export_*.json`) are strictly excluded via `.gitignore` to preserve patient confidentiality.
