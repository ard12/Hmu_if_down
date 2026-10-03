# Docs — Regulatory, Clinical & Engineering Documentation

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `docs/` directory maintains regulatory compliance matrices, clinical validation reports, risk analyses, and automated traceability tooling:

- **`TRACEABILITY_MATRIX.md`**: IEC 62304 Class B/C Software Requirement-to-Test verification matrix (69/69 requirements verified).
- **`RISK_ANALYSIS.md`**: ISO 14971:2019 Medical Device Hazard Identification & FMEA Risk Mitigation matrix (37/37 hazards ALARP).
- **`HIPAA_COMPLIANCE_REPORT.md`**: Verification across all 9 technical safeguards of HIPAA Security Rule 45 CFR § 164.312.
- **`HE75_HUMAN_FACTORS_EVALUATION.md`**: Human factors engineering report according to ANSI/AAMI HE75 and IEC 62366-1.
- **`CLINICAL_PERFORMANCE_REPORT.md`**: FDA Class II SaMD sensitivity, specificity, and ROC/PR curve clinical evidence.
- **`SAST_REPORT.md`**: Automated Bandit Static Application Security Testing audit report.
- **Compliance Tooling**:
  - `generate_traceability.py`: Automated scanning tool that maps `@req` and `@covers` annotations to requirements.
  - `validate_risk_analysis.py`: Verifies residual risk acceptability across all identified hazard codes.
