# Static Application Security Testing (SAST) Audit Report

> **FDA SaMD / IEC 62304 / HIPAA Technical Safeguards Security Assessment**

## 1. Executive Summary

- **Scan Date**: `2026-09-22T20:46:39Z`
- **Lines of Code Scanned**: `9215`
- **Critical / High Severity Vulnerabilities**: `0` (Requirement: **0**)
- **Medium Severity Findings**: `15`
- **Low Severity Warnings**: `38`
- **Overall Status**: **PASSED** (0 High Severity)

## 2. Severity Summary

| Severity Level | Finding Count | Threshold Allowed | Status |
| :--- | :--- | :--- | :--- |
| **HIGH** | 0 | 0 | PASS |
| **MEDIUM** | 15 | Acceptable with Compensating Control | REVIEWED |
| **LOW** | 38 | Informational | ACCEPTED |

## 3. Detailed Findings & Mitigation Traceability

| Test ID | CWE | Severity | File & Line | Description | SRS Mapping | Compensating Control / Remediation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `B310` | `CWE-22` | `MEDIUM` | `hub/alert_dispatcher.py:152` | Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. | `SRS-SEC-006` | URL schemes restricted; outbound webhook dispatch operates with strict 5-second timeouts. |
| `B608` | `CWE-89` | `MEDIUM` | `hub/audit_log.py:182` | Possible SQL injection vector through string-based query construction. | `SRS-SEC-007` | Dynamic SQL clauses use parameterized SQLite queries; WHERE conditions strictly whitelist column names. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/calibrate.py:167` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B301` | `CWE-502` | `MEDIUM` | `hub/csi_pipeline/classifier.py:231` | Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue. | `SRS-SEC-004` | Enforce SHA-256 / HMAC cryptographic verification before unpickling; restricted to internal SQLite model store. |
| `B301` | `CWE-502` | `MEDIUM` | `hub/fall_type_classifier.py:211` | Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue. | `SRS-SEC-004` | Enforce SHA-256 / HMAC cryptographic verification before unpickling; restricted to internal SQLite model store. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/hl7_listener.py:22` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B301` | `CWE-502` | `MEDIUM` | `hub/model_registry.py:127` | Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue. | `SRS-SEC-004` | Enforce SHA-256 / HMAC cryptographic verification before unpickling; restricted to internal SQLite model store. |
| `B301` | `CWE-502` | `MEDIUM` | `hub/model_registry.py:200` | Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue. | `SRS-SEC-004` | Enforce SHA-256 / HMAC cryptographic verification before unpickling; restricted to internal SQLite model store. |
| `B301` | `CWE-502` | `MEDIUM` | `hub/onnx_runner.py:106` | Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue. | `SRS-SEC-004` | Enforce SHA-256 / HMAC cryptographic verification before unpickling; restricted to internal SQLite model store. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/recorder.py:192` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/recorder.py:197` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/server.py:757` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/server.py:850` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B104` | `CWE-605` | `MEDIUM` | `hub/server.py:865` | Possible binding to all interfaces. | `SRS-SEC-005` | Enforced by container network namespace isolation and Kubernetes Ingress / Service boundary. |
| `B301` | `CWE-502` | `MEDIUM` | `hub/train.py:486` | Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue. | `SRS-SEC-004` | Enforce SHA-256 / HMAC cryptographic verification before unpickling; restricted to internal SQLite model store. |

## 4. Compensating Controls & Architectural Hardening

1. **Model Deserialization (`B301` / `CWE-502`)**:
   - SQLite `model_versions` table stores SHA-256 checksums alongside serialized model blobs.
   - Prior to invoking `pickle.loads()`, `ModelRegistry.load_model()` and `ModelRegistry.get_active()` compute the SHA-256 hash of the blob and compare against the stored hash.
   - If tampering is detected, a `ValueError` is raised immediately, halting deserialization.

2. **Network Socket Binds (`B104` / `CWE-605`)**:
   - Binding to `0.0.0.0` is standard for containerized Microservices deployed in Kubernetes pods.
   - All pod ingress is strictly gated by Kubernetes NetworkPolicies, Ingress TLS termination, and mutual authentication.

3. **SQL Query Construction (`B608` / `CWE-89`)**:
   - Dynamic SQL clauses in `hub/audit_log.py` use strictly parameterized queries (`?` place-holders).
   - Column names in conditional clauses are hardcoded internal identifiers, preventing external SQL injection.

4. **Outbound Requests (`B310` / `CWE-22`)**:
   - Alert dispatcher webhook delivery enforces explicit 5-second socket timeouts and URL validation.

---
*Report generated automatically by `docs/parse_bandit_report.py` as part of CI/CD SAST pipeline.*
