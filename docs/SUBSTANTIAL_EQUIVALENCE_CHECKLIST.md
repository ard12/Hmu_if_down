# FDA 510(k) Substantial Equivalence Decision-Making Checklist

**Document ID**: SE-CHK-K151548-V3  
**Subject Device**: Autonomous Non-Invasive Fall Detection SaMD (v3.3.0)  
**Predicate Device**: Philips Lifeline HomeSafe with AutoAlert (K151548)  
**Guidance Document**: FDA Guidance *The 510(k) Program: Evaluating Substantial Equivalence in Premarket Notifications [510(k)]* (Issued July 28, 2014)

---

## Substantial Equivalence Decision Flowchart

```
[1] Same Intended Use? ------------> YES
         |
[2] Same Technological Characteristics? ---> NO (Ambient RF/Radar vs. Wearable Accelerometer)
         |
[3] Do different technological characteristics raise DIFFERENT questions of safety and effectiveness? ---> NO
         |
[4] Are the methods used to evaluate acceptable? ---> YES (IEC 62304, ISO 14971, Kinematic Benchmarks)
         |
[5] Do data demonstrate device is as safe and effective as predicate? ---> YES (Sensitivity 99.1% vs 95.3%, Specificity 98.4% vs 92.0%)
         |
======================================================
  DECISION: SUBSTANTIALLY EQUIVALENT (SE) DETERMINATION
======================================================
```

---

## Detailed Evaluation Checklist

| Item ID | Evaluation Question / Criterion | Predicate (K151548) | Subject Device (v3.3.0) | Compliance Status | Evidentiary Reference |
|---|---|---|---|---|---|
| **SE-001** | Is the predicate device legally marketed under the FD&C Act? | Cleared under 510(k) K151548, Class II. | N/A (Subject) | **PASS** | FDA 510(k) Database K151548 |
| **SE-002** | Does the subject device have the same intended use as the predicate? | Detect fall events in at-risk individuals and notify caregivers. | Detect fall events in at-risk individuals and notify caregivers. | **PASS** | `docs/PREDICATE_COMPARISON.md` §3 |
| **SE-003** | Does the subject device have the same target patient population? | Senior living, individuals with gait/balance impairments. | Senior living, individuals with gait/balance impairments. | **PASS** | `docs/PREDICATE_COMPARISON.md` §3 |
| **SE-004** | Do differences in technological characteristics introduce new types of safety hazards? | Wearable pendant with accelerometer/barometer. | Ambient WiFi CSI + 60 GHz mmWave radar. No new hazards identified; eliminates patient non-compliance. | **PASS** | `docs/RISK_ANALYSIS.md`, `docs/PREDICATE_COMPARISON.md` §4, §6 |
| **SE-005** | Are non-clinical performance testing methods scientifically valid and repeatable? | Bench drop tests and human volunteer simulated falls. | Standardized simulated human falls (500+ trials), ADL rejection, Doppler velocity validation. | **PASS** | `docs/CLINICAL_PERFORMANCE_REPORT.md` |
| **SE-006** | Is fall detection sensitivity equal to or better than predicate? | 95.3% sensitivity in clinical literature. | 99.1% sensitivity achieved on test validation sets. | **PASS** | `models/evaluation_report.json`, `docs/PREDICATE_COMPARISON.md` §5 |
| **SE-007** | Is fall detection specificity (ADL rejection) equal to or better than predicate? | 92.0% specificity. | 98.4% specificity achieved on validation sets. | **PASS** | `models/evaluation_report.json`, `docs/PREDICATE_COMPARISON.md` §5 |
| **SE-008** | Does the software lifecycle process comply with medical device standards? | IEC 62304 lifecycle documentation. | IEC 62304 Class B lifecycle process with 100% SRS traceability matrix. | **PASS** | `docs/TRACEABILITY_MATRIX.md`, `tests/test_traceability.py` |
| **SE-009** | Are cybersecurity and clinical audit trail requirements satisfied? | Base station internal logging. | SHA-256 hash-chained immutable audit log, 21 CFR Part 11 compliant. | **PASS** | `hub/audit_log.py`, `tests/test_audit_log.py` |
| **SE-010** | Does the overall risk/benefit assessment support clearance? | Acceptable risk profile for wearable PERS. | Favorable risk profile; eliminates fatal non-compliance risks of wearable pendants. | **PASS** | `docs/RISK_ANALYSIS.md`, `docs/PREDICATE_COMPARISON.md` §7 |

---

## Conclusion

The subject device satisfies all criteria (SE-001 through SE-010) established in the FDA 510(k) Substantial Equivalence decision-making guidance. The system is determined to be **Substantially Equivalent** to the predicate device.
