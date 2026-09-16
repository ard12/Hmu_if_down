# Clinical Human Trial Data Collection Protocol
## Multimodal Wi-Fi CSI & 60 GHz mmWave Radar Fall Detection System

**Document ID:** SOP-CLIN-FD-001  
**Revision:** 1.0  
**Effective Date:** 2026-09-16  
**Regulatory Context:** Aligned with FDA Class II Medical Device Guidance (Software as a Medical Device - SaMD) & IEC 62304 Medical Device Software Life Cycle Processes.

---

## 1. Executive Summary & Objectives
This Standard Operating Procedure (SOP) defines the operational, safety, and data-integrity requirements for conducting empirical human trials to capture synchronized Wi-Fi Channel State Information (CSI) and 60 GHz frequency-modulated continuous-wave (FMCW) mmWave radar telemetry. The captured datasets form the empirical corpus used for training, validating, and certifying the probabilistic fall classification pipeline (`hub/train.py`).

The primary objectives are:
1. **Kinematic Precision**: Acquire ground-truth synchronized multi-bistatic RF and radar Doppler signatures across diverse anthropometric profiles.
2. **False Alarm Stress-Testing**: Rigorously capture challenging Activities of Daily Living (ADLs)—including dropping objects, pet movements, abrupt chair sitting, and floor picking—to evaluate and maximize system specificity ($\ge 98\%$).
3. **Safety & Ethics**: Ensure zero-injury experimental execution using certified crash-attenuation apparatus and multi-spotter safety controls.

---

## 2. Experimental Geometry & Hardware Setup

### 2.1 Environmental Enclosure
- **Room Dimensions**: $5.0\,\text{m} \times 5.0\,\text{m} \times 2.8\,\text{m}$ testing area with non-conductive boundary walls (drywall / timber frame).
- **Floor Surfaces**: Conduct trials across two distinct flooring regimes:
  - High-friction low-pile commercial carpet.
  - Smooth vinyl / polished hardwood (for slip simulations).

### 2.2 Transceiver & Sensor Positioning
```
                [Wi-Fi AP Router] (Height: 1.5m, North Wall)
                          ▲
                          │
       Link 1 (Elevated)  │  Link 2 (Mid-Level)
      ┌───────────────────┼───────────────────┐
      │                   │                   │
[Node 1: 2.4m]      [Radar: 2.2m, 45°]   [Node 2: 1.2m]
(West Wall)         (Center Ceiling)     (East Wall)
                          │
                          │ Link 3 (Floor Level)
                          ▼
                   [Node 3: 0.3m]
                   (South Baseboard)
```

1. **Tracker Node 1 (Elevated)**: ESP32-S3 mounted on West wall at $Z = 2.40\,\text{m}$. Sensitive to whole-body vertical posture transitions and upper-torso kinetics.
2. **Tracker Node 2 (Mid-Elevation)**: ESP32-S3 mounted on East wall at $Z = 1.20\,\text{m}$. Captures center-of-mass (COM) traversal and torso velocity bursts.
3. **Tracker Node 3 (Floor Baseboard)**: ESP32-S3 mounted on South baseboard at $Z = 0.30\,\text{m}$. Critical for the **Elevation Perturbation Ratio (EPR)** disambiguation of ground-level pets vs human floor impacts.
4. **mmWave Radar Gateway**: MR60FDA1 60 GHz radar module positioned at ceiling height $Z = 2.20\,\text{m}$, tilted downward at a $45^\circ$ angle facing the test perimeter center.
5. **Central Hub Server**: Linux/Windows host workstation running `hub/server.py` and `hub/recorder.py`, connected to the dedicated test Wi-Fi network via gigabit ethernet.

---

## 3. Cohort Stratification & Eligibility

### 3.1 Cohort A: Staged High-Velocity Dynamic Falls (Healthy Adults)
- **Age**: 20 – 45 years.
- **Inclusion Criteria**: Healthy individuals without acute musculoskeletal, vestibular, cardiovascular, or spinal disorders; BMI between 18.5 and 32.0.
- **Role**: Execute dynamic, high-impact fall simulations (forward trips, backward slips, lateral impacts).

### 3.2 Cohort B: Ambulatory & Geriatric Kinematics (Elderly Volunteers / Physical Therapy)
- **Age**: 65+ years.
- **Inclusion Criteria**: Ambulatory individuals with or without mild gait instability (Timed Up and Go test score between 8 and 20 seconds).
- **Role**: Execute exclusively non-fall Activities of Daily Living (ADLs) and controlled bed/chair slumps assisted by physical therapists onto high-density safety mats. **No high-velocity drops permitted for Cohort B.**

---

## 4. Test Battery & Activity Protocols

Each subject performs a randomized sequence of trials across both Class 1 (Falls) and Class 0 (ADLs). Every trial is recorded for exactly $15.0\,\text{seconds}$ using `hub/recorder.py`.

### 4.1 Class 1: Fall Scenarios (Target: Ground Truth = 1)
| Code | Scenario | Kinematic Description | Target Phase Duration |
| :--- | :--- | :--- | :--- |
| **FALL_FWD** | Forward Trip | Participant walks 2 steps, trips over a soft obstacle, and falls forward onto the chest/hands on the mat. | Impact at $t \approx 3.0\text{--}4.0\,\text{s}$, followed by 10s immobility. |
| **FALL_BWD** | Backward Slip | Participant experiences sudden loss of foot traction, arms flail upward, falling backward onto buttocks and back. | Impact at $t \approx 3.0\text{--}3.5\,\text{s}$, followed by 10s immobility. |
| **FALL_LAT** | Lateral Fall | Participant loses balance sideways, impacting on hip/shoulder. | Impact at $t \approx 3.0\text{--}4.0\,\text{s}$, followed by 10s immobility. |
| **SLUMP_CHR** | Geriatric Chair Slump | Participant seated upright on an armchair gradually slides down the seat cushions over $2.0\text{--}3.5\,\text{s}$ until resting on the floor. | Descent $t = 2.0\text{--}5.5\,\text{s}$, dwell on floor for remaining 9s. |
| **SLUMP_BED** | Bed Roll-off | Participant lying near mattress edge rolls off onto the floor mat. | Transition $t = 2.5\text{--}4.0\,\text{s}$, dwell on floor. |

### 4.2 Class 0: Activities of Daily Living (Target: Ground Truth = 0)
| Code | Scenario | Stress Test Objective | Execution Details |
| :--- | :--- | :--- | :--- |
| **ADL_WALK** | Normal & Rapid Walking | Doppler baseline calibration | Continuous pacing across the room at $1.0\text{--}1.8\,\text{m/s}$. |
| **ADL_SIT_HRD**| Abrupt Chair Seating | Disambiguate downward sitting from fall | Participant walks to chair and sits down forcefully within 0.8s. |
| **ADL_SOFA** | Reclining on Soft Sofa | Evaluate soft deceleration profiles | Participant falls back comfortably onto sofa cushions. |
| **ADL_PICK** | Bending to Pick Object | Elevate Link 3 without sustained floor dwell | Participant bends at waist, grabs keys from floor, returns upright within 2.0s. |
| **ADL_DROP** | Dropping Heavy Item | **Radar Veto Stress Test** | Standing participant ($Z = 1.7\,\text{m}$) drops a 3 kg medicine ball / metal box. |
| **ADL_PET** | Ground Pet / Vacuum | **EPR Filter Stress Test** | Robotic vacuum or domestic dog traversing room floor while Links 1 & 2 remain quiet. |

---

## 5. Participant Safety & Risk Mitigation SOP

> [!CAUTION]
> High-velocity impact falls must strictly adhere to the safety standards below. No exceptions are permitted.

1. **Impact Attenuation Apparatus**:
   - Primary landing zone must feature a certified $2.0\,\text{m} \times 2.0\,\text{m} \times 0.15\,\text{m}$ high-density polyurethane gymnastics crash mat with vinyl cover.
   - Mat edges must be beveled or taped to avoid unintended trip hazards outside designated test sequences.
2. **Personal Protective Equipment (PPE)**:
   - Cohort A participants must wear low-profile CE-certified soft-shell elbow pads, knee pads, and reinforced hip-protector shorts underneath standard athletic clothing.
3. **Spotter Protocol**:
   - Two trained research assistants must stand at the mat periphery.
   - Spotters are tasked with decelerating the participant's descent if erratic rotational or head-first trajectories occur.
4. **Medical Protocol**:
   - Automated External Defibrillator (AED) and standard first aid kit present in the testing facility.
   - Post-trial physiological screening: blood pressure and pulse checked before and after session sets.

---

## 6. Data Acquisition & Recording SOP

### 6.1 Recording Invocation
For each experimental trial, execute `hub/recorder.py` via command-line interface:

```powershell
python hub/recorder.py `
  --label "fall_forward" `
  --subject "sub_014" `
  --duration 15.0 `
  --notes "carpet_floor, 78kg_male, 181cm, athletic_wear"
```

### 6.2 Data File Hierarchy
Recordings are automatically indexed and saved to the `datasets/` repository:
```
datasets/
├── fall_forward_sub_014_20260916_143022.npz
├── fall_forward_sub_014_20260916_143022.json
├── adl_drop_sub_014_20260916_143215.npz
└── adl_drop_sub_014_20260916_143215.json
```

### 6.3 Data Verification Checklist
Before releasing the participant from a session set, the lead investigator must run dataset verification:
1. **Packet Continuity**: Verify packet arrival rate on all 3 nodes is $100 \pm 5\,\text{Hz}$ (minimum 1,400 packets per node per 15s trial).
2. **Missing Frame Rate**: Packet loss must be $< 1.0\%$. If sequence number gaps exceed 15 frames, discard trial and re-record.
3. **Radar Telemetry Continuity**: Confirm mmWave gateway logged $\ge 140$ frames (nominal $10\,\text{Hz}$).
4. **Signal-to-Noise Ratio (SNR)**: Check that baseline RSSI is between $-40\,\text{dBm}$ and $-65\,\text{dBm}$. Discard trials where RSSI drops below $-75\,\text{dBm}$.

---

## 7. Model Training & Pipeline Ingestion

Upon completion of the recording campaign:
1. Run automated cross-validation and production model export:
   ```powershell
   python -m hub.train --data-dir datasets/ --output-model models/fall_classifier.pkl --n-splits 5
   ```
2. Verify cross-validation sensitivity exceeds $98.5\%$ and specificity exceeds $98.0\%$.
3. Check `models/evaluation_report.json` for per-fold ROC-AUC and PR-AUC validation.

---

## 8. Ethics, Anonymization & Governance
- All participant recordings must be pseudo-anonymized (`sub_XXX`). No identifying personal health information (PHI) or video recordings are stored in the `.npz` archive.
- Subject demographic keys (height, weight, age, sex) must be stored in an encrypted, password-protected offline spreadsheet accessible only by the Principal Investigator.
- Research conducted under Institutional Review Board (IRB) Protocol `#2026-FD-CSI-04`.
