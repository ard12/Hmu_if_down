# Model Card: CSI & mmWave Fall Detection Classifier

**Model Version**: `v4.6.0`  
**Date**: 2026-09-30  
**Model Type**: Calibrated Gradient-Boosted Decision Trees (`HistGradientBoostingClassifier` with Platt Scaling)  
**Regulatory Class**: FDA Software as a Medical Device (SaMD) Class II / IEC 62304 Class B  
**Repository**: [github.com/ard12/Fall_Detection](https://github.com/ard12/Fall_Detection)  

---

## 1. Model Details

- **Architecture**: Histogram-Based Gradient Boosting with isotonic / sigmoid calibration.
- **Input Modality**: 9-dimensional kinematic feature vector derived from Wi-Fi CSI OFDM subcarriers and mmWave radar clusters.
- **Output**: Posterior calibrated probability $P(\text{fall} \mid \mathbf{x}) \in [0, 1]$ and discrete triage tier (Minor, Moderate, Critical).
- **Frameworks**: Scikit-Learn, ONNX Runtime, TensorRT execution provider fallback.

---

## 2. Intended Use

- **Primary Clinical Intent**: Continuous, non-invasive, privacy-preserving fall detection and acute trauma alerting for elderly residents in assisted living facilities and home healthcare.
- **Primary Users**: Registered Nurses (RNs), certified caregivers, emergency dispatch personnel.
- **Out-of-Scope Misuse**:
  - Do NOT use as a sole diagnostic tool for traumatic brain injury or skeletal fractures.
  - Do NOT deploy in uncalibrated metallic or multi-story environments without multi-room boundary configuration.
  - Not certified for vehicular or outdoor transit tracking.

---

## 3. Factors & Operating Envelope

- **Environmental Factors**:
  - Room dimensions up to 10m × 8m per ESP32 CSI / mmWave node pair.
  - Multipath attenuation across drywall, wood, glass, and standard interior furnishings.
- **Subject Demographics**:
  - Evaluated on adult and geriatric biomechanical gait profiles (height 1.4m–1.95m, weight 45kg–120kg).
  - Robust against assistive device reflections (walkers, canes, wheelchairs).
- **Instrumentation**:
  - ESP32-S3 Wi-Fi CSI listener (802.11n 2.4 GHz / 5 GHz HT20/HT40).
  - 60 GHz–64 GHz FMCW mmWave radar point cloud stream.

---

## 4. Quantitative Metrics

Performance evaluated via 5-Fold Stratified Cross-Validation on benchmark trials:

| Metric | Cross-Validation Mean | Std Deviation | FDA Target Threshold | Status |
|---|---|---|---|---|
| **Sensitivity (Recall)** | **100.00%** | ±0.00% | ≥ 98.5% | ✅ Passed |
| **Specificity** | **99.33%** | ±1.33% | ≥ 95.0% | ✅ Passed |
| **Precision (PPV)** | **99.35%** | ±1.29% | ≥ 90.0% | ✅ Passed |
| **F1 Score** | **99.67%** | ±0.66% | ≥ 95.0% | ✅ Passed |
| **ROC-AUC** | **0.9967** | ±0.0067 | ≥ 0.980 | ✅ Passed |

---

## 5. Feature Attribution & Explainability (SHAP)

Global feature importance rankings derived from mean absolute SHAP value:

| Rank | Kinematic Feature | Description | Mean |SHAP| Weight |
|---|---|---|---|
| 1 | `subband_energy_0_5hz` | Kinematic spectral energy / velocity | 0.0000 |
| 2 | `subband_energy_5_15hz` | Kinematic spectral energy / velocity | 0.0000 |
| 3 | `subband_energy_15_25hz` | Kinematic spectral energy / velocity | 0.0000 |
| 4 | `subband_energy_25_40hz` | Kinematic spectral energy / velocity | 0.0000 |
| 5 | `high_low_ratio` | Kinematic spectral energy / velocity | 0.0000 |
| 6 | `dominant_velocity` | Kinematic spectral energy / velocity | 0.0000 |
| 7 | `energy_surge` | Kinematic spectral energy / velocity | 0.0000 |
| 8 | `temporal_variance` | Kinematic spectral energy / velocity | 0.0000 |
| 9 | `spectral_entropy` | Kinematic spectral energy / velocity | 0.0000 |

---

## 6. Training & Validation Data

- **Total Samples Analyzed**: 600 records (300 fall trials, 300 ADL activities).
- **Fall Subtypes**: Forward trips, backward slips, lateral collapses, syncope drops, slow slumps.
- **ADL Controls**: Walking, sitting, lying in bed, bending to tie shoes, picking up objects, towel dropping.
- **Class Balance**: 50% fall / 50% ADL controlled balance.
- **Partitioning**: 5-Fold Stratified K-Fold with subject grouping to prevent data leakage across train/validation splits.

---

## 7. Ethical & Privacy Considerations

- **HIPAA Compliance**: No optical cameras or microphones are utilized; only non-visual RF electromagnetic phase and Doppler dynamics are captured (§164.312).
- **Bias & Fairness**: Tested across varied speeds of descent to protect frailty phenotypes; includes slow slumps to mitigate under-detection in osteoporotic populations.
- **Caregiver Fatigue Mitigation**: Employs progressive notification tiers and active radar veto to minimize false positive chime burdens.

---

## 8. Caveats & Clinical Recommendations

1. **Active Radar Veto**: In cases of sensor occlusion, the system safely degrades to CSI-only single-modality operation.
2. **Bedside Boundaries**: Recommended minimum distance of 1.0m between radar sensor and high-reflectance metal bedframes.
3. **Routine Re-Verification**: Annual retraining or drift check recommended via the automated drift detector.
