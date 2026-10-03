# Models — Calibrated Machine Learning Classifiers

> [!NOTE]
> **Repository Architecture**:
> - **Internal Production Repository**: `Fall_Detection` (`https://github.com/ard12/Fall_Detection.git`) — Primary internal engineering, clinical validation, and production codebase.
> - **Public-Facing Repository**: `Hmu_if_down` (`https://github.com/ard12/Hmu_if_down.git`) — Public-facing open-source distribution and external documentation portal.

## Overview

The `models/` directory houses calibrated production inference models, cryptographic integrity sidecars, and evaluation metrics:

- **`fall_classifier.pkl`**: Production `HistGradientBoostingClassifier` trained on multi-modal kinematic feature vectors (9-D representations across sub-band Doppler energies, surge velocity ratios, and spectral entropy). Deserialization is hardened for cross-version compatibility across NumPy 1.x and 2.x.
- **`fall_classifier.pkl.sha256`**: Cryptographic SHA-256 integrity checksum enforced by model runners prior to execution.
- **`evaluation_report.json`**: Stratified 5-fold cross-validation performance metrics (sensitivity, specificity, F1-score, Brier score, ROC-AUC).
- **`clinical_trial_cohort_report.json`**: Demographic stratification and multi-cohort subgroup performance validation data.
