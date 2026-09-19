"""Probabilistic Machine Learning Classifier for CSI Fall Detection.

Extracts multi-band spectral Doppler energy, velocity dynamics, and variance
decay signatures from multi-subcarrier CSI streams. Employs a Gradient-Boosted
Classifier to estimate the posterior probability of a human fall event P(fall).
"""

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import signal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from hub.csi_pipeline.pca_features import CSIDynamicFeatures


@dataclass
class MLFeatures:
    """Rich kinematic feature vector for machine learning classification."""
    subband_energy_0_5hz: float
    subband_energy_5_15hz: float
    subband_energy_15_25hz: float
    subband_energy_25_40hz: float
    high_low_ratio: float
    dominant_velocity: float
    energy_surge: float
    temporal_variance: float
    spectral_entropy: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.subband_energy_0_5hz,
            self.subband_energy_5_15hz,
            self.subband_energy_15_25hz,
            self.subband_energy_25_40hz,
            self.high_low_ratio,
            self.dominant_velocity,
            self.energy_surge,
            self.temporal_variance,
            self.spectral_entropy,
        ], dtype=np.float32)


class CSIFeatureExtractor:
    """Computes multidimensional spectral and kinematic features from CSI windows."""

    def __init__(self, sampling_rate_hz: float = 100.0, wavelength_m: float = 0.123):
        self.fs = sampling_rate_hz
        self.wavelength = wavelength_m

    def extract_from_window(
        self,
        window: np.ndarray,
        baseline_energy: float = 1.0,
    ) -> MLFeatures:
        """Extract ML feature vector from a (T, subcarriers) filtered amplitude matrix."""
        if window.shape[0] < 16:
            return MLFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        # 1. Temporal variance across window
        temp_var = float(np.mean(np.var(window, axis=0)))

        # 2. PCA dominant component
        centered = window - np.mean(window, axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            pc1 = np.dot(centered, vh[0])
        except Exception:
            pc1 = np.mean(centered, axis=1)

        # 3. Welch Power Spectral Density
        freqs, psd = signal.welch(pc1, fs=self.fs, nperseg=min(len(pc1), 64))

        # 4. Doppler Sub-Band Energies
        # 0-5 Hz (static posture, breathing, slow drift)
        idx_0_5 = np.where((freqs >= 0.5) & (freqs < 5.0))[0]
        # 5-15 Hz (normal walking, arm swinging)
        idx_5_15 = np.where((freqs >= 5.0) & (freqs < 15.0))[0]
        # 15-25 Hz (moderate movement, quick sitting)
        idx_15_25 = np.where((freqs >= 15.0) & (freqs < 25.0))[0]
        # 25-40 Hz (rapid vertical descent / impact kinetic burst)
        idx_25_40 = np.where((freqs >= 25.0) & (freqs <= 40.0))[0]

        e_0_5 = float(np.sum(psd[idx_0_5])) if len(idx_0_5) > 0 else 1e-6
        e_5_15 = float(np.sum(psd[idx_5_15])) if len(idx_5_15) > 0 else 1e-6
        e_15_25 = float(np.sum(psd[idx_15_25])) if len(idx_15_25) > 0 else 1e-6
        e_25_40 = float(np.sum(psd[idx_25_40])) if len(idx_25_40) > 0 else 1e-6

        # Ratio of rapid kinetic energy to baseline motion energy
        high_low = (e_15_25 + e_25_40) / max(e_0_5 + e_5_15, 1e-6)

        # 5. Dominant velocity
        peak_idx = np.argmax(psd)
        dom_freq = float(freqs[peak_idx])
        dom_velocity = (self.wavelength * dom_freq) / 2.0

        # 6. Total energy surge
        tot_energy = float(np.sum(psd))
        surge = tot_energy / max(baseline_energy, 1e-6)

        # 7. Spectral Entropy (measure of frequency dispersion)
        psd_norm = psd / max(np.sum(psd), 1e-9)
        spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

        return MLFeatures(
            subband_energy_0_5hz=round(e_0_5, 4),
            subband_energy_5_15hz=round(e_5_15, 4),
            subband_energy_15_25hz=round(e_15_25, 4),
            subband_energy_25_40hz=round(e_25_40, 4),
            high_low_ratio=round(high_low, 3),
            dominant_velocity=round(dom_velocity, 2),
            energy_surge=round(surge, 2),
            temporal_variance=round(temp_var, 4),
            spectral_entropy=round(spectral_entropy, 3),
        )


class FallClassifier:
    """Supervised Machine Learning classifier predicting fall probability."""

    def __init__(self, model_path: Optional[Path] = None):
        self.extractor = CSIFeatureExtractor()
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
        )
        self.is_fitted = False

        if model_path and model_path.exists():
            self.load(model_path)
        else:
            self._init_baseline_model()

    @property
    def is_calibrated(self) -> bool:
        """Return True if model employs Platt probability calibration."""
        return isinstance(self.model, CalibratedClassifierCV)

    def _init_baseline_model(self):
        """Train on a robust synthetic distribution of typical falls vs ADLs (Activities of Daily Living)."""
        np.random.seed(42)
        n_adl = 300
        n_fall = 200

        # Normal activities (ADLs: walking, sitting, bending)
        # Low high-to-low ratio, velocity 0.2 - 1.2 m/s, surge 1.0 - 2.5
        adl_0_5 = np.random.uniform(5.0, 25.0, n_adl)
        adl_5_15 = np.random.uniform(3.0, 15.0, n_adl)
        adl_15_25 = np.random.uniform(0.1, 2.0, n_adl)
        adl_25_40 = np.random.uniform(0.01, 0.5, n_adl)
        adl_hl = (adl_15_25 + adl_25_40) / (adl_0_5 + adl_5_15)
        adl_v = np.random.uniform(0.2, 1.3, n_adl)
        adl_surge = np.random.uniform(1.0, 2.8, n_adl)
        adl_var = np.random.uniform(0.05, 0.4, n_adl)
        adl_ent = np.random.uniform(2.5, 4.5, n_adl)
        X_adl = np.column_stack([adl_0_5, adl_5_15, adl_15_25, adl_25_40, adl_hl, adl_v, adl_surge, adl_var, adl_ent])

        # Falls (rapid descent, sudden surge in 20-35 Hz, high Doppler)
        fall_0_5 = np.random.uniform(2.0, 10.0, n_fall)
        fall_5_15 = np.random.uniform(5.0, 18.0, n_fall)
        fall_15_25 = np.random.uniform(8.0, 30.0, n_fall)
        fall_25_40 = np.random.uniform(10.0, 45.0, n_fall)
        fall_hl = (fall_15_25 + fall_25_40) / (fall_0_5 + fall_5_15)
        fall_v = np.random.uniform(1.85, 3.2, n_fall)
        fall_surge = np.random.uniform(3.5, 12.0, n_fall)
        fall_var = np.random.uniform(0.8, 3.5, n_fall)
        fall_ent = np.random.uniform(4.0, 5.8, n_fall)
        X_fall = np.column_stack([fall_0_5, fall_5_15, fall_15_25, fall_25_40, fall_hl, fall_v, fall_surge, fall_var, fall_ent])

        X = np.vstack([X_adl, X_fall])
        y = np.array([0] * n_adl + [1] * n_fall)

        self.model.fit(X, y)
        calibrated = CalibratedClassifierCV(self.model, method="sigmoid", cv=3)
        calibrated.fit(X, y)
        self.model = calibrated
        self.is_fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray, calibrate: bool = True):
        """Train the classifier on empirical feature matrices."""
        base_model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
        )
        if calibrate:
            calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
            calibrated.fit(X, y)
            self.model = calibrated
        else:
            base_model.fit(X, y)
            self.model = base_model
        self.is_fitted = True

    def predict_proba(self, features: Union[MLFeatures, np.ndarray]) -> float:
        """Return probability P(fall) in range [0.0, 1.0]."""
        if not self.is_fitted:
            return 0.0

        if isinstance(features, MLFeatures):
            feat_arr = features.to_array().reshape(1, -1)
        else:
            feat_arr = features.reshape(1, -1)

        probas = self.model.predict_proba(feat_arr)[0]
        # Class 1 is Fall
        return float(probas[1])

    def save(self, file_path: Path):
        """Persist model state to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: Path):
        """Load trained model state from file."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
