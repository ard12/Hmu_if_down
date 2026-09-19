"""Second-stage ML classifier for categorizing confirmed falls into clinical fall types."""

from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class FallTypeClassifier:
    """Second-stage classifier that categorizes confirmed falls into distinct clinical phenotypes."""

    FALL_TYPES = [
        "forward_trip",
        "backward_slip",
        "lateral_collapse",
        "slow_slump",
        "syncope_drop",
    ]

    ALERT_PRIORITIES = {
        "forward_trip": "HIGH",
        "backward_slip": "HIGH",
        "lateral_collapse": "MEDIUM",
        "slow_slump": "MEDIUM",
        "syncope_drop": "HIGH",
    }

    def __init__(self, model_path: Optional[Path] = None):
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

    def generate_synthetic_features(
        self, n_per_class: int = 100, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 11-dimensional synthetic kinematic features for each fall type.

        Features:
            0: subband_energy_0_5hz
            1: subband_energy_5_15hz
            2: subband_energy_15_25hz
            3: subband_energy_25_40hz
            4: high_low_ratio
            5: dominant_velocity
            6: energy_surge
            7: temporal_variance
            8: spectral_entropy
            9: radar_dwell_time
            10: peak_velocity
        """
        rng = np.random.default_rng(random_state)
        X_list = []
        y_list = []

        # 0: forward_trip - high horizontal velocity, rapid onset, moderate dwell
        X_trip = np.column_stack([
            rng.uniform(2.0, 6.0, n_per_class),     # 0-5Hz
            rng.uniform(8.0, 16.0, n_per_class),    # 5-15Hz
            rng.uniform(15.0, 30.0, n_per_class),   # 15-25Hz
            rng.uniform(15.0, 35.0, n_per_class),   # 25-40Hz
            rng.uniform(3.0, 8.0, n_per_class),     # high_low_ratio
            rng.uniform(2.4, 3.5, n_per_class),     # dominant_velocity
            rng.uniform(5.0, 12.0, n_per_class),    # surge
            rng.uniform(1.2, 3.0, n_per_class),     # variance
            rng.uniform(4.5, 5.8, n_per_class),     # entropy
            rng.uniform(2.0, 8.0, n_per_class),     # dwell
            rng.uniform(2.8, 4.0, n_per_class),     # peak_vel
        ])
        X_list.append(X_trip)
        y_list.extend([0] * n_per_class)

        # 1: backward_slip - high velocity, sudden backward acceleration, high surge
        X_slip = np.column_stack([
            rng.uniform(1.5, 5.0, n_per_class),
            rng.uniform(10.0, 20.0, n_per_class),
            rng.uniform(20.0, 40.0, n_per_class),
            rng.uniform(25.0, 50.0, n_per_class),
            rng.uniform(4.0, 12.0, n_per_class),
            rng.uniform(2.6, 3.8, n_per_class),
            rng.uniform(7.0, 15.0, n_per_class),
            rng.uniform(1.8, 4.0, n_per_class),
            rng.uniform(4.8, 6.0, n_per_class),
            rng.uniform(3.0, 10.0, n_per_class),
            rng.uniform(3.0, 4.5, n_per_class),
        ])
        X_list.append(X_slip)
        y_list.extend([1] * n_per_class)

        # 2: lateral_collapse - moderate velocity, wide azimuth dispersion, moderate variance
        X_lat = np.column_stack([
            rng.uniform(3.0, 8.0, n_per_class),
            rng.uniform(6.0, 14.0, n_per_class),
            rng.uniform(10.0, 25.0, n_per_class),
            rng.uniform(10.0, 25.0, n_per_class),
            rng.uniform(2.0, 5.0, n_per_class),
            rng.uniform(1.8, 2.7, n_per_class),
            rng.uniform(4.0, 8.0, n_per_class),
            rng.uniform(1.0, 2.5, n_per_class),
            rng.uniform(4.0, 5.2, n_per_class),
            rng.uniform(2.0, 9.0, n_per_class),
            rng.uniform(2.0, 3.0, n_per_class),
        ])
        X_list.append(X_lat)
        y_list.extend([2] * n_per_class)

        # 3: slow_slump - low velocity, gradual descent, low surge
        X_slump = np.column_stack([
            rng.uniform(5.0, 12.0, n_per_class),
            rng.uniform(3.0, 8.0, n_per_class),
            rng.uniform(2.0, 7.0, n_per_class),
            rng.uniform(1.0, 5.0, n_per_class),
            rng.uniform(0.3, 1.2, n_per_class),
            rng.uniform(0.5, 1.4, n_per_class),
            rng.uniform(2.0, 4.5, n_per_class),
            rng.uniform(0.3, 1.0, n_per_class),
            rng.uniform(3.0, 4.2, n_per_class),
            rng.uniform(5.0, 20.0, n_per_class),
            rng.uniform(0.8, 1.6, n_per_class),
        ])
        X_list.append(X_slump)
        y_list.extend([3] * n_per_class)

        # 4: syncope_drop - sudden loss of posture with near-zero initial warning velocity
        X_syncope = np.column_stack([
            rng.uniform(0.5, 3.0, n_per_class),
            rng.uniform(2.0, 6.0, n_per_class),
            rng.uniform(8.0, 18.0, n_per_class),
            rng.uniform(12.0, 28.0, n_per_class),
            rng.uniform(2.0, 6.0, n_per_class),
            rng.uniform(1.4, 2.2, n_per_class),
            rng.uniform(3.0, 6.5, n_per_class),
            rng.uniform(0.8, 2.0, n_per_class),
            rng.uniform(2.2, 3.8, n_per_class),     # lower spectral entropy
            rng.uniform(8.0, 30.0, n_per_class),    # prolonged unconscious dwell
            rng.uniform(1.5, 2.5, n_per_class),
        ])
        X_list.append(X_syncope)
        y_list.extend([4] * n_per_class)

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)
        return X, y

    def _init_baseline_model(self):
        """Train baseline model on synthetic fall phenotypes."""
        X, y = self.generate_synthetic_features(n_per_class=80)
        self.model.fit(X, y)
        self.is_fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model on feature matrix and labels."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, features_11d: Sequence[float]) -> Tuple[str, float]:
        """Classify 11-D feature vector into fall type and return (label, confidence)."""
        if not self.is_fitted:
            return self.FALL_TYPES[0], 0.0

        feat_arr = np.asarray(features_11d, dtype=np.float32).reshape(1, -1)
        probas = self.model.predict_proba(feat_arr)[0]
        best_idx = int(np.argmax(probas))
        conf = float(probas[best_idx])
        label = self.FALL_TYPES[best_idx]
        return label, conf

    def train_and_export(self, X: np.ndarray, y: np.ndarray, output_path: Path) -> Dict[str, Any]:
        """Train model and persist to disk with metadata report."""
        self.fit(X, y)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)

        # Accuracy on training set
        train_acc = float(np.mean(self.model.predict(X) == y))
        report = {
            "model_path": str(output_path),
            "num_samples": len(y),
            "num_classes": len(self.FALL_TYPES),
            "train_accuracy": round(train_acc, 4),
            "fall_types": self.FALL_TYPES,
        }
        return report

    def save(self, file_path: Path):
        """Save model to pickle file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: Path):
        """Load model from pickle file."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
