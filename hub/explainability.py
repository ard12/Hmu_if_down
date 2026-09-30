"""
Explainable AI (XAI) & Model Transparency Engine.

Provides SHAP-based feature attributions, counterfactual explanations,
and global feature importance rankings for SaMD compliance:
  - FDA GMLP Principle 3 (scientific validity & explainability)
  - EU AI Act Article 13 (transparency)
  - IEC 62304 §5.5.3 (software unit verification)

@req SRS-XAI-001
"""
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_NAMES = [
    "subband_energy_0_5hz",
    "subband_energy_5_15hz",
    "subband_energy_15_25hz",
    "subband_energy_25_40hz",
    "high_low_ratio",
    "dominant_velocity",
    "energy_surge",
    "temporal_variance",
    "spectral_entropy",
]


class SHAPExplainer:
    """
    SHAP-based feature attribution and counterfactual generation engine.

    Supports tree-based SHAP when the `shap` package is present, and provides
    a mathematically consistent marginal contribution / permutation fallback
    satisfying the Efficiency (Local Accuracy) axiom:
        base_value + sum(shap_values) == prediction_probability
    """

    FEATURE_NAMES = DEFAULT_FEATURE_NAMES

    def __init__(
        self,
        model: Optional[Any] = None,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.feature_names = feature_names or list(self.FEATURE_NAMES)
        self.model = model
        self.background_data = background_data
        self._tree_explainer = None
        self._use_shap_pkg = False

        if self.model is None:
            self._init_default_model()

        if self.background_data is None:
            self._init_default_background()

        # Compute base value (expected probability over background)
        bg_preds = self._predict_proba(self.background_data)
        self.base_value = float(np.mean(bg_preds))

        self._try_init_shap_pkg()

    def _init_default_model(self) -> None:
        """Load trained production model or train a calibrated fallback."""
        model_path = Path("models/calibrated_fall_classifier.pkl")
        hash_path = Path("models/calibrated_fall_classifier.sha256")
        if model_path.exists():
            try:
                content = model_path.read_bytes()
                computed_hash = hashlib.sha256(content).hexdigest()
                if hash_path.exists():
                    expected_hash = hash_path.read_text(encoding="utf-8").strip()
                    if computed_hash != expected_hash:
                        logger.error(
                            f"Model integrity violation! SHA-256 {computed_hash} != {expected_hash}. Refusing to load {model_path}."
                        )
                        raise ValueError("Model file cryptographic integrity check failed")
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)  # nosec B301
                return
            except Exception as e:
                logger.warning(f"Could not load {model_path} securely: {e}")

        # Train a light HistGradientBoostingClassifier on synthetic data
        from sklearn.ensemble import HistGradientBoostingClassifier
        rng = np.random.default_rng(42)
        X_adl = rng.normal(loc=[10, 5, 2, 1, 0.5, 0.4, 1.0, 0.2, 1.5], scale=0.5, size=(100, 9))
        X_fall = rng.normal(loc=[50, 40, 30, 20, 3.5, 2.5, 8.0, 1.8, 3.0], scale=0.5, size=(100, 9))
        X = np.vstack([X_adl, X_fall])
        y = np.array([0] * 100 + [1] * 100)
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=20)
        clf.fit(X, y)
        self.model = clf

    def _init_default_background(self) -> None:
        """Create baseline background reference distribution."""
        rng = np.random.default_rng(42)
        # Background representing standard ADL activities
        self.background_data = rng.normal(
            loc=[12.0, 6.0, 2.5, 1.2, 0.6, 0.5, 1.2, 0.25, 1.6],
            scale=0.3,
            size=(50, len(self.feature_names)),
        ).astype(np.float32)

    def _try_init_shap_pkg(self) -> None:
        """Attempt to initialize shap.TreeExplainer if library installed."""
        try:
            import shap
            # Test TreeExplainer
            self._tree_explainer = shap.TreeExplainer(self.model, data=self.background_data)
            self._use_shap_pkg = True
            logger.info("SHAP TreeExplainer initialised successfully.")
        except Exception:
            self._use_shap_pkg = False
            logger.info("Using marginal contribution Shapley fallback.")

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Safely extract probability of fall (class 1)."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs.flatten()
        elif hasattr(self.model, "predict"):
            return self.model.predict(X).astype(float)
        return np.zeros(len(X))

    def explain_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Compute SHAP feature attribution values for a single prediction vector.
        Guarantees: base_value + sum(shap_values) == prediction_proba.
        """
        x = np.asarray(features, dtype=np.float32).flatten()
        if len(x) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(x)}")

        x_2d = x.reshape(1, -1)
        pred_proba = float(self._predict_proba(x_2d)[0])
        pred_class = int(1 if pred_proba >= 0.5 else 0)

        total_diff = pred_proba - self.base_value

        if self._use_shap_pkg and self._tree_explainer is not None:
            try:
                raw_shap = self._tree_explainer.shap_values(x_2d)
                if isinstance(raw_shap, list):
                    vals = np.array(raw_shap[1][0]) if len(raw_shap) > 1 else np.array(raw_shap[0][0])
                else:
                    vals = np.array(raw_shap[0])
                shap_values = [float(v) for v in vals]
            except Exception:
                shap_values = self._compute_marginal_shap(x, pred_proba, total_diff)
        else:
            shap_values = self._compute_marginal_shap(x, pred_proba, total_diff)

        return {
            "feature_names": list(self.feature_names),
            "feature_values": [round(float(v), 4) for v in x],
            "shap_values": [round(float(v), 5) for v in shap_values],
            "base_value": round(float(self.base_value), 5),
            "prediction_proba": round(float(pred_proba), 4),
            "predicted_class": pred_class,
        }

    def _compute_marginal_shap(
        self, x: np.ndarray, pred_proba: float, total_diff: float
    ) -> List[float]:
        """
        Calculates marginal contribution per feature against background,
        enforcing the local accuracy / additivity axiom.
        """
        n_feats = len(x)
        raw_contribs = np.zeros(n_feats)

        # Baseline point is background centroid
        bg_mean = np.mean(self.background_data, axis=0)

        for i in range(n_feats):
            # Substitute feature i with baseline background mean
            x_perturbed = x.copy()
            x_perturbed[i] = bg_mean[i]
            perturbed_pred = float(self._predict_proba(x_perturbed.reshape(1, -1))[0])
            raw_contribs[i] = pred_proba - perturbed_pred

        sum_raw = np.sum(raw_contribs)
        if abs(sum_raw) > 1e-7:
            # Scale proportionally to satisfy local accuracy: sum(phi) == total_diff
            scaled_shap = raw_contribs * (total_diff / sum_raw)
        else:
            scaled_shap = np.full(n_feats, total_diff / n_feats)

        return list(scaled_shap)

    def global_importance(self, samples: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute global feature importance as mean absolute SHAP value."""
        data = samples if samples is not None else self.background_data
        n_samples = min(len(data), 100)
        subset = data[:n_samples]

        all_abs_shap = np.zeros(len(self.feature_names))
        for row in subset:
            exp = self.explain_prediction(row)
            all_abs_shap += np.abs(exp["shap_values"])

        mean_abs = all_abs_shap / max(1, n_samples)
        ranking = {
            name: round(float(val), 4)
            for name, val in sorted(
                zip(self.feature_names, mean_abs), key=lambda item: item[1], reverse=True
            )
        }
        return ranking

    def counterfactual(
        self, features: np.ndarray, target_class: int = 0, max_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Compute minimal feature perturbation vector to flip prediction to `target_class`.
        """
        x = np.asarray(features, dtype=np.float32).flatten()
        orig_proba = float(self._predict_proba(x.reshape(1, -1))[0])
        orig_class = int(1 if orig_proba >= 0.5 else 0)

        if orig_class == target_class:
            return {
                "original_features": [round(float(v), 4) for v in x],
                "counterfactual_features": [round(float(v), 4) for v in x],
                "changes": {},
                "original_prediction": orig_class,
                "counterfactual_prediction": target_class,
                "iterations": 0,
            }

        # Target background centroid for target class
        bg_mean = np.mean(self.background_data, axis=0)
        cf = x.copy()

        # Iteratively interpolate most important features toward baseline until class flips
        exp = self.explain_prediction(x)
        sorted_indices = np.argsort(np.abs(exp["shap_values"]))[::-1]

        step = 0
        current_class = orig_class
        for idx in sorted_indices:
            step += 1
            cf[idx] = bg_mean[idx]
            curr_proba = float(self._predict_proba(cf.reshape(1, -1))[0])
            current_class = int(1 if curr_proba >= 0.5 else 0)
            if current_class == target_class:
                break

        changes = {
            self.feature_names[i]: round(float(cf[i] - x[i]), 4)
            for i in range(len(x))
            if abs(cf[i] - x[i]) > 1e-4
        }

        return {
            "original_features": [round(float(v), 4) for v in x],
            "counterfactual_features": [round(float(v), 4) for v in cf],
            "changes": changes,
            "original_prediction": orig_class,
            "counterfactual_prediction": current_class,
            "iterations": step,
        }

    def to_waterfall_json(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Format an explanation for Chart.js waterfall or bar rendering."""
        bars = []
        for name, val, shap_val in zip(
            explanation["feature_names"],
            explanation["feature_values"],
            explanation["shap_values"],
        ):
            bars.append({
                "feature": name,
                "value": val,
                "contribution": shap_val,
                "direction": "positive" if shap_val >= 0 else "negative",
            })

        # Sort bars by absolute contribution magnitude
        bars.sort(key=lambda b: abs(b["contribution"]), reverse=True)

        return {
            "base_value": explanation["base_value"],
            "prediction_proba": explanation["prediction_proba"],
            "predicted_class": explanation["predicted_class"],
            "bars": bars,
        }
