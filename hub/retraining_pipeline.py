"""Orchestrates model drift detection, retraining, evaluation, and safe A/B swapping."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from hub.drift_detector import DriftDetector
from hub.model_registry import ModelRegistry
from hub.training_buffer import InsufficientDataError, TrainingBuffer


class RetrainingPipeline:
    """Orchestrates: detect drift -> gather buffer -> fit -> evaluate

    -> swap if quality OK -> rollback if sensitivity < 98.5%.
    """

    SENSITIVITY_FLOOR = 0.985
    SPECIFICITY_FLOOR = 0.970

    def __init__(
        self,
        registry: ModelRegistry,
        buffer: TrainingBuffer,
        detector: DriftDetector,
        audit_log: Optional[Any] = None,
    ):
        self.registry = registry
        self.buffer = buffer
        self.detector = detector
        self.audit_log = audit_log
        self.last_result: Optional[Dict[str, Any]] = None

    def _do_retrain(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit a calibrated classifier on training data."""
        base_clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        unique_classes = np.unique(y)

        if len(unique_classes) > 1:
            min_class_count = int(np.min(np.bincount(y)))
            cv_folds = min(3, min_class_count)
            if cv_folds >= 2:
                try:
                    clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=cv_folds)
                except TypeError:
                    clf = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv=cv_folds)
            else:
                clf = base_clf
        else:
            clf = base_clf

        clf.fit(X, y)
        return clf

    def run(self) -> Dict[str, Any]:
        """Execute retraining evaluation workflow."""
        drift = self.detector.drift_status()
        if drift.get("status") != "RETRAIN_REQUIRED":
            self.last_result = {
                "status": "NO_DRIFT",
                "new_version_id": None,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "brier": 0.0,
                "reason": "No concept drift detected",
            }
            return self.last_result

        try:
            X, y = self.buffer.get_batch(min_positives=20)
        except InsufficientDataError as e:
            self.last_result = {
                "status": "RETRAIN_FAILED",
                "new_version_id": None,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "brier": 0.0,
                "reason": str(e),
            }
            return self.last_result

        # Split held-out evaluation slice (20%)
        strat = y if (len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )

        new_clf = self._do_retrain(X_train, y_train)

        # Evaluate on validation slice
        if hasattr(new_clf, "predict_proba"):
            probs = new_clf.predict_proba(X_val)
            if probs.shape[1] > 1:
                p1 = probs[:, 1]
            else:
                p1 = probs[:, 0]
        else:
            p1 = new_clf.predict(X_val).astype(float)

        preds = (p1 >= 0.5).astype(int)

        tp = np.sum((preds == 1) & (y_val == 1))
        fn = np.sum((preds == 0) & (y_val == 1))
        tn = np.sum((preds == 0) & (y_val == 0))
        fp = np.sum((preds == 1) & (y_val == 0))

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        brier = float(np.mean((p1 - y_val) ** 2))

        # Check safety floors
        if sensitivity >= self.SENSITIVITY_FLOOR and specificity >= self.SPECIFICITY_FLOOR:
            version_id = self.registry.save_model(
                new_clf,
                metadata={
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "brier": brier,
                    "training_n": len(X),
                    "algorithm": type(new_clf).__name__,
                    "active": True,
                    "notes": "Adaptive retraining auto-swap",
                },
            )
            self.registry.set_active(version_id)
            self.detector.reset_reference()

            if self.audit_log is not None and hasattr(self.audit_log, "append"):
                try:
                    self.audit_log.append(
                        "RETRAIN_SUCCESS",
                        {
                            "version_id": version_id,
                            "sensitivity": sensitivity,
                            "specificity": specificity,
                            "brier": brier,
                        },
                    )
                except Exception:
                    pass

            self.last_result = {
                "status": "RETRAINED",
                "new_version_id": version_id,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "brier": brier,
                "reason": "Model retrained and verified above clinical safety floor",
            }
            return self.last_result
        else:
            # Rollback: do not activate new model
            reason = f"Sensitivity {sensitivity:.4f} < {self.SENSITIVITY_FLOOR} or Specificity {specificity:.4f} < {self.SPECIFICITY_FLOOR}"
            if self.audit_log is not None and hasattr(self.audit_log, "append"):
                try:
                    self.audit_log.append(
                        "RETRAIN_ROLLED_BACK",
                        {
                            "sensitivity": sensitivity,
                            "specificity": specificity,
                            "reason": reason,
                        },
                    )
                except Exception:
                    pass

            self.last_result = {
                "status": "ROLLED_BACK",
                "new_version_id": None,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "brier": brier,
                "reason": reason,
            }
            return self.last_result
