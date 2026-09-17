"""ONNX Runtime wrapper for the Fall Detection classifier (Milestone 6.2).

Provides a drop-in replacement for the scikit-learn FallClassifier that uses
ONNX Runtime for inference. Falls back to the pickle model if ONNX is unavailable.

Usage:
    from hub.onnx_runner import ONNXFallClassifier
    clf = ONNXFallClassifier()          # auto-discovers models/fall_classifier_int8.onnx
    label, confidence = clf.predict(features_9d)
    report = clf.benchmark_inference(n=1000)
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("onnx_runner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ONNX_INT8 = PROJECT_ROOT / "models" / "fall_classifier_int8.onnx"
_DEFAULT_ONNX_FP32 = PROJECT_ROOT / "models" / "fall_classifier.onnx"
_DEFAULT_PICKLE = PROJECT_ROOT / "models" / "fall_classifier.pkl"

FEATURE_DIM = 9  # Must match MLFeatures vector length


class ONNXFallClassifier:
    """ONNX Runtime inference wrapper for the fall detection classifier.

    Priority order:
      1. INT8 quantized ONNX (smallest, fastest)
      2. FP32 ONNX
      3. scikit-learn pickle fallback

    Args:
        model_path: Explicit path to a .onnx file. Auto-detected if None.
        fallback_pickle: Path to the sklearn pickle to fall back to.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        fallback_pickle: Optional[Path] = None,
    ):
        self._session = None
        self._fallback_model = None
        self._input_name: Optional[str] = None
        self._using_onnx = False

        # Resolve model paths
        onnx_path = Path(model_path) if model_path else None
        pickle_path = Path(fallback_pickle) if fallback_pickle else _DEFAULT_PICKLE

        # Try loading ONNX session
        if onnx_path is None:
            for candidate in (_DEFAULT_ONNX_INT8, _DEFAULT_ONNX_FP32):
                if candidate.exists():
                    onnx_path = candidate
                    break

        if onnx_path and onnx_path.exists():
            self._load_onnx(onnx_path)
        else:
            logger.info("No ONNX model found; falling back to pickle classifier.")

        # Load pickle fallback if ONNX not available
        if not self._using_onnx:
            self._load_pickle(pickle_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_onnx(self, path: Path) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._using_onnx = True
            logger.info("ONNX session loaded: %s", path.name)
        except ImportError:
            logger.warning(
                "onnxruntime not installed. "
                "Install with: pip install onnxruntime>=1.18"
            )
        except Exception as exc:
            logger.warning("Failed to load ONNX model (%s): %s", path, exc)

    def _load_pickle(self, path: Path) -> None:
        if not path.exists():
            logger.error("Pickle model not found at %s", path)
            return
        try:
            with open(path, "rb") as f:
                self._fallback_model = pickle.load(f)
            logger.info("Pickle classifier loaded from %s", path.name)
        except Exception as exc:
            logger.error("Failed to load pickle model: %s", exc)

    def _to_float32_array(self, features: Sequence[float]) -> np.ndarray:
        arr = np.array(features, dtype=np.float32).reshape(1, FEATURE_DIM)
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if at least one backend (ONNX or pickle) is loaded."""
        return self._using_onnx or (self._fallback_model is not None)

    def predict(self, features: Sequence[float]) -> Tuple[int, float]:
        """Classify a 9-dimensional feature vector.

        Returns:
            (label, confidence) where label is 1=fall / 0=ADL and
            confidence is the probability of the predicted class.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self.is_ready:
            raise RuntimeError("No classifier model loaded. Check model paths.")

        arr = self._to_float32_array(features)

        if self._using_onnx and self._session is not None:
            # ONNX Runtime inference
            outputs = self._session.run(None, {self._input_name: arr})
            # Typical skl2onnx output: [label_array, probability_map_list]
            label = int(outputs[0][0])
            try:
                prob_map = outputs[1][0]  # dict or list of dicts
                if isinstance(prob_map, dict):
                    confidence = float(prob_map.get(label, 0.5))
                else:
                    # prob_map is list-like [prob_class0, prob_class1]
                    confidence = float(prob_map[label])
            except Exception:
                confidence = 1.0 if label == 1 else 0.0
            return label, confidence

        # Pickle fallback
        proba = self._fallback_model.predict_proba(arr)[0]  # shape (2,)
        label = int(np.argmax(proba))
        confidence = float(proba[label])
        return label, confidence

    def predict_proba(self, features: Sequence[float]) -> List[float]:
        """Return [prob_ADL, prob_fall] probability pair."""
        if not self.is_ready:
            raise RuntimeError("No classifier model loaded.")

        arr = self._to_float32_array(features)

        if self._using_onnx and self._session is not None:
            outputs = self._session.run(None, {self._input_name: arr})
            try:
                prob_map = outputs[1][0]
                if isinstance(prob_map, dict):
                    return [float(prob_map.get(0, 0.5)), float(prob_map.get(1, 0.5))]
                return [float(v) for v in prob_map]
            except Exception:
                label = int(outputs[0][0])
                return [0.0, 1.0] if label == 1 else [1.0, 0.0]

        return list(self._fallback_model.predict_proba(arr)[0])

    def benchmark_inference(self, n: int = 1000) -> Dict[str, float]:
        """Run n inference calls on a random feature vector and report latency.

        Returns:
            Dict with keys: mean_ms, p50_ms, p95_ms, p99_ms, n_calls.
        """
        if not self.is_ready:
            raise RuntimeError("No classifier model loaded.")

        rng = np.random.default_rng(seed=0)
        sample = rng.uniform(0.0, 1.0, size=FEATURE_DIM).tolist()

        latencies_ms: List[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(sample)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        arr = np.array(latencies_ms)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "n_calls": n,
            "backend": "onnx" if self._using_onnx else "sklearn_pickle",
        }
