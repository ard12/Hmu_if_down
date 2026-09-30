"""Empirical Dataset Training & Evaluation Pipeline for CSI Fall Detection.

Loads recorded multi-modal trial datasets (.npz) or generates verified synthetic
benchmark distributions, computes 9-dimensional kinematic feature representations,
performs Stratified K-Fold Cross-Validation, evaluates ROC/PR metrics, and exports
production classifier models to disk.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from hub.csi_pipeline.classifier import CSIFeatureExtractor, FallClassifier, MLFeatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSI Fall Detection Empirical Training Pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets",
        help="Path to directory containing .npz dataset sessions (default: datasets/)",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Path to save trained classifier model (overrides --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory for model and report outputs (default: models/)",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Path to save cross-validation metrics JSON (overrides --output-dir)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for Stratified K-Fold Cross Validation (default: 5)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force generation of synthetic benchmark corpus if real datasets are sparse",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=600,
        help="Number of synthetic feature samples to generate (default: 600)",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export ONNX FP32 and INT8-quantized models after training (requires skl2onnx + onnxruntime)",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.7,
        help="Fraction of real-data samples when blending real + synthetic (default: 0.7)",
    )
    parser.add_argument(
        "--fall-type",
        action="store_true",
        help="Train second-stage FallTypeClassifier model on synthetic fall phenotypes",
    )
    return parser.parse_args()


def generate_synthetic_features(
    n_samples: int = 600,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced, physically calibrated feature distributions for ADLs and falls."""
    rng = np.random.RandomState(random_state)
    n_adl = n_samples // 2
    n_fall = n_samples - n_adl

    # 1. Activities of Daily Living (ADLs: walking, slow sitting, bending, pet ground noise)
    adl_0_5 = rng.uniform(5.0, 30.0, n_adl)
    adl_5_15 = rng.uniform(3.0, 16.0, n_adl)
    adl_15_25 = rng.uniform(0.1, 2.5, n_adl)
    adl_25_40 = rng.uniform(0.01, 0.6, n_adl)
    adl_hl = (adl_15_25 + adl_25_40) / (adl_0_5 + adl_5_15 + 1e-6)
    adl_v = rng.uniform(0.1, 1.4, n_adl)
    adl_surge = rng.uniform(1.0, 2.8, n_adl)
    adl_var = rng.uniform(0.02, 0.45, n_adl)
    adl_ent = rng.uniform(2.2, 4.4, n_adl)
    X_adl = np.column_stack([adl_0_5, adl_5_15, adl_15_25, adl_25_40, adl_hl, adl_v, adl_surge, adl_var, adl_ent])

    # 2. Human Fall Signatures (hard impact, forward trip, backward slip, slump)
    fall_0_5 = rng.uniform(2.0, 12.0, n_fall)
    fall_5_15 = rng.uniform(5.0, 20.0, n_fall)
    fall_15_25 = rng.uniform(8.0, 35.0, n_fall)
    fall_25_40 = rng.uniform(12.0, 50.0, n_fall)
    fall_hl = (fall_15_25 + fall_25_40) / (fall_0_5 + fall_5_15 + 1e-6)
    fall_v = rng.uniform(1.80, 3.4, n_fall)
    fall_surge = rng.uniform(3.8, 14.0, n_fall)
    fall_var = rng.uniform(0.75, 4.0, n_fall)
    fall_ent = rng.uniform(4.0, 5.9, n_fall)
    X_fall = np.column_stack([fall_0_5, fall_5_15, fall_15_25, fall_25_40, fall_hl, fall_v, fall_surge, fall_var, fall_ent])

    X = np.vstack([X_adl, X_fall])
    y = np.array([0] * n_adl + [1] * n_fall, dtype=np.int32)

    # Shuffle dataset
    indices = rng.permutation(len(y))
    return X[indices], y[indices]


def extract_features_from_npz(
    npz_path: Path,
    extractor: Optional[CSIFeatureExtractor] = None,
    window_len: int = 100,
    stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sliding window ML feature vectors from a recorded session file."""
    if extractor is None:
        extractor = CSIFeatureExtractor()

    data = np.load(npz_path, allow_pickle=True)
    amplitudes = data.get("csi_amplitudes")
    if amplitudes is None or len(amplitudes) < 16:
        return np.empty((0, 9)), np.empty((0,), dtype=np.int32)

    # Determine ground-truth label from filename or metadata JSON
    label_str = npz_path.stem.lower()
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                label_str = meta.get("label", label_str).lower()
        except Exception:
            pass

    is_fall_session = ("fall" in label_str) or ("slump" in label_str)

    # Baseline quiet energy from the first 50 samples
    baseline_energy = float(np.mean(np.var(amplitudes[: min(50, len(amplitudes))], axis=0)))
    baseline_energy = max(baseline_energy, 1e-4)

    features_list = []
    labels_list = []

    # Slide window
    num_samples = amplitudes.shape[0]
    for start_idx in range(0, num_samples - window_len + 1, stride):
        window = amplitudes[start_idx : start_idx + window_len]
        feat = extractor.extract_from_window(window, baseline_energy=baseline_energy)
        feat_vec = feat.to_array()

        # For fall sessions, classify high-energy or velocity bursts as fall windows
        if is_fall_session:
            # If window exhibits rapid velocity or surge, label as 1; else pre/post-impact 0
            if feat.dominant_velocity >= 1.5 or feat.energy_surge >= 3.0 or feat.high_low_ratio >= 1.2:
                win_label = 1
            else:
                win_label = 0
        else:
            win_label = 0

        features_list.append(feat_vec)
        labels_list.append(win_label)

    if not features_list:
        return np.empty((0, 9)), np.empty((0,), dtype=np.int32)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


def load_all_datasets(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all .npz dataset files from data_dir and compile feature matrix."""
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        return np.empty((0, 9)), np.empty((0,), dtype=np.int32)

    extractor = CSIFeatureExtractor()
    all_x = []
    all_y = []

    for f in npz_files:
        x_sub, y_sub = extract_features_from_npz(f, extractor=extractor)
        if len(x_sub) > 0:
            all_x.append(x_sub)
            all_y.append(y_sub)

    if not all_x:
        return np.empty((0, 9)), np.empty((0,), dtype=np.int32)

    return np.vstack(all_x), np.concatenate(all_y)


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform Stratified K-Fold Cross Validation and calculate comprehensive metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_list = []
    prec_list = []
    rec_list = []
    spec_list = []
    f1_list = []
    roc_auc_list = []
    pr_auc_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state + fold_idx,
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        roc_auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 1.0
        pr_auc = average_precision_score(y_val, probs) if len(np.unique(y_val)) > 1 else 1.0

        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        spec = float(tn) / max(float(tn + fp), 1.0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        spec_list.append(spec)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)
        pr_auc_list.append(pr_auc)

    results = {
        "n_samples": int(len(y)),
        "n_falls": int(np.sum(y == 1)),
        "n_adls": int(np.sum(y == 0)),
        "n_splits": n_splits,
        "metrics": {
            "accuracy": {"mean": float(np.mean(acc_list)), "std": float(np.std(acc_list))},
            "precision": {"mean": float(np.mean(prec_list)), "std": float(np.std(prec_list))},
            "recall_sensitivity": {"mean": float(np.mean(rec_list)), "std": float(np.std(rec_list))},
            "specificity": {"mean": float(np.mean(spec_list)), "std": float(np.std(spec_list))},
            "f1_score": {"mean": float(np.mean(f1_list)), "std": float(np.std(f1_list))},
            "roc_auc": {"mean": float(np.mean(roc_auc_list)), "std": float(np.std(roc_auc_list))},
            "pr_auc": {"mean": float(np.mean(pr_auc_list)), "std": float(np.std(pr_auc_list))},
        },
        "per_fold": {
            "accuracy": [round(v, 4) for v in acc_list],
            "recall": [round(v, 4) for v in rec_list],
            "specificity": [round(v, 4) for v in spec_list],
            "f1": [round(v, 4) for v in f1_list],
            "roc_auc": [round(v, 4) for v in roc_auc_list],
        },
    }
    return results


def train_and_export(
    X: np.ndarray,
    y: np.ndarray,
    output_model_path: Path,
    report_file_path: Optional[Path] = None,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Evaluate via cross-validation, train final model on full dataset, and persist artifacts."""
    print(f"Running {n_splits}-Fold Stratified Cross Validation on {len(y)} samples...")
    cv_report = cross_validate_model(X, y, n_splits=n_splits)

    m = cv_report["metrics"]
    print("\n---------------- Cross-Validation Results ----------------")
    print(f"  Accuracy:    {m['accuracy']['mean']:.4f} (+/- {m['accuracy']['std']:.4f})")
    print(f"  Sensitivity: {m['recall_sensitivity']['mean']:.4f} (+/- {m['recall_sensitivity']['std']:.4f})")
    print(f"  Specificity: {m['specificity']['mean']:.4f} (+/- {m['specificity']['std']:.4f})")
    print(f"  F1 Score:    {m['f1_score']['mean']:.4f} (+/- {m['f1_score']['std']:.4f})")
    print(f"  ROC-AUC:     {m['roc_auc']['mean']:.4f} (+/- {m['roc_auc']['std']:.4f})")
    print(f"  PR-AUC:      {m['pr_auc']['mean']:.4f} (+/- {m['pr_auc']['std']:.4f})")
    print("----------------------------------------------------------\n")

    # Fit final model on full dataset
    final_model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
    )
    final_model.fit(X, y)

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    # Calibrate probability using Platt scaling (sigmoid)
    calibrated_model = CalibratedClassifierCV(final_model, method="sigmoid", cv=3)
    calibrated_model.fit(X, y)

    probas = calibrated_model.predict_proba(X)[:, 1]
    brier = float(brier_score_loss(y, probas))
    cv_report["brier_score"] = round(brier, 4)
    cv_report["calibrated"] = True

    # Persist model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"[OK] Trained production model exported to: {output_model_path}")

    # Persist calibrated model
    calibrated_path = output_model_path.with_name(output_model_path.stem + "_calibrated.pkl")
    with open(calibrated_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    print(f"[OK] Calibrated model exported to: {calibrated_path}")

    # Persist report
    if report_file_path:
        report_file_path.parent.mkdir(parents=True, exist_ok=True)
        cv_report["trained_at"] = datetime.now().isoformat()
        cv_report["model_file"] = output_model_path.name
        with open(report_file_path, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)
        print(f"[OK] Evaluation report saved to: {report_file_path}")

    return cv_report


def convert_and_quantize(model, output_dir: Path) -> Dict[str, Any]:
    """Export the trained sklearn model to ONNX FP32 and INT8-quantized formats.

    Requires: skl2onnx>=1.16 and onnxruntime>=1.18

    Args:
        model: Fitted scikit-learn estimator.
        output_dir: Directory where .onnx files will be saved.

    Returns:
        Dict with onnx_path, int8_path, onnx_size_kb, int8_size_kb, and
        benchmark latency metrics (mean_ms, p99_ms) for both variants.
    """
    result: Dict[str, Any] = {}
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("[WARN] skl2onnx not installed — ONNX export skipped.")
        print("       Install with: pip install skl2onnx>=1.16 onnxruntime>=1.18")
        return result

    try:
        import onnx  # noqa: F401
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[WARN] onnxruntime not installed — ONNX export skipped.")
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "fall_classifier.onnx"
    int8_path = output_dir / "fall_classifier_int8.onnx"

    # FP32 conversion
    initial_type = [("float_input", FloatTensorType([None, 9]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[OK] FP32 ONNX model exported to: {onnx_path}")

    # INT8 quantization
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
    print(f"[OK] INT8 quantized ONNX exported to: {int8_path}")

    result["onnx_path"] = str(onnx_path)
    result["int8_path"] = str(int8_path)
    result["onnx_size_kb"] = round(onnx_path.stat().st_size / 1024, 1)
    result["int8_size_kb"] = round(int8_path.stat().st_size / 1024, 1)

    # Benchmark both variants using the ONNX runner
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from hub.onnx_runner import ONNXFallClassifier

        for variant_name, model_path in [("onnx_fp32", onnx_path), ("onnx_int8", int8_path)]:
            runner = ONNXFallClassifier(model_path=model_path)
            bench = runner.benchmark_inference(n=500)
            result[f"{variant_name}_mean_ms"] = bench["mean_ms"]
            result[f"{variant_name}_p99_ms"] = bench["p99_ms"]
            print(
                f"[OK] {variant_name} benchmark: mean={bench['mean_ms']:.3f}ms "
                f"p99={bench['p99_ms']:.3f}ms"
            )
    except Exception as exc:
        print(f"[WARN] ONNX benchmark failed: {exc}")

    return result


def main():
    args = parse_args()

    # Resolve output paths (explicit --output-model / --report-file override --output-dir)
    output_dir: Path = args.output_dir
    output_model = args.output_model or (output_dir / "fall_classifier.pkl")
    report_file = args.report_file or (output_dir / "evaluation_report.json")

    X, y = np.empty((0, 9)), np.empty((0,), dtype=np.int32)
    if not args.synthetic and args.data_dir.exists():
        print(f"Scanning '{args.data_dir}' for recorded sessions...")
        X, y = load_all_datasets(args.data_dir)
        print(f"Extracted {len(y)} samples from empirical recordings.")

    # Fallback to synthetic benchmark distribution if requested or no recordings exist
    if len(y) < 20 or args.synthetic:
        print(f"Generating {args.n_synthetic} synthetic feature distributions...")
        X_syn, y_syn = generate_synthetic_features(n_samples=args.n_synthetic)
        if len(y) > 0:
            # Mix real + synthetic at the specified ratio
            n_real = len(y)
            n_syn_target = max(1, int(n_real / args.mix_ratio * (1.0 - args.mix_ratio)))
            n_syn_actual = min(len(y_syn), n_syn_target)
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(y_syn), size=n_syn_actual, replace=False)
            X = np.vstack([X, X_syn[idx]])
            y = np.concatenate([y, y_syn[idx]])
            print(
                f"Mixed corpus: {n_real} real + {n_syn_actual} synthetic "
                f"(mix-ratio={args.mix_ratio})"
            )
        else:
            X, y = X_syn, y_syn

    print(f"Total training corpus: {len(y)} instances ({np.sum(y == 1)} falls, {np.sum(y == 0)} ADLs)")
    cv_report = train_and_export(
        X,
        y,
        output_model_path=output_model,
        report_file_path=report_file,
        n_splits=args.n_splits,
    )

    # Optional ONNX export
    if args.onnx:
        print("\n[ONNX] Starting model conversion and quantization...")
        # Re-load the saved model for ONNX conversion with integrity verification
        model_bytes = output_model.read_bytes()
        hash_file = output_model.with_suffix(output_model.suffix + ".sha256")
        if hash_file.exists():
            expected = hash_file.read_text(encoding="utf-8").strip()
            if hashlib.sha256(model_bytes).hexdigest() != expected:
                raise ValueError("Saved model failed cryptographic integrity check")
        saved_model = pickle.loads(model_bytes)  # nosec B301
        onnx_result = convert_and_quantize(saved_model, output_dir=output_dir)
        if onnx_result and report_file.exists():
            # Merge ONNX metadata back into evaluation report
            with open(report_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.update(onnx_result)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            print(f"[OK] ONNX metadata merged into: {report_file}")

    if args.fall_type:
        from hub.fall_type_classifier import FallTypeClassifier
        print("\n[FALL-TYPE] Training second-stage FallTypeClassifier...")
        ft_clf = FallTypeClassifier()
        X_ft, y_ft = ft_clf.generate_synthetic_features(n_per_class=100)
        ft_model_path = output_dir / "fall_type_classifier.pkl"
        ft_report = ft_clf.train_and_export(X_ft, y_ft, output_path=ft_model_path)
        ft_report_file = output_dir / "fall_type_classifier_report.json"
        with open(ft_report_file, "w", encoding="utf-8") as f:
            json.dump(ft_report, f, indent=2)
        print(f"[OK] Fall-type classifier exported to: {ft_model_path}")
        print(f"[OK] Fall-type report saved to: {ft_report_file}")


if __name__ == "__main__":
    main()
