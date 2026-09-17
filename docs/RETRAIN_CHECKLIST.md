# Model Retrain Checklist — SOP-CLIN-FD-002

This document governs the process for retraining the fall detection classifier
on empirical human-subject data collected per **SOP-CLIN-FD-001**
(`docs/DATA_COLLECTION_PROTOCOL.md`).

---

## Prerequisites

- [ ] At least **15 subjects** enrolled per SOP-CLIN-FD-001
- [ ] At least **270 labeled sessions** in `datasets/trials/` (≥6 fall types × ≥3 room configs per subject)
- [ ] Each `.npz` file accompanied by a `.json` metadata sidecar (label, subject_id, room_id, notes)
- [ ] `hub/train.py` tests pass: `pytest -v tests/test_train_pipeline.py`

---

## Phase A — Data Validation

```bash
# List collected sessions
python -c "from hub.train import load_all_datasets; from pathlib import Path; X,y = load_all_datasets(Path('datasets/trials')); print(f'{len(y)} samples: {sum(y==1)} falls, {sum(y==0)} ADLs')"
```

- [ ] Confirm ≥ 270 labeled samples with balanced fall / ADL classes (target ratio ≤ 2:1)
- [ ] Inspect class balance; if falls < 40% of corpus, add synthetic augmentation via `--mix-ratio`

---

## Phase B — Cross-Validation Run (Trial Data Only)

```bash
python hub/train.py \
  --data-dir datasets/trials/ \
  --output-dir models/v2/ \
  --n-splits 5
```

Inspect `models/v2/evaluation_report.json`. Minimum thresholds before promotion:

| Metric | Minimum Required |
|---|---|
| Sensitivity (Recall) | ≥ 98.5% |
| Specificity | ≥ 98.0% |
| ROC-AUC | ≥ 0.995 |
| PR-AUC | ≥ 0.990 |

- [ ] All metrics meet targets → proceed to Phase C
- [ ] Any metric misses target → run feature engineering loop (see below)

### Feature Engineering Loop (if metrics fail)
1. Add `link3_raw_variance` as a 10th feature dimension to `MLFeatures`
2. Try window sizes: 32, 64, 128 samples
3. Re-run cross-validation until metrics pass
4. Document changes in `SYSTEM_AUDIT_AND_FINDINGS.md`

---

## Phase C — Mixed Real + Synthetic Training

```bash
python hub/train.py \
  --data-dir datasets/trials/ \
  --output-dir models/v2/ \
  --mix-ratio 0.7 \
  --onnx
```

- [ ] `models/v2/fall_classifier.pkl` created
- [ ] `models/v2/fall_classifier.onnx` created
- [ ] `models/v2/fall_classifier_int8.onnx` created (requires skl2onnx + onnxruntime)
- [ ] `models/v2/evaluation_report.json` includes ONNX latency metrics

---

## Phase D — Model Promotion

```bash
# Promote v2 model to production
cp models/v2/fall_classifier.pkl models/fall_classifier.pkl
cp models/v2/evaluation_report.json models/evaluation_report.json
# Optionally promote ONNX models:
cp models/v2/fall_classifier.onnx models/fall_classifier.onnx
cp models/v2/fall_classifier_int8.onnx models/fall_classifier_int8.onnx
```

- [ ] Run full test suite to confirm no regressions:
  ```bash
  pytest -v tests/ --tb=short
  ```
- [ ] All tests pass (≥ 80 passed, 0 failed)

---

## Phase E — Git Tag & Release

```bash
# Commit model files and report
git add models/ SYSTEM_AUDIT_AND_FINDINGS.md
git commit -m "feat: promote empirical trial v2 model (sensitivity=XX.X%, specificity=XX.X%)"

# Tag and push
make release TAG=v3.1.0
```

- [ ] Update `SYSTEM_AUDIT_AND_FINDINGS.md` Section 11 (revision table) with new metrics
- [ ] Update `README.md` feature list with empirical trial numbers
- [ ] Notify clinical team of model promotion

---

## Rollback Procedure

If the promoted model shows degraded performance in production:

```bash
git checkout v3.0.0 -- models/fall_classifier.pkl models/evaluation_report.json
git commit -m "revert: roll back to synthetic-baseline model v3.0.0"
git push origin main
```

Investigate discrepancy between CV metrics and production performance before re-promoting.
