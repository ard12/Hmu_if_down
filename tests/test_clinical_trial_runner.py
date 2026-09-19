"""Tests for Clinical Trial Cohort Runner & Demographic Disparity (Milestone 10.2)."""

import pytest
from hub.clinical_trial_runner import (
    ClinicalTrialRunner,
    compute_cohens_kappa,
    wilson_score_interval,
)


def test_wilson_score_interval_properties():
    # 50 out of 50 successes -> interval should be close to 1.0, not exactly 1.0
    low, high = wilson_score_interval(50, 50)
    assert 0.90 <= low < 1.0
    assert abs(high - 1.0) < 1e-4

    # 0 out of 50 -> lower is 0.0, upper > 0.0
    low0, high0 = wilson_score_interval(0, 50)
    assert low0 == 0.0
    assert 0.0 < high0 <= 0.10

    # Symmetry at 50%
    l50, h50 = wilson_score_interval(25, 50)
    assert abs((l50 + h50) / 2.0 - 0.5) < 0.01


def test_cohens_kappa_perfect_agreement():
    # Perfect agreement
    kappa = compute_cohens_kappa(tp=50, fp=0, tn=50, fn=0)
    assert math_close(kappa, 1.0)


def test_cohens_kappa_chance_agreement():
    # Completely random / uncorrelated
    kappa = compute_cohens_kappa(tp=25, fp=25, tn=25, fn=25)
    assert abs(kappa) < 0.10


def math_close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-4


def test_cohort_features_generation_shape():
    runner = ClinicalTrialRunner(seed=123)
    for cohort in runner.COHORTS:
        X, y = runner.generate_cohort_features(cohort, n_sessions=40)
        assert X.shape == (40, 9)
        assert len(y) == 40
        assert sum(y == 1) == 20
        assert sum(y == 0) == 20


def test_clinical_trial_runner_fairness_disparity_ratio():
    runner = ClinicalTrialRunner(seed=42)
    report = runner.run_all_cohorts(n_sessions_per_cohort=50)

    assert "cohorts" in report
    assert len(report["cohorts"]) == 4
    for name, c_data in report["cohorts"].items():
        assert c_data["sensitivity"] >= 0.90
        assert c_data["specificity"] >= 0.90
        assert c_data["cohens_kappa"] >= 0.80

    assert report["disparity_ratio"] >= 0.85
    assert report["gmlp_principle_7_passed"] is True


def test_generate_report_markdown():
    runner = ClinicalTrialRunner(seed=42)
    report = runner.run_all_cohorts(n_sessions_per_cohort=30)
    md = runner.generate_report_markdown(report)

    assert "# FDA GMLP Principle 7: Clinical Trial Cohort Disparity Report" in md
    assert "young_adults" in md
    assert "frail_geriatric" in md
    assert "COMPLIANT" in md
