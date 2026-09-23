"""Tests for SHAP Feature Attribution and XAI Engine.

@req SRS-XAI-001
"""
import numpy as np
import pytest
from hub.explainability import SHAPExplainer


@pytest.fixture
def explainer():
    return SHAPExplainer()


def test_shap_explanation_has_feature_names(explainer):
    feats = np.array([20.0, 10.0, 5.0, 2.0, 1.0, 0.8, 2.0, 0.4, 2.0])
    exp = explainer.explain_prediction(feats)

    assert "feature_names" in exp
    assert len(exp["feature_names"]) == 9
    assert "feature_values" in exp
    assert "shap_values" in exp
    assert "base_value" in exp
    assert "prediction_proba" in exp
    assert "predicted_class" in exp


def test_shap_values_sum_to_prediction(explainer):
    feats = np.array([55.0, 42.0, 28.0, 18.0, 3.8, 2.6, 8.5, 1.9, 3.2])
    exp = explainer.explain_prediction(feats)

    # Local Accuracy / Efficiency axiom: base_value + sum(shap_values) == prediction_proba
    reconstructed = exp["base_value"] + sum(exp["shap_values"])
    assert abs(reconstructed - exp["prediction_proba"]) < 1e-3


def test_global_importance_ranks_all_features(explainer):
    ranking = explainer.global_importance()
    assert len(ranking) == 9
    for name in explainer.feature_names:
        assert name in ranking
        assert ranking[name] >= 0.0

    # Values should be in descending order
    vals = list(ranking.values())
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_counterfactual_changes_prediction(explainer):
    # A clear fall vector
    fall_feats = np.array([60.0, 45.0, 32.0, 22.0, 4.0, 3.0, 9.0, 2.0, 3.5])
    cf = explainer.counterfactual(fall_feats, target_class=0)

    assert cf["original_prediction"] == 1
    assert cf["counterfactual_prediction"] == 0
    assert len(cf["changes"]) > 0
    assert "counterfactual_features" in cf


def test_waterfall_json_schema(explainer):
    feats = np.array([25.0, 15.0, 8.0, 4.0, 1.2, 0.9, 3.0, 0.5, 2.2])
    exp = explainer.explain_prediction(feats)
    waterfall = explainer.to_waterfall_json(exp)

    assert "base_value" in waterfall
    assert "prediction_proba" in waterfall
    assert "predicted_class" in waterfall
    assert "bars" in waterfall
    assert len(waterfall["bars"]) == 9
    bar = waterfall["bars"][0]
    assert "feature" in bar
    assert "value" in bar
    assert "contribution" in bar
    assert bar["direction"] in ("positive", "negative")


def test_explainer_without_shap_library_fallback():
    # Force _use_shap_pkg = False
    exp = SHAPExplainer()
    exp._use_shap_pkg = False
    feats = np.array([12.0, 6.0, 2.5, 1.2, 0.6, 0.5, 1.2, 0.25, 1.6])
    result = exp.explain_prediction(feats)
    assert len(result["shap_values"]) == 9
    assert abs(result["base_value"] + sum(result["shap_values"]) - result["prediction_proba"]) < 1e-3


def test_explainer_invalid_dimensions_raises(explainer):
    with pytest.raises(ValueError):
        explainer.explain_prediction(np.array([1.0, 2.0]))  # Only 2 features instead of 9
