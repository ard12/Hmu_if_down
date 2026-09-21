"""Tests for FRAX-Style Clinical Fall Risk Score (Milestone 14.3)."""

import pytest
from hub.frax_risk import FRAXFallRiskScore


def test_young_healthy_female_low_risk():
    """Young healthy female (25F, BMI 22, no prior fall) -> LOW risk (< 0.10)."""
    calc = FRAXFallRiskScore()
    res = calc.compute(
        age=25,
        gender="F",
        bmi=22.0,
        prior_fall=False,
        morse_fall_scale=0,
        n_high_risk_meds=0,
    )
    assert res["ten_year_fall_risk"] < 0.10
    assert res["risk_category"] == "LOW"


def test_elderly_high_risk_profile():
    """Elderly (82F, BMI 18, prior fall, 3 high-risk meds, morse 65) -> HIGH risk (> 0.60)."""
    calc = FRAXFallRiskScore()
    res = calc.compute(
        age=82,
        gender="F",
        bmi=18.0,
        prior_fall=True,
        morse_fall_scale=65,
        n_high_risk_meds=3,
    )
    assert res["ten_year_fall_risk"] > 0.60
    assert res["risk_category"] == "HIGH"
    assert len(res["primary_risk_factors"]) >= 3


def test_morse_scale_increases_risk_monotonically():
    """Morse Fall Scale 60 increases risk score vs. Morse 20."""
    calc = FRAXFallRiskScore()
    res20 = calc.compute(age=70, gender="F", bmi=24.0, prior_fall=False, morse_fall_scale=20, n_high_risk_meds=0)
    res60 = calc.compute(age=70, gender="F", bmi=24.0, prior_fall=False, morse_fall_scale=60, n_high_risk_meds=0)

    assert res60["ten_year_fall_risk"] > res20["ten_year_fall_risk"]


def test_male_gender_reduces_risk_vs_female():
    """Male gender reduces risk vs. female holding other factors constant."""
    calc = FRAXFallRiskScore()
    res_female = calc.compute(age=72, gender="F", bmi=24.0, prior_fall=False, morse_fall_scale=25, n_high_risk_meds=1)
    res_male = calc.compute(age=72, gender="M", bmi=24.0, prior_fall=False, morse_fall_scale=25, n_high_risk_meds=1)

    assert res_male["ten_year_fall_risk"] < res_female["ten_year_fall_risk"]


def test_underweight_bmi_increases_risk():
    """BMI < 19 increases risk compared to normal BMI 22."""
    calc = FRAXFallRiskScore()
    res_normal = calc.compute(age=75, gender="F", bmi=22.0, prior_fall=False, morse_fall_scale=10, n_high_risk_meds=0)
    res_underweight = calc.compute(age=75, gender="F", bmi=17.5, prior_fall=False, morse_fall_scale=10, n_high_risk_meds=0)

    assert res_underweight["ten_year_fall_risk"] > res_normal["ten_year_fall_risk"]
    assert any("Underweight" in factor for factor in res_underweight["primary_risk_factors"])


def test_risk_category_boundaries():
    """Output risk_category boundaries correctly map LOW (<0.20), MODERATE (0.20-0.50), HIGH (>=0.50)."""
    calc = FRAXFallRiskScore()

    res_low = calc.compute(age=30, gender="M", bmi=23.0, prior_fall=False, morse_fall_scale=0, n_high_risk_meds=0)
    assert res_low["risk_category"] == "LOW"

    res_mod = calc.compute(age=72, gender="F", bmi=23.0, prior_fall=False, morse_fall_scale=25, n_high_risk_meds=0)
    assert res_mod["risk_category"] == "MODERATE"

    res_high = calc.compute(age=85, gender="F", bmi=18.0, prior_fall=True, morse_fall_scale=50, n_high_risk_meds=2)
    assert res_high["risk_category"] == "HIGH"
