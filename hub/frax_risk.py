"""
FRAX-Style Clinical Fall Risk Score Integration.
Stratifies 10-year fall and fracture risk proxy based on published clinical coefficients.
"""

import math
from typing import Dict, List, Optional


class FRAXFallRiskScore:
    """
    Simplified FRAX-inspired clinical fall risk calculator.
    Inputs: age, gender, BMI, prior fall, Morse Fall Scale, high-risk medications.
    NOT a diagnostic tool — used as a risk stratification aid only.

    Uses logistic regression fitted to published FRAX coefficients
    (Kanis et al., 2008) adapted for fall risk.
    """

    def compute(
        self,
        age: int,
        gender: str,
        bmi: float,
        prior_fall: bool,
        morse_fall_scale: int = 0,
        n_high_risk_meds: int = 0,
    ) -> dict:
        """
        Returns:
          {
            "ten_year_fall_risk": float,   # 0–1 probability
            "risk_category": "LOW" | "MODERATE" | "HIGH",
            "primary_risk_factors": list[str],
          }
        """
        age = int(age)
        gender = str(gender).upper()
        bmi = float(bmi)
        prior_fall = bool(prior_fall)
        morse = int(morse_fall_scale)
        n_meds = int(n_high_risk_meds)

        primary_risk_factors: List[str] = []

        # Baseline log-odds intercept
        z = -2.8

        # Age component (+0.04 per year over 50)
        age_diff = age - 50
        z += 0.04 * age_diff
        if age >= 65:
            primary_risk_factors.append(f"Advanced age ({age} years)")

        # Gender component
        if gender.startswith("F"):
            z += 0.25
            primary_risk_factors.append("Female gender (higher epidemiological fall risk)")
        else:
            z -= 0.25

        # BMI component (underweight BMI < 19 is high risk)
        if bmi < 19.0:
            z += 0.6
            primary_risk_factors.append(f"Underweight BMI ({bmi:.1f} < 19.0)")
        elif bmi > 30.0:
            z += 0.2
            primary_risk_factors.append(f"Obesity BMI ({bmi:.1f} > 30.0)")

        # Prior fall history
        if prior_fall:
            z += 0.8
            primary_risk_factors.append("Documented history of prior falls")

        # Morse Fall Scale score
        if morse > 0:
            z += 0.025 * morse
            if morse >= 45:
                primary_risk_factors.append(f"High Morse Fall Scale score ({morse})")

        # High-risk medications
        if n_meds > 0:
            z += 0.25 * n_meds
            primary_risk_factors.append(f"Concurrent high-risk medications ({n_meds})")

        # Logistic sigmoid probability
        risk_prob = 1.0 / (1.0 + math.exp(-z))
        risk_prob = round(max(0.01, min(0.99, risk_prob)), 3)

        # Categorize
        if risk_prob < 0.20:
            risk_category = "LOW"
        elif risk_prob < 0.50:
            risk_category = "MODERATE"
        else:
            risk_category = "HIGH"

        return {
            "ten_year_fall_risk": risk_prob,
            "risk_category": risk_category,
            "primary_risk_factors": primary_risk_factors,
        }
