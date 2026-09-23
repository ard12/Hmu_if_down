"""Tests for Automated Model Card Generator.

@req SRS-XAI-002
"""
from pathlib import Path
import pytest
from docs.generate_model_card import generate_model_card, load_evaluation_report


def test_model_card_generates_markdown(tmp_path):
    out_file = tmp_path / "TEST_MODEL_CARD.md"
    md = generate_model_card(output_path=out_file)

    assert out_file.exists()
    assert "# Model Card: CSI & mmWave Fall Detection Classifier" in md
    assert len(md) > 500


def test_model_card_contains_intended_use(tmp_path):
    out_file = tmp_path / "TEST_MODEL_CARD.md"
    md = generate_model_card(output_path=out_file)

    assert "## 2. Intended Use" in md
    assert "Primary Clinical Intent" in md
    assert "Out-of-Scope Misuse" in md


def test_model_card_contains_metrics(tmp_path):
    out_file = tmp_path / "TEST_MODEL_CARD.md"
    md = generate_model_card(output_path=out_file)

    assert "## 4. Quantitative Metrics" in md
    assert "Sensitivity (Recall)" in md
    assert "Specificity" in md
    assert "ROC-AUC" in md


def test_model_card_contains_ethical_section(tmp_path):
    out_file = tmp_path / "TEST_MODEL_CARD.md"
    md = generate_model_card(output_path=out_file)

    assert "## 7. Ethical & Privacy Considerations" in md
    assert "HIPAA Compliance" in md
    assert "Bias & Fairness" in md


def test_model_card_reproducible(tmp_path):
    out1 = tmp_path / "CARD1.md"
    out2 = tmp_path / "CARD2.md"
    md1 = generate_model_card(output_path=out1)
    md2 = generate_model_card(output_path=out2)
    assert md1 == md2
