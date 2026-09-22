"""Tests for Helm chart validation (Milestone 17.1)."""

from pathlib import Path
import shutil
import subprocess
import yaml
import pytest

CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "fall-detection-hub"


def test_chart_yaml_structure():
    chart_file = CHART_DIR / "Chart.yaml"
    assert chart_file.exists(), "Chart.yaml must exist"
    with open(chart_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["name"] == "fall-detection-hub"
    assert data["apiVersion"] == "v2"
    assert data["version"] == "4.1.0"
    assert data["appVersion"] == "4.1.0"
    assert "description" in data


def test_values_yaml_structure():
    values_file = CHART_DIR / "values.yaml"
    assert values_file.exists(), "values.yaml must exist"
    with open(values_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["replicaCount"] >= 1
    assert "image" in data
    assert "repository" in data["image"]
    assert "resources" in data
    assert "requests" in data["resources"]
    assert "limits" in data["resources"]
    assert "persistence" in data
    assert data["persistence"]["size"] == "10Gi"
    assert "autoscaling" in data
    assert data["autoscaling"]["enabled"] is True
    assert "livenessProbe" in data
    assert data["livenessProbe"]["httpGet"]["path"] == "/api/diagnostics/health"
    assert "readinessProbe" in data
    assert data["readinessProbe"]["httpGet"]["path"] == "/health"


def test_all_required_templates_exist():
    templates_dir = CHART_DIR / "templates"
    required = [
        "_helpers.tpl",
        "deployment.yaml",
        "service.yaml",
        "ingress.yaml",
        "hpa.yaml",
        "pvc.yaml",
        "configmap.yaml",
        "secret.yaml",
        "serviceaccount.yaml",
        "servicemonitor.yaml",
    ]
    for template_name in required:
        file_path = templates_dir / template_name
        assert file_path.exists(), f"Missing template: {template_name}"


def test_deployment_template_probe_references():
    dep_file = CHART_DIR / "templates" / "deployment.yaml"
    content = dep_file.read_text(encoding="utf-8")
    assert "livenessProbe:" in content
    assert ".Values.livenessProbe" in content
    assert "readinessProbe:" in content
    assert ".Values.readinessProbe" in content
    assert "models-storage" in content


def test_service_template_ports():
    svc_file = CHART_DIR / "templates" / "service.yaml"
    content = svc_file.read_text(encoding="utf-8")
    assert ".Values.service.httpPort" in content
    assert ".Values.service.csiUdpPort" in content
    assert ".Values.service.radarUdpPort" in content


def test_helm_lint_if_installed():
    helm_bin = shutil.which("helm")
    if not helm_bin:
        pytest.skip("helm binary not found on PATH; skipping helm lint subprocess check")

    res = subprocess.run([helm_bin, "lint", str(CHART_DIR)], capture_output=True, text=True)
    assert res.returncode == 0, f"helm lint failed: {res.stdout}\n{res.stderr}"
