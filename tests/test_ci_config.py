"""Tests for Continuous Delivery and Release Configuration (Milestone 17.3)."""

from pathlib import Path
import re
import yaml
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_FILE = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
CHART_FILE = PROJECT_ROOT / "helm" / "fall-detection-hub" / "Chart.yaml"


def test_release_workflow_is_valid_yaml():
    assert WORKFLOW_FILE.exists(), f"{WORKFLOW_FILE} must exist"
    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "name" in data
    assert "jobs" in data


def test_release_workflow_tag_trigger():
    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    on_triggers = data.get("on") or data.get(True)  # pyyaml parses 'on' as True sometimes
    assert on_triggers is not None
    assert "push" in on_triggers
    tags = on_triggers["push"].get("tags", [])
    assert any("v*" in tag for tag in tags)


def test_docker_build_step_uses_build_push_action():
    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    build_job = data["jobs"].get("build-and-push")
    assert build_job is not None
    steps = build_job.get("steps", [])
    step_uses = [s.get("uses", "") for s in steps]
    assert any("docker/build-push-action" in u for u in step_uses)


def test_helm_release_depends_on_build_and_push():
    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    helm_job = data["jobs"].get("helm-release")
    assert helm_job is not None
    needs = helm_job.get("needs")
    assert needs == "build-and-push" or (isinstance(needs, list) and "build-and-push" in needs)


def test_chart_version_follows_semver():
    assert CHART_FILE.exists(), f"{CHART_FILE} must exist"
    with open(CHART_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ver = data.get("version", "")
    semver_pattern = r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$"
    assert re.match(semver_pattern, ver), f"Version '{ver}' does not match SemVer"
