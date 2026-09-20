"""Tests for SQLite-backed model registry, cryptographic integrity, and changelog generation."""

import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from hub.model_registry import ModelRegistry


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "registry.db")
        yield db_path


def test_save_load_round_trip(temp_db):
    """save_model / load_model round-trip preserves classifier predictions."""
    registry = ModelRegistry(db_path=temp_db)

    X = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0], [6.0, 7.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression()
    clf.fit(X, y)

    v_id = registry.save_model(
        clf,
        metadata={"algorithm": "LogisticRegression", "sensitivity": 1.0, "specificity": 1.0},
    )

    loaded_clf = registry.load_model(v_id)
    assert np.array_equal(clf.predict(X), loaded_clf.predict(X))
    assert np.allclose(clf.predict_proba(X), loaded_clf.predict_proba(X))


def test_set_active_atomic_swap(temp_db):
    """set_active atomic swap: only one version active at a time."""
    registry = ModelRegistry(db_path=temp_db)

    v1 = registry.save_model("model_v1", metadata={"active": True})
    v2 = registry.save_model("model_v2", metadata={"active": False})
    v3 = registry.save_model("model_v3", metadata={"active": False})

    assert registry.get_active_version_id() == v1
    assert registry.get_active() == "model_v1"

    registry.set_active(v2)
    assert registry.get_active_version_id() == v2
    assert registry.get_active() == "model_v2"

    registry.set_active(v3)
    assert registry.get_active_version_id() == v3
    assert registry.get_active() == "model_v3"

    # Confirm in raw DB that exactly one row has active=1
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM model_versions WHERE active = 1")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 1


def test_verify_integrity_detects_tampering(temp_db):
    """verify_integrity detects bit-flipped or corrupted pickle bytes."""
    registry = ModelRegistry(db_path=temp_db)
    v1 = registry.save_model("clean_model_1")
    v2 = registry.save_model("clean_model_2")

    # Initially 100% clean
    status = registry.verify_integrity()
    assert status["ok"] is True
    assert len(status["corrupt_versions"]) == 0

    # Tamper with v2's model_bytes in SQLite
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE model_versions SET model_bytes = X'DEADBEEF' WHERE version_id = ?",
        (v2,),
    )
    conn.commit()
    conn.close()

    status_tampered = registry.verify_integrity()
    assert status_tampered["ok"] is False
    assert v2 in status_tampered["corrupt_versions"]
    assert v1 not in status_tampered["corrupt_versions"]


def test_list_versions_sorted_descending(temp_db):
    """list_versions sorted by created_at descending."""
    registry = ModelRegistry(db_path=temp_db)

    t0 = time.time()
    v1 = registry.save_model("m1", metadata={"created_at": t0 - 100})
    v2 = registry.save_model("m2", metadata={"created_at": t0})
    v3 = registry.save_model("m3", metadata={"created_at": t0 + 100})

    versions = registry.list_versions()
    assert len(versions) == 3
    # v3 is newest, v1 is oldest
    assert versions[0]["version_id"] == v3
    assert versions[1]["version_id"] == v2
    assert versions[2]["version_id"] == v1


def test_concurrent_save_model(temp_db):
    """Concurrent save_model from multiple threads does not corrupt DB."""
    registry = ModelRegistry(db_path=temp_db)

    def worker(worker_id):
        for i in range(15):
            registry.save_model(f"model_{worker_id}_{i}")

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    versions = registry.list_versions()
    assert len(versions) == 60


def test_generate_model_changelog_script(temp_db):
    """generate_model_changelog.py script runs without error and produces markdown table."""
    registry = ModelRegistry(db_path=temp_db)
    registry.save_model(
        "sample_model",
        metadata={
            "algorithm": "RandomForestClassifier",
            "sensitivity": 0.991,
            "specificity": 0.984,
            "brier": 0.018,
            "training_n": 1000,
            "active": True,
            "notes": "Baseline model",
        },
    )

    out_md = os.path.join(os.path.dirname(temp_db), "MODEL_CHANGELOG.md")

    cmd = [
        sys.executable,
        "docs/generate_model_changelog.py",
        "--db",
        temp_db,
        "--output",
        out_md,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(out_md)

    with open(out_md, "r", encoding="utf-8") as f:
        content = f.read()

    assert "IEC 62304 §8.2" in content
    assert "RandomForestClassifier" in content
    assert "99.10%" in content
    assert "ACTIVE" in content


def test_load_nonexistent_model_raises_keyerror(temp_db):
    """load_model with non-existent version_id raises KeyError."""
    registry = ModelRegistry(db_path=temp_db)
    with pytest.raises(KeyError):
        registry.load_model("nonexistent_id")


def test_set_active_nonexistent_model_raises_keyerror(temp_db):
    """set_active with non-existent version_id raises KeyError."""
    registry = ModelRegistry(db_path=temp_db)
    with pytest.raises(KeyError):
        registry.set_active("nonexistent_id")

