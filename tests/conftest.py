"""
Centralized pytest configuration and shared fixtures (Phase 29, F-15).
Provides hermetic test isolation, DB teardown, and deterministic test environments.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator
import pytest


@pytest.fixture(autouse=True)
def clean_token_store():
    """Ensure _token_store in hub.auth is reset before and after each test."""
    from hub.auth import _token_store, _token_lock
    with _token_lock:
        original = dict(_token_store)
    yield
    with _token_lock:
        _token_store.clear()
        _token_store.update(original)


@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """Provide a thread-safe, isolated temporary directory with automatic teardown."""
    tmp = Path(tempfile.mkdtemp(prefix="falldetect_test_"))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_audit_log(isolated_temp_dir):
    """Provide an isolated, fresh AuditLog instance."""
    from hub.audit_log import AuditLog
    db_path = isolated_temp_dir / "audit.db"
    return AuditLog(db_path=db_path)
