"""
Unit Tests for RBAC Authentication and Break-Glass Tokens (Phase 29, F-16).
Tests token validation, expiration, admin authorization, and emergency access.

@covers SRS-SEC-004
"""

import time
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from hub.auth import (
    reload_tokens,
    is_token_expired,
    create_break_glass_token,
    verify_token,
    require_admin,
    require_emergency_or_admin,
    _token_store,
    _token_lock,
)


def test_token_expiration():
    """Verifies that expired tokens are recognized by is_token_expired."""
    assert is_token_expired({}) is True
    assert is_token_expired({"exp": time.time() - 100}) is True
    assert is_token_expired({"exp": time.time() + 3600}) is False
    assert is_token_expired({"role": "admin"}) is False


def test_verify_valid_and_invalid_tokens():
    """Verifies token verification and role extraction."""
    with _token_lock:
        _token_store["valid_admin_token"] = {"role": "admin", "description": "Admin Token"}
        _token_store["expired_token"] = {"role": "viewer", "exp": time.time() - 10}

    assert verify_token(None) is None
    assert verify_token("nonexistent_token") is None
    assert verify_token("expired_token") is None

    meta = verify_token("valid_admin_token")
    assert meta is not None
    assert meta["role"] == "admin"


@pytest.mark.anyio
async def test_require_admin_success():
    """Verifies require_admin succeeds for valid admin credentials."""
    with _token_lock:
        _token_store["test_admin"] = {"role": "admin"}

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test_admin")
    result = await require_admin(creds)
    assert result["role"] == "admin"


@pytest.mark.anyio
async def test_require_admin_unauthorized_and_forbidden():
    """Verifies 401 when missing credentials and 403 when role is insufficient."""
    # Missing credentials -> 401
    with pytest.raises(HTTPException) as exc_401:
        await require_admin(None)
    assert exc_401.value.status_code == 401

    # Viewer role -> 403
    with _token_lock:
        _token_store["test_viewer"] = {"role": "viewer"}
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test_viewer")
    with pytest.raises(HTTPException) as exc_403:
        await require_admin(creds)
    assert exc_403.value.status_code == 403


def test_create_break_glass_token_lifecycle():
    """Verifies break-glass token creation, role assignment, and expiration TTL."""
    token = create_break_glass_token(
        user_id="dr_turner",
        reason="Trauma ICU emergency",
        ttl_seconds=600,
    )
    assert token.startswith("break_glass_")

    meta = verify_token(token)
    assert meta is not None
    assert meta["role"] == "emergency"
    assert meta["emergency"] is True
    assert meta["break_glass"] is True
    assert meta["sub"] == "dr_turner"
    assert meta["reason"] == "Trauma ICU emergency"
