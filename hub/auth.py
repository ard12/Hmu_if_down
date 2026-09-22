"""Lightweight Token-Based RBAC for Clinical Audit Endpoints (Milestone 6.3).

Reads API tokens from config/api_tokens.yaml and verifies them against
incoming Authorization: Bearer <token> headers.

Roles:
  admin  — full access to audit endpoints, FHIR export, chain verification
  viewer — read-only access to telemetry and status endpoints only

Usage (FastAPI dependency):
    from hub.auth import require_admin
    @app.get("/api/audit/verify")
    async def verify_chain(token_data: dict = Depends(require_admin)):
        ...

If config/api_tokens.yaml does not exist, all admin requests are DENIED
(fail-secure mode). During development, create the file from the template.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("auth")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOKENS_FILE = PROJECT_ROOT / "config" / "api_tokens.yaml"

_bearer_scheme = HTTPBearer(auto_error=False)

# In-memory token cache (refreshed on first import or manual call to reload_tokens)
_token_store: dict = {}


def reload_tokens() -> None:
    """Load or reload token definitions from config/api_tokens.yaml."""
    global _token_store
    _token_store = {}

    if not _TOKENS_FILE.exists():
        logger.warning(
            "api_tokens.yaml not found at %s — all admin access DENIED. "
            "Copy config/api_tokens.yaml.template to enable authenticated endpoints.",
            _TOKENS_FILE,
        )
        return

    try:
        import yaml

        with open(_TOKENS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tokens = data.get("tokens", {})
        for token, meta in tokens.items():
            _token_store[str(token)] = {
                "role": meta.get("role", "viewer"),
                "description": meta.get("description", ""),
            }
        logger.info("Loaded %d API tokens from %s", len(_token_store), _TOKENS_FILE)
    except Exception as exc:
        logger.error("Failed to load api_tokens.yaml: %s", exc)


# Load tokens on module import
reload_tokens()


# Default JWT / session expiration in hours for HIPAA §164.312(a)(2)(ii) Automatic Logoff
JWT_EXPIRATION_HOURS = 1
AUTOMATIC_LOGOFF_SECONDS = JWT_EXPIRATION_HOURS * 3600


def is_token_expired(meta: dict) -> bool:
    """Check if token or session has exceeded expiration time (Automatic Logoff)."""
    if not meta:
        return True
    exp = meta.get("exp")
    if exp is not None:
        return time.time() > float(exp)
    return False


def create_break_glass_token(user_id: str, reason: str, ttl_seconds: int = 3600) -> str:
    """Generate an emergency break-glass token for critical clinical access (§164.312(a)(2)(i)).

    Mandates justification reason and emits emergency audit log entry.
    """
    import secrets
    import time

    token = f"break_glass_{secrets.token_hex(16)}"
    exp_time = time.time() + ttl_seconds
    _token_store[token] = {
        "role": "emergency",
        "sub": user_id,
        "reason": reason,
        "exp": exp_time,
        "emergency": True,
        "break_glass": True,
        "description": f"Emergency Break-Glass access for {user_id}: {reason}",
    }
    logger.warning(
        "EMERGENCY BREAK-GLASS ACCESS GRANTED to user=%s reason=%s (expires in %ds)",
        user_id,
        reason,
        ttl_seconds,
    )
    return token


def _get_token_meta(token: Optional[str]) -> Optional[dict]:
    """Return token metadata dict, or None if not found or expired."""
    if not token:
        return None
    meta = _token_store.get(token)
    if meta and is_token_expired(meta):
        logger.info("Token expired — Automatic logoff triggered for user %s", meta.get("sub", "unknown"))
        return None
    return meta


async def require_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency that requires a valid admin-role Bearer token.

    Raises:
        HTTPException 401 if no token provided or expired (automatic logoff).
        HTTPException 403 if token is invalid or has insufficient role.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    meta = _get_token_meta(credentials.credentials)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unrecognised API token.",
        )

    if meta.get("role") not in ("admin", "emergency"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient role: '{meta['role']}' cannot access admin endpoints.",
        )

    return meta


async def require_emergency_or_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency allowing either admin or emergency break-glass token."""
    return await require_admin(credentials)


def verify_token(token: str) -> Optional[dict]:
    """Programmatic token verification (not a FastAPI dependency).

    Returns the token metadata dict (with 'role') or None if invalid or expired.
    """
    return _get_token_meta(token)

