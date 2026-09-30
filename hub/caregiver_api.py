"""
Caregiver Authentication & Mobile REST/WebSocket API.

Provides endpoints for mobile apps used by professional nurses, home
caregivers, and emergency response teams to manage falls and triage alerts.

@req SRS-UX-002
"""
from dataclasses import dataclass, field
import hashlib
import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from hub.auth import _token_lock, _token_store, is_token_expired

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/caregiver", tags=["caregiver"])
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class CaregiverProfile:
    caregiver_id: str
    name: str
    phone: str
    email: str
    role: str = "primary"  # "primary" or "secondary"
    password_hash: str = ""
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caregiver_id": self.caregiver_id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "role": self.role,
            "registered_at": self.registered_at,
        }


class CaregiverManager:
    """Manages registered caregivers and their sessions."""

    def __init__(self):
        self._caregivers: Dict[str, CaregiverProfile] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> str:
        """Derive salted PBKDF2-HMAC-SHA256 hash (100,000 iterations)."""
        if salt is None:
            salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return f"{salt}${key.hex()}"

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        """Verify password against salted PBKDF2 hash or legacy SHA-256."""
        if "$" in stored_hash:
            salt, key_hex = stored_hash.split("$", 1)
            computed = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                iterations=100_000,
            )
            return secrets.compare_digest(computed.hex(), key_hex)
        else:
            legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
            return secrets.compare_digest(legacy, stored_hash)

    def register(
        self, caregiver_id: str, name: str, phone: str, email: str, role: str, password: str
    ) -> CaregiverProfile:
        pwd_hash = self._hash_password(password)
        profile = CaregiverProfile(
            caregiver_id=caregiver_id,
            name=name,
            phone=phone,
            email=email,
            role=role,
            password_hash=pwd_hash,
            registered_at=time.time(),
        )
        with self._lock:
            self._caregivers[caregiver_id] = profile
        return profile

    def authenticate(self, caregiver_id: str, password: str) -> Optional[str]:
        with self._lock:
            profile = self._caregivers.get(caregiver_id)
        if not profile:
            return None
        if not self._verify_password(password, profile.password_hash):
            return None

        # Issue token valid for 1 hour
        token = f"cg_{secrets.token_hex(20)}"
        with _token_lock:
            _token_store[token] = {
                "sub": caregiver_id,
                "role": "caregiver",
                "name": profile.name,
                "exp": time.time() + 3600,
                "description": f"Caregiver token for {caregiver_id}",
            }
        return token

    def get(self, caregiver_id: str) -> Optional[CaregiverProfile]:
        with self._lock:
            return self._caregivers.get(caregiver_id)

    def list_caregivers(self) -> List[CaregiverProfile]:
        with self._lock:
            return list(self._caregivers.values())


# Singleton manager
caregiver_manager = CaregiverManager()


async def require_caregiver_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """Dependency verifying caregiver token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Format: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    with _token_lock:
        meta = _token_store.get(token)
    if not meta or is_token_expired(meta):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired caregiver token",
        )
    return meta


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_caregiver(request: Request):
    """Register a new caregiver."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    required = ("caregiver_id", "name", "phone", "email", "password")
    for field_name in required:
        if not body.get(field_name):
            raise HTTPException(status_code=422, detail=f"Missing required field: {field_name}")

    caregiver_id = str(body["caregiver_id"]).strip()
    name = str(body["name"]).strip()
    phone = str(body["phone"]).strip()
    email = str(body["email"]).strip()
    role = str(body.get("role", "primary")).strip()
    password = str(body["password"])

    profile = caregiver_manager.register(caregiver_id, name, phone, email, role, password)
    return {
        "status": "created",
        "caregiver": profile.to_dict(),
    }


@router.post("/login")
async def login_caregiver(request: Request):
    """Authenticate caregiver and return Bearer token."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    caregiver_id = body.get("caregiver_id")
    password = body.get("password")
    if not caregiver_id or not password:
        raise HTTPException(status_code=422, detail="caregiver_id and password required")

    token = caregiver_manager.authenticate(str(caregiver_id), str(password))
    if not token:
        raise HTTPException(status_code=401, detail="Invalid caregiver credentials")

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
        "caregiver_id": caregiver_id,
    }


@router.get("/alerts")
async def list_caregiver_alerts(
    token_meta: Dict[str, Any] = Depends(require_caregiver_auth),
):
    """List active and recent alerts from notification escalator."""
    from hub.dashboard.app import _get_notification_escalator
    escalator = _get_notification_escalator()
    if escalator is None:
        return {"alerts": [], "count": 0}

    active = escalator.get_active_escalations()
    all_events = escalator.get_all_events()
    return {
        "active_alerts": active,
        "all_alerts": all_events,
        "count": len(all_events),
    }


@router.post("/alerts/{event_id}/ack")
async def acknowledge_alert(
    event_id: str,
    token_meta: Dict[str, Any] = Depends(require_caregiver_auth),
):
    """Caregiver acknowledges a fall alert, stopping escalation."""
    from hub.dashboard.app import _get_notification_escalator
    escalator = _get_notification_escalator()
    if escalator is None:
        raise HTTPException(status_code=404, detail="Notification escalator not active")

    caregiver_id = token_meta.get("sub", "unknown")
    success = escalator.acknowledge(event_id, caregiver_id=caregiver_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert event '{event_id}' not found or cannot be acknowledged")

    event = escalator.get_event(event_id)
    return {
        "status": "acknowledged",
        "event_id": event_id,
        "caregiver_id": caregiver_id,
        "event": event.to_dict() if event else None,
    }


@router.get("/alerts/{event_id}/timeline")
async def get_alert_timeline(event_id: str):
    """Fetch progressive escalation timeline for a fall event."""
    from hub.dashboard.app import _get_notification_escalator
    escalator = _get_notification_escalator()
    if escalator is None:
        raise HTTPException(status_code=404, detail="Notification escalator not active")

    event = escalator.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Alert event '{event_id}' not found")

    return {
        "event_id": event_id,
        "current_tier": event.current_tier.name,
        "state": event.state.value,
        "timeline": event.history,
    }


@router.get("/fatigue")
async def get_fatigue_analytics(days: int = 30, caregiver_id: Optional[str] = None):
    """Return caregiver fatigue metrics and response times."""
    from hub.dashboard.app import broadcaster, _get_notification_escalator
    escalator = _get_notification_escalator()
    events = escalator.get_all_events() if escalator else []

    from hub.analytics import FallAnalytics
    audit_log = getattr(broadcaster, "audit_log", None)
    if audit_log is None:
        from hub.audit_log import AuditLog
        audit_log = AuditLog()
    analytics = FallAnalytics(audit_log)

    score = analytics.alert_fatigue_score(caregiver_id=caregiver_id, days=days, events=events)
    dist = analytics.response_time_distribution(caregiver_id=caregiver_id, days=days, events=events)
    tod = analytics.time_of_day_fatigue(days=days, events=events)

    return {
        "fatigue_score": score,
        "response_time_distribution": dist,
        "time_of_day_fatigue": tod,
        "days": days,
    }

