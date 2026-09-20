"""SMART-on-FHIR backend service client with OAuth2 token auto-refresh."""

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx


class FHIRConnectionError(Exception):
    """Raised when communication with FHIR server fails."""
    pass


class SMARTFHIRClient:
    """SMART-on-FHIR backend service authentication (client_credentials grant).

    Fetches and publishes patient and observation resources with automatic OAuth2 token refresh.
    """

    def __init__(
        self,
        fhir_base_url: str,
        client_id: str,
        client_secret: str,
        token_url: Optional[str] = None,
        scope: str = "system/Patient.read system/Observation.read",
    ):
        self.fhir_base_url = fhir_base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url or f"{self.fhir_base_url}/oauth/token"
        self.scope = scope

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Fetches/caches OAuth2 token. Auto-refreshes 60s before expiry."""
        now = time.time()
        if self._token is not None and (self._token_expires_at - now) > 60:
            return self._token

        async with self._lock:
            # Double check inside lock
            now = time.time()
            if self._token is not None and (self._token_expires_at - now) > 60:
                return self._token

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        self.token_url,
                        data={
                            "grant_type": "client_credentials",
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "scope": self.scope,
                        },
                    )
                    if resp.status_code != 200:
                        raise FHIRConnectionError(
                            f"OAuth2 token request failed: HTTP {resp.status_code} - {resp.text}"
                        )
                    data = resp.json()
                    self._token = data["access_token"]
                    expires_in = float(data.get("expires_in", 3600))
                    self._token_expires_at = now + expires_in
                    return self._token
            except httpx.RequestError as e:
                raise FHIRConnectionError(f"OAuth2 token connection error: {e}") from e

    async def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """GET {fhir_base_url}/Patient/{patient_id} with Bearer token."""
        token = await self.get_token()
        url = f"{self.fhir_base_url}/Patient/{patient_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/fhir+json",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    raise FHIRConnectionError(
                        f"Failed to fetch patient {patient_id}: HTTP {resp.status_code} - {resp.text}"
                    )
                return resp.json()
        except httpx.RequestError as e:
            raise FHIRConnectionError(f"FHIR get_patient connection error: {e}") from e

    async def search_observations(self, patient_id: str, code: str) -> List[Dict[str, Any]]:
        """GET /Observation?subject={patient_id}&code={code}."""
        token = await self.get_token()
        url = f"{self.fhir_base_url}/Observation"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/fhir+json",
        }
        params = {"subject": patient_id, "code": code}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code != 200:
                    raise FHIRConnectionError(
                        f"Failed to search observations: HTTP {resp.status_code} - {resp.text}"
                    )
                data = resp.json()
                entries = data.get("entry", [])
                return [e.get("resource", {}) for e in entries]
        except httpx.RequestError as e:
            raise FHIRConnectionError(f"FHIR search_observations connection error: {e}") from e

    async def post_observation(self, observation: Dict[str, Any]) -> str:
        """POST /Observation — returns server-assigned ID."""
        token = await self.get_token()
        url = f"{self.fhir_base_url}/Observation"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, json=observation)
                if resp.status_code not in (200, 201):
                    raise FHIRConnectionError(
                        f"Failed to post observation: HTTP {resp.status_code} - {resp.text}"
                    )

                # Check Location header or body id
                loc = resp.headers.get("Location", "")
                if loc:
                    return loc.rstrip("/").split("/")[-1]
                data = resp.json() if resp.text else {}
                return str(data.get("id") or observation.get("id") or "")
        except httpx.RequestError as e:
            raise FHIRConnectionError(f"FHIR post_observation connection error: {e}") from e
