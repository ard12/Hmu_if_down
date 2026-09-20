import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from hub.smart_fhir_client import FHIRConnectionError, SMARTFHIRClient


def test_token_fetch_uses_client_credentials():
    """Token fetch uses client_credentials grant and returns access token."""
    async def _run():
        client = SMARTFHIRClient(
            fhir_base_url="https://fhir.hospital.org/r4",
            client_id="client_123",
            client_secret="secret_abc",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "token_xyz_123", "expires_in": 3600}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            token = await client.get_token()

            assert token == "token_xyz_123"
            mock_post.assert_called_once()
            data = mock_post.call_args[1]["data"]
            assert data["grant_type"] == "client_credentials"
            assert data["client_id"] == "client_123"
            assert data["client_secret"] == "secret_abc"

    asyncio.run(_run())


def test_cached_token_returned_when_valid():
    """Cached token returned without network request if not expired."""
    async def _run():
        client = SMARTFHIRClient(
            fhir_base_url="https://fhir.hospital.org/r4",
            client_id="cid",
            client_secret="sec",
        )
        client._token = "cached_token_1"
        client._token_expires_at = time.time() + 1000  # valid for 1000s

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            token = await client.get_token()
            assert token == "cached_token_1"
            mock_post.assert_not_called()

    asyncio.run(_run())


def test_token_auto_refreshed_when_near_expiry():
    """Token is auto-refreshed when < 60s remain before expiration."""
    async def _run():
        client = SMARTFHIRClient(
            fhir_base_url="https://fhir.hospital.org/r4",
            client_id="cid",
            client_secret="sec",
        )
        client._token = "expiring_token"
        client._token_expires_at = time.time() + 30  # only 30s remain (< 60s)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "fresh_token_2", "expires_in": 3600}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            token = await client.get_token()

            assert token == "fresh_token_2"
            mock_post.assert_called_once()

    asyncio.run(_run())


def test_get_patient_headers_and_url():
    """get_patient constructs correct URL and Authorization Bearer header."""
    async def _run():
        client = SMARTFHIRClient(
            fhir_base_url="https://fhir.hospital.org/r4",
            client_id="cid",
            client_secret="sec",
        )
        client.get_token = AsyncMock(return_value="mock_bearer_token")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"resourceType": "Patient", "id": "PAT_42"}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            patient = await client.get_patient("PAT_42")

            assert patient["id"] == "PAT_42"
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert url == "https://fhir.hospital.org/r4/Patient/PAT_42"
            headers = mock_get.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer mock_bearer_token"

    asyncio.run(_run())


def test_post_observation_content_type():
    """post_observation sends POST with application/fhir+json Content-Type."""
    async def _run():
        client = SMARTFHIRClient(
            fhir_base_url="https://fhir.hospital.org/r4",
            client_id="cid",
            client_secret="sec",
        )
        client.get_token = AsyncMock(return_value="mock_token")

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {"Location": "https://fhir.hospital.org/r4/Observation/OBS_99"}
        mock_resp.text = '{"id": "OBS_99"}'
        mock_resp.json.return_value = {"id": "OBS_99"}

        obs = {"resourceType": "Observation", "status": "final"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            obs_id = await client.post_observation(obs)

            assert obs_id == "OBS_99"
            mock_post.assert_called_once()
            headers = mock_post.call_args[1]["headers"]
            assert headers["Content-Type"] == "application/fhir+json"

    asyncio.run(_run())


def test_network_error_raises_fhir_connection_error():
    """Network connection errors raise FHIRConnectionError."""
    async def _run():
        client = SMARTFHIRClient(
            fhir_base_url="https://fhir.hospital.org/r4",
            client_id="cid",
            client_secret="sec",
        )
        client.get_token = AsyncMock(return_value="mock_token")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(FHIRConnectionError) as exc_info:
                await client.get_patient("PAT_FAIL")
            assert "connection error" in str(exc_info.value).lower()

    asyncio.run(_run())

