"""
Unit and integration tests for Voice Emergency Responder & Intercom (Phase 24).
"""

import pytest
from fastapi.testclient import TestClient
from hub.voice_responder import VoiceResponder, VoiceState, VoiceEvent
from hub.dashboard.app import app


@pytest.fixture
def responder():
    return VoiceResponder(room_id="room_101", confidence_threshold=0.65)


def test_distress_keyword_detection(responder):
    """
    Covers: SRS-VOC-001
    Verifies that vocal distress keywords ('help me', 'I have fallen') trigger an alert.
    """
    event = responder.process_transcript("Please help me I cannot get up")
    assert event is not None
    assert event.event_type == "DISTRESS_KEYWORD"
    assert event.confidence >= 0.65
    assert responder.state == VoiceState.ALERT_TRIGGERED


def test_voice_alert_cancellation(responder):
    """
    Covers: SRS-VOC-002
    Verifies hands-free voice cancellation when patient states they are okay.
    """
    # Trigger first
    responder.process_transcript("help me")
    assert responder.state == VoiceState.ALERT_TRIGGERED

    # Cancel via voice
    cancel_event = responder.process_transcript("I'm okay, false alarm")
    assert cancel_event is not None
    assert cancel_event.event_type == "CANCEL_KEYWORD"
    assert responder.state == VoiceState.ALERT_SUPPRESSED


def test_conflict_resolution_and_negation(responder):
    """
    Covers: HAZ-033, SRS-VOC-001
    F-02: Verifies safety-first conflict resolution when transcript contains both
    distress and cancel phrases ('do not cancel, help me!').
    Also verifies negated cancellation is rejected.
    """
    # 1. Mixed distress and cancel -> Distress MUST win
    event = responder.process_transcript("do not cancel, help me!")
    assert event is not None
    assert event.event_type == "DISTRESS_KEYWORD"
    assert responder.state == VoiceState.ALERT_TRIGGERED

    # 2. Negated cancellation -> Should not suppress or trigger cancel
    responder.state = VoiceState.IDLE
    neg_event = responder.process_transcript("do not cancel")
    assert neg_event is None
    assert responder.state == VoiceState.IDLE



def test_acoustic_clipping_and_fuzziness(responder):
    """
    Covers: HAZ-033
    Verifies phonetic/Levenshtein matching under noisy or slurred speech.
    """
    # Slurred or slightly misspelled emergency phrases
    event = responder.process_transcript("heelp meee")
    assert event is not None
    assert event.event_type == "DISTRESS_KEYWORD"

    # Energy calculation check
    samples = [0.05, -0.05, 0.1, -0.1]
    db = responder.calculate_audio_energy(samples)
    assert isinstance(db, float)
    assert db < 0.0


def test_two_way_intercom_lifecycle(responder):
    """
    Covers: SRS-VOC-001
    Verifies opening and closing the two-way audio intercom channel.
    """
    assert responder.intercom_channel_open is False
    assert responder.open_intercom() is True
    assert responder.intercom_channel_open is True
    assert responder.state == VoiceState.INTERCOM_ACTIVE

    status = responder.get_status()
    assert status["intercom_channel_open"] is True
    assert status["room_id"] == "room_101"

    assert responder.close_intercom() is True
    assert responder.intercom_channel_open is False
    assert responder.state == VoiceState.IDLE


def test_voice_api_endpoints():
    """
    Covers: SRS-VOC-001, SRS-VOC-002
    Verifies REST API endpoints for transcript processing, triggering, and intercom.
    """
    client = TestClient(app)

    # 1. Process transcript
    resp = client.post(
        "/api/voice/process",
        json={"transcript": "Somebody help me please", "confidence": 0.95},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched"] is True
    assert data["event_type"] == "DISTRESS_KEYWORD"

    # 2. Intercom open
    resp_open = client.post("/api/voice/intercom/open")
    assert resp_open.status_code == 200
    assert resp_open.json()["intercom_open"] is True

    # 3. Intercom close
    resp_close = client.post("/api/voice/intercom/close")
    assert resp_close.status_code == 200
    assert resp_close.json()["intercom_open"] is False

    # 4. Status check
    resp_status = client.get("/api/voice/status")
    assert resp_status.status_code == 200
    assert "state" in resp_status.json()

    # 5. Cancel endpoint
    resp_cancel = client.post(
        "/api/voice/cancel",
        json={"phrase": "I am fine, do not worry"},
    )
    assert resp_cancel.status_code == 200
    assert resp_cancel.json()["status"] == "CANCELLED"
