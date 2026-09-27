"""
Voice Emergency Response REST API — Phase 24 (v4.8.0).
Provides endpoints for audio transcript ingestion, voice-activated alert
cancellation, two-way caregiver intercom toggling, and microphone telemetry.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from hub.voice_responder import voice_responder

router = APIRouter(prefix="/api/voice", tags=["voice"])


class TranscriptPayload(BaseModel):
    transcript: str
    confidence: Optional[float] = 1.0
    room_id: Optional[str] = "room_default"


class ManualVoiceTriggerPayload(BaseModel):
    phrase: str
    room_id: Optional[str] = "room_default"
    patient_id: Optional[str] = "patient_unknown"


@router.post("/process")
def process_transcript(req: TranscriptPayload):
    """
    Ingest a speech-to-text transcript from ambient room mic.
    Detects distress keywords (e.g. 'Help me') or cancellation ('I am fine').
    """
    if req.room_id and req.room_id != voice_responder.room_id:
        voice_responder.room_id = req.room_id

    event = voice_responder.process_transcript(
        transcript=req.transcript, confidence=req.confidence or 1.0
    )
    if not event:
        return {
            "status": "NO_ACTION",
            "matched": False,
            "transcript": req.transcript,
            "voice_state": voice_responder.state.value,
        }

    return {
        "status": "MATCHED",
        "matched": True,
        "event_type": event.event_type,
        "phrase": event.phrase,
        "confidence": event.confidence,
        "voice_state": voice_responder.state.value,
    }


@router.post("/trigger")
def trigger_voice_sos(req: ManualVoiceTriggerPayload):
    """Directly trigger vocal SOS emergency protocol."""
    event = voice_responder.process_transcript(req.phrase, confidence=1.0)
    return {
        "status": "TRIGGERED",
        "room_id": req.room_id,
        "phrase": req.phrase,
        "event": (
            {
                "type": event.event_type,
                "confidence": event.confidence,
            }
            if event
            else None
        ),
        "voice_state": voice_responder.state.value,
    }


@router.post("/cancel")
def cancel_via_voice(req: ManualVoiceTriggerPayload):
    """Directly cancel/suppress alert using voice confirmation."""
    event = voice_responder.process_transcript(req.phrase or "I am fine", confidence=1.0)
    return {
        "status": "CANCELLED" if event and event.event_type == "CANCEL_KEYWORD" else "IGNORED",
        "voice_state": voice_responder.state.value,
    }


@router.post("/intercom/open")
def open_intercom():
    """Open two-way audio intercom between caregiver station and patient room."""
    success = voice_responder.open_intercom()
    return {"status": "OPEN", "intercom_open": success}


@router.post("/intercom/close")
def close_intercom():
    """Terminate two-way audio intercom session."""
    success = voice_responder.close_intercom()
    return {"status": "CLOSED", "intercom_open": False}


@router.get("/status")
def get_voice_status():
    """Retrieve current voice responder diagnostics and event statistics."""
    return voice_responder.get_status()
