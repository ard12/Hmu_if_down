"""
Voice Emergency Response & Ambient Audio Intercom Module (v4.8.0).
Provides on-device keyword spotting for hands-free distress calls,
ambient false-alarm voice cancellation, and two-way intercom state machine.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("voice_responder")


class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    INTERCOM_ACTIVE = "INTERCOM_ACTIVE"
    ALERT_TRIGGERED = "ALERT_TRIGGERED"
    ALERT_SUPPRESSED = "ALERT_SUPPRESSED"


@dataclass
class VoiceEvent:
    timestamp: float
    room_id: str
    event_type: str  # "DISTRESS_KEYWORD", "CANCEL_KEYWORD", "AUDIO_ENERGY_SURGE", "INTERCOM_OPEN"
    phrase: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VoiceResponder:
    """Ambient voice emergency responder and hands-free intercom manager."""

    DISTRESS_KEYWORDS = [
        "help",
        "help me",
        "i have fallen",
        "i fell",
        "fallen",
        "can't get up",
        "cannot get up",
        "emergency",
        "call 911",
        "somebody help",
        "need help",
    ]

    CANCEL_KEYWORDS = [
        "i'm okay",
        "im ok",
        "i am fine",
        "i'm fine",
        "false alarm",
        "cancel alert",
        "cancel",
        "don't call",
        "dont call",
        "all good",
    ]

    def __init__(
        self,
        room_id: str = "room_default",
        confidence_threshold: float = 0.65,
        energy_threshold_db: float = -35.0,
    ):
        self.room_id = room_id
        self.confidence_threshold = confidence_threshold
        self.energy_threshold_db = energy_threshold_db
        self.state = VoiceState.IDLE
        self.last_event: Optional[VoiceEvent] = None
        self.intercom_channel_open = False
        self.intercom_connected_at: Optional[float] = None
        self.events_history: List[VoiceEvent] = []
        self._on_distress_callback: Optional[Callable[[VoiceEvent], None]] = None
        self._on_cancel_callback: Optional[Callable[[VoiceEvent], None]] = None

    def register_distress_callback(self, cb: Callable[[VoiceEvent], None]) -> None:
        """Register hook invoked when an emergency distress phrase is matched."""
        self._on_distress_callback = cb

    def register_cancel_callback(self, cb: Callable[[VoiceEvent], None]) -> None:
        """Register hook invoked when a cancellation phrase is matched."""
        self._on_cancel_callback = cb

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two normalized strings."""
        if len(s1) < len(s2):
            return VoiceResponder._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _match_phrase_list(
        self, query: str, targets: List[str]
    ) -> Tuple[Optional[str], float]:
        """Fuzzy match query text against a target keyword dictionary using word boundaries."""
        query_norm = re.sub(r"[^\w\s]", "", query.lower()).strip()
        if not query_norm:
            return None, 0.0

        best_target = None
        best_score = 0.0

        for target in targets:
            target_norm = re.sub(r"[^\w\s]", "", target.lower()).strip()
            # Exact match or word-bounded regex match (prevents partial word false positives)
            pattern = r"(?:\b|^)" + re.escape(target_norm) + r"(?:\b|$)"
            if target_norm == query_norm or re.search(pattern, query_norm):
                return target, 1.0

            # Token overlap or Levenshtein ratio
            max_len = max(len(query_norm), len(target_norm))
            dist = self._levenshtein_distance(query_norm, target_norm)
            similarity = 1.0 - (dist / max_len)

            if similarity > best_score:
                best_score = similarity
                best_target = target

        return best_target, round(best_score, 3)

    def process_transcript(
        self, transcript: str, confidence: float = 1.0
    ) -> Optional[VoiceEvent]:
        """Process speech-to-text transcript from ambient room microphone.

        Safety-First Design (ISO 14971):
        Emergency distress keywords take absolute precedence over cancellation phrases.
        Cancellation phrases are also verified against negation prefixes (e.g. 'do not cancel').
        """
        if not transcript or not isinstance(transcript, str):
            return None
        now = time.time()

        # 1. Check for distress keywords FIRST
        distress_target, distress_score = self._match_phrase_list(
            transcript, self.DISTRESS_KEYWORDS
        )
        combined_distress_conf = distress_score * confidence

        # 2. Check for cancellation keywords
        cancel_target, cancel_score = self._match_phrase_list(
            transcript, self.CANCEL_KEYWORDS
        )
        combined_cancel_conf = cancel_score * confidence

        # Safety-First Conflict Resolution:
        # If distress is detected above threshold, distress ALWAYS wins over cancellation.
        if combined_distress_conf >= self.confidence_threshold:
            event = VoiceEvent(
                timestamp=now,
                room_id=self.room_id,
                event_type="DISTRESS_KEYWORD",
                phrase=distress_target or transcript,
                confidence=combined_distress_conf,
                metadata={"raw_transcript": transcript},
            )
            self.state = VoiceState.ALERT_TRIGGERED
            self.last_event = event
            self.events_history.append(event)
            logger.warning(
                f"[{self.room_id}] Distress Keyword Detected: '{distress_target}' ({combined_distress_conf:.2f})"
            )
            if self._on_distress_callback:
                self._on_distress_callback(event)
            return event

        # 3. Check for cancellation ONLY if no distress was detected
        if combined_cancel_conf >= self.confidence_threshold:
            query_norm = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
            cancel_norm = re.sub(r"[^\w\s]", "", (cancel_target or "").lower()).strip()
            negation_pattern = r"\b(not|dont|do not|never|cant|cannot)\s+" + re.escape(cancel_norm)
            if re.search(negation_pattern, query_norm):
                logger.info(
                    f"[{self.room_id}] Cancellation phrase '{cancel_target}' rejected due to negation in: '{transcript}'"
                )
                return None

            event = VoiceEvent(
                timestamp=now,
                room_id=self.room_id,
                event_type="CANCEL_KEYWORD",
                phrase=cancel_target or transcript,
                confidence=combined_cancel_conf,
                metadata={"raw_transcript": transcript},
            )
            self.state = VoiceState.ALERT_SUPPRESSED
            self.last_event = event
            self.events_history.append(event)
            logger.info(
                f"[{self.room_id}] Voice Alert Suppressed via phrase: '{cancel_target}' ({combined_cancel_conf:.2f})"
            )
            if self._on_cancel_callback:
                self._on_cancel_callback(event)
            return event

        return None

    def calculate_audio_energy(self, pcm_samples: List[float]) -> float:
        """Calculate RMS signal energy in dBFS from normalized PCM float samples [-1.0, 1.0]."""
        if not pcm_samples:
            return -100.0
        valid_samples = [s for s in pcm_samples if not (math.isnan(s) or math.isinf(s))]
        if not valid_samples:
            return -100.0
        sum_sq = sum(s * s for s in valid_samples)
        rms = math.sqrt(sum_sq / len(valid_samples))
        if rms <= 1e-6:
            return -100.0
        db = 20.0 * math.log10(rms)
        return round(db, 2)

    def open_intercom(self) -> bool:
        """Activate two-way audio intercom between the room speaker/mic and caregiver."""
        self.intercom_channel_open = True
        self.intercom_connected_at = time.time()
        self.state = VoiceState.INTERCOM_ACTIVE
        event = VoiceEvent(
            timestamp=self.intercom_connected_at,
            room_id=self.room_id,
            event_type="INTERCOM_OPEN",
            phrase="intercom_connected",
            confidence=1.0,
            metadata={"channel": "half_duplex_webrtc"},
        )
        self.last_event = event
        self.events_history.append(event)
        logger.info(f"[{self.room_id}] Intercom channel opened.")
        return True

    def close_intercom(self) -> bool:
        """Terminate two-way audio intercom session."""
        self.intercom_channel_open = False
        self.intercom_connected_at = None
        self.state = VoiceState.IDLE
        logger.info(f"[{self.room_id}] Intercom channel closed.")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and status metrics for voice subsystem."""
        return {
            "room_id": self.room_id,
            "state": self.state.value,
            "intercom_channel_open": self.intercom_channel_open,
            "intercom_connected_at": self.intercom_connected_at,
            "confidence_threshold": self.confidence_threshold,
            "energy_threshold_db": self.energy_threshold_db,
            "last_event": (
                {
                    "type": self.last_event.event_type,
                    "phrase": self.last_event.phrase,
                    "confidence": self.last_event.confidence,
                    "timestamp": self.last_event.timestamp,
                }
                if self.last_event
                else None
            ),
            "events_count": len(self.events_history),
        }


# Default global instance
voice_responder = VoiceResponder()
