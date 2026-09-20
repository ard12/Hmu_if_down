"""Tests for HL7 v2.x ADT parser and MLLP listener."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from hub.hl7_listener import HL7MLLPListener


SAMPLE_ADT_A01 = (
    b"\x0b"
    b"MSH|^~\\&|HOSPITAL|EPIC|FALL_HUB|STATION|20260920120000||ADT^A01|MSG00001|P|2.5\r"
    b"PID|1||PAT12345^^^MRN||DOE^JOHN||19550315|M|||123 MAIN ST^^CITY^STATE^12345\r"
    b"PV1|1|I|ICU^ROOM_101^BED_1||||1234^DOCTOR^BOB||||||||||||123456|||||||||||||||||||||||||20260920110000\r"
    b"ZFR|65|20260920113000\r"
    b"RXA|0|1|20260920|20260920|WARFARIN^Warfarin sodium^NDC\r"
    b"RXA|0|1|20260920|20260920|FUROSEMIDE^Furosemide 40mg^NDC\r"
    b"\x1c\x0d"
)

SAMPLE_ADT_A03 = (
    b"\x0b"
    b"MSH|^~\\&|HOSPITAL|EPIC|FALL_HUB|STATION|20260920140000||ADT^A03|MSG00002|P|2.5\r"
    b"PID|1||PAT12345^^^MRN||DOE^JOHN||19550315|M\r"
    b"PV1|1|I|ICU^ROOM_101^BED_1\r"
    b"\x1c\x0d"
)

SAMPLE_ADT_A08 = (
    b"\x0b"
    b"MSH|^~\\&|HOSPITAL|EPIC|FALL_HUB|STATION|20260920130000||ADT^A08|MSG00003|P|2.5\r"
    b"PID|1||PAT12345^^^MRN||DOE^JOHN||19550315|M\r"
    b"PV1|1|I|ICU^ROOM_101^BED_1\r"
    b"ZFR|80|20260920130000\r"
    b"\x1c\x0d"
)


def test_mllp_stripping():
    """MLLP start/end byte stripping is correct."""
    listener = HL7MLLPListener()
    raw = b"\x0bMSH|test\x1c\x0d"
    stripped = listener._strip_mllp(raw)
    assert stripped == b"MSH|test"

    raw_partial = b"\x0bMSH|test\x1c"
    stripped_partial = listener._strip_mllp(raw_partial)
    assert stripped_partial == b"MSH|test"


def test_adt_a01_admit_parsing():
    """ADT^A01 admit message parsed correctly (PID, PV1, ZFR, RXA segments)."""
    listener = HL7MLLPListener()
    parsed = listener._parse_adt(SAMPLE_ADT_A01)

    assert parsed["msg_type"] == "ADT^A01"
    assert parsed["msg_id"] == "MSG00001"
    assert parsed["patient_id"] == "PAT12345"
    assert parsed["dob"] == "19550315"
    assert parsed["gender"] == "M"
    assert parsed["room_id"] == "ROOM_101"
    assert parsed["admit_dt"] == "20260920110000"
    assert parsed["morse_fall_scale"] == 65
    assert "Warfarin sodium" in parsed["medications"]
    assert "Furosemide 40mg" in parsed["medications"]


def test_adt_a03_discharge_parsing():
    """ADT^A03 discharge message parsed with correct room_id."""
    listener = HL7MLLPListener()
    parsed = listener._parse_adt(SAMPLE_ADT_A03)

    assert parsed["msg_type"] == "ADT^A03"
    assert parsed["room_id"] == "ROOM_101"
    assert parsed["patient_id"] == "PAT12345"


def test_adt_a08_update_parsing():
    """ADT^A08 update parses updated Morse Fall Scale."""
    listener = HL7MLLPListener()
    parsed = listener._parse_adt(SAMPLE_ADT_A08)

    assert parsed["msg_type"] == "ADT^A08"
    assert parsed["morse_fall_scale"] == 80


def test_multiple_rxa_medication_segments_captured():
    """Multiple RXA medication segments are all captured in list."""
    listener = HL7MLLPListener()
    msg = (
        b"\x0b"
        b"MSH|^~\\&|HOSP||||||ADT^A01|1\r"
        b"PID|1||P1|||||F\r"
        b"PV1|1|I|R1\r"
        b"RXA|0|1|20260920|20260920|MED1^Aspirin\r"
        b"RXA|0|1|20260920|20260920|MED2^Metoprolol\r"
        b"RXA|0|1|20260920|20260920|MED3^Lorazepam\r"
        b"\x1c\x0d"
    )
    parsed = listener._parse_adt(msg)
    assert len(parsed["medications"]) == 3
    assert parsed["medications"] == ["Aspirin", "Metoprolol", "Lorazepam"]


def test_send_ack_formatting():
    """HL7 ACK is formatted with AA or AE and framed with MLLP bytes."""
    listener = HL7MLLPListener()
    writer = MagicMock()

    listener._send_ack(writer, "MSG999", "AA")
    writer.write.assert_called_once()
    data = writer.write.call_args[0][0]

    assert data.startswith(b"\x0b")
    assert data.endswith(b"\x1c\x0d")
    assert b"MSA|AA|MSG999" in data


import asyncio


def test_handle_client_dispatches_to_context_store():
    """handle_client dispatches parsed ADT to context_store."""
    async def _run():
        mock_store = MagicMock()
        listener = HL7MLLPListener(context_store=mock_store)

        reader = AsyncMock()
        # Return sample ADT A01 on first read, then empty bytes to close
        reader.read.side_effect = [SAMPLE_ADT_A01, b""]
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()

        await listener.handle_client(reader, writer)

        mock_store.upsert.assert_called_once()
        call_args = mock_store.upsert.call_args[0]
        assert call_args[0] == "ROOM_101"
        assert call_args[1]["patient_id"] == "PAT12345"

    asyncio.run(_run())


def test_handle_client_discharge_clears_context():
    """handle_client with ADT^A03 calls clear on context_store."""
    async def _run():
        mock_store = MagicMock()
        listener = HL7MLLPListener(context_store=mock_store)

        reader = AsyncMock()
        reader.read.side_effect = [SAMPLE_ADT_A03, b""]
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()

        await listener.handle_client(reader, writer)

        mock_store.clear.assert_called_once_with("ROOM_101")

    asyncio.run(_run())


def test_malformed_mllp_framing_sends_ae_ack():
    """Malformed MLLP framing returns AE acknowledgement."""
    async def _run():
        listener = HL7MLLPListener()
        reader = AsyncMock()
        # Bad message without valid MSH
        reader.read.side_effect = [b"\x0bBAD_PAYLOAD\x1c\x0d", b""]
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()

        await listener.handle_client(reader, writer)

        # Check writer wrote AE ack
        written = [call[0][0] for call in writer.write.call_args_list]
        assert any(b"MSA|AE|" in w for w in written)

    asyncio.run(_run())
