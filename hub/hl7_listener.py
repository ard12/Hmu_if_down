"""Minimal MLLP (Minimal Lower Layer Protocol) HL7 v2.x ADT parser and TCP listener."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hl7_listener")


class HL7MLLPListener:
    """Minimal MLLP (Minimal Lower Layer Protocol) TCP server.

    Parses HL7 v2.x ADT^A01/A02/A03/A08 message types and updates patient context.
    """

    MLLP_START = b"\x0b"
    MLLP_END = b"\x1c\x0d"

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 2575,
        context_store: Optional[Any] = None,
    ):
        self.host = host
        self.port = port
        self.context_store = context_store
        self.server: Optional[asyncio.Server] = None
        self.is_running = False

    def _strip_mllp(self, raw: bytes) -> bytes:
        """Strip MLLP envelope characters from raw byte buffer."""
        cleaned = raw
        if cleaned.startswith(self.MLLP_START):
            cleaned = cleaned[len(self.MLLP_START) :]
        if cleaned.endswith(self.MLLP_END):
            cleaned = cleaned[: -len(self.MLLP_END)]
        elif cleaned.endswith(b"\x1c"):
            cleaned = cleaned[:-1]
        return cleaned

    def _parse_adt(self, raw: bytes) -> Dict[str, Any]:
        """Parse HL7 v2.x ADT message into a normalized dictionary."""
        stripped = self._strip_mllp(raw)
        try:
            text = stripped.decode("utf-8")
        except UnicodeDecodeError:
            text = stripped.decode("latin-1", errors="replace")

        # Split into segments by \r\n, \r, or \n
        lines = [line.strip() for line in text.replace("\r\n", "\r").replace("\n", "\r").split("\r") if line.strip()]

        if not lines or lines[0][:3] != "MSH":
            raise ValueError("Invalid HL7 message: missing MSH header segment")

        msg_type = ""
        msg_id = ""
        patient_id = ""
        dob = ""
        gender = "U"
        room_id = ""
        admit_dt = ""
        morse_fall_scale: Optional[int] = None
        medications: List[str] = []

        for line in lines:
            if not line:
                continue
            delimiter = line[3] if len(line) > 3 and line[:3] == "MSH" else "|"
            fields = line.split(delimiter)
            seg_name = fields[0]

            if seg_name == "MSH":
                # MSH-9 is message type (index 8 because fields[0] is MSH and MSH-1 is delimiter)
                # In standard HL7: fields[0]=MSH, fields[1]=delimiter, fields[2]=enc_chars...
                # Splitting MSH by delimiter makes fields[0]="MSH", fields[1]=enc_chars, fields[2]=sending_app...
                # Thus MSH-9 is fields[8], MSH-10 is fields[9]
                if len(fields) > 8:
                    msg_type = fields[8]
                if len(fields) > 9:
                    msg_id = fields[9]
            elif seg_name == "PID":
                # PID-3: Patient Identifier
                if len(fields) > 3 and fields[3]:
                    patient_id = fields[3].split("^")[0]
                # PID-7: Date/Time of Birth
                if len(fields) > 7 and fields[7]:
                    dob = fields[7]
                # PID-8: Sex
                if len(fields) > 8 and fields[8]:
                    gender = fields[8].upper()
            elif seg_name == "PV1":
                # PV1-3: Assigned Patient Location (Point of care ^ Room ^ Bed)
                if len(fields) > 3 and fields[3]:
                    loc_parts = fields[3].split("^")
                    # Use room component if present, or first part
                    room_id = loc_parts[1] if len(loc_parts) > 1 and loc_parts[1] else loc_parts[0]
                # PV1-44: Admit Date/Time
                if len(fields) > 44 and fields[44]:
                    admit_dt = fields[44]
            elif seg_name == "ZFR":
                # Custom ZFR segment: ZFR|<score>|<assessment_dt>
                if len(fields) > 1 and fields[1]:
                    try:
                        morse_fall_scale = int(fields[1])
                    except ValueError:
                        pass
            elif seg_name == "RXA":
                # RXA-5: Administered Code & Description (Code ^ Name ^ ...)
                if len(fields) > 5 and fields[5]:
                    med_parts = fields[5].split("^")
                    med_name = med_parts[1] if len(med_parts) > 1 and med_parts[1] else med_parts[0]
                    if med_name:
                        medications.append(med_name.strip())

        return {
            "msg_type": msg_type,
            "msg_id": msg_id,
            "patient_id": patient_id,
            "dob": dob,
            "gender": gender,
            "room_id": room_id,
            "admit_dt": admit_dt,
            "morse_fall_scale": morse_fall_scale,
            "medications": medications,
        }

    def _send_ack(self, writer: asyncio.StreamWriter, msg_id: str, ack_code: str = "AA"):
        """Send HL7 ACK (AA=accept, AE=error) back to sender with MLLP framing."""
        ts = time.strftime("%Y%m%d%H%M%S")
        ack_body = (
            f"MSH|^~\\&|FALL_HUB|HOSPITAL|SENDER|RECEIVER|{ts}||ACK|{msg_id}|P|2.5\r"
            f"MSA|{ack_code}|{msg_id}\r"
        )
        packet = self.MLLP_START + ack_body.encode("utf-8") + self.MLLP_END
        try:
            writer.write(packet)
        except Exception as e:
            logger.warning(f"Error writing HL7 ACK: {e}")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming MLLP TCP connection."""
        buffer = bytearray()
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                buffer.extend(data)

                # Process complete MLLP packets
                while self.MLLP_START in buffer and (self.MLLP_END in buffer or b"\x1c" in buffer):
                    start_idx = buffer.index(self.MLLP_START)
                    if self.MLLP_END in buffer[start_idx:]:
                        end_idx = buffer.index(self.MLLP_END, start_idx) + len(self.MLLP_END)
                    elif b"\x1c" in buffer[start_idx:]:
                        end_idx = buffer.index(b"\x1c", start_idx) + 1
                    else:
                        break

                    raw_msg = bytes(buffer[start_idx:end_idx])
                    buffer = buffer[end_idx:]

                    try:
                        parsed = self._parse_adt(raw_msg)
                        msg_id = parsed.get("msg_id", "1")
                        room_id = parsed.get("room_id", "")
                        msg_type = parsed.get("msg_type", "")

                        if self.context_store is not None and room_id:
                            if "A03" in msg_type:
                                # Discharge: clear context for room
                                self.context_store.clear(room_id)
                            else:
                                # Admit / Update: upsert patient context
                                self.context_store.upsert(room_id, parsed)

                        self._send_ack(writer, msg_id, "AA")
                        await writer.drain()
                    except Exception as e:
                        logger.error(f"Error processing HL7 ADT message: {e}")
                        self._send_ack(writer, "UNKNOWN", "AE")
                        await writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"HL7 connection exception: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        """Launch asyncio TCP server."""
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        self.is_running = True
        logger.info(f"HL7 MLLP Listener started on {self.host}:{self.port}")

    async def stop(self):
        """Stop asyncio TCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        self.is_running = False
