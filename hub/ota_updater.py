"""
OTA Updater Module — Handles Over-The-Air model and configuration updates for the Edge Fleet.
"""
import hashlib
import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger("ota_updater")

class OTAManager:
    """Manages the generation and verification of OTA update payloads."""
    
    def __init__(self):
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        
    def generate_update_payload(self, version: str, model_bytes: bytes, target_facility: Optional[str] = None) -> Dict[str, Any]:
        """
        Package a model into an OTA payload with a cryptographic SHA-256 hash.
        In a real deployment, this would be signed with a private key.
        """
        if model_bytes is None or not isinstance(model_bytes, (bytes, bytearray)):
            raise ValueError("model_bytes must be a non-null bytes or bytearray instance")
        payload_hash = hashlib.sha256(model_bytes).hexdigest()
        
        deployment_id = f"ota_{payload_hash[:8]}"
        payload = {
            "deployment_id": deployment_id,
            "version": version,
            "target_facility": target_facility,
            "hash": payload_hash,
            "size_bytes": len(model_bytes),
            # In a real REST API, we'd provide a pre-signed URL to download the binary.
            # Here we mock the binary content as a hex string for simplicity.
            "data_hex": model_bytes.hex()
        }
        
        self.active_deployments[deployment_id] = payload
        logger.info(f"Generated OTA payload {deployment_id} (Version: {version})")
        return payload

    def verify_and_apply_update(self, payload: Dict[str, Any]) -> bool:
        """
        Called by the Edge Hub to verify the integrity of an incoming OTA payload.
        Satisfies HAZ-031 control.
        """
        try:
            expected_hash = payload["hash"]
            data_bytes = bytes.fromhex(payload["data_hex"])
            
            actual_hash = hashlib.sha256(data_bytes).hexdigest()
            if actual_hash != expected_hash:
                logger.error(f"OTA Verification Failed: Hash mismatch! Expected {expected_hash}, got {actual_hash}")
                return False
                
            logger.info(f"OTA Verification Succeeded for deployment {payload.get('deployment_id')}")
            # Here the Edge Hub would save data_bytes to disk and hot-reload the model
            return True
            
        except (KeyError, ValueError) as e:
            logger.error(f"OTA Verification Failed: Malformed payload ({str(e)})")
            return False

ota_manager = OTAManager()
