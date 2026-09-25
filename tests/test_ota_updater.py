import pytest
import hashlib
from hub.ota_updater import OTAManager

@pytest.fixture
def ota_manager():
    return OTAManager()

def test_generate_update_payload(ota_manager):
    """
    Covers: SRS-FLT-003
    Verifies generation of SHA-256 hashed OTA payloads.
    """
    model_bytes = b"mock_onnx_tensorrt_weights_data"
    payload = ota_manager.generate_update_payload("4.7.1", model_bytes, "FAC-1")
    
    assert "deployment_id" in payload
    assert payload["version"] == "4.7.1"
    assert payload["target_facility"] == "FAC-1"
    assert payload["size_bytes"] == len(model_bytes)
    assert payload["data_hex"] == model_bytes.hex()
    
    expected_hash = hashlib.sha256(model_bytes).hexdigest()
    assert payload["hash"] == expected_hash

def test_verify_and_apply_update_success(ota_manager):
    model_bytes = b"mock_onnx_tensorrt_weights_data"
    payload = ota_manager.generate_update_payload("4.7.1", model_bytes)
    
    # Simulate Edge Hub verifying the payload
    assert ota_manager.verify_and_apply_update(payload) is True

def test_verify_and_apply_update_tampered_hash(ota_manager):
    """
    Covers: HAZ-031
    Verifies that tampered payloads are rejected via strict SHA-256 validation.
    """
    model_bytes = b"mock_onnx_tensorrt_weights_data"
    payload = ota_manager.generate_update_payload("4.7.1", model_bytes)
    
    # Tamper with the hash (Man-in-the-middle attack)
    payload["hash"] = "deadbeef" + payload["hash"][8:]
    
    # Should fail SHA-256 validation
    assert ota_manager.verify_and_apply_update(payload) is False

def test_verify_and_apply_update_tampered_data(ota_manager):
    model_bytes = b"mock_onnx_tensorrt_weights_data"
    payload = ota_manager.generate_update_payload("4.7.1", model_bytes)
    
    # Tamper with the data
    tampered_bytes = b"malicious_weights_data"
    payload["data_hex"] = tampered_bytes.hex()
    
    # Hash of malicious_weights_data won't match original payload hash
    assert ota_manager.verify_and_apply_update(payload) is False

def test_verify_malformed_payload(ota_manager):
    payload = {"missing_hash_and_data": True}
    assert ota_manager.verify_and_apply_update(payload) is False
