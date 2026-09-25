"""
Fleet Management REST API — Phase 23
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Any

from hub.fleet_manager import fleet_manager
from hub.ota_updater import ota_manager

router = APIRouter(prefix="/api/fleet", tags=["fleet"])

class ProvisionRequest(BaseModel):
    facility_id: str
    room_id: str
    firmware_ver: str
    model_ver: str

class ProvisionResponse(BaseModel):
    device_id: str
    status: str

class HeartbeatRequest(BaseModel):
    device_id: str
    firmware_ver: Optional[str] = None
    model_ver: Optional[str] = None

class OTARequest(BaseModel):
    version: str
    model_data_hex: str
    target_facility: Optional[str] = None


@router.post("/provision", response_model=ProvisionResponse)
def provision_device(req: ProvisionRequest):
    """Provision a new Edge Hub."""
    device_id = fleet_manager.provision_device(
        req.facility_id, req.room_id, req.firmware_ver, req.model_ver
    )
    return {"device_id": device_id, "status": "provisioned"}

@router.post("/heartbeat")
def heartbeat(req: HeartbeatRequest):
    """Record a heartbeat from an Edge Hub."""
    success = fleet_manager.record_heartbeat(req.device_id, req.firmware_ver, req.model_ver)
    if not success:
        raise HTTPException(status_code=404, detail="Device not found in registry")
    return {"status": "ok"}

@router.get("/devices")
def list_devices(facility_id: Optional[str] = None):
    """Get the fleet status matrix."""
    devices = fleet_manager.get_fleet_status(facility_id)
    return {"devices": devices, "total": len(devices)}

@router.post("/ota/deploy")
def deploy_ota(req: OTARequest):
    """Generate an OTA update payload for the fleet."""
    try:
        model_bytes = bytes.fromhex(req.model_data_hex)
    except ValueError:
        raise HTTPException(status_code=400, detail="model_data_hex must be a valid hex string")
        
    payload = ota_manager.generate_update_payload(req.version, model_bytes, req.target_facility)
    return {"status": "deployed", "deployment": payload}
