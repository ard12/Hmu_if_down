"""Home Assistant MQTT Auto-Discovery integration for Fall Detection Hub.

Generates MQTT discovery topics and JSON payloads adhering to the official
Home Assistant MQTT Discovery specification for binary sensors, telemetry sensors,
and command buttons.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("HADiscovery")

DEVICE_INFO = {
    "identifiers": ["fall_detection_hub"],
    "name": "Multi-Modal Fall Detection Hub",
    "model": "ESP32-CSI-mmWave-Fusion",
    "manufacturer": "OpenFall",
    "sw_version": "1.0.0",
}


def build_discovery_configs(
    discovery_prefix: str = "homeassistant",
    base_topic: str = "fall_detection",
    node_ids: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, Dict[str, Any]]:
    """Generate all Home Assistant MQTT discovery topics and payload configurations.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of MQTT discovery topic -> configuration dictionary.
    """
    configs: Dict[str, Dict[str, Any]] = {}
    hub_id = "fall_detection_hub"

    # 1. Binary Sensor: Fall Alert (Critical Emergency)
    topic_binary_fall = f"{discovery_prefix}/binary_sensor/{hub_id}/fall_detected/config"
    configs[topic_binary_fall] = {
        "name": "Fall Detected",
        "unique_id": f"{hub_id}_fall_detected",
        "device_class": "safety",
        "state_topic": f"{base_topic}/state",
        "payload_on": "ON",
        "payload_off": "OFF",
        "value_template": "{{ 'ON' if value == 'FALL DETECTED' else 'OFF' }}",
        "device": DEVICE_INFO,
    }

    # 2. Sensor: Unified System State
    topic_state = f"{discovery_prefix}/sensor/{hub_id}/system_state/config"
    configs[topic_state] = {
        "name": "System Fall State",
        "unique_id": f"{hub_id}_system_state",
        "state_topic": f"{base_topic}/state",
        "icon": "mdi:shield-alert",
        "device": DEVICE_INFO,
    }

    # 3. Sensor: Radar Target Height
    topic_height = f"{discovery_prefix}/sensor/{hub_id}/radar_height/config"
    configs[topic_height] = {
        "name": "Target Height",
        "unique_id": f"{hub_id}_radar_height",
        "state_topic": f"{base_topic}/radar",
        "device_class": "distance",
        "unit_of_measurement": "m",
        "value_template": "{{ value_json.height }}",
        "icon": "mdi:human-male-height",
        "device": DEVICE_INFO,
    }

    # 4. Sensor: Radar Detected Posture
    topic_posture = f"{discovery_prefix}/sensor/{hub_id}/radar_posture/config"
    configs[topic_posture] = {
        "name": "Detected Posture",
        "unique_id": f"{hub_id}_radar_posture",
        "state_topic": f"{base_topic}/radar",
        "value_template": "{{ value_json.posture }}",
        "icon": "mdi:human",
        "device": DEVICE_INFO,
    }

    # 5. Sensor: Radar Floor Dwell Time
    topic_dwell = f"{discovery_prefix}/sensor/{hub_id}/radar_dwell/config"
    configs[topic_dwell] = {
        "name": "Floor Dwell Time",
        "unique_id": f"{hub_id}_radar_dwell",
        "state_topic": f"{base_topic}/radar",
        "device_class": "duration",
        "unit_of_measurement": "s",
        "value_template": "{{ value_json.dwell }}",
        "icon": "mdi:timer-sand",
        "device": DEVICE_INFO,
    }

    # 6. Sensor: Probabilistic ML Fall Score
    topic_ml = f"{discovery_prefix}/sensor/{hub_id}/ml_fall_probability/config"
    configs[topic_ml] = {
        "name": "ML Fall Probability",
        "unique_id": f"{hub_id}_ml_fall_probability",
        "state_topic": f"{base_topic}/ml",
        "unit_of_measurement": "%",
        "value_template": "{{ (value_json.probability * 100) | round(1) }}",
        "icon": "mdi:chart-bell-curve-cumulative",
        "device": DEVICE_INFO,
    }

    # 7. Per-Node CSI Telemetry Sensors
    for nid in node_ids:
        # CSI Velocity
        topic_v = f"{discovery_prefix}/sensor/{hub_id}/csi_node_{nid}_velocity/config"
        configs[topic_v] = {
            "name": f"CSI Node {nid} Doppler Velocity",
            "unique_id": f"{hub_id}_csi_node_{nid}_velocity",
            "state_topic": f"{base_topic}/csi/{nid}",
            "device_class": "speed",
            "unit_of_measurement": "m/s",
            "value_template": "{{ value_json.velocity }}",
            "icon": "mdi:speedometer",
            "device": DEVICE_INFO,
        }

        # CSI Energy Surge Ratio
        topic_s = f"{discovery_prefix}/sensor/{hub_id}/csi_node_{nid}_surge/config"
        configs[topic_s] = {
            "name": f"CSI Node {nid} Energy Surge",
            "unique_id": f"{hub_id}_csi_node_{nid}_surge",
            "state_topic": f"{base_topic}/csi/{nid}",
            "value_template": "{{ value_json.surge }}",
            "icon": "mdi:waveform",
            "device": DEVICE_INFO,
        }

    # 8. Button: Alarm Reset Command
    topic_btn = f"{discovery_prefix}/button/{hub_id}/reset_alarm/config"
    configs[topic_btn] = {
        "name": "Reset Fall Alarm",
        "unique_id": f"{hub_id}_reset_alarm",
        "device_class": "restart",
        "command_topic": f"{base_topic}/command/reset",
        "payload_press": "RESET",
        "icon": "mdi:bell-cancel",
        "device": DEVICE_INFO,
    }

    return configs


class HomeAssistantMQTTDiscoveryManager:
    """Manages publishing Home Assistant discovery payloads and sensor state updates."""

    def __init__(
        self,
        mqtt_client: Any = None,
        discovery_prefix: str = "homeassistant",
        base_topic: str = "fall_detection",
        node_ids: Tuple[int, ...] = (1, 2, 3),
    ):
        self.client = mqtt_client
        self.discovery_prefix = discovery_prefix
        self.base_topic = base_topic
        self.node_ids = node_ids
        self._configs = build_discovery_configs(discovery_prefix, base_topic, node_ids)

    def announce_discovery(self) -> int:
        """Publish discovery configurations to MQTT broker with retain=True.

        Returns:
            int: Total number of entity configurations announced.
        """
        count = 0
        for topic, payload in self._configs.items():
            payload_str = json.dumps(payload)
            if self.client and hasattr(self.client, "publish"):
                try:
                    self.client.publish(topic, payload_str, retain=True)
                except Exception as e:
                    logger.error(f"Failed to publish HA discovery to {topic}: {e}")
            count += 1
        return count

    def remove_discovery(self) -> int:
        """Clear discovery entities from Home Assistant by publishing empty retained messages."""
        count = 0
        for topic in self._configs.keys():
            if self.client and hasattr(self.client, "publish"):
                try:
                    self.client.publish(topic, "", retain=True)
                except Exception as e:
                    logger.error(f"Failed to clear HA discovery on {topic}: {e}")
            count += 1
        return count

    def publish_state(self, state: str):
        """Publish unified fall detection state (e.g. 'Normal', 'Fall Suspected', 'FALL DETECTED')."""
        topic = f"{self.base_topic}/state"
        if self.client and hasattr(self.client, "publish"):
            self.client.publish(topic, state, retain=False)

    def publish_radar(self, height: float, posture: str, fall_state: str, dwell: int = 0):
        """Publish mmWave radar telemetry JSON."""
        topic = f"{self.base_topic}/radar"
        payload = json.dumps({
            "height": round(float(height), 2),
            "posture": posture,
            "fall_state": fall_state,
            "dwell": int(dwell),
        })
        if self.client and hasattr(self.client, "publish"):
            self.client.publish(topic, payload, retain=False)

    def publish_csi(self, node_id: int, velocity: float, surge: float, state: str):
        """Publish CSI dynamic feature telemetry JSON for a specific tracker node."""
        topic = f"{self.base_topic}/csi/{node_id}"
        payload = json.dumps({
            "velocity": round(float(velocity), 2),
            "surge": round(float(surge), 2),
            "state": state,
        })
        if self.client and hasattr(self.client, "publish"):
            self.client.publish(topic, payload, retain=False)

    def publish_ml_probability(self, probability: float):
        """Publish ML classifier fall probability JSON."""
        topic = f"{self.base_topic}/ml"
        payload = json.dumps({
            "probability": round(float(probability), 4),
        })
        if self.client and hasattr(self.client, "publish"):
            self.client.publish(topic, payload, retain=False)
