"""Unit tests for Home Assistant MQTT Auto-Discovery and Automations."""

import json
from pathlib import Path
import pytest
import yaml

from hub.ha_discovery import (
    DEVICE_INFO,
    HomeAssistantMQTTDiscoveryManager,
    build_discovery_configs,
)


class MockMQTTClient:
    """Mock MQTT client tracking publish calls."""

    def __init__(self):
        self.published = []

    def publish(self, topic: str, payload: str = "", retain: bool = False):
        self.published.append({
            "topic": topic,
            "payload": payload,
            "retain": retain,
        })


def test_build_discovery_configs_structure():
    """Verify all entities conform to Home Assistant discovery specs."""
    configs = build_discovery_configs(
        discovery_prefix="homeassistant",
        base_topic="fall_detection",
        node_ids=(1, 2, 3),
    )

    # 1 binary sensor + 5 scalar sensors (state, height, posture, dwell, ml)
    # + 6 node sensors (2 per node for 3 nodes) + 1 reset button = 13 entities
    assert len(configs) == 13

    # Check binary sensor
    binary_topic = "homeassistant/binary_sensor/fall_detection_hub/fall_detected/config"
    assert binary_topic in configs
    b_conf = configs[binary_topic]
    assert b_conf["device_class"] == "safety"
    assert b_conf["payload_on"] == "ON"
    assert b_conf["payload_off"] == "OFF"
    assert b_conf["device"]["identifiers"] == ["fall_detection_hub"]

    # Check radar sensors
    h_topic = "homeassistant/sensor/fall_detection_hub/radar_height/config"
    assert h_topic in configs
    assert configs[h_topic]["unit_of_measurement"] == "m"

    # Check button
    btn_topic = "homeassistant/button/fall_detection_hub/reset_alarm/config"
    assert btn_topic in configs
    assert configs[btn_topic]["command_topic"] == "fall_detection/command/reset"

    # Verify JSON serializability of all payloads
    for topic, payload in configs.items():
        assert topic.startswith("homeassistant/")
        serialized = json.dumps(payload)
        assert len(serialized) > 0


def test_discovery_manager_announce_and_remove():
    """Verify manager announces discovery configs with retain=True and cleans up."""
    mock_client = MockMQTTClient()
    manager = HomeAssistantMQTTDiscoveryManager(
        mqtt_client=mock_client,
        discovery_prefix="homeassistant",
        base_topic="fall_detection",
        node_ids=(1, 2),
    )

    # Announce
    count = manager.announce_discovery()
    assert count == len(mock_client.published)
    assert all(msg["retain"] is True for msg in mock_client.published)

    # Clear mock calls
    mock_client.published.clear()

    # Remove
    clear_count = manager.remove_discovery()
    assert clear_count == count
    assert all(msg["payload"] == "" and msg["retain"] is True for msg in mock_client.published)


def test_discovery_manager_publish_telemetry():
    """Verify state updates are published to the correct topics without retain."""
    mock_client = MockMQTTClient()
    manager = HomeAssistantMQTTDiscoveryManager(
        mqtt_client=mock_client,
        base_topic="fall_detection",
    )

    # 1. State update
    manager.publish_state("FALL DETECTED")
    assert mock_client.published[-1]["topic"] == "fall_detection/state"
    assert mock_client.published[-1]["payload"] == "FALL DETECTED"
    assert mock_client.published[-1]["retain"] is False

    # 2. Radar update
    manager.publish_radar(height=0.25, posture="Lying Down", fall_state="FALL_CONFIRMED", dwell=8)
    assert mock_client.published[-1]["topic"] == "fall_detection/radar"
    radar_data = json.loads(mock_client.published[-1]["payload"])
    assert radar_data["height"] == 0.25
    assert radar_data["posture"] == "Lying Down"
    assert radar_data["dwell"] == 8

    # 3. CSI update
    manager.publish_csi(node_id=1, velocity=2.34, surge=4.12, state="Fall Suspected")
    assert mock_client.published[-1]["topic"] == "fall_detection/csi/1"
    csi_data = json.loads(mock_client.published[-1]["payload"])
    assert csi_data["velocity"] == 2.34
    assert csi_data["surge"] == 4.12

    # 4. ML Probability update
    manager.publish_ml_probability(0.9412)
    assert mock_client.published[-1]["topic"] == "fall_detection/ml"
    ml_data = json.loads(mock_client.published[-1]["payload"])
    assert ml_data["probability"] == 0.9412


def test_ha_automations_yaml_syntax():
    """Verify config/ha_automations.yaml parses valid Home Assistant automation schemas."""
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "ha_automations.yaml"
    assert yaml_path.exists()

    with open(yaml_path, "r", encoding="utf-8") as f:
        automations = yaml.safe_load(f)

    assert isinstance(automations, list)
    assert len(automations) >= 3

    ids = [a["id"] for a in automations]
    assert "fall_detection_emergency_alert" in ids
    assert "fall_detection_suspected_warning" in ids
    assert "fall_detection_recovered_clear" in ids
