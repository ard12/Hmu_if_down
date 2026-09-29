"""Unit tests for AlertDispatcher MQTT and webhook extensions."""

import json
from unittest.mock import MagicMock, patch
import pytest

from hub.alert_dispatcher import AlertDispatcher


def test_default_init(tmp_path):
    dispatcher = AlertDispatcher(enable_sound=False, log_dir=str(tmp_path))
    assert dispatcher._mqtt_client is None
    assert dispatcher._webhook_url is None
    assert dispatcher._mqtt_topic == "falldetect/alert"
    assert dispatcher.cooldown_sec == 5.0


def test_mqtt_paho_missing_warning(tmp_path, capsys, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "paho" in name:
            raise ImportError("No module named 'paho'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    dispatcher = AlertDispatcher(
        enable_sound=False,
        log_dir=str(tmp_path),
        mqtt_broker="127.0.0.1",
        mqtt_port=1883,
        mqtt_topic="test/topic",
    )
    assert dispatcher._mqtt_client is None
    assert dispatcher._mqtt_topic == "test/topic"
    captured = capsys.readouterr()
    assert "paho-mqtt is not installed" in captured.out or "paho-mqtt is not installed" in captured.err


def test_mqtt_publish_payload(tmp_path):
    dispatcher = AlertDispatcher(
        enable_sound=False,
        log_dir=str(tmp_path),
        mqtt_topic="home/falls",
    )
    mock_client = MagicMock()
    dispatcher._mqtt_client = mock_client

    dispatcher.trigger_alarm(modality="csi", event_name="FALL_CONFIRMED", details="CSI drop detected")

    mock_client.publish.assert_called_once()
    call_args = mock_client.publish.call_args
    topic = call_args[0][0]
    payload_str = call_args[0][1]

    assert topic == "home/falls"
    payload = json.loads(payload_str)
    assert payload["modality"] == "csi"
    assert payload["event"] == "FALL_CONFIRMED"
    assert payload["details"] == "CSI drop detected"
    assert "timestamp" in payload


def test_webhook_post_payload(tmp_path):
    dispatcher = AlertDispatcher(
        enable_sound=False,
        log_dir=str(tmp_path),
        webhook_url="https://alerts.example.com/webhook",
    )
    assert dispatcher._webhook_url == "https://alerts.example.com/webhook"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        dispatcher.trigger_alarm(modality="radar", event_name="FALL_CONFIRMED", details="Height=0.2m")

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://alerts.example.com/webhook"
        assert req.get_method() == "POST"
        assert req.headers.get("Content-type") == "application/json"

        body = json.loads(req.data.decode("utf-8"))
        assert body["modality"] == "radar"
        assert body["event"] == "FALL_CONFIRMED"
        assert body["details"] == "Height=0.2m"
        assert "timestamp" in body


def test_cooldown_suppresses_secondary_dispatch(tmp_path):
    dispatcher = AlertDispatcher(
        enable_sound=False,
        log_dir=str(tmp_path),
        cooldown_sec=10.0,
        webhook_url="https://alerts.example.com/webhook",
    )
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        dispatcher.trigger_alarm("camera", "FALL_DETECTED", "Subject prone")
        assert mock_urlopen.call_count == 1

        # Second trigger within cooldown
        dispatcher.trigger_alarm("camera", "FALL_DETECTED", "Subject still prone")
        assert mock_urlopen.call_count == 1


def test_network_error_resilience(tmp_path):
    dispatcher = AlertDispatcher(
        enable_sound=False,
        log_dir=str(tmp_path),
        webhook_url="https://bad.invalid/webhook",
    )
    mock_client = MagicMock()
    mock_client.publish.side_effect = RuntimeError("MQTT connection lost")
    dispatcher._mqtt_client = mock_client

    with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
        # Should not raise exception
        dispatcher.trigger_alarm("csi", "FALL_CONFIRMED", "test error resilience")


def test_incident_csv_rotation_and_callback(tmp_path):
    """
    Covers: SRS-016
    F-12, F-19: Verifies decoupled on_alert callback and size-based incident CSV rotation.
    """
    received_alerts = []
    def alert_listener(modality, event, details):
        received_alerts.append((modality, event, details))

    dispatcher = AlertDispatcher(
        enable_sound=False,
        log_dir=str(tmp_path),
        cooldown_sec=0.0,
        on_alert=alert_listener,
    )

    dispatcher.trigger_alarm("fusion", "FALL_CONFIRMED", "Patient collapse in Room 101")
    assert len(received_alerts) == 1
    assert received_alerts[0][0] == "fusion"

    # Simulate large CSV file (> 10MB) to test rotation
    with open(dispatcher.csv_path, "wb") as f:
        f.write(b"0" * (10 * 1024 * 1024 + 100))

    # Next alarm trigger should trigger rotation
    dispatcher.trigger_alarm("radar", "FALL_CONFIRMED", "Bed fall detected")
    rotated_files = list(tmp_path.glob("incidents_*.csv"))
    assert len(rotated_files) >= 1
    assert dispatcher.csv_path.exists()

