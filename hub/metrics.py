"""Prometheus Metrics Exporter for Fall Detection SaMD Hub (Milestone 17.2)."""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Counters
fall_alerts_total = Counter(
    "fall_alerts_total",
    "Total fall alerts dispatched by the system",
    ["room_id", "severity"],
)

hl7_messages_received = Counter(
    "hl7_messages_received_total",
    "Total HL7 ADT messages ingested via MLLP",
    ["msg_type"],
)

retrain_events_total = Counter(
    "retrain_events_total",
    "Total adaptive model retraining pipeline runs",
    ["outcome"],
)

# Gauges
active_rooms = Gauge(
    "active_rooms_total",
    "Number of rooms currently actively monitored",
)

drift_psi = Gauge(
    "model_drift_psi",
    "Current Population Stability Index (PSI) drift score",
)

buffer_size = Gauge(
    "training_buffer_size",
    "Current number of samples buffered for incremental retraining",
)

# Histograms
fall_detection_latency = Histogram(
    "fall_detection_latency_seconds",
    "Processing latency from CSI/Radar packet ingestion to fall decision",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)


def get_metrics_text() -> bytes:
    """Serialize all metrics to Prometheus text exposition format."""
    return generate_latest()
