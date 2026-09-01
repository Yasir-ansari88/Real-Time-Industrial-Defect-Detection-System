import time

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

PREDICTION_COUNTER = Counter(
    "defect_detection_predictions_total",
    "Total number of prediction requests served",
)

INFERENCE_LATENCY = Histogram(
    "defect_detection_inference_seconds",
    "Model inference latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

UPTIME_START_TIME = Gauge(
    "defect_detection_start_time_seconds",
    "Unix timestamp when the service started",
)

def track_uptime_start():
    UPTIME_START_TIME.set(time.time())

def render_metrics() -> bytes:
    return generate_latest()

__all__ = [
    "PREDICTION_COUNTER",
    "INFERENCE_LATENCY",
    "UPTIME_START_TIME",
    "track_uptime_start",
    "render_metrics",
    "CONTENT_TYPE_LATEST",
]
