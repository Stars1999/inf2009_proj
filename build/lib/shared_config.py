import os

# CSI geometry shared across training and server ingest.
MAX_LOWER = 4
MAX_UPPER = 60
DC_NULL = 32
CSI_HEADERS = [f"SC_{i}" for i in range(MAX_LOWER, MAX_UPPER + 1) if i != DC_NULL]

# Network defaults used by dashboard/server when .env is not set.
DEFAULT_MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
DEFAULT_MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Upload geometry (must stay aligned with firmware unless intentionally changed).
DEFAULT_SUB_BATCH_SIZE = int(os.getenv("SUB_BATCH_SIZE", "40"))

# Keep training defaults aligned with firmware extraction window.
DEFAULT_FEATURE_WINDOW = int(os.getenv("FEATURE_WINDOW", "16"))

# Feature layout identifiers shared across training and model metadata.
FEATURE_MODE_GROUPED = "grouped"
FEATURE_MODE_SELECTED_SUBCARRIERS = "selected_subcarriers"
