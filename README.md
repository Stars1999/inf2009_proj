# inf2009_proj

This folder contains the Pi‑side Python services for the CSI project. After
migrating `server.py` from the `edge-esp32` repo, both the dashboard and the
HTTP ingest server live here.

## Setup

```bash
cd inf2009_proj
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# edit .env — set MQTT_BROKER to your Pi's IP or localhost
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MQTT_BROKER` | `localhost` | MQTT broker hostname / IP |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `SERVER_HOST` | `0.0.0.0` | Flask ingest server bind address |
| `SERVER_PORT` | `5000` | Flask ingest server port |
| `MODEL_STORE_DIR` | `model_store` | Directory where trained models are saved |
| `CSI_DATA_DIR` | `csi_data` | Directory where CSI CSVs are written |

## Running

Run the server and dashboard in separate terminals:

```bash
python server.py       # terminal 1: HTTP ingest + MQTT merge-complete publish
python dashboard.py    # terminal 2: UI + MQTT orchestration
```

## Model training (notebook + CLI)

`Edge_ML (1).ipynb` is the source-of-truth for model training behavior.

The canonical production trainer is `edge_ml.py` (a CLI-aligned version of the
notebook logic).

`dashboard.py` calls `edge_ml.py` after calibration is complete. You can also
run it manually:

```bash
python edge_ml.py \
  --data-dir csi_data \
  --node RACK_1 \
  --output model_store/_tmp/RACK_1/model.tflite \
  --scaler-output model_store/_tmp/RACK_1/scaler_params.json
```

The script:

* loads CSVs from `csi_data/<node>/<state>/...` recursively,
* keeps notebook feature-selection behavior (variance threshold on `SC_4..SC_59`, excluding `SC_32`),
* appends the notebook variation feature (`AVG_VARIATION`),
* trains the notebook dense classifier and exports strict int8 `.tflite`,
* emits `scaler_params.json` with `mean/std` arrays and notebook metadata used by ESP32 inference.

### ESP32-C3 limitations

* Keep models compact (int8 export and early stopping help memory use).
* Feature count is data-dependent (variance-selected subcarriers + 1 variation feature).
* Ensure model and scaler are updated together for each node.

## Notes

* Configuration (broker, port, CSI data directory) is stored in
  `config.json` and used by both components.
* CSI uploads are saved under `csi_data/<node>/<state>/<session>`; the server accepts an optional `X-Session-ID` header to differentiate concurrent collections.
* After a full set of sub-batches is uploaded, the server **merges them into a CSV and then publishes a `collection_complete` MQTT event** with a `session` field.
* The dashboard prefers this server-authoritative completion event for status/logging, but if it is delayed/missing it can use firmware upload-complete as a fallback unlock path so calibration does not remain blocked.
* Trained models and scaler parameters live in `model_store/<node>`.
* Models can be fetched via `/model/<node>` or the legacy `/static/models/<node>` URL; the service will return either `model.tflite` or a file matching the node name.

### MQTT training/download flow

Rather than immediately pushing a model to the ESP32 as soon as training starts,
`dashboard.py` now waits until training has successfully finished and the
artifacts have been copied to the server.  At that point it publishes a
`/commands/<node>/training_complete` message (JSON payload contains a
`session` string).  The firmware subscriptions include this topic; receiving it
arms model download and starts a pull of new model + scaler parameters.  The
old `/commands/<node>/update_model` topic remains for backwards compatibility,
but it is ignored until at least one `training_complete` message has been seen.

This guarantees that the ESP will not begin its HTTP pull until the model is
fully trained and committed to the server, and lets the dashboard know when the
training step succeeded.

For ESP32 firmware instructions refer to the top‑level
`edge-esp32/README.md`.