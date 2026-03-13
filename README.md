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
pip install flask numpy pandas paho-mqtt customtkinter scikit-learn tensorflow
```

No separate requirements file is used; the packages installed above are all that
are needed by `dashboard.py`, `server.py`, and the helper scripts.

## Running

Run the server and dashboard in separate terminals:

```bash
python server.py       # terminal 1: HTTP ingest + MQTT merge-complete publish
python dashboard.py    # terminal 2: UI + MQTT orchestration
```

## Model training (notebook + CLI)

`Edge_ML.ipynb` is kept as the reference notebook for feature selection,
normalization, model architecture, and TFLite conversion.

`dashboard.py` calls `train_model.py` after calibration is complete. You can
also run it manually:

```bash
python train_model.py \
  --data-dir csi_data \
  --node RACK_1 \
  --output model_store/_tmp/RACK_1/model.tflite \
  --scaler-output model_store/_tmp/RACK_1/scaler_params.json
```

The script:

* loads CSVs from `csi_data/<node>/<state>/...` recursively,
* uses notebook-compatible features (`SC_4..SC_59` excluding `SC_32`),
* trains a compact dense classifier and exports `.tflite`,
* emits `scaler_params.json` in `{"mean": [...], "std": [...]}` format.

### ESP32-C3 limitations

* Keep models compact (warning threshold is ~100 KB in the trainer).
* Keep input feature count fixed to the notebook-compatible set (55 features).
* Prefer early stopping / modest model depth to reduce RAM and CPU load.

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