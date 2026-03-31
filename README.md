# inf2009_proj

Pi-side services for CSI collection, calibration, training, and ESP32 model delivery.

This repository works together with `edge-esp32`:

- the ESP32 firmware captures CSI and uploads sub-batches over HTTP
- this repo receives and merges the data, manages calibration, trains models, and publishes model-load commands

For profiling notes and PASO methodology, see `TECHNICAL_REFERENCE.md`.

## What’s in here

| File / folder | Purpose |
| --- | --- |
| `server.py` | Flask ingest server and MQTT bridge for CSI uploads |
| `dashboard.py` | Desktop dashboard for calibration, node status, and model workflow |
| `launch.sh` | Helper launcher for `server.py`, `dashboard.py`, and `broadcast_generator.py` |
| `train_improved_grouped_model.py` | Portable grouped-feature training pipeline |
| `edge_ml.py` | TensorFlow-based training/export pipeline |
| `export_model_for_esp32.py` | Utility for aligning and exporting model artifacts for ESP32 use |
| `push_model.py` | Copies model artifacts into `model_store/<NODE>/` and can trigger a load |
| `broadcast_generator.py` | Optional UDP traffic helper for testing CSI capture |
| `reset_helpers.py` | Removes node data/model artifacts and can trigger cleanup actions |
| `shared_config.py` | Shared constants for CSI geometry, feature windowing, and defaults |
| `csi_data/` | Collected CSI CSVs, organized by node and label |
| `model_store/` | Per-node active model artifacts plus backups |
| `profiling/` | PASO and performance profiling outputs |
| `calib_states.json`, `config.json`, `keys.json`, `view_state.json`, `heartbeat_state.json` | Runtime state persisted by the dashboard and scripts |

## Setup

### Prerequisites

- Python 3.9+
- Raspberry Pi OS or another Linux environment
- Mosquitto broker

### Create a Python environment

```bash
cd inf2009_proj
python3 -m venv inf2009_venv
source inf2009_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps -r requirements.txt
```

If you prefer the packaged install, `python -m pip install .` also works in supported environments.

### Configure environment variables

Copy the template and fill in the values that match your Pi and broker:

```bash
cp .env.example .env
```

At minimum, check these values in `.env`:

- `MQTT_BROKER`
- `MQTT_PORT`
- `SERVER_HOST`
- `SERVER_PORT`
- `MODEL_STORE_DIR`
- `CSI_DATA_DIR`

## How to run

### Start the main services

Open separate terminals and run:

```bash
cd inf2009_proj
source venv/bin/activate # if you created a virtual environment
python ./launch.sh # starts server and dashboard
```

### Optional helper

```bash
python broadcast_generator.py
```

### One-command launcher

`launch.sh` can start the services interactively or in the background:

```bash
./launch.sh
./launch.sh --headless
./launch.sh --headless --components server,dashboard
./launch.sh --status
./launch.sh --stop --components server,dashboard
```

In headless mode, logs go to `mqtt_logs/` and PID files go to `.run/`.

## How to use the project

### 1. Collect calibration data

Use the dashboard to mark calibration states such as `door_closed`, `door_open`, and `person_standing`, then trigger collection from the ESP32 side.

### 2. Train a model

For the portable grouped pipeline:

```bash
python train_improved_grouped_model.py
```

For the notebook-derived TensorFlow flow:

```bash
python edge_ml.py \
  --data-dir csi_data \
  --node RACK_1 \
  --output model_store/_tmp/RACK_1/model.tflite \
  --scaler-output model_store/_tmp/RACK_1/scaler_params.json
```

### 3. Push the model to a node

```bash
python push_model.py model_store/RACK_1/model.tflite --node RACK_1 --load
```

You can also provide a custom scaler file with `--params`.

### 4. Manage or reset node data

```bash
python reset_helpers.py RACK_1
```

### 5. Inspect profiling output

```bash
profiling/scripts/run_worktree_compare.sh <baseline_commit>
```

## Things to note

- MQTT topic names must stay in sync with `edge-esp32`, especially `/commands/<node>/collect`, `/commands/<node>/training_complete`, `/commands/<node>/load_model`, and `/sensors/<node>/status`.
- The current training path uses the grouped feature layout: 8 averaged CSI groups plus `AVG_VARIATION`.
- `model_store/<NODE>/` contains the active artifacts; `model_store/_tmp/` is for temporary training output.
- `heartbeat_state.json` is a local runtime file used by the dashboard to avoid stale online/offline state.
- TensorFlow is optional; if it is unavailable, use `train_improved_grouped_model.py`.
- `server.py` exposes `/health` for firmware profile validation before the ESP32 settles on a Wi-Fi/MQTT/HTTP combination.
- For firmware build and flash steps, see `edge-esp32/README.md`.
