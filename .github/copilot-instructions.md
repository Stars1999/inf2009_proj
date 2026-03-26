# Copilot instructions for inf2009_proj

Purpose
- Provide precise, repository-aware instructions so future Copilot sessions can act with accurate context (build/run/test/deploy, architecture, and conventions).

---

1) Fast setup / build / run commands

- Create and activate venv, install pinned deps (project uses many ML libs including TF; TF may fail on some Python versions):
  python3 -m venv inf2009_venv
  source inf2009_venv/bin/activate
  pip install --upgrade pip
  pip install --no-deps -r requirements.txt
  # Preferred venv name: inf2009_venv/ (the owner uses /home/iankoh/inf2009_venv)

- Start main services (two terminals):
  python server.py
  python dashboard.py

- Start components with helper script (single command launcher):
  ./launch.sh             # interactive terminals
  ./launch.sh --headless  # background (writes logs + pid files)
  ./launch.sh --headless --components server,broadcast
  ./launch.sh --status
  ./launch.sh --stop --components server,broadcast

- Train (ESP32-compatible grouped pipeline — non-TF, sklearn):
  python train_improved_grouped_model.py
  # This writes model_store/<NODE>/scaler_params.json (default node: RACK_1)

- Export / prepare model metadata for ESP32 (optional):
  python export_model_for_esp32.py
  # Produces joblib/model_config and updates scaler metadata where needed

- Push model + scaler to server and trigger device load (uses .env MQTT settings):
  python push_model.py model_store/RACK_1/model.tflite --node RACK_1 --load
  python push_model.py --node RACK_1 --load  # trigger load of existing model

- Verify scaler/model pairing (quick check):
  python - <<'PY'
import json
j=json.load(open('model_store/RACK_1/scaler_params.json'))
print('features', len(j.get('feature_columns',[])), 'feature_mode', j.get('notebook_alignment',{}).get('feature_mode'))
PY

Notes on testing and linting
- There are no automated unit tests or CI scripts in the repository root. Use the single-script commands above to run components. If test files are added later, run them directly with pytest or the supplied runner.

---

2) High-level architecture (big picture)

- Data collection
  - Devices publish CSI samples to the Pi-side server via MQTT; raw data are stored under `csi_data/<NODE>/<STATE>/*.csv`.
  - States present in this project: `door_open`, `door_closed`, `person_standing`.

- Server & dashboard
  - `server.py` handles MQTT ingestion and the API endpoint surface.
  - `dashboard.py` is the UI and orchestration tool: calibration, triggering training, and model lifecycle handlers.
  - `launch.sh` is a convenience wrapper to start server/dashboard/broadcast components (headless option creates log and pid files under `.run` and `mqtt_logs`).

- Training & feature pipeline
  - `Edge_ML (1).ipynb` is the canonical original training notebook and primary source of truth for feature engineering and model architecture. All changes should be based on it unless a clear, documented improvement or bug fix is found.
  - `edge_ml.py` implements the TF-based training/export flow derived from the notebook. Note: TF may not be available on all environments; treat this as experimental unless TF compatibility is verified.
  - `train_improved_grouped_model.py` is a portable sklearn pipeline used to produce ESP32-compatible grouped features (56 SC → 8 groups + AVG_VARIATION). Use when TF is unavailable or for quick iteration.
  - `export_model_for_esp32.py` helps convert/align trained sklearn models (joblib + JSON metadata) and update scaler metadata for TFLite/ESP32 use.

- Deployment
  - `model_store/<NODE>/` holds per-node artifacts: `model.tflite` and `scaler_params.json` (push_model.py makes backups: `.backup`).
  - `push_model.py` copies files into `model_store/<NODE>/`, validates schema, and can publish an MQTT `/commands/<NODE>/load_model` to tell the ESP32 to load the new assets.

- Device/firmware contract
  - Firmware expects grouped input layout: 8 averaged groups + AVG_VARIATION (9 floats total). See `model_store/RACK_1/scaler_params.json` for the canonical format used here.
  - Core MQTT topics used by server and firmware: `/commands/<node>/load_model`, `/sensors/<node>/status`, `/commands/<node>/collect`, `/commands/<node>/training_complete`.
  - Cross-repo compatibility:
    - This workspace contains two repositories: `inf2009_proj` (Pi-side: server, training, deployment) and `edge-esp32` (firmware). Keep them compatible when changing feature layout, scaling, or label ordering.
    - Key checks before deployment:
      - `receiver.cpp` grouping must match training grouping (SC_4..SC_60 excluding SC_32 → groups of 7 in the same order).
      - Firmware label index mapping must match `label_order` in `scaler_params.json`.
      - Quantization/min-max ranges used in firmware inference must be consistent with scaler `min`/`max` or quantization metadata.
    - If you change grouping or feature ordering, update firmware (`edge-esp32/receiver.cpp`) and retrain with the exact grouping.

---

3) Key repo conventions and patterns (important for Copilot)

- Feature layout
  - Raw subcarriers: SC_4 .. SC_60 inclusive, excluding SC_32 (the DC null). There are 56 usable SCs.
  - Grouping convention used by training/firmware: split the 56 SCs into 8 groups of 7 in ascending SC order. GROUP_0 corresponds to the first 7 SCs (SC_4..SC_10), GROUP_1 the next 7, and so on.
  - Final feature vector order is: [GROUP_0, ..., GROUP_7, AVG_VARIATION]. AVG_VARIATION is computed as mean(std(window, axis=0)) over a sliding window.

- Scaler metadata schema (scaler_params.json)
  - Required fields used by server/firmware: `schema_version`, `scaler_type`, `mean`, `std`, `min`, `max`, `feature_columns`, `notebook_alignment`, `training_metadata`, `label_order`.
  - `notebook_alignment` includes: `feature_mode` ("grouped" or other), `group_count` (8 for grouped), `train_baseline`, `use_session_offset` and `session_offset_formula`. The firmware & push script use `group_count + 1 == len(mean)` to validate grouped layout.
  - `label_order` shows the mapping from model output indices → labels. Keep this stable between training/export and firmware interpretation. Current order: `["door_open","door_closed","person_standing"]`.

- Deployment & backups
  - `push_model.py` will backup existing `model.tflite` and `scaler_params.json` to `.backup` before replacing. To restore a previous artifact, copy `<file>.backup` back into the node folder or run `git checkout` if previously versioned.

- Environment & Python runtime
  - `launch.sh` attempts to resolve Python in this order: explicit PYTHON_BIN env var → `/home/iankoh/inf2009_venv/bin/python3` → `./inf2009_venv/bin/python3` → system python3. Preferred venv: `inf2009_venv/` (the project owner's venv path is `/home/iankoh/inf2009_venv`). Copilot actions that run scripts should activate this venv before running commands.
  - `.env.example` → copy to `.env` and set `MQTT_BROKER`/`MQTT_PORT` for push_model.py and server to connect.

- Model compatibility checks
  - Use `push_model.py`'s `validate_scaler_schema()` logic when updating scaler JSON; it enforces either grouped layout (group_count+1 == len(mean)) or selected_subcarriers layout.
  - If TensorFlow tooling is unavailable in the environment, prefer `train_improved_grouped_model.py` + `export_model_for_esp32.py` which produce scaler metadata and joblib artifacts. Rebuilding a TFLite model requires TF (and a compatible Python version).

- Logging & debugging artifacts
  - MQTT logs are stored in `mqtt_logs/` (server-side captures) and `heartbeat_state.json` persists heartbeat timestamps.
  - `model_store/<NODE>/` contains both active and `.backup` files; inspect these when device reports compatibility errors.

---

4) Quick diagnostics Copilot should try when asked to debug a misclassification or model-loading issue

- Confirm the node loaded the updated scaler and model: subscribe to `/sensors/<NODE>/status` and confirm `model_ready: true` and no `model_download_incompatible` events.
- Confirm `model_store/<NODE>/scaler_params.json` has `feature_columns` length == expected feature count.
- If device reports `input_elements_not_multiple_of_features` or similar, check `feature_mode`+`group_count` vs `len(mean)`.
- To validate a sample from the device, request the device to publish computed grouped features (GROUP_0..GROUP_7 + AVG_VARIATION) and compute distances to training centroids (the repository contains analysis scripts that compute centroids from `csi_data/`).

---

Where to look for more context
- In-repo docs: `README.md`, `TECHNICAL_REFERENCE.md` (PASO/profiling), and `profiling/scripts` for measured workflows.
- Training code: `edge_ml.py` (TF pipeline; may be experimental), `train_improved_grouped_model.py` (sklearn grouped pipeline), `export_model_for_esp32.py`.
- Deployment: `push_model.py`, `model_store/<NODE>/`.
- Data: `csi_data/<NODE>/` (raw CSVs).

---

Final notes
- This file is intentionally prescriptive about the runtime contract (feature layout, metadata schema, MQTT topics) because those are critical for model inference compatibility across server and firmware.
- If you want a follow-up: add a small `scripts/` helper that validates a `model.tflite` vs `scaler_params.json` pair locally (attempt tflite input inspection where tflite_runtime available), or create a unit test runner for the feature extraction logic.

---

If you'd like edits (ease-of-use shortcuts, extra verification scripts, or MCP server configuration for testing), say which area to expand and I’ll update this file.
