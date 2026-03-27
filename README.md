# inf2009_proj

Pi-side services for CSI collection, model training, and ESP32 model delivery.

For report-grade profiling and PASO methodology, see `TECHNICAL_REFERENCE.md`.

## Setup (UV packaging)

This project now supports PEP 621/`pyproject.toml` packaging with optional UV tooling.

### Option A: standard package install

```bash
cd inf2009_proj
python3 -m venv inf2009_venv
source inf2009_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

### Option B: using UV packaging helper

```bash
cd inf2009_proj
python3 -m venv inf2009_venv
source inf2009_venv/bin/activate
python -m pip install --upgrade pip
pip install uv
uv install
```

### Legacy requirements file

If you need strict pinned dependencies in this repo (for reproducibility):

```bash
pip install --no-deps -r requirements.txt
```

Copy environment variables:

```bash
cp .env.example .env
```

## Run

Start in separate terminals:

```bash
python server.py
python dashboard.py
```

Optional UDP broadcaster:

```bash
python broadcast_generator.py
```

One-command launcher (interactive terminals or headless):

```bash
./launch.sh
./launch.sh --headless
./launch.sh --status
./launch.sh --stop
```

## Train model

`dashboard.py` invokes `edge_ml.py` after calibration, or run manually:

```bash
python edge_ml.py \
  --data-dir csi_data \
  --node RACK_1 \
  --output model_store/_tmp/RACK_1/model.tflite \
  --scaler-output model_store/_tmp/RACK_1/scaler_params.json
```

Push model to ESP32:

```bash
python push_model.py model_store/RACK_1/model.tflite --node RACK_1 --load
```

## Core MQTT topics

- `/commands/<node>/collect`
- `/commands/<node>/training_complete`
- `/commands/<node>/load_model`
- `/sensors/<node>/status`
- `/sensors/<node>/perf_bin`
- `device/<node>/status`

## Node presence and liveliness model (dashboard)

- Dashboard now derives node presence from both `device/<node>/status` and `/sensors/<node>/status` heartbeat events.
- Presence states:
  - `ONLINE`: explicit online + fresh heartbeat
  - `STALE`: explicit online but heartbeat timed out (fallback safety state)
  - `OFFLINE`: explicit offline or never seen
- Liveliness defaults:
  - heartbeat timeout: `90s`
  - liveliness check interval: `30s`
- Dashboard persists heartbeat timestamps in `heartbeat_state.json` (local runtime file, gitignored) to avoid ghost online state after restarts.

## Profiling (before vs after, non-destructive)

Run baseline-vs-current profiling without reverting your working tree:

```bash
profiling/scripts/run_worktree_compare.sh <baseline_commit>
```

Outputs in `profiling/results/`:

- `before_baseline.json`, `after_optimized.json`
- `perf_before.csv`, `perf_after.csv`
- `summary_table.csv`, `summary_table.md`
- `runbook.md`

Method/workload lock files:

- `profiling/scripts/profile_matrix.json`
- `profiling/scripts/workload_lock.json`

For firmware-specific instructions, see `edge-esp32/README.md`.
