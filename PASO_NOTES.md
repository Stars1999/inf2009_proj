# PASO Implementation Notes (Profiling Branch)

## Best-practice references consulted

The implementation choices were guided by docs research gathered from Context7 queries:

1. **Flask background work / non-blocking routes** (`/pallets/flask`)
   - Pattern: avoid long-running work inside request handlers; offload to background workers/queues.
   - Applied in `server.py` by queueing finalize jobs and processing in background workers.

2. **TensorFlow Lite optimization workflow** (`/tensorflow/docs`)
   - Pattern: use representative datasets for INT8 quantization and evaluate post-conversion model quality.
   - Applied in `edge_ml.py` with representative dataset conversion and pruning+accuracy guardrail.

## PASO phase mapping

- **Profiling**
  - ESP32 `Invoke()` microsecond instrumentation in `edge-esp32/main/tflm_inference.cpp`.
  - Server callback profiling via `@profile` in `server.py`.

- **Analysing**
  - Budget constants and alerts in both firmware and server.
  - Stage-level timing logs and perf payload decode.

- **Scheduling**
  - Existing CSI callback -> inference task split retained and profiled.
  - Server heavy finalize path moved to async worker queue with overflow worker path.

- **Optimisation**
  - Model: optional magnitude pruning + guardrail (`--max-accuracy-drop`) before TFLite export.
  - Feature condensation: grouped feature mode in trainer and firmware metadata/model layout support.
  - Payload: packed binary telemetry (`/sensors/<node>/perf_bin`) for lower overhead metric transport.

## Safety and compatibility

- Existing JSON command/status topics remain intact.
- New optimization paths are opt-in via training flags and environment variables.
- Queue saturation no longer forces sync finalize blocking in request path.
