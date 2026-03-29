# Project Profiling & Justification Report

## 1) Executive Summary

- Project: Wi-Fi CSI edge-based occupancy/activity classification.
- Stack: ESP32-C3 (edge capture + inference) + Raspberry Pi host (server, dashboard, training).
- Value proposition:
  - privacy-preserving (no camera/audio data), low-cost, low-power edge inference.
  - significant performance wins in preprocessing and ingestion with profiling-driven optimization.
  - hardened intermittent-connectivity behavior via MQTT and batched HTTP.

---

## 2) Measured Profiling Outcomes (Pi side)

Source: `inf2009_proj/TECHNICAL_REFERENCE.md` and `profiling/results/summary_table.md`.

Metrics table

| metric | baseline | optimized | delta_pct | speedup_x |
|---|---:|---:|---:|---:|
| upload_endpoint_avg_ms | 5.0992 | 3.2733 | -35.81 | 1.558 |
| extract_old_avg_ms | 104.7007 | 103.8013 | -0.86 | 1.009 |
| extract_new_avg_ms | 104.487 | 8.87 | -91.51 | 11.78 |
| extract_speedup_x | 1.002 | 11.7025 | +1067.9 | 11.679 |
| finalize_merge_peak_kib_avg | 75862.8 | 2180.1 | -97.13 | 34.798 |
| finalize_merge_rss_max_kib | 162976 | 90992 | -44.17 | 1.791 |

### Key insights

- `edge_ml.extract_features` reduced from ~104ms to ~8.9ms, a 91.5% improvement.
- Upload endpoint now under ~3.3ms.
- Peak memory in finalize payloads shrank from ~74MB to ~2MB (34x).
- RSS max shrank from 159MB to 89MB (1.8x), enabling stable Pi-side operation with lower swap pressure.

---

## 3) Edge device (ESP32-C3) resource expectations

### CPU
- ESP32-C3 160MHz CPU uses inference via TFLM.
- Per-sample inference: sub-ms (9 inputs, small dense model).
- Wi-Fi/MQTT background load for periodic transports.

### RAM
- Total available ~400KB SRAM.
- Firmware + network stacks likely use ~100-250KB.
- Model buffer and runtime context ~50-120KB.
- CSI upload chunk buffer minimal (kB) when batched properly.

### Flash
- Firmware image ~1.5–2MB.
- `model.tflite` typically 30–300KB after quantization.
- `scaler_params.json` ~ <10KB.

### GPU
- None for edge; CPU-only inference on ESP32.
- Raspberry Pi side does not use GPU in current setup.

### Storage (host)
- CSI data logs: 1–10MB per node/day (config dependent).
- Model store: ~0.5MB per node.
- Profiling artifacts: ~100MB+ in `profiling/results`.

---

## 4) Intermittent connectivity and resilience strategies

### ESP32 strategies
- MQTT command-based control topics:
  - `/commands/<NODE>/collect`
  - `/commands/<NODE>/training_complete`
  - `/commands/<NODE>/load_model`
- Auto model reload on MQTT reconnect.
- Batched HTTP upload (`/upload_data` splits) reduces partial-failure risk.
- `broadcast_generator.py` separate file, optional for traffic generation.

### Pi side strategies
- `/upload_data` endpoint chunked merge and memory-stable path.
- Idempotency and throttling checks in server logic.
- Workload-locking for deterministic behavior in profiling and regression tuning.

### Recommended improvements
- Persistent queue on ESP32 (e.g., NVS or flash segments) to survive reboot before upload.
- Exponential backoff + retry for HTTP and MQTT.
- ETag/hashes for model/params to avoid redundant downloads when reconnecting.
- In-dashboard health checks for stale node connectivity and round-trip latencies.

---

## 5) Privacy boundaries

- No raw CSI is offloaded as fine-grained location; only grouped features (8 group means + avg variation) are used.
- No camera/audio; sensor domain maintains strong privacy posture.
- Retention policy should be enforced for `csi_data/<NODE>/<STATE>` (not yet in code; operational process required).
- Hard-coded credentials are documented; production should switch to secrets store + TLS.

---

## 6) Justification summary (for stakeholders)

- deliverable: A complete sensor pipeline from collection to edge inference and retraining.
- high ROI: 91.5% time cut on hotspot, 97% memory cut on merge path.
- low-cost hardware: Raspberry Pi + ESP32‑C3; no server GPU.
- aligned with privacy and intermittent-connectivity constraints.

---

## 7) Next steps (roadmap)

1. Add explicit tool in `inf2009_proj` to auto-archive old `csi_data` after X days.
2. Extend `profiling/scripts` with link-failure simulation (`tc qdisc netem`) for intermittent validation.
3. Add host-level CPU/RAM logging in `server.py` while running ingest loops.
4. Add optional `MQTT+HTTP` auth (JWT/TLS) and secure config handling.
5. Add a small `scripts/validate_model_consistency.py` to check TFLM input shapes match `scaler_params.json`.
