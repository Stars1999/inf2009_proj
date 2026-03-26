#!/usr/bin/env python3
"""Non-interactive profiling harness for before/after hotspot measurements.

This script is designed for Pi-side profiling (`inf2009_proj`) and supports:
- time.perf_counter timing
- timeit repeated microbenchmarks
- cProfile call attribution snapshots
- tracemalloc peak memory during merge paths

It intentionally avoids GUI/live walkthrough requirements.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import inspect
import io
import json
import os
import platform
import resource
import shutil
import sys
import statistics
import subprocess
import tempfile
import time
import timeit
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from shared_config import DEFAULT_FEATURE_WINDOW


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _stats_from_runs(runs_ms: List[float]) -> Dict[str, float]:
    if not runs_ms:
        return {"avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    q = statistics.quantiles(runs_ms, n=20, method="inclusive")
    p95 = q[18] if len(q) >= 19 else max(runs_ms)
    return {
        "avg_ms": round(sum(runs_ms) / len(runs_ms), 4),
        "median_ms": round(statistics.median(runs_ms), 4),
        "p95_ms": round(p95, 4),
        "min_ms": round(min(runs_ms), 4),
        "max_ms": round(max(runs_ms), 4),
    }


def _import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_value(repo_root: Path, args: List[str], fallback: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()
        return out if out else fallback
    except Exception:
        return fallback


def _profile_top_lines(func, *args, top_n: int = 20, **kwargs) -> List[str]:
    import pstats

    prof = cProfile.Profile()
    prof.enable()
    func(*args, **kwargs)
    prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
    stats.print_stats(top_n)
    return [line.rstrip() for line in buf.getvalue().splitlines() if line.strip()]


def _scan_dataset_once(dataset_root: Path, node_id: str, states: List[str]) -> Dict[str, Any]:
    node_dir = dataset_root / node_id
    if not node_dir.is_dir():
        return {"manifest_count": 0, "csv_count": 0, "rows_scanned": 0}

    manifest_paths = [
        p
        for p in node_dir.rglob("*_manifest.json")
        if p.is_file() and p.name != "session_manifest.json"
    ]

    csv_count = 0
    rows_scanned = 0
    for state in states:
        state_dir = node_dir / state
        if not state_dir.is_dir():
            continue
        for path in state_dir.rglob("*.csv"):
            if not path.is_file() or path.name.startswith("part_"):
                continue
            csv_count += 1
            try:
                rows_scanned += int(len(pd.read_csv(path)))
            except Exception:
                pass

    return {
        "manifest_count": len(manifest_paths),
        "csv_count": csv_count,
        "rows_scanned": rows_scanned,
    }


def bench_summary_scan(dataset_root: Path, node_id: str, states: List[str], runs: int) -> Dict[str, Any]:
    run_ms: List[float] = []
    snapshot: Dict[str, Any] = {}

    for _ in range(runs):
        t0 = time.perf_counter()
        snapshot = _scan_dataset_once(dataset_root, node_id, states)
        run_ms.append((time.perf_counter() - t0) * 1000.0)

    # Cache emulation: first call miss, second call hit.
    cache: Dict[str, Dict[str, Any]] = {}

    def _cached_scan() -> Dict[str, Any]:
        if node_id in cache:
            return cache[node_id]
        result = _scan_dataset_once(dataset_root, node_id, states)
        cache[node_id] = result
        return result

    t0 = time.perf_counter()
    _cached_scan()
    miss_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _cached_scan()
    hit_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "runs_ms": [round(v, 4) for v in run_ms],
        "stats": _stats_from_runs(run_ms),
        "manifest_count": snapshot.get("manifest_count", 0),
        "csv_count": snapshot.get("csv_count", 0),
        "rows_scanned": snapshot.get("rows_scanned", 0),
        "cache_emulation": {
            "miss_ms": round(miss_ms, 4),
            "hit_ms": round(hit_ms, 4),
            "demo_speedup_x": round(miss_ms / hit_ms, 2) if hit_ms > 0 else None,
        },
    }


def bench_dataframe_build(iterations: int, sub_batch_size: int = 40, sub_count: int = 56) -> Dict[str, Any]:
    headers = [f"SC_{i}" for i in range(4, 61) if i != 32][:sub_count]
    raw = np.random.randint(0, 255, size=(sub_batch_size, len(headers)), dtype=np.uint8)

    def old_build():
        df = pd.DataFrame(raw, columns=headers)
        sub_batch_idx = 3
        received_at = "2026-03-26T00:00:00"
        idem_key = "idem_key"
        computed_crc = 0xABCDEF
        split_group = "train"
        run_id = "run_1"
        campaign_id = "campaign_1"
        sess = "session_1"
        room_state = "door_closed"
        esp32_id = "RACK_1"
        df.insert(0, "row_in_sub_batch", np.arange(len(df), dtype=np.int32))
        df.insert(0, "global_sample_idx", sub_batch_idx * sub_batch_size + df["row_in_sub_batch"].to_numpy(dtype=np.int32))
        df.insert(0, "upload_received_at", received_at)
        df.insert(0, "idempotency_key", idem_key or "")
        df.insert(0, "payload_crc32", f"{computed_crc:08x}")
        df.insert(0, "collection_split", split_group)
        df.insert(0, "calibration_run_id", run_id)
        df.insert(0, "campaign_id", campaign_id)
        df.insert(0, "sub_batch_idx", sub_batch_idx)
        df.insert(0, "session_id", sess)
        df.insert(0, "collection_label", room_state)
        df.insert(0, "node_id", esp32_id)
        return df

    def new_build():
        row_count = len(raw)
        sub_batch_idx = 3
        received_at = "2026-03-26T00:00:00"
        idem_key = "idem_key"
        computed_crc = 0xABCDEF
        split_group = "train"
        run_id = "run_1"
        campaign_id = "campaign_1"
        sess = "session_1"
        room_state = "door_closed"
        esp32_id = "RACK_1"
        metadata_columns = {
            "node_id": [esp32_id] * row_count,
            "collection_label": [room_state] * row_count,
            "session_id": [sess] * row_count,
            "sub_batch_idx": [sub_batch_idx] * row_count,
            "campaign_id": [campaign_id] * row_count,
            "calibration_run_id": [run_id] * row_count,
            "collection_split": [split_group] * row_count,
            "payload_crc32": [f"{computed_crc:08x}"] * row_count,
            "idempotency_key": [idem_key or ""] * row_count,
            "upload_received_at": [received_at] * row_count,
            "global_sample_idx": sub_batch_idx * sub_batch_size + np.arange(row_count, dtype=np.int32),
            "row_in_sub_batch": np.arange(row_count, dtype=np.int32),
        }
        csi_df = pd.DataFrame(raw, columns=headers)
        return pd.concat([pd.DataFrame(metadata_columns), csi_df], axis=1)

    old_sec = timeit.repeat(old_build, number=iterations, repeat=3)
    new_sec = timeit.repeat(new_build, number=iterations, repeat=3)
    old_ms = [v * 1000.0 for v in old_sec]
    new_ms = [v * 1000.0 for v in new_sec]
    old_avg = sum(old_ms) / len(old_ms)
    new_avg = sum(new_ms) / len(new_ms)

    return {
        "iterations": iterations,
        "old_insert_chain": {
            "runs_ms": [round(v, 4) for v in old_ms],
            "stats": _stats_from_runs(old_ms),
            "per_iter_us": round((old_avg * 1000.0) / iterations, 4),
        },
        "new_bulk_build": {
            "runs_ms": [round(v, 4) for v in new_ms],
            "stats": _stats_from_runs(new_ms),
            "per_iter_us": round((new_avg * 1000.0) / iterations, 4),
        },
        "speedup_x": round(old_avg / new_avg, 3) if new_avg > 0 else None,
    }


def bench_upload_endpoint(repo_root: Path, iterations: int, node_id: str) -> Dict[str, Any]:
    server = _import_module_from_path(f"server_mod_{int(time.time() * 1000)}", repo_root / "server.py")
    tmp_root = Path(tempfile.mkdtemp(prefix="upload_bench_"))

    try:
        # Keep endpoint microbench stable and non-blocking regardless of server defaults.
        if hasattr(server, "PASO_ASYNC_FINALIZE_ENABLED"):
            server.PASO_ASYNC_FINALIZE_ENABLED = False
        server.SAVE_DIR = str(tmp_root / "csi_data")
        Path(server.SAVE_DIR).mkdir(parents=True, exist_ok=True)
        server.active_sessions.clear()
        client = server.app.test_client()

        sub_batch_size = int(getattr(server, "SUB_BATCH_SIZE", 40))
        sub_count = int(getattr(server, "SUB_COUNT", 56))
        raw = np.random.randint(0, 255, size=(sub_batch_size, sub_count), dtype=np.uint8).tobytes()

        runs_ms: List[float] = []
        failures = 0
        for i in range(iterations):
            headers = {
                "X-Room-State": "door_closed",
                "X-ESP32-ID": node_id,
                "X-Sub-Batch-Index": "0",
                "X-Total-Sub-Batches": "2",
                "X-Session-ID": f"bench_sess_{i}",
                "X-Idempotency-Key": f"bench_key_{i}",
                "X-Campaign-ID": f"bench_campaign_{i}",
                "X-Run-ID": f"bench_run_{i}",
                "X-Split-Group": "train",
                "Content-Type": "application/octet-stream",
            }
            t0 = time.perf_counter()
            resp = client.post("/upload_data", data=raw, headers=headers)
            runs_ms.append((time.perf_counter() - t0) * 1000.0)
            if resp.status_code != 200:
                failures += 1

        top = _profile_top_lines(client.post, "/upload_data", data=raw, headers=headers, top_n=15)
        return {
            "iterations": iterations,
            "runs_ms": [round(v, 4) for v in runs_ms],
            "stats": _stats_from_runs(runs_ms),
            "failures": failures,
            "cprofile_top": top[:20],
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def bench_finalize_merge(
    repo_root: Path,
    runs: int,
    parts: int,
    rows_per_part: int,
    sub_count: int = 56,
) -> Dict[str, Any]:
    server = _import_module_from_path(f"server_mod_merge_{int(time.time() * 1000)}", repo_root / "server.py")
    server.mqtt_client = None
    tmp_root = Path(tempfile.mkdtemp(prefix="merge_bench_"))
    server.SAVE_DIR = str(tmp_root / "csi_data")
    Path(server.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    headers = [f"SC_{i}" for i in range(4, 61) if i != 32][:sub_count]
    run_ms: List[float] = []
    peaks_kib: List[float] = []
    cprofile_top: List[str] = []

    def _legacy_finalize(job: Dict[str, Any]) -> None:
        session_dir = str(job["session_dir"])
        esp32_id = str(job["esp32_id"])
        room_state = str(job["room_state"])
        split_group = str(job["split_group"])
        campaign_slug = str(job["campaign_slug"])
        run_slug = str(job["run_slug"])
        campaign_id = str(job["campaign_id"])
        run_id = str(job["run_id"])
        sess = str(job["sess"])
        total_sub_batches = int(job["total_sub_batches"])

        existing_parts = sorted(Path(session_dir).glob("part_*.csv"))
        if len(existing_parts) < total_sub_batches:
            return

        full_df_list = [pd.read_csv(part) for part in existing_parts]
        combined_df = pd.concat(full_df_list, ignore_index=True)

        manifest = {}
        if hasattr(server, "_json_load") and hasattr(server, "_session_manifest_path"):
            manifest = server._json_load(server._session_manifest_path(session_dir), {})

        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        final_filename = os.path.join(
            server.SAVE_DIR,
            esp32_id,
            room_state,
            f"csi_{room_state}_{split_group}_{timestamp}.csv",
        )
        os.makedirs(os.path.dirname(final_filename), exist_ok=True)
        combined_df.to_csv(final_filename, index=False)

        if hasattr(server, "_final_manifest_path"):
            final_manifest_filename = server._final_manifest_path(
                esp32_id,
                room_state,
                split_group,
                campaign_slug,
                run_slug,
                timestamp,
            )
        else:
            final_manifest_filename = os.path.join(
                server.SAVE_DIR,
                esp32_id,
                room_state,
                f"csi_{room_state}_{split_group}_{timestamp}_manifest.json",
            )

        manifest.update(
            {
                "merged_at": datetime.now().isoformat(timespec="seconds"),
                "final_csv": final_filename,
                "final_manifest": final_manifest_filename,
                "row_count": int(len(combined_df)),
                "feature_count": int(getattr(server, "SUB_COUNT", sub_count)),
                "sub_batch_count": int(total_sub_batches),
                "campaign_id": campaign_id,
                "run_id": run_id,
                "split_group": split_group,
                "session_id": sess,
            }
        )
        if hasattr(server, "_json_save"):
            server._json_save(final_manifest_filename, manifest)
        else:
            with open(final_manifest_filename, "w", encoding="utf-8") as fp:
                json.dump(manifest, fp, indent=2, sort_keys=True)

        for part in existing_parts:
            try:
                os.remove(part)
            except Exception:
                pass
        try:
            os.rmdir(session_dir)
        except OSError:
            pass

        if hasattr(server, "active_sessions"):
            try:
                server.active_sessions.pop(esp32_id, None)
            except Exception:
                pass

    if hasattr(server, "_finalize_session_artifacts"):
        finalize_fn: Callable[[Dict[str, Any]], None] = server._finalize_session_artifacts
        finalize_mode = "helper"
    else:
        finalize_fn = _legacy_finalize
        finalize_mode = "legacy_inline"

    try:
        for idx in range(runs):
            session_dir = (
                Path(server.SAVE_DIR) / "RACK_1" / "door_closed" / f"campaign_{idx}" / f"run_{idx}"
            )
            session_dir.mkdir(parents=True, exist_ok=True)

            for pidx in range(parts):
                arr = np.random.randint(0, 255, size=(rows_per_part, len(headers)), dtype=np.uint8)
                df = pd.DataFrame(arr, columns=headers)
                df.insert(0, "session_id", f"sess_{idx}")
                df.insert(0, "collection_label", "door_closed")
                df.insert(0, "node_id", "RACK_1")
                df.to_csv(session_dir / f"part_{pidx:03d}.csv", index=False)

            job = {
                "session_dir": str(session_dir),
                "esp32_id": "RACK_1",
                "room_state": "door_closed",
                "split_group": "train",
                "campaign_slug": f"campaign_{idx}",
                "run_slug": f"run_{idx}",
                "campaign_id": f"campaign_{idx}",
                "run_id": f"run_{idx}",
                "sess": f"sess_{idx}",
                "completed_label": "door_closed",
                "total_sub_batches": parts,
            }

            tracemalloc_started = False
            try:
                import tracemalloc

                tracemalloc.start()
                tracemalloc_started = True
            except Exception:
                tracemalloc_started = False

            t0 = time.perf_counter()
            if idx == 0:
                cprofile_top = _profile_top_lines(finalize_fn, job, top_n=20)
            else:
                finalize_fn(job)
            run_ms.append((time.perf_counter() - t0) * 1000.0)

            if tracemalloc_started:
                _, peak = tracemalloc.get_traced_memory()
                peaks_kib.append(peak / 1024.0)
                tracemalloc.stop()

        peak_stats = _stats_from_runs(peaks_kib) if peaks_kib else {"avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "runs": runs,
            "finalize_mode": finalize_mode,
            "parts": parts,
            "rows_per_part": rows_per_part,
            "total_rows_per_run": parts * rows_per_part,
            "runs_ms": [round(v, 4) for v in run_ms],
            "stats": _stats_from_runs(run_ms),
            "peak_kib_runs": [round(v, 4) for v in peaks_kib],
            "peak_kib_stats": {
                "avg_kib": round(peak_stats["avg_ms"], 4),
                "median_kib": round(peak_stats["median_ms"], 4),
                "p95_kib": round(peak_stats["p95_ms"], 4),
                "min_kib": round(peak_stats["min_ms"], 4),
                "max_kib": round(peak_stats["max_ms"], 4),
            },
            "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cprofile_top": cprofile_top[:20],
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _old_extract_loop(df: pd.DataFrame, feature_cols: List[str], window_size: int = 16) -> np.ndarray:
    raw_data = df[feature_cols].to_numpy(dtype=np.float32)
    calibrated_data = raw_data
    num_rows = calibrated_data.shape[0]
    all_features: List[np.ndarray] = []
    for i in range(num_rows):
        start = max(0, i - int(window_size) + 1)
        window = calibrated_data[start : i + 1, :]
        current_frame = calibrated_data[i, :]
        temp_std = np.std(window, axis=0)
        avg_variation = float(np.mean(temp_std))
        feat = np.concatenate([current_frame, np.asarray([avg_variation], dtype=np.float32)])
        all_features.append(feat)
    return np.asarray(all_features, dtype=np.float32)


def _call_extract(func, df: pd.DataFrame, feature_cols: List[str], window_size: int):
    sig = inspect.signature(func)
    kwargs = {}
    if "window_size" in sig.parameters:
        kwargs["window_size"] = window_size
    if "session_offset" in sig.parameters:
        kwargs["session_offset"] = 0.0
    if "calibration_val" in sig.parameters:
        kwargs["calibration_val"] = None
    if "group_count" in sig.parameters:
        kwargs["group_count"] = 0
    try:
        return func(df, feature_cols, **kwargs)
    except TypeError:
        return func(df, feature_cols)


def bench_extract_features(
    repo_root: Path,
    dataset_root: Path,
    node_id: str,
    runs: int,
    window_size: int,
) -> Dict[str, Any]:
    edge_ml = _import_module_from_path(f"edge_ml_mod_{int(time.time() * 1000)}", repo_root / "edge_ml.py")

    node_dir = dataset_root / node_id
    if not (
        (node_dir / "door_open").is_dir()
        and (node_dir / "door_closed").is_dir()
        and (node_dir / "person_standing").is_dir()
    ):
        raise RuntimeError(
            f"Dataset root missing expected node/state layout: {node_dir}"
        )
    data_dir = str(dataset_root)

    if hasattr(edge_ml, "_load_dataset"):
        df_open, df_close, df_person = edge_ml._load_dataset(data_dir, node_id)
    else:
        raise RuntimeError("edge_ml._load_dataset not found; cannot profile extract_features reliably.")

    feature_cols = list(getattr(edge_ml, "RAW_COLS", [c for c in df_open.columns if str(c).startswith("SC_")]))
    df_combined = pd.concat([df_open[feature_cols], df_close[feature_cols], df_person[feature_cols]], axis=0, ignore_index=True)

    current_extract = getattr(edge_ml, "extract_features")

    # Warmup
    _old_extract_loop(df_combined, feature_cols, window_size=window_size)
    _call_extract(current_extract, df_combined, feature_cols, window_size)

    old_runs: List[float] = []
    cur_runs: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _old_extract_loop(df_combined, feature_cols, window_size=window_size)
        old_runs.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _call_extract(current_extract, df_combined, feature_cols, window_size)
        cur_runs.append((time.perf_counter() - t0) * 1000.0)

    old_avg = sum(old_runs) / len(old_runs)
    cur_avg = sum(cur_runs) / len(cur_runs)

    cprofile_top = _profile_top_lines(_call_extract, current_extract, df_combined, feature_cols, window_size, top_n=20)

    return {
        "rows": int(len(df_combined)),
        "feature_count": int(len(feature_cols)),
        "window_size": int(window_size),
        "reference_old": {
            "runs_ms": [round(v, 4) for v in old_runs],
            "stats": _stats_from_runs(old_runs),
        },
        "current_impl": {
            "runs_ms": [round(v, 4) for v in cur_runs],
            "stats": _stats_from_runs(cur_runs),
        },
        "speedup_x": round(old_avg / cur_avg, 4) if cur_avg > 0 else None,
        "cprofile_top": cprofile_top[:20],
    }


def load_workload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pi-side hotspot profiling harness")
    parser.add_argument("--repo-root", default=".", help="Path to repository root to profile")
    parser.add_argument("--label", required=True, help="Label for this run (before/after)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--workload-lock", default="profiling/scripts/workload_lock.json", help="Workload lock JSON path")
    parser.add_argument("--upload-iterations", type=int, default=80, help="Upload endpoint benchmark iterations")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Dataset root containing <node>/<state>/... (default from workload lock path_hint)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    workload_lock = Path(args.workload_lock).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Ensure imports inside target modules (e.g., shared_config) resolve from
    # the profiled tree, not from the harness repository path.
    os.chdir(repo_root)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    cfg = load_workload(workload_lock)
    node_id = cfg.get("dataset", {}).get("node", "RACK_1")
    states = cfg.get("dataset", {}).get("states", ["door_closed", "door_open", "person_standing"])
    dataset_hint = str(cfg.get("dataset", {}).get("path_hint", "csi_data"))
    iters = cfg.get("microbench", {}).get("iterations", {})
    merge_cfg = cfg.get("microbench", {}).get("merge_synthetic", {})

    summary_runs = _safe_int(iters.get("summary_scan"), 5)
    df_iters = _safe_int(iters.get("dataframe_build"), 2000)
    extract_runs = _safe_int(iters.get("extract_features"), 3)
    merge_runs = _safe_int(iters.get("merge_runs"), 1)
    merge_parts = _safe_int(merge_cfg.get("parts"), 10)
    merge_rows = _safe_int(merge_cfg.get("rows_per_part"), 20000)
    feature_window = _safe_int(os.getenv("FEATURE_WINDOW", str(DEFAULT_FEATURE_WINDOW)), DEFAULT_FEATURE_WINDOW)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
    else:
        hint_path = Path(dataset_hint)
        dataset_root = hint_path.resolve() if hint_path.is_absolute() else (repo_root / hint_path).resolve()

    result: Dict[str, Any] = {
        "label": args.label,
        "timestamp_utc": _now_iso(),
        "repo_root": str(repo_root),
        "git": {
            "commit": _git_value(repo_root, ["rev-parse", "HEAD"]),
            "commit_short": _git_value(repo_root, ["rev-parse", "--short", "HEAD"]),
            "branch": _git_value(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "environment": {
            "python": os.sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "workload_lock": str(workload_lock),
        "dataset_root": str(dataset_root),
    }

    result["summary_scan"] = bench_summary_scan(dataset_root, node_id, states, summary_runs)
    result["upload_dataframe_microbench"] = bench_dataframe_build(df_iters)
    result["upload_endpoint"] = bench_upload_endpoint(repo_root, args.upload_iterations, node_id)
    result["finalize_merge"] = bench_finalize_merge(repo_root, merge_runs, merge_parts, merge_rows)
    result["extract_features"] = bench_extract_features(
        repo_root,
        dataset_root,
        node_id,
        extract_runs,
        feature_window,
    )

    with output.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    print(f"[OK] wrote profiling output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
