# Technical Reference (PASO + Profiling)

This document is the concise report-ready reference for PASO work in `inf2009_proj`.

## 1) Setup used for profiling

- Host: Raspberry Pi 5
- Project: `inf2009_proj` (Pi-side `server.py`, `dashboard.py`, `edge_ml.py`)
- Python: `/home/iankoh/inf2009_venv/bin/python3`
- Dataset: `csi_data/RACK_1` (same dataset for before and after)
- Baseline commit: `8ab0462`
- Current optimized commit profiled: `4f8913e`

## 2) PASO section (report-ready)

### Profile

Profiling was done before further optimization using:

- `time.perf_counter` for elapsed-time measurement
- `timeit` for repeatable microbenchmarks
- `cProfile` for call-level attribution
- `perf stat` for CPU-level counters
- `tracemalloc` and RSS for memory behavior

`valgrind` was not used because the current scope is Python runtime hotspots; for this workload it adds noise and low signal compared to the tools above.

### Analyze

The measured bottlenecks selected for optimization were:

- Upload endpoint latency path (`/upload_data`)
- Feature extraction path (`edge_ml.extract_features`)
- Merge/finalization memory footprint

### Schedule

A non-destructive before/after schedule was used:

1. Create temporary baseline worktree at commit `8ab0462`
2. Run identical profiling harness on baseline and current tree
3. Keep workload fixed with `profiling/scripts/workload_lock.json`
4. Save artifacts in `profiling/results/`
5. Auto-remove temporary worktree after capture

This ensured repeatability and avoided reverting the active branch.

### Optimize (measured outcome)

Key before/after results:

- Upload endpoint avg: `5.0992 ms -> 3.2733 ms` (**1.56x faster**, -35.81%)
- Extract current implementation avg: `104.487 ms -> 8.87 ms` (**11.78x faster**, -91.51%)
- Extract speedup ratio: `1.002x -> 11.7025x`
- Finalize merge peak allocation avg: `75862.8 KiB -> 2180.1 KiB` (**~34.8x lower**)
- Finalize merge max RSS: `162976 KiB -> 90992 KiB` (**~1.79x lower**)
- Finalize merge elapsed avg was slightly higher (`17595 ms -> 18260 ms`, +3.78%), indicating trade-off: much lower memory use for similar/slightly slower completion time.

## 3) How profiling was run

Command:

```bash
profiling/scripts/run_worktree_compare.sh 8ab0462
```

What this script does:

1. Creates detached baseline worktree
2. Runs baseline harness capture
3. Runs current-tree harness capture
4. Runs `perf stat` baseline and current captures
5. Builds summary table (`summary_table.csv`, `summary_table.md`)
6. Cleans up worktree
7. Writes run context to `profiling/results/runbook.md`

## 4) Artifacts produced

- `profiling/results/before_baseline.json`
- `profiling/results/after_optimized.json`
- `profiling/results/perf_before.csv`
- `profiling/results/perf_after.csv`
- `profiling/results/summary_table.csv`
- `profiling/results/summary_table.md`
- `profiling/results/runbook.md`

## 5) Notes

- ESP32 battery optimization is not prioritized for this deployment (constant power).
- `broadcast_generator.py` is intentionally retained.
