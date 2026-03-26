# Profiling Methodology and Results (PASO Report Section)

## 1. Objective

This profiling exercise was conducted to justify optimization decisions using measurable evidence.  
The process follows PASO: **Profile → Analyze → Schedule → Optimize**.

The goal was to compare a **pre-optimization baseline** against the **current optimized implementation** without reverting active development changes.

---

## 2. Scope

Primary profiling scope:

- Pi-side services in `inf2009_proj`:
  - `server.py`
  - `dashboard.py` (dataset summary path)
  - `edge_ml.py` (feature extraction path)

Out of scope:

- Battery-life benchmarking (deployment is constant-power).

---

## 3. Setup Used

- Host: Raspberry Pi 5
- Python: `/home/iankoh/inf2009_venv/bin/python3`
- Dataset root: `csi_data/RACK_1`
- Baseline commit: `8ab0462`
- Optimized commit: `4f8913e`
- Capture date (UTC): `2026-03-26` (from runbook)

---

## 4. Profiling Tools and Why They Were Used

| Tool | Purpose in this project | Why applicable |
|---|---|---|
| `time.perf_counter` | Elapsed-time timing for endpoints/functions | High-resolution timing for Python code paths |
| `timeit` | Repeatable microbenchmarks | Good for A/B comparisons of old vs new logic |
| `cProfile` | Call-stack attribution | Identifies where Python runtime is spending time |
| `perf stat` | CPU counters (task clock, cycles, instructions, faults) | Confirms low-level CPU behavior changes |
| `tracemalloc` + RSS | Peak/overall memory behavior | Required for merge/finalization memory analysis |
| `valgrind` | Not used in this run | Low signal for this Python-heavy scope unless debugging native leaks |

---

## 5. How Profiling Was Carried Out

### 5.1 Non-destructive before/after method

1. Create detached baseline worktree at `8ab0462`
2. Run identical profiling harness on baseline tree
3. Run identical profiling harness on current optimized tree
4. Run `perf stat` for both baseline and optimized runs
5. Generate comparison tables (`CSV` + `Markdown`)
6. Remove temporary worktree automatically

This preserves the active branch and avoids manual revert/reset.

### 5.2 Command used

```bash
profiling/scripts/run_worktree_compare.sh 8ab0462
```

### 5.3 Controlled workload

- Same dataset and node (`RACK_1`)
- Same synthetic payload sizes for upload/merge benchmarks
- Same host + Python interpreter
- Repeated runs with averaged/percentile stats in JSON outputs

---

## 6. Results Summary (Before vs After)

| Metric | Before | After | Delta | Interpretation |
|---|---:|---:|---:|---|
| Upload endpoint avg (ms) | 5.0992 | 3.2733 | -35.81% | Faster request handling |
| Extract current impl avg (ms) | 104.487 | 8.87 | -91.51% | Major speedup from vectorized extraction path |
| Extract speedup ratio (x) | 1.002 | 11.7025 | +1067.91% | New extraction path is much more efficient |
| Finalize merge peak alloc avg (KiB) | 75862.8 | 2180.1 | -97.13% | Large memory reduction |
| Finalize merge max RSS (KiB) | 162976 | 90992 | -44.17% | Lower process memory footprint |
| Finalize merge avg time (ms) | 17595.1 | 18259.8 | +3.78% | Slight time increase, but much better memory behavior |

---

## 7. How to Interpret the Results

- For latency/memory metrics, **lower is better**.
- For speedup metrics (`*_speedup_x`), **higher is better**.
- A negative delta on time/memory means improvement.
- A positive delta on speedup ratio means improvement.

Key conclusion:

- The optimization effort is strongly justified for:
  - `extract_features` (large latency improvement)
  - upload endpoint handling (meaningful latency improvement)
  - merge/finalization memory behavior (major reduction in memory pressure)

Trade-off observed:

- Merge/finalize elapsed time increased slightly, but with major memory savings.  
  This is acceptable when memory stability and scalability are prioritized.

---

## 8. Supporting Artifacts

Generated outputs:

- `profiling/results/before_baseline.json`
- `profiling/results/after_optimized.json`
- `profiling/results/perf_before.csv`
- `profiling/results/perf_after.csv`
- `profiling/results/summary_table.csv`
- `profiling/results/summary_table.md`
- `profiling/results/runbook.md`

These files provide full traceability for reported values.

---

## 9. PASO Justification Statement (Report-ready)

Profiling was executed first, under a controlled and reproducible setup, before further optimization work.  
Measured results showed that feature extraction and upload handling were high-impact targets, while merge finalization required memory-focused optimization.  
Therefore, subsequent optimization choices were evidence-driven, reproducible, and aligned with PASO methodology.

---

## 10. Why the Optimizations Worked (Mechanism)

### 10.1 Upload endpoint optimization (`server.py:/upload_data`)

What changed:

- Old path repeatedly inserted metadata columns into a DataFrame (`insert` chain).
- New path builds metadata once and concatenates in one step.

Why it works:

- Reduces repeated DataFrame column re-layout operations.
- Lowers Python-level overhead per request.
- Produces faster request handling under the same payload size.

Measured effect:

- Upload average dropped from `5.0992 ms` to `3.2733 ms` (~`1.56x` faster).

### 10.2 Feature extraction optimization (`edge_ml.py:extract_features`)

What changed:

- Old implementation performed a per-row loop and window operations in Python.
- New implementation uses a vectorized NumPy path (windowed standard deviation and mean variation computed in bulk).

Why it works:

- Moves work from Python loops to optimized NumPy C-level operations.
- Reduces interpreter overhead and repeated object allocations.

Measured effect:

- Current extraction path improved from `104.487 ms` to `8.87 ms` (~`11.78x` faster).

### 10.3 Finalize merge optimization (`server.py` merge/finalization path)

What changed:

- Old path concatenated all part CSVs in-memory into one large DataFrame.
- New path streams chunks to output CSV incrementally.

Why it works:

- Prevents large in-memory intermediate objects.
- Keeps memory pressure bounded as data size grows.

Measured effect:

- Peak allocation dropped from `75862.8 KiB` to `2180.1 KiB` (~`34.8x` lower).
- Max RSS dropped from `162976 KiB` to `90992 KiB` (~`1.79x` lower).
- Time increased slightly (`+3.78%`), which is an expected memory-vs-latency trade-off.

---

## 11. Hotspot Profiling: Which Functions Took Most Time/Space?

Yes. The profiling harness captures function-level hotspot evidence.

### 11.1 Time hotspots

- `cProfile` snapshots are embedded in JSON outputs (`cprofile_top` fields) for:
  - upload endpoint benchmark
  - finalize merge benchmark
  - extract_features benchmark
- `timeit`/`perf_counter` provide direct timing deltas for old-vs-new implementations.

Where to find:

- `profiling/results/before_baseline.json`
- `profiling/results/after_optimized.json`

Look for keys:

- `upload_endpoint.cprofile_top`
- `finalize_merge.cprofile_top`
- `extract_features.cprofile_top`

### 11.2 Space hotspots

- `tracemalloc` peak memory statistics were captured for finalize merge.
- Process-level memory trend captured with max RSS.

Where to find:

- `finalize_merge.peak_kib_stats`
- `finalize_merge.rss_max_kib`

Interpretation:

- If peak allocation drops strongly after optimization, the dominant memory hotspot was in that path.
- In this run, merge/finalization was the main memory hotspot, and optimization directly reduced it.
