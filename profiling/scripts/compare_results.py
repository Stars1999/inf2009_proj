#!/usr/bin/env python3
"""Build report table from before/after profiling JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _metric_lower_is_better(before: float, after: float) -> Tuple[float, float]:
    if before == 0:
        return (0.0, 0.0)
    delta_pct = ((after - before) / before) * 100.0
    speedup = (before / after) if after > 0 else 0.0
    return (delta_pct, speedup)


def _metric_higher_is_better(before: float, after: float) -> Tuple[float, float]:
    if before == 0:
        return (0.0, 0.0)
    delta_pct = ((after - before) / before) * 100.0
    gain_x = (after / before) if before > 0 else 0.0
    return (delta_pct, gain_x)


def _safe_get(d: Dict[str, Any], path: List[str], default: float = 0.0) -> float:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    try:
        return float(cur)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare before/after profiling JSON outputs")
    parser.add_argument("--before", required=True, help="before_baseline.json")
    parser.add_argument("--after", required=True, help="after_optimized.json")
    parser.add_argument("--out-csv", required=True, help="summary_table.csv")
    parser.add_argument("--out-md", required=True, help="summary_table.md")
    args = parser.parse_args()

    before = _read_json(Path(args.before))
    after = _read_json(Path(args.after))

    rows: List[Dict[str, Any]] = []

    measures = [
        ("summary_scan_avg_ms", ["summary_scan", "stats", "avg_ms"], "lower"),
        ("upload_dataframe_old_avg_ms", ["upload_dataframe_microbench", "old_insert_chain", "stats", "avg_ms"], "lower"),
        ("upload_dataframe_new_avg_ms", ["upload_dataframe_microbench", "new_bulk_build", "stats", "avg_ms"], "lower"),
        ("upload_dataframe_speedup_x", ["upload_dataframe_microbench", "speedup_x"], "higher"),
        ("upload_endpoint_avg_ms", ["upload_endpoint", "stats", "avg_ms"], "lower"),
        ("finalize_merge_avg_ms", ["finalize_merge", "stats", "avg_ms"], "lower"),
        ("finalize_merge_peak_kib_avg", ["finalize_merge", "peak_kib_stats", "avg_kib"], "lower"),
        ("finalize_merge_rss_max_kib", ["finalize_merge", "rss_max_kib"], "lower"),
        ("extract_old_avg_ms", ["extract_features", "reference_old", "stats", "avg_ms"], "lower"),
        ("extract_new_avg_ms", ["extract_features", "current_impl", "stats", "avg_ms"], "lower"),
        ("extract_speedup_x", ["extract_features", "speedup_x"], "higher"),
    ]

    for metric_name, path, direction in measures:
        b = _safe_get(before, path, 0.0)
        a = _safe_get(after, path, 0.0)
        if direction == "higher":
            delta_pct, speedup = _metric_higher_is_better(b, a)
        else:
            delta_pct, speedup = _metric_lower_is_better(b, a)
        rows.append(
            {
                "metric": metric_name,
                "before": round(b, 4),
                "after": round(a, 4),
                "delta_pct": round(delta_pct, 2),
                "speedup_x": round(speedup, 3),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "before", "after", "delta_pct", "speedup_x"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out_md = Path(args.out_md)
    with out_md.open("w", encoding="utf-8") as fp:
        fp.write("| metric | before | after | delta_pct | speedup_x |\n")
        fp.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            fp.write(
                f"| {row['metric']} | {row['before']} | {row['after']} | {row['delta_pct']} | {row['speedup_x']} |\n"
            )

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
