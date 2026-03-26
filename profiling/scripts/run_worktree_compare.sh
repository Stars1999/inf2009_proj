#!/usr/bin/env bash
set -euo pipefail

# Non-destructive before/after profiling runner using git worktree baseline.
#
# Usage:
#   profiling/scripts/run_worktree_compare.sh <baseline_commit>
#
# Example:
#   profiling/scripts/run_worktree_compare.sh 8ab0462

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE_DIR="${ROOT_DIR}/../inf2009_proj_baseline_worktree"
RESULTS_DIR="${ROOT_DIR}/profiling/results"
RUNBOOK="${RESULTS_DIR}/runbook.md"
PY="/home/iankoh/inf2009_venv/bin/python3"
BASELINE_COMMIT="${1:-}"
DATASET_ROOT="${ROOT_DIR}/csi_data"

if [[ -z "${BASELINE_COMMIT}" ]]; then
  echo "Usage: $0 <baseline_commit>"
  exit 1
fi

cleanup_worktree() {
  if [[ -d "${WORKTREE_DIR}" ]]; then
    git -C "${ROOT_DIR}" worktree remove --force "${WORKTREE_DIR}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_worktree EXIT

mkdir -p "${RESULTS_DIR}"

echo "# Profiling runbook" > "${RUNBOOK}"
echo "" >> "${RUNBOOK}"
echo "- Timestamp (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "${RUNBOOK}"
echo "- Current repo: ${ROOT_DIR}" >> "${RUNBOOK}"
echo "- Baseline commit: ${BASELINE_COMMIT}" >> "${RUNBOOK}"
echo "- Dataset root: ${DATASET_ROOT}" >> "${RUNBOOK}"
echo "" >> "${RUNBOOK}"

echo "[1/6] Preparing baseline worktree..."
if [[ -d "${WORKTREE_DIR}" ]]; then
  git -C "${ROOT_DIR}" worktree remove --force "${WORKTREE_DIR}" || true
fi
git -C "${ROOT_DIR}" worktree add --detach "${WORKTREE_DIR}" "${BASELINE_COMMIT}"

echo "- Baseline worktree: ${WORKTREE_DIR}" >> "${RUNBOOK}"
echo "- Baseline HEAD: $(git -C "${WORKTREE_DIR}" rev-parse --short HEAD)" >> "${RUNBOOK}"
echo "- Current HEAD: $(git -C "${ROOT_DIR}" rev-parse --short HEAD)" >> "${RUNBOOK}"
echo "" >> "${RUNBOOK}"

echo "[2/6] Running baseline harness..."
"${PY}" "${ROOT_DIR}/profiling/scripts/profile_hotspots.py" \
  --repo-root "${WORKTREE_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --label "before_baseline" \
  --output "${RESULTS_DIR}/before_baseline.json" \
  --workload-lock "${ROOT_DIR}/profiling/scripts/workload_lock.json"

echo "[3/6] Running optimized harness..."
"${PY}" "${ROOT_DIR}/profiling/scripts/profile_hotspots.py" \
  --repo-root "${ROOT_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --label "after_optimized" \
  --output "${RESULTS_DIR}/after_optimized.json" \
  --workload-lock "${ROOT_DIR}/profiling/scripts/workload_lock.json"

echo "[4/6] Running perf stat (baseline)..."
perf stat -x, -o "${RESULTS_DIR}/perf_before.csv" \
  "${PY}" "${ROOT_DIR}/profiling/scripts/profile_hotspots.py" \
  --repo-root "${WORKTREE_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --label "before_perf" \
  --output "${RESULTS_DIR}/before_perf.json" \
  --workload-lock "${ROOT_DIR}/profiling/scripts/workload_lock.json" \
  --upload-iterations 40

echo "[5/6] Running perf stat (optimized)..."
perf stat -x, -o "${RESULTS_DIR}/perf_after.csv" \
  "${PY}" "${ROOT_DIR}/profiling/scripts/profile_hotspots.py" \
  --repo-root "${ROOT_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --label "after_perf" \
  --output "${RESULTS_DIR}/after_perf.json" \
  --workload-lock "${ROOT_DIR}/profiling/scripts/workload_lock.json" \
  --upload-iterations 40

echo "[6/6] Building comparison tables and cleaning up..."
"${PY}" "${ROOT_DIR}/profiling/scripts/compare_results.py" \
  --before "${RESULTS_DIR}/before_baseline.json" \
  --after "${RESULTS_DIR}/after_optimized.json" \
  --out-csv "${RESULTS_DIR}/summary_table.csv" \
  --out-md "${RESULTS_DIR}/summary_table.md"

git -C "${ROOT_DIR}" worktree remove --force "${WORKTREE_DIR}"

echo "" >> "${RUNBOOK}"
echo "## Outputs" >> "${RUNBOOK}"
echo "- before_baseline.json" >> "${RUNBOOK}"
echo "- after_optimized.json" >> "${RUNBOOK}"
echo "- perf_before.csv / perf_after.csv" >> "${RUNBOOK}"
echo "- before_perf.json / after_perf.json" >> "${RUNBOOK}"
echo "- summary_table.csv / summary_table.md" >> "${RUNBOOK}"

echo "Done. Results in ${RESULTS_DIR}"
