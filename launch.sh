#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/mqtt_logs"
PID_DIR="${ROOT_DIR}/.run"

MODE="start"
HEADLESS=0
COMPONENTS="server,dashboard,broadcast"

usage() {
  cat << 'EOF'
Usage:
  ./launch.sh [options]

Options:
  --headless                 Start in background (logs + pid files)
  --components LIST          Comma-separated: server,dashboard,broadcast
  --stop                     Stop selected components started in headless mode
  --status                   Show status of selected components
  -h, --help                 Show help

Examples:
  ./launch.sh
  ./launch.sh --headless
  ./launch.sh --headless --components server,broadcast
  ./launch.sh --status
  ./launch.sh --stop --components server,broadcast
EOF
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if [[ -x "/home/iankoh/inf2009_venv/bin/python3" ]]; then
    echo "/home/iankoh/inf2009_venv/bin/python3"
    return
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
    echo "${ROOT_DIR}/.venv/bin/python3"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo ""
}

is_running() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -d "/proc/${pid}" ]]
}

component_script() {
  case "$1" in
    server) echo "server.py" ;;
    dashboard) echo "dashboard.py" ;;
    broadcast) echo "broadcast_generator.py" ;;
    *) return 1 ;;
  esac
}

open_terminal() {
  local title="$1"
  local cmd="$2"

  if command -v lxterminal >/dev/null 2>&1; then
    lxterminal --title="$title" -e "bash -lc '${cmd}; exec bash'" &
    return
  fi

  if command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "${cmd}; exec bash" &
    return
  fi

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -lc "${cmd}; exec bash" &
    return
  fi

  echo "No terminal emulator found. Use --headless mode." >&2
  exit 1
}

start_component() {
  local name="$1"
  local script="$2"
  local pid_file="${PID_DIR}/${name}.pid"
  local run_cmd="cd \"${ROOT_DIR}\" && \"${PYTHON}\" \"${script}\""

  if [[ "$HEADLESS" -eq 1 ]]; then
    mkdir -p "$LOG_DIR" "$PID_DIR"

    if [[ -f "$pid_file" ]]; then
      local existing_pid
      existing_pid="$(cat "$pid_file" 2>/dev/null || true)"
      if is_running "$existing_pid"; then
        echo "[SKIP] ${name} already running (pid ${existing_pid})"
        return
      fi
      rm -f "$pid_file"
    fi

    nohup bash -lc "$run_cmd" >"${LOG_DIR}/${name}.log" 2>&1 &
    local pid=$!
    echo "$pid" >"$pid_file"
    echo "[OK] started ${name} (pid ${pid}) log=${LOG_DIR}/${name}.log"
    return
  fi

  open_terminal "INF2009 ${name}" "$run_cmd"
  echo "[OK] opened terminal for ${name}"
}

stop_component() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "[INFO] ${name} not tracked"
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  if is_running "$pid"; then
    "${PYTHON}" - "$pid" <<'PY'
import os
import signal
import sys

if len(sys.argv) < 2 or not sys.argv[1].isdigit():
    raise SystemExit(1)
os.kill(int(sys.argv[1]), signal.SIGTERM)
PY
    echo "[OK] stopped ${name} (pid ${pid})"
  else
    echo "[INFO] ${name} pid file exists but process is not running"
  fi
  rm -f "$pid_file"
}

status_component() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "${name}: not running (no pid file)"
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if is_running "$pid"; then
    echo "${name}: running (pid ${pid})"
  else
    echo "${name}: stale pid file"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless)
      HEADLESS=1
      shift
      ;;
    --components)
      COMPONENTS="${2:-}"
      shift 2
      ;;
    --stop)
      MODE="stop"
      shift
      ;;
    --status)
      MODE="status"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON="$(resolve_python)"
if [[ -z "$PYTHON" ]]; then
  echo "Python3 not found. Set PYTHON_BIN or install python3." >&2
  exit 1
fi

IFS=',' read -r -a requested <<< "$COMPONENTS"
components=()
for c in "${requested[@]}"; do
  c="$(echo "$c" | xargs)"
  case "$c" in
    server|dashboard|broadcast)
      components+=("$c")
      ;;
    *)
      echo "Invalid component: $c" >&2
      exit 1
      ;;
  esac
done

if [[ "${#components[@]}" -eq 0 ]]; then
  echo "No valid components selected." >&2
  exit 1
fi

case "$MODE" in
  start)
    for name in "${components[@]}"; do
      script="$(component_script "$name")"
      start_component "$name" "$script"
    done
    if [[ "$HEADLESS" -eq 1 ]]; then
      echo "Done. Use './launch.sh --status' or './launch.sh --stop'."
    fi
    ;;
  stop)
    for name in "${components[@]}"; do
      stop_component "$name"
    done
    ;;
  status)
    for name in "${components[@]}"; do
      status_component "$name"
    done
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac
