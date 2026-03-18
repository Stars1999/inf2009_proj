from flask import Flask, request, send_file, jsonify
import numpy as np
import pandas as pd
import os
import glob
import json
import threading
import time
import zlib
import paho.mqtt.client as mqtt
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# --- CONFIGURATION (Must match ESP32 exactly) ---
MAX_LOWER = 4
MAX_UPPER = 60
DC_NULL = 32
# Note: BATCH_SIZE is only for validation of the FULL batch. 
# On server, we care about SUB_BATCH_SIZE (e.g., 40)
SUB_BATCH_SIZE = 40 
# make directories relative to this script so moving repo won't break paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "csi_data")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

os.makedirs(MODEL_STORE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Calculate exactly how many bytes per packet based on the exclusion
# create header names for every subcarrier index between lower and upper
# inclusive, excluding the DC null carrier
CSI_HEADERS = [f"SC_{i}" for i in range(MAX_LOWER, MAX_UPPER + 1) if i != DC_NULL]
# match firmware definition: ACTIVE_SUBCARRIERS = (MAX_UPPER - MAX_LOWER) with DC excluded
SUB_COUNT = len(CSI_HEADERS)
ALLOWED_LABELS = {"door_closed", "door_open", "person_standing"}
ALLOWED_SPLITS = {"train", "dev", "test"}
active_sessions = {}
# The idempotency cache is used to detect retries of the same logical upload.
# It grows with every new (session, sub-batch) combination until the cleanup
# thread evicts old entries. In a typical deployment this should remain small
# (e.g., <1000 keys) because each collection has only a handful of sub-batches,
# and entries expire quickly.
_seen_idempotency_keys: dict[str, tuple[float, int]] = {}  # key -> (timestamp, payload_crc)

# Protect access to shared structures used by Flask request threads + the cleanup thread
_active_sessions_lock = threading.Lock()

# How long to keep the idempotency keys in memory before evicting them
IDEMPOTENCY_TTL_SECONDS = 600  # 10 minutes

# How long to keep incomplete/in-progress session directories before cleanup.
# Session directories are the timestamped folders under <SAVE_DIR>/<node>/<label>/
# used while a multi-part upload is being collected.
SESSION_DIR_TTL_SECONDS = 600  # 10 minutes
SESSION_CLEANUP_INTERVAL_SECONDS = 300  # run cleanup every 5 minutes


def _json_load(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return default


def _json_save(path: str, payload) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _safe_slug(value: str, fallback: str = "unknown") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(value).strip())
    cleaned = cleaned.strip("-")
    return cleaned or fallback


def _session_manifest_path(session_dir: str) -> str:
    return os.path.join(session_dir, "session_manifest.json")


def _final_manifest_path(esp32_id: str, room_state: str, split_group: str, campaign_id: str, run_id: str, timestamp: str) -> str:
    return os.path.join(
        SAVE_DIR,
        esp32_id,
        room_state,
        f"csi_{room_state}_{split_group}_{timestamp}_manifest.json",
    )


def _update_session_manifest(
    session_dir: str,
    esp32_id: str,
    room_state: str,
    campaign_id: str,
    sess: str,
    split_group: str,
    sub_batch_idx: int,
    total_sub_batches: int,
    raw_size: int,
    computed_crc: int,
    idem_key: str | None,
    received_at: str,
    row_count: int,
) -> None:
    manifest_path = _session_manifest_path(session_dir)
    with _active_sessions_lock:
        manifest = _json_load(
            manifest_path,
            {
                "node_id": esp32_id,
                "label": room_state,
                "campaign_id": campaign_id,
                "session_id": sess,
                "split_group": split_group,
                "created_at": received_at,
                "updated_at": received_at,
                "expected_sub_batches": total_sub_batches,
                "parts": [],
            },
        )
        parts = [p for p in manifest.get("parts", []) if p.get("sub_batch_idx") != sub_batch_idx]
        parts.append(
            {
                "sub_batch_idx": sub_batch_idx,
                "total_sub_batches": total_sub_batches,
                "received_at": received_at,
                "split_group": split_group,
                "raw_bytes": raw_size,
                "payload_crc32": f"{computed_crc:08x}",
                "idempotency_key": idem_key or "",
                "row_count": row_count,
                "status": "accepted",
            }
        )
        parts.sort(key=lambda item: item.get("sub_batch_idx", 0))
        manifest.update(
            {
                "node_id": esp32_id,
                "label": room_state,
                "campaign_id": campaign_id,
                "session_id": sess,
                "split_group": split_group,
                "updated_at": received_at,
                "expected_sub_batches": total_sub_batches,
                "parts": parts,
                "last_sub_batch_idx": sub_batch_idx,
                "total_rows_observed": len(parts) * row_count,
            }
        )
        _json_save(manifest_path, manifest)


def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("/commands/+/collect")
        print("MQTT connected. Listening on /commands/+/collect")
    else:
        print(f"MQTT connection failed: {rc}")


def on_mqtt_message(client, userdata, message):
    topic_parts = message.topic.strip("/").split("/")
    if len(topic_parts) == 3 and topic_parts[0] == "commands" and topic_parts[2] == "collect":
        node_id = topic_parts[1]
        if not node_id or node_id == "None":
            print(f"WARNING: ignoring collect command with invalid node topic: {message.topic}")
            return
        raw = message.payload.decode(errors="replace").strip()
        # payload may be plain label or JSON {"label":"...","session":"..."}
        label = raw
        sess = None
        campaign_id = None
        run_id = None
        split_group = "train"
        try:
            j = json.loads(raw)
            if isinstance(j, dict):
                if "label" in j:
                    label = j.get("label", raw)
                if "session" in j:
                    sess = j.get("session")
                if "campaign_id" in j:
                    campaign_id = j.get("campaign_id")
                if "run_id" in j:
                    run_id = j.get("run_id")
                if "split_group" in j and isinstance(j.get("split_group"), str):
                    split_group = j.get("split_group")
        except json.JSONDecodeError:
            pass
        if sess is None or not isinstance(sess, str) or not sess:
            # generate simple session id based on timestamp
            sess = datetime.now().strftime("%Y%m%d%H%M%S")
        if run_id is None or not isinstance(run_id, str) or not run_id:
            run_id = sess
        if campaign_id is None or not isinstance(campaign_id, str) or not campaign_id:
            campaign_id = sess
        if split_group not in ALLOWED_SPLITS:
            split_group = "train"
        with _active_sessions_lock:
            active_sessions[node_id] = {
                "label": label,
                "session": run_id,
                "campaign_id": campaign_id,
                "split_group": split_group,
                "created_at": datetime.now().timestamp(),
            }
        print(f"Collection command received: node={node_id}, label={label}, campaign={campaign_id}, run={run_id}, split={split_group}")


mqtt_client = mqtt.Client()
# connection is deferred; call init_mqtt() once broker settings are known

def init_mqtt():
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    print(f"Initializing MQTT connection to {MQTT_BROKER}:{MQTT_PORT}")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"MQTT setup warning: {e}")


def publish_status_event(topic: str, payload: str, qos: int = 1) -> bool:
    """Publish MQTT status payload with explicit error checking/logging."""
    try:
        info = mqtt_client.publish(topic, payload, qos=qos)
    except Exception as e:
        print(f"WARNING: MQTT publish exception topic={topic}: {e}")
        return False

    rc = getattr(info, "rc", mqtt.MQTT_ERR_UNKNOWN)
    if rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"WARNING: MQTT publish failed topic={topic} rc={rc}")
        return False

    try:
        if hasattr(info, "wait_for_publish"):
            info.wait_for_publish()
    except Exception as e:
        print(f"WARNING: MQTT wait_for_publish error topic={topic}: {e}")

    return True


def _node_model_dir(node_id: str) -> str:
    return os.path.join(MODEL_STORE_DIR, node_id)


@app.route('/model/<node_id>', methods=['GET'])
def get_model(node_id):
    # serve model file; allow either explicit filename or bare id
    client_ip = request.remote_addr
    print(f"[MODEL DOWNLOAD] {client_ip} requesting model for {node_id}")
    dirpath = _node_model_dir(node_id)
    candidates = [
        os.path.join(dirpath, "model.tflite"),
        os.path.join(dirpath, node_id),
        os.path.join(dirpath, f"{node_id}.tflite"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
                print(f"[MODEL DOWNLOAD] Serving {path} ({size} bytes) to {client_ip}")
            except Exception:
                pass
            return send_file(path, mimetype="application/octet-stream", as_attachment=False)
    print(f"[MODEL DOWNLOAD] Model not found for {node_id} (searched {candidates})")
    return jsonify({"error": "model not found", "node": node_id}), 404


# alias that matches collaborator's '/static/models/<esp32_id>' pattern
@app.route('/static/models/<node_id>', methods=['GET'])
def download_model(node_id):
    # behaviour identical to /model/<node_id>
    return get_model(node_id)

@app.route('/params/<node_id>', methods=['GET'])
def get_scaler_params(node_id):
    client_ip = request.remote_addr
    print(f"[SCALER DOWNLOAD] {client_ip} requesting scaler params for {node_id}")
    path = os.path.join(_node_model_dir(node_id), "scaler_params.json")
    if not os.path.exists(path):
        print(f"[SCALER DOWNLOAD] scaler params not found for {node_id} (expected {path})")
        return jsonify({"error": "scaler params not found", "node": node_id}), 404
    try:
        size = os.path.getsize(path)
        print(f"[SCALER DOWNLOAD] Serving {path} ({size} bytes) to {client_ip}")
    except Exception:
        pass
    return send_file(path, mimetype="application/json", as_attachment=False)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    room_state = request.headers.get('X-Room-State', 'unknown')
    esp32_id = request.headers.get('X-ESP32-ID', '0')
    sub_batch_idx = int(request.headers.get('X-Sub-Batch-Index', -1))
    total_sub_batches = int(request.headers.get('X-Total-Sub-Batches', 0))
    session_hdr = request.headers.get('X-Session-ID')
    idem_key = request.headers.get('X-Idempotency-Key')

    if sub_batch_idx < 0 or total_sub_batches <= 0:
        return "Invalid Headers", 400

    if room_state != 'unknown' and room_state not in ALLOWED_LABELS:
        return jsonify({"error": "invalid label", "allowed": sorted(ALLOWED_LABELS)}), 400

    # prefer explicit session header, fall back to active_sessions table
    sess = session_hdr if session_hdr else None
    if room_state == 'unknown':
        with _active_sessions_lock:
            info = active_sessions.get(esp32_id)
        if isinstance(info, dict):
            room_state = info.get('label', room_state)
            if sess is None:
                sess = info.get('session')
        elif isinstance(info, str):
            # backward compatibility for legacy in-memory state values
            room_state = info

    if room_state not in ALLOWED_LABELS:
        return jsonify({"error": "invalid label", "allowed": sorted(ALLOWED_LABELS)}), 400

    if sess is None:
        sess = datetime.now().strftime('%Y%m%d%H%M%S')
        # update active_sessions so subsequent batches match
        with _active_sessions_lock:
            active_sessions[esp32_id] = {
                "label": room_state,
                "session": sess,
                "campaign_id": sess,
                "split_group": "train",
                "created_at": datetime.now().timestamp(),
            }

    with _active_sessions_lock:
        info = active_sessions.get(esp32_id, {})

    campaign_id = request.headers.get('X-Campaign-ID')
    run_id = request.headers.get('X-Run-ID') or sess
    split_group = request.headers.get('X-Split-Group')
    if not campaign_id and isinstance(info, dict):
        campaign_id = info.get('campaign_id')
    if not split_group and isinstance(info, dict):
        split_group = info.get('split_group')
    if split_group not in ALLOWED_SPLITS:
        split_group = 'train'
    if not campaign_id or not isinstance(campaign_id, str):
        campaign_id = sess
    campaign_slug = _safe_slug(campaign_id, fallback=sess)
    run_slug = _safe_slug(run_id, fallback=sess)

    raw_data = request.get_data()

    # CRC32 integrity check
    computed_crc = zlib.crc32(raw_data) & 0xFFFFFFFF
    received_at = datetime.now().isoformat(timespec='seconds')
    crc_header = request.headers.get('X-CRC32')
    if crc_header:
        try:
            received_crc = int(crc_header, 16)
        except ValueError:
            return "Invalid CRC header", 400
        if computed_crc != received_crc:
            return f"CRC mismatch: got {crc_header}, expected {computed_crc:08x}", 422

    # Idempotency: if we already processed this chunk, ensure payload matches
    if idem_key:
        with _active_sessions_lock:
            entry = _seen_idempotency_keys.get(idem_key)
        if entry is not None:
            _, stored_crc = entry
            if stored_crc == computed_crc:
                return jsonify({"status": "duplicate", "accepted": False}), 200
            return jsonify({"status": "conflict", "accepted": False, "reason": "payload mismatch"}), 409

    # Debug: Log CRC mismatch details when it occurs so we can track down firmware issues
    if crc_header:
        try:
            received_crc = int(crc_header, 16)
        except ValueError:
            return "Invalid CRC header", 400
        if computed_crc != received_crc:
            print(
                f"[CRC MISMATCH] node={esp32_id} label={room_state} session={sess} "
                f"part={sub_batch_idx}/{total_sub_batches} received_crc={crc_header} expected={computed_crc:08x} "
                f"len={len(raw_data)}"
            )
            return jsonify({
                "error": "crc_mismatch",
                "node": esp32_id,
                "label": room_state,
                "session": sess,
                "part": sub_batch_idx,
                "total_parts": total_sub_batches,
                "received_crc": crc_header,
                "expected_crc": f"{computed_crc:08x}",
                "length": len(raw_data),
            }), 422

    # Validation: The ESP32 sends a buffer of SUB_BATCH_SIZE
    expected_size = SUB_BATCH_SIZE * SUB_COUNT
    
    if len(raw_data) != expected_size:
        print(f"DATA MISMATCH: Received {len(raw_data)}, expected {expected_size}")
        return "Wrong Size", 400

    # Register this idempotency key now that the payload is valid
    if idem_key:
        with _active_sessions_lock:
            _seen_idempotency_keys[idem_key] = (time.time(), computed_crc)

    # Periodically cleanup old incomplete session directories.
    # This keeps the store from accumulating stale "in-progress" partial uploads.
    _cleanup_old_session_dirs()

    # Reshape binary data to DataFrame
    csi_matrix = np.frombuffer(raw_data, dtype=np.uint8).reshape(SUB_BATCH_SIZE, SUB_COUNT)
    df = pd.DataFrame(csi_matrix, columns=CSI_HEADERS)
    df.insert(0, "row_in_sub_batch", np.arange(len(df), dtype=np.int32))
    df.insert(0, "global_sample_idx", sub_batch_idx * SUB_BATCH_SIZE + df["row_in_sub_batch"].to_numpy(dtype=np.int32))
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
    
    # Path setup: Use a unique sub-directory for this session (handles multiple
    # collections from the same node/state running concurrently).
    with _active_sessions_lock:
        info = active_sessions.get(esp32_id)
    if isinstance(info, dict):
        sess = info.get("session", sess)
    if not sess:
        sess = datetime.now().strftime('%Y%m%d%H%M%S')

    session_dir = os.path.join(SAVE_DIR, esp32_id, room_state, campaign_slug, run_slug)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    # Save each sub-batch as its own individual file named by its index (e.g., part_000.csv)
    # This allows requests to arrive in any order (out-of-sequence)
    part_filename = os.path.join(session_dir, f"part_{sub_batch_idx:03d}.csv")
    df.to_csv(part_filename, index=False)
    _update_session_manifest(
        session_dir=session_dir,
        esp32_id=esp32_id,
        room_state=room_state,
        campaign_id=campaign_id,
        sess=sess,
        split_group=split_group,
        sub_batch_idx=sub_batch_idx,
        total_sub_batches=total_sub_batches,
        raw_size=len(raw_data),
        computed_crc=computed_crc,
        idem_key=idem_key,
        received_at=received_at,
        row_count=len(df),
    )
    
    # Check how many parts we have collected so far
    existing_parts = glob.glob(os.path.join(session_dir, "part_*.csv"))
    print(f"[{esp32_id}-{room_state}] Part {sub_batch_idx + 1}/{total_sub_batches} received (Current: {len(existing_parts)})")

    # Only merge when ALL parts have arrived
    if len(existing_parts) == total_sub_batches:
        print(f"FULL BATCH RECEIVED: Merging {total_sub_batches} parts...")
        
        # Sort files by name to ensure sequence (part_000, part_001, etc)
        existing_parts.sort()
        
        # Merge all parts into one large dataframe
        full_df_list = [pd.read_csv(f) for f in existing_parts]
        combined_df = pd.concat(full_df_list, ignore_index=True)

        # Keep a compact manifest alongside the merged CSV so the session
        # can be audited after the temporary sub-batch directory is removed.
        manifest = _json_load(_session_manifest_path(session_dir), {})
        
        # Save final combined CSV with timestamp
        timestamp = datetime.now().strftime('%m-%d_%H-%M-%S')
        final_filename = os.path.join(
            SAVE_DIR,
            esp32_id,
            room_state,
            f"csi_{room_state}_{split_group}_{timestamp}.csv",
        )
        combined_df.to_csv(final_filename, index=False)
        final_manifest_filename = _final_manifest_path(esp32_id, room_state, split_group, campaign_slug, run_slug, timestamp)
        manifest.update(
            {
                "merged_at": datetime.now().isoformat(timespec='seconds'),
                "final_csv": final_filename,
                "final_manifest": final_manifest_filename,
                "row_count": int(len(combined_df)),
                "feature_count": int(SUB_COUNT),
                "sub_batch_count": int(total_sub_batches),
                "campaign_id": campaign_id,
                "run_id": run_id,
                "split_group": split_group,
            }
        )
        _json_save(final_manifest_filename, manifest)
        
        # Cleanup: Remove temporary parts and their directory
        for f in existing_parts:
            os.remove(f)
        try:
            os.rmdir(session_dir)
        except OSError:
            pass # Directory might not be empty if another batch started simultaneously

        if isinstance(info, dict):
            completed_label = info.get("label", room_state)
        elif isinstance(info, str):
            completed_label = info
        payload = json.dumps({
            "event": "collection_complete",
            "label": completed_label,
            "session": sess,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "split_group": split_group,
            "source": "server",
        })
        if mqtt_client is not None:
            print(f"Publishing collection_complete for {esp32_id} (session={sess})")
            ok = publish_status_event(f"/sensors/{esp32_id}/status", payload, qos=1)
            if not ok:
                print(f"WARNING: publish collection_complete failed for {esp32_id} (session={sess})")
        else:
            print("WARNING: mqtt_client is None, cannot publish collection_complete")
        with _active_sessions_lock:
            active_sessions.pop(esp32_id, None)
        
        print(f"SAVED: {final_filename} with {len(combined_df)} rows.")
    
    return "OK", 200

_cleanup_thread_started = False
_cleanup_thread_lock = threading.Lock()


def start_server(host=os.getenv("SERVER_HOST", "0.0.0.0"), port=int(os.getenv("SERVER_PORT", "5000"))):
    # Ensure the background cleanup thread is running before we start serving.
    _start_cleanup_thread()

    # initialize MQTT with whatever broker/port have been configured
    init_mqtt()

    # Debug mode can be enabled via environment variable for development.
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug, use_reloader=debug)


def _cleanup_old_session_dirs() -> None:
    """Remove old incomplete session directories.

    In-progress uploads create per-session folders under <SAVE_DIR>/<node>/<label>.
    If the device never completes the upload, those folders can linger.
    This helper deletes folders containing a session_manifest.json that are older
    than SESSION_DIR_TTL_SECONDS.
    """
    now = time.time()
    cutoff = now - SESSION_DIR_TTL_SECONDS

    if not os.path.isdir(SAVE_DIR):
        return

    for node_id in os.listdir(SAVE_DIR):
        node_dir = os.path.join(SAVE_DIR, node_id)
        if not os.path.isdir(node_dir):
            continue
        for label in os.listdir(node_dir):
            label_dir = os.path.join(node_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for campaign in os.listdir(label_dir):
                campaign_dir = os.path.join(label_dir, campaign)
                if not os.path.isdir(campaign_dir):
                    continue
                for run in os.listdir(campaign_dir):
                    run_dir = os.path.join(campaign_dir, run)
                    if not os.path.isdir(run_dir):
                        continue
                    manifest_path = os.path.join(run_dir, "session_manifest.json")
                    if not os.path.exists(manifest_path):
                        continue
                    try:
                        mtime = os.path.getmtime(manifest_path)
                    except Exception:
                        mtime = os.path.getmtime(run_dir)
                    if mtime < cutoff:
                        try:
                            shutil.rmtree(run_dir)
                            print(f"[CLEANUP] Removed stale session directory: {run_dir}")
                        except Exception:
                            pass


def _cleanup_stale_entries(session_ttl_seconds: int = 1800, interval_seconds: int = 300):
    """Background thread: evict stale sessions and idempotency keys.

    By default, sessions are considered stale after 30 minutes (1800 seconds),
    and cleanup runs every 5 minutes (300 seconds). These values balance the
    need to avoid unbounded in-memory growth with tolerating intermittent
    reconnects and retries from devices.

    This function is designed to run forever in a daemon thread. Exceptions
    are caught and logged to prevent the thread from silently dying.
    """
    while True:
        try:
            time.sleep(interval_seconds)
            now = datetime.now().timestamp()

            # Evict old session entries
            cutoff = now - session_ttl_seconds
            with _active_sessions_lock:
                stale_sessions = [
                    k for k, v in active_sessions.items()
                    if isinstance(v, dict) and v.get("created_at", float("inf")) < cutoff
                ]
                for k in stale_sessions:
                    active_sessions.pop(k, None)
                    print(f"[TTL] Evicted stale session: {k}")

                # Evict old idempotency keys
                idem_cutoff = now - IDEMPOTENCY_TTL_SECONDS
                stale_idems = [k for k, v in _seen_idempotency_keys.items() if v[0] < idem_cutoff]
                for k in stale_idems:
                    _seen_idempotency_keys.pop(k, None)

            # Remove stale incomplete session directories as well.
            _cleanup_old_session_dirs()
        except Exception as e:
            print(f"[TTL] cleanup error: {e}")


def _start_cleanup_thread():
    global _cleanup_thread_started
    with _cleanup_thread_lock:
        if _cleanup_thread_started:
            return
        _cleanup_thread_started = True
    threading.Thread(target=_cleanup_stale_entries, daemon=True).start()


if __name__ == '__main__':
    start_server()
