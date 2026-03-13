from flask import Flask, request, send_file, jsonify
import numpy as np
import pandas as pd
import os
import glob
import json
import paho.mqtt.client as mqtt
from datetime import datetime

app = Flask(__name__)

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
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

os.makedirs(MODEL_STORE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Calculate exactly how many bytes per packet based on the exclusion
# create header names for every subcarrier index between lower and upper
# inclusive, excluding the DC null carrier
CSI_HEADERS = [f"SC_{i}" for i in range(MAX_LOWER, MAX_UPPER + 1) if i != DC_NULL]
# match firmware definition: ACTIVE_SUBCARRIERS = (MAX_UPPER - MAX_LOWER) with DC excluded
SUB_COUNT = len(CSI_HEADERS)
active_sessions = {}


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
        raw = message.payload.decode(errors="replace").strip()
        # payload may be plain label or JSON {"label":"...","session":"..."}
        label = raw
        sess = None
        try:
            j = json.loads(raw)
            if isinstance(j, dict):
                if "label" in j:
                    label = j.get("label", raw)
                if "session" in j:
                    sess = j.get("session")
        except json.JSONDecodeError:
            pass
        if sess is None or not isinstance(sess, str) or not sess:
            # generate simple session id based on timestamp
            sess = datetime.now().strftime("%Y%m%d%H%M%S")
        active_sessions[node_id] = {"label": label, "session": sess}
        print(f"Collection command received: node={node_id}, label={label}, session={sess}")


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
    dirpath = _node_model_dir(node_id)
    candidates = [os.path.join(dirpath, "model.tflite"), os.path.join(dirpath, node_id), os.path.join(dirpath, f"{node_id}.tflite")]
    for path in candidates:
        if os.path.exists(path):
            return send_file(path, mimetype="application/octet-stream", as_attachment=False)
    return jsonify({"error": "model not found", "node": node_id}), 404


# alias that matches collaborator's '/static/models/<esp32_id>' pattern
@app.route('/static/models/<node_id>', methods=['GET'])
def download_model(node_id):
    # behaviour identical to /model/<node_id>
    return get_model(node_id)

@app.route('/params/<node_id>', methods=['GET'])
def get_scaler_params(node_id):
    path = os.path.join(_node_model_dir(node_id), "scaler_params.json")
    if not os.path.exists(path):
        return jsonify({"error": "scaler params not found", "node": node_id}), 404
    return send_file(path, mimetype="application/json", as_attachment=False)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    room_state = request.headers.get('X-Room-State', 'unknown')
    esp32_id = request.headers.get('X-ESP32-ID', '0')
    sub_batch_idx = int(request.headers.get('X-Sub-Batch-Index', -1))
    total_sub_batches = int(request.headers.get('X-Total-Sub-Batches', 0))
    session_hdr = request.headers.get('X-Session-ID')

    if sub_batch_idx < 0 or total_sub_batches <= 0:
        return "Invalid Headers", 400

    # prefer explicit session header, fall back to active_sessions table
    sess = session_hdr if session_hdr else None
    if room_state == 'unknown' and esp32_id in active_sessions:
        info = active_sessions[esp32_id]
        if isinstance(info, dict):
            room_state = info.get('label', room_state)
            if sess is None:
                sess = info.get('session')
        elif isinstance(info, str):
            # backward compatibility for legacy in-memory state values
            room_state = info

    if sess is None:
        sess = datetime.now().strftime('%Y%m%d%H%M%S')
        # update active_sessions so subsequent batches match
        active_sessions[esp32_id] = {"label": room_state, "session": sess}

    raw_data = request.get_data()
    
    # Validation: The ESP32 sends a buffer of SUB_BATCH_SIZE
    expected_size = SUB_BATCH_SIZE * SUB_COUNT
    
    if len(raw_data) != expected_size:
        print(f"DATA MISMATCH: Received {len(raw_data)}, expected {expected_size}")
        return "Wrong Size", 400

    # Reshape binary data to DataFrame
    csi_matrix = np.frombuffer(raw_data, dtype=np.uint8).reshape(SUB_BATCH_SIZE, SUB_COUNT)
    df = pd.DataFrame(csi_matrix, columns=CSI_HEADERS)
    
    # Path setup: Use a unique sub-directory for this session (handles multiple
    # collections from the same node/state running concurrently).
    info = active_sessions.get(esp32_id)
    if isinstance(info, dict):
        sess = info.get("session", sess)
    if not sess:
        sess = datetime.now().strftime('%Y%m%d%H%M%S')

    session_dir = os.path.join(SAVE_DIR, esp32_id, room_state, sess)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    # Save each sub-batch as its own individual file named by its index (e.g., part_000.csv)
    # This allows requests to arrive in any order (out-of-sequence)
    part_filename = os.path.join(session_dir, f"part_{sub_batch_idx:03d}.csv")
    df.to_csv(part_filename, index=False)
    
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
        
        # Save final combined CSV with timestamp
        timestamp = datetime.now().strftime('%H%M%S')
        final_filename = os.path.join(SAVE_DIR, esp32_id, room_state, f"csi_{room_state}_{timestamp}.csv")
        combined_df.to_csv(final_filename, index=False)
        
        # Cleanup: Remove temporary parts and their directory
        for f in existing_parts:
            os.remove(f)
        try:
            os.rmdir(session_dir)
        except OSError:
            pass # Directory might not be empty if another batch started simultaneously

        # active_sessions stores a dict {"label":..., "session":...}
        # publish just the label string so dashboard doesn't receive a dict
        completed_label = room_state
        if esp32_id in active_sessions:
            info = active_sessions[esp32_id]
            if isinstance(info, dict):
                completed_label = info.get("label", room_state)
            elif isinstance(info, str):
                completed_label = info
        payload = json.dumps({
            "event": "collection_complete",
            "label": completed_label,
            "session": sess,
            "source": "server",
        })
        if mqtt_client is not None:
            print(f"Publishing collection_complete for {esp32_id} (session={sess})")
            ok = publish_status_event(f"/sensors/{esp32_id}/status", payload, qos=1)
            if not ok:
                print(f"WARNING: publish collection_complete failed for {esp32_id} (session={sess})")
        else:
            print("WARNING: mqtt_client is None, cannot publish collection_complete")
        active_sessions.pop(esp32_id, None)
        
        print(f"SAVED: {final_filename} with {len(combined_df)} rows.")
    
    return "OK", 200

def start_server(host='0.0.0.0', port=5000):
    # initialize MQTT with whatever broker/port have been configured
    init_mqtt()
    app.run(host=host, port=port, debug=True, use_reloader=False)

if __name__ == '__main__':
    start_server()
