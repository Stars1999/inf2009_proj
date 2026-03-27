import customtkinter as ctk
import paho.mqtt.client as mqtt
import os
import json
import glob
import time
import shutil
import subprocess
import sys
import queue
import pandas as pd
import traceback
import tkinter as tk
# matplotlib is only needed for adΓÇæhoc plotting; import lazily inside
# the plotting helper functions so that any backend initialization happens
# after the main Tk root is up.  This avoids sporadic segfaults on some
# Windows/Tk builds.
from datetime import datetime

import threading
from dotenv import load_dotenv
from shared_config import DEFAULT_MQTT_BROKER, DEFAULT_MQTT_PORT

# utility to enable mouse-wheel scrolling on CTkScrollableFrame widgets
# (CustomTkinter doesn't wire this up by default).  We bind when the
# cursor enters the frame and unbind on leave, mimicking native behavior.
def _enable_mousewheel(widget):
    """Enable mouse-wheel scrolling on the given scrollable frame.

    Previous implementation used bind_all() which registered the handler on
the entire application every time the pointer entered any scroll frame.  This
quickly produced cascading events and even crashed Tk with recursion when the
window was first shown.  We now bind directly to the widget, which is safe and
sufficient for most cases.
    """
    def _on_mousewheel(event):
        try:
            # normalized delta for Windows (multiples of 120) and Linux
            if event.delta:
                step = int(-1 * (event.delta / 120))
            elif event.num in (4, 5):
                step = 1 if event.num == 5 else -1
            else:
                step = 0
            if step:
                widget.yview_scroll(step, "units")
        except Exception:
            # swallow any Tk errors; we don't want stray callbacks crashing the app
            pass
    # bind wheel events directly; they fire when widget has focus or under cursor
    widget.bind("<MouseWheel>", _on_mousewheel)
    widget.bind("<Button-4>", _on_mousewheel)  # linux
    widget.bind("<Button-5>", _on_mousewheel)

# server module is no longer imported; launch the ingest service separately

load_dotenv()

# --- Global Config ---
BROKER = DEFAULT_MQTT_BROKER
PORT = DEFAULT_MQTT_PORT
LOG_DIR = "mqtt_logs"
CALIB_STATES = ["door_closed", "door_open", "person_standing"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# default to local server output path; can still be overridden in Settings
CSI_DATA_DIR = os.path.join(BASE_DIR, "csi_data")
TRAINING_SCRIPT = os.path.join(BASE_DIR, "edge_ml.py")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

# PASO optimization defaults (override with environment variables when needed)
PASO_ENABLE_PRUNING = os.getenv("PASO_ENABLE_PRUNING", "1").strip().lower() in ("1", "true", "yes")
PASO_PRUNING_SPARSITY = float(os.getenv("PASO_PRUNING_SPARSITY", "0.5"))
PASO_PRUNING_FINETUNE_EPOCHS = int(os.getenv("PASO_PRUNING_FINETUNE_EPOCHS", "30"))
PASO_MAX_ACCURACY_DROP = float(os.getenv("PASO_MAX_ACCURACY_DROP", "0.01"))
PASO_FEATURE_CONDENSE_GROUPS = int(os.getenv("PASO_FEATURE_CONDENSE_GROUPS", "8"))
TRAINING_RESULT_QUEUE_MAXSIZE = int(os.getenv("TRAINING_RESULT_QUEUE_MAXSIZE", "20"))

# legacy behaviour: fall back to local directory if CSI_DATA_DIR doesnΓÇÖt exist

CONFIG_FILE = "config.json"  # persists broker/port/csi directory

# filenames for various saved state
CALIB_FILE = "calib_states.json"           # per-node completed flags
KEYS_FILE = "keys.json"                    # issued/in vault status
VIEW_FILE = "view_state.json"              # last page/node
MODEL_FILE = "models.json"                 # per-node model validity flag
HEARTBEAT_FILE = "heartbeat_state.json"    # per-node last heartbeat timestamps

# Liveliness check configuration - tuned for faster offline detection
HEARTBEAT_TIMEOUT_S = 20             # Mark node stale if no heartbeat for 20s (matches ESP32 15s interval + margin)
LIVELINESS_CHECK_INTERVAL_MS = 5000  # Run liveliness check every 5s (fast detection)
STALE_ON_STARTUP_THRESHOLD_S = 300   # 5 minutes - mark offline if last heartbeat older


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.makedirs(MODEL_STORE_DIR, exist_ok=True)


class BoundedLogBuffer(list):
    """List-like buffer that keeps only the most recent log entries."""

    def __init__(self, max_items=5000, prune_to=2500):
        super().__init__()
        self.max_items = max_items
        self.prune_to = prune_to

    def append(self, item):
        super().append(item)
        if len(self) > self.max_items:
            del self[:-self.prune_to]


class DataSummaryCache:
    """Simple TTL cache for dataset summary information."""

    def __init__(self, ttl_seconds=30):
        self._cache = {}
        self._ttl = ttl_seconds

    def get(self, node_id):
        cached = self._cache.get(node_id)
        if cached:
            timestamp, summary = cached
            age = time.time() - timestamp
            if age < self._ttl:
                out = summary.copy()
                out["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                out["cache_age_seconds"] = round(age, 1)
                return out
        return None

    def set(self, node_id, summary):
        self._cache[node_id] = (time.time(), summary.copy())

    def invalidate(self, node_id=None):
        if node_id:
            self._cache.pop(node_id, None)
        else:
            self._cache.clear()

    def cleanup(self):
        now = time.time()
        expired = [
            node_id for node_id, (timestamp, _) in self._cache.items()
            if (now - timestamp) >= self._ttl
        ]
        for node_id in expired:
            del self._cache[node_id]
        return len(expired)


class GovernanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Zero-Trust Physical Key Governance")
        self.geometry("1100x750")
        
        # State Management
        # Keys 1-24: Key 2 is "out"
        self.keys = {f"Key {i}": "in" for i in range(1, 25)}
        self.all_logs = BoundedLogBuffer(max_items=5000, prune_to=2500)
        self.MAX_LOGS = 5000
        # By default every node has no completed calibrations
        self.esp_nodes = {
            f"RACK_{i}": {state: False for state in CALIB_STATES}
            for i in range(1, 5)
        }
        # keep track of when we last heard from each node (for health)
        self.node_last_seen = {}
        self.node_online = {node_id: False for node_id in self.esp_nodes}
        # track last heartbeat timestamp for liveliness detection
        self.node_last_heartbeat = {node_id: 0.0 for node_id in self.esp_nodes}
        # store the last radio selection per node so the UI remembers it
        self.last_calib_choice = {}
        # remember which label we requested for each node so we can validate
        self.expected_calib = {}
        # remember the active collection session per node for completion matching
        self.expected_session = {}
        # campaign/run bookkeeping so repeated calibrations stay grouped but
        # each calibration attempt still gets its own CSI file.
        self.collection_campaign = {}
        self.collection_run_counter = {}
        self.last_split_choice = {}
        # thresholds removed; model now handles inference
        # model lifecycle per-node (True if latest model is trained/valid)
        self.model_state = {node_id: False for node_id in self.esp_nodes}
        # prevent duplicate concurrent training triggers per node
        self.training_in_progress = {node_id: False for node_id in self.esp_nodes}
        # suppress repeated "auto-training held" log spam until split readiness changes
        self.auto_training_hold_logged = {node_id: False for node_id in self.esp_nodes}
        # debounce dashboard-wide refreshes under MQTT burst traffic
        self._ui_refresh_scheduled = False
        self._ui_refresh_delay_ms = 250
        self._pending_completion_events = []
        self._resize_refresh_job = None
        self._last_root_size = (0, 0)
        self._dataset_summary_cache = DataSummaryCache(ttl_seconds=30)
        # latest inferred state per node (published by ESP)
        self.node_detected_state = {node_id: "" for node_id in self.esp_nodes}
        # configuration state
        self.config = {"broker": BROKER, "port": PORT, "csi_data_dir": CSI_DATA_DIR}
        self.csi_data_dir = CSI_DATA_DIR
        # state for remembering page/node
        self.last_view = None
        self.last_node = None
        # load any previously saved data
        self._load_keys()
        self._load_calib_states()
        self._load_model_state()
        self._load_view_state()
        self._load_config()
        self._load_heartbeat_state()
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (LoginPage, MainDashboard):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")
        # refresh on resize so child views can redraw properly
        self.bind("<Configure>", self._on_resize)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        try:
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.client.connect_async(self.config.get("broker", BROKER), self.config.get("port", PORT), 60)
            self.client.loop_start()
        except Exception as e:
            print(f"WARNING: MQTT client setup failed: {e}")

        # Start periodic liveliness check for ESP32 nodes
        self._schedule_liveliness_check()

    def _log(self, category, message):
        """Add log entry with ring buffer to prevent unbounded growth."""
        ts = datetime.now().strftime('%H:%M:%S')
        self.all_logs.append((ts, category, message))

    def on_connect(self, client, userdata, flags, reason_code, properties):
        try:
            client.subscribe("#", qos=1)
            client.subscribe("device/+/status", qos=1)
            ts = datetime.now().strftime("%H:%M:%S")
            self._log("MQTT", f"Connected (reason_code={reason_code})")
        except Exception as e:
            print(f"WARNING: on_connect error: {e}")

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._log("MQTT", f"Disconnected (reason_code={reason_code})")
        # mark nodes offline until fresh status messages arrive
        for node_id in self.node_online:
            self.node_online[node_id] = False

    def _node_presence_state(self, node_id):
        """Return node presence state: online, stale, or offline.
        
        - online: Explicit online message received AND recent heartbeat
        - stale: Was online but heartbeat timed out (no explicit offline)
        - offline: Explicit offline message or never connected
        """
        if not self.node_online.get(node_id, False):
            return "offline"
        
        # Node is marked online, check heartbeat freshness
        last_hb = self.node_last_heartbeat.get(node_id, 0)
        if last_hb > 0:
            age = time.time() - last_hb
            if age > HEARTBEAT_TIMEOUT_S:
                return "stale"
        return "online"

    @staticmethod
    def _presence_colors(state):
        if state == "online":
            return "#2e7d32"  # green
        if state == "stale":
            return "#f9a825"  # amber/yellow
        return "gray"  # offline

    def _queue_ui_refresh(self, completion_event=None):
        """Schedule a debounced UI refresh on the Tk main loop.

        MQTT callbacks may arrive in bursts; this coalesces repeated refresh work
        while preserving collection-complete callbacks.
        """
        if completion_event:
            self._pending_completion_events.append(completion_event)

        if self._ui_refresh_scheduled:
            return

        self._ui_refresh_scheduled = True
        self.after(self._ui_refresh_delay_ms, self._flush_ui_refresh)

    def _flush_ui_refresh(self):
        self._ui_refresh_scheduled = False
        completion_events = self._pending_completion_events
        self._pending_completion_events = []

        main_dash = self.frames.get("MainDashboard")
        if not main_dash:
            return

        cfg = main_dash.sub_frames.get("ESP32-C3 Configuration")
        if completion_events and cfg:
            for completion_node, completion_state, completion_session, completion_source in completion_events:
                try:
                    cfg.on_calibration_complete(
                        completion_node,
                        completion_state,
                        completion_session,
                        completion_source,
                    )
                except Exception as e:
                    print("[DASHBOARD] on_calibration_complete exception")
                    traceback.print_exc()
                    self._log("Error", f"calibration callback failed: {e}")

        try:
            main_dash.refresh_active_view(force=False)
            main_dash.update_idletasks()
        except Exception:
            pass

    def _publish_collect_with_retry(self, node_id, label, split_group="train", retries=3):
        if not node_id or str(node_id) == "None" or node_id not in self.esp_nodes:
            self._log("Model", f"Refused to publish collect command for invalid node id: {node_id!r}")
            return False

        if split_group not in {"train", "dev"}:
            split_group = "train"

        campaign_id = self.collection_campaign.get(node_id)
        if not campaign_id:
            campaign_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.collection_campaign[node_id] = campaign_id

        self._dataset_summary_cache.invalidate(node_id)

        run_counter = self.collection_run_counter.get(node_id, 0) + 1
        self.collection_run_counter[node_id] = run_counter

        # each calibration attempt gets a distinct run id, while campaign_id
        # remains stable across the whole collection session
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_counter:02d}"
        topic = f"/commands/{node_id}/collect"
        payload = json.dumps({
            "label": label,
            "session": run_id,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "split_group": split_group,
        })
        for attempt in range(1, retries + 1):
            msg_info = self.client.publish(topic, payload, qos=1)
            if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
                # only set expected values once publish actually succeeds
                self.expected_calib[node_id] = label
                self.expected_session[node_id] = run_id
                return True
            time.sleep(0.2 * attempt)
        # clear stale expectations if publish failed after all retries
        self.expected_calib.pop(node_id, None)
        self.expected_session.pop(node_id, None)
        return False

    def _mark_model_dirty(self, node_id):
        if self.model_state.get(node_id, False):
            self.model_state[node_id] = False
            self._save_model_state()

    def _copy_model_to_server(self, node_id, model_path, scaler_path):
        """Copy the trained artifacts into the server model store.

        Returns True on success, False otherwise.  This does *not* notify the
        ESP32; notification is handled separately so we can wait for an explicit
        MQTT signal before the device ever attempts a download.
        """
        node_dir = os.path.join(MODEL_STORE_DIR, node_id)
        os.makedirs(node_dir, exist_ok=True)
        model_dst = os.path.join(node_dir, "model.tflite")
        scaler_dst = os.path.join(node_dir, "scaler_params.json")
        extra_files = [
            "split_metadata.json",
            "dataset_summary.json",
        ]
        extra_dirs = ["dataset_splits"]
        try:
            shutil.copy2(model_path, model_dst)
            shutil.copy2(scaler_path, scaler_dst)
            src_dir = os.path.dirname(os.path.abspath(model_path))
            for name in extra_files:
                src_path = os.path.join(src_dir, name)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(node_dir, name))
            for dirname in extra_dirs:
                src_path = os.path.join(src_dir, dirname)
                dst_path = os.path.join(node_dir, dirname)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            return True
        except Exception as e:
            self._log("Model", f"Artifact copy failed for {node_id}: {e}")
            return False

    def _notify_training_complete(self, node_id, session):
        """Publish a training-complete command so the ESP can begin downloading."""
        topic = f"/commands/{node_id}/training_complete"
        payload = json.dumps({"session": session})
        msg_info = self.client.publish(topic, payload, qos=1)
        if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
            self._log("Model", f"Notified {node_id} training_complete (session={session})")
            return True
        else:
            self._log("Model", f"Failed to publish training_complete for {node_id}")
            return False

    def _notify_model_reset(self, node_id):
        """Publish a model-reset command so ESP clears model from RAM immediately."""
        topic = f"/commands/{node_id}/reset_model"
        payload = json.dumps({"reason": "dashboard_rack_reset"})
        msg_info = self.client.publish(topic, payload, qos=1)
        if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
            self._log("Model", f"Requested runtime model reset for {node_id}")
            return True
        self._log("Model", f"Failed to publish runtime model reset for {node_id}")
        return False

    def _send_model_to_esp(self, node_id, model_path, scaler_path):
        """Legacy helper used by manual flows: copy and then send update_model.

        We keep this around for backwards compatibility (e.g. `trigger_collect`)
        but dashboard.training now uses training_complete instead.
        """
        if not self._copy_model_to_server(node_id, model_path, scaler_path):
            return False
        topic = f"/commands/{node_id}/update_model"
        session = datetime.now().strftime("%Y%m%d%H%M%S")
        payload = json.dumps({"session": session})
        msg_info = self.client.publish(topic, payload, qos=1)
        if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
            self._log("Model", f"Notified {node_id} to pull model (session={session})")
            return True
        self._log("Model", f"Failed to notify {node_id} for model update")
        return False

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        # remember last shown page and give it a chance to redraw
        self.last_view = page_name
        if hasattr(frame, "refresh"):
            try:
                frame.refresh()
            except Exception:
                pass

    # persistence helpers --------------------------------------------------
    def _save_calib_states(self):
        try:
            with open(CALIB_FILE, "w") as f:
                json.dump(self.esp_nodes, f)
        except Exception as e:
            print(f"WARNING: could not save calibration file: {e}")

    # keys persistence -------------------------------------------------
    def _save_keys(self):
        try:
            with open(KEYS_FILE, "w") as f:
                json.dump(self.keys, f)
        except Exception as e:
            print(f"WARNING: could not save keys: {e}")

    def _load_keys(self):
        if not os.path.exists(KEYS_FILE):
            return
        try:
            with open(KEYS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.keys.update(data)
        except Exception as e:
            print(f"WARNING: failed to load keys: {e}")

    # resize handler --------------------------------------------------

    def _on_resize(self, event):
        # Handle only root window size changes; child widget Configure events
        # are frequent and can trigger visual flashing if we rerender on each one.
        if event.widget is not self:
            return

        size = (event.width, event.height)
        if size == self._last_root_size:
            return
        self._last_root_size = size

        if self._resize_refresh_job is not None:
            try:
                self.after_cancel(self._resize_refresh_job)
            except Exception:
                pass

        self._resize_refresh_job = self.after(140, self._apply_resize_refresh)

    def _apply_resize_refresh(self):
        self._resize_refresh_job = None
        main_dash = self.frames.get("MainDashboard")
        if main_dash and main_dash.winfo_ismapped():
            try:
                main_dash.refresh_active_view(force=False)
            except Exception:
                pass

        try:
            self.update_idletasks()
        except Exception:
            pass

    # view state persistence ------------------------------------------
    def _save_view_state(self):
        state = {"last_page": self.last_view, "last_node": self.last_node}
        try:
            with open(VIEW_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"WARNING: could not save view state: {e}")

    def _load_view_state(self):
        if not os.path.exists(VIEW_FILE):
            return
        try:
            with open(VIEW_FILE, "r") as f:
                data = json.load(f)
            self.last_view = data.get("last_page")
            self.last_node = data.get("last_node")
        except Exception as e:
            print(f"WARNING: failed to load view state: {e}")

    # model persistence ------------------------------------------
    def _save_model_state(self):
        try:
            with open(MODEL_FILE, "w") as f:
                json.dump(self.model_state, f)
        except Exception as e:
            print(f"WARNING: could not save model state: {e}")

    # configuration persistence ---------------------------------------
    def _load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.config.update(data)
                # update attributes
                self.csi_data_dir = self.config.get("csi_data_dir", CSI_DATA_DIR)
        except Exception as e:
            print(f"WARNING: failed to load config: {e}")

    def _save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f)
        except Exception as e:
            print(f"WARNING: could not save config: {e}")


    def _load_model_state(self):
        if not os.path.exists(MODEL_FILE):
            return
        try:
            with open(MODEL_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for node_id, ready in data.items():
                    if node_id in self.model_state and isinstance(ready, bool):
                        self.model_state[node_id] = ready
        except Exception as e:
            print(f"WARNING: failed to load model state: {e}")


    def _load_calib_states(self):
        if not os.path.exists(CALIB_FILE):
            return
        try:
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
            # merge into default structure, ignore unknown nodes/states
            for node, states in data.items():
                if node in self.esp_nodes:
                    for state, val in states.items():
                        if state in self.esp_nodes[node] and isinstance(val, bool):
                            self.esp_nodes[node][state] = val
        except Exception as e:
            print(f"WARNING: failed to load calib states: {e}")

    # heartbeat state persistence for liveliness detection -----------------
    def _load_heartbeat_state(self):
        """Load last heartbeat timestamps; mark nodes offline if stale on startup."""
        if not os.path.exists(HEARTBEAT_FILE):
            return
        try:
            with open(HEARTBEAT_FILE, "r") as f:
                data = json.load(f)
            now = time.time()
            for node_id, ts in data.items():
                if node_id in self.node_last_heartbeat and isinstance(ts, (int, float)):
                    self.node_last_heartbeat[node_id] = ts
                    # If heartbeat is ancient, ensure node starts offline
                    if now - ts > STALE_ON_STARTUP_THRESHOLD_S:
                        self.node_online[node_id] = False
                    else:
                        # Recent heartbeat - tentatively mark online, will be confirmed by next heartbeat
                        self.node_online[node_id] = True
        except Exception as e:
            print(f"WARNING: failed to load heartbeat state: {e}")

    def _clear_node_heartbeat_state(self, node_id):
        """Clear persisted heartbeat timestamp for one node and mark it offline."""
        if node_id in self.node_last_heartbeat:
            self.node_last_heartbeat[node_id] = 0.0
        if node_id in self.node_online:
            self.node_online[node_id] = False
        if node_id in self.node_last_seen:
            self.node_last_seen.pop(node_id, None)
        self._save_heartbeat_state()

    def _save_heartbeat_state(self):
        """Persist heartbeat timestamps for recovery after restart."""
        try:
            with open(HEARTBEAT_FILE, "w") as f:
                json.dump(self.node_last_heartbeat, f)
        except Exception as e:
            print(f"WARNING: could not save heartbeat state: {e}")

    # liveliness check -----------------------------------------------------
    def _schedule_liveliness_check(self):
        """Schedule periodic liveliness check for ESP32 nodes."""
        self.after(LIVELINESS_CHECK_INTERVAL_MS, self._liveliness_check)

    def _liveliness_check(self):
        """Check heartbeat freshness and mark stale nodes; reschedule next check."""
        now = time.time()
        stale_detected = False
        
        for node_id in self.esp_nodes:
            if not self.node_online.get(node_id, False):
                continue  # already offline
            
            last_hb = self.node_last_heartbeat.get(node_id, 0)
            if last_hb > 0:
                age = now - last_hb
                if age > HEARTBEAT_TIMEOUT_S:
                    # Node heartbeat is stale - presence state will return "stale"
                    # Log once when transitioning to stale
                    prev_state = "online" if age <= HEARTBEAT_TIMEOUT_S + LIVELINESS_CHECK_INTERVAL_MS / 1000 else "stale"
                    if prev_state == "online":
                        print(f"\033[93m[LIVELINESS] Node {node_id} heartbeat stale ({age:.0f}s old)\033[0m")
                        self._log("LIVELINESS", f"Node {node_id} heartbeat stale ({age:.0f}s)")
                    stale_detected = True
        
        if stale_detected:
            self._queue_ui_refresh()
        
        # Reschedule next check
        self.after(LIVELINESS_CHECK_INTERVAL_MS, self._liveliness_check)

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="replace")
        now_epoch = time.time()
        self._log(msg.topic, payload)

        topic = msg.topic.strip()
        if topic.startswith("device/") and topic.endswith("/status"):
            topic_parts = topic.split("/")
            node_id = topic_parts[1] if len(topic_parts) >= 3 else "unknown"
            status_payload = payload.strip().lower()

            if status_payload == "online":
                self.node_last_seen[node_id] = now_epoch
                self.node_online[node_id] = True
                # Also update heartbeat timestamp when going online
                if node_id in self.node_last_heartbeat:
                    self.node_last_heartbeat[node_id] = now_epoch
                    self._save_heartbeat_state()
                print(f"\033[92m[SYSTEM] Node {node_id} is ONLINE\033[0m")
                self._log("SYSTEM", f"Node {node_id} is ONLINE")
                self._queue_ui_refresh()
            elif status_payload == "offline":
                self.node_last_seen[node_id] = now_epoch
                self.node_online[node_id] = False
                print(f"\033[91m[ALERT] Node {node_id} is OFFLINE\033[0m")
                self._log("ALERT", f"Node {node_id} is OFFLINE")
                self._queue_ui_refresh()

        completion_node = None
        completion_state = None
        completion_session = None
        completion_source = None

        topic_parts = msg.topic.strip("/").split("/")
        if len(topic_parts) == 3 and topic_parts[0] == "sensors" and topic_parts[2] == "status":
            node_id = topic_parts[1]
            try:
                payload_json = json.loads(payload)
                evt = payload_json.get("event")
                if evt == "node_online":
                    # Accept model status updates when the node first joins.
                    # Update heartbeat timestamp and mark online - proof of life
                    if node_id in self.node_last_heartbeat:
                        self.node_last_heartbeat[node_id] = now_epoch
                        self.node_last_seen[node_id] = now_epoch
                        self.node_online[node_id] = True
                        self._save_heartbeat_state()
                    mr = payload_json.get("model_ready")
                    if isinstance(mr, bool):
                        if self.model_state.get(node_id) != mr:
                            self.model_state[node_id] = mr
                            self._save_model_state()
                elif evt == "heartbeat":
                    # Heartbeats are proof of life - update timestamp and mark online
                    if node_id in self.node_last_heartbeat:
                        self.node_last_heartbeat[node_id] = now_epoch
                        self.node_last_seen[node_id] = now_epoch
                        self.node_online[node_id] = True
                        self._save_heartbeat_state()
                    # Heartbeats can also include model_ready to keep dashboard in sync.
                    mr = payload_json.get("model_ready")
                    if isinstance(mr, bool):
                        if self.model_state.get(node_id) != mr:
                            self.model_state[node_id] = mr
                            self._save_model_state()
                elif evt == "collection_progress":
                    label = payload_json.get("label", "?")
                    cur = payload_json.get("sub_batch_idx", "?")
                    total = payload_json.get("total_sub_batches", "?")
                    pct = payload_json.get("percent", "?")
                    self._log("Progress", f"{node_id} {label}: {cur}/{total} ({pct}%)")
                    # Update progress bar in the configuration view if visible
                    if self.frames.get("MainDashboard"):
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                        if cfg and cfg.current_node_id == node_id:
                            try:
                                ci = int(cur)
                                ti = int(total)
                                def _apply_collection_progress():
                                    try:
                                        if not getattr(cfg, "winfo_exists", lambda: False)():
                                            return
                                        cfg.set_progress(ci, ti)
                                        # if we've just reached the end, update status text
                                        if ci >= ti:
                                            cfg._safe_configure(
                                                cfg.status_lbl_ref,
                                                text="Status: Upload finished; awaiting server merge",
                                                text_color="#1a4d66",
                                            )
                                    except Exception:
                                        pass

                                self.after(0, _apply_collection_progress)
                            except Exception:
                                pass
                elif evt == "ack":
                    cmd = payload_json.get("cmd", "?")
                    self._log("ACK", f"{node_id} acknowledged {cmd}")
                elif evt == "identify_confirmed":
                    assigned = payload_json.get("name", node_id)
                    self._log("Identify", f"{node_id} confirmed name {assigned}")
                elif evt == "upload":
                    # upload progress from ESP32 (same handling as server-side "progress")
                    sub = payload_json.get("sub")
                    total = payload_json.get("total")
                    if isinstance(sub, (int, float)) and isinstance(total, (int, float)):
                        if self.frames.get("MainDashboard"):
                            subs = self.frames["MainDashboard"].sub_frames
                            cfg = subs.get("ESP32-C3 Configuration")
                            if cfg and cfg.current_node_id == node_id:
                                try:
                                    si = int(sub)
                                    ti = int(total)
                                    def _apply_upload_progress():
                                        try:
                                            if not getattr(cfg, "winfo_exists", lambda: False)():
                                                return
                                            cfg.set_progress(si, ti)
                                            # fallback unlock path: allow next calibration when
                                            # firmware confirms all uploads were sent, even if
                                            # server merge-complete MQTT is delayed/missing.
                                            if si >= ti:
                                                cfg._safe_configure(cfg.start_btn_ref, state="normal")
                                                cfg._safe_configure(
                                                    cfg.status_lbl_ref,
                                                    text="Status: Upload complete (server merge pending)",
                                                    text_color="#1a4d66",
                                                )
                                        except Exception:
                                            pass

                                    self.after(0, _apply_upload_progress)
                                except Exception:
                                    pass
                elif evt == "state_change":
                    state = payload_json.get("state")
                    if isinstance(state, str):
                        self.node_detected_state[node_id] = state
                        self._log("State", f"{node_id} -> {state}")
                elif evt == "collection_complete":
                    completion_node = node_id
                    completion_state = payload_json.get("label")
                    completion_session = payload_json.get("session")
                    completion_source = payload_json.get("source")
                    if completion_node not in self.esp_nodes:
                        completion_node = None
                        completion_state = None
                        completion_session = None
                        completion_source = None
                    elif not (isinstance(completion_state, str) and completion_state in self.esp_nodes[completion_node]):
                        completion_node = None
                        completion_state = None
                        completion_session = None
                        completion_source = None
                elif evt == "model_ready":
                    self.model_state[node_id] = True
                    self._save_model_state()
                    input_elements = payload_json.get("input_elements")
                    window_size = payload_json.get("window_size")
                    if isinstance(input_elements, (int, float)) and isinstance(window_size, (int, float)):
                        self._log(
                            "Model",
                            (
                                f"{node_id} reported model_ready (input_elements={int(input_elements)}, window_size={int(window_size)})",
                            )
                        )
                    else:
                        self._log("Model", f"{node_id} reported model_ready")

                    cfg = None
                    if "MainDashboard" in self.frames:
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                    if cfg and cfg.current_node_id == node_id:
                        def _show_model_ready():
                            try:
                                cfg._safe_configure(
                                    cfg.status_lbl_ref,
                                    text="Status: Model ready on device",
                                    text_color="#2e7d32",
                                )
                            except Exception:
                                pass

                        self.after(0, _show_model_ready)
                elif evt == "model_download_failed":
                    self.model_state[node_id] = False
                    self._save_model_state()
                    self._log("Model", f"{node_id} model download failed")

                    cfg = None
                    if "MainDashboard" in self.frames:
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                    if cfg and cfg.current_node_id == node_id:
                        def _show_model_failed():
                            try:
                                cfg._safe_configure(
                                    cfg.status_lbl_ref,
                                    text="Status: Model download failed (check ESP logs)",
                                    text_color="red",
                                )
                            except Exception:
                                pass

                        self.after(0, _show_model_failed)
                elif evt == "model_download_incompatible":
                    self.model_state[node_id] = False
                    self._save_model_state()

                    reason = payload_json.get("reason", "unknown")
                    input_elements = payload_json.get("input_elements", "?")
                    feature_count = payload_json.get("feature_count", "?")
                    self._log(
                        "Model",
                        (
                            f"{node_id} model incompatible: reason={reason}, input_elements={input_elements}, feature_count={feature_count}",
                        )
                    )

                    cfg = None
                    if "MainDashboard" in self.frames:
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                    if cfg and cfg.current_node_id == node_id:
                        def _show_model_incompatible():
                            try:
                                cfg._safe_configure(
                                    cfg.status_lbl_ref,
                                    text=f"Status: Model incompatible ({reason})",
                                    text_color="red",
                                )
                            except Exception:
                                pass

                        self.after(0, _show_model_incompatible)
                elif evt == "model_download_memory_error":
                    self.model_state[node_id] = False
                    self._save_model_state()

                    asset = payload_json.get("asset", "unknown")
                    reason = payload_json.get("reason", "unspecified")
                    attempt = payload_json.get("attempt", "?")
                    requested = payload_json.get("requested_bytes", "?")
                    free_heap = payload_json.get("free_heap", "?")
                    largest_block = payload_json.get("largest_block", "?")
                    err_name = payload_json.get("err_name", "UNKNOWN")

                    self._log(
                        "Model",
                        (
                            f"{node_id} model download memory error [{asset}/{reason}] "
                            f"attempt={attempt} req={requested} free={free_heap} largest={largest_block} err={err_name}",
                        )
                    )

                    cfg = None
                    if "MainDashboard" in self.frames:
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                    if cfg and cfg.current_node_id == node_id:
                        def _show_model_oom():
                            try:
                                cfg._safe_configure(
                                    cfg.status_lbl_ref,
                                    text=(
                                        f"Status: Model download OOM ({asset}, free={free_heap}, largest={largest_block})"
                                    ),
                                    text_color="red",
                                )
                            except Exception:
                                pass

                        self.after(0, _show_model_oom)
                elif evt == "model_cleared":
                    self.model_state[node_id] = False
                    self._save_model_state()
                    self._log("Model", f"{node_id} cleared runtime model")

                    cfg = None
                    if "MainDashboard" in self.frames:
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                    if cfg and cfg.current_node_id == node_id:
                        def _show_model_cleared():
                            try:
                                cfg._safe_configure(
                                    cfg.status_lbl_ref,
                                    text="Status: Model cleared on device",
                                    text_color="#d32f2f",
                                )
                            except Exception:
                                pass

                        self.after(0, _show_model_cleared)
            except json.JSONDecodeError:
                pass
        
        # Color Toggle Logic
        if payload in self.keys:
            if msg.topic == "Key Unlocked":
                self.keys[payload] = "out" # Red
            elif msg.topic == "Key Returned":
                self.keys[payload] = "in"  # Green
            # persist and log change
            self._save_keys()
            self._log("Key", f"{payload} {self.keys[payload]}")

        completion_event = None
        if completion_node and completion_state:
            completion_event = (completion_node, completion_state, completion_session, completion_source)
        self._queue_ui_refresh(completion_event)

# --- LOGIN PAGE ---
class LoginPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white")
        self.controller = controller

        # Centered Login UI
        login_box = ctk.CTkFrame(self, width=600, height=350, corner_radius=40, 
                                 border_width=2, border_color="#1a4d66", fg_color="white")
        login_box.place(relx=0.5, rely=0.5, anchor="center")
        login_box.pack_propagate(False)

        ctk.CTkLabel(login_box, text="Username:", font=("Arial", 22), text_color="black").place(x=50, y=70)
        self.user_ent = ctk.CTkEntry(login_box, width=320, height=50, fg_color="#1a5d7a", corner_radius=10)
        self.user_ent.place(x=230, y=60)

        ctk.CTkLabel(login_box, text="Password:", font=("Arial", 22), text_color="black").place(x=50, y=140)
        self.pass_ent = ctk.CTkEntry(login_box, width=320, height=50, fg_color="#1a5d7a", corner_radius=10, show="*")
        self.pass_ent.place(x=230, y=130)

        login_btn = ctk.CTkButton(login_box, text="Login", font=("Arial", 22), 
                                 fg_color="white", text_color="black", border_width=2, 
                                 border_color="#1a4d66", width=320, height=50,
                                 command=lambda: controller.show_frame("MainDashboard"))
        login_btn.place(x=230, y=220)

# --- MAIN DASHBOARD ---
class MainDashboard(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="#f0f0f0")
        self.controller = controller
        
        # Sidebar Navigation
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, border_width=1, border_color="black")
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="Control Centre", font=("Arial", 20, "bold")).pack(pady=20)
        
        self.nav_btns = {}
        items = ["Homepage", "Key Management", "ESP32-C3 Configuration", "Logs", "Graphs", "Settings", "Logout"]
        for item in items:
            btn = ctk.CTkButton(self.sidebar, text=item, fg_color="white", text_color="black", 
                               corner_radius=10, border_width=1, border_color="black", height=45,
                               command=lambda i=item: self.switch_view(i))
            btn.pack(fill="x", padx=15, pady=8)
            self.nav_btns[item] = btn

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.sub_frames = {}
        self.current_view = None
        for F in (HomeView, KeyMgmtView, ESPConfigView, LogsView, GraphsView, SettingsView):
            self.sub_frames[F.name] = F(self.content, self.controller)
            self.sub_frames[F.name].place(relx=0, rely=0, relwidth=1, relheight=1)
        # restore last-view if available (stored on controller)
        if self.controller.last_view and self.controller.last_view in self.sub_frames:
            # defer until after frames are laid out
            self.after(100, lambda: self.switch_view(self.controller.last_view))
            if self.controller.last_view == "ESP32-C3 Configuration" and self.controller.last_node:
                # show node when user returns
                def show_node():
                    vf = self.sub_frames.get("ESP32-C3 Configuration")
                    if vf:
                        vf.show_details(self.controller.last_node)
                self.after(150, show_node)
        else:
            self.switch_view("Homepage")

        self.after(2000, self._periodic_status_refresh)

    def _periodic_status_refresh(self):
        try:
            if self.current_view == "ESP32-C3 Configuration":
                cfg = self.sub_frames.get("ESP32-C3 Configuration")
                if cfg:
                    cfg.refresh(force_table=False)
            self.controller._dataset_summary_cache.cleanup()
        except Exception:
            pass
        self.after(2000, self._periodic_status_refresh)

    def refresh_active_view(self, force=False):
        name = self.current_view
        if not name or name not in self.sub_frames:
            return

        frame = self.sub_frames[name]
        if not hasattr(frame, "refresh"):
            return

        try:
            if name == "ESP32-C3 Configuration":
                frame.refresh(force_table=force)
            else:
                frame.refresh()
        except TypeError:
            frame.refresh()
        finally:
            try:
                frame.update_idletasks()
            except Exception:
                pass

    def switch_view(self, name):
        # logout gets special handling regardless of current view
        if name == "Logout":
            self.controller.show_frame("LoginPage")
            return

        # update button highlights and raise requested frame
        for n, b in self.nav_btns.items():
            b.configure(fg_color="#d9d9d9" if n == name else "white")
        if name in self.sub_frames:
            self.current_view = name
            self.sub_frames[name].tkraise()
            if hasattr(self.sub_frames[name], "refresh"):
                try:
                    if name == "ESP32-C3 Configuration":
                        self.sub_frames[name].refresh(force_table=True)
                    else:
                        self.sub_frames[name].refresh()
                except Exception:
                    pass
            try:
                self.sub_frames[name].update_idletasks()
            except Exception:
                pass

        # remember state for persistence
        self.controller.last_view = name
        if name == "ESP32-C3 Configuration":
            cur = self.sub_frames[name].current_node_id
            self.controller.last_node = cur
        else:
            self.controller.last_node = None
        self.controller._save_view_state()

# --- VIEWS ---

class HomeView(ctk.CTkFrame):
    name = "Homepage"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        # use pack with expand so the container resizes with the window
        self.grid_container = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_container.pack(fill="both", expand=True)
        self._last_stats_signature = None
        self.refresh()

    def refresh(self):
        # Calculate dynamic counts
        total_keys = len(self.controller.keys)
        issued_keys = sum(1 for status in self.controller.keys.values() if status == "out")
        vault_keys = sum(1 for status in self.controller.keys.values() if status == "in")
        online = sum(1 for node_id in self.controller.esp_nodes if self.controller._node_presence_state(node_id) == "online")
        total_nodes = len(self.controller.esp_nodes)
        offline = total_nodes - online
        # Define stats list with calculated values
        stats = [
            (str(total_keys), "Total Keys"), 
            (str(issued_keys), "Key Issued"), 
            (str(vault_keys), "Key In Vault"),
            (str(total_nodes), "Total ESP32-C3"), 
            (str(online), "ESP32-C3 Online"), 
            (str(offline), "ESP32-C3 Offline")
        ]

        stats_signature = tuple(stats)
        if stats_signature == self._last_stats_signature:
            return
        self._last_stats_signature = stats_signature

        # Clear old stats only when we know a redraw is needed.
        for widget in self.grid_container.winfo_children():
            widget.destroy()

        for i, (v, t) in enumerate(stats):
            box = ctk.CTkFrame(self.grid_container, width=180, height=130, 
                               border_width=1, border_color="black", corner_radius=15, fg_color="white")
            box.grid(row=i//3, column=i%3, padx=15, pady=15)
            box.pack_propagate(False)
            ctk.CTkLabel(box, text=v, font=("Arial", 28, "bold")).pack(pady=(30,0))
            ctk.CTkLabel(box, text=t, font=("Arial", 14)).pack()

class KeyMgmtView(ctk.CTkFrame):
    name = "Key Management"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="Key Status", font=("Arial", 40, "bold")).pack(pady=10)
        
        self.scroll = ctk.CTkScrollableFrame(self, corner_radius=30, border_width=1, border_color="#1a4d66")
        self.scroll.pack(fill="both", expand=True, padx=30, pady=10)
        _enable_mousewheel(self.scroll)
        self._last_keys_signature = None

    def refresh(self):
        keys_signature = tuple(sorted(self.controller.keys.items()))
        if keys_signature == self._last_keys_signature:
            return
        self._last_keys_signature = keys_signature

        for w in self.scroll.winfo_children(): w.destroy()
        for i, (name, status) in enumerate(self.controller.keys.items()):
            color = "#7ed957" if status == "in" else "#ff3131" # Green if in vault, red if issued
            ctk.CTkButton(self.scroll, text=name, fg_color=color, text_color="black", border_width=1, 
                          width=100, height=45, hover_color=color).grid(row=i//6, column=i%6, padx=10, pady=10)




class ESPConfigView(ctk.CTkFrame):
    name = "ESP32-C3 Configuration"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="ESP32-C3 Settings", font=("Arial", 40, "bold")).pack(pady=10)

        # build UI once during initialization; refresh will only touch the table
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20)
        
        # Left Panel (Node Selection List)
        left = ctk.CTkFrame(main, border_width=1, border_color="black", corner_radius=30, fg_color="white")
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        node_scroll = ctk.CTkScrollableFrame(left, fg_color="transparent")
        node_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        _enable_mousewheel(node_scroll)

        self.bulk_selected = {}
        self.node_status_labels = {}
        self.node_state_labels = {}
        self.node_buttons = {}
        self.bulk_label_var = ctk.StringVar(value="door_closed")
        self.bulk_split_var = ctk.StringVar(value="train")

        # 4 Nodes total for now
        for i in range(1, 5):
            node_id = f"RACK_{i}"
            row = ctk.CTkFrame(node_scroll, fg_color="transparent")
            row.pack(pady=6, padx=10, fill="x")

            var = ctk.BooleanVar(value=False)
            self.bulk_selected[node_id] = var
            ctk.CTkCheckBox(row, text="", variable=var, width=20).pack(side="left", padx=(0, 6))

            btn = ctk.CTkButton(row, text=f"ESP32 Node {i}", fg_color="white", text_color="black", border_width=1,
                                command=lambda n=node_id: self.show_details(n))
            btn.pack(side="left", fill="x", expand=True)
            self.node_buttons[node_id] = btn

            state_lbl = ctk.CTkLabel(row, text="Unknown", text_color="gray", width=140, anchor="e")
            state_lbl.pack(side="right", padx=(6, 0))
            self.node_state_labels[node_id] = state_lbl

            status_dot = ctk.CTkLabel(row, text="ΓùÅ", text_color="gray", width=20)
            status_dot.pack(side="right", padx=(6, 0))
            self.node_status_labels[node_id] = status_dot

        bulk_box = ctk.CTkFrame(left, fg_color="transparent")
        bulk_box.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(bulk_box, text="Bulk Label", font=("Arial", 13, "bold")).pack(anchor="w", pady=(6, 4))
        for state in CALIB_STATES:
            ctk.CTkRadioButton(bulk_box, text=state, variable=self.bulk_label_var, value=state).pack(anchor="w")
        ctk.CTkLabel(bulk_box, text="Bulk Split", font=("Arial", 13, "bold")).pack(anchor="w", pady=(8, 4))
        ctk.CTkOptionMenu(bulk_box, values=["train", "dev"], variable=self.bulk_split_var, width=120).pack(anchor="w")
        ctk.CTkButton(
            bulk_box,
            text="Start Selected Racks",
            fg_color="#3b8ed0",
            command=self.start_selected_racks
        ).pack(fill="x", pady=(8, 0))
        
        # Right Panel (Node Specific Details)
        # right panel becomes scrollable so contents never get clipped
        self.right = ctk.CTkScrollableFrame(main, border_width=1, border_color="black",
                                            corner_radius=30, fg_color="white")
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        _enable_mousewheel(self.right)
        # windowΓÇælevel resize refresh is handled by the app's _on_resize
        # (no need to bind individual Configure events here)
        
        self.detail_widgets = []
        self.calibration_table_widgets = []
        self.current_node_id = None
        self.radio_var = ctk.StringVar(value="door_closed")
        self.start_btn_ref = None
        self.status_lbl_ref = None
        self.train_btn_ref = None
        self.summary_btn_ref = None
        self.model_status_lbl_ref = None
        self.split_var = ctk.StringVar(value="train")
        self._training_result_queue = queue.Queue(maxsize=TRAINING_RESULT_QUEUE_MAXSIZE)
        self._training_popups = {}
        self._calibration_table_node_id = None
        self._calibration_title_ref = None
        self._calibration_campaign_ref = None
        self._calibration_row_refs = {}
        self._training_eligibility_lbl_ref = None
        # placeholder label lives inside scrollable frame
        self.placeholder = ctk.CTkLabel(self.right, text="Select a node to view details", text_color="gray")
        # use pack instead of place so it scrolls naturally
        self.placeholder.pack(pady=20)
        self._dataset_summary_cache = DataSummaryCache(ttl_seconds=30)
        self.after(150, self._process_training_results)

    def refresh(self, force_table=False):
        # when the frame is raised, make sure the UI reflects the current state
        # (node status, connection, door state, calibration table)
        self._refresh_node_status_list()
        if self.current_node_id:
            self._update_right_panel(self.current_node_id, refresh_table=force_table)

    def _ask_yes_no(self, title: str, message: str) -> bool:
        """Show a modal yes/no dialog and return True if user confirms."""
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.geometry("420x170")
        popup.transient(self)
        popup.lift()
        popup.focus_force()

        result = {"ok": False}

        def _yes():
            result["ok"] = True
            popup.destroy()

        def _no():
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _no)

        ctk.CTkLabel(popup, text=message, wraplength=400, justify="left").pack(pady=(16, 10), padx=20)

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(0, 16))
        ctk.CTkButton(btn_frame, text="Yes", width=90, command=_yes).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="No", width=90, command=_no).pack(side="left", padx=8)

        popup.grab_set()
        self.wait_window(popup)
        return result["ok"]

    def _ask_calibration_mode(self, node_id: str, state: str, split_group: str) -> str:
        """Ask user whether to append, replace, reset, or cancel calibration data."""
        popup = ctk.CTkToplevel(self)
        popup.title("Calibration conflict")
        popup.geometry("520x220")
        popup.transient(self)
        popup.lift()
        popup.focus_force()

        result = {"choice": "cancel"}

        def _append():
            result["choice"] = "append"
            popup.destroy()

        def _replace():
            result["choice"] = "replace"
            popup.destroy()

        def _reset():
            result["choice"] = "reset"
            popup.destroy()

        def _cancel():
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _cancel)

        ctk.CTkLabel(
            popup,
            text=(
                f"Node {node_id} already has calibration data for '{state}' (split: {split_group}).\n"
                "You can append another session, replace the existing data for this state,\n"
                "or reset all states and start fresh."
            ),
            wraplength=480,
            justify="left"
        ).pack(pady=(16, 10), padx=20)

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(0, 16))
        ctk.CTkButton(btn_frame, text="Append session", width=160, command=_append).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Replace state", width=140, command=_replace).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Reset rack", width=120, command=_reset).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=_cancel).pack(side="left", padx=8)

        popup.grab_set()
        self.wait_window(popup)
        return result["choice"]

    def _ask_reset_scope(self, node_id: str, selected_state: str) -> str:
        """Ask whether to reset only one state or the entire rack."""
        popup = ctk.CTkToplevel(self)
        popup.title("Reset calibration")
        popup.geometry("560x240")
        popup.transient(self)
        popup.lift()
        popup.focus_force()

        result = {"choice": "cancel"}

        def _state_only():
            result["choice"] = "state"
            popup.destroy()

        def _rack():
            result["choice"] = "rack"
            popup.destroy()

        def _cancel():
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _cancel)

        ctk.CTkLabel(
            popup,
            text=(
                f"Choose reset scope for {node_id}.\n\n"
                f"State reset: only clear '{selected_state}' calibration data.\n"
                "Rack reset: clear all states for this rack, remove stored model files,\n"
                "and request ESP32 runtime model reset."
            ),
            wraplength=520,
            justify="left"
        ).pack(pady=(16, 12), padx=20)

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(0, 16))
        ctk.CTkButton(btn_frame, text=f"Reset state ({selected_state})", width=180, command=_state_only).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Reset entire rack", width=160, fg_color="#ff5555", command=_rack).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=_cancel).pack(side="left", padx=8)

        popup.grab_set()
        self.wait_window(popup)
        return result["choice"]

    def _refresh_node_status_list(self):
        any_key_out = any(v == "out" for v in self.controller.keys.values())
        for node_id, dot in self.node_status_labels.items():
            presence = self.controller._node_presence_state(node_id)
            state = self.controller.node_detected_state.get(node_id)
            state_lbl = self.node_state_labels.get(node_id)
            dot.configure(text_color=self.controller._presence_colors(presence))
            if state == "door_closed":
                if state_lbl:
                    state_lbl.configure(text="door_closed", text_color="#2e7d32")
            elif state == "door_open":
                if any_key_out:
                    if state_lbl:
                        state_lbl.configure(text="door_open (authorized)", text_color="#ff9800")
                else:
                    if state_lbl:
                        state_lbl.configure(text="door_open (unauthorized)", text_color="#d32f2f")
            elif state == "person_standing":
                if state_lbl:
                    state_lbl.configure(text="person_standing", text_color="#ff9800")
            else:
                if state_lbl:
                    state_lbl.configure(text=presence, text_color=self.controller._presence_colors(presence))

    def show_details(self, node_id):
        self.current_node_id = node_id
        # remember selection for next launch
        self.controller.last_node = node_id
        self.controller._save_view_state()
        self.placeholder.pack_forget()
        for w in list(self.detail_widgets):
            try:
                w.destroy()
            except tk.TclError:
                # Some CTk widgets (e.g., OptionMenu) can raise on destroy if their
                # underlying Tk trace callback has already been removed.
                # This is non-fatal; ignore and continue cleaning up remaining widgets.
                pass
        self.detail_widgets = []
        self.calibration_table_widgets = []
        self.start_btn_ref = None
        self.status_lbl_ref = None
        self.train_btn_ref = None
        self.summary_btn_ref = None
        self.model_status_lbl_ref = None
        self._training_eligibility_lbl_ref = None
        self._calibration_title_ref = None
        self._calibration_campaign_ref = None
        self._calibration_row_refs = {}
        self._calibration_table_node_id = node_id
        self.split_var = ctk.StringVar(value=self.controller.last_split_choice.get(node_id, "train"))
        # restore last radio choice for this node if available
        default_choice = self.controller.last_calib_choice.get(node_id, "door_closed")
        self.radio_var = ctk.StringVar(value=default_choice)

        # Dynamic values
        node_state = self.controller.node_detected_state.get(node_id, "")
        if node_state == "door_closed":
            door_status = "CLOSE"
        elif node_state == "door_open":
            door_status = "OPEN"
        elif node_state == "person_standing":
            door_status = "PERSON PRESENT"
        else:
            door_status = "UNKNOWN"

        # Store references to these labels so we can update them dynamically
        self.node_number_label = None
        self.connection_label = None
        self.last_seen_label = None
        self.heartbeat_label = None
        self.door_status_label = None

        f = ctk.CTkFrame(self.right, fg_color="transparent")
        f.pack(fill="x", padx=30, pady=8)
        self.detail_widgets.append(f)
        ctk.CTkLabel(f, text="Node Number:", text_color="black", font=("Arial", 16)).pack(side="left")
        self.node_number_label = ctk.CTkLabel(f, text=node_id, text_color="gray")
        self.node_number_label.pack(side="right")

        f = ctk.CTkFrame(self.right, fg_color="transparent")
        f.pack(fill="x", padx=30, pady=8)
        self.detail_widgets.append(f)
        ctk.CTkLabel(f, text="Connection:", text_color="black", font=("Arial", 16)).pack(side="left")
        presence = self.controller._node_presence_state(node_id)
        self.connection_label = ctk.CTkLabel(f, text=presence.upper(), text_color=self.controller._presence_colors(presence))
        self.connection_label.pack(side="right")

        f = ctk.CTkFrame(self.right, fg_color="transparent")
        f.pack(fill="x", padx=30, pady=0)
        self.detail_widgets.append(f)
        self.last_seen_label = ctk.CTkLabel(f, text="", text_color="gray")
        self.last_seen_label.pack(side="right")

        # Heartbeat indicator row
        f = ctk.CTkFrame(self.right, fg_color="transparent")
        f.pack(fill="x", padx=30, pady=4)
        self.detail_widgets.append(f)
        ctk.CTkLabel(f, text="Last Heartbeat:", text_color="black", font=("Arial", 14)).pack(side="left")
        self.heartbeat_label = ctk.CTkLabel(f, text="", text_color="gray")
        self.heartbeat_label.pack(side="right")

        f = ctk.CTkFrame(self.right, fg_color="transparent")
        f.pack(fill="x", padx=30, pady=8)
        self.detail_widgets.append(f)
        ctk.CTkLabel(f, text="Door Status:", text_color="black", font=("Arial", 16)).pack(side="left")
        self.door_status_label = ctk.CTkLabel(f, text=door_status, text_color="gray")
        self.door_status_label.pack(side="right")

        divider = ctk.CTkFrame(self.right, height=1, fg_color="#cccccc")
        divider.pack(fill="x", padx=30, pady=(8, 14))
        self.detail_widgets.append(divider)

        radio_group = ctk.CTkFrame(self.right, fg_color="transparent")
        radio_group.pack(fill="x", padx=30, pady=4)
        self.detail_widgets.append(radio_group)
        ctk.CTkLabel(radio_group, text="Calibration Label:", text_color="black", font=("Arial", 16)).pack(anchor="w")

        for state in CALIB_STATES:
            rb = ctk.CTkRadioButton(radio_group, text=state, variable=self.radio_var, value=state)
            rb.pack(anchor="w", pady=2)

        ctk.CTkLabel(radio_group, text="Collection Split:", text_color="black", font=("Arial", 16)).pack(anchor="w", pady=(8, 0))
        split_menu = ctk.CTkOptionMenu(
            radio_group,
            values=["train", "dev", "test"],
            variable=self.split_var,
            command=lambda v, n=node_id: self.controller.last_split_choice.__setitem__(n, v),
            width=140,
        )
        split_menu.pack(anchor="w", pady=(2, 4))
        self.detail_widgets.append(split_menu)

        # By default we append new collection sessions (no interruption), but
        # allow the user to explicitly request an overwrite/replace prompt.
        self.ask_overwrite_var = tk.BooleanVar(value=False)
        self.ask_overwrite_checkbox = ctk.CTkCheckBox(
            radio_group,
            text="Prompt before overwriting existing split data",
            variable=self.ask_overwrite_var,
        )
        self.ask_overwrite_checkbox.pack(anchor="w", pady=(0, 8))
        self.detail_widgets.append(self.ask_overwrite_checkbox)

        self.start_btn_ref = ctk.CTkButton(
            self.right,
            text="Start Calibration",
            fg_color="#3b8ed0",
            command=lambda n=node_id: self.start_calibration(n)
        )
        self.start_btn_ref.pack(pady=(10, 6))
        self.detail_widgets.append(self.start_btn_ref)

        self.status_lbl_ref = ctk.CTkLabel(self.right, text="Status: Idle", text_color="gray")
        self.status_lbl_ref.pack(pady=(0, 10))
        self.detail_widgets.append(self.status_lbl_ref)

        summary_btn = ctk.CTkButton(
            self.right,
            text="Dataset Summary",
            fg_color="#666666",
            command=lambda n=node_id: self.show_dataset_summary(n)
        )
        summary_btn.pack(pady=(0, 8))
        self.summary_btn_ref = summary_btn
        self.detail_widgets.append(summary_btn)

        btn_f = ctk.CTkFrame(self.right, fg_color="transparent")
        btn_f.pack(pady=20)
        self.detail_widgets.append(btn_f)
        ctk.CTkButton(btn_f, text="Reset Calibrations", width=140, fg_color="#ff5555", text_color="white",
                      command=lambda n=node_id: self._reset_calibrations(n)).pack(side="left", padx=10)

        # Update dynamic values (connection/door status + calibration table)
        self._update_right_panel(node_id, refresh_table=True)
        self._scroll_details_to_top()

    def _update_right_panel(self, node_id, refresh_table=False):
        # Update the connection/door status labels (and refresh calibration table)
        if not node_id:
            return
        if self.node_number_label:
            self._safe_configure(self.node_number_label, text=node_id)
        if self.connection_label:
            presence = self.controller._node_presence_state(node_id)
            self._safe_configure(
                self.connection_label,
                text=presence.upper(),
                text_color=self.controller._presence_colors(presence),
            )

        # Show the last device-status update time for offline nodes.
        if self.last_seen_label:
            last_seen = self.controller.node_last_seen.get(node_id, 0)
            presence = self.controller._node_presence_state(node_id)
            now = time.time()
            if last_seen and presence != "online":
                age_s = int(now - last_seen)
                if age_s < 60:
                    ago = f"{age_s}s ago"
                else:
                    ago = f"{age_s//60}m ago"
                self._safe_configure(self.last_seen_label, text=f"Last seen: {ago}")
            else:
                self._safe_configure(self.last_seen_label, text="")

        # Show heartbeat age with color coding
        if self.heartbeat_label:
            last_hb = self.controller.node_last_heartbeat.get(node_id, 0)
            now = time.time()
            if last_hb > 0:
                age_s = int(now - last_hb)
                if age_s < 60:
                    ago_text = f"{age_s}s ago"
                elif age_s < 3600:
                    ago_text = f"{age_s//60}m ago"
                else:
                    ago_text = f"{age_s//3600}h ago"
                # Color coding: green (<60s), yellow (60-90s), red (>90s)
                if age_s < 60:
                    hb_color = "#2e7d32"  # green
                elif age_s < HEARTBEAT_TIMEOUT_S:
                    hb_color = "#f9a825"  # amber
                else:
                    hb_color = "#c62828"  # red
                self._safe_configure(self.heartbeat_label, text=ago_text, text_color=hb_color)
            else:
                self._safe_configure(self.heartbeat_label, text="Never", text_color="gray")

        if self.door_status_label:
            node_state = self.controller.node_detected_state.get(node_id, "")
            if node_state == "door_closed":
                door_status = "CLOSE"
            elif node_state == "door_open":
                door_status = "OPEN"
            elif node_state == "person_standing":
                door_status = "PERSON PRESENT"
            else:
                door_status = "UNKNOWN"
            self._safe_configure(self.door_status_label, text=door_status)

        if refresh_table:
            # rebuild the calibration table only when required; recreating this
            # widget tree too often causes visible flashing.
            self._render_calibration_table(node_id)

    def _latest_dataset_summary_path(self, node_id):
        candidates = [
            os.path.join(MODEL_STORE_DIR, node_id, "dataset_summary.json"),
            os.path.join(MODEL_STORE_DIR, "_tmp", node_id, "dataset_summary.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _current_campaign_id(self, node_id):
        campaign = self.controller.collection_campaign.get(node_id)
        if isinstance(campaign, str) and campaign.strip():
            return campaign.strip()

        data_base = self._resolve_data_base()
        node_dir = os.path.join(str(data_base), str(node_id))
        if not os.path.isdir(node_dir):
            return None

        latest = None
        for path in glob.glob(os.path.join(node_dir, "**", "*_manifest.json"), recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    manifest = json.load(fp)
                campaign_id = manifest.get("campaign_id")
                if not isinstance(campaign_id, str) or not campaign_id.strip():
                    continue
                mtime = os.path.getmtime(path)
                if latest is None or mtime > latest[0]:
                    latest = (mtime, campaign_id.strip())
            except Exception:
                continue

        return latest[1] if latest else None

    def _build_live_dataset_summary(self, node_id):
        cached = self._dataset_summary_cache.get(node_id)
        if cached is not None:
            return cached

        data_base = self._resolve_data_base()
        node_dir = os.path.join(str(data_base), str(node_id))
        if not os.path.isdir(node_dir):
            return None

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "node_id": node_id,
            "source": "live_scan",
            "campaign_id": self._current_campaign_id(node_id),
            "labels": {},
            "splits": {},
        }

        active_campaign = summary["campaign_id"]
        manifest_paths = [
            path for path in glob.glob(os.path.join(node_dir, "**", "*_manifest.json"), recursive=True)
            if os.path.isfile(path) and os.path.basename(path) != "session_manifest.json"
        ]

        has_campaign_data = False
        for state in CALIB_STATES:
            summary["labels"][state] = {
                "files": 0,
                "sessions": 0,
                "rows_raw": 0,
                "rows_clean": 0,
                "split_files": {"train": 0, "dev": 0, "test": 0},
                "split_rows": {"train": 0, "dev": 0, "test": 0},
            }

        if active_campaign and manifest_paths:
            for path in manifest_paths:
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        manifest = json.load(fp)
                except Exception:
                    continue

                if manifest.get("campaign_id") != active_campaign:
                    continue
                label = manifest.get("label")
                if label not in summary["labels"]:
                    continue

                row_count = int(manifest.get("row_count", 0))
                split_group = manifest.get("split_group", "train")
                if split_group not in {"train", "dev", "test"}:
                    split_group = "train"

                label_summary = summary["labels"][label]
                label_summary["files"] += 1
                label_summary["sessions"] += 1
                label_summary["rows_raw"] += row_count
                label_summary["rows_clean"] += row_count
                label_summary["split_files"][split_group] += 1
                label_summary["split_rows"][split_group] += row_count
                has_campaign_data = True

        if not has_campaign_data:
            for state in CALIB_STATES:
                state_dir = os.path.join(node_dir, state)
                csv_paths = [
                    path for path in glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True)
                    if os.path.isfile(path) and not os.path.basename(path).startswith("part_")
                ]
                total_rows = 0
                split_files = {"train": 0, "dev": 0, "test": 0}
                split_rows = {"train": 0, "dev": 0, "test": 0}
                sessions = set()
                for path in csv_paths:
                    sessions.add(os.path.splitext(os.path.basename(path))[0])
                    try:
                        df = pd.read_csv(path)
                        total_rows += int(len(df))
                        split_col = None
                        detected_split = ""
                        for candidate in ("collection_split", "split_group", "dataset_split"):
                            if candidate in df.columns:
                                split_col = candidate
                                break
                        if split_col:
                            split_series = df[split_col].astype(str).str.lower()
                            split_series = split_series[split_series.isin(("train", "dev", "test"))]
                            if not split_series.empty:
                                detected_split = str(split_series.mode().iloc[0])
                            counts = split_series.value_counts().to_dict()
                            for split_name in split_rows:
                                split_rows[split_name] += int(counts.get(split_name, 0))
                        else:
                            split_rows["train"] += int(len(df))

                        if not detected_split:
                            base_name = os.path.basename(path).lower()
                            for split_name in ("train", "dev", "test"):
                                if f"_{split_name}_" in base_name:
                                    detected_split = split_name
                                    break
                        if detected_split not in ("train", "dev", "test"):
                            detected_split = "train"
                        split_files[detected_split] += 1
                    except Exception:
                        pass
                summary["labels"][state] = {
                    "files": int(len(csv_paths)),
                    "sessions": int(len(sessions)),
                    "rows_raw": int(total_rows),
                    "rows_clean": int(total_rows),
                    "split_files": split_files,
                    "split_rows": split_rows,
                }

        for split_name in ("train", "dev", "test"):
            summary["splits"][split_name] = {
                "rows": int(sum(info.get("split_rows", {}).get(split_name, 0) for info in summary["labels"].values())),
            }

        self._dataset_summary_cache.set(node_id, summary)
        return summary

    def show_dataset_summary(self, node_id):
        summary = None
        summary_path = self._latest_dataset_summary_path(node_id)
        if summary_path:
            try:
                with open(summary_path, "r", encoding="utf-8") as fp:
                    summary = json.load(fp)
            except Exception as e:
                self.controller._log("Model", f"Failed to read dataset summary for {node_id}: {e}")

        if summary is None:
            summary = self._build_live_dataset_summary(node_id)

        popup = ctk.CTkToplevel(self)
        popup.title(f"Dataset Summary - {node_id}")
        popup.geometry("760x540")
        popup.transient(self)
        popup.lift()

        box = ctk.CTkTextbox(popup, wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=12)

        if not summary:
            box.insert("end", "No dataset summary is available yet.\n\n")
            box.insert("end", "Collect at least one session per label and run training once to generate the curated split summary.\n")
            box.configure(state="disabled")
            return

        lines = [f"Node: {node_id}", f"Generated: {summary.get('generated_at', 'unknown')}"]
        if summary.get("source"):
            lines.append(f"Source: {summary['source']}")
        overall = summary.get("overall", {})
        if overall:
            lines.append("")
            lines.append("Overall")
            lines.append(f"  Raw rows: {overall.get('rows_raw', 'n/a')}")
            lines.append(f"  Clean rows: {overall.get('rows_clean', 'n/a')}")
            lines.append(f"  Redundant rows: {overall.get('redundant_rows', 'n/a')}")
            lines.append(f"  Redundancy rate: {overall.get('redundancy_rate', 'n/a')}")

        labels = summary.get("labels", {})
        if labels:
            lines.append("")
            lines.append("Per-label coverage")
            for state in CALIB_STATES:
                info = labels.get(state, {})
                lines.append(
                    f"  {state}: raw_rows={info.get('rows_raw', info.get('files', 'n/a'))}, "
                    f"clean_rows={info.get('rows_clean', 'n/a')}, sessions={info.get('sessions_raw', info.get('sessions', 'n/a'))}, "
                    f"redundant={info.get('redundant_rows', 'n/a')}"
                )
                split_rows = info.get("split_rows", {})
                if split_rows:
                    # Only show the test split if it actually has data
                    if split_rows.get('test', 0):
                        lines.append(
                            f"    split rows -> train={split_rows.get('train', 0)}, dev={split_rows.get('dev', 0)}, test={split_rows.get('test', 0)}"
                        )
                    else:
                        lines.append(
                            f"    split rows -> train={split_rows.get('train', 0)}, dev={split_rows.get('dev', 0)}"
                        )

        splits = summary.get("splits", {})
        if splits:
            lines.append("")
            lines.append("Split coverage")
            for split_name in ("train", "dev", "test"):
                split_info = splits.get(split_name, {})
                rows = split_info.get('rows', 0)
                if split_name == "test" and not rows:
                    continue
                lines.append(
                    f"  {split_name}: rows={split_info.get('rows', 'n/a')}, sessions={split_info.get('sessions', 'n/a')}"
                )
                per_label = split_info.get("per_label", {})
                for state in CALIB_STATES:
                    state_info = per_label.get(state, {})
                    lines.append(
                        f"    - {state}: rows={state_info.get('rows', 'n/a')}, sessions={state_info.get('sessions', 'n/a')}"
                    )

        session_splits = summary.get("session_splits", {})
        if session_splits:
            lines.append("")
            lines.append("Session split allocation")
            for state in CALIB_STATES:
                state_info = session_splits.get(state, {})
                train_count = state_info.get('train_count', 'n/a')
                dev_count = state_info.get('dev_count', 'n/a')
                test_count = state_info.get('test_count', 0)
                if test_count:
                    lines.append(
                        f"  {state}: train={train_count}, dev={dev_count}, test={test_count}"
                    )
                else:
                    lines.append(
                        f"  {state}: train={train_count}, dev={dev_count}"
                    )

        box.insert("end", "\n".join(str(line) for line in lines))
        box.configure(state="disabled")
        try:
            self.right.update_idletasks()
        except Exception:
            pass

    def set_progress(self, cur, total):
        # initialize progress bar if missing
        if not hasattr(self, 'progress_bar'):
            self.progress_bar = ctk.CTkProgressBar(self.right, width=300)
            self.progress_bar.pack(pady=(4,10))
        self.progress_bar.set(cur/total)
        self._safe_configure(self.status_lbl_ref, text=f"Status: {cur}/{total} sub-batches uploaded")

    def _render_calibration_table(self, node_id):
        # ensure the map exists (in case we added a node later)
        state_map = self.controller.esp_nodes.setdefault(node_id, {s: False for s in CALIB_STATES})
        live_summary = self._build_live_dataset_summary(node_id) or {}
        current_campaign = live_summary.get("campaign_id", "")

        if self._calibration_title_ref is None:
            self._calibration_title_ref = ctk.CTkLabel(
                self.right,
                text="Last Calibrated States",
                text_color="black",
                font=("Arial", 16, "bold"),
            )
            self._calibration_title_ref.pack(anchor="w", padx=30, pady=(2, 6))
            self.calibration_table_widgets.append(self._calibration_title_ref)
            self.detail_widgets.append(self._calibration_title_ref)

        if self._calibration_campaign_ref is None:
            self._calibration_campaign_ref = ctk.CTkLabel(
                self.right,
                text="Current session: -",
                text_color="#666666",
                font=("Arial", 12, "italic"),
            )
            self._calibration_campaign_ref.pack(anchor="w", padx=30, pady=(0, 8))
            self.calibration_table_widgets.append(self._calibration_campaign_ref)
            self.detail_widgets.append(self._calibration_campaign_ref)

        campaign_text = f"Current session: {current_campaign}" if current_campaign else "Current session: -"
        self._safe_configure(self._calibration_campaign_ref, text=campaign_text)

        for state in CALIB_STATES:
            row_refs = self._calibration_row_refs.get(state)
            if row_refs is None:
                row = ctk.CTkFrame(self.right, fg_color="transparent")
                row.pack(fill="x", padx=30, pady=2)
                state_lbl = ctk.CTkLabel(row, text=state, text_color="black", font=("Arial", 14))
                state_lbl.pack(side="left")
                count_lbl = ctk.CTkLabel(row, text="", text_color="#666666", font=("Arial", 12))
                count_lbl.pack(side="right", padx=(8, 0))
                status_lbl = ctk.CTkLabel(row, text="", text_color="gray", font=("Arial", 14))
                status_lbl.pack(side="right")
                row_refs = {
                    "row": row,
                    "count_lbl": count_lbl,
                    "status_lbl": status_lbl,
                }
                self._calibration_row_refs[state] = row_refs
                self.calibration_table_widgets.append(row)
                self.detail_widgets.append(row)

            info = live_summary.get("labels", {}).get(state, {})
            row_count = int(info.get("rows_raw", 0))
            file_count = int(info.get("files", 0))
            split_files = info.get("split_files", {}) if isinstance(info, dict) else {}
            train_files = int(split_files.get("train", 0))
            dev_files = int(split_files.get("dev", 0))
            test_files = int(split_files.get("test", 0))
            count_text = (
                f"{file_count} file{'s' if file_count != 1 else ''}, {row_count} row{'s' if row_count != 1 else ''} "
                f"| train: {train_files}, dev: {dev_files}, test: {test_files}"
            )
            self._safe_configure(row_refs["count_lbl"], text=count_text)
            done = state_map.get(state, False)
            status_text = "Γ£ô Done" if done else "Γùï Pending"
            status_color = "#2e7d32" if done else "gray"
            self._safe_configure(row_refs["status_lbl"], text=status_text, text_color=status_color)

        all_done = all(state_map.get(s, False) for s in CALIB_STATES)
        if all_done:
            trained = self.controller.model_state.get(node_id, False)
            status_txt = "Model: Ready" if trained else "Model: Invalid (retrain required)"
            status_col = "#2e7d32" if trained else "#d32f2f"

            if self.model_status_lbl_ref is None:
                self.model_status_lbl_ref = ctk.CTkLabel(
                    self.right,
                    text=status_txt,
                    text_color=status_col,
                    font=("Arial", 13, "bold"),
                )
                self.model_status_lbl_ref.pack(anchor="w", padx=30, pady=(8, 4))
                self.calibration_table_widgets.append(self.model_status_lbl_ref)
                self.detail_widgets.append(self.model_status_lbl_ref)
            else:
                if not self.model_status_lbl_ref.winfo_manager():
                    self.model_status_lbl_ref.pack(anchor="w", padx=30, pady=(8, 4))
                self._safe_configure(self.model_status_lbl_ref, text=status_txt, text_color=status_col)

            eligible = self._auto_training_splits_ready(node_id)
            if eligible:
                if self._training_eligibility_lbl_ref is not None and self._training_eligibility_lbl_ref.winfo_manager():
                    self._training_eligibility_lbl_ref.pack_forget()

                if self.train_btn_ref is None:
                    self.train_btn_ref = ctk.CTkButton(
                        self.right,
                        text="Retrain Model" if trained else "Train Model",
                        fg_color="#3b8ed0",
                        command=lambda n=node_id: self.train_model(n),
                    )
                    self.calibration_table_widgets.append(self.train_btn_ref)
                    self.detail_widgets.append(self.train_btn_ref)
                else:
                    self._safe_configure(
                        self.train_btn_ref,
                        text="Retrain Model" if trained else "Train Model",
                        command=lambda n=node_id: self.train_model(n),
                    )
                if not self.train_btn_ref.winfo_manager():
                    self.train_btn_ref.pack(anchor="w", padx=30, pady=(2, 10))
            else:
                if self.train_btn_ref is not None and self.train_btn_ref.winfo_manager():
                    self.train_btn_ref.pack_forget()
                if self._training_eligibility_lbl_ref is None:
                    self._training_eligibility_lbl_ref = ctk.CTkLabel(
                        self.right,
                        text="Training locked: each state needs ΓëÑ1 train and ΓëÑ1 dev session (test split optional; min 6 total).",
                        text_color="#ff9800",
                        font=("Arial", 12, "italic"),
                    )
                    self.calibration_table_widgets.append(self._training_eligibility_lbl_ref)
                    self.detail_widgets.append(self._training_eligibility_lbl_ref)
                if not self._training_eligibility_lbl_ref.winfo_manager():
                    self._training_eligibility_lbl_ref.pack(anchor="w", padx=30, pady=(2, 10))
        else:
            if self.model_status_lbl_ref is not None and self.model_status_lbl_ref.winfo_manager():
                self.model_status_lbl_ref.pack_forget()
            if self.train_btn_ref is not None and self.train_btn_ref.winfo_manager():
                self.train_btn_ref.pack_forget()
            if self._training_eligibility_lbl_ref is not None and self._training_eligibility_lbl_ref.winfo_manager():
                self._training_eligibility_lbl_ref.pack_forget()

    def _auto_training_splits_ready(self, node_id):
        """Return True when each state has >=1 train and >=1 dev session.

        Eligibility per rack:
        - For every state in CALIB_STATES, split_files.train >= 1 and split_files.dev >= 1
        - Test split is optional for eligibility.
        - Minimum required total is 6 files across the 3 states.
        """
        summary = self._build_live_dataset_summary(node_id) or {}
        labels = summary.get("labels", {})
        for state in CALIB_STATES:
            info = labels.get(state, {})
            split_files = info.get("split_files", {}) if isinstance(info, dict) else {}
            train_files = int(split_files.get("train", 0))
            dev_files = int(split_files.get("dev", 0))
            if train_files < 1 or dev_files < 1:
                return False
        return True

    def _resolve_data_base(self):
        configured = getattr(self.controller, "csi_data_dir", None)
        candidates = []
        if isinstance(configured, str) and configured.strip():
            candidates.append(configured)
        if isinstance(CSI_DATA_DIR, str) and CSI_DATA_DIR.strip():
            candidates.append(CSI_DATA_DIR)
        candidates.append("csi_data")

        # Prefer existing directories; fall back to a sane default.
        for path in candidates:
            try:
                if path and os.path.exists(path):
                    return path
            except Exception:
                continue

        # If nothing exists, ensure we still return a usable path.
        fallback = candidates[-1] if candidates else "csi_data"
        try:
            os.makedirs(fallback, exist_ok=True)
        except Exception:
            pass
        return fallback

    def _safe_configure(self, widget, **kwargs):
        """Safely configure a widget that may have been destroyed.

        Tkinter widgets can be destroyed while background work is still running.
        This helper ensures we only call .configure on an existing widget.
        """
        try:
            if widget is not None and getattr(widget, "winfo_exists", lambda: False)():
                widget.configure(**kwargs)
        except Exception:
            pass

    def _scroll_details_to_top(self):
        """Best-effort reset of the right-side scrollable frame to the top."""
        try:
            if hasattr(self.right, "yview_moveto"):
                self.right.yview_moveto(0)
                return
        except Exception:
            pass

        for attr_name in ("_parent_canvas", "canvas", "_canvas"):
            try:
                canvas = getattr(self.right, attr_name, None)
                if canvas is not None and hasattr(canvas, "yview_moveto"):
                    canvas.yview_moveto(0)
                    return
            except Exception:
                pass

    def train_model(self, node_id):
        state_map = self.controller.esp_nodes.get(node_id, {})
        if not all(state_map.get(s, False) for s in CALIB_STATES):
            popup = ctk.CTkToplevel(self)
            popup.title("Training blocked")
            popup.geometry("380x140")
            ctk.CTkLabel(popup, text="Complete all calibration states first.", font=("Arial", 14)).pack(pady=20)
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            return

        if not self._auto_training_splits_ready(node_id):
            popup = ctk.CTkToplevel(self)
            popup.title("Training blocked")
            popup.geometry("520x170")
            ctk.CTkLabel(
                popup,
                text="Training requires per state: at least 1 train and 1 dev session.",
                font=("Arial", 14),
                wraplength=480,
            ).pack(pady=(20, 8))
            ctk.CTkLabel(
                popup,
                text="Minimum total per rack: 6 files (3 states ├ù train+dev).",
                text_color="gray",
            ).pack(pady=(0, 12))
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            return

        if self.controller.training_in_progress.get(node_id, False):
            self.controller._log("Model", f"Training already in progress for {node_id}")
            self._safe_configure(self.status_lbl_ref, text="Status: Training already in progress", text_color="#1a4d66")
            return

        self.controller.training_in_progress[node_id] = True

        # pick a valid CSI directory, avoid NoneType
        data_base = self._resolve_data_base()
        output_tmp_dir = os.path.join(MODEL_STORE_DIR, "_tmp", node_id)
        os.makedirs(output_tmp_dir, exist_ok=True)
        model_output = os.path.join(output_tmp_dir, "model.tflite")
        scaler_output = os.path.join(output_tmp_dir, "scaler_params.json")

        cmd = [
            sys.executable,
            TRAINING_SCRIPT,
            "--data-dir", data_base,
            "--node", node_id,
            "--output", model_output,
            "--scaler-output", scaler_output,
        ]

        if PASO_ENABLE_PRUNING:
            cmd.extend([
                "--enable-pruning",
                "--pruning-sparsity", str(PASO_PRUNING_SPARSITY),
                "--pruning-finetune-epochs", str(PASO_PRUNING_FINETUNE_EPOCHS),
                "--max-accuracy-drop", str(PASO_MAX_ACCURACY_DROP),
            ])
        if PASO_FEATURE_CONDENSE_GROUPS > 0:
            cmd.extend([
                "--feature-condense-groups", str(PASO_FEATURE_CONDENSE_GROUPS),
            ])

        # disable the train button to prevent reΓÇæentry
        self._safe_configure(self.train_btn_ref, state="disabled")
        self._safe_configure(self.status_lbl_ref, text=f"Status: Training model for {node_id}...", text_color="#1a4d66")

        # show a simple progress popup while training runs
        progress_popup = ctk.CTkToplevel(self)
        progress_popup.title("Training in progress")
        progress_popup.geometry("320x100")
        ctk.CTkLabel(progress_popup, text="Training in progress, please wait...").pack(pady=20)
        
        # Add cleanup handler to prevent memory leak
        def cleanup_popup():
            if node_id in self._training_popups:
                del self._training_popups[node_id]
            progress_popup.destroy()
        
        progress_popup.protocol("WM_DELETE_WINDOW", cleanup_popup)
        self._training_popups[node_id] = progress_popup

        worker = threading.Thread(
            target=self._training_worker,
            args=(node_id, cmd, model_output, scaler_output),
            daemon=True,
        )
        worker.start()

    def _training_worker(self, node_id, cmd, model_output, scaler_output):
        returncode = None
        stdout = ""
        stderr = ""
        err = ""

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if proc.returncode != 0:
                err = stderr.strip()
        except Exception as e:
            err = str(e)

        self._training_result_queue.put({
            "node_id": node_id,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "err": err,
            "model_output": model_output,
            "scaler_output": scaler_output,
        }, block=True)

    def _process_training_results(self):
        if not getattr(self, "winfo_exists", lambda: False)():
            try:
                self.controller._log("UI", "Skipped training-result callback: widget no longer exists")
            except Exception:
                pass
            return

        try:
            while True:
                result = self._training_result_queue.get_nowait()
                self._handle_training_result(result)
        except queue.Empty:
            pass
        finally:
            if getattr(self, "winfo_exists", lambda: False)():
                self.after(150, self._process_training_results)

    def _handle_training_result(self, result):
        node_id = result.get("node_id")
        returncode = result.get("returncode")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        err = result.get("err", "")
        model_output = result.get("model_output")
        scaler_output = result.get("scaler_output")

        popup = self._training_popups.pop(node_id, None)
        try:
            if popup is not None and getattr(popup, "winfo_exists", lambda: False)():
                popup.destroy()
        except Exception:
            pass

        if stdout:
            for line in stdout.splitlines():
                if line.strip():
                    self.controller._log("Model", line.strip())
                    if self.current_node_id == node_id and ("Dataset shape" in line or "Test loss" in line):
                        self._safe_configure(self.status_lbl_ref, text=line.strip(), text_color="#1a4d66")

        training_failed = (returncode is None) or (returncode != 0)
        if training_failed:
            if stdout:
                print("[TRAINING] stdout:\n", stdout)
            if stderr:
                print("[TRAINING] stderr:\n", stderr)
            for line in stderr.splitlines():
                self.controller._log("Model", line)

            self.controller._log("Model", f"Training failed for {node_id}: {err or 'unknown error'}")

            if self.current_node_id == node_id:
                self._safe_configure(self.status_lbl_ref, text="Status: Training failed", text_color="red")
                popup = ctk.CTkToplevel(self)
                popup.title("Training failed")
                popup.geometry("480x180")
                ctk.CTkLabel(popup, text="Model training failed.", font=("Arial", 16, "bold")).pack(pady=(16, 8))
                ctk.CTkLabel(popup, text=err or "Check logs for details.", wraplength=440, text_color="gray").pack(pady=(0, 12))
                ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
                self._safe_configure(self.train_btn_ref, state="normal")

            self.controller.training_in_progress[node_id] = False
            return

        copied = self.controller._copy_model_to_server(node_id, model_output, scaler_output)
        if copied:
            session = datetime.now().strftime("%Y%m%d%H%M%S")
            self.controller.model_state[node_id] = False  # set true when ESP reports model_ready
            self.controller._save_model_state()
            self.controller._notify_training_complete(node_id, session)
            if self.current_node_id == node_id:
                self._safe_configure(self.status_lbl_ref, text="Status: Model trained; waiting for device to pull", text_color="#1a4d66")
        else:
            self.controller._log("Model", f"Failed to copy model to server for {node_id}")
            if self.current_node_id == node_id:
                self._safe_configure(self.status_lbl_ref, text="Status: Model trained but copy failed", text_color="#d32f2f")

        if self.current_node_id == node_id:
            self._safe_configure(self.train_btn_ref, state="normal")
            self._render_calibration_table(node_id)
        self.controller.training_in_progress[node_id] = False

    def start_calibration(self, node_id):
        if not node_id or node_id not in self.controller.esp_nodes:
            popup = ctk.CTkToplevel(self)
            popup.title("Calibration error")
            popup.geometry("360x140")
            ctk.CTkLabel(popup, text="Select a valid rack before starting calibration.", wraplength=320).pack(pady=20)
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            return

        selected_state = self.radio_var.get()
        selected_split = self.split_var.get()
        if selected_split not in {"train", "dev", "test"}:
            selected_split = "train"
        self.controller.last_split_choice[node_id] = selected_split
        # remember choice so UI can restore it later
        self.controller.last_calib_choice[node_id] = selected_state

        # Determine if we already have collected any data for this node/state/split
        data_base = self._resolve_data_base()
        has_split_files = False
        if isinstance(data_base, (str, os.PathLike)) and data_base:
            state_dir = os.path.join(str(data_base), str(node_id), str(selected_state))
            if os.path.isdir(state_dir):
                pattern = f"csi_{selected_state}_{selected_split}_*.csv"
                has_split_files = bool(glob.glob(os.path.join(state_dir, pattern)))

        # Append mode is the default (no prompt).  Prompt only if the user explicitly
        # enables it via the checkbox and we have existing data for the selected split.
        existing = self.controller.esp_nodes.get(node_id, {})
        prompt_conflict = has_split_files and self.ask_overwrite_var.get()
        if prompt_conflict:
            choice = self._ask_calibration_mode(node_id, selected_state, selected_split)
            if choice == "cancel":
                self._safe_configure(self.status_lbl_ref, text="Status: Calibration cancelled.", text_color="gray")
                return
            if choice == "reset":
                # Reset the entire rack for this selected node.
                self._reset_calibrations(node_id=node_id, scope="rack")
            elif choice == "replace":
                # Remove only the existing files for the selected split so we can start fresh.
                if isinstance(data_base, (str, os.PathLike)) and data_base:
                    state_dir = os.path.join(str(data_base), str(node_id), str(selected_state))
                    if os.path.isdir(state_dir):
                        for f in glob.glob(os.path.join(state_dir, f"csi_{selected_state}_{selected_split}_*.csv")):
                            try:
                                os.remove(f)
                            except Exception:
                                pass
                        # If no data remains for this state at all, clear the completed flag.
                        remaining = bool(glob.glob(os.path.join(state_dir, f"csi_{selected_state}_*.csv")))
                        if not remaining:
                            existing[selected_state] = False
                            self.controller._save_calib_states()
            # append is a no-op: keep old data and add another session

        # any fresh calibration invalidates previously trained model
        self.controller._mark_model_dirty(node_id)

        # keep prior CSI sessions so each label can accumulate multiple runs;
        # only the calibration flags change here.
        data_base = self._resolve_data_base()
        if not isinstance(data_base, (str, os.PathLike)) or not data_base:
            self.controller._log("Model", f"Invalid CSI base directory: {data_base!r}")
            popup = ctk.CTkToplevel(self)
            popup.title("Calibration error")
            popup.geometry("420x160")
            ctk.CTkLabel(popup, text="Cannot determine CSI data directory.", font=("Arial", 14, "bold")).pack(pady=(16, 8))
            ctk.CTkLabel(popup, text="Please configure a valid CSI data directory in Settings.", wraplength=400, text_color="gray").pack(pady=(0, 12))
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            return

        state_dir = os.path.join(str(data_base), str(node_id), str(selected_state))
        if os.path.isdir(state_dir):
            self.controller._log("Model", f"Appending new CSI session under {node_id}/{selected_state}")
        self.controller._dataset_summary_cache.invalidate(node_id)

        # reset progress UI
        try:
            self.set_progress(0,1)
        except Exception:
            pass
        # attempt to publish command with retries
        ok = self.controller._publish_collect_with_retry(node_id, selected_state, split_group=selected_split, retries=3)
        if not ok:
            # publication failed locally; inform user and leave button enabled
            self._safe_configure(
                self.status_lbl_ref,
                text="Status: Publish failed",
                text_color="red"
            )
            # revert to idle after a moment
            def restore_idle():
                self._safe_configure(self.status_lbl_ref, text="Status: Idle", text_color="gray")
            self.after(3000, restore_idle)
            return

        # disable button only after we know the message was sent successfully
        if self.start_btn_ref:
            self._safe_configure(self.start_btn_ref, state="disabled")

        self._safe_configure(
            self.status_lbl_ref,
            text=f"Status: Calibrating {node_id} for {selected_state} [{selected_split}]... Please wait (~20s)",
            text_color="#1a4d66"
        )

    def start_selected_racks(self):
        selected_nodes = [node for node, var in self.bulk_selected.items() if var.get()]
        if not selected_nodes:
            popup = ctk.CTkToplevel(self)
            popup.title("No racks selected")
            popup.geometry("320x120")
            ctk.CTkLabel(popup, text="Select at least one rack.", font=("Arial", 14)).pack(pady=20)
            ctk.CTkButton(popup, text="OK", width=80, command=popup.destroy).pack()
            return

        selected_state = self.bulk_label_var.get()
        selected_split = self.bulk_split_var.get()
        if selected_split not in {"train", "dev", "test"}:
            selected_split = "train"
        success_count = 0
        failed_nodes = []
        for node_id in selected_nodes:
            self.controller.last_calib_choice[node_id] = selected_state
            self.controller.last_split_choice[node_id] = selected_split
            self.controller._mark_model_dirty(node_id)
            self.controller._dataset_summary_cache.invalidate(node_id)

            data_base = self._resolve_data_base()
            state_dir = os.path.join(data_base, node_id, selected_state)
            if os.path.isdir(state_dir):
                self.controller._log("Bulk", f"Appending new CSI session under {node_id}/{selected_state}")

            if self.controller._publish_collect_with_retry(node_id, selected_state, split_group=selected_split, retries=3):
                success_count += 1
            else:
                failed_nodes.append(node_id)

        if failed_nodes:
            self.controller._log(
                "Bulk",
                f"Started {selected_state} [{selected_split}] for {success_count}/{len(selected_nodes)} racks; failed: {', '.join(failed_nodes)}",
            )
        else:
            self.controller._log("Bulk", f"Started {selected_state} [{selected_split}] for {success_count}/{len(selected_nodes)} racks")

        popup = ctk.CTkToplevel(self)
        popup.title("Bulk calibration")
        popup.geometry("420x170")
        msg = f"Started {selected_state} [{selected_split}] on {success_count}/{len(selected_nodes)} selected racks."
        if failed_nodes:
            msg += f"\nFailed: {', '.join(failed_nodes)}"
        ctk.CTkLabel(popup, text=msg, font=("Arial", 13), justify="left").pack(pady=24)
        ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()

    def on_calibration_complete(self, node_id, state, session=None, source=None):
        expected_session = self.controller.expected_session.get(node_id)
        completion_source = source
        completion_session = session
        # if we previously sent a command, verify the returned label
        expected = self.controller.expected_calib.get(node_id)
        if expected is not None and expected != state:
            # mismatch occurred; warn and let user retry explicitly
            print(f"WARNING: node {node_id} reported '{state}' but expected '{expected}'")
            if self.current_node_id == node_id:
                # still on the same node, show a warning popup
                self._show_mismatch_popup(node_id, state, expected)
                # recover UI so user can retry
                self._safe_configure(self.start_btn_ref, state="normal")
                self._safe_configure(self.status_lbl_ref, text="Status: Idle", text_color="gray")
            # clear expected values; require explicit retry from user
            self.controller.expected_calib.pop(node_id, None)
            self.controller.expected_session.pop(node_id, None)
            return

        # validate completion session against expected session
        if expected_session is not None and session and session != expected_session:
            self.controller._log("Mismatch", f"{node_id} session mismatch: got {session}, expected {expected_session}; event ignored")
            print(f"WARNING: node {node_id} reported session '{session}' but expected '{expected_session}'")
            if self.current_node_id == node_id:
                self._safe_configure(self.start_btn_ref, state="normal")
                self._safe_configure(self.status_lbl_ref, text="Status: Idle", text_color="gray")
            self.controller.expected_calib.pop(node_id, None)
            self.controller.expected_session.pop(node_id, None)
            return

        # ignore no-session fallback completions when we have no active expected session
        if expected_session is None and not session:
            self.controller._log("Model", f"Ignored unsolicited firmware fallback completion for {node_id}/{state} (no active expected session)")
            self.controller.expected_calib.pop(node_id, None)
            self.controller.expected_session.pop(node_id, None)
            return

        # either we had no expectation, or the labels match
        # `completion_source` explicitly marks whether this event came from the server
        # or from firmware fallback.  Server events are authoritative; firmware
        # events are a best-effort unlock path.
        if completion_source == "server":
            server_complete = True
        elif completion_source == "firmware":
            server_complete = False
        else:
            server_complete = bool(completion_session)

        was_done = self.controller.esp_nodes.get(node_id, {}).get(state, False)
        newly_marked_done = False
        if node_id in self.controller.esp_nodes and state in self.controller.esp_nodes[node_id]:
            if not was_done:
                self.controller.esp_nodes[node_id][state] = True
                self.controller._save_calib_states()
                newly_marked_done = True
        self.controller._dataset_summary_cache.invalidate(node_id)

        if server_complete:
            self.controller._log("Model", f"Server merge complete for {node_id}/{state} (session={completion_session})")
        else:
            fallback_desc = "firmware fallback" if completion_source == "firmware" else "no server session"
            self.controller._log("Model", f"Firmware completion for {node_id}/{state} ({fallback_desc})")

        all_done = all(self.controller.esp_nodes[node_id].get(s, False) for s in CALIB_STATES)
        splits_ready = self._auto_training_splits_ready(node_id)
        if all_done and splits_ready:
            self.controller.auto_training_hold_logged[node_id] = False
            if not self.controller.training_in_progress.get(node_id, False):
                self.controller._log("Model", f"Training eligible for {node_id}; press Train Model to start")
            if self.current_node_id == node_id:
                self._safe_configure(
                    self.status_lbl_ref,
                    text="Status: Training eligible ΓÇö press Train Model",
                    text_color="#1a4d66"
                )
        elif all_done and not splits_ready:
            if self.current_node_id == node_id:
                self._safe_configure(
                    self.status_lbl_ref,
                    text="Status: Waiting for train+dev data for each state",
                    text_color="#ff9800"
                )
            if not self.controller.auto_training_hold_logged.get(node_id, False):
                self.controller._log("Model", f"Training held for {node_id}: each state needs 1 train + 1 dev")
                self.controller.auto_training_hold_logged[node_id] = True
        elif all_done and self.controller.training_in_progress.get(node_id, False):
            self.controller.auto_training_hold_logged[node_id] = False
            self.controller._log("Model", f"Training already running for {node_id}; skipping duplicate trigger")
        else:
            self.controller.auto_training_hold_logged[node_id] = False

        # remove the stored expectations (if any)
        self.controller.expected_calib.pop(node_id, None)
        self.controller.expected_session.pop(node_id, None)

        # if user has navigated away, no further UI changes needed
        if self.current_node_id != node_id:
            return

        # unlock next calibration on either authoritative server complete or
        # firmware fallback completion.
        self._safe_configure(self.start_btn_ref, state="normal")
        if all_done and splits_ready:
            self._safe_configure(self.status_lbl_ref, text="Status: Training eligible ΓÇö press Train Model", text_color="#1a4d66")
        elif server_complete:
            self._safe_configure(self.status_lbl_ref, text="Status: Idle", text_color="gray")
        else:
            self._safe_configure(self.status_lbl_ref, text="Status: Idle (firmware fallback)", text_color="#1a4d66")

        self._render_calibration_table(node_id)
        # clear progress bar after completion
        if self.current_node_id == node_id and hasattr(self, 'progress_bar'):
            self.progress_bar.set(0)
        if newly_marked_done:
            self._show_calibration_popup(node_id, state)

    def _show_calibration_popup(self, node_id, state):
        popup = ctk.CTkToplevel(self)
        popup.title("Calibration Complete!")
        popup.geometry("340x180")

        ctk.CTkLabel(popup, text="Calibration Complete!", font=("Arial", 20, "bold")).pack(pady=(20, 8))
        ctk.CTkLabel(popup, text=f"{node_id} - {state}", text_color="gray").pack(pady=(0, 16))

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(pady=8)

        ctk.CTkButton(btn_row, text="Close", fg_color="white", text_color="black", border_width=1,
                      command=popup.destroy).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="View Graph", fg_color="#3b8ed0",
                      command=lambda: self._open_graph_and_close(popup, node_id, state)).pack(side="left", padx=8)

    def _open_graph_and_close(self, popup, node_id, state):
        popup.destroy()
        self.view_graph(node_id, state)

    def view_graph(self, node_id, state):
        # pick base directory dynamically; preference given to configured path
        base = self._resolve_data_base()
        state_dir = os.path.join(base, node_id, state)
        csv_files = glob.glob(os.path.join(state_dir, "*.csv"))

        if not csv_files:
            no_data = ctk.CTkToplevel(self)
            no_data.title("No Data")
            no_data.geometry("300x120")
            ctk.CTkLabel(no_data, text="No CSV files found for this calibration.").pack(pady=16)
            ctk.CTkButton(no_data, text="Close", command=no_data.destroy).pack(pady=8)
            return

        latest_csv = max(csv_files, key=os.path.getmtime)
        df = pd.read_csv(latest_csv)

        subcarrier_cols = [col for col in df.columns if col.startswith("SC_")]
        if not subcarrier_cols:
            subcarrier_cols = df.select_dtypes(include="number").columns.tolist()

        if not subcarrier_cols:
            invalid = ctk.CTkToplevel(self)
            invalid.title("Invalid CSV")
            invalid.geometry("300x120")
            ctk.CTkLabel(invalid, text="No numeric subcarrier columns found.").pack(pady=16)
            ctk.CTkButton(invalid, text="Close", command=invalid.destroy).pack(pady=8)
            return

        mean_series = df[subcarrier_cols].mean(axis=1)

        # import pyplot only when needed
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(mean_series, label=f"{node_id} - {state} mean")
        plt.title(f"CSI Mean Signal ({node_id} | {state})")
        plt.xlabel("Sample Index")
        plt.ylabel("Mean Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _show_mismatch_popup(self, node_id, received, expected):
        # show visible warning popup
        popup = ctk.CTkToplevel(self)
        popup.title("Calibration Label Mismatch")
        popup.geometry("420x170")
        ctk.CTkLabel(popup, text="Calibration mismatch detected", font=("Arial", 18, "bold")).pack(pady=(16, 8))
        ctk.CTkLabel(
            popup,
            text=f"{node_id} reported '{received}' but expected '{expected}'.",
            text_color="gray"
        ).pack(pady=(0, 12))
        ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()

        # also add to log
        self.controller._log('Mismatch', f'{node_id} reported {received} (expected {expected})')
        # update log panel if visible
        if "MainDashboard" in self.controller.frames:
            subs = self.controller.frames["MainDashboard"].sub_frames
            if "Logs" in subs:
                subs["Logs"].refresh()

    def _delete_state_data(self, node_id: str, state: str):
        data_base = self._resolve_data_base()
        state_dir = os.path.join(str(data_base), str(node_id), str(state))
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)

    def _delete_rack_data(self, node_id: str):
        data_base = self._resolve_data_base()
        node_dir = os.path.join(str(data_base), str(node_id))
        if os.path.isdir(node_dir):
            shutil.rmtree(node_dir)

    def _delete_rack_model_artifacts(self, node_id: str):
        node_model_dir = os.path.join(MODEL_STORE_DIR, node_id)
        node_tmp_dir = os.path.join(MODEL_STORE_DIR, "_tmp", node_id)
        if os.path.isdir(node_model_dir):
            shutil.rmtree(node_model_dir)
        if os.path.isdir(node_tmp_dir):
            shutil.rmtree(node_tmp_dir)

    def _reset_calibrations(self, node_id=None, scope=None):
        node_id = node_id or self.current_node_id
        if not node_id or node_id not in self.controller.esp_nodes:
            return

        selected_state = self.radio_var.get() if hasattr(self, "radio_var") else CALIB_STATES[0]
        reset_scope = scope or self._ask_reset_scope(node_id, selected_state)
        if reset_scope == "cancel":
            self.controller._log('Action', f'Reset cancelled for {node_id}')
            return

        try:
            if reset_scope == "state":
                self._delete_state_data(node_id, selected_state)
                self.controller.esp_nodes[node_id][selected_state] = False
                self.controller._mark_model_dirty(node_id)
                self.controller._dataset_summary_cache.invalidate(node_id)
                self.controller._log('Action', f'Reset calibration state {node_id}/{selected_state}')
            elif reset_scope == "rack":
                self._delete_rack_data(node_id)
                self._delete_rack_model_artifacts(node_id)
                for state in self.controller.esp_nodes[node_id]:
                    self.controller.esp_nodes[node_id][state] = False
                self.controller.model_state[node_id] = False
                self.controller.collection_campaign.pop(node_id, None)
                self.controller.collection_run_counter.pop(node_id, None)
                self.controller.last_split_choice.pop(node_id, None)
                self.controller.expected_calib.pop(node_id, None)
                self.controller.expected_session.pop(node_id, None)
                self.controller.auto_training_hold_logged[node_id] = False
                self.controller._dataset_summary_cache.invalidate(node_id)
                self.controller._clear_node_heartbeat_state(node_id)
                self.controller._notify_model_reset(node_id)
                self.controller._log('Action', f'Reset entire rack {node_id} (including model artifacts)')
            else:
                return
        except Exception as e:
            self.controller._log('Action', f'Failed to reset {node_id}: {e}')
            return

        self.controller._save_calib_states()
        self.controller._save_model_state()
        if "MainDashboard" in self.controller.frames:
            subs = self.controller.frames["MainDashboard"].sub_frames
            if "Logs" in subs:
                subs["Logs"].refresh()
        if self.current_node_id:
            self._update_right_panel(self.current_node_id, refresh_table=True)

class GraphsView(ctk.CTkFrame):
    name = "Graphs"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="Saved Graphs", font=("Arial", 40, "bold")).pack(pady=5)
        self.scroll = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.scroll.pack(fill="both", expand=True, padx=20, pady=20)
        self.refresh()

    def refresh(self):
        for w in self.scroll.winfo_children():
            w.destroy()
        configured = getattr(self.controller, 'csi_data_dir', None)
        if isinstance(configured, str) and configured.strip():
            base = configured
        else:
            base = CSI_DATA_DIR
        if not os.path.exists(base):
            ctk.CTkLabel(self.scroll, text="No CSI data directory found.", font=("Arial",16)).pack(pady=20)
            return
        for node in sorted(os.listdir(base)):
            node_dir = os.path.join(base, node)
            if not os.path.isdir(node_dir):
                continue
            ctk.CTkLabel(self.scroll, text=node, font=("Arial",18,"bold")).pack(anchor="w", pady=(10,2))
            for state in sorted(os.listdir(node_dir)):
                state_dir = os.path.join(node_dir, state)
                if not os.path.isdir(state_dir):
                    continue
                ctk.CTkLabel(self.scroll, text="  "+state, font=("Arial",16)).pack(anchor="w", padx=10)
                files = sorted(
                    path for path in glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True)
                    if os.path.isfile(path) and not os.path.basename(path).startswith("part_")
                )
                for f in files:
                    b = ctk.CTkButton(self.scroll, text=os.path.basename(f), anchor="w",
                                      command=lambda f=f: self.plot_file(f))
                    b.pack(fill="x", padx=30, pady=2)

    def plot_file(self, path):
        df = pd.read_csv(path)
        subcarrier_cols = [col for col in df.columns if col.startswith("SC_")]
        if not subcarrier_cols:
            subcarrier_cols = df.select_dtypes(include="number").columns.tolist()
        if not subcarrier_cols:
            return
        mean_series = df[subcarrier_cols].mean(axis=1)
        # import matplotlib lazily here too
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(mean_series, label=os.path.basename(path))
        plt.title(f"CSI Mean Signal {os.path.basename(path)}")
        plt.xlabel("Sample Index")
        plt.ylabel("Mean Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

class SettingsView(ctk.CTkFrame):
    name = "Settings"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="Configuration", font=("Arial", 40, "bold")).pack(pady=10)
        frm = ctk.CTkFrame(self, fg_color="transparent")
        frm.pack(padx=30, pady=20)
        # Broker
        ctk.CTkLabel(frm, text="MQTT Broker:", font=("Arial", 16)).grid(row=0, column=0, sticky="w")
        self.broker_entry = ctk.CTkEntry(frm, width=300)
        self.broker_entry.grid(row=0, column=1, pady=5)
        # Port
        ctk.CTkLabel(frm, text="MQTT Port:", font=("Arial", 16)).grid(row=1, column=0, sticky="w")
        self.port_entry = ctk.CTkEntry(frm, width=100)
        self.port_entry.grid(row=1, column=1, pady=5)
        # CSI dir
        ctk.CTkLabel(frm, text="CSI Data Dir:", font=("Arial", 16)).grid(row=2, column=0, sticky="w")
        self.csi_entry = ctk.CTkEntry(frm, width=300)
        self.csi_entry.grid(row=2, column=1, pady=5)
        # load current values
        self._load_values()
        ctk.CTkButton(self, text="Save", fg_color="#3b8ed0", command=self.save).pack(pady=10)

    def _load_values(self):
        conf = self.controller.config
        self.broker_entry.delete(0, "end")
        self.broker_entry.insert(0, conf.get("broker", ""))
        self.port_entry.delete(0, "end")
        self.port_entry.insert(0, str(conf.get("port", "")))
        self.csi_entry.delete(0, "end")
        self.csi_entry.insert(0, conf.get("csi_data_dir", ""))

    def save(self):
        self.controller.config["broker"] = self.broker_entry.get()
        try:
            self.controller.config["port"] = int(self.port_entry.get())
        except ValueError:
            pass
        self.controller.config["csi_data_dir"] = self.csi_entry.get()
        self.controller.csi_data_dir = self.controller.config["csi_data_dir"]
        self.controller._save_config()
        # restart MQTT client with new settings
        try:
            self.controller.client.disconnect()
            self.controller.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.controller.client.on_connect = self.controller.on_connect
            self.controller.client.on_disconnect = self.controller.on_disconnect
            self.controller.client.on_message = self.controller.on_message
            self.controller.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.controller.client.connect_async(self.controller.config.get("broker"), self.controller.config.get("port"), 60)
            self.controller.client.loop_start()
        except Exception as e:
            print(f"WARNING: failed to restart mqtt with new config: {e}")
        # note: ingest server is now decoupled; make sure to restart it manually
        popup = ctk.CTkToplevel(self)
        popup.title("Settings Saved")
        popup.geometry("300x120")
        ctk.CTkLabel(popup, text="Configuration saved.").pack(pady=20)
        ctk.CTkButton(popup, text="OK", command=popup.destroy).pack(pady=8)

class LogsView(ctk.CTkFrame):
    name = "Logs"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="Logs", font=("Arial", 40, "bold")).pack(pady=5)
        # Search Bar for Logs
        self.search = ctk.CTkEntry(self, placeholder_text="Search Bar", height=40)
        self.search.pack(fill="x", padx=30, pady=10)
        self.box = ctk.CTkTextbox(self, corner_radius=30, border_width=1, border_color="black")
        self.box.pack(fill="both", expand=True, padx=30, pady=10)

    def refresh(self):
        self.box.delete("1.0", "end")
        for ts, top, msg in self.controller.all_logs:
            self.box.insert("end", f"[{ts}] {top}: {msg}\n")
        # Keep the latest log visible at the bottom instead of jumping to the top.
        try:
            self.box.see("end")
        except Exception:
            # customtkinter text widget should support see(); if not, ignore.
            pass

if __name__ == "__main__":
    app = GovernanceApp()
    # the ingest server is now started separately; run `python server.py` in
    # another terminal before launching the dashboard if you want HTTP/MQTT
    # services available.  the dashboard itself will still function for
    # key governance and local status displays.
    app.mainloop()
