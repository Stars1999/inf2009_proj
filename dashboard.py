import customtkinter as ctk
import paho.mqtt.client as mqtt
import os
import json
import glob
import time
import shutil
import subprocess
import sys
import pandas as pd
import traceback
# matplotlib is only needed for ad‑hoc plotting; import lazily inside
# the plotting helper functions so that any backend initialization happens
# after the main Tk root is up.  This avoids sporadic segfaults on some
# Windows/Tk builds.
from datetime import datetime

import threading

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

# --- Global Config ---
BROKER = "localhost"
PORT = 1883
LOG_DIR = "mqtt_logs"
CALIB_STATES = ["door_closed", "door_open", "person_standing"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# default to local server output path; can still be overridden in Settings
CSI_DATA_DIR = os.path.join(BASE_DIR, "csi_data")
TRAINING_SCRIPT = os.path.join(BASE_DIR, "train_model.py")
NOTEBOOK_TRAINING_FILE = os.path.join(BASE_DIR, "Edge_ML.ipynb")
NOTEBOOK_MODEL_FILE = os.path.join(BASE_DIR, "model (1).tflite")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

# legacy behaviour: fall back to local directory if CSI_DATA_DIR doesn’t exist

CONFIG_FILE = "config.json"  # persists broker/port/csi directory

# filenames for various saved state
CALIB_FILE = "calib_states.json"           # per-node completed flags
KEYS_FILE = "keys.json"                    # issued/in vault status
VIEW_FILE = "view_state.json"              # last page/node
MODEL_FILE = "models.json"                # per-node model validity flag
ONLINE_TTL_SECONDS = 60


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.makedirs(MODEL_STORE_DIR, exist_ok=True)

class GovernanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Zero-Trust Physical Key Governance")
        self.geometry("1100x750")
        
        # State Management
        # Keys 1-24: Key 2 is "out"
        self.keys = {f"Key {i}": "in" for i in range(1, 25)}
        self.all_logs = []
        # By default every node has no completed calibrations
        self.esp_nodes = {
            f"RACK_{i}": {state: False for state in CALIB_STATES}
            for i in range(1, 5)
        }
        # keep track of when we last heard from each node (for health)
        self.node_last_seen = {}
        self.node_online = {node_id: False for node_id in self.esp_nodes}
        # store the last radio selection per node so the UI remembers it
        self.last_calib_choice = {}
        # remember which label we requested for each node so we can validate
        self.expected_calib = {}
        # remember the active collection session per node for completion matching
        self.expected_session = {}
        # thresholds removed; model now handles inference
        # model lifecycle per-node (True if latest model is trained/valid)
        self.model_state = {node_id: False for node_id in self.esp_nodes}
        # prevent duplicate concurrent training triggers per node
        self.training_in_progress = {node_id: False for node_id in self.esp_nodes}
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

    def on_connect(self, client, userdata, flags, reason_code, properties):
        try:
            client.subscribe("#", qos=1)
            ts = datetime.now().strftime("%H:%M:%S")
            self.all_logs.append((ts, "MQTT", f"Connected (reason_code={reason_code})"))
        except Exception as e:
            print(f"WARNING: on_connect error: {e}")

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        ts = datetime.now().strftime("%H:%M:%S")
        self.all_logs.append((ts, "MQTT", f"Disconnected (reason_code={reason_code})"))
        # mark nodes offline until fresh status messages arrive
        for node_id in self.node_online:
            self.node_online[node_id] = False

    def _publish_collect_with_retry(self, node_id, label, retries=3):
        topic = f"/commands/{node_id}/collect"
        # generate a simple session identifier (timestamp-based)
        session = datetime.now().strftime("%Y%m%d%H%M%S")
        payload = json.dumps({"label": label, "session": session})
        for attempt in range(1, retries + 1):
            msg_info = self.client.publish(topic, payload, qos=1)
            if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
                # only set expected values once publish actually succeeds
                self.expected_calib[node_id] = label
                self.expected_session[node_id] = session
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
        try:
            shutil.copy2(model_path, model_dst)
            shutil.copy2(scaler_path, scaler_dst)
            return True
        except Exception as e:
            ts = datetime.now().strftime('%H:%M:%S')
            self.all_logs.append((ts, "Model", f"Artifact copy failed for {node_id}: {e}"))
            return False

    def _notify_training_complete(self, node_id, session):
        """Publish a training-complete command so the ESP can begin downloading."""
        topic = f"/commands/{node_id}/training_complete"
        payload = json.dumps({"session": session})
        msg_info = self.client.publish(topic, payload, qos=1)
        ts = datetime.now().strftime('%H:%M:%S')
        if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.all_logs.append((ts, "Model", f"Notified {node_id} training_complete (session={session})"))
            return True
        else:
            self.all_logs.append((ts, "Model", f"Failed to publish training_complete for {node_id}"))
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
        ts = datetime.now().strftime('%H:%M:%S')
        if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.all_logs.append((ts, "Model", f"Notified {node_id} to pull model (session={session})"))
            return True
        self.all_logs.append((ts, "Model", f"Failed to notify {node_id} for model update"))
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
        # when the main window is resized, force every view to re-layout
        # itself.  older behaviour only touched the currently visible frame,
        # which meant components in other tabs never updated and could appear
        # clipped when the user returned to them.
        for frame in self.frames.values():
            if hasattr(frame, "refresh"):
                try:
                    frame.refresh()
                except Exception:
                    pass
        # ensure the geometry manager repositions everything
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

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="replace")
        ts = datetime.now().strftime("%H:%M:%S")
        now_epoch = time.time()
        self.all_logs.append((ts, msg.topic, payload))

        completion_node = None
        completion_state = None
        completion_session = None

        topic_parts = msg.topic.strip("/").split("/")
        if len(topic_parts) == 3 and topic_parts[0] == "sensors" and topic_parts[2] == "status":
            node_id = topic_parts[1]
            # update health timestamp for this node
            self.node_last_seen[node_id] = now_epoch
            self.node_online[node_id] = True
            try:
                payload_json = json.loads(payload)
                evt = payload_json.get("event")
                if evt == "node_offline":
                    self.node_online[node_id] = False
                elif evt == "node_online":
                    self.node_online[node_id] = True
                elif evt == "collection_progress":
                    label = payload_json.get("label", "?")
                    cur = payload_json.get("sub_batch_idx", "?")
                    total = payload_json.get("total_sub_batches", "?")
                    pct = payload_json.get("percent", "?")
                    self.all_logs.append((ts, "Progress", f"{node_id} {label}: {cur}/{total} ({pct}%)"))
                    # Update progress bar in the configuration view if visible
                    if self.frames.get("MainDashboard"):
                        subs = self.frames["MainDashboard"].sub_frames
                        cfg = subs.get("ESP32-C3 Configuration")
                        if cfg and cfg.current_node_id == node_id:
                            try:
                                ci = int(cur)
                                ti = int(total)
                                cfg.set_progress(ci, ti)
                                # if we've just reached the end, re-enable the start button
                                if ci == ti and cfg.status_lbl_ref:
                                    cfg.status_lbl_ref.configure(text="Status: Upload finished; awaiting server merge", text_color="#1a4d66")
                            except Exception:
                                pass
                elif evt == "ack":
                    cmd = payload_json.get("cmd", "?")
                    self.all_logs.append((ts, "ACK", f"{node_id} acknowledged {cmd}"))
                elif evt == "identify_confirmed":
                    assigned = payload_json.get("name", node_id)
                    self.all_logs.append((ts, "Identify", f"{node_id} confirmed name {assigned}"))
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
                                    cfg.set_progress(si, ti)
                                    # fallback unlock path: allow next calibration when
                                    # firmware confirms all uploads were sent, even if
                                    # server merge-complete MQTT is delayed/missing.
                                    if si >= ti:
                                        if cfg.start_btn_ref:
                                            cfg.start_btn_ref.configure(state="normal")
                                        if cfg.status_lbl_ref:
                                            cfg.status_lbl_ref.configure(
                                                text="Status: Upload complete (server merge pending)",
                                                text_color="#1a4d66"
                                            )
                                except Exception:
                                    pass
                elif evt == "state_change":
                    state = payload_json.get("state")
                    if isinstance(state, str):
                        self.node_detected_state[node_id] = state
                        self.all_logs.append((ts, "State", f"{node_id} -> {state}"))
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
                    self.all_logs.append((ts, "Model", f"{node_id} reported model_ready"))
                elif evt == "model_download_failed":
                    self.model_state[node_id] = False
                    self._save_model_state()
                    self.all_logs.append((ts, "Model", f"{node_id} model download failed"))
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
            self.all_logs.append((ts, "Key", f"{payload} {self.keys[payload]}"))

        # UI Refresh (thread-safe)
        def refresh_ui():
            if "MainDashboard" in self.frames:
                subs = self.frames["MainDashboard"].sub_frames
                if "Homepage" in subs:
                    subs["Homepage"].refresh()
                if "Logs" in subs:
                    subs["Logs"].refresh()
                if "Key Management" in subs:
                    subs["Key Management"].refresh()
                if "ESP32-C3 Configuration" in subs:
                    subs["ESP32-C3 Configuration"].refresh()
                if completion_node and completion_state and "ESP32-C3 Configuration" in subs:
                    try:
                        subs["ESP32-C3 Configuration"].on_calibration_complete(completion_node, completion_state, completion_session)
                    except Exception as e:
                        print("[DASHBOARD] on_calibration_complete exception")
                        traceback.print_exc()
                        self.all_logs.append((datetime.now().strftime("%H:%M:%S"), "Error", f"calibration callback failed: {e}"))

        self.after(0, refresh_ui)

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
            cfg = self.sub_frames.get("ESP32-C3 Configuration")
            if cfg:
                cfg._refresh_node_status_list()
        except Exception:
            pass
        self.after(2000, self._periodic_status_refresh)

    def switch_view(self, name):
        # logout gets special handling regardless of current view
        if name == "Logout":
            self.controller.show_frame("LoginPage")
            return

        # update button highlights and raise requested frame
        for n, b in self.nav_btns.items():
            b.configure(fg_color="#d9d9d9" if n == name else "white")
        if name in self.sub_frames:
            self.sub_frames[name].tkraise()
            if hasattr(self.sub_frames[name], "refresh"):
                try:
                    self.sub_frames[name].refresh()
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
        self.refresh()

    def refresh(self):
        # Clear old stats
        for widget in self.grid_container.winfo_children():
            widget.destroy()

        # Calculate dynamic counts
        total_keys = len(self.controller.keys)
        issued_keys = sum(1 for status in self.controller.keys.values() if status == "out")
        vault_keys = sum(1 for status in self.controller.keys.values() if status == "in")
        # estimate online nodes: seen within last ONLINE_TTL_SECONDS
        now = time.time()
        online = 0
        for node_id in self.controller.esp_nodes:
            seen = self.controller.node_last_seen.get(node_id, 0)
            is_recent = (now - seen) < ONLINE_TTL_SECONDS
            if self.controller.node_online.get(node_id, False) and is_recent:
                online += 1
            elif not is_recent:
                self.controller.node_online[node_id] = False
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

    def refresh(self):
        for w in self.scroll.winfo_children(): w.destroy()
        for i, (name, status) in enumerate(self.controller.keys.items()):
            color = "#7ed957" if status == "in" else "#ff3131" # Green if in vault, red if issued
            ctk.CTkButton(self.scroll, text=name, fg_color=color, text_color="black", border_width=1, 
                          width=100, height=45, hover_color=color).grid(row=i//6, column=i%6, padx=10, pady=10)

    def add(self):
        n = self.ent.get()
        if n:
            self.controller.keys[n] = "in"
            self.controller._save_keys()
            self.refresh()
    def rem(self):
        n = self.ent.get()
        if n in self.controller.keys:
            del self.controller.keys[n]
            self.controller._save_keys()
            self.refresh()


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

            status_dot = ctk.CTkLabel(row, text="●", text_color="gray", width=20)
            status_dot.pack(side="right", padx=(6, 0))
            self.node_status_labels[node_id] = status_dot

        bulk_box = ctk.CTkFrame(left, fg_color="transparent")
        bulk_box.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(bulk_box, text="Bulk Label", font=("Arial", 13, "bold")).pack(anchor="w", pady=(6, 4))
        for state in CALIB_STATES:
            ctk.CTkRadioButton(bulk_box, text=state, variable=self.bulk_label_var, value=state).pack(anchor="w")
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
        # window‑level resize refresh is handled by the app's _on_resize
        # (no need to bind individual Configure events here)
        
        self.detail_widgets = []
        self.calibration_table_widgets = []
        self.current_node_id = None
        self.radio_var = ctk.StringVar(value="door_closed")
        self.start_btn_ref = None
        self.status_lbl_ref = None
        self.train_btn_ref = None
        self.model_status_lbl_ref = None
        # placeholder label lives inside scrollable frame
        self.placeholder = ctk.CTkLabel(self.right, text="Select a node to view details", text_color="gray")
        # use pack instead of place so it scrolls naturally
        self.placeholder.pack(pady=20)

    def refresh(self):
        # when the frame is raised, make sure the table reflects the current
        # saved state for whatever node is selected
        self._refresh_node_status_list()
        if self.current_node_id:
            self._render_calibration_table(self.current_node_id)

    def _refresh_node_status_list(self):
        now = time.time()
        any_key_out = any(v == "out" for v in self.controller.keys.values())
        for node_id, dot in self.node_status_labels.items():
            seen = self.controller.node_last_seen.get(node_id, 0)
            online = self.controller.node_online.get(node_id, False) and ((now - seen) < ONLINE_TTL_SECONDS)
            state = self.controller.node_detected_state.get(node_id)
            state_lbl = self.node_state_labels.get(node_id)
            if state == "door_closed":
                dot.configure(text_color="#2e7d32")
                if state_lbl:
                    state_lbl.configure(text="door_closed", text_color="#2e7d32")
            elif state == "door_open":
                if any_key_out:
                    dot.configure(text_color="#ff9800")
                    if state_lbl:
                        state_lbl.configure(text="door_open (authorized)", text_color="#ff9800")
                else:
                    dot.configure(text_color="#d32f2f")
                    if state_lbl:
                        state_lbl.configure(text="door_open (unauthorized)", text_color="#d32f2f")
            elif state == "person_standing":
                dot.configure(text_color="#ff9800")
                if state_lbl:
                    state_lbl.configure(text="person_standing", text_color="#ff9800")
            else:
                dot.configure(text_color="#2e7d32" if online else "gray")
                if state_lbl:
                    state_lbl.configure(text="online" if online else "offline", text_color="#2e7d32" if online else "gray")

    def show_details(self, node_id):
        self.current_node_id = node_id
        # remember selection for next launch
        self.controller.last_node = node_id
        self.controller._save_view_state()
        self.placeholder.pack_forget()
        for w in self.detail_widgets: w.destroy()
        self.detail_widgets = []
        self.calibration_table_widgets = []
        self.start_btn_ref = None
        self.status_lbl_ref = None
        self.train_btn_ref = None
        self.model_status_lbl_ref = None
        # restore last radio choice for this node if available
        default_choice = self.controller.last_calib_choice.get(node_id, "door_closed")
        self.radio_var = ctk.StringVar(value=default_choice)

        # Hardcoded static values
        fields = [
            ("Node Number:", node_id),
            ("Connection:", "UP" if self.controller.node_online.get(node_id, False) else "DOWN"),
            ("Door Status:", "CLOSE")
        ]

        for lbl, val in fields:
            f = ctk.CTkFrame(self.right, fg_color="transparent")
            f.pack(fill="x", padx=30, pady=8)
            self.detail_widgets.append(f)
            ctk.CTkLabel(f, text=lbl, text_color="black", font=("Arial", 16)).pack(side="left")
            ctk.CTkLabel(f, text=val, text_color="gray").pack(side="right")

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

        self._render_calibration_table(node_id)
        try:
            self.right.update_idletasks()
        except Exception:
            pass

        btn_f = ctk.CTkFrame(self.right, fg_color="transparent")
        btn_f.pack(pady=20); self.detail_widgets.append(btn_f)
        ctk.CTkButton(btn_f, text="Reset Calibrations", width=140, fg_color="#ff5555", text_color="white",
                      command=self._reset_calibrations).pack(side="left", padx=10)

    def set_progress(self, cur, total):
        # initialize progress bar if missing
        if not hasattr(self, 'progress_bar'):
            self.progress_bar = ctk.CTkProgressBar(self.right, width=300)
            self.progress_bar.pack(pady=(4,10))
        self.progress_bar.set(cur/total)
        if self.status_lbl_ref:
            self.status_lbl_ref.configure(text=f"Status: {cur}/{total} sub-batches uploaded")

    def _render_calibration_table(self, node_id):
        for w in self.calibration_table_widgets:
            if w in self.detail_widgets:
                self.detail_widgets.remove(w)
            w.destroy()
        self.calibration_table_widgets = []

        title = ctk.CTkLabel(self.right, text="Last Calibrated States", text_color="black", font=("Arial", 16, "bold"))
        title.pack(anchor="w", padx=30, pady=(2, 6))
        self.calibration_table_widgets.append(title)
        self.detail_widgets.append(title)

        # ensure the map exists (in case we added a node later)
        state_map = self.controller.esp_nodes.setdefault(node_id, {s: False for s in CALIB_STATES})
        for state in CALIB_STATES:
            row = ctk.CTkFrame(self.right, fg_color="transparent")
            row.pack(fill="x", padx=30, pady=2)
            self.calibration_table_widgets.append(row)
            self.detail_widgets.append(row)

            ctk.CTkLabel(row, text=state, text_color="black", font=("Arial", 14)).pack(side="left")
            done = state_map.get(state, False)
            status_text = "✓ Done" if done else "○ Pending"
            status_color = "#2e7d32" if done else "gray"
            ctk.CTkLabel(row, text=status_text, text_color=status_color, font=("Arial", 14)).pack(side="right")

        all_done = all(state_map.get(s, False) for s in CALIB_STATES)
        if all_done:
            trained = self.controller.model_state.get(node_id, False)
            status_txt = "Model: Ready" if trained else "Model: Invalid (retrain required)"
            status_col = "#2e7d32" if trained else "#d32f2f"
            self.model_status_lbl_ref = ctk.CTkLabel(self.right, text=status_txt, text_color=status_col, font=("Arial", 13, "bold"))
            self.model_status_lbl_ref.pack(anchor="w", padx=30, pady=(8, 4))
            self.calibration_table_widgets.append(self.model_status_lbl_ref)
            self.detail_widgets.append(self.model_status_lbl_ref)

            self.train_btn_ref = ctk.CTkButton(
                self.right,
                text="Retrain Model" if trained else "Train Model",
                fg_color="#3b8ed0",
                command=lambda n=node_id: self.train_model(n)
            )
            self.train_btn_ref.pack(anchor="w", padx=30, pady=(2, 10))
            self.calibration_table_widgets.append(self.train_btn_ref)
            self.detail_widgets.append(self.train_btn_ref)

    def _resolve_data_base(self):
        configured = getattr(self.controller, "csi_data_dir", None)
        candidates = []
        if isinstance(configured, str) and configured.strip():
            candidates.append(configured)
        if isinstance(CSI_DATA_DIR, str) and CSI_DATA_DIR.strip():
            candidates.append(CSI_DATA_DIR)
        candidates.append("csi_data")

        for path in candidates:
            try:
                if os.path.exists(path):
                    return path
            except Exception:
                continue
        return candidates[-1]

    def train_model(self, node_id):
        state_map = self.controller.esp_nodes.get(node_id, {})
        if not all(state_map.get(s, False) for s in CALIB_STATES):
            popup = ctk.CTkToplevel(self)
            popup.title("Training blocked")
            popup.geometry("380x140")
            ctk.CTkLabel(popup, text="Complete all calibration states first.", font=("Arial", 14)).pack(pady=20)
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            return

        if self.controller.training_in_progress.get(node_id, False):
            ts = datetime.now().strftime('%H:%M:%S')
            self.controller.all_logs.append((ts, "Model", f"Training already in progress for {node_id}"))
            if self.status_lbl_ref:
                self.status_lbl_ref.configure(text="Status: Training already in progress", text_color="#1a4d66")
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
            "--notebook", NOTEBOOK_TRAINING_FILE,
            "--notebook-model", NOTEBOOK_MODEL_FILE,
        ]

        # disable the train button to prevent re‑entry
        if self.train_btn_ref:
            self.train_btn_ref.configure(state="disabled")

        if self.status_lbl_ref:
            self.status_lbl_ref.configure(text=f"Status: Training model for {node_id}...", text_color="#1a4d66")

        # show a simple progress popup while training runs
        progress_popup = ctk.CTkToplevel(self)
        progress_popup.title("Training in progress")
        progress_popup.geometry("320x100")
        ctk.CTkLabel(progress_popup, text="Training in progress, please wait...").pack(pady=20)

        # helper to close later
        training_popup_ref = progress_popup

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            proc = None
            err = str(e)
        else:
            err = proc.stderr.strip() if proc.returncode != 0 else ""

        # log stdout for additional visibility
        if proc and proc.stdout:
            for line in proc.stdout.splitlines():
                if line.strip():
                    ts = datetime.now().strftime('%H:%M:%S')
                    self.controller.all_logs.append((ts, "Model", line.strip()))
                    # also update status label with key info lines
                    if "Dataset shape" in line or "Test loss" in line:
                        if self.status_lbl_ref:
                            self.status_lbl_ref.configure(text=line.strip(), text_color="#1a4d66")

        # once subprocess finishes remove progress popup
        try:
            training_popup_ref.destroy()
        except Exception:
            pass

        if proc is None or proc.returncode != 0:
            # dump output to terminal for immediate debugging
            if proc:
                print("[TRAINING] stdout:\n", proc.stdout)
                print("[TRAINING] stderr:\n", proc.stderr)
                # also add every line to the dashboard log so Logs tab shows it
                for line in proc.stdout.splitlines():
                    ts = datetime.now().strftime('%H:%M:%S')
                    self.controller.all_logs.append((ts, "Model", line))
                for line in proc.stderr.splitlines():
                    ts = datetime.now().strftime('%H:%M:%S')
                    self.controller.all_logs.append((ts, "Model", line))
            ts = datetime.now().strftime('%H:%M:%S')
            self.controller.all_logs.append((ts, "Model", f"Training failed for {node_id}: {err or 'unknown error'}"))
            if self.status_lbl_ref:
                self.status_lbl_ref.configure(text="Status: Training failed", text_color="red")
            popup = ctk.CTkToplevel(self)
            popup.title("Training failed")
            popup.geometry("480x180")
            ctk.CTkLabel(popup, text="Model training failed.", font=("Arial", 16, "bold")).pack(pady=(16, 8))
            ctk.CTkLabel(popup, text=err or "Check logs for details.", wraplength=440, text_color="gray").pack(pady=(0, 12))
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            # re-enable button
            if self.train_btn_ref:
                self.train_btn_ref.configure(state="normal")
            self.controller.training_in_progress[node_id] = False
            return

        # training succeeded; copy artifacts to server and then notify ESP
        copied = self.controller._copy_model_to_server(node_id, model_output, scaler_output)
        if copied:
            # generate a session id for notification
            session = datetime.now().strftime("%Y%m%d%H%M%S")
            self.controller.model_state[node_id] = False  # will be set true when ESP reports model_ready
            self.controller._save_model_state()
            self.controller._notify_training_complete(node_id, session)
            if self.status_lbl_ref:
                self.status_lbl_ref.configure(text="Status: Model trained; waiting for device to pull", text_color="#1a4d66")
        else:
            ts = datetime.now().strftime('%H:%M:%S')
            self.controller.all_logs.append((ts, "Model", f"Failed to copy model to server for {node_id}"))
            if self.status_lbl_ref:
                self.status_lbl_ref.configure(text="Status: Model trained but copy failed", text_color="#d32f2f")
        # re-enable train button regardless of outcome
        if self.train_btn_ref:
            self.train_btn_ref.configure(state="normal")
        self.controller.training_in_progress[node_id] = False
        if self.current_node_id == node_id:
            self._render_calibration_table(node_id)

    def start_calibration(self, node_id):
        selected_state = self.radio_var.get()
        # remember choice so UI can restore it later
        self.controller.last_calib_choice[node_id] = selected_state
        # any fresh calibration invalidates previously trained model
        self.controller._mark_model_dirty(node_id)

        # ensure a fresh dataset for each new calibration cycle
        data_base = self._resolve_data_base()
        state_dir = os.path.join(data_base, node_id, selected_state)
        if os.path.isdir(state_dir):
            try:
                shutil.rmtree(state_dir)
                ts = datetime.now().strftime('%H:%M:%S')
                self.controller.all_logs.append((ts, "Model", f"Cleared old CSI data for {node_id}/{selected_state}"))
            except Exception as e:
                ts = datetime.now().strftime('%H:%M:%S')
                self.controller.all_logs.append((ts, "Model", f"Failed to clear CSI data for {node_id}/{selected_state}: {e}"))

        # reset progress UI
        try:
            self.set_progress(0,1)
        except Exception:
            pass
        # attempt to publish command with retries
        ok = self.controller._publish_collect_with_retry(node_id, selected_state, retries=3)
        if not ok:
            # publication failed locally; inform user and leave button enabled
            if self.status_lbl_ref:
                self.status_lbl_ref.configure(
                    text="Status: Publish failed",
                    text_color="red"
                )
                # revert to idle after a moment
                def restore_idle():
                    if self.status_lbl_ref:
                        self.status_lbl_ref.configure(text="Status: Idle", text_color="gray")
                self.after(3000, restore_idle)
            return

        # disable button only after we know the message was sent successfully
        if self.start_btn_ref:
            self.start_btn_ref.configure(state="disabled")

        if self.status_lbl_ref:
            self.status_lbl_ref.configure(
                text=f"Status: Calibrating {node_id} for {selected_state}... Please wait (~20s)",
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
        success_count = 0
        failed_nodes = []
        for node_id in selected_nodes:
            self.controller.last_calib_choice[node_id] = selected_state
            self.controller._mark_model_dirty(node_id)

            data_base = self._resolve_data_base()
            state_dir = os.path.join(data_base, node_id, selected_state)
            if os.path.isdir(state_dir):
                try:
                    shutil.rmtree(state_dir)
                    ts = datetime.now().strftime('%H:%M:%S')
                    self.controller.all_logs.append((ts, "Model", f"Cleared old CSI data for {node_id}/{selected_state}"))
                except Exception as e:
                    ts = datetime.now().strftime('%H:%M:%S')
                    self.controller.all_logs.append((ts, "Model", f"Failed to clear CSI data for {node_id}/{selected_state}: {e}"))

            if self.controller._publish_collect_with_retry(node_id, selected_state, retries=3):
                success_count += 1
            else:
                failed_nodes.append(node_id)

        ts = datetime.now().strftime('%H:%M:%S')
        if failed_nodes:
            self.controller.all_logs.append((
                ts,
                "Bulk",
                f"Started {selected_state} for {success_count}/{len(selected_nodes)} racks; failed: {', '.join(failed_nodes)}"
            ))
        else:
            self.controller.all_logs.append((ts, "Bulk", f"Started {selected_state} for {success_count}/{len(selected_nodes)} racks"))

        popup = ctk.CTkToplevel(self)
        popup.title("Bulk calibration")
        popup.geometry("420x170")
        msg = f"Started {selected_state} on {success_count}/{len(selected_nodes)} selected racks."
        if failed_nodes:
            msg += f"\nFailed: {', '.join(failed_nodes)}"
        ctk.CTkLabel(popup, text=msg, font=("Arial", 13), justify="left").pack(pady=24)
        ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()

    def on_calibration_complete(self, node_id, state, session=None):
        expected_session = self.controller.expected_session.get(node_id)
        # if we previously sent a command, verify the returned label
        expected = self.controller.expected_calib.get(node_id)
        if expected is not None and expected != state:
            # mismatch occurred; warn and let user retry explicitly
            print(f"WARNING: node {node_id} reported '{state}' but expected '{expected}'")
            if self.current_node_id == node_id:
                # still on the same node, show a warning popup
                self._show_mismatch_popup(node_id, state, expected)
                # recover UI so user can retry
                if self.start_btn_ref:
                    self.start_btn_ref.configure(state="normal")
                if self.status_lbl_ref:
                    self.status_lbl_ref.configure(text="Status: Idle", text_color="gray")
            # clear expected values; require explicit retry from user
            self.controller.expected_calib.pop(node_id, None)
            self.controller.expected_session.pop(node_id, None)
            return

        # validate completion session against expected session
        if expected_session is not None and session and session != expected_session:
            ts = datetime.now().strftime('%H:%M:%S')
            self.controller.all_logs.append((
                ts,
                "Mismatch",
                f"{node_id} session mismatch: got {session}, expected {expected_session}; event ignored"
            ))
            print(f"WARNING: node {node_id} reported session '{session}' but expected '{expected_session}'")
            if self.current_node_id == node_id:
                if self.start_btn_ref:
                    self.start_btn_ref.configure(state="normal")
                if self.status_lbl_ref:
                    self.status_lbl_ref.configure(text="Status: Idle", text_color="gray")
            self.controller.expected_calib.pop(node_id, None)
            self.controller.expected_session.pop(node_id, None)
            return

        # ignore no-session fallback completions when we have no active expected session
        if expected_session is None and not session:
            ts = datetime.now().strftime('%H:%M:%S')
            self.controller.all_logs.append((
                ts,
                "Model",
                f"Ignored unsolicited firmware fallback completion for {node_id}/{state} (no active expected session)"
            ))
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

        ts = datetime.now().strftime('%H:%M:%S')
        if server_complete:
            self.controller.all_logs.append((
                ts,
                "Model",
                f"Server merge complete for {node_id}/{state} (session={completion_session})"
            ))
        else:
            fallback_desc = "firmware fallback" if completion_source == "firmware" else "no server session"
            self.controller.all_logs.append((
                ts,
                "Model",
                f"Firmware completion for {node_id}/{state} ({fallback_desc})"
            ))

        all_done = all(self.controller.esp_nodes[node_id].get(s, False) for s in CALIB_STATES)
        if all_done and not self.controller.model_state.get(node_id, False) and not self.controller.training_in_progress.get(node_id, False):
            self.controller.all_logs.append((ts, "Model", f"Auto-training triggered for {node_id}"))
            if self.current_node_id == node_id and self.status_lbl_ref:
                self.status_lbl_ref.configure(text=f"Status: Auto-training model for {node_id}...", text_color="#1a4d66")
            popup = ctk.CTkToplevel(self)
            popup.title("Auto Training")
            popup.geometry("420x140")
            ctk.CTkLabel(
                popup,
                text=f"All calibration states complete for {node_id}.\nStarting model training...",
                font=("Arial", 14)
            ).pack(pady=22)
            ctk.CTkButton(popup, text="OK", width=90, command=popup.destroy).pack()
            self.train_model(node_id)
        elif all_done and self.controller.training_in_progress.get(node_id, False):
            self.controller.all_logs.append((ts, "Model", f"Training already running for {node_id}; skipping duplicate trigger"))

        # remove the stored expectations (if any)
        self.controller.expected_calib.pop(node_id, None)
        self.controller.expected_session.pop(node_id, None)

        # if user has navigated away, no further UI changes needed
        if self.current_node_id != node_id:
            return

        # unlock next calibration on either authoritative server complete or
        # firmware fallback completion.
        if self.start_btn_ref:
            self.start_btn_ref.configure(state="normal")
        if self.status_lbl_ref:
            if server_complete:
                self.status_lbl_ref.configure(text="Status: Idle", text_color="gray")
            else:
                self.status_lbl_ref.configure(text="Status: Idle (firmware fallback)", text_color="#1a4d66")

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
        ts = datetime.now().strftime('%H:%M:%S')
        self.controller.all_logs.append((ts, 'Mismatch', f'{node_id} reported {received} (expected {expected})'))
        # update log panel if visible
        if "MainDashboard" in self.controller.frames:
            subs = self.controller.frames["MainDashboard"].sub_frames
            if "Logs" in subs:
                subs["Logs"].refresh()

    def _reset_calibrations(self):
        # clear all esp_nodes flags and delete persisted file
        for node in self.controller.esp_nodes:
            for state in self.controller.esp_nodes[node]:
                self.controller.esp_nodes[node][state] = False
            self.controller.model_state[node] = False
        try:
            if os.path.exists(CALIB_FILE):
                os.remove(CALIB_FILE)
        except Exception:
            pass
        self.controller._save_model_state()
        ts = datetime.now().strftime('%H:%M:%S')
        self.controller.all_logs.append((ts, 'Action', 'Reset all calibrations'))
        if "MainDashboard" in self.controller.frames:
            subs = self.controller.frames["MainDashboard"].sub_frames
            if "Logs" in subs:
                subs["Logs"].refresh()
        if self.current_node_id:
            self._render_calibration_table(self.current_node_id)

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
                files = sorted(glob.glob(os.path.join(state_dir, "*.csv")))
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

if __name__ == "__main__":
    app = GovernanceApp()
    # the ingest server is now started separately; run `python server.py` in
    # another terminal before launching the dashboard if you want HTTP/MQTT
    # services available.  the dashboard itself will still function for
    # key governance and local status displays.
    app.mainloop()
