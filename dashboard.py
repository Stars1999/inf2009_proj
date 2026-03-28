import os
import sys
import json
import glob
import time
import threading
import traceback
import subprocess
import queue
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QStackedWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QTextEdit,
    QCheckBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QProgressBar,
    QFormLayout,
    QSpinBox,
    QFrame,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
)

import pandas as pd
import paho.mqtt.client as mqtt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from shared_config import DEFAULT_MQTT_BROKER, DEFAULT_MQTT_PORT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSI_DATA_DIR = os.path.join(BASE_DIR, "csi_data")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

CALIB_STATES = ["door_closed", "door_open", "person_standing"]

CSI_METADATA_COLUMNS = {
    "node_id",
    "collection_label",
    "session_id",
    "sub_batch_idx",
    "subbatchidx",
    "campaign_id",
    "calibration_run_id",
    "collection_split",
    "payload_crc32",
    "idempotency_key",
    "upload_received_at",
    "global_sample_idx",
    "global_sampleidx",
    "globalsampleidx",
    "row_in_sub_batch",
    "rowinsubbatch",
    "row_in_subbatch",
    "timestamp",
    "time",
    "label",
}

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
CALIB_FILE = os.path.join(BASE_DIR, "calib_states.json")
KEYS_FILE = os.path.join(BASE_DIR, "keys.json")
VIEW_FILE = os.path.join(BASE_DIR, "view_state.json")
MODEL_FILE = os.path.join(BASE_DIR, "models.json")
HEARTBEAT_FILE = os.path.join(BASE_DIR, "heartbeat_state.json")

HEARTBEAT_TIMEOUT_S = 20
REFRESH_INTERVAL_MS = 1500

MODERN_STYLE = """
QMainWindow {
    background-color: #f5f6fa;
}

QWidget {
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    color: #333333;
}

/* Sidebar styling */
QFrame#SidebarFrame {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}

#SidebarFrame QPushButton {
    background-color: transparent;
    text-align: left;
    padding: 12px 20px;
    border: none;
    border-radius: 6px;
    margin: 4px 8px;
    font-weight: 500;
    color: #555555;
}

#SidebarFrame QPushButton:hover {
    background-color: #e3f2fd;
    color: #1976d2;
}

#SidebarFrame QPushButton:checked {
    background-color: #e3f2fd;
    color: #1976d2;
    font-weight: bold;
    border-left: 4px solid #1976d2;
    border-radius: 4px;
}

/* General Buttons */
QPushButton {
    background-color: #1976d2;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #0d47a1;
}

QPushButton:disabled {
    background-color: #b0bec5;
    color: #ffffff;
}

/* Special Buttons */
QPushButton#ActionBtn {
    background-color: #4caf50;
}
QPushButton#ActionBtn:hover {
    background-color: #43a047;
}
QPushButton#ActionBtn:disabled {
    background-color: #b0bec5;
    color: #ffffff;
}

QPushButton#DangerBtn {
    background-color: #f44336;
}
QPushButton#DangerBtn:hover {
    background-color: #e53935;
}
QPushButton#DangerBtn:disabled {
    background-color: #b0bec5;
    color: #ffffff;
}

/* Inputs */
QLineEdit, QComboBox, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px 10px;
    min-height: 24px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #1976d2;
}

QProgressBar {
    background-color: #e0e0e0;
    border-radius: 6px;
    text-align: center;
    color: #333333;
    font-weight: bold;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #4caf50;
    border-radius: 6px;
}

/* Tables */
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    gridline-color: #f0f0f0;
    selection-background-color: #e3f2fd;
    selection-color: #000000;
}

QHeaderView::section {
    background-color: #fafafa;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #e0e0e0;
    border-right: 1px solid #e0e0e0;
    font-weight: bold;
    text-align: left;
}

/* Text edits */
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 8px;
}

/* Page Titles */
QLabel#PageTitle {
    font-size: 26px;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e0e0e0;
}

/* Stat Cards */
QFrame#StatCard, QFrame.StatCard {
    background-color: #ffffff;
    border-radius: 8px;
    border: 2px solid #cfd8dc;
    padding: 16px;
}
QLabel.StatValue {
    font-size: 32px;
    font-weight: bold;
    color: #1976d2;
}
QLabel.StatLabel {
    font-size: 14px;
    color: #7f8c8d;
    text-transform: uppercase;
    font-weight: bold;
}
"""

class DashboardSignals(QObject):
    node_status = Signal(str, str)  # node_id, status
    key_status = Signal(str, str)  # key_id, status
    collection_progress = Signal(str, int, int)
    collection_complete = Signal(str, str, str)  # node_id, label, session
    model_event = Signal(str, dict)
    training_result = Signal(str, bool, str)
    error = Signal(str)


class MqttClient(threading.Thread):
    def __init__(self, state, signals):
        super().__init__(daemon=True)
        self.state = state
        self.signals = signals
        self.event_queue = queue.Queue()
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="DashboardClient",
        )
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._stop = threading.Event()
        self._has_logged_connected = False

    def _enqueue_event(self, event_type, *args):
        self.event_queue.put((event_type, *args))

    def run(self):
        host = self.state.config.get("broker", DEFAULT_MQTT_BROKER)
        port = int(self.state.config.get("port", DEFAULT_MQTT_PORT))
        try:
            self.client.connect(host, port, 60)
            self.client.loop_start()
            while not self._stop.is_set():
                time.sleep(0.3)
        except Exception as e:
            self._enqueue_event("error", f"MQTT start failed: {e}")

    def stop(self):
        self._stop.set()
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def on_connect(self, client, userdata, connect_flags, reason_code, properties=None):
        if reason_code == 0:
            client.subscribe("#", qos=1)
            client.subscribe("device/+/status", qos=1)
            client.subscribe("/sensors/+/status", qos=1)
            if not self._has_logged_connected:
                self._enqueue_event("error", "MQTT connected")
            self._has_logged_connected = True
        else:
            self._enqueue_event("error", f"MQTT connect fail {reason_code}")

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        if self._stop.is_set():
            return
        self._has_logged_connected = False
        if reason_code != 0:
            self._enqueue_event("error", f"MQTT disconnected (rc={reason_code})")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode(errors="ignore").strip()
            
            if topic.startswith("device/") and topic.endswith("/status"):
                node_id = topic.split("/")[1]
                status = payload.lower()
                self.state.node_online[node_id] = (status == "online")
                if status == "online":
                    self.state.node_last_heartbeat[node_id] = time.time()
                self._enqueue_event("node_status", node_id, status)
            
            elif topic.startswith("/sensors/") and topic.endswith("/status"):
                node_id = topic.split("/")[2]
                try:
                    body = json.loads(payload)
                except json.JSONDecodeError:
                    return
                event = body.get("event")
                if not event:
                    return
                if event == "collection_progress":
                    sub = int(body.get("sub_batch_idx", 0))
                    total = int(body.get("total_sub_batches", 1))
                    self._enqueue_event("collection_progress", node_id, sub, total)
                elif event == "upload":
                    sub = body.get("sub")
                    total = body.get("total")
                    if isinstance(sub, (int, float)) and isinstance(total, (int, float)):
                        self._enqueue_event("collection_progress", node_id, int(sub), int(total))
                elif event == "collection_complete":
                    label = body.get("label")
                    session = body.get("session")
                    if label in CALIB_STATES:
                        self.state.esp_nodes[node_id][label] = True
                        self._enqueue_event("collection_complete", node_id, label, session)
                elif event == "model_ready":
                    self.state.model_state[node_id] = True
                    self._enqueue_event("model_event", node_id, body)
                elif event == "model_download_failed" or event == "model_download_incompatible":
                    self.state.model_state[node_id] = False
                    self._enqueue_event("model_event", node_id, body)
                elif event == "model_cleared":
                    self.state.model_state[node_id] = False
                    self._enqueue_event("model_event", node_id, body)
                elif event == "heartbeat":
                    self.state.node_last_heartbeat[node_id] = time.time()
                elif event == "state_change":
                    state = body.get("state")
                    if isinstance(state, str):
                        self.state.node_detected_state[node_id] = state
                elif event == "ack":
                    cmd = body.get("cmd", "?")
                    self._enqueue_event("error", f"{node_id} acknowledged {cmd}")
                elif event == "identify_confirmed":
                    assigned = body.get("name", node_id)
                    self._enqueue_event("error", f"{node_id} confirmed name {assigned}")
            
            # Color Toggle Logic - matches legacy exactly
            if payload in self.state.keys:
                if topic == "Key Unlocked":
                    self.state.keys[payload] = "out"
                    self._enqueue_event("key_status", payload, "out")
                elif topic == "Key Returned":
                    self.state.keys[payload] = "in"
                    self._enqueue_event("key_status", payload, "in")
        except Exception as e:
            self._enqueue_event("error", f"MQTT message handling error: {e}")


class AppState:
    def __init__(self):
        self.keys = {f"Key {i}": "in" for i in range(1, 25)}
        self.esp_nodes = {f"RACK_{i}": {s: False for s in CALIB_STATES} for i in range(1, 5)}
        self.node_online = {n: False for n in self.esp_nodes}
        self.node_last_heartbeat = {n: 0.0 for n in self.esp_nodes}
        self.last_node = None
        self.last_view = "Homepage"
        self.model_state = {n: False for n in self.esp_nodes}
        self.node_detected_state = {n: "" for n in self.esp_nodes}
        self.expected_calib = {}
        self.expected_session = {}
        self.collection_campaign = {}
        self.collection_run_counter = {}
        self.collection_in_progress = {n: False for n in self.esp_nodes}
        self.collection_progress = {n: (0, 0) for n in self.esp_nodes}
        self.last_calib_choice = {}
        self.last_split_choice = {}
        self.training_in_progress = {n: False for n in self.esp_nodes}
        self.config = {"broker": DEFAULT_MQTT_BROKER, "port": DEFAULT_MQTT_PORT, "csi_data_dir": CSI_DATA_DIR}

    def save_json(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_json(self, path, default):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return default

    def load_all(self):
        self.keys.update(self.load_json(KEYS_FILE, {}))
        calib = self.load_json(CALIB_FILE, {})
        for k, v in calib.items():
            if k in self.esp_nodes and isinstance(v, dict):
                for s, val in v.items():
                    if s in CALIB_STATES and isinstance(val, bool):
                        self.esp_nodes[k][s] = val
        self.model_state.update(self.load_json(MODEL_FILE, {}))
        view = self.load_json(VIEW_FILE, {})
        if isinstance(view, dict):
            self.last_view = view.get("last_page", self.last_view)
            self.last_node = view.get("last_node", self.last_node)
        self.config.update(self.load_json(CONFIG_FILE, {}))
        # Keep CSI path usable on the current host (legacy configs may contain
        # stale absolute paths from another machine/OS).
        self.config["csi_data_dir"] = self.resolve_csi_data_dir()
        self.node_last_heartbeat.update(self.load_json(HEARTBEAT_FILE, {}))

    def persist(self):
        self.save_json(KEYS_FILE, self.keys)
        self.save_json(CALIB_FILE, self.esp_nodes)
        self.save_json(MODEL_FILE, self.model_state)
        self.save_json(VIEW_FILE, {"last_page": self.last_view, "last_node": self.last_node})
        self.save_json(CONFIG_FILE, self.config)
        self.save_json(HEARTBEAT_FILE, self.node_last_heartbeat)

    def discount_stale_heartbeat(self, node_id):
        last = self.node_last_heartbeat.get(node_id, 0)
        if not last:
            return False
        return (time.time() - last) > HEARTBEAT_TIMEOUT_S

    def resolve_csi_data_dir(self):
        configured = self.config.get("csi_data_dir", "")
        candidates = []
        if isinstance(configured, str) and configured.strip():
            candidates.append(configured.strip())
        candidates.extend([
            CSI_DATA_DIR,
            os.path.join(BASE_DIR, "csi_data"),
            "csi_data",
        ])

        seen = set()
        for path in candidates:
            if not isinstance(path, str) or not path.strip():
                continue
            try:
                abs_path = os.path.abspath(path)
            except Exception:
                continue
            if abs_path in seen:
                continue
            seen.add(abs_path)
            if os.path.isdir(abs_path):
                return abs_path

        fallback = os.path.abspath(CSI_DATA_DIR)
        os.makedirs(fallback, exist_ok=True)
        return fallback

    def node_data_dir(self, node):
        base = self.resolve_csi_data_dir()
        return os.path.join(base, node)

    def count_state_csv_files(self, node, state):
        """Count CSV files recursively under csi_data/<node>/<state>, excluding files starting with 'part_'."""
        node_dir = self.node_data_dir(node)
        state_dir = os.path.join(node_dir, state)
        if not os.path.isdir(state_dir):
            return 0
        csv_paths = sorted(glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True))
        files = [p for p in csv_paths if not os.path.basename(p).startswith("part_")]
        return len(files)

    def build_dataset_summary(self, node):
        node_dir = self.node_data_dir(node)
        if not os.path.isdir(node_dir):
            return None

        summary = {
            "node": node,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_files": 0,
            "total_rows": 0,
            "labels": {},
        }

        for state in CALIB_STATES:
            state_dir = os.path.join(node_dir, state)
            summary["labels"][state] = {
                "files": 0,
                "sessions": 0,
                "rows_raw": 0,
                "rows_clean": 0,
                "split_files": {"train": 0, "dev": 0},
                "split_rows": {"train": 0, "dev": 0},
            }
            if not os.path.isdir(state_dir):
                continue
            csv_paths = sorted(glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True))
            sessions = set()
            for csv_path in csv_paths:
                if os.path.basename(csv_path).startswith("part_"):
                    continue
                session_name = os.path.splitext(os.path.basename(csv_path))[0]
                sessions.add(session_name)
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                rows = len(df)
                label_summary = summary["labels"][state]
                label_summary["files"] += 1
                label_summary["rows_raw"] += rows
                label_summary["rows_clean"] += rows
                split_group = "train"
                if "collection_split" in df.columns:
                    split_group = str(df["collection_split"].mode().iloc[0]).lower() if not df["collection_split"].mode().empty else "train"
                elif "split_group" in df.columns:
                    split_group = str(df["split_group"].mode().iloc[0]).lower() if not df["split_group"].mode().empty else "train"
                elif "dataset_split" in df.columns:
                    split_group = str(df["dataset_split"].mode().iloc[0]).lower() if not df["dataset_split"].mode().empty else "train"
                # Map legacy 'test' -> 'dev' and default unknown splits to 'train'
                if split_group == "test":
                    split_group = "dev"
                if split_group not in ("train", "dev"):
                    split_group = "train"
                label_summary["split_files"][split_group] += 1
                label_summary["split_rows"][split_group] += rows

            summary["labels"][state]["sessions"] = len(sessions)
            summary["total_files"] += summary["labels"][state]["files"]
            summary["total_rows"] += summary["labels"][state]["rows_clean"]

        return summary


class DashboardMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zero-Trust Physical Key Governance")
        self.resize(1300, 820)

        self.state = AppState()
        self.state.load_all()

        self.signals = DashboardSignals()
        self.mqtt = MqttClient(self.state, self.signals)

        self.signals.node_status.connect(self.on_node_status)
        self.signals.key_status.connect(self.on_key_status)
        self.signals.collection_progress.connect(self.on_collection_progress)
        self.signals.collection_complete.connect(self.on_collection_complete)
        self.signals.model_event.connect(self.on_model_event)
        self.signals.training_result.connect(lambda node, success, msg: self.pages["ESP32-C3 Configuration"].on_training_result(node, success, msg))
        self.signals.error.connect(self.log_message)

        self.mqtt.start()

        self.setup_ui()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.periodic_refresh)
        self.update_timer.start(REFRESH_INTERVAL_MS)

    def setup_ui(self):
        self.setStyleSheet(MODERN_STYLE)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("SidebarFrame")
        sidebar_frame.setFixedWidth(260)
        sidebar = QVBoxLayout(sidebar_frame)
        sidebar.setContentsMargins(0, 20, 0, 20)
        sidebar.setSpacing(5)

        # Title/Logo area in sidebar
        logo_label = QLabel("Zero-Trust\nKey Governance")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1976d2; margin-bottom: 20px;")
        sidebar.addWidget(logo_label)

        self.buttons = {}
        items = ["Homepage", "Key Management", "ESP32-C3 Configuration", "Logs", "Graphs", "Settings"]
        for item in items:
            btn = QPushButton(item)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, i=item: self.switch_view(i))
            sidebar.addWidget(btn)
            self.buttons[item] = btn
        sidebar.addStretch(1)

        # Main Content Area
        main_content = QWidget()
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(30, 30, 30, 30)

        self.stack = QStackedWidget()
        self.pages = {}

        self.pages["Homepage"] = HomePage(self.state)
        self.pages["Key Management"] = KeyMgmtPage(self.state)
        self.pages["ESP32-C3 Configuration"] = ESPConfigPage(self.state, self)
        self.pages["Logs"] = LogsPage(self)
        self.pages["Graphs"] = GraphsPage(self.state)
        self.pages["Settings"] = SettingsPage(self.state, self)

        for p in items:
            self.stack.addWidget(self.pages[p])

        main_layout.addWidget(self.stack)

        layout.addWidget(sidebar_frame)
        layout.addWidget(main_content, 1)

        self.setCentralWidget(container)

        self.switch_view(self.state.last_view)

    def switch_view(self, name):
        for btn_name, btn in self.buttons.items():
            btn.setChecked(btn_name == name)

        if name in ("ESP32-C3 Configuration", "Homepage", "Key Management", "Logs", "Graphs"):
            self.pages[name].refresh()

        self.state.last_view = name
        self.state.persist()
        idx = list(self.pages.keys()).index(name)
        self.stack.setCurrentIndex(idx)

    def process_mqtt_events(self):
        while True:
            try:
                event = self.mqtt.event_queue.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "error":
                self.signals.error.emit(event[1])
            elif kind == "node_status":
                self.signals.node_status.emit(event[1], event[2])
            elif kind == "key_status":
                self.signals.key_status.emit(event[1], event[2])
            elif kind == "collection_progress":
                self.signals.collection_progress.emit(event[1], event[2], event[3])
            elif kind == "collection_complete":
                self.signals.collection_complete.emit(event[1], event[2], event[3])
            elif kind == "model_event":
                self.signals.model_event.emit(event[1], event[2])

    def periodic_refresh(self):
        self.process_mqtt_events()
        if self.state.node_online:
            for node_id, last in self.state.node_last_heartbeat.items():
                if not last:
                    continue
                age = time.time() - last
                if age > HEARTBEAT_TIMEOUT_S:
                    self.state.node_online[node_id] = False
        if self.stack.currentWidget() in [self.pages["ESP32-C3 Configuration"], self.pages["Homepage"], self.pages["Key Management"]]:
            self.pages["ESP32-C3 Configuration"].refresh(force=True)
            self.pages["Homepage"].refresh()
            self.pages["Key Management"].refresh()

    def on_node_status(self, node_id, status):
        self.log_message(f"Node {node_id} status {status}")
        self.state.node_online[node_id] = (status == "online")
        if status == "online":
            self.state.node_last_heartbeat[node_id] = time.time()
        self.state.persist()
        if self.stack.currentWidget() in [self.pages["ESP32-C3 Configuration"], self.pages["Homepage"], self.pages["Key Management"]]:
            self.pages["ESP32-C3 Configuration"].refresh(force=True)
            self.pages["Homepage"].refresh()
            self.pages["Key Management"].refresh()

    def on_key_status(self, key_name, status):
        self.log_message(f"{key_name} status changed to {status}")
        self.state.keys[key_name] = status
        self.state.persist()
        if self.stack.currentWidget() == self.pages["Key Management"]:
            self.pages["Key Management"].refresh()
        self.pages["Homepage"].refresh()

    def on_collection_progress(self, node_id, cur, total):
        self.state.collection_in_progress[node_id] = True
        self.state.collection_progress[node_id] = (cur, total)
        page = self.pages["ESP32-C3 Configuration"]
        page.set_progress(node_id, cur, total)
        # Refresh counts/display if the page is visible
        if self.stack.currentWidget() == page:
            page.refresh()
    
    def on_collection_complete(self, node_id, label, session):
        self.log_message(f"Collection complete: {node_id}/{label} session={session}")
        self.state.esp_nodes[node_id][label] = True
        self.state.collection_in_progress[node_id] = False
        self.state.collection_progress[node_id] = (0, 0)
        self.state.expected_calib.pop(node_id, None)
        self.state.expected_session.pop(node_id, None)
        self.state.persist()

        # Immediately reset the calibration progress UI to default (avoid lingering 100%)
        page = self.pages.get("ESP32-C3 Configuration")
        if page:
            try:
                page.set_progress(node_id, 0, 0)
                # ensure UI refresh reflects new state
                page.refresh()
            except Exception:
                pass

        if self.stack.currentWidget() == self.pages["ESP32-C3 Configuration"]:
            self.pages["ESP32-C3 Configuration"].refresh()

    def on_model_event(self, node_id, body):
        event = body.get("event", "")
        if event == "model_ready":
            self.state.model_state[node_id] = True
            self.state.persist()
        elif event in ("model_download_failed", "model_download_incompatible"):
            self.state.model_state[node_id] = False
            self.state.persist()
        self.pages["ESP32-C3 Configuration"].on_model_event(node_id, body)

    def log_message(self, text):
        self.pages["Logs"].append_log(text)

    def closeEvent(self, event):
        self.state.persist()
        self.mqtt.stop()
        event.accept()


class HomePage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(20)
        
        title = QLabel("Dashboard Overview")
        title.setObjectName("PageTitle")
        self.layout.addWidget(title)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)
        
        self.cards = {}
        for label_text in ["Total Keys", "Issued", "In Vault", "Nodes Online", "Nodes Offline"]:
            card = QFrame()
            card.setProperty("class", "StatCard")
            cl = QVBoxLayout(card)
            
            val_lbl = QLabel("0")
            val_lbl.setProperty("class", "StatValue")
            val_lbl.setAlignment(Qt.AlignCenter)
            
            title_lbl = QLabel(label_text)
            title_lbl.setProperty("class", "StatLabel")
            title_lbl.setAlignment(Qt.AlignCenter)
            
            cl.addWidget(val_lbl)
            cl.addWidget(title_lbl)
            cards_layout.addWidget(card)
            self.cards[label_text] = val_lbl

        self.layout.addLayout(cards_layout)

        # Node summary area (compact, mirrors Key Management style)
        nodes_title = QLabel("Nodes")
        nodes_title.setProperty("class", "StatLabel")
        nodes_title.setAlignment(Qt.AlignLeft)
        nodes_title.setStyleSheet("font-size: 16px; margin-top: 8px;")
        self.layout.addWidget(nodes_title)

        # Use a grid layout that wraps into rows so nodes always fit on screen
        self.nodes_widget = QWidget()
        self.nodes_grid = QGridLayout(self.nodes_widget)
        self.nodes_grid.setSpacing(10)
        self.nodes_widget.setObjectName("NodesGridWidget")
        # Allow the widget to expand and let the main layout manage wrapping
        self.nodes_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Add the nodes widget directly (no scroll area) so tiles wrap to next rows
        self.layout.addWidget(self.nodes_widget)

        self.layout.addStretch(1)

    def refresh(self):
        total_keys = len(self.state.keys)
        issued = sum(1 for v in self.state.keys.values() if v == "out")
        vault = total_keys - issued
        online = sum(1 for n in self.state.esp_nodes if self.state.node_online.get(n, False))
        offline = len(self.state.esp_nodes) - online
        
        self.cards["Total Keys"].setText(str(total_keys))
        self.cards["Issued"].setText(str(issued))
        self.cards["In Vault"].setText(str(vault))
        self.cards["Nodes Online"].setText(str(online))
        self.cards["Nodes Offline"].setText(str(offline))

        # Rebuild node tiles
        for i in reversed(range(self.nodes_grid.count())):
            item = self.nodes_grid.takeAt(i)
            if item is None:
                continue
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        nodes = sorted(self.state.esp_nodes.keys())
        if not nodes:
            return
        cols = min(max(1, len(nodes)), 6)

        for i, node in enumerate(nodes):
            online = bool(self.state.node_online.get(node, False))
            model_ready = bool(self.state.model_state.get(node, False))
            node_state = (self.state.node_detected_state.get(node) or "").strip()
            if node_state == "door_open":
                node_state_text = "DOOR OPEN"
                node_state_color = "#d32f2f"
            elif node_state == "door_closed":
                node_state_text = "DOOR CLOSED"
                node_state_color = "#2e7d32"
            elif node_state == "person_standing":
                node_state_text = "PERSON STANDING"
                node_state_color = "#ff8f00"
            elif online:
                node_state_text = "ONLINE"
                node_state_color = "#1976d2"
            else:
                node_state_text = "UNKNOWN"
                node_state_color = "#607d8b"

            if online and model_ready:
                color = "#4caf50"
            elif online and not model_ready:
                color = "#ffb300"
            else:
                color = "#9e9e9e"

            tile = QFrame()
            tile.setStyleSheet("QFrame { background-color: #ffffff; border-radius: 8px; padding: 8px; border: 1px solid #e0e0e0; }")
            tl = QHBoxLayout(tile)
            tl.setContentsMargins(10, 6, 10, 6)
            tl.setSpacing(12)

            badge = QLabel()
            badge.setFixedSize(16, 16)
            badge.setStyleSheet(f"background-color: {color}; border-radius: 7px; border: 1px solid rgba(0,0,0,0.08);")

            name_lbl = QLabel(node)
            name_lbl.setStyleSheet("font-size: 18px; font-weight: bold; background: transparent;")

            state_lbl = QLabel(node_state_text)
            state_lbl.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {node_state_color}; background: transparent;")

            tl.addWidget(badge)
            tl.addWidget(name_lbl)
            tl.addWidget(state_lbl)
            tl.addStretch(1)

            row = i // cols
            col = i % cols
            self.nodes_grid.addWidget(tile, row, col)


class KeyMgmtPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("Key Management")
        title.setObjectName("PageTitle")
        self.layout.addWidget(title)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(15)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_widget)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; } QWidget#GridWidget { background-color: transparent; }")
        self.grid_widget.setObjectName("GridWidget")
        
        self.layout.addWidget(scroll)

    def refresh(self):
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        def extract_num(k_str):
            parts = k_str.split()
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
            return 0
            
        keys = sorted(self.state.keys.items(), key=lambda x: extract_num(x[0]))
        
        cols = 6
        for i, (k, v) in enumerate(keys):
            frame = QFrame()
            color = "#4caf50" if v == "in" else "#f44336"
            frame.setStyleSheet(f"QFrame {{ background-color: {color}; color: white; border-radius: 8px; padding: 16px; border: 1px solid rgba(0,0,0,0.1); }}")
            
            flay = QVBoxLayout(frame)
            flay.setContentsMargins(5, 10, 5, 10)
            flay.setSpacing(8)
            
            lbl = QLabel(k)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 18px; font-weight: bold; background: transparent; border: none;")
            
            status_lbl = QLabel(v.upper())
            status_lbl.setAlignment(Qt.AlignCenter)
            status_lbl.setStyleSheet("font-size: 21px; font-weight: bold; opacity: 0.95; background: transparent; border: none;")
            
            flay.addWidget(lbl)
            flay.addWidget(status_lbl)
            
            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(frame, row, col)


class ESPConfigPage(QWidget):
    def __init__(self, state, parent):
        super().__init__()
        self.state = state
        self.parent = parent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("ESP32-C3 Calibration & Training")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        top_widget = QFrame()
        top_widget.setObjectName("StatCard")
        top_layout = QVBoxLayout(top_widget)
        
        controls_layout = QHBoxLayout()
        
        self.node_selector = QComboBox()
        self.node_selector.addItems(sorted(self.state.esp_nodes.keys()))
        self.node_selector.currentTextChanged.connect(self.update_details)
        controls_layout.addWidget(QLabel("Select Node:"))
        controls_layout.addWidget(self.node_selector)

        self.label_selector = QComboBox()
        self.label_selector.addItems(CALIB_STATES)
        controls_layout.addWidget(QLabel("Calibration:"))
        controls_layout.addWidget(self.label_selector)

        self.split_selector = QComboBox()
        self.split_selector.addItems(["train", "dev"])
        controls_layout.addWidget(QLabel("Split:"))
        controls_layout.addWidget(self.split_selector)

        self.start_button = QPushButton("Start Calibration")
        self.start_button.setObjectName("ActionBtn")
        self.start_button.clicked.connect(self.start_calibration)
        controls_layout.addWidget(self.start_button)
        
        top_layout.addLayout(controls_layout)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: #1976d2; margin-top: 10px; font-size: 15px;")
        top_layout.addWidget(self.status_label)

        self.calib_progress = QProgressBar()
        self.calib_progress.setRange(0, 100)
        self.calib_progress.setFormat("Calibration: %p%")
        top_layout.addWidget(self.calib_progress)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("Model Training Progress")
        top_layout.addWidget(self.train_progress)

        layout.addWidget(top_widget)

        actions_layout = QHBoxLayout()
        self.training_button = QPushButton("Train Model")
        self.training_button.setObjectName("ActionBtn")
        self.training_button.clicked.connect(self.train_model)
        actions_layout.addWidget(self.training_button)

        self.load_button = QPushButton("Load Model")
        self.load_button.setObjectName("ActionBtn")
        self.load_button.clicked.connect(self.load_model)
        actions_layout.addWidget(self.load_button)

        self.reset_button = QPushButton("Reset Calibrations")
        self.reset_button.setObjectName("DangerBtn")
        self.reset_button.clicked.connect(self.reset_calibrations)
        actions_layout.addWidget(self.reset_button)
        actions_layout.addStretch(1)
        
        layout.addLayout(actions_layout)

        table_label = QLabel("Calibration Status")
        table_label.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        layout.addWidget(table_label)

        self.detail_table = QTableWidget(0, 4)
        self.detail_table.setHorizontalHeaderLabels(["State", "Trained?", "Train Rows", "Dev Rows"])
        # Make columns equal width and rows uniform
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detail_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        # Slightly larger default row height; will be adjusted on resize
        self.detail_table.verticalHeader().setDefaultSectionSize(44)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setWordWrap(False)
        self.detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Increase default font size for readability; will be adjusted dynamically
        self.detail_table.setStyleSheet("QTableWidget { font-size: 14px; } QHeaderView::section { font-weight: bold; font-size: 15px; }")
        self.detail_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.detail_table)

    def resizeEvent(self, event):
        # Adjust table row height and font size proportionally when the page is resized
        try:
            total_h = max(300, self.height())
            rows = max(1, max(1, self.detail_table.rowCount()))
            # Reserve some space for controls; use remaining height for rows
            reserved = 220
            avail = max(100, total_h - reserved)
            # Ensure rows are comfortably tall for larger text
            row_h = max(36, int(avail / (rows + 0.5)))
            self.detail_table.verticalHeader().setDefaultSectionSize(row_h)

            # Compute header height (fallback if not yet shown)
            header_h = self.detail_table.horizontalHeader().height() or 36
            desired_h = header_h + rows * row_h + 8

            # Force table to expand to fit content and avoid scrollbars
            self.detail_table.setMinimumHeight(desired_h)
            self.detail_table.setMaximumHeight(desired_h)
            self.detail_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.detail_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.detail_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            # Scale font size with row height and make it bold for readability
            font = self.detail_table.font()
            font.setPointSize(max(12, int(row_h / 2)))
            font.setBold(True)
            self.detail_table.setFont(font)

            header_font = self.detail_table.horizontalHeader().font()
            header_font.setPointSize(max(13, int(row_h / 2.2)))
            header_font.setBold(True)
            self.detail_table.horizontalHeader().setFont(header_font)
        except Exception:
            pass
        return super().resizeEvent(event)

    def refresh(self, force=False):
        node = self.node_selector.currentText()
        if not node:
            return
        status = "online" if self.state.node_online.get(node) else "offline"

        # Check for model artifacts on disk
        node_model_dir = os.path.join(MODEL_STORE_DIR, node)
        model_path = os.path.join(node_model_dir, "model.tflite")
        scaler_path = os.path.join(node_model_dir, "scaler_params.json")
        model_exists = os.path.isfile(model_path)

        prev_model_state = self.state.model_state.get(node, False)
        self.state.model_state[node] = bool(model_exists)
        if prev_model_state != self.state.model_state[node]:
            try:
                self.state.persist()
            except Exception:
                pass

        # Compose model status text
        if model_exists:
            try:
                size = os.path.getsize(model_path)
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(model_path)))
                scaler_present = os.path.isfile(scaler_path)
                model_text = f"Model: model.tflite ({size} bytes) | Scaler: {'present' if scaler_present else 'missing'} | Built: {mtime}"
            except Exception:
                model_text = "Model: model.tflite (unknown)"
        else:
            model_text = "No model loaded"

        base_text = f"Node: {node} | Status: {status.upper()} | Current State: {self.state.node_detected_state.get(node, 'unknown')} | {model_text}"
        cur, total = self.state.collection_progress.get(node, (0, 0))
        ready_from_progress = total > 0 and cur >= total
        collecting = bool(self.state.collection_in_progress.get(node, False))

        # Calibration action state mirrors legacy behavior: disable while collecting,
        # then re-enable once upload reaches the final sub-batch.
        self.start_button.setEnabled(not (collecting and not ready_from_progress))

        if self.state.training_in_progress.get(node, False):
            self.status_label.setText("Training in progress...")
        elif collecting:
            if ready_from_progress:
                self.status_label.setText(f"Status: Upload complete ({cur}/{total}) - waiting for server merge")
            elif total > 0:
                self.status_label.setText(f"Status: Calibrating {cur}/{total} sub-batches uploaded")
            else:
                self.status_label.setText("Status: Calibration started")
        else:
            self.status_label.setText(base_text)

        # Enable/disable training and load buttons appropriately
        can_train = all(self.state.esp_nodes.get(node, {}).get(s, False) for s in CALIB_STATES) and not self.state.training_in_progress.get(node, False)
        self.training_button.setEnabled(bool(can_train))
        if hasattr(self, "load_button"):
            self.load_button.setEnabled(bool(model_exists))

        self.detail_table.setRowCount(len(CALIB_STATES))
        summary = self.state.build_dataset_summary(node)
        labels_info = summary.get("labels", {}) if summary else {}

        for i, st in enumerate(CALIB_STATES):
            # Show per-state CSV file count excluding files starting with 'part_'
            file_count = self.state.count_state_csv_files(node, st)
            self.detail_table.setItem(i, 0, QTableWidgetItem(f"{st} ({file_count})"))
            done = self.state.esp_nodes.get(node, {}).get(st, False)
            self.detail_table.setItem(i, 1, QTableWidgetItem("Yes" if done else "No"))
            
            info = labels_info.get(st, {})
            splits = info.get("split_rows", {}) if isinstance(info, dict) else {}
            self.detail_table.setItem(i, 2, QTableWidgetItem(str(splits.get("train", 0))))
            self.detail_table.setItem(i, 3, QTableWidgetItem(str(splits.get("dev", 0))))

    def count_state_rows(self, node, state, base):
        path = os.path.join(base, node, state)
        if not os.path.isdir(path):
            return 0
        csvs = glob.glob(os.path.join(path, "*.csv"))
        total = 0
        for c in csvs:
            try:
                df = pd.read_csv(c)
                total += len(df)
            except Exception:
                pass
        return total

    def update_details(self, _node):
        self.refresh()

    def _publish_collect(self, node, label, split, retries=3):
        if not node:
            return False

        campaign_id = self.state.collection_campaign.get(node)
        if not campaign_id:
            campaign_id = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.state.collection_campaign[node] = campaign_id

        self.state.collection_run_counter[node] = self.state.collection_run_counter.get(node, 0) + 1
        run_id = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.state.collection_run_counter[node]:02d}"

        topic = f"/commands/{node}/collect"
        payload = json.dumps({
            "label": label,
            "session": run_id,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "split_group": split,
        })

        for i in range(retries):
            info = self.parent.mqtt.client.publish(topic, payload, qos=1)
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                self.state.expected_calib[node] = label
                self.state.expected_session[node] = run_id
                self.state.collection_in_progress[node] = True
                self.state.collection_progress[node] = (0, 0)
                return True
            time.sleep(0.2 * (i + 1))

        self.state.expected_calib.pop(node, None)
        self.state.expected_session.pop(node, None)
        return False

    def start_calibration(self):
        node = self.node_selector.currentText()
        label = self.label_selector.currentText()
        split = self.split_selector.currentText()
        if not node:
            QMessageBox.warning(self, "No Node", "Please select a node")
            return

        if self._publish_collect(node, label, split):
            self.status_label.setText(f"Status: Calibration started for {node} ({label}, {split})")
            self.start_button.setEnabled(False)
            self.calib_progress.setValue(0)
            self.state.persist()
            self.refresh(force=False)
        else:
            self.state.collection_in_progress[node] = False
            self.state.collection_progress[node] = (0, 0)
            self.status_label.setText("Collect publish failed")

    def train_model(self):
        node = self.node_selector.currentText()
        if not node:
            return

        if self.training_button.isEnabled() is False:
            return

        # Pre-flight dataset checks: ensure each calibration label has files/rows
        summary = self.state.build_dataset_summary(node)
        if summary is None:
            QMessageBox.warning(self, "Training Not Ready", f"No calibration data found for {node}.")
            return
        missing = []
        for s in CALIB_STATES:
            info = summary["labels"].get(s, {})
            files = info.get("files", 0)
            rows = info.get("rows_clean", 0)
            if files <= 0 or rows <= 0:
                missing.append(f"{s}: files={files}, rows={rows}")
        if missing:
            details = "\n".join(missing)
            QMessageBox.critical(self, "Training Not Ready", "Training requires all 3 classes with data: door_open, door_closed, person_standing.\nMissing or incomplete:\n" + details)
            return

        self.state.training_in_progress[node] = True
        self.state.persist()
        self.training_button.setEnabled(False)
        self.status_label.setText("Training in progress...")
        self.train_progress.setRange(0, 0)

        worker = threading.Thread(target=self._training_worker, args=(node,), daemon=True)
        worker.start()

    def _training_worker(self, node):
        success = False
        msg = ""
        try:
            model_output = os.path.join(MODEL_STORE_DIR, node, "model.tflite")
            scaler_output = os.path.join(MODEL_STORE_DIR, node, "scaler_params.json")
            data_dir = self.state.config.get("csi_data_dir", CSI_DATA_DIR)

            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "edge_ml.py"),
                "--data-dir", data_dir,
                "--node", node,
                "--output", model_output,
                "--scaler-output", scaler_output,
                "--feature-condense-groups", "8"
            ]
            self.parent.signals.error.emit(f"Training command: {' '.join(cmd)}")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                msg = f"Training process failed ({r.returncode}): {r.stderr.strip()}"
            else:
                msg = f"Training completed for {node}: {r.stdout.splitlines()[-1] if r.stdout else ''}"
                success = True

            if success:
                if not os.path.isfile(model_output):
                    raise RuntimeError(f"Trained model file missing: {model_output}")
                if not os.path.isfile(scaler_output):
                    raise RuntimeError(f"Scaler params file missing: {scaler_output}")

                # Use the existing dashboard MQTT connection to avoid spawning
                # transient helper clients that appear as repeated connect/disconnect.
                try:
                    topic = f"/commands/{node}/load_model"
                    info = self.parent.mqtt.client.publish(topic, "", qos=1)
                    if getattr(info, "rc", None) != mqtt.MQTT_ERR_SUCCESS:
                        self.parent.log_message(f"Failed to publish load_model for {node}: rc={getattr(info, 'rc', None)}")
                except Exception as e:
                    self.parent.log_message(f"Failed to publish load_model for {node}: {e}")

                self.state.model_state[node] = True
                self.state.persist()
                self._notify_training_complete(node)
            else:
                self.state.model_state[node] = False
                self.state.persist()

        except subprocess.TimeoutExpired as e:
            msg = f"Training timeout: {e}"
        except Exception as e:
            msg = f"Training exception: {e}"
        finally:
            try:
                self.state.training_in_progress[node] = False
                self.state.persist()
            except Exception:
                pass
            self.parent.signals.training_result.emit(node, success, msg)
            self.parent.signals.error.emit(msg)

    def on_training_result(self, node, success, msg):
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(100 if success else 0)
        if self.node_selector.currentText() == node:
            self.status_label.setText("Model trained" if success else "Training failed")
        self.training_button.setEnabled(True)
        self.parent.log_message(msg)
        # Show a user-friendly popup when training fails to make the issue visible to the user
        if not success:
            display_msg = msg or "Training failed. See logs for details."
            try:
                QMessageBox.warning(self, "Training Failed", display_msg)
            except Exception:
                # If GUI popup fails for any reason, ensure message is still logged
                self.parent.log_message(f"Failed to show popup: {display_msg}")

    def reset_calibrations(self):
        node = self.node_selector.currentText()
        if not node:
            return
        
        reply = QMessageBox.question(
            self, 
            'Reset Calibrations',
            f'Delete ALL calibration data for {node}?\n\nThis will:\n- Clear all state flags\n- Delete CSI data files\n- Delete model artifacts\n- Cannot be undone',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return

        try:
            # Update in-memory state immediately and persist so UI reflects reset
            for state in CALIB_STATES:
                self.state.esp_nodes[node][state] = False

            self.state.collection_campaign.pop(node, None)
            self.state.collection_run_counter.pop(node, None)
            self.state.expected_calib.pop(node, None)
            self.state.expected_session.pop(node, None)
            self.state.collection_in_progress[node] = False
            self.state.collection_progress[node] = (0, 0)
            self.state.model_state[node] = False
            self.state.training_in_progress[node] = False
            self.state.node_last_heartbeat[node] = 0.0

            self.state.persist()

            # Run filesystem deletion + MQTT trigger in background to avoid UI freeze
            def _bg_reset():
                try:
                    import reset_helpers as _reset_helpers
                    csi_root = self.state.resolve_csi_data_dir()
                    ok = _reset_helpers.reset_node(node, csi_root=csi_root, model_root=MODEL_STORE_DIR, trigger_mqtt=False)
                    if ok:
                        self.parent.log_message(f"Reset calibrations for {node} (background)")
                    else:
                        self.parent.log_message(f"Reset calibrations for {node} (background: partial failures)")
                except Exception as e:
                    self.parent.log_message(f"Background reset failed: {e}")

            threading.Thread(target=_bg_reset, daemon=True).start()
            try:
                topic = f"/commands/{node}/reset_model"
                payload = json.dumps({"reason": "dashboard_rack_reset"})
                info = self.parent.mqtt.client.publish(topic, payload, qos=1)
                if getattr(info, "rc", None) != mqtt.MQTT_ERR_SUCCESS:
                    self.parent.log_message(f"Failed to publish reset_model for {node}: rc={getattr(info, 'rc', None)}")
            except Exception as e:
                self.parent.log_message(f"Failed to publish reset_model for {node}: {e}")

            # Update UI immediately to reflect reset
            try:
                self.status_label.setText("Status: No model loaded")
                # make it stand out as cleared
                self.status_label.setStyleSheet("font-weight: bold; color: #d32f2f; margin-top: 10px; font-size: 15px;")
                self.calib_progress.setValue(0)
                self.start_button.setEnabled(True)
                self.train_progress.setValue(0)
                self.training_button.setEnabled(False)
            except Exception:
                pass

            self.parent.log_message(f"Reset calibrations initiated for {node}")
            self.refresh()
        except Exception as e:
            self.parent.log_message(f"Failed to reset {node}: {e}")

    def _notify_training_complete(self, node):
        topic = f"/commands/{node}/training_complete"
        payload = json.dumps({"session": str(int(time.time()))})
        self.parent.mqtt.client.publish(topic, payload, qos=1)

    def load_model(self):
        node = self.node_selector.currentText()
        if not node:
            return
        node_model_dir = os.path.join(MODEL_STORE_DIR, node)
        model_path = os.path.join(node_model_dir, "model.tflite")
        if not os.path.isfile(model_path):
            QMessageBox.information(self, "No Model", f"No model file found for {node}")
            return
        try:
            topic = f"/commands/{node}/load_model"
            info = self.parent.mqtt.client.publish(topic, "", qos=1)
            success_rc = getattr(info, "rc", None)
            if success_rc == mqtt.MQTT_ERR_SUCCESS or success_rc == 0:
                self.status_label.setText("Triggered model load on device")
                self.parent.log_message(f"Triggered model load for {node}")
            else:
                self.parent.log_message(f"Failed to trigger model load for {node}: rc={success_rc}")
        except Exception as e:
            self.parent.log_message(f"Exception triggering model load: {e}")

    def set_progress(self, node, cur, total):
        if self.node_selector.currentText() != node:
            return
        if total <= 0:
            self.calib_progress.setValue(0)
            self.start_button.setEnabled(False)
            return
        self.calib_progress.setValue(int(cur / total * 100))
        self.start_button.setEnabled(cur >= total)
        if cur < total:
            self.status_label.setText(f"Status: {cur}/{total} sub-batches uploaded")
        else:
            self.status_label.setText(f"Status: Upload complete ({cur}/{total}) - waiting for server merge")

    def on_model_event(self, node, body):
        if self.node_selector.currentText() != node:
            return
        event = body.get("event", "")
        if event == "model_ready":
            self.status_label.setText("Model ready on device")


class GraphsPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.graph_popup = None
        self.state_file_map = {}
        self.state_data_cache = {}
        self.feature_columns = []
        self.state_stats = {}

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Dashboard Analytics")
        title.setObjectName("PageTitle")
        self.layout.addWidget(title)

        top_widget = QFrame()
        top_widget.setObjectName("StatCard")
        top_layout = QHBoxLayout(top_widget)

        self.node_selector = QComboBox()
        self.node_selector.addItems(sorted(self.state.esp_nodes.keys()))
        self.node_selector.currentTextChanged.connect(self.refresh_data)
        top_layout.addWidget(QLabel("Select Node:"))
        top_layout.addWidget(self.node_selector)

        self.split_selector = QComboBox()
        self.split_selector.addItems(["train", "dev", "train + dev"])
        self.split_selector.currentTextChanged.connect(self.refresh_data)
        top_layout.addWidget(QLabel("Split:"))
        top_layout.addWidget(self.split_selector)

        self.show_graph_btn = QPushButton("Show Graph")
        self.show_graph_btn.setObjectName("ActionBtn")
        self.show_graph_btn.clicked.connect(self.show_graph)
        self.show_graph_btn.setEnabled(False)
        top_layout.addWidget(self.show_graph_btn)

        top_layout.addStretch(1)
        self.layout.addWidget(top_widget)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 13px;")
        self.layout.addWidget(self.summary_text)

        self.refresh_data()

    def _close_graph_popup(self):
        if self.graph_popup is not None:
            try:
                self.graph_popup.close()
            except Exception:
                pass
        self.graph_popup = None

    def _on_graph_popup_closed(self):
        self.graph_popup = None

    def node_data_dir(self, node):
        base = self.state.config.get("csi_data_dir") if self.state.config.get("csi_data_dir") else CSI_DATA_DIR
        return os.path.join(base, node)

    def list_files_for(self, node, label, split):
        node_dir = self.node_data_dir(node)
        if not os.path.isdir(node_dir):
            return []
        label_dir = os.path.join(node_dir, label)
        if not os.path.isdir(label_dir):
            return []
        files = sorted(glob.glob(os.path.join(label_dir, "*.csv")))
        matched = []
        for p in files:
            if os.path.basename(p).startswith("part_"):
                continue
            if split is None:
                matched.append(p)
                continue
            try:
                df = pd.read_csv(p, nrows=10)
                split_group = None
                for col in ("collection_split", "split_group", "dataset_split"):
                    if col in df.columns:
                        split_group = str(df[col].mode().iloc[0]).lower() if not df[col].mode().empty else None
                        break
                if split_group == "test":
                    split_group = "dev"
                if split_group is None or split_group == split:
                    matched.append(p)
            except Exception:
                matched.append(p)
        return matched

    def _feature_sort_key(self, name):
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        if digits:
            return (0, int(digits), str(name))
        return (1, str(name))

    def _feature_columns_from_df(self, df):
        cols = []
        for col in df.columns:
            name = str(col).strip()
            if name.lower() in CSI_METADATA_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(name)
        return cols

    def _selected_splits(self):
        mode = self.split_selector.currentText()
        if mode == "train + dev":
            return ["train", "dev"]
        if mode in ("train", "dev"):
            return [mode]
        return ["train"]

    def refresh_data(self, *args):
        node = self.node_selector.currentText()
        split_mode = self.split_selector.currentText()
        selected_splits = self._selected_splits()
        self._close_graph_popup()

        self.state_file_map = {split: {} for split in selected_splits}
        self.state_data_cache = {}
        self.feature_columns = []
        self.state_stats = {split: {} for split in selected_splits}

        feature_seen = set()
        temp_frames = {split: {} for split in selected_splits}
        summary_lines = [
            f"Node: {node}",
            f"Split mode: {split_mode}",
            "",
        ]

        total_files = 0
        total_usable_files = 0
        total_rows = 0

        for split in selected_splits:
            split_total_files = 0
            split_usable_files = 0
            split_rows = 0
            summary_lines.append(f"{split}:")

            for state in CALIB_STATES:
                files = self.list_files_for(node, state, split)
                self.state_file_map[split][state] = files
                split_total_files += len(files)
                total_files += len(files)

                state_rows = 0
                state_usable = 0
                state_frames = []

                for path in files:
                    try:
                        df = pd.read_csv(path)
                    except Exception:
                        continue

                    feature_cols = self._feature_columns_from_df(df)
                    if not feature_cols:
                        continue

                    state_rows += len(df)
                    state_usable += 1
                    split_rows += len(df)
                    total_rows += len(df)
                    state_frames.append(df[feature_cols].copy())
                    for col in feature_cols:
                        if col not in feature_seen:
                            feature_seen.add(col)
                            self.feature_columns.append(col)

                temp_frames[split][state] = state_frames
                split_usable_files += state_usable
                total_usable_files += state_usable
                self.state_stats[split][state] = {
                    "files": len(files),
                    "usable_files": state_usable,
                    "rows": state_rows,
                }
                summary_lines.append(
                    f"  {state}: files={len(files)} usable={state_usable} rows={state_rows}"
                )

            summary_lines.append(
                f"  total: files={split_total_files} usable={split_usable_files} rows={split_rows}"
            )
            summary_lines.append("")

        self.feature_columns.sort(key=self._feature_sort_key)

        for split, states in temp_frames.items():
            self.state_data_cache[split] = {}
            for state, frames in states.items():
                aligned_frames = [frame.reindex(columns=self.feature_columns) for frame in frames]
                if aligned_frames:
                    self.state_data_cache[split][state] = pd.concat(aligned_frames, ignore_index=True)
                else:
                    self.state_data_cache[split][state] = pd.DataFrame(columns=self.feature_columns)

        summary_lines.extend([
            "",
            f"Feature channels detected: {len(self.feature_columns)}",
            f"Matched files: {total_files}",
            f"Usable files: {total_usable_files}",
            f"Total rows loaded: {total_rows}",
            "",
            "All SC channels are selected automatically for the comparison graph.",
        ])

        self.summary_text.setPlainText("\n".join(summary_lines))
        self.show_graph_btn.setEnabled(bool(self.feature_columns))

    def _state_display_name(self, state):
        return {
            "door_closed": "Door Closed",
            "door_open": "Door Open",
            "person_standing": "Person Standing",
        }.get(state, state)

    def _render_graph(self):
        if self.graph_popup is None:
            self.graph_popup = GraphPopupWindow(on_close=self._on_graph_popup_closed)

        split_mode = self.split_selector.currentText()
        self.graph_popup.setWindowTitle(
            f"CSI Signature Comparison ({self.node_selector.currentText()}, {split_mode})"
        )
        self.graph_popup.render_comparison(
            node=self.node_selector.currentText(),
            split_mode=split_mode,
            active_splits=self._selected_splits(),
            selected_channels=self.feature_columns,
            state_data_cache=self.state_data_cache,
            state_display_name=self._state_display_name,
        )
        self.graph_popup.showMaximized()
        self.graph_popup.raise_()
        self.graph_popup.activateWindow()

    def show_graph(self):
        if not self.feature_columns:
            QMessageBox.information(self, "No Data", "No CSI channels were found for this rack/split.")
            return
        self._render_graph()

    def refresh(self):
        node = self.node_selector.currentText()
        split_mode = self.split_selector.currentText()
        if not node:
            self.summary_text.setPlainText("No node selected")
            return
        if not self.feature_columns:
            self.summary_text.setPlainText(f"No CSI data available for {node} ({split_mode})")
            return

        lines = [f"Node: {node}", f"Split mode: {split_mode}", ""]
        for split in self._selected_splits():
            lines.append(f"{split}:")
            for state in CALIB_STATES:
                info = self.state_stats.get(split, {}).get(state, {})
                lines.append(
                    f"  {state}: files={info.get('files', 0)} usable={info.get('usable_files', 0)} rows={info.get('rows', 0)}"
                )
            lines.append("")
        lines.extend([
            "",
            f"Feature channels: {len(self.feature_columns)}",
            "",
            "Click Show Graph to open the popup comparison.",
        ])
        self.summary_text.setPlainText("\n".join(lines))


class GraphPopupWindow(QMainWindow):
    def __init__(self, on_close=None):
        super().__init__()
        self._on_close = on_close
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.resize(1200, 760)

    def closeEvent(self, event):
        if callable(self._on_close):
            self._on_close()
        self.deleteLater()
        super().closeEvent(event)

    def clear_message(self, message):
        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=16)
        ax.set_axis_off()
        self.canvas.draw()

    def render_comparison(self, node, split_mode, active_splits, selected_channels, state_data_cache, state_display_name):
        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if not selected_channels:
            self.clear_message("No channels selected")
            return

        x = list(range(len(selected_channels)))
        colors = {
            "door_closed": "#7b1fa2",
            "door_open": "#2e7d32",
            "person_standing": "#212121",
        }
        split_styles = {
            "train": "-",
            "dev": "--",
        }

        plotted_any = False
        for split in active_splits:
            split_cache = state_data_cache.get(split, {})
            for state in CALIB_STATES:
                df = split_cache.get(state)
                if df is None or df.empty:
                    continue
                aligned = df.reindex(columns=selected_channels)
                series = aligned.mean(axis=0)
                if series.isna().all():
                    continue
                ax.plot(
                    x,
                    series.values,
                    label=f"{split.title()}: {state_display_name(state)}",
                    color=colors.get(state, "#1976d2"),
                    linestyle=split_styles.get(split, "-"),
                    linewidth=2,
                    alpha=0.95 if split == "train" else 0.85,
                )
                plotted_any = True

        if not plotted_any:
            self.clear_message("No matching CSI data found for the selected channels")
            return

        ax.set_title(f"CSI Signature Comparison ({node}, {split_mode})")
        ax.set_xlabel("CSI feature channels")
        ax.set_ylabel("Mean signal")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_channels, rotation=90, fontsize=7)
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
        fig.tight_layout()
        self.canvas.draw()


class SettingsPage(QWidget):
    def __init__(self, state, parent):
        super().__init__()
        self.state = state
        self.parent = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("System Settings")
        title.setObjectName("PageTitle")
        self.layout.addWidget(title)
        
        form_frame = QFrame()
        form_frame.setObjectName("StatCard")
        form_layout = QFormLayout(form_frame)
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setSpacing(15)

        self.broker_input = QLineEdit(str(self.state.config.get("broker", DEFAULT_MQTT_BROKER)))
        self.broker_input.setFixedWidth(300)
        
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setButtonSymbols(QSpinBox.NoButtons)
        self.port_input.setValue(int(self.state.config.get("port", DEFAULT_MQTT_PORT)))
        self.port_input.setFixedWidth(100)
        
        self.csi_input = QLineEdit(self.state.config.get("csi_data_dir", CSI_DATA_DIR))
        self.csi_input.setFixedWidth(400)

        form_layout.addRow(QLabel("<b>MQTT Broker:</b>"), self.broker_input)
        form_layout.addRow(QLabel("<b>MQTT Port:</b>"), self.port_input)
        form_layout.addRow(QLabel("<b>CSI Data Directory:</b>"), self.csi_input)

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setObjectName("ActionBtn")
        self.save_btn.setFixedWidth(200)
        self.save_btn.clicked.connect(self.save_settings)
        form_layout.addRow("", self.save_btn)
        
        self.layout.addWidget(form_frame)
        self.layout.addStretch(1)

    def save_settings(self):
        self.state.config["broker"] = self.broker_input.text().strip()
        self.state.config["port"] = self.port_input.value()
        self.state.config["csi_data_dir"] = self.csi_input.text().strip()
        self.state.persist()

        self.parent.mqtt.stop()
        self.parent.mqtt = MqttClient(self.state, self.parent.signals)
        self.parent.mqtt.start()

        QMessageBox.information(self, "Settings", "Saved and restarted MQTT connection")

    def refresh(self):
        pass


class LogsPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("System Logs")
        title.setObjectName("PageTitle")
        self.layout.addWidget(title)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 13px; background-color: #2b2b2b; color: #a9b7c6; border-radius: 6px;")
        self.layout.addWidget(self.text_area)

    def append_log(self, message):
        self.text_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def refresh(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardMain()
    window.show()
    sys.exit(app.exec())
