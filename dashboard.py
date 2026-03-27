import os
import sys
import json
import glob
import time
import threading
import traceback
import subprocess
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
    QFileDialog,
    QProgressBar,
    QFormLayout,
    QSpinBox,
    QFrame,
    QGridLayout,
    QScrollArea,
)

import pandas as pd
import paho.mqtt.client as mqtt

from shared_config import DEFAULT_MQTT_BROKER, DEFAULT_MQTT_PORT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSI_DATA_DIR = os.path.join(BASE_DIR, "csi_data")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

CALIB_STATES = ["door_closed", "door_open", "person_standing"]

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

QPushButton#DangerBtn {
    background-color: #f44336;
}
QPushButton#DangerBtn:hover {
    background-color: #e53935;
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
    collection_progress = Signal(str, int, int)
    model_event = Signal(str, dict)
    training_result = Signal(str, bool, str)
    error = Signal(str)


class MqttClient(threading.Thread):
    def __init__(self, state, signals):
        super().__init__(daemon=True)
        self.state = state
        self.signals = signals
        self.client = mqtt.Client(client_id="DashboardClient")
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self._stop = threading.Event()

    def run(self):
        host = self.state.config.get("broker", DEFAULT_MQTT_BROKER)
        port = int(self.state.config.get("port", DEFAULT_MQTT_PORT))
        try:
            self.client.connect(host, port, 60)
            self.client.loop_start()
            while not self._stop.is_set():
                time.sleep(0.3)
        except Exception as e:
            self.signals.error.emit(f"MQTT start failed: {e}")

    def stop(self):
        self._stop.set()
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            topics = [("device/+/status", 1), ("/sensors/+/status", 1)]
            for topic, qos in topics:
                client.subscribe(topic, qos=qos)
            self.signals.error.emit("MQTT connected")
        else:
            self.signals.error.emit(f"MQTT connect fail {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.signals.error.emit("MQTT disconnected")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode(errors="ignore")
            if topic.startswith("device/") and topic.endswith("/status"):
                node_id = topic.split("/")[1]
                status = payload.strip().lower()
                self.state.node_online[node_id] = (status == "online")
                if status == "online":
                    self.state.node_last_heartbeat[node_id] = time.time()
                self.signals.node_status.emit(node_id, status)
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
                    self.signals.collection_progress.emit(node_id, sub, total)
                elif event == "model_ready":
                    self.state.model_state[node_id] = True
                    self.signals.model_event.emit(node_id, body)
                elif event == "model_download_failed" or event == "model_download_incompatible":
                    self.state.model_state[node_id] = False
                    self.signals.model_event.emit(node_id, body)
                elif event == "heartbeat":
                    self.state.node_last_heartbeat[node_id] = time.time()
                elif event == "state_change":
                    self.state.node_detected_state[node_id] = body.get("state", "")
        except Exception as e:
            self.signals.error.emit(f"MQTT message handling error: {e}")


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

    def node_data_dir(self, node):
        base = self.config.get("csi_data_dir", CSI_DATA_DIR)
        return os.path.join(base, node)

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
                "split_files": {"train": 0, "dev": 0, "test": 0},
                "split_rows": {"train": 0, "dev": 0, "test": 0},
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
                if "split_group" in df.columns:
                    split_group = str(df["split_group"].mode().iloc[0]).lower() if not df["split_group"].mode().empty else "train"
                elif "dataset_split" in df.columns:
                    split_group = str(df["dataset_split"].mode().iloc[0]).lower() if not df["dataset_split"].mode().empty else "train"
                if split_group not in ("train", "dev", "test"):
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
        self.signals.collection_progress.connect(self.on_collection_progress)
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
        logo_label = QLabel("Zero-Trust\\nKey Governance")
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
            
        if name == "ESP32-C3 Configuration":
            self.pages[name].refresh()
        if name == "Logs":
            self.pages[name].refresh()
        if name == "Graphs":
            self.pages[name].refresh()

        self.state.last_view = name
        self.state.persist()
        idx = list(self.pages.keys()).index(name)
        self.stack.setCurrentIndex(idx)

    def periodic_refresh(self):
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

    def on_collection_progress(self, node_id, cur, total):
        self.pages["ESP32-C3 Configuration"].set_progress(node_id, cur, total)

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
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                
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
            frame.setStyleSheet(f"QFrame {{ background-color: {color}; color: white; border-radius: 8px; padding: 15px; border: 1px solid rgba(0,0,0,0.1); }}")
            
            flay = QVBoxLayout(frame)
            flay.setContentsMargins(5, 10, 5, 10)
            
            lbl = QLabel(k)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 16px; font-weight: bold; background: transparent; border: none;")
            
            status_lbl = QLabel(v.upper())
            status_lbl.setAlignment(Qt.AlignCenter)
            status_lbl.setStyleSheet("font-size: 12px; font-weight: normal; opacity: 0.9; background: transparent; border: none;")
            
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
        self.split_selector.addItems(["train", "dev", "test"])
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

        self.reset_button = QPushButton("Reset Calibrations")
        self.reset_button.setObjectName("DangerBtn")
        self.reset_button.clicked.connect(self.reset_calibrations)
        actions_layout.addWidget(self.reset_button)
        actions_layout.addStretch(1)
        
        layout.addLayout(actions_layout)

        table_label = QLabel("Calibration Status")
        table_label.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        layout.addWidget(table_label)

        self.detail_table = QTableWidget(0, 5)
        self.detail_table.setHorizontalHeaderLabels(["State", "Trained?", "Train Rows", "Dev Rows", "Test Rows"])
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.detail_table)

    def refresh(self, force=False):
        node = self.node_selector.currentText()
        if not node:
            return
        status = "online" if self.state.node_online.get(node) else "offline"
        self.status_label.setText(f"Node: {node} | Status: {status.upper()} | Current State: {self.state.node_detected_state.get(node, 'unknown')}")

        self.detail_table.setRowCount(len(CALIB_STATES))
        summary = self.state.build_dataset_summary(node)
        labels_info = summary.get("labels", {}) if summary else {}

        for i, st in enumerate(CALIB_STATES):
            self.detail_table.setItem(i, 0, QTableWidgetItem(st))
            done = self.state.esp_nodes.get(node, {}).get(st, False)
            self.detail_table.setItem(i, 1, QTableWidgetItem("Yes" if done else "No"))
            
            info = labels_info.get(st, {})
            splits = info.get("split_rows", {}) if isinstance(info, dict) else {}
            self.detail_table.setItem(i, 2, QTableWidgetItem(str(splits.get("train", 0))))
            self.detail_table.setItem(i, 3, QTableWidgetItem(str(splits.get("dev", 0))))
            self.detail_table.setItem(i, 4, QTableWidgetItem(str(splits.get("test", 0))))

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
            self.status_label.setText(f"Sent collect to {node} for {label} [{split}]")
            self.state.esp_nodes[node][label] = True
            self.state.persist()
            self.refresh(initial=False)
        else:
            self.status_label.setText("Collect publish failed")

    def train_model(self):
        node = self.node_selector.currentText()
        if not node:
            return

        if self.training_button.isEnabled() is False:
            return

        if any(not self.state.esp_nodes[node][s] for s in CALIB_STATES):
            QMessageBox.information(self, "Training Not Ready", "Complete all calibration states first.")
            return

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
            self.parent.signals.training_result.emit(node, success, msg)
            self.parent.signals.error.emit(msg)

    def on_training_result(self, node, success, msg):
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(100 if success else 0)
        if self.node_selector.currentText() == node:
            self.status_label.setText("Model trained" if success else "Training failed")
        self.training_button.setEnabled(True)
        self.parent.log_message(msg)

    def reset_calibrations(self):
        node = self.node_selector.currentText()
        if not node:
            return
        for state in CALIB_STATES:
            self.state.esp_nodes[node][state] = False
        self.state.collection_campaign.pop(node, None)
        self.state.collection_run_counter.pop(node, None)
        self.state.expected_calib.pop(node, None)
        self.state.expected_session.pop(node, None)
        self.state.persist()
        self.parent.signals.error.emit(f"Reset calibrations for {node}")
        self.refresh()

    def _notify_training_complete(self, node):
        topic = f"/commands/{node}/training_complete"
        payload = json.dumps({"session": str(int(time.time()))})
        self.parent.mqtt.client.publish(topic, payload, qos=1)

    def set_progress(self, node, cur, total):
        if self.node_selector.currentText() != node:
            return
        if total <= 0:
            self.calib_progress.setValue(0)
            return
        self.calib_progress.setValue(int(cur / total * 100))

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
        self.node_selector.currentTextChanged.connect(self.refresh)
        
        top_layout.addWidget(QLabel("Select Node:"))
        top_layout.addWidget(self.node_selector)

        self.refresh_btn = QPushButton("Refresh Summary")
        self.refresh_btn.clicked.connect(self.refresh)
        top_layout.addWidget(self.refresh_btn)

        self.plot_btn = QPushButton("Show Visual Data Distribution")
        self.plot_btn.setObjectName("ActionBtn")
        self.plot_btn.clicked.connect(self.plot_summary)
        top_layout.addWidget(self.plot_btn)
        top_layout.addStretch(1)

        self.layout.addWidget(top_widget)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 13px;")
        self.layout.addWidget(self.summary_text)

    def refresh(self):
        node = self.node_selector.currentText()
        if not node:
            self.summary_text.setPlainText("No node selected")
            return

        summary = self.state.build_dataset_summary(node)
        if not summary:
            self.summary_text.setPlainText(f"No CSI data available for {node}")
            return

        lines = [f"Node: {node}", f"Generated: {summary.get('generated_at')}", f"Total files: {summary.get('total_files')}", f"Total rows: {summary.get('total_rows')}", ""]
        for label, info in summary.get("labels", {}).items():
            lines.append(f"{label}: files={info.get('files')} sessions={info.get('sessions')} rows={info.get('rows_clean')}")
            lines.append(f"  splits -> train={info['split_rows'].get('train',0)} dev={info['split_rows'].get('dev',0)} test={info['split_rows'].get('test',0)}")
        self.summary_text.setPlainText("\n".join(lines))

    def plot_summary(self):
        node = self.node_selector.currentText()
        if not node:
            return
        summary = self.state.build_dataset_summary(node)
        if not summary or "labels" not in summary:
            QMessageBox.information(self, "No Data", f"No dataset data for {node}")
            return

        import matplotlib.pyplot as plt

        labels = list(summary["labels"].keys())
        rows = [summary["labels"][s]["rows_clean"] for s in labels]

        plt.figure(figsize=(10, 4))
        plt.bar(labels, rows, color=["#4caf50", "#2196f3", "#ff9800"])
        plt.title(f"CSI row count by label for {node}")
        plt.ylabel("Rows")
        plt.xlabel("Label")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


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
