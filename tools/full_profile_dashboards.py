# Profiling script for dashboard comparison
import os
import sys
import time
import cProfile
import pstats
import importlib
import traceback
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make output dir
OUT = ROOT / "profiling"
OUT.mkdir(exist_ok=True)

# Prefer offscreen Qt for headless environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Try to set a DISPLAY for tkinter; if not available, legacy UI may fail and will be captured
os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))

# Monkeypatch paho.mqtt.client.Client to avoid network activity
try:
    import paho.mqtt.client as mqtt
    class DummyClient:
        def __init__(self, *args, **kwargs):
            self._callbacks = {}
        def connect(self, *a, **k):
            return 0
        def connect_async(self, *a, **k):
            return 0
        def loop_start(self):
            return None
        def loop_stop(self):
            return None
        def disconnect(self):
            return None
        def subscribe(self, *a, **k):
            return None
        def publish(self, *a, **k):
            class Info: rc = 0
            return Info()
        def reconnect(self):
            return None
    mqtt.Client = DummyClient
except Exception:
    pass

# Helper to profile a callable and dump stats
def profile_call(name, func, *a, **kw):
    prof = cProfile.Profile()
    prof.enable()
    start = time.perf_counter()
    result = None
    exc = None
    try:
        result = func(*a, **kw)
    except Exception as e:
        exc = e
        traceback.print_exc()
    end = time.perf_counter()
    prof.disable()
    prof_path = OUT / f"{name}.prof"
    prof.dump_stats(str(prof_path))
    stats_txt = OUT / f"{name}_stats.txt"
    with open(stats_txt, "w") as fh:
        ps = pstats.Stats(prof, stream=fh).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(60)
    return {
        "name": name,
        "elapsed_s": end - start,
        "exception": repr(exc) if exc else None,
        "prof_path": str(prof_path),
        "stats_txt": str(stats_txt),
    }

results = []

# 1) Profile imports
for mod_name, file in [("new_dashboard", "dashboard"), ("legacy_dashboard", "_legacy_dashboard_ref")]:
    # remove from sys.modules if present
    if file in sys.modules:
        del sys.modules[file]
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(file)
        importlib.reload(mod)
        exc = None
    except Exception as e:
        mod = None
        exc = e
        traceback.print_exc()
    t1 = time.perf_counter()
    results.append({
        "phase": "import",
        "module": file,
        "elapsed_s": t1 - t0,
        "exception": repr(exc) if exc else None,
    })

# 2) Profile UI instantiation and key functions
# For new dashboard (PySide6)
try:
    import dashboard as newd
    # Prevent MQTT thread from connecting by replacing MqttClient with a dummy
    class DummyMqttThread:
        def __init__(self, state, signals):
            self.event_queue = None
        def start(self):
            return None
        def stop(self):
            return None
    newd.MqttClient = DummyMqttThread

    # Ensure QApplication exists
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
    except Exception:
        app = None

    # Profile constructing DashboardMain
    def make_new():
        win = newd.DashboardMain()
        # call a few lightweight methods to exercise refresh paths
        try:
            win.periodic_refresh()
        except Exception:
            pass
        try:
            win.process_mqtt_events()
        except Exception:
            pass
        # Do not exec app loop; just close
        try:
            win.close()
        except Exception:
            pass
        return win

    results.append(profile_call("new_dashboard_init", make_new))
except Exception as e:
    results.append({"phase": "new_dashboard_setup", "exception": repr(e)})

# For legacy dashboard (customtkinter)
try:
    import _legacy_dashboard_ref as legacy
    # Monkeypatch mqtt.Client already done earlier; ensure any background connect/publish is safe

    def make_legacy():
        # Attempt to instantiate GovernanceApp. This may fail on headless systems.
        app_ref = None
        try:
            # customtkinter uses tkinter; create root via their app class
            app_ref = legacy.GovernanceApp()
            # exercise a few methods if present
            try:
                app_ref._flush_ui_refresh()
            except Exception:
                pass
            try:
                app_ref._queue_ui_refresh()
            except Exception:
                pass
            try:
                app_ref._save_model_state()
            except Exception:
                pass
            # try closing
            try:
                app_ref.destroy()
            except Exception:
                pass
        except Exception:
            # propagate to be captured by profiler wrapper
            raise
        return app_ref

    results.append(profile_call("legacy_dashboard_init", make_legacy))
except Exception as e:
    results.append({"phase": "legacy_dashboard_setup", "exception": repr(e)})

# 3) Profile a heavy pure-Python function: dataset summary if available
# Create minimal fake CSI data for a node
node_dir = ROOT / "csi_data" / "RACK_1" / "door_closed"
node_dir.mkdir(parents=True, exist_ok=True)
csv_path = node_dir / "session1.csv"
if not csv_path.exists():
    with open(csv_path, "w") as fh:
        fh.write("timestamp,label,global_sample_idx\n")
        for i in range(200):
            fh.write(f"{int(time.time())},{'door_closed'},{i}\n")

# call newd.state.build_dataset_summary if available
try:
    if 'newd' in globals() and hasattr(newd, 'DashboardMain'):
        st = newd.AppState()
        def call_build():
            return st.build_dataset_summary('RACK_1')
        results.append(profile_call('new_build_dataset_summary', call_build))
except Exception as e:
    results.append({"phase": "new_build_dataset_summary", "exception": repr(e)})

# call legacy build if present (search for similar method)
try:
    if 'legacy' in globals():
        # legacy's dataset summary may be a method on GovernanceApp or helper; try both
        if hasattr(legacy, 'GovernanceApp'):
            ga = legacy.GovernanceApp()
            if hasattr(ga, 'build_dataset_summary'):
                results.append(profile_call('legacy_build_dataset_summary', ga.build_dataset_summary, 'RACK_1'))
            else:
                # attempt to find function
                pass
except Exception as e:
    results.append({"phase": "legacy_build_dataset_summary", "exception": repr(e)})

# 4) Save a summary JSON
import json
with open(OUT / 'summary.json', 'w') as fh:
    json.dump(results, fh, indent=2)

print("Profiling complete. Results written to:")
for p in OUT.iterdir():
    print(" -", p)
