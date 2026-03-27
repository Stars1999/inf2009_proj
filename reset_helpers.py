"""Headless reset helpers for calibration and model artifacts.

Provides reset_node(node_id, ...) which removes CSI data and model artifacts
and triggers the ESP32 to reload its model state via MQTT (so the device drops
any local model). Designed to be imported and called without GUI dependencies.
"""

from __future__ import annotations
import os
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to load defaults from project modules when available
try:
    from shared_config import CSI_DATA_DIR
except Exception:
    CSI_DATA_DIR = os.path.join(os.path.dirname(__file__), "csi_data")

try:
    import push_model
    MODEL_STORE_DIR_DEFAULT = getattr(push_model, "MODEL_STORE_DIR", os.path.join(os.path.dirname(__file__), "model_store"))
except Exception:
    push_model = None
    MODEL_STORE_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "model_store")


def reset_node(node_id: str, csi_root: Optional[str] = None, model_root: Optional[str] = None, trigger_mqtt: bool = True) -> bool:
    """Remove CSI files and model artifacts for a node and optionally trigger MQTT.

    Returns True if operations completed without uncaught exceptions (best-effort).
    """
    ok = True
    try:
        base_dir = os.path.dirname(__file__)

        if csi_root is None:
            csi_root = CSI_DATA_DIR
        node_dir = os.path.join(csi_root, node_id)
        if os.path.isdir(node_dir):
            try:
                shutil.rmtree(node_dir)
                logger.info("Removed CSI data for %s at %s", node_id, node_dir)
            except Exception as e:
                logger.exception("Failed to remove CSI data %s: %s", node_dir, e)
                ok = False

        if model_root is None:
            model_root = MODEL_STORE_DIR_DEFAULT
        node_model_dir = os.path.join(model_root, node_id)
        if os.path.isdir(node_model_dir):
            try:
                # Remove known artifacts first
                tflite_path = os.path.join(node_model_dir, "model.tflite")
                if os.path.exists(tflite_path):
                    try:
                        os.remove(tflite_path)
                    except Exception:
                        pass
                scaler_path = os.path.join(node_model_dir, "scaler_params.json")
                if os.path.exists(scaler_path):
                    try:
                        os.remove(scaler_path)
                    except Exception:
                        pass
                # Finally remove directory if still present
                try:
                    shutil.rmtree(node_model_dir)
                except Exception:
                    pass
                logger.info("Removed model artifacts for %s at %s", node_id, node_model_dir)
            except Exception as e:
                logger.exception("Failed to remove model artifacts %s: %s", node_model_dir, e)
                ok = False

        if trigger_mqtt:
            try:
                if push_model is not None:
                    # Ask device to reload model state (device should drop model if none present)
                    push_model.trigger_model_load_mqtt(node_id)
                else:
                    logger.warning("push_model unavailable; cannot trigger MQTT model reload")
            except Exception as e:
                logger.exception("Failed to trigger model load MQTT for %s: %s", node_id, e)
                ok = False

    except Exception as e:
        logger.exception("Reset node failed: %s", e)
        ok = False

    return ok


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Headless reset helper')
    parser.add_argument('--node', required=True, help='Node id (e.g., RACK_1)')
    parser.add_argument('--csi-root', default=None, help='CSI data root')
    parser.add_argument('--model-root', default=None, help='Model store root')
    parser.add_argument('--no-mqtt', action='store_true', help='Do not trigger MQTT')
    args = parser.parse_args()
    ok = reset_node(args.node, csi_root=args.csi_root, model_root=args.model_root, trigger_mqtt=not args.no_mqtt)
    exit(0 if ok else 2
)