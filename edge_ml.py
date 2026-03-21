#!/usr/bin/env python3
"""Notebook-aligned CSI model training entrypoint.

This file is the productionized form of `Edge_ML (1).ipynb` and keeps the same
modeling logic (variance-selected subcarriers, temporal variation feature,
StandardScaler, dense Keras classifier, and full-int8 TFLite export).
"""

import argparse
import glob
import importlib
import json
import os
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


STATE_TO_LABEL: Dict[str, int] = {
    "door_open": 0,
    "door_closed": 1,
    "person_standing": 2,
}
TARGET_NAMES = ["door_open", "door_closed", "person_standing"]

MAX_LOWER = 4
MAX_UPPER = 59
DC_NULL = 32
RAW_COLS = [f"SC_{i}" for i in range(MAX_LOWER, MAX_UPPER + 1) if i != DC_NULL]


def _import_tf():
    try:
        return importlib.import_module("tensorflow")
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow import failed. Install tensorflow in the Python environment used for training."
        ) from exc


def extract_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 5,
    session_offset: float = 0.0,
    calibration_val: float | None = None,
) -> np.ndarray:
    """Notebook-equivalent feature extraction."""
    raw_data = df[feature_cols].to_numpy(dtype=np.float32)
    calibrated_data = raw_data + float(session_offset)
    num_rows = calibrated_data.shape[0]
    all_features: List[np.ndarray] = []

    for i in range(num_rows):
        start = max(0, i - int(window_size) + 1)
        window = calibrated_data[start:i + 1, :]
        current_frame = calibrated_data[i, :]
        temp_std = np.std(window, axis=0)
        avg_variation = float(np.mean(temp_std))

        if calibration_val is not None:
            avg_variation = avg_variation / (float(calibration_val) + 1e-8)

        feat = np.concatenate([current_frame, np.asarray([avg_variation], dtype=np.float32)])
        all_features.append(feat)

    return np.asarray(all_features, dtype=np.float32)


def _representative_dataset_gen(x_calib: np.ndarray, max_samples: int = 100):
    if x_calib.size == 0:
        raise RuntimeError("Calibration dataset is empty; cannot quantize to int8")

    sample_count = min(int(max_samples), int(x_calib.shape[0]))
    for i in range(sample_count):
        yield [np.asarray(x_calib[i:i + 1], dtype=np.float32)]


def build_model(tf, input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _find_state_csvs(base_dir: str, node: str, state: str) -> List[str]:
    state_dir = os.path.join(base_dir, node, state)
    if not os.path.isdir(state_dir):
        return []

    csvs = sorted(glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True))
    return [
        path for path in csvs
        if os.path.isfile(path)
        and not os.path.basename(path).startswith("part_")
        and not os.path.basename(path).endswith("_manifest.csv")
    ]


def _load_state_df(base_dir: str, node: str, state: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in _find_state_csvs(base_dir, node, state):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable CSV: {path} ({exc})")
            continue

        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            print(f"[WARN] Skipping CSV missing required notebook columns: {path}")
            continue

        rows.append(df[RAW_COLS].copy())

    if not rows:
        return pd.DataFrame(columns=RAW_COLS)

    return pd.concat(rows, axis=0, ignore_index=True)


def _load_dataset(data_dir: str, node: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_open = _load_state_df(data_dir, node, "door_open")
    df_close = _load_state_df(data_dir, node, "door_closed")
    df_person = _load_state_df(data_dir, node, "person_standing")

    if df_open.empty or df_close.empty or df_person.empty:
        raise RuntimeError(
            "Training requires all 3 classes with data: door_open, door_closed, person_standing"
        )

    return df_open, df_close, df_person


def _select_feature_cols(df_open: pd.DataFrame, df_close: pd.DataFrame, df_person: pd.DataFrame, threshold: float) -> List[str]:
    df_temp = pd.concat([df_open[RAW_COLS], df_close[RAW_COLS], df_person[RAW_COLS]], axis=0, ignore_index=True)
    variances = df_temp.std()
    feature_cols = variances[variances > float(threshold)].index.tolist()
    if not feature_cols:
        raise RuntimeError(
            f"No features passed variance threshold={threshold}. Collect more diverse calibration data."
        )
    return feature_cols


def _save_scaler_params(
    path: str,
    scaler: StandardScaler,
    x_train_raw: np.ndarray,
    feature_cols: List[str],
    train_baseline: float,
    train_mean: float,
    cal_frames: int,
    feature_window: int,
) -> None:
    feature_columns = list(feature_cols) + ["AVG_VARIATION"]
    selected_subcarriers: List[int] = []
    for col in feature_cols:
        if col.startswith("SC_"):
            try:
                selected_subcarriers.append(int(col.split("_", 1)[1]))
            except ValueError:
                pass

    payload = {
        "schema_version": 2,
        "scaler_type": "standard",
        "mean": scaler.mean_.astype(float).tolist(),
        "std": scaler.scale_.astype(float).tolist(),
        "min": np.min(x_train_raw, axis=0).astype(float).tolist(),
        "max": np.max(x_train_raw, axis=0).astype(float).tolist(),
        "feature_columns": feature_columns,
        "selected_subcarriers": selected_subcarriers,
        "label_order": TARGET_NAMES,
        "notebook_alignment": {
            "calibration_frames": int(cal_frames),
            "extract_window_size": int(feature_window),
            "train_baseline": float(train_baseline),
            "train_mean": float(train_mean),
            "variation_feature": "mean(std(window, axis=0))",
            "use_session_offset": True,
            "session_offset_formula": "offset = train_mean - session_raw_mean",
        },
    }

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train notebook-aligned CSI model and export TFLite")
    parser.add_argument("--data-dir", required=True, help="Base csi_data directory")
    parser.add_argument("--node", required=True, help="Node id (e.g. RACK_1)")
    parser.add_argument("--output", required=True, help="Output model.tflite path")
    parser.add_argument("--scaler-output", required=True, help="Output scaler_params.json path")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--feature-window", type=int, default=5)
    parser.add_argument("--calibration-frames", type=int, default=20)
    parser.add_argument("--variance-threshold", type=float, default=0.1)

    # accepted for compatibility with previous dashboard invocation
    parser.add_argument("--notebook", default="")
    parser.add_argument("--notebook-model", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        tf = _import_tf()

        print(f"[INFO] Loading dataset node={args.node} from {args.data_dir}")
        df_open, df_close, df_person = _load_dataset(args.data_dir, args.node)

        feature_cols = _select_feature_cols(df_open, df_close, df_person, threshold=args.variance_threshold)

        df_open = df_open[feature_cols].copy()
        df_close = df_close[feature_cols].copy()
        df_person = df_person[feature_cols].copy()
        df_open["label"] = STATE_TO_LABEL["door_open"]
        df_close["label"] = STATE_TO_LABEL["door_closed"]
        df_person["label"] = STATE_TO_LABEL["person_standing"]
        df_combined = pd.concat([df_open, df_close, df_person], axis=0, ignore_index=True)

        print(f"[INFO] Dataset shape: {df_combined.shape}")
        print(f"[INFO] Label distribution: {df_combined['label'].value_counts().to_dict()}")
        print(f"[INFO] Selected subcarriers: {len(feature_cols)}")

        cal_frames = max(1, int(args.calibration_frames))
        feature_window = max(1, int(args.feature_window))

        open_calibration_seed = df_open.head(cal_frames)
        if open_calibration_seed.empty:
            raise RuntimeError("door_open calibration frames are empty; cannot compute train baseline")

        train_baseline = float(
            np.mean(
                extract_features(
                    open_calibration_seed,
                    feature_cols,
                    window_size=feature_window,
                    calibration_val=None,
                )[:, -1]
            )
        )

        x = extract_features(
            df_combined,
            feature_cols,
            window_size=feature_window,
            calibration_val=train_baseline,
        )
        y = df_combined["label"].to_numpy(dtype=np.int32)

        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(args.test_size),
            random_state=int(args.random_seed),
            stratify=y,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        print(f"[INFO] Training set shape: {x_train.shape}")
        print(f"[INFO] Testing set shape: {x_test.shape}")

        model = build_model(tf, x_train.shape[1])
        print("[INFO] Starting model training...")
        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=max(1, int(args.patience)),
                    restore_best_weights=True,
                )
            ],
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Test loss={test_loss:.5f}, acc={test_acc:.5f}")

        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        print("[INFO] Confusion Matrix:")
        print(cm)
        print("[INFO] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["open", "close", "person"], zero_division=0))

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset_gen(x_train, max_samples=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        in_dtype = interpreter.get_input_details()[0]["dtype"]
        out_dtype = interpreter.get_output_details()[0]["dtype"]
        if in_dtype != np.int8 or out_dtype != np.int8:
            raise RuntimeError("Converted model is not strict int8 I/O")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.scaler_output)), exist_ok=True)

        with open(args.output, "wb") as f:
            f.write(tflite_model)

        train_mean = float(df_combined[feature_cols].to_numpy(dtype=np.float32).mean())
        _save_scaler_params(
            args.scaler_output,
            scaler,
            x_train_raw,
            feature_cols,
            train_baseline=train_baseline,
            train_mean=train_mean,
            cal_frames=cal_frames,
            feature_window=feature_window,
        )

        print(f"[INFO] Wrote TFLite model: {args.output} ({os.path.getsize(args.output)} bytes)")
        print(f"[INFO] Wrote scaler params: {args.scaler_output}")
        return 0
    except Exception as exc:
        print(f"[ERROR] exception during training: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())