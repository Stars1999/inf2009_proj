#!/usr/bin/env python3
"""Train CSI model from calibration CSVs and export TFLite + scaler params.

This script is adapted from `Edge_ML.ipynb` and is intended to be called by
`dashboard.py`.
"""

import argparse
import glob
import json
import os
import sys
import importlib
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _import_tf():
    try:
        tf = importlib.import_module("tensorflow")
        return tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow import failed. Install tensorflow in the Python environment "
            "used by dashboard/train_model.py"
        ) from exc


# Notebook-compatible feature layout
FEATURE_COLS = [f"SC_{i}" for i in range(4, 60) if i != 32]

# Keep class indices compatible with notebook semantics:
# open=0, close=1, person=2
STATE_TO_LABEL: Dict[str, int] = {
    "door_open": 0,
    "door_closed": 1,
    "person_standing": 2,
}

TARGET_NAMES = ["door_open", "door_closed", "person_standing"]


def find_state_csvs(base_dir: str, node: str, state: str) -> List[str]:
    """Find CSVs for one state, including nested session folders."""
    state_dir = os.path.join(base_dir, node, state)
    if not os.path.isdir(state_dir):
        return []
    # server output may include nested session folders, so recurse
    return sorted(glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True))


def load_dataset(data_dir: str, node: str) -> Tuple[pd.DataFrame, np.ndarray]:
    frames: List[pd.DataFrame] = []

    for state, label in STATE_TO_LABEL.items():
        csv_paths = find_state_csvs(data_dir, node, state)
        if not csv_paths:
            continue

        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Skipping unreadable CSV: {path} ({exc})")
                continue

            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"[WARN] Skipping CSV missing features: {path}")
                continue

            chunk = df[FEATURE_COLS].copy()
            chunk["label"] = label
            frames.append(chunk)

    if not frames:
        raise RuntimeError(
            f"No valid calibration CSVs found under {os.path.join(data_dir, node)}"
        )

    dataset = pd.concat(frames, axis=0, ignore_index=True)

    labels_present = set(dataset["label"].astype(int).tolist())
    if labels_present != {0, 1, 2}:
        raise RuntimeError(
            "Training requires all 3 classes (door_open, door_closed, person_standing). "
            f"Found labels: {sorted(labels_present)}"
        )

    return dataset, np.array(FEATURE_COLS)


def build_model(tf, input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_scaler_params(
    path: str,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    payload = {
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSI model and export TFLite")
    parser.add_argument("--data-dir", required=True, help="Base csi_data directory")
    parser.add_argument("--node", required=True, help="Node id (e.g. RACK_1)")
    parser.add_argument("--output", required=True, help="Output .tflite path")
    parser.add_argument("--scaler-output", required=True, help="Output scaler_params.json path")

    # accepted for dashboard compatibility (informational only)
    parser.add_argument("--notebook", default="", help="Reference notebook path")
    parser.add_argument("--notebook-model", default="", help="Reference notebook model path")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--model-size-warn-bytes", type=int, default=100 * 1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        tf = _import_tf()

        if args.notebook:
            print(f"[INFO] Notebook reference: {args.notebook}")

        print(f"[INFO] Loading data for node={args.node} from {args.data_dir}")
        data, feature_cols = load_dataset(args.data_dir, args.node)

        x = data[feature_cols].values
        y = np.asarray(data["label"].to_numpy(dtype=np.int32), dtype=np.int32)

        unique_labels, label_counts = np.unique(y, return_counts=True)
        print(f"[INFO] Dataset shape: {x.shape}; labels: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")

        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=y,
        )

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_train_raw)
        x_train = scaler.transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        min_vals = x_train_raw.min(axis=0)
        max_vals = x_train_raw.max(axis=0)
        mean = x_train_raw.mean(axis=0)
        std = x_train_raw.std(axis=0)
        std = np.where(std == 0, 1.0, std)

        print(f"[INFO] Train shape: {x_train.shape}; Test shape: {x_test.shape}")

        # sanity: ensure we aren't training on an absurdly small dataset
        if x_train.shape[0] < 10:
            raise RuntimeError("Dataset too small for reliable training; check calibration data")

        model = build_model(tf, input_dim=x_train.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ]

        print("[INFO] Starting model training...")
        model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            verbose=1,
            callbacks=callbacks,
        )

        eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Test loss={eval_loss:.5f}, acc={eval_acc:.5f}")

        preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
        from sklearn.metrics import classification_report, confusion_matrix

        print("[INFO] Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("[INFO] Classification Report:")
        print(classification_report(y_test, preds, target_names=TARGET_NAMES))

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.scaler_output)), exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(args.output, "wb") as fp:
            fp.write(tflite_model)

        size_bytes = os.path.getsize(args.output)
        print(f"[INFO] Wrote TFLite model: {args.output} ({size_bytes} bytes)")
        if size_bytes > args.model_size_warn_bytes:
            print(
                f"[WARN] Model size {size_bytes} exceeds recommended "
                f"{args.model_size_warn_bytes} bytes for ESP32-C3 memory budgets"
            )

        save_scaler_params(args.scaler_output, min_vals, max_vals, mean, std)
        print(f"[INFO] Wrote scaler parameters: {args.scaler_output}")
        return 0
    except Exception as exc:
        print(f"[ERROR] exception during training: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
