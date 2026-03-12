#!/usr/bin/env python3
"""Train a CSI classification model and export TFLite/scaler artifacts.

This script is adapted from the `Edge_ML.ipynb` notebook in the repository.
It reads all CSV files in the specified data directory for a given node,
labels them according to calibration state (door_closed, door_open,
person_standing), trains a small Keras model, converts it to TFLite, and
writes out the scaler parameters as JSON.  The format of the scaler JSON
matches the expectations of the ESP32 firmware:

    {"mean": [...], "std": [...]}  # arrays of floats, one per feature

Usage example:

    python train_model.py \
        --data-dir csi_data \
        --node RACK_1 \
        --output /tmp/model.tflite \
        --scaler-output /tmp/scaler_params.json

The script also accepts optional hyperparameters and will print warnings if
the generated model is too large for typical ESP32 memory (100 KB heuristic).
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

# silence excessive TF logging
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CALIB_STATES = ["door_closed", "door_open", "person_standing"]


def collect_data(data_dir: str, node: str):
    """Load CSV files for each calibration state and return a combined DataFrame.

    The notebook used a hardcoded list of feature columns (SC_4..SC_59, skip
    SC_32).  We replicate that here so the ordering is consistent with
    previously‑trained models.
    """
    feature_cols = [f"SC_{i}" for i in range(4, 60) if i != 32]
    dfs = []
    for idx, state in enumerate(CALIB_STATES):
        state_dir = os.path.join(data_dir, node, state)
        if not os.path.isdir(state_dir):
            continue
        files = glob.glob(os.path.join(state_dir, "*.csv"))
        for fp in files:
            df = pd.read_csv(fp)
            # keep only the predefined feature columns if they exist
            df = df.reindex(columns=feature_cols)
            df = df.copy()
            df['label'] = idx
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No data files found for node {node} in {data_dir}")
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    return combined, feature_cols


def build_and_train(X_train, y_train, epochs=100, batch_size=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(CALIB_STATES), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return model, history


def save_scaler_params(scaler: MinMaxScaler, output_path: str):
    # compute mean/std of the scaled training data
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        # inverse transform to get raw mean/std? easiest: compute from stored data
        # scaler.transform(X) = (X - min) * scale.  We'll compute mean and std using
        # the scaler parameters themselves; although firmware doesn't currently use
        # them, we store mean=0,std=1 as a fallback if unavailable.
        mean = (scaler.data_min_ + scaler.data_max_) / 2.0
        std = (scaler.data_max_ - scaler.data_min_) / 2.0
    else:
        mean = np.zeros(len(scaler.scale_))
        std = np.ones(len(scaler.scale_))

    obj = {"mean": mean.tolist(), "std": std.tolist()}
    with open(output_path, 'w') as f:
        json.dump(obj, f)


def parse_args():
    p = argparse.ArgumentParser(description="Train CSI classification model")
    p.add_argument("--data-dir", required=True,
                   help="Base CSI data directory (contains <node>/<state> folders)")
    p.add_argument("--node", required=True, help="Node identifier, e.g. RACK_1")
    p.add_argument("--output", required=True, help="Path to write TFLite model")
    p.add_argument("--scaler-output", required=True,
                   help="Path to write scaler parameter JSON")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default 100)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for training (default 32)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading data for node {args.node} from {args.data_dir}...")
    df, feat_cols = collect_data(args.data_dir, args.node)
    X = df[feat_cols].values
    y = df['label'].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("Training model...")
    model, history = build_and_train(X_train, y_train,
                                     epochs=args.epochs,
                                     batch_size=args.batch_size)

    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(args.output, 'wb') as f:
        f.write(tflite_model)

    print(f"Model written to {args.output} ({os.path.getsize(args.output)} bytes)")
    if os.path.getsize(args.output) > 100 * 1024:
        print("WARNING: model size >100KB, may not fit in ESP32 RAM")

    save_scaler_params(scaler, args.scaler_output)
    print(f"Scaler parameters written to {args.scaler_output}")

    # simple evaluation on test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    from sklearn.metrics import classification_report
    print("\nEvaluation on test set:")
    print(classification_report(y_test, y_pred, target_names=CALIB_STATES))


if __name__ == '__main__':
    main()
