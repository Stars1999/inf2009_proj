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
from datetime import datetime


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
ALLOWED_SPLITS = {"train", "dev", "test"}

SESSION_COL_CANDIDATES = ["session_id", "collection_session", "session", "X-Session-ID"]
LABEL_COL_CANDIDATES = ["collection_label", "room_state", "state_name"]
SPLIT_COL_CANDIDATES = ["collection_split", "split_group", "dataset_split"]
CAMPAIGN_COL_CANDIDATES = ["campaign_id", "collection_campaign"]
RUN_COL_CANDIDATES = ["calibration_run_id", "run_id"]
REDUNDANCY_SIGNATURE_BIN = 16
REDUNDANCY_L1_THRESHOLD = 10


def _first_present_scalar(df: pd.DataFrame, candidates: List[str], fallback: str) -> str:
    for col in candidates:
        if col in df.columns:
            series = df[col].dropna().astype(str)
            if not series.empty:
                value = series.iloc[0].strip()
                if value:
                    return value
    return fallback


def _normalize_split_tag(value: str) -> str:
    value = str(value).strip().lower()
    return value if value in ALLOWED_SPLITS else ""


def _session_sort_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "sub_batch_idx",
        "row_in_sub_batch",
        "global_sample_idx",
        "row_in_file",
        "source_file",
    ]
    return [col for col in candidates if col in df.columns]


def _allocate_split_counts(n_sessions: int, train_frac: float, dev_frac: float, test_frac: float) -> Tuple[int, int, int]:
    if n_sessions < 3:
        raise RuntimeError(f"Need at least 3 sessions to create train/dev/test splits; found {n_sessions}")

    raw = np.array([train_frac, dev_frac, test_frac], dtype=float) * float(n_sessions)
    counts = np.floor(raw).astype(int)
    fractions = raw - counts
    minimums = np.array([1, 1, 1], dtype=int)

    for idx in range(3):
        if counts[idx] < minimums[idx]:
            counts[idx] = minimums[idx]

    while int(counts.sum()) > n_sessions:
        reducible = [i for i in range(3) if counts[i] > minimums[i]]
        if not reducible:
            break
        idx = max(reducible, key=lambda i: (counts[i] - minimums[i], fractions[i]))
        counts[idx] -= 1

    while int(counts.sum()) < n_sessions:
        idx = int(np.argmax(fractions))
        counts[idx] += 1
        fractions[idx] = -1

    return int(counts[0]), int(counts[1]), int(counts[2])


def _detect_redundant_rows(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    annotated = df.copy()
    annotated["is_redundant"] = False
    annotated["redundancy_reason"] = ""
    annotated["redundancy_signature"] = ""

    if annotated.empty:
        return annotated

    session_col = "split_key" if "split_key" in annotated.columns else "session_id"
    sort_cols = _session_sort_columns(annotated)

    for _, session_idx in annotated.groupby(session_col, sort=False).groups.items():
        session_df = annotated.loc[session_idx]
        if sort_cols:
            session_df = session_df.sort_values(by=sort_cols)

        seen_signatures = set()
        prev_vals = None

        for idx, row in session_df.iterrows():
            vals = row[feature_cols].to_numpy(dtype=np.int16)
            signature = tuple((vals // REDUNDANCY_SIGNATURE_BIN).tolist())
            annotated.at[idx, "redundancy_signature"] = str(signature)

            reason = ""
            if signature in seen_signatures:
                reason = "duplicate_signature"
            elif prev_vals is not None:
                delta = int(np.abs(vals - prev_vals).sum())
                if delta <= REDUNDANCY_L1_THRESHOLD:
                    reason = f"near_duplicate_l1_{delta}"

            if reason:
                annotated.at[idx, "is_redundant"] = True
                annotated.at[idx, "redundancy_reason"] = reason
            else:
                seen_signatures.add(signature)
            prev_vals = vals

    return annotated


def _build_session_assignments(df: pd.DataFrame, seed: int, train_frac: float, dev_frac: float, test_frac: float):
    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}
    details = {}

    for state in TARGET_NAMES:
        state_df = df[df["state_name"] == state]
        session_keys = sorted(state_df["split_key"].dropna().astype(str).unique().tolist())
        if len(session_keys) < 3:
            raise RuntimeError(f"Need at least 3 sessions for {state}; found {len(session_keys)}")

        rng.shuffle(session_keys)
        train_n, dev_n, test_n = _allocate_split_counts(len(session_keys), train_frac, dev_frac, test_frac)
        train_keys = session_keys[:train_n]
        dev_keys = session_keys[train_n:train_n + dev_n]
        test_keys = session_keys[train_n + dev_n:train_n + dev_n + test_n]

        details[state] = {
            "session_count": int(len(session_keys)),
            "train_sessions": train_keys,
            "dev_sessions": dev_keys,
            "test_sessions": test_keys,
            "train_count": int(len(train_keys)),
            "dev_count": int(len(dev_keys)),
            "test_count": int(len(test_keys)),
        }

        for key in train_keys:
            assignment[key] = "train"
        for key in dev_keys:
            assignment[key] = "dev"
        for key in test_keys:
            assignment[key] = "test"

    return assignment, details


def _rebalance_explicit_splits_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict], List[int]]:
    """Ensure the train split has at least one session for each class.

    When operators explicitly tag collections as train/dev, it's possible to
    accidentally place an entire class only in dev. Training cannot proceed if
    train is missing any target class, so we promote one session per missing
    class from dev/test into train.
    """
    adjusted = df.copy()
    moves: List[dict] = []
    required_labels = {0, 1, 2}

    train_mask = adjusted["split_group"] == "train"
    train_labels = set(adjusted.loc[train_mask, "label"].astype(int).tolist())
    missing_labels = sorted(required_labels - train_labels)

    for label in missing_labels:
        candidates = adjusted[
            (adjusted["label"].astype(int) == int(label))
            & (adjusted["split_group"].isin(["dev", "test"]))
        ]
        if candidates.empty:
            continue

        # Prefer moving from dev first, then test. Keep session integrity.
        candidate_sessions = (
            candidates[["split_key", "split_group"]]
            .dropna()
            .drop_duplicates()
            .sort_values(by=["split_group", "split_key"], key=lambda col: col.map({"dev": 0, "test": 1}).fillna(2) if col.name == "split_group" else col)
        )

        if not candidate_sessions.empty:
            chosen = candidate_sessions.iloc[0]
            chosen_split_key = str(chosen["split_key"])
            from_split = str(chosen["split_group"])
            move_mask = adjusted["split_key"].astype(str) == chosen_split_key
        else:
            # Fallback to row-level move if split_key is unavailable.
            if (candidates["split_group"] == "dev").any():
                from_split = "dev"
            else:
                from_split = "test"
            move_mask = (
                (adjusted["label"].astype(int) == int(label))
                & (adjusted["split_group"] == from_split)
            )
            chosen_split_key = ""

        moved_rows = int(move_mask.sum())
        if moved_rows <= 0:
            continue

        adjusted.loc[move_mask, "split_group"] = "train"
        moves.append(
            {
                "label": int(label),
                "label_name": TARGET_NAMES[int(label)] if 0 <= int(label) < len(TARGET_NAMES) else str(label),
                "from_split": from_split,
                "rows_moved": moved_rows,
                "split_key": chosen_split_key,
            }
        )

    final_train_labels = set(adjusted.loc[adjusted["split_group"] == "train", "label"].astype(int).tolist())
    still_missing = sorted(required_labels - final_train_labels)
    return adjusted, moves, still_missing


def _export_split_csvs(output_dir: str, df: pd.DataFrame, feature_cols: List[str]) -> None:
    split_root = os.path.join(output_dir, "dataset_splits")
    os.makedirs(split_root, exist_ok=True)

    export_cols = [
        "node_id",
        "collection_label",
        "state_name",
        "session_id",
        "split_key",
        "split_group",
        "sub_batch_idx",
        "row_in_sub_batch",
        "global_sample_idx",
        "upload_received_at",
        "payload_crc32",
        "idempotency_key",
        "is_redundant",
        "redundancy_reason",
        "redundancy_signature",
        "label",
    ]
    export_cols = [col for col in export_cols if col in df.columns]
    export_cols.extend([col for col in feature_cols if col in df.columns])

    for split_name in ("train", "dev", "test"):
        split_dir = os.path.join(split_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_df = df[df["split_group"] == split_name]
        for state in TARGET_NAMES:
            label_df = split_df[split_df["state_name"] == state].copy()
            if label_df.empty:
                continue
            out_path = os.path.join(split_dir, f"{state}.csv")
            label_df.to_csv(out_path, index=False, columns=export_cols)


def _build_dataset_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame, feature_cols: List[str], split_details) -> dict:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_count": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "overall": {
            "rows_raw": int(len(raw_df)),
            "rows_clean": int(len(clean_df)),
            "redundant_rows": int(raw_df["is_redundant"].sum()) if "is_redundant" in raw_df.columns else 0,
            "redundancy_rate": float(raw_df["is_redundant"].mean()) if len(raw_df) and "is_redundant" in raw_df.columns else 0.0,
        },
        "labels": {},
        "splits": {},
        "session_splits": split_details,
    }

    for state in TARGET_NAMES:
        raw_state = raw_df[raw_df["state_name"] == state]
        clean_state = clean_df[clean_df["state_name"] == state]
        means = clean_state[feature_cols].mean(numeric_only=True).fillna(0.0) if not clean_state.empty else pd.Series([0.0] * len(feature_cols), index=feature_cols)
        stds = clean_state[feature_cols].std(ddof=0, numeric_only=True).fillna(0.0) if not clean_state.empty else pd.Series([0.0] * len(feature_cols), index=feature_cols)
        variances = clean_state[feature_cols].var(ddof=0, numeric_only=True).fillna(0.0) if not clean_state.empty else pd.Series([0.0] * len(feature_cols), index=feature_cols)

        top_variance = [
            feature_cols[i]
            for i in np.argsort(variances.to_numpy())[-5:][::-1]
        ] if len(feature_cols) else []

        summary["labels"][state] = {
            "rows_raw": int(len(raw_state)),
            "rows_clean": int(len(clean_state)),
            "sessions_raw": int(raw_state["split_key"].nunique()) if "split_key" in raw_state.columns else 0,
            "sessions_clean": int(clean_state["split_key"].nunique()) if "split_key" in clean_state.columns else 0,
            "redundant_rows": int(raw_state["is_redundant"].sum()) if "is_redundant" in raw_state.columns else 0,
            "redundancy_rate": float(raw_state["is_redundant"].mean()) if len(raw_state) and "is_redundant" in raw_state.columns else 0.0,
            "feature_mean": [float(v) for v in means.round(3).tolist()],
            "feature_std": [float(v) for v in stds.round(3).tolist()],
            "top_variance_features": top_variance,
        }

    for split_name in ("train", "dev", "test"):
        split_df = clean_df[clean_df["split_group"] == split_name]
        per_label = {}
        for state in TARGET_NAMES:
            label_df = split_df[split_df["state_name"] == state]
            per_label[state] = {
                "rows": int(len(label_df)),
                "sessions": int(label_df["split_key"].nunique()) if "split_key" in label_df.columns else 0,
            }
        summary["splits"][split_name] = {
            "rows": int(len(split_df)),
            "sessions": int(split_df["split_key"].nunique()) if "split_key" in split_df.columns else 0,
            "per_label": per_label,
        }

    return summary


def find_state_csvs(base_dir: str, node: str, state: str) -> List[str]:
    """Find CSVs for one state, including nested session folders."""
    state_dir = os.path.join(base_dir, node, state)
    if not os.path.isdir(state_dir):
        return []
    # server output may include nested session folders, so recurse
    csvs = sorted(glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True))
    return [
        path for path in csvs
        if os.path.isfile(path)
        and not os.path.basename(path).startswith("part_")
        and not os.path.basename(path).endswith("_manifest.csv")
    ]


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
            fallback_session = os.path.splitext(os.path.basename(path))[0]
            session_id = _first_present_scalar(df, SESSION_COL_CANDIDATES, fallback_session)
            chunk["session_id"] = session_id
            chunk["state_name"] = state
            chunk["collection_label"] = _first_present_scalar(df, LABEL_COL_CANDIDATES, state)
            chunk["collection_split"] = _normalize_split_tag(_first_present_scalar(df, SPLIT_COL_CANDIDATES, ""))
            chunk["campaign_id"] = _first_present_scalar(df, CAMPAIGN_COL_CANDIDATES, session_id)
            chunk["calibration_run_id"] = _first_present_scalar(df, RUN_COL_CANDIDATES, session_id)
            chunk["source_file"] = os.path.basename(path)
            chunk["source_path"] = path
            chunk["row_in_file"] = np.arange(len(chunk), dtype=np.int32)
            if "node_id" in df.columns:
                chunk["node_id"] = _first_present_scalar(df, ["node_id"], node)
            else:
                chunk["node_id"] = node
            for extra_col in ["sub_batch_idx", "row_in_sub_batch", "global_sample_idx", "upload_received_at", "payload_crc32", "idempotency_key"]:
                if extra_col in df.columns:
                    chunk[extra_col] = df[extra_col].values
            chunk["split_key"] = chunk["state_name"].astype(str) + "|" + chunk["session_id"].astype(str)
            chunk["label"] = label
            frames.append(chunk)

    if not frames:
        raise RuntimeError(
            f"No valid calibration CSVs found under {os.path.join(data_dir, node)}"
        )

    dataset = pd.concat(frames, axis=0, ignore_index=True)

    labels_present = set(dataset["label"].astype(int).tolist())
    expected = {0, 1, 2}
    missing = sorted(expected - labels_present)
    if missing:
        missing_names = [TARGET_NAMES[i] for i in missing if 0 <= i < len(TARGET_NAMES)]
        raise RuntimeError(
            "Training requires all 3 classes (door_open, door_closed, person_standing). "
            f"Missing: {missing_names} (found labels: {sorted(labels_present)})"
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
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--split-test-fraction", type=float, default=0.15)
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
        annotated = _detect_redundant_rows(data, feature_cols.tolist())

        split_values = annotated["collection_split"].map(_normalize_split_tag) if "collection_split" in annotated.columns else pd.Series([""] * len(annotated))
        has_explicit_splits = split_values.ne("").any()
        if has_explicit_splits:
            annotated["split_group"] = split_values.replace("", "train")
            annotated, rebalance_moves, still_missing = _rebalance_explicit_splits_for_training(annotated)
            if rebalance_moves:
                for move in rebalance_moves:
                    print(
                        "[WARN] Rebalanced explicit splits: "
                        f"moved {move['rows_moved']} row(s) of {move['label_name']} "
                        f"from {move['from_split']} to train"
                    )
            if still_missing:
                missing_names = [TARGET_NAMES[i] for i in still_missing if 0 <= i < len(TARGET_NAMES)]
                raise RuntimeError(
                    "Train split is missing required labels even after rebalancing. "
                    f"Missing: {missing_names}"
                )
            split_details = {"explicit_rebalance": rebalance_moves}
            split_source = "explicit"
        else:
            split_assignments, split_details = _build_session_assignments(
                annotated,
                seed=args.random_seed,
                train_frac=args.train_fraction,
                dev_frac=args.dev_fraction,
                test_frac=args.split_test_fraction,
            )
            annotated["split_group"] = annotated["split_key"].map(split_assignments)
            if annotated["split_group"].isna().any():
                missing = sorted(annotated.loc[annotated["split_group"].isna(), "split_key"].astype(str).unique().tolist())
                raise RuntimeError(f"Missing split assignment for sessions: {missing[:10]}")
            split_source = "session"

        clean_data = annotated[~annotated["is_redundant"]].copy()
        if clean_data.empty:
            raise RuntimeError("All rows were flagged as redundant; cannot train a model")

        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        _export_split_csvs(output_dir, clean_data, feature_cols.tolist())

        summary = _build_dataset_summary(annotated, clean_data, feature_cols.tolist(), split_details)
        split_metadata = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "random_seed": args.random_seed,
            "fractions": {
                "train": args.train_fraction,
                "dev": args.dev_fraction,
                "test": args.split_test_fraction,
            },
            "label_names": TARGET_NAMES,
            "feature_columns": feature_cols.tolist(),
            "split_source": split_source,
            "session_assignments": split_details,
            "row_counts": {
                split_name: int((clean_data["split_group"] == split_name).sum())
                for split_name in ("train", "dev", "test")
            },
            "redundant_rows": int(annotated["is_redundant"].sum()),
        }

        with open(os.path.join(output_dir, "split_metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(split_metadata, fp, indent=2, sort_keys=True)
        with open(os.path.join(output_dir, "dataset_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, sort_keys=True)

        train_df = clean_data[clean_data["split_group"] == "train"].copy()
        dev_df = clean_data[clean_data["split_group"] == "dev"].copy()
        test_df = clean_data[clean_data["split_group"] == "test"].copy()

        # The dashboard currently only manages train/dev collection; test may be empty.
        # If test split is empty, reuse dev as a proxy for evaluation (no held-out split).
        if train_df.empty or dev_df.empty:
            raise RuntimeError(
                f"Train/dev splits must be non-empty after session assignment: train={len(train_df)}, dev={len(dev_df)}"
            )
        if test_df.empty:
            print("[WARN] Test split is empty; using dev split for evaluation.")
            test_df = dev_df.copy()

        x_train_raw = train_df[feature_cols].values
        y_train = np.asarray(train_df["label"].to_numpy(dtype=np.int32), dtype=np.int32)
        x_dev_raw = dev_df[feature_cols].values
        y_dev = np.asarray(dev_df["label"].to_numpy(dtype=np.int32), dtype=np.int32)
        x_test_raw = test_df[feature_cols].values
        y_test = np.asarray(test_df["label"].to_numpy(dtype=np.int32), dtype=np.int32)

        unique_labels, label_counts = np.unique(clean_data["label"].to_numpy(dtype=np.int32), return_counts=True)
        print(f"[INFO] Dataset shape: {clean_data[feature_cols].shape}; labels: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
        print(
            f"[INFO] Rows by split: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}; "
            f"redundant_rows={int(annotated['is_redundant'].sum())}"
        )

        train_labels = set(np.unique(y_train).tolist())
        dev_labels = set(np.unique(y_dev).tolist())
        test_labels = set(np.unique(y_test).tolist())

        # Train must include all labels; dev/test may be partial in small datasets.
        required_label_set = {0, 1, 2}
        if train_labels != required_label_set:
            raise RuntimeError(
                f"Training split must contain all labels; train={sorted(train_labels)}"
            )
        if dev_labels != required_label_set:
            print(f"[WARN] Dev split does not include all labels: dev={sorted(dev_labels)}")
        if test_df is not dev_df and test_labels != required_label_set:
            print(f"[WARN] Test split does not include all labels: test={sorted(test_labels)}")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_train_raw)
        x_train = scaler.transform(x_train_raw)
        x_dev = scaler.transform(x_dev_raw)
        x_test = scaler.transform(x_test_raw)

        min_vals = x_train_raw.min(axis=0)
        max_vals = x_train_raw.max(axis=0)
        mean = x_train_raw.mean(axis=0)
        std = x_train_raw.std(axis=0)
        std = np.where(std == 0, 1.0, std)

        print(f"[INFO] Train shape: {x_train.shape}; Dev shape: {x_dev.shape}; Test shape: {x_test.shape}")

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
            validation_data=(x_dev, y_dev),
            verbose=1,
            callbacks=callbacks,
        )

        dev_loss, dev_acc = model.evaluate(x_dev, y_dev, verbose=0)
        print(f"[INFO] Dev loss={dev_loss:.5f}, acc={dev_acc:.5f}")

        eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Test loss={eval_loss:.5f}, acc={eval_acc:.5f}")

        preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
        from sklearn.metrics import classification_report, confusion_matrix

        print("[INFO] Confusion Matrix:")
        print(confusion_matrix(y_test, preds, labels=[0, 1, 2]))
        print("[INFO] Classification Report:")
        print(classification_report(y_test, preds, labels=[0, 1, 2], target_names=TARGET_NAMES, zero_division=0))

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
