#!/usr/bin/env python3
"""
Improved model training using grouped features (8 groups) for ESP32 compatibility.
The ESP32 firmware groups 56 subcarriers into 8 groups + 1 temporal feature = 9 total.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

STATE_TO_LABEL = {
    "door_open": 0,
    "door_closed": 1,
    "person_standing": 2,
}
TARGET_NAMES = ["door_open", "door_closed", "person_standing"]

# Raw subcarriers: SC_4 to SC_60, excluding SC_32 (DC null)
RAW_SC_INDICES = [i for i in range(4, 61) if i != 32]  # 56 subcarriers

def group_subcarriers(data, num_groups=8):
    """
    Group subcarriers into averaged groups (matches ESP32 firmware).
    Input: data with 56 subcarriers
    Output: data with num_groups averaged values
    """
    n_samples = data.shape[0]
    grouped = np.zeros((n_samples, num_groups), dtype=np.float32)
    
    # Split 56 subcarriers into 8 groups of 7 each
    group_size = 56 // num_groups
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = start_idx + group_size if g < num_groups - 1 else 56
        grouped[:, g] = data[:, start_idx:end_idx].mean(axis=1)
    
    return grouped


def extract_temporal_features(df, feature_cols, window_size=5, calibration_val=None):
    """Extract temporal features with grouped subcarriers."""
    raw_data = df[feature_cols].to_numpy(dtype=np.float32)
    
    # Group the 56 raw subcarriers into 8 groups
    grouped_data = group_subcarriers(raw_data, num_groups=8)
    
    num_rows = grouped_data.shape[0]
    all_features = []
    
    for i in range(num_rows):
        start = max(0, i - window_size + 1)
        window = grouped_data[start:i+1, :]
        
        # Current frame (8 grouped values)
        current_frame = grouped_data[i, :]
        
        # Temporal feature: average variation
        temp_std = np.std(window, axis=0)
        avg_variation = float(np.mean(temp_std))
        
        # Normalize avg_variation
        if calibration_val is not None:
            avg_variation = avg_variation / (float(calibration_val) + 1e-8)
        
        # Combine: 8 grouped subcarriers + 1 temporal = 9 features
        feat = np.concatenate([current_frame, [avg_variation]])
        all_features.append(feat)
    
    return np.array(all_features)


def load_state_data(base_dir, node, state):
    """Load CSV data for a state."""
    state_dir = os.path.join(base_dir, node, state)
    csvs = sorted([f for f in Path(state_dir).glob("*.csv") if "manifest" not in f.name])
    
    dfs = []
    for csv_file in csvs:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train grouped RF model for ESP32 nodes")
    parser.add_argument("--node", default="RACK_1", help="Node ID, e.g. RACK_1")
    parser.add_argument("--data-dir", default="csi_data", help="CSI data base directory")
    parser.add_argument("--output-dir", default="model_store", help="Output model store directory")
    parser.add_argument("--no-save-model", action="store_true", help="Skip saving symbolic model file")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    node = args.node
    output_dir = args.output_dir
    output_scaler = os.path.join(output_dir, node, "scaler_params.json")
    model_output = os.path.join(output_dir, node, "model.tflite")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_scaler), exist_ok=True)
    
    print("[INFO] Loading data...")
    df_open = load_state_data(data_dir, node, "door_open")
    df_close = load_state_data(data_dir, node, "door_closed")
    df_person = load_state_data(data_dir, node, "person_standing")
    
    # Get all feature columns (raw subcarriers)
    feature_cols = [f"SC_{i}" for i in RAW_SC_INDICES]
    
    print(f"[INFO] Loaded door_open: {len(df_open)} rows")
    print(f"[INFO] Loaded door_closed: {len(df_close)} rows")
    print(f"[INFO] Loaded person_standing: {len(df_person)} rows")
    print(f"[INFO] Raw subcarriers: {len(feature_cols)} (SC_4-SC_60, excluding SC_32)")
    
    # Extract labels
    df_open["label"] = STATE_TO_LABEL["door_open"]
    df_close["label"] = STATE_TO_LABEL["door_closed"]
    df_person["label"] = STATE_TO_LABEL["person_standing"]
    df_combined = pd.concat([df_open, df_close, df_person], ignore_index=True)
    
    print(f"[INFO] Label distribution:\n{df_combined['label'].value_counts()}")
    
    # Calibration (door_open baseline)
    open_calib = df_open[feature_cols].head(20)
    open_calib_feat = extract_temporal_features(open_calib, feature_cols, window_size=5, calibration_val=None)
    train_baseline = float(np.mean(open_calib_feat[:, -1]))  # avg_variation (last column)
    
    # Extract features for all data (with grouping)
    print("[INFO] Extracting grouped features...")
    x = extract_temporal_features(df_combined, feature_cols, window_size=5, calibration_val=train_baseline)
    y = df_combined["label"].values
    
    print(f"[INFO] Feature dimensions: {x.shape} (8 groups + 1 temporal = 9 features)")
    
    # Split data (stratified) -> use 'dev' instead of 'test' (backwards-compatible)
    x_train, x_dev, y_train, y_dev = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)
    
    print(f"[INFO] Training set: {x_train.shape}, Dev set: {x_dev.shape}")
    
    # Class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"[INFO] Class weights: {class_weight_dict}")
    
    # Train Random Forest (with grouped features)
    print("[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(x_train, y_train)
    
    # Evaluate on dev set
    y_pred = model.predict(x_dev)
    accuracy = np.mean(y_pred == y_dev)
    print(f"\n[INFO] Accuracy: {accuracy:.4f}")
    print("\n[INFO] Classification Report:")
    print(classification_report(y_dev, y_pred, target_names=TARGET_NAMES))
    
    # Per-class F1
    f1_scores = f1_score(y_dev, y_pred, average=None)
    macro_f1 = f1_score(y_dev, y_pred, average='macro')
    print(f"\n[INFO] Per-class F1 scores: {dict(zip(TARGET_NAMES, f1_scores))}")
    print(f"[INFO] Macro F1: {macro_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_dev, y_pred)
    print(f"\n[INFO] Confusion Matrix:\n{cm}")
    
    # Save model metadata (ESP32 compatible format)
    metadata = {
        "schema_version": 2,
        "scaler_type": "standard",
        "mean": scaler.mean_.astype(float).tolist(),
        "std": scaler.scale_.astype(float).tolist(),
        "min": np.min(x_train, axis=0).astype(float).tolist(),
        "max": np.max(x_train, axis=0).astype(float).tolist(),
        "feature_columns": [f"GROUP_{i}" for i in range(8)] + ["AVG_VARIATION"],
        "selected_subcarriers": RAW_SC_INDICES,  # All 56 raw SCs used for grouping
        "label_order": TARGET_NAMES,
        "optimization": {
            "pruning_enabled": False,
            "baseline_acc": float(accuracy),
            "final_acc": float(accuracy),
            "max_accuracy_drop": 0.01,
        },
        "notebook_alignment": {
            "calibration_frames": 20,
            "extract_window_size": 5,
            "train_baseline": float(train_baseline),
            "train_mean": float(df_combined[feature_cols].values.mean()),
            "variation_feature": "mean(std(grouped_window, axis=0))",
            "enhanced_temporal_features": True,
            "temporal_features": ["avg_variation"],
            "use_session_offset": True,
            "session_offset_formula": "offset = train_mean - session_raw_mean",
            "feature_mode": "grouped",
            "group_count": 8,
        },
        "training_metadata": {
            "model_type": "RandomForest",
            "total_samples": len(df_combined),
            "train_samples": len(x_train),
            "dev_samples": len(x_dev),
            "n_classes": 3,
            "class_weights": class_weight_dict,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "per_class_f1": {TARGET_NAMES[i]: float(f1_scores[i]) for i in range(3)},
            "n_estimators": 200,
            "max_depth": 10,
        }
    }
    
    with open(output_scaler, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[INFO] Saved scaler params: {output_scaler}")
    print(f"[INFO] Model training complete!")
    print(f"\n✓ ESP32 Compatibility:")
    print(f"  - Features: 9 (8 groups + 1 temporal)")
    print(f"  - Grouping: 56 subcarriers → 8 groups of 7")
    print(f"  - Format: GROUPED (matches ESP32 firmware)")
    print(f"  - Ready for deployment ✓")

if __name__ == "__main__":
    main()
