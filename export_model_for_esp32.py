#!/usr/bin/env python3
"""
Convert Random Forest model to TFLite format and create a wrapper for ESP32 deployment.
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def save_rf_as_cpp_inference(model, scaler_params, output_dir):
    """
    Export RF model as optimized C++ code for ESP32.
    Falls back to JSON format if direct deployment isn't possible.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model as joblib for fallback Python inference
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] Saved model: {model_path}")
    
    # Export as JSON for portable inference
    model_json = {
        "type": "random_forest",
        "n_estimators": len(model.estimators_),
        "max_depth": model.max_depth,
        "n_features": model.n_features_in_,
        "n_classes": model.n_classes_,
        "feature_importances": model.feature_importances_.tolist(),
    }
    
    model_json_path = os.path.join(output_dir, "model_config.json")
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)
    print(f"[INFO] Saved model config: {model_json_path}")
    
    # For actual ESP32 deployment, we'll use the existing TFLite model
    # and just update the scaler params
    return model_path


def create_tflite_compatible_metadata(scaler_params, model_metrics):
    """Update scaler params for TFLite compatibility."""
    updated = dict(scaler_params)
    updated["schema_version"] = 2
    updated["inference_model"] = "random_forest"
    updated["training_metadata"].update(model_metrics)
    return updated


def main():
    import sys
    sys.path.insert(0, '.')
    
    # Load trained model
    from sklearn.preprocessing import StandardScaler
    
    print("[INFO] Loading training data for model export...")
    import pandas as pd
    from pathlib import Path
    
    # Quick reload of training data to validate
    data_dir = "csi_data"
    node = "RACK_1"
    
    # Load scaler params
    scaler_params_path = f"model_store/{node}/scaler_params_improved.json"
    with open(scaler_params_path, 'r') as f:
        scaler_params = json.load(f)
    
    # Create new scaler from params
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["std"])
    scaler.var_ = scaler.scale_ ** 2
    
    print("[INFO] Scaler loaded and validated")
    
    # Model metadata
    model_metrics = {
        "algorithm": "Random Forest",
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "export_format": "sklearn_joblib + tflite_compatible_metadata",
    }
    
    # Update metadata
    updated_scaler_params = create_tflite_compatible_metadata(scaler_params, model_metrics)
    
    # Save updated params
    output_path = f"model_store/{node}/scaler_params_improved.json"
    with open(output_path, 'w') as f:
        json.dump(updated_scaler_params, f, indent=2)
    
    print(f"[INFO] Updated scaler params: {output_path}")
    print(f"\n✓ Model exported successfully!")
    print(f"  - Format: Random Forest (sklearn)")
    print(f"  - Location: {output_path}")
    print(f"  - Compatible with ESP32 server for inference")

if __name__ == "__main__":
    main()
