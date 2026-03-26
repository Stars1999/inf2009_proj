#!/usr/bin/env python3
"""
Validate a TFLite model and scaler_params.json pair.

Usage:
    validate_deploy.py model.tflite scaler_params.json

Checks:
 - scaler JSON contains keys mean, std, feature_columns, label_order, and types.
 - If TFLite runtime or TensorFlow Lite available, load model and compare input tensor shape/dtype and output tensor size to scaler mean length and label_order length.
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a TFLite model and scaler_params.json pair."
    )
    parser.add_argument("model", help="Path to .tflite model file")
    parser.add_argument("scaler", help="Path to scaler_params.json")
    return parser.parse_args()


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_scaler(scaler):
    errors = []
    keys = ['mean','std','feature_columns','label_order']
    for k in keys:
        if k not in scaler:
            errors.append(f"Missing key '{k}' in scaler JSON")
    # if missing keys, return
    if errors:
        return errors
    mean = scaler['mean']
    std = scaler['std']
    feature_columns = scaler['feature_columns']
    label_order = scaler['label_order']
    if not isinstance(mean, list):
        errors.append("scaler['mean'] is not a list")
    else:
        if not all(is_number(x) for x in mean):
            errors.append("scaler['mean'] must be list of numbers")
    if not isinstance(std, list):
        errors.append("scaler['std'] is not a list")
    else:
        if not all(is_number(x) for x in std):
            errors.append("scaler['std'] must be list of numbers")
    if isinstance(mean, list) and isinstance(std, list) and len(mean) != len(std):
        errors.append(f"Length mismatch: mean({len(mean)}) != std({len(std)})")
    if not isinstance(feature_columns, list):
        errors.append("scaler['feature_columns'] is not a list")
    else:
        if not all(isinstance(x, str) for x in feature_columns):
            errors.append("scaler['feature_columns'] must be list of strings")
    if not isinstance(label_order, list):
        errors.append("scaler['label_order'] is not a list")
    else:
        if not all(isinstance(x, str) for x in label_order):
            errors.append("scaler['label_order'] must be list of strings")
    if isinstance(mean, list) and isinstance(feature_columns, list) and len(mean) != len(feature_columns):
        errors.append(f"Length mismatch: mean({len(mean)}) != feature_columns({len(feature_columns)})")
    return errors


def product(iterable):
    p = 1
    for x in iterable:
        p *= x
    return p


def validate_tflite_model(model_path, mean_len, label_len):
    model_errors = []
    Interpreter = None
    try:
        from tflite_runtime.interpreter import Interpreter
        runtime = 'tflite_runtime'
    except Exception:
        try:
            import tensorflow as _tf
            Interpreter = _tf.lite.Interpreter
            runtime = 'tensorflow.lite'
        except Exception:
            Interpreter = None
            runtime = None
    if Interpreter is None:
        print("Warning: tflite runtime not available (neither tflite_runtime nor tensorflow.lite). Skipping model checks.")
        return model_errors, False
    # Try to load model
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        model_errors.append(f"Failed to load TFLite model: {e}")
        return model_errors, True
    # Input details
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        model_errors.append(f"Failed to get tensor details: {e}")
        return model_errors, True
    if not input_details:
        model_errors.append("Model has no inputs")
        return model_errors, True
    # Use first input tensor
    inp = input_details[0]
    shape = list(inp.get('shape', []))
    dtype = inp.get('dtype', None)
    # convert dims to ints safely
    dims = []
    dynamic = False
    for d in shape:
        try:
            di = int(d)
            dims.append(di)
            if di <= 0:
                dynamic = True
        except Exception:
            dynamic = True
            dims.append(-1)
    if dynamic:
        print("Model input has dynamic dimensions; skipping size comparison.")
    else:
        if len(dims) == 1:
            input_feature_count = dims[0]
        else:
            if dims[0] == 1:
                input_feature_count = product(dims[1:])
            else:
                input_feature_count = product(dims)
        if input_feature_count != mean_len:
            model_errors.append(f"Input feature count mismatch: model expects {input_feature_count}, scaler mean length {mean_len}")
    # dtype check
    dtype_name = getattr(dtype, 'name', None)
    if dtype_name is None:
        dtype_name = str(dtype).lower()
    else:
        dtype_name = str(dtype_name).lower()
    if 'float' not in dtype_name:
        model_errors.append(f"Input tensor dtype is {dtype_name}; expected a float dtype compatible with scaler (float32/float16).")
    # Output checks
    if not output_details:
        model_errors.append("Model has no outputs")
        return model_errors, True
    out = output_details[0]
    out_shape = list(out.get('shape', []))
    out_dims = []
    out_dynamic = False
    for d in out_shape:
        try:
            di = int(d)
            out_dims.append(di)
            if di <= 0:
                out_dynamic = True
        except Exception:
            out_dynamic = True
            out_dims.append(-1)
    if out_dynamic:
        print("Model output has dynamic dimensions; skipping output size comparison.")
    else:
        if len(out_dims) == 1:
            out_size = out_dims[0]
        else:
            if out_dims[0] == 1:
                out_size = product(out_dims[1:])
            else:
                out_size = product(out_dims)
        if out_size != label_len:
            model_errors.append(f"Output size mismatch: model output size {out_size}, scaler label_order length {label_len}")
    return model_errors, True


def main():
    args = parse_args()
    exit_code = 0
    # JSON checks
    try:
        scaler = load_json(args.scaler)
    except Exception as e:
        print(f"Failed to load scaler JSON '{args.scaler}': {e}", file=sys.stderr)
        sys.exit(2)
    json_errors = validate_scaler(scaler)
    if json_errors:
        print("Scaler JSON validation errors:")
        for e in json_errors:
            print(" -", e)
        exit_code = 3
    else:
        print("Scaler JSON validation: OK")
    mean_len = len(scaler.get('mean', [])) if isinstance(scaler.get('mean', []), list) else 0
    label_len = len(scaler.get('label_order', [])) if isinstance(scaler.get('label_order', []), list) else 0
    # Model checks
    model_errors = []
    model_checks_performed = False
    model_errors, model_checks_performed = validate_tflite_model(args.model, mean_len, label_len)
    if model_checks_performed:
        if model_errors:
            print("Model validation errors:")
            for e in model_errors:
                print(" -", e)
            exit_code = exit_code or 4
        else:
            print("Model validation: OK")
    # Final
    if exit_code != 0:
        sys.exit(exit_code)
    print("Validation successful")
    sys.exit(0)


if __name__ == '__main__':
    main()
