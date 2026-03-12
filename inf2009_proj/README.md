### Training models

A Jupyter notebook `Edge_ML.ipynb` lives in this directory and demonstrates
how to load the CSI CSV files, fit a simple Keras network, evaluate it
and convert it to TensorFlow Lite.  That notebook is used as the reference
for the `train_model.py` command‑line helper, which the dashboard invokes
via `subprocess`.

To run training manually use:

```bash
python train_model.py \
    --data-dir csi_data \
    --node RACK_1 \
    --output /path/to/model.tflite \
    --scaler-output /path/to/scaler_params.json
```

The script will:

1. concatenate all CSVs under `data_dir/node/<state>` and assign numeric
   labels (0=door_closed, 1=door_open, 2=person_standing),
2. split into train/test sets and scale features to [0,1],
3. train a small dense model (64→32→3 units) and convert it to TFLite,
4. write the model and scaler JSON in the format expected by the
   ESP32 firmware (`{"mean": [...], "std": [...]}`),
5. print an evaluation report and warn if the model exceeds ~100 KB.

**Hardware notes**

- The ESP32-C3 has limited RAM; aim for models under ~100 KB when converted
  to TFLite and no more than ~60 input features (currently SC_4..SC_59).
- The firmware currently ignores the scaler `mean`/`std`, but these values
  are required by the download protocol and may be used in future updates.
