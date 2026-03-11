# inf2009_proj

This folder contains the Pi‑side Python services for the CSI project. After
migrating `server.py` from the `edge-esp32` repo, both the dashboard and the
HTTP ingest server live here.

## Setup

```bash
cd inf2009_proj
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install flask numpy pandas paho-mqtt customtkinter
```

No separate requirements file is used; the packages installed above are all that
are needed by `dashboard.py`, `server.py`, and the helper scripts.

## Running

Simply start the dashboard – the ingest server will be spawned automatically:

```bash
python dashboard.py
```

A background thread launches the Flask instance on port 5000 and connects to
`mosquitto` using the broker/port specified in the dashboard's configuration.

You can still run the server standalone for debugging using:

```bash
python server.py       # launches server only
```

## Notes

* Configuration (broker, port, CSI data directory) is stored in
  `config.json` and used by both components.
* CSI uploads are saved under `csi_data/<node>/<state>/<session>`.
* Trained models and scaler parameters live in `model_store/<node>`.

For ESP32 firmware instructions refer to the top‑level
`edge-esp32/README.md`.