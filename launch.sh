#!/bin/bash

# Path to your virtual environment (adjust if it's elsewhere)
VENV_PATH="./venv/bin/activate"

# Launch Script 1
lxterminal -e "bash -c 'source $VENV_PATH && python3 dashboard.py; exec bash'" &

# Launch Script 2
lxterminal -e "bash -c 'source $VENV_PATH && python3 server.py; exec bash'" &

# Launch Script 3
lxterminal -e "bash -c 'source $VENV_PATH && python3 broadcast_generator.py; exec bash'"
