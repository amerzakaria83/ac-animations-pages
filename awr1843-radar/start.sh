#!/usr/bin/env sh
# Set up (first run only) and launch AWR1843 Radar Studio on Linux / macOS.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "Creating Python environment, this happens only once..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m awr1843 gui "$@"
