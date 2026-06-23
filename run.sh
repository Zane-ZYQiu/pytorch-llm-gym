#!/usr/bin/env bash
# Launch the PyTorch LLM Gym web app.
set -e
cd "$(dirname "$0")"
exec python web/app.py
