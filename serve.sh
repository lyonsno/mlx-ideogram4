#!/bin/bash
# Launch Ideogram4 NF4 Gradio server with auto-restart
# Designed to run as a LaunchAgent via phylax or directly.
#
# Usage:
#   ./serve.sh              # local only
#   ./serve.sh --share      # public Gradio URL
#   ./serve.sh --share --public  # public with rate limits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${HOME}/Library/Logs/ideogram4-nf4"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/serve.log"

cd "$SCRIPT_DIR"

echo "[$(date -Iseconds)] Starting Ideogram4 NF4 server" >> "$LOG_FILE"

# Use uv to handle deps automatically
exec uv run \
  --with safetensors --with huggingface_hub --with numpy \
  --with requests --with transformers --with pillow \
  --with tqdm --with mlx-lm --with gradio \
  python -u app.py "$@" \
  >> "$LOG_FILE" 2>&1
