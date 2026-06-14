#!/bin/bash
# Launch Ideogram4 NF4 Gradio server with auto-restart
# Designed to run as a LaunchAgent via phylax or directly.
#
# Usage:
#   ./serve.sh              # local only
#   ./serve.sh --share      # public Gradio URL
#   ./serve.sh --share --public  # public with rate limits
#   NGROK_DOMAIN=name.ngrok-free.dev ./serve.sh --public --tunnel ngrok

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${HOME}/Library/Logs/ideogram4-nf4"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/serve.log"
TUNNEL_LOG_FILE="${LOG_DIR}/tunnel.log"

cd "$SCRIPT_DIR"

echo "[$(date -Iseconds)] Starting Ideogram4 NF4 server" >> "$LOG_FILE"

TUNNEL=""
APP_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tunnel)
      if [[ $# -lt 2 ]]; then
        echo "--tunnel requires a value, currently only: ngrok" >&2
        exit 2
      fi
      TUNNEL="$2"
      shift 2
      ;;
    --tunnel=*)
      TUNNEL="${1#--tunnel=}"
      shift
      ;;
    *)
      APP_ARGS+=("$1")
      shift
      ;;
  esac
done

APP_CMD=(
  uv run
  --with safetensors --with huggingface_hub --with numpy
  --with requests --with transformers --with pillow
  --with tqdm --with mlx-lm --with mlx-vlm --with gradio
  --with "mlx @ git+https://github.com/lyonsno/mlx.git@nf4"
  python -u app.py
)

if [[ -z "$TUNNEL" ]]; then
  exec "${APP_CMD[@]}" "${APP_ARGS[@]}" >> "$LOG_FILE" 2>&1
fi

if [[ "$TUNNEL" != "ngrok" ]]; then
  echo "Unsupported tunnel '$TUNNEL'; currently only 'ngrok' is implemented." >&2
  exit 2
fi

if [[ -z "${NGROK_DOMAIN:-}" ]]; then
  echo "NGROK_DOMAIN is required for --tunnel ngrok." >&2
  exit 2
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok is not installed or not on PATH." >&2
  exit 2
fi

PORT="${GRADIO_SERVER_PORT:-7860}"
"${APP_CMD[@]}" "${APP_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
APP_PID=$!
TUNNEL_PID=""

cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" >/dev/null 2>&1 || true
  fi
  kill "$APP_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in {1..60}; do
  if ! kill -0 "$APP_PID" >/dev/null 2>&1; then
    echo "Gradio app exited before tunnel startup; see $LOG_FILE." >&2
    exit 1
  fi
  if python - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = sys.argv[1]
urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=1).close()
PY
  then
    break
  fi
  sleep 1
done

if ! python - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = sys.argv[1]
urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=1).close()
PY
then
  echo "Gradio app did not answer on 127.0.0.1:$PORT within 60s; see $LOG_FILE." >&2
  exit 1
fi

echo "[$(date -Iseconds)] Starting ngrok tunnel https://${NGROK_DOMAIN} -> localhost:${PORT}" >> "$TUNNEL_LOG_FILE"
ngrok http "$PORT" --url "https://${NGROK_DOMAIN}" >> "$TUNNEL_LOG_FILE" 2>&1 &
TUNNEL_PID=$!

while kill -0 "$APP_PID" >/dev/null 2>&1 && kill -0 "$TUNNEL_PID" >/dev/null 2>&1; do
  sleep 2
done
