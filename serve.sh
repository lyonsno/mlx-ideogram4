#!/bin/bash
# Launch the Ideogram4 NF4 Gradio server, optionally through a stable ngrok dev domain.
#
# Usage:
#   ./serve.sh --public
#   ./serve.sh --public --port 7861
#   NGROK_DOMAIN=your-assigned-name.ngrok-free.dev ./serve.sh --public --tunnel ngrok
#   ./serve.sh --public --tunnel ngrok --ngrok-domain your-assigned-name.ngrok-free.dev
#   ./serve.sh --tunnel ngrok --tunnel-only --port 7861 --ngrok-domain your-assigned-name.ngrok-free.dev
#
# Notes:
#   - The live launcher uses .venv/bin/python by default so uv does not resync the
#     environment and shadow the NF4 fork with stock MLX.
#   - Gradio's --share URL is intentionally not stable; use --tunnel ngrok for a
#     repeatable public URL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${HOME}/Library/Logs/ideogram4-nf4"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/serve.log"
TUNNEL_LOG_FILE="${LOG_DIR}/tunnel.log"

TUNNEL="none"
TUNNEL_ONLY="0"
PORT="${GRADIO_SERVER_PORT:-7860}"
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
APP_ARGS=()

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tunnel)
      TUNNEL="${2:-}"
      shift 2
      ;;
    --ngrok-domain)
      NGROK_DOMAIN="${2:-}"
      shift 2
      ;;
    --tunnel-only)
      TUNNEL_ONLY="1"
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      APP_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$SCRIPT_DIR"

if [[ "$TUNNEL_ONLY" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found or not executable: $PYTHON_BIN" >&2
  echo "Create/repair .venv first, then reinstall the NF4 MLX fork last." >&2
  exit 2
fi

if [[ "$TUNNEL" != "none" && "$TUNNEL" != "ngrok" ]]; then
  echo "Unsupported tunnel: $TUNNEL" >&2
  echo "Supported values: none, ngrok" >&2
  exit 2
fi

start_server() {
  echo "[$(date -Iseconds)] Starting Ideogram4 NF4 server on port ${PORT}: ${PYTHON_BIN} app.py ${APP_ARGS[*]}" >> "$LOG_FILE"
  GRADIO_SERVER_PORT="$PORT" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u app.py "${APP_ARGS[@]}" >> "$LOG_FILE" 2>&1
}

wait_for_server() {
  local url="http://127.0.0.1:${PORT}/"
  local attempt
  for attempt in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Server did not answer at ${url} after 120s. See ${LOG_FILE}." >&2
  return 1
}

if [[ "$TUNNEL" == "none" ]]; then
  exec env GRADIO_SERVER_PORT="$PORT" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u app.py "${APP_ARGS[@]}" >> "$LOG_FILE" 2>&1
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok is not installed. Install it, then run this command again." >&2
  echo "macOS/Homebrew example: brew install ngrok/ngrok/ngrok" >&2
  exit 3
fi

if [[ -z "$NGROK_DOMAIN" ]]; then
  echo "NGROK_DOMAIN is required for a stable no-paid-domain tunnel." >&2
  echo "Find the free dev domain in the ngrok dashboard under Universal Gateway > Domains." >&2
  echo "It should look like: something.ngrok-free.dev" >&2
  exit 4
fi

if [[ "$TUNNEL_ONLY" == "1" ]]; then
  APP_PID=""
else
  start_server &
  APP_PID=$!
  cleanup() {
    if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" >/dev/null 2>&1; then
      kill "$APP_PID" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT INT TERM
fi

wait_for_server

echo "[$(date -Iseconds)] Starting ngrok tunnel https://${NGROK_DOMAIN} -> http://127.0.0.1:${PORT}" >> "$TUNNEL_LOG_FILE"
echo "Public demo: https://${NGROK_DOMAIN}"
ngrok http "$PORT" --url "https://${NGROK_DOMAIN}" --log=stdout --log-format=logfmt >> "$TUNNEL_LOG_FILE" 2>&1
