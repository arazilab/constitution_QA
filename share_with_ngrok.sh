#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REQUESTED_PORT="${GRADIO_SERVER_PORT:-}"
PORT="${REQUESTED_PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv"
DEPS_DIR="$ROOT_DIR/.deps"
NGROK_BIN="$DEPS_DIR/ngrok"
APP_LOG="$DEPS_DIR/gradio-app.log"
NGROK_LOG="$DEPS_DIR/ngrok.log"

APP_PID=""
NGROK_PID=""

cleanup() {
  if [[ -n "${APP_PID}" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "${NGROK_PID}" ]] && kill -0 "$NGROK_PID" 2>/dev/null; then
    kill "$NGROK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_ngrok_asset() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Darwin) os="darwin" ;;
    Linux) os="linux" ;;
    *) echo "Unsupported OS for automatic ngrok download: $os" >&2; exit 1 ;;
  esac

  case "$arch" in
    arm64|aarch64) arch="arm64" ;;
    x86_64|amd64) arch="amd64" ;;
    *) echo "Unsupported CPU architecture for automatic ngrok download: $arch" >&2; exit 1 ;;
  esac

  echo "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-${os}-${arch}.zip"
}

install_ngrok_if_needed() {
  mkdir -p "$DEPS_DIR"

  if command -v ngrok >/dev/null 2>&1; then
    NGROK_BIN="$(command -v ngrok)"
    return
  fi

  if [[ -x "$NGROK_BIN" ]]; then
    return
  fi

  require_command curl
  require_command unzip

  local url zip_path
  url="$(detect_ngrok_asset)"
  zip_path="$DEPS_DIR/ngrok.zip"

  echo "Downloading ngrok to .deps/ ..."
  curl -fsSL "$url" -o "$zip_path"
  unzip -o "$zip_path" -d "$DEPS_DIR" >/dev/null
  chmod +x "$NGROK_BIN"
}

wait_for_app() {
  echo "Waiting for Gradio on http://127.0.0.1:${PORT} ..."
  for _ in {1..60}; do
    if curl -fsS "http://127.0.0.1:${PORT}" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done

  echo "Gradio did not become ready. See $APP_LOG" >&2
  exit 1
}

port_is_free() {
  "$VENV_DIR/bin/python" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)
PY
}

choose_port() {
  local candidate

  if [[ -n "$REQUESTED_PORT" ]]; then
    if port_is_free "$PORT"; then
      return
    fi
    echo "Port $PORT is already in use." >&2
    echo "Stop the process using it, or run with a different port." >&2
    echo "Example: GRADIO_SERVER_PORT=9000 ./share_with_ngrok.sh" >&2
    exit 1
  fi

  for candidate in $(seq "$PORT" "$((PORT + 100))"); do
    if port_is_free "$candidate"; then
      if [[ "$candidate" != "$PORT" ]]; then
        echo "Port $PORT is busy. Using port $candidate instead."
      fi
      PORT="$candidate"
      return
    fi
  done

  echo "No free port found from $PORT to $((PORT + 100))." >&2
  echo "Stop an old Gradio process, or set GRADIO_SERVER_PORT." >&2
  exit 1
}

wait_for_ngrok_url() {
  echo "Waiting for ngrok public URL ..."
  for _ in {1..60}; do
    local url
    url="$("$VENV_DIR/bin/python" - <<'PY' 2>/dev/null || true
import json
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit

for tunnel in payload.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    if public_url.startswith("https://"):
        print(public_url)
        break
PY
)"
    if [[ -n "$url" ]]; then
      echo
      echo "Share this link:"
      echo "$url"
      echo
      echo "Keep this terminal open while people use the app."
      echo "Press Ctrl-C to stop Gradio and ngrok."
      return
    fi
    sleep 1
  done

  echo "ngrok did not return a public URL. See $NGROK_LOG" >&2
  exit 1
}

process_is_running() {
  local pid state
  pid="$1"
  state="$(ps -p "$pid" -o state= 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z ]]
}

wait_for_processes() {
  while true; do
    if ! process_is_running "$APP_PID"; then
      echo "Gradio stopped unexpectedly. Last log lines:" >&2
      tail -n 40 "$APP_LOG" >&2 || true
      exit 1
    fi

    if ! process_is_running "$NGROK_PID"; then
      echo "ngrok stopped unexpectedly. Last log lines:" >&2
      tail -n 40 "$NGROK_LOG" >&2 || true
      exit 1
    fi

    sleep 2
  done
}

config_ngrok_auth_token() {
  "$VENV_DIR/bin/python" - <<'PY'
import json
from pathlib import Path

path = Path("config.json")
if not path.exists():
    raise SystemExit

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit

if not isinstance(payload, dict):
    raise SystemExit

ngrok = payload.get("ngrok", {})
if not isinstance(ngrok, dict):
    raise SystemExit

auth_token = str(ngrok.get("auth_token", "") or "").strip()
if auth_token:
    print(auth_token)
PY
}

if [[ ! -f "$ROOT_DIR/config.json" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "No config.json or OPENAI_API_KEY found." >&2
  echo "Create config.json from config.example.json or export OPENAI_API_KEY before sharing." >&2
  exit 1
fi

mkdir -p "$DEPS_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing Python requirements ..."
"$VENV_DIR/bin/pip" install -qqq -r requirements.txt

choose_port
install_ngrok_if_needed

NGROK_AUTH_TOKEN="${NGROK_AUTHTOKEN:-}"
if [[ -z "$NGROK_AUTH_TOKEN" ]]; then
  NGROK_AUTH_TOKEN="$(config_ngrok_auth_token)"
fi

if [[ -n "$NGROK_AUTH_TOKEN" ]]; then
  "$NGROK_BIN" config add-authtoken "$NGROK_AUTH_TOKEN" >/dev/null
fi

echo "Starting Gradio app on port $PORT ..."
: >"$APP_LOG"
: >"$NGROK_LOG"
GRADIO_SERVER_PORT="$PORT" GRADIO_SHARE=0 "$VENV_DIR/bin/python" app.py >"$APP_LOG" 2>&1 &
APP_PID="$!"

wait_for_app

echo "Starting ngrok tunnel ..."
"$NGROK_BIN" http "http://127.0.0.1:${PORT}" --log=stdout >"$NGROK_LOG" 2>&1 &
NGROK_PID="$!"

wait_for_ngrok_url

wait_for_processes
