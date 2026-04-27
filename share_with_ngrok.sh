#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${GRADIO_SERVER_PORT:-7860}"
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
"$VENV_DIR/bin/pip" install -r requirements.txt

install_ngrok_if_needed

NGROK_AUTH_TOKEN="${NGROK_AUTHTOKEN:-}"
if [[ -z "$NGROK_AUTH_TOKEN" ]]; then
  NGROK_AUTH_TOKEN="$(config_ngrok_auth_token)"
fi

if [[ -n "$NGROK_AUTH_TOKEN" ]]; then
  "$NGROK_BIN" config add-authtoken "$NGROK_AUTH_TOKEN" >/dev/null
fi

echo "Starting Gradio app on port $PORT ..."
GRADIO_SERVER_PORT="$PORT" GRADIO_SHARE=0 "$VENV_DIR/bin/python" app.py >"$APP_LOG" 2>&1 &
APP_PID="$!"

wait_for_app

echo "Starting ngrok tunnel ..."
"$NGROK_BIN" http "http://127.0.0.1:${PORT}" --log=stdout >"$NGROK_LOG" 2>&1 &
NGROK_PID="$!"

wait_for_ngrok_url

wait "$APP_PID"
