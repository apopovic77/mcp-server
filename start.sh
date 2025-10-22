#!/usr/bin/env bash

# Wrapper for running the Arkturian MCP server. Designed for systemd usage.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

: "${ARKTURIAN_API_BASE:=https://api.arkturian.com}"
: "${MCP_HOST:=127.0.0.1}"
: "${MCP_PORT:=8080}"

if [[ -z "${ARKTURIAN_API_KEY:-}" ]]; then
  echo "ARKTURIAN_API_KEY must be set in the environment" >&2
  exit 1
fi

UVICORN_BIN="$BASE_DIR/venv/bin/uvicorn"
if [[ ! -x "$UVICORN_BIN" ]]; then
  echo "Expected uvicorn at $UVICORN_BIN (install deps first)" >&2
  exit 1
fi

exec "$UVICORN_BIN" server:app --host "$MCP_HOST" --port "$MCP_PORT" --proxy-headers --log-level info
