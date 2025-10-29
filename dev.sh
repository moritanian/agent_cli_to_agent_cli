#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend_dir="${repo_root}/backend"
frontend_dir="${repo_root}/frontend"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed or not on PATH." >&2
  exit 1
fi

if ! command -v yarn >/dev/null 2>&1; then
  echo "Error: yarn is not installed or not on PATH." >&2
  exit 1
fi

cleanup() {
  trap - INT TERM EXIT
  kill 0 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "Starting backend (FastAPI)..."
(cd "${backend_dir}" && uv run serve --reload) &
backend_pid=$!

echo "Starting frontend (Vite)..."
(cd "${frontend_dir}" && yarn dev) &
frontend_pid=$!

wait "${backend_pid}" "${frontend_pid}"
