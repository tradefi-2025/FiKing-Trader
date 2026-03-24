#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p logs

python -m src.server.flask_server > logs/flask_server.log 2>&1 &
FLASK_PID=$!

echo "Flask server started (pid=$FLASK_PID)"

python -m src.services.signaling.worker > logs/signaling_worker.log 2>&1 &
WORKER_PID=$!

echo "Signaling worker started (pid=$WORKER_PID)"

cleanup() {
  echo "Stopping services..."
  kill "$WORKER_PID" "$FLASK_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait "$FLASK_PID" "$WORKER_PID"
