#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p logs

python3 -m src.server.flask_server 2>&1 | tee -a logs/flask_server.log &
FLASK_PID=$!
echo "Flask server started (pid=$FLASK_PID)"

python3 -m src.services.signaling.worker 2>&1 | tee -a logs/signaling_worker.log &
WORKER_PID=$!
echo "Signaling worker started (pid=$WORKER_PID)"

python3 -m src.encoders.contextualizer 2>&1 | tee -a logs/contextualizer.log &
CONTEXTUALIZER_PID=$!
echo "Contextualizer started (pid=$CONTEXTUALIZER_PID)"

cleanup() {
  trap - INT TERM EXIT
  echo "Stopping services..."

  kill "$WORKER_PID" "$FLASK_PID" "$CONTEXTUALIZER_PID" 2>/dev/null || true
  wait "$WORKER_PID" 2>/dev/null || true
  wait "$FLASK_PID" 2>/dev/null || true
  wait "$CONTEXTUALIZER_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

wait -n "$FLASK_PID" "$WORKER_PID" "$CONTEXTUALIZER_PID" || true
cleanup
wait "$FLASK_PID" "$WORKER_PID" "$CONTEXTUALIZER_PID" 2>/dev/null || true