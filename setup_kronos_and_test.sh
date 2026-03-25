#!/usr/bin/env bash
set -euo pipefail

# Setup and smoke-test script for src/encoders/ts_handler.py
# Usage:
#   bash setup_kronos_and_test.sh
# Optional env vars:
#   KRONOS_REPO_PATH, KRONOS_TOKENIZER_ID, KRONOS_MODEL_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KRONOS_REPO_PATH="${KRONOS_REPO_PATH:-$SCRIPT_DIR/Kronos}"
KRONOS_TOKENIZER_ID="${KRONOS_TOKENIZER_ID:-NeoQuasar/Kronos-Tokenizer-base}"
KRONOS_MODEL_ID="${KRONOS_MODEL_ID:-NeoQuasar/Kronos-small}"

echo "[1/5] Checking Python"
PYTHON_EXE=""

if command -v python >/dev/null 2>&1 && python -m pip --version >/dev/null 2>&1; then
  PYTHON_EXE="python"
elif command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
  PYTHON_EXE="python3"
fi

if [ -z "$PYTHON_EXE" ]; then
  USER_ROOT_GUESS_1="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || true)"
  USER_ROOT_GUESS_2="$(cd "$SCRIPT_DIR/../../.." 2>/dev/null && pwd || true)"
  for candidate in \
    "$HOME/Miniconda3/python.exe" \
    "$HOME/miniconda3/python.exe" \
    "$HOME/Anaconda3/python.exe" \
    "$HOME/anaconda3/python.exe" \
    "$USER_ROOT_GUESS_1/Miniconda3/python.exe" \
    "$USER_ROOT_GUESS_1/miniconda3/python.exe" \
    "$USER_ROOT_GUESS_1/Anaconda3/python.exe" \
    "$USER_ROOT_GUESS_1/anaconda3/python.exe" \
    "$USER_ROOT_GUESS_2/Miniconda3/python.exe" \
    "$USER_ROOT_GUESS_2/miniconda3/python.exe" \
    "$USER_ROOT_GUESS_2/Anaconda3/python.exe" \
    "$USER_ROOT_GUESS_2/anaconda3/python.exe" \
    "/mnt/c/Users/$USER/Miniconda3/python.exe" \
    "/mnt/c/Users/$USER/miniconda3/python.exe" \
    "/mnt/c/Users/$USER/Anaconda3/python.exe" \
    "/mnt/c/Users/$USER/anaconda3/python.exe"; do
    if [ -x "$candidate" ] && "$candidate" -m pip --version >/dev/null 2>&1; then
      PYTHON_EXE="$candidate"
      break
    fi
  done
fi

if [ -z "$PYTHON_EXE" ]; then
  echo "ERROR: No usable Python with pip found (tried python/python3 and common Conda paths)."
  exit 1
fi

echo "Using interpreter: $PYTHON_EXE"
"$PYTHON_EXE" --version

PYTHON_IS_WINDOWS=0
if [[ "$PYTHON_EXE" == *.exe ]]; then
  PYTHON_IS_WINDOWS=1
fi

to_python_path() {
  local p="$1"
  if [ "$PYTHON_IS_WINDOWS" -eq 1 ]; then
    if command -v wslpath >/dev/null 2>&1; then
      wslpath -w "$p"
      return 0
    fi
  fi
  echo "$p"
}

KRONOS_REPO_PATH_PY="$(to_python_path "$KRONOS_REPO_PATH")"
TS_HANDLER_PATH_PY="$(to_python_path "$SCRIPT_DIR/src/encoders/ts_handler.py")"

echo "[2/5] Cloning/updating Kronos repo"
if [ ! -d "$KRONOS_REPO_PATH/.git" ]; then
  git clone https://github.com/shiyu-coder/Kronos "$KRONOS_REPO_PATH"
else
  git -C "$KRONOS_REPO_PATH" pull --ff-only
fi

echo "[3/5] Installing Kronos requirements"
"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install -r "$KRONOS_REPO_PATH_PY/requirements.txt"

echo "[4/5] Exporting Kronos runtime env vars"
export KRONOS_REPO_PATH="$KRONOS_REPO_PATH_PY"
export KRONOS_TOKENIZER_ID
export KRONOS_MODEL_ID

echo "KRONOS_REPO_PATH=$KRONOS_REPO_PATH"
echo "KRONOS_TOKENIZER_ID=$KRONOS_TOKENIZER_ID"
echo "KRONOS_MODEL_ID=$KRONOS_MODEL_ID"

echo "[5/5] Running ts_handler smoke test"
output_file="$(mktemp)"
"$PYTHON_EXE" "$TS_HANDLER_PATH_PY" | tee "$output_file"

# Gate: fail unless Kronos backend is actually used.
if ! grep -q "Backend: kronos-local-repo" "$output_file"; then
  echo "ERROR: ts_handler.py did not run with Kronos backend (expected 'kronos-local-repo')."
  echo "Check model download/network access and KRONOS_* env vars."
  rm -f "$output_file"
  exit 1
fi

rm -f "$output_file"
echo "SUCCESS: Kronos setup and ts_handler test passed. Safe to push."
