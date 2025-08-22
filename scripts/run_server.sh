#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing venv at $VENV_DIR. Run scripts/setup_venv.sh first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export FLASK_APP=$PROJECT_ROOT/app.py
export DEBUG=${DEBUG:-false}
# Override model path if desired: export DETECTION_ONNX_MODEL=/path/model.onnx

python "$PROJECT_ROOT/app.py"
