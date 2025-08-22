#!/usr/bin/env bash
set -euo pipefail

# Create and initialize a dedicated venv for serving the Flask app (CPU-only)
# Usage:
#   bash scripts/setup_venv.sh            # creates .venv and installs deps
#   bash scripts/setup_venv.sh --recreate # remove and recreate venv

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

if [[ "${1:-}" == "--recreate" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo "==> Venv ready at $VENV_DIR"
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
