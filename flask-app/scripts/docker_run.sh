#!/usr/bin/env bash
set -euo pipefail

# Build and run the Flask egg-counting API in Docker
# Usage:
#   bash scripts/docker_run.sh build   # build image
#   bash scripts/docker_run.sh run     # run container
#   bash scripts/docker_run.sh dev     # rebuild and run

IMAGE_NAME=${IMAGE_NAME:-egg-counter-api}
TAG=${TAG:-latest}
PORT=${PORT:-5000}
MODEL_PATH=${DETECTION_ONNX_MODEL:-$(pwd)/models/rtdetr_eggs.onnx}

case "${1:-}" in
  build)
    docker build -t "$IMAGE_NAME:$TAG" .
    ;;
  run)
    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "Warning: model not found at $MODEL_PATH. Set DETECTION_ONNX_MODEL to override." >&2
    fi
    docker run --rm -p "$PORT:5000" \
      -e DETECTION_ONNX_MODEL="/app/models/rtdetr_eggs.onnx" \
      -e DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.3}" \
      -e ENABLE_CORS="${ENABLE_CORS:-true}" \
      -v "$MODEL_PATH:/app/models/rtdetr_eggs.onnx:ro" \
      "$IMAGE_NAME:$TAG"
    ;;
  dev)
    bash "$0" build && bash "$0" run
    ;;
  *)
    echo "Usage: $0 {build|run|dev}" >&2
    exit 1
    ;;
 esac
