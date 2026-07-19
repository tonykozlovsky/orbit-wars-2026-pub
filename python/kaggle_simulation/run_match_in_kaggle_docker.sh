#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KAGGLE_DOCKER_IMAGE="${KAGGLE_DOCKER_IMAGE:-gcr.io/kaggle-images/python-simulations}"
DOCKER_GPU_ARGS=()
if [[ "${LOCAL_SIMULATION:-}" == "1" ]]; then
  DOCKER_GPU_ARGS=(--gpus all -e LOCAL_SIMULATION=1)
fi
DOCKER_SILENT_ARGS=()
if [[ "${SILENT:-}" == "1" ]]; then
  DOCKER_SILENT_ARGS=(-e SILENT=1)
fi
DOCKER_MODEL_PROFILE_ARGS=()
if [[ "${MODEL_PROFILE:-}" == "1" ]]; then
  DOCKER_MODEL_PROFILE_ARGS=(-e MODEL_PROFILE=1)
fi

docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  "${DOCKER_SILENT_ARGS[@]}" \
  "${DOCKER_MODEL_PROFILE_ARGS[@]}" \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -v /tmp:/tmp \
  -w "${REPO_ROOT}" \
  "${KAGGLE_DOCKER_IMAGE}" \
  python python/kaggle_simulation/run_match.py "$@"
