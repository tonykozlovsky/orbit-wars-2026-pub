#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 SUBMISSION.tar.gz" >&2
  exit 2
fi

SUBMISSION_TAR="$1"
[[ -f "${SUBMISSION_TAR}" ]] || {
  echo "Submission archive not found: ${SUBMISSION_TAR}" >&2
  exit 1
}

SUBMISSION_TAR_ABS="$(cd "$(dirname "${SUBMISSION_TAR}")" && pwd)/$(basename "${SUBMISSION_TAR}")"
SUBMISSION_TAR_DIR="$(dirname "${SUBMISSION_TAR_ABS}")"
SUBMISSION_TAR_BASENAME="$(basename "${SUBMISSION_TAR_ABS}")"
KAGGLE_DOCKER_IMAGE="gcr.io/kaggle-images/python-simulations"

docker run --rm -i \
  -e SUBMISSION_TAR_IN_CONTAINER="/submission_input/${SUBMISSION_TAR_BASENAME}" \
  -v "${SUBMISSION_TAR_DIR}:/submission_input:ro" \
  "${KAGGLE_DOCKER_IMAGE}" \
  python - <<'PY'
from __future__ import annotations

import importlib
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def _fmt(seconds: float) -> str:
    return f"{seconds:.6f}s"


script_t0 = time.perf_counter()
tar_path = Path(os.environ["SUBMISSION_TAR_IN_CONTAINER"])
assert tar_path.is_file(), tar_path

with tempfile.TemporaryDirectory(prefix="kaggle_submission_unpack_") as tmp_dir_raw:
    tmp_dir = Path(tmp_dir_raw)

    extract_t0 = time.perf_counter()
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp_dir)
    extract_s = time.perf_counter() - extract_t0

    artifacts = (
        tmp_dir / "kaggle_submission" / "checkpoint_policy_logits_aoti_2p.so",
        tmp_dir / "kaggle_submission" / "checkpoint_policy_logits_aoti_4p.so",
    )
    for artifact in artifacts:
        assert artifact.is_file(), artifact
    print(f"archive={tar_path}", flush=True)
    print(f"unpack_dir={tmp_dir}", flush=True)
    for artifact in artifacts:
        print(f"aoti_package={artifact}", flush=True)
        print(f"aoti_package_bytes={artifact.stat().st_size}", flush=True)
    print(f"extract_elapsed={_fmt(extract_s)}", flush=True)

    sys.path.insert(0, str(tmp_dir))

    import_t0 = time.perf_counter()
    submission = importlib.import_module("kaggle_submission.submission")
    import_s = time.perf_counter() - import_t0
    print(f"import_submission_elapsed={_fmt(import_s)}", flush=True)

    assert submission._submission_model_artifact_paths(2)[3] == artifacts[0], (
        submission._submission_model_artifact_paths(2)[3],
        artifacts[0],
    )
    assert submission._submission_model_artifact_paths(4)[3] == artifacts[1], (
        submission._submission_model_artifact_paths(4)[3],
        artifacts[1],
    )

    for num_agents in (2, 4):
        load_t0 = time.perf_counter()
        loaded_model = submission._get_model(num_agents)
        load_s = time.perf_counter() - load_t0
        assert loaded_model.artifact_kind == "policy_logits_aoti", loaded_model.artifact_kind
        print(f"get_model_aoti_load_elapsed num_agents={num_agents} elapsed={_fmt(load_s)}", flush=True)

        cached_t0 = time.perf_counter()
        cached_model = submission._get_model(num_agents)
        cached_s = time.perf_counter() - cached_t0
        assert cached_model is loaded_model, (cached_model, loaded_model)
        print(f"get_model_cached_elapsed num_agents={num_agents} elapsed={_fmt(cached_s)}", flush=True)

total_s = time.perf_counter() - script_t0
print(f"total_elapsed={_fmt(total_s)}", flush=True)
PY
