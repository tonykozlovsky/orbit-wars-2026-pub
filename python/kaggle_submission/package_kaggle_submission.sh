#!/usr/bin/env bash
set -euo pipefail

# Build a minimal tar.gz for Kaggle Orbit Wars: only modules needed to import
# kaggle_submission.submission.agent and run ImpalaOrbitModel forward.
#
# Native code: package a prebuilt orbit_wars_cpp extension, or build it locally
# inside Kaggle's official simulations Docker image when ORBIT_WARS_CPP_PREBUILT_SO is unset.
#
# Why a stub src/gym/__init__.py: the repo's package __init__ imports create_env
# (training stack). Importing src.gym.orbit_wars_env loads src.gym first; the stub
# avoids pulling torchbeast, env factories, etc.
#
# Usage:
#   ./package_kaggle_submission.sh [OUTPUT.tar.gz] [CHECKPOINT.pt]
#   ./package_kaggle_submission.sh [OUTPUT.tar.gz] [CHECKPOINT_2P.pt] [CHECKPOINT_4P.pt]
#   ./package_kaggle_submission.sh [OUTPUT.tar.gz] [CHECKPOINT_2P.pt] [CHECKPOINT_4P.pt] [CHECKPOINT_OVERAGE_PRIMARY.pt]
#
# If checkpoints are omitted, uses env ORBIT_IMPALA_CHECKPOINT,
# ORBIT_IMPALA_CHECKPOINT_2P/4P, or ORBIT_IMPALA_CHECKPOINT_2P/4P plus
# ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY when set; otherwise the archive has no weights.
#
# Recommended from repo root:
#   python/kaggle_submission/package_kaggle_submission.sh \
#     orbit_wars_submit.tar.gz /path/to/2p/ckpt.pt /path/to/4p/ckpt.pt
#     orbit_wars_submit.tar.gz /path/to/2p/ckpt.pt /path/to/4p/ckpt.pt /path/to/overage_primary/ckpt.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPALA_PY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -gt 4 ]]; then
  echo "Usage: $0 [OUTPUT.tar.gz] [CHECKPOINT.pt]" >&2
  echo "   or: $0 [OUTPUT.tar.gz] [CHECKPOINT_2P.pt] [CHECKPOINT_4P.pt]" >&2
  echo "   or: $0 [OUTPUT.tar.gz] [CHECKPOINT_2P.pt] [CHECKPOINT_4P.pt] [CHECKPOINT_OVERAGE_PRIMARY.pt]" >&2
  exit 2
fi

OUT="${1:-${IMPALA_PY_ROOT}/orbit_wars_kaggle_submission.tar.gz}"
CKPT_SHARED=""
CKPT_2P=""
CKPT_4P=""
CKPT_OVERAGE_PRIMARY=""
if [[ "$#" -eq 2 ]]; then
  CKPT_SHARED="$2"
elif [[ "$#" -eq 3 ]]; then
  CKPT_2P="$2"
  CKPT_4P="$3"
elif [[ "$#" -eq 4 ]]; then
  CKPT_2P="$2"
  CKPT_4P="$3"
  CKPT_OVERAGE_PRIMARY="$4"
else
  CKPT_SHARED="${ORBIT_IMPALA_CHECKPOINT:-}"
  CKPT_2P="${ORBIT_IMPALA_CHECKPOINT_2P:-}"
  CKPT_4P="${ORBIT_IMPALA_CHECKPOINT_4P:-}"
  CKPT_OVERAGE_PRIMARY="${ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY:-}"
fi
if [[ -n "${CKPT_SHARED}" && ( -n "${CKPT_2P}" || -n "${CKPT_4P}" || -n "${CKPT_OVERAGE_PRIMARY}" ) ]]; then
  echo "Use either ORBIT_IMPALA_CHECKPOINT or ORBIT_IMPALA_CHECKPOINT_2P/4P[/OVERAGE_PRIMARY], not both." >&2
  exit 1
fi
if [[ -z "${CKPT_SHARED}" && ( -n "${CKPT_2P}" || -n "${CKPT_4P}" || -n "${CKPT_OVERAGE_PRIMARY}" ) ]]; then
  [[ -n "${CKPT_2P}" && -n "${CKPT_4P}" ]] || {
    echo "Both 2p and 4p checkpoints are required for per-player checkpoint mode." >&2
    exit 1
  }
  [[ -z "${CKPT_OVERAGE_PRIMARY}" || ( -n "${CKPT_2P}" && -n "${CKPT_4P}" ) ]] || {
    echo "2p and 4p checkpoints are required with ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY." >&2
    exit 1
  }
fi

read_bool_env() {
  local name="$1"
  local value="${!name:-0}"
  case "${value}" in
    ""|0) echo "0" ;;
    1) echo "1" ;;
    *)
      echo "${name} must be unset, 0, or 1; got '${value}'" >&2
      exit 1
      ;;
  esac
}

ORBIT_SUBMISSION_DYNAMIC_QUANTIZE="$(read_bool_env ORBIT_SUBMISSION_DYNAMIC_QUANTIZE)"
ORBIT_SUBMISSION_TORCHSCRIPT="$(read_bool_env ORBIT_SUBMISSION_TORCHSCRIPT)"
ORBIT_SUBMISSION_AOTI="$(read_bool_env ORBIT_SUBMISSION_AOTI)"
ORBIT_SUBMISSION_IS_SAMPLE="${ORBIT_SUBMISSION_IS_SAMPLE:-1}"
ORBIT_SUBMISSION_IS_SAMPLE="$(read_bool_env ORBIT_SUBMISSION_IS_SAMPLE)"
ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS="${ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS:-1}"
ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS="$(read_bool_env ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS)"
ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS="${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS:-}"
ORBIT_SUBMISSION_CPU_TORCH_VERSION="2.6.0"
ORBIT_SUBMISSION_CPU_CAPABILITY="avx2"
ORBIT_SUBMISSION_CPU_MARCH="x86-64-v3"
PACK_FOR_NOTEBOOK="$(read_bool_env PACK_FOR_NOTEBOOK)"
if [[ "${ORBIT_SUBMISSION_TORCHSCRIPT}" == "1" && "${ORBIT_SUBMISSION_AOTI}" == "1" ]]; then
  echo "ORBIT_SUBMISSION_TORCHSCRIPT and ORBIT_SUBMISSION_AOTI are mutually exclusive." >&2
  exit 1
fi
if [[ -n "${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS}" && "${ORBIT_SUBMISSION_AOTI}" != "1" ]]; then
  echo "ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS requires ORBIT_SUBMISSION_AOTI=1." >&2
  exit 1
fi

STAGING="$(mktemp -d)"
CPP_BUILD_DIR="$(mktemp -d)"
if [[ "${PACK_FOR_NOTEBOOK}" == "1" ]]; then
  KAGGLE_DOCKER_IMAGE="gcr.io/kaggle-images/python"
else
  KAGGLE_DOCKER_IMAGE="gcr.io/kaggle-images/python-simulations"
fi
INSTALL_CPU_TORCH_CMD="python -m pip install --no-cache-dir --force-reinstall torch==${ORBIT_SUBMISSION_CPU_TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
cat > "${CPP_BUILD_DIR}/cxx_cpu_target.sh" <<'SH'
#!/usr/bin/env bash
exec g++ "$@" "-march=${ORBIT_SUBMISSION_CPU_MARCH}"
SH
chmod +x "${CPP_BUILD_DIR}/cxx_cpu_target.sh"
cleanup() {
  rm -rf "${STAGING}"
  rm -rf "${CPP_BUILD_DIR}"
}
trap cleanup EXIT

mkdir -p \
  "${STAGING}/kaggle_submission" \
  "${STAGING}/src/gym" \
  "${STAGING}/src/models" \
  "${STAGING}/src/configs" \
  "${STAGING}/cpp/orbit_wars/reference_kaggle_upstream_github_no_edit"

cp "${IMPALA_PY_ROOT}/main.py" "${STAGING}/"
cp "${SCRIPT_DIR}/submission.py" "${STAGING}/kaggle_submission/"
cp "${SCRIPT_DIR}/debug_aoti_package.py" "${STAGING}/kaggle_submission/"
if [[ "${ORBIT_SUBMISSION_IS_SAMPLE}" == "1" ]]; then
  SUBMISSION_IS_SAMPLE_PY=True
else
  SUBMISSION_IS_SAMPLE_PY=False
fi
if [[ "${ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS}" == "1" ]]; then
  SUBMISSION_SHUFFLE_IDENTITY_IDS_PY=True
else
  SUBMISSION_SHUFFLE_IDENTITY_IDS_PY=False
fi
cat > "${STAGING}/kaggle_submission/submission_runtime_config.py" <<PY
SUBMISSION_IS_SAMPLE = ${SUBMISSION_IS_SAMPLE_PY}
SUBMISSION_SHUFFLE_IDENTITY_IDS = ${SUBMISSION_SHUFFLE_IDENTITY_IDS_PY}
PY
if [[ "${ORBIT_SUBMISSION_AOTI}" == "1" ]]; then
  cp "${SCRIPT_DIR}/aoti_policy_runner_cpp.cpp" "${STAGING}/kaggle_submission/"
fi

: > "${STAGING}/src/__init__.py"
: > "${STAGING}/kaggle_submission/__init__.py"
: > "${STAGING}/src/gym/__init__.py"
# Minimal package namespace: avoid repo configs/__init__.py (imports training YAML stack).
: > "${STAGING}/src/configs/__init__.py"

cp "${IMPALA_PY_ROOT}/src/gym/dict_io_contract.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_honest_mask_dataset_io.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/wall_tree_profiler.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/obs_wrapper.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_cpp_plain_sync.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_kaggle_cpp_cache.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_kaggle_replay_fleet_gt.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_reference_upstream_random_state.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_tape_feature_pack.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_wars_env.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_wars_cpp_ext.py" "${STAGING}/src/gym/"
cp "${IMPALA_PY_ROOT}/src/gym/orbit_wars_cpp_obs_stub.py" "${STAGING}/src/gym/"
: > "${STAGING}/src/models/__init__.py"
cp "${IMPALA_PY_ROOT}/src/models/models.py" "${STAGING}/src/models/"
cp "${IMPALA_PY_ROOT}/src/models/orbit_obs_feature_layout.py" "${STAGING}/src/models/"
cp "${IMPALA_PY_ROOT}/src/models/orbit_obs_feature_input_contract.py" "${STAGING}/src/models/"
cp "${IMPALA_PY_ROOT}/src/configs/impala_orbit_model_hyperparams.py" "${STAGING}/src/configs/"
cp "${IMPALA_PY_ROOT}/src/configs/impala_orbit_obs_feature_normalization.py" "${STAGING}/src/configs/"
cp "${IMPALA_PY_ROOT}/cpp/orbit_wars/reference_kaggle_upstream_github_no_edit/orbit_wars.py" \
  "${STAGING}/cpp/orbit_wars/reference_kaggle_upstream_github_no_edit/"
cp "${IMPALA_PY_ROOT}/cpp/orbit_wars/reference_kaggle_upstream_github_no_edit/orbit_wars.json" \
  "${STAGING}/cpp/orbit_wars/reference_kaggle_upstream_github_no_edit/"

export IMPALA_PY_ROOT STAGING CPP_BUILD_DIR
export CCACHE_DISABLE=1

if [[ -n "${ORBIT_WARS_CPP_PREBUILT_SO:-}" ]]; then
  [[ -f "${ORBIT_WARS_CPP_PREBUILT_SO}" ]] || {
    echo "ORBIT_WARS_CPP_PREBUILT_SO not found: ${ORBIT_WARS_CPP_PREBUILT_SO}" >&2
    exit 1
  }
  PREBUILT_STAGED_SO="${STAGING}/src/gym/$(basename "${ORBIT_WARS_CPP_PREBUILT_SO}")"
  cp "${ORBIT_WARS_CPP_PREBUILT_SO}" "${PREBUILT_STAGED_SO}"
  strip --strip-unneeded "${PREBUILT_STAGED_SO}"
  {
    echo "python=<prebuilt>"
    echo "torch=<prebuilt>"
    echo "torch_cuda=<prebuilt>"
    echo "module_file=$(basename "${ORBIT_WARS_CPP_PREBUILT_SO}")"
  } > "${STAGING}/src/gym/orbit_wars_cpp_build_meta.txt"
  echo "Packaged prebuilt native module: ${ORBIT_WARS_CPP_PREBUILT_SO}"
else
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required to build orbit_wars_cpp inside Kaggle image." >&2
    echo "Set ORBIT_WARS_CPP_PREBUILT_SO=/path/to/orbit_wars_cpp.so to package an existing build." >&2
    exit 1
  }
  # Build the native module in Kaggle's Python image and package only the compiled binary.
  # Same translation units and flags as src/gym/orbit_wars_cpp_ext.py / cpp/orbit_wars/setup.py.
  cat > "${CPP_BUILD_DIR}/build_orbit_wars_cpp.py" <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

root = Path(os.environ["IMPALA_PY_ROOT"]).resolve()
staging = Path(os.environ["STAGING"]).resolve()
build_dir = Path(os.environ["CPP_BUILD_DIR"]).resolve()
cpp_dir = root / "cpp" / "orbit_wars"
sources = [
    cpp_dir / "bindings.cpp",
    cpp_dir / "io.cpp",
    cpp_dir / "library.cpp",
    cpp_dir / "simulation.cpp",
    cpp_dir / "masks.cpp",
    cpp_dir / "kaggle_integration.cpp",
    cpp_dir / "honest_shared_intercept.cpp",
    cpp_dir / "honest_shared_features.cpp",
    cpp_dir / "cpp_env_v2" / "cpp_env_static_cache_v2.cpp",
    cpp_dir / "cpp_env_v2" / "cpp_env_static_cache_v2_mask.cpp",
    cpp_dir / "cpp_env_v2" / "cpp_env_static_cache_v2_features.cpp",
    cpp_dir / "cpp_env_v2" / "cpp_env_live_v2.cpp",
]
for p in sources:
    assert p.is_file(), p

print(f"Building native module in Kaggle image with torch={torch.__version__}, cuda={torch.version.cuda}")
mod = load(
    name="orbit_wars_cpp",
    sources=[str(p) for p in sources],
    extra_cflags=[
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        f"-march={os.environ['ORBIT_SUBMISSION_CPU_MARCH']}",
        f"-I{cpp_dir}",
    ],
    with_cuda=False,
    verbose=False,
    build_directory=str(build_dir),
)
mod_file = Path(mod.__file__).resolve()
assert mod_file.is_file(), mod_file
dst = staging / "src" / "gym" / mod_file.name
shutil.copy2(mod_file, dst)
subprocess.run(["strip", "--strip-unneeded", str(dst)], check=True)
meta = staging / "src" / "gym" / "orbit_wars_cpp_build_meta.txt"
meta.write_text(
    f"python={sys.version}\n"
    f"torch={torch.__version__}\n"
    f"torch_cuda={torch.version.cuda}\n"
    f"module_file={mod_file.name}\n"
)
print(f"Packaged native module: {dst}")
PY
  docker run --rm \
    -e IMPALA_PY_ROOT=/workspace \
    -e STAGING=/staging \
    -e CPP_BUILD_DIR=/build \
    -e CCACHE_DISABLE=1 \
    -e ATEN_CPU_CAPABILITY="${ORBIT_SUBMISSION_CPU_CAPABILITY}" \
    -e ORBIT_SUBMISSION_CPU_MARCH="${ORBIT_SUBMISSION_CPU_MARCH}" \
    -e CXX=/build/cxx_cpu_target.sh \
    -e CFLAGS="-march=${ORBIT_SUBMISSION_CPU_MARCH}" \
    -e CXXFLAGS="-march=${ORBIT_SUBMISSION_CPU_MARCH}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${IMPALA_PY_ROOT}:/workspace:ro" \
    -v "${STAGING}:/staging" \
    -v "${CPP_BUILD_DIR}:/build" \
    -w /workspace \
    "${KAGGLE_DOCKER_IMAGE}" \
    bash -lc "${INSTALL_CPU_TORCH_CMD} && python /build/build_orbit_wars_cpp.py"
fi

if [[ -n "${CKPT_SHARED}" || -n "${CKPT_2P}" || -n "${CKPT_4P}" || -n "${CKPT_OVERAGE_PRIMARY}" ]]; then
  if [[ -n "${CKPT_SHARED}" ]]; then
    [[ -f "${CKPT_SHARED}" ]] || {
      echo "checkpoint not found: ${CKPT_SHARED}" >&2
      exit 1
    }
    CKPT_MODE="shared"
  else
    [[ -f "${CKPT_2P}" ]] || {
      echo "2p checkpoint not found: ${CKPT_2P}" >&2
      exit 1
    }
    [[ -f "${CKPT_4P}" ]] || {
      echo "4p checkpoint not found: ${CKPT_4P}" >&2
      exit 1
    }
    if [[ -n "${CKPT_OVERAGE_PRIMARY}" ]]; then
      [[ -f "${CKPT_OVERAGE_PRIMARY}" ]] || {
        echo "overage primary checkpoint not found: ${CKPT_OVERAGE_PRIMARY}" >&2
        exit 1
      }
      CKPT_MODE="by_num_agents_with_overage_primary"
    else
      CKPT_MODE="by_num_agents"
    fi
  fi
  if [[ "${ORBIT_SUBMISSION_DYNAMIC_QUANTIZE}" == "0" && "${ORBIT_SUBMISSION_TORCHSCRIPT}" == "0" && "${ORBIT_SUBMISSION_AOTI}" == "0" ]]; then
    if [[ "${CKPT_MODE}" == "shared" ]]; then
      cp "${CKPT_SHARED}" "${STAGING}/kaggle_submission/checkpoint_weights.pt"
    else
      cp "${CKPT_2P}" "${STAGING}/kaggle_submission/checkpoint_weights_2p.pt"
      cp "${CKPT_4P}" "${STAGING}/kaggle_submission/checkpoint_weights_4p.pt"
      if [[ "${CKPT_MODE}" == "by_num_agents_with_overage_primary" ]]; then
        cp "${CKPT_OVERAGE_PRIMARY}" "${STAGING}/kaggle_submission/checkpoint_weights_overage_primary.pt"
      fi
    fi
  else
    command -v docker >/dev/null 2>&1 || {
      echo "docker is required to build optimized submission model artifacts in Kaggle's Python image." >&2
      exit 1
    }
    SOURCE_CKPT_ENV_ARGS=(-e ORBIT_SUBMISSION_SOURCE_CKPT_MODE="${CKPT_MODE}")
    SOURCE_CKPT_DOCKER_ARGS=()
    if [[ "${CKPT_MODE}" == "shared" ]]; then
      CKPT_SHARED_ABS="$(cd "$(dirname "${CKPT_SHARED}")" && pwd)/$(basename "${CKPT_SHARED}")"
      CKPT_SHARED_MOUNT_DIR="$(dirname "${CKPT_SHARED_ABS}")"
      CKPT_SHARED_BASENAME="$(basename "${CKPT_SHARED_ABS}")"
      SOURCE_CKPT_ENV_ARGS+=(-e ORBIT_SUBMISSION_SOURCE_CKPT="/checkpoint_shared/${CKPT_SHARED_BASENAME}")
      SOURCE_CKPT_DOCKER_ARGS+=(-v "${CKPT_SHARED_MOUNT_DIR}:/checkpoint_shared:ro")
    else
      CKPT_2P_ABS="$(cd "$(dirname "${CKPT_2P}")" && pwd)/$(basename "${CKPT_2P}")"
      CKPT_2P_MOUNT_DIR="$(dirname "${CKPT_2P_ABS}")"
      CKPT_2P_BASENAME="$(basename "${CKPT_2P_ABS}")"
      CKPT_4P_ABS="$(cd "$(dirname "${CKPT_4P}")" && pwd)/$(basename "${CKPT_4P}")"
      CKPT_4P_MOUNT_DIR="$(dirname "${CKPT_4P_ABS}")"
      CKPT_4P_BASENAME="$(basename "${CKPT_4P_ABS}")"
      SOURCE_CKPT_ENV_ARGS+=(
        -e ORBIT_SUBMISSION_SOURCE_CKPT_2P="/checkpoint_2p/${CKPT_2P_BASENAME}"
        -e ORBIT_SUBMISSION_SOURCE_CKPT_4P="/checkpoint_4p/${CKPT_4P_BASENAME}"
      )
      SOURCE_CKPT_DOCKER_ARGS+=(
        -v "${CKPT_2P_MOUNT_DIR}:/checkpoint_2p:ro"
        -v "${CKPT_4P_MOUNT_DIR}:/checkpoint_4p:ro"
      )
      if [[ "${CKPT_MODE}" == "by_num_agents_with_overage_primary" ]]; then
        CKPT_OVERAGE_PRIMARY_ABS="$(cd "$(dirname "${CKPT_OVERAGE_PRIMARY}")" && pwd)/$(basename "${CKPT_OVERAGE_PRIMARY}")"
        CKPT_OVERAGE_PRIMARY_MOUNT_DIR="$(dirname "${CKPT_OVERAGE_PRIMARY_ABS}")"
        CKPT_OVERAGE_PRIMARY_BASENAME="$(basename "${CKPT_OVERAGE_PRIMARY_ABS}")"
        SOURCE_CKPT_ENV_ARGS+=(
          -e ORBIT_SUBMISSION_SOURCE_CKPT_OVERAGE_PRIMARY="/checkpoint_overage_primary/${CKPT_OVERAGE_PRIMARY_BASENAME}"
        )
        SOURCE_CKPT_DOCKER_ARGS+=(
          -v "${CKPT_OVERAGE_PRIMARY_MOUNT_DIR}:/checkpoint_overage_primary:ro"
        )
      fi
    fi
    AOTI_EXAMPLE_INPUTS_CONTAINER=""
    AOTI_EXAMPLE_INPUTS_DOCKER_ARGS=()
    if [[ "${ORBIT_SUBMISSION_AOTI}" == "1" && -n "${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS}" ]]; then
      [[ -f "${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS}" ]] || {
        echo "ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS not found: ${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS}" >&2
        exit 1
      }
      AOTI_EXAMPLE_INPUTS_ABS="$(cd "$(dirname "${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS}")" && pwd)/$(basename "${ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS}")"
      AOTI_EXAMPLE_INPUTS_MOUNT_DIR="$(dirname "${AOTI_EXAMPLE_INPUTS_ABS}")"
      AOTI_EXAMPLE_INPUTS_BASENAME="$(basename "${AOTI_EXAMPLE_INPUTS_ABS}")"
      AOTI_EXAMPLE_INPUTS_CONTAINER="/aoti_example_inputs/${AOTI_EXAMPLE_INPUTS_BASENAME}"
      AOTI_EXAMPLE_INPUTS_DOCKER_ARGS=(-v "${AOTI_EXAMPLE_INPUTS_MOUNT_DIR}:/aoti_example_inputs:ro")
    fi
    cat > "${CPP_BUILD_DIR}/build_optimized_submission_artifact.py" <<'PY'
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

print("build_optimized_submission_artifact_import_torch_start", flush=True)
import torch
import torch.nn as nn
import torch._inductor
from torch.quantization import quantize_dynamic
from torch.utils.cpp_extension import load
print(f"build_optimized_submission_artifact_import_torch_done torch={torch.__version__}", flush=True)

print("build_optimized_submission_artifact_import_project_start", flush=True)
from src.gym.obs_wrapper import (
    ORBIT_EDGE_FEATURES,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
)
from src.models.models import ImpalaOrbitModel, impala_orbit_model_init_kwargs_from_flags
print("build_optimized_submission_artifact_import_project_done", flush=True)


_AOTI_EXAMPLE_INPUT_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
)
_AOTI_EXAMPLE_INPUT_CAPTURE_STEP = 50


def _submission_model_init_kwargs(model_config: Mapping[str, Any]) -> dict[str, Any]:
    flags = SimpleNamespace(
        model=model_config,
        target_min_entropy={"send": 0.0, "dst": 0.0, "amount": 0.0},
        split_head_uniform_mixture_epsilon={"send": 0.0, "dst": 0.0, "amount": 0.0},
        entropy_floor_max_temperature=10000.0,
        entropy_floor_num_iters=16,
    )
    kw = impala_orbit_model_init_kwargs_from_flags(flags)
    kw["entropy_floor_target"] = (0.0, 0.0, 0.0)
    kw["entropy_floor_max_temperature"] = 10000.0
    kw["entropy_floor_num_iters"] = 16
    kw["include_rl_policy_value_heads"] = True
    kw["include_rl_value_head"] = False
    return kw


def _strip_torch_compile_orig_mod_prefix(sd: dict[Any, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    keys = tuple(str(k) for k in sd.keys())
    prefixed = tuple(k.startswith(prefix) for k in keys)
    assert all(prefixed) or not any(prefixed), keys[:8]
    if not any(prefixed):
        return {str(k): v for k, v in sd.items()}
    return {str(k)[len(prefix) :]: v for k, v in sd.items()}


def _is_stale_submission_state_key(key: str) -> bool:
    return (
        key.startswith("_global_value_head.")
        or key.startswith("_global_value_head_production_delta.")
        or key == "_policy_head._split_head_uniform_mixture_epsilon"
    )


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict), type(ckpt)
    assert "model_state_dict" in ckpt, (
        f"Expected IMPALA checkpoint with model_state_dict; keys={sorted(ckpt.keys())}"
    )
    assert "model_config" in ckpt, (
        f"Expected IMPALA checkpoint with model_config; keys={sorted(ckpt.keys())}"
    )
    assert isinstance(ckpt["model_config"], Mapping), type(ckpt["model_config"])
    return ckpt


def _load_model_weights(model: ImpalaOrbitModel, ckpt: dict[str, Any]) -> None:
    raw_sd = ckpt["model_state_dict"]
    assert isinstance(raw_sd, dict), type(raw_sd)
    sd = _strip_torch_compile_orig_mod_prefix(raw_sd)
    model_sd = model.state_dict()
    checkpoint_only = sorted(k for k in sd if k not in model_sd)
    assert all(_is_stale_submission_state_key(k) for k in checkpoint_only), checkpoint_only
    load_sd = {k: v for k, v in sd.items() if k in model_sd}
    missing = sorted(k for k in model_sd if k not in load_sd)
    assert not missing, missing
    for k, v in load_sd.items():
        assert isinstance(v, torch.Tensor), (k, type(v))
        assert v.shape == model_sd[k].shape, (k, tuple(v.shape), tuple(model_sd[k].shape))
    model.load_state_dict(load_sd, strict=True)


def _configure_submission_model_runtime(
    model: ImpalaOrbitModel,
    *,
    is_sample: bool,
    shuffle_identity_ids: bool,
) -> None:
    zero_heads = torch.zeros(3, dtype=torch.float32)
    model.set_entropy_floor_targets(zero_heads)
    model.set_split_head_uniform_mixture_epsilon(zero_heads)
    model.set_is_sample(is_sample)
    model.set_compile_friendly_sample(True)
    model.set_shuffle_identity_ids(shuffle_identity_ids)


class _OrbitPolicyActionsTraceWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        orbit_planet_features: torch.Tensor,
        orbit_planet_arrival_features: torch.Tensor,
        orbit_enemy_mask: torch.Tensor,
        orbit_planet_mask: torch.Tensor,
        orbit_planet_pairwise_mask: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        available_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self._model(
            {
                "obs_LEARN_INFER": {
                    "orbit_planet_features": orbit_planet_features,
                    "orbit_planet_arrival_features": orbit_planet_arrival_features,
                    "orbit_enemy_mask": orbit_enemy_mask,
                    "orbit_planet_mask": orbit_planet_mask,
                    "orbit_planet_pairwise_mask": orbit_planet_pairwise_mask,
                    "orbit_planet_pairwise_features": orbit_planet_pairwise_features,
                    "available_action_mask": available_action_mask,
                },
            },
            output_full_policy_log_probs=False,
            include_final_policy_logits=False,
            include_policy_logits_pre_action_mask=False,
            include_value_head=False,
        )
        actions = out["actions_LEARN"]["spawn_fleet"]
        assert isinstance(actions, torch.Tensor), type(actions)
        return actions


def _example_inputs() -> tuple[torch.Tensor, ...]:
    b_e = 1
    b_p = 1
    available_action_mask = torch.zeros(
        (
            b_e,
            b_p,
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PER_PLANET_MOVE_CLASSES,
        ),
        dtype=torch.int8,
    )
    src = torch.arange(ORBIT_MAX_PLANETS, dtype=torch.int64)
    noop_cls = src * ORBIT_MOVE_CLASSES_PER_TARGET
    available_action_mask[:, :, src, noop_cls] = 1
    return (
        torch.zeros((b_e, b_p, ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES), dtype=torch.float32),
        torch.zeros(
            (
                b_e,
                b_p,
                ORBIT_MAX_PLANETS,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ),
            dtype=torch.float32,
        ),
        torch.ones((b_e, b_p, ORBIT_PLAYER_AXIS_SLOTS - 1), dtype=torch.float32),
        torch.ones((b_e, b_p, ORBIT_MAX_PLANETS), dtype=torch.float32),
        torch.ones((b_e, b_p, ORBIT_PLANET_PAIRWISE_COUNT), dtype=torch.float32),
        torch.zeros((b_e, b_p, ORBIT_PLANET_PAIRWISE_COUNT, ORBIT_EDGE_FEATURES), dtype=torch.float32),
        available_action_mask,
    )


def _example_inputs_from_capture(path: Path) -> tuple[torch.Tensor, ...]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(payload, dict), type(payload)
    assert payload["format"] == "orbit_policy_logits_aoti_example_inputs_v1", payload["format"]
    assert tuple(payload["input_keys"]) == _AOTI_EXAMPLE_INPUT_KEYS, payload["input_keys"]
    samples = payload["samples"]
    assert isinstance(samples, list), type(samples)
    assert len(samples) > 0, "AOTI example input capture contains no samples"
    samples_per_step = 4
    assert samples_per_step == 4, samples_per_step
    sample_index = _AOTI_EXAMPLE_INPUT_CAPTURE_STEP * samples_per_step
    assert sample_index < len(samples), (
        _AOTI_EXAMPLE_INPUT_CAPTURE_STEP,
        samples_per_step,
        len(samples),
    )
    sample = samples[sample_index]
    assert isinstance(sample, tuple), type(sample)
    assert len(sample) == len(_AOTI_EXAMPLE_INPUT_KEYS), len(sample)
    example_inputs = tuple(t.detach().cpu().contiguous() for t in sample)
    for key, t in zip(_AOTI_EXAMPLE_INPUT_KEYS, example_inputs, strict=True):
        assert isinstance(t, torch.Tensor), (key, type(t))
    assert tuple(example_inputs[0].shape) == (
        1,
        1,
        ORBIT_MAX_PLANETS,
        ORBIT_PLANET_FEATURES,
    ), tuple(example_inputs[0].shape)
    assert tuple(example_inputs[1].shape) == (
        1,
        1,
        ORBIT_MAX_PLANETS,
        ORBIT_PLANET_ARRIVAL_HORIZON,
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_TEMPORAL_FEATURES,
    ), tuple(example_inputs[1].shape)
    assert tuple(example_inputs[2].shape) == (
        1,
        1,
        ORBIT_PLAYER_AXIS_SLOTS - 1,
    ), tuple(example_inputs[2].shape)
    assert tuple(example_inputs[3].shape) == (1, 1, ORBIT_MAX_PLANETS), tuple(
        example_inputs[3].shape
    )
    assert tuple(example_inputs[4].shape) == (
        1,
        1,
        ORBIT_PLANET_PAIRWISE_COUNT,
    ), tuple(example_inputs[4].shape)
    assert tuple(example_inputs[5].shape) == (
        1,
        1,
        ORBIT_PLANET_PAIRWISE_COUNT,
        ORBIT_EDGE_FEATURES,
    ), tuple(example_inputs[5].shape)
    assert tuple(example_inputs[6].shape) == (
        1,
        1,
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), tuple(example_inputs[6].shape)
    return example_inputs


def _build_aoti_policy_runner_cpp(out_dir: Path) -> None:
    source = out_dir / "aoti_policy_runner_cpp.cpp"
    assert source.is_file(), source
    build_dir = Path("/tmp/aoti_policy_runner_cpp_build").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    mod = load(
        name="aoti_policy_runner_cpp",
        sources=[str(source)],
        extra_cflags=[
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            f"-march={os.environ['ORBIT_SUBMISSION_CPU_MARCH']}",
        ],
        with_cuda=False,
        verbose=False,
        build_directory=str(build_dir),
    )
    mod_file = Path(mod.__file__).resolve()
    assert mod_file.is_file(), mod_file
    dst = out_dir / mod_file.name
    shutil.copy2(mod_file, dst)
    subprocess.run(["strip", "--strip-unneeded", str(dst)], check=True)


def _extract_aoti_model_so(package_path: Path, model_so_path: Path) -> None:
    assert package_path.is_file(), package_path
    with zipfile.ZipFile(package_path) as zf:
        so_names = tuple(name for name in zf.namelist() if name.endswith(".so"))
        assert len(so_names) == 1, so_names
        with zf.open(so_names[0]) as src, model_so_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    model_so_path.chmod(0o755)
    subprocess.run(["strip", "--strip-unneeded", str(model_so_path)], check=True)
    package_path.unlink()


source_checkpoint_mode = os.environ["ORBIT_SUBMISSION_SOURCE_CKPT_MODE"]
assert source_checkpoint_mode in (
    "shared",
    "by_num_agents",
    "by_num_agents_with_overage_primary",
), source_checkpoint_mode
if source_checkpoint_mode == "shared":
    source_checkpoints = (
        ("", 4, Path(os.environ["ORBIT_SUBMISSION_SOURCE_CKPT"]).resolve()),
    )
elif source_checkpoint_mode == "by_num_agents":
    source_checkpoints = (
        ("_2p", 2, Path(os.environ["ORBIT_SUBMISSION_SOURCE_CKPT_2P"]).resolve()),
        ("_4p", 4, Path(os.environ["ORBIT_SUBMISSION_SOURCE_CKPT_4P"]).resolve()),
    )
else:
    source_checkpoints = (
        ("_2p", 2, Path(os.environ["ORBIT_SUBMISSION_SOURCE_CKPT_2P"]).resolve()),
        ("_4p", 4, Path(os.environ["ORBIT_SUBMISSION_SOURCE_CKPT_4P"]).resolve()),
        (
            "_overage_primary",
            4,
            Path(os.environ["ORBIT_SUBMISSION_SOURCE_CKPT_OVERAGE_PRIMARY"]).resolve(),
        ),
    )
staging = Path(os.environ["STAGING"]).resolve()
dynamic_quantize_raw = os.environ["ORBIT_SUBMISSION_DYNAMIC_QUANTIZE"]
torchscript_raw = os.environ["ORBIT_SUBMISSION_TORCHSCRIPT"]
aoti_raw = os.environ["ORBIT_SUBMISSION_AOTI"]
is_sample_raw = os.environ["ORBIT_SUBMISSION_IS_SAMPLE"]
shuffle_identity_ids_raw = os.environ["ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS"]
aoti_example_inputs_raw = os.environ["ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS"]
assert dynamic_quantize_raw in ("0", "1"), dynamic_quantize_raw
assert torchscript_raw in ("0", "1"), torchscript_raw
assert aoti_raw in ("0", "1"), aoti_raw
assert is_sample_raw in ("0", "1"), is_sample_raw
assert shuffle_identity_ids_raw in ("0", "1"), shuffle_identity_ids_raw
dynamic_quantize = dynamic_quantize_raw == "1"
torchscript = torchscript_raw == "1"
aoti = aoti_raw == "1"
is_sample = is_sample_raw == "1"
shuffle_identity_ids = shuffle_identity_ids_raw == "1"
assert not (torchscript and aoti)

out_dir = staging / "kaggle_submission"
if aoti:
    assert aoti_example_inputs_raw != "", "ORBIT_SUBMISSION_AOTI requires replay example inputs"
    _build_aoti_policy_runner_cpp(out_dir)

for suffix, validation_num_agents, source_ckpt in source_checkpoints:
    assert suffix in ("", "_2p", "_4p", "_overage_primary"), suffix
    assert int(validation_num_agents) in (2, 4), validation_num_agents
    assert source_ckpt.is_file(), source_ckpt
    if aoti:
        package_path = out_dir / f"checkpoint_policy_logits_aoti{suffix}.pt2"
        debug_args = [
            sys.executable,
            "-u",
            str(staging / "kaggle_submission" / "debug_aoti_package.py"),
            str(source_ckpt),
            aoti_example_inputs_raw,
            "--package-path",
            str(package_path),
            "--compare-samples",
            "50",
            "--kaggle-env-steps",
            "5",
            "--kaggle-env-agents",
            str(validation_num_agents),
        ]
        if dynamic_quantize:
            debug_args.append("--dynamic-quantize")
        print(
            f"build_optimized_submission_artifact_exec_debug_aoti suffix={suffix!r} validation_num_agents={validation_num_agents} args={debug_args}",
            flush=True,
        )
        subprocess.run(debug_args, check=True)
        _extract_aoti_model_so(package_path, out_dir / f"checkpoint_policy_logits_aoti{suffix}.so")
    elif torchscript:
        source_ckpt_payload = _load_checkpoint_payload(source_ckpt)
        model = ImpalaOrbitModel(
            **_submission_model_init_kwargs(source_ckpt_payload["model_config"])
        )
        _load_model_weights(model, source_ckpt_payload)
        model.eval()
        _configure_submission_model_runtime(
            model,
            is_sample=is_sample,
            shuffle_identity_ids=shuffle_identity_ids,
        )
        if dynamic_quantize:
            model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)
            assert isinstance(model, ImpalaOrbitModel), type(model)
            model.eval()
            _configure_submission_model_runtime(
                model,
                is_sample=is_sample,
                shuffle_identity_ids=shuffle_identity_ids,
            )
        wrapper = _OrbitPolicyActionsTraceWrapper(model).eval()
        traced = torch.jit.trace(wrapper, _example_inputs(), strict=True, check_trace=not is_sample)
        frozen = torch.jit.freeze(traced.eval())
        optimized = torch.jit.optimize_for_inference(frozen)
        torch.jit.save(optimized, str(out_dir / f"checkpoint_policy_logits_torchscript{suffix}.pt"))
    else:
        source_ckpt_payload = _load_checkpoint_payload(source_ckpt)
        model = ImpalaOrbitModel(
            **_submission_model_init_kwargs(source_ckpt_payload["model_config"])
        )
        _load_model_weights(model, source_ckpt_payload)
        model.eval()
        _configure_submission_model_runtime(
            model,
            is_sample=is_sample,
            shuffle_identity_ids=shuffle_identity_ids,
        )
        if dynamic_quantize:
            model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)
            assert isinstance(model, ImpalaOrbitModel), type(model)
            model.eval()
            _configure_submission_model_runtime(
                model,
                is_sample=is_sample,
                shuffle_identity_ids=shuffle_identity_ids,
            )
        torch.save(
            {
                "artifact_kind": "impala_eager_module",
                "dynamic_quantized_linear": dynamic_quantize,
                "is_sample": is_sample,
                "shuffle_identity_ids": shuffle_identity_ids,
                "model_config": source_ckpt_payload["model_config"],
                "model": model,
            },
            str(out_dir / f"checkpoint_model_eager{suffix}.pt"),
        )
PY
    echo "Building optimized submission model artifact in Kaggle image..."
    docker run --rm \
      "${SOURCE_CKPT_ENV_ARGS[@]}" \
      -e ORBIT_SUBMISSION_DYNAMIC_QUANTIZE="${ORBIT_SUBMISSION_DYNAMIC_QUANTIZE}" \
      -e ORBIT_SUBMISSION_TORCHSCRIPT="${ORBIT_SUBMISSION_TORCHSCRIPT}" \
      -e ORBIT_SUBMISSION_AOTI="${ORBIT_SUBMISSION_AOTI}" \
      -e ORBIT_SUBMISSION_IS_SAMPLE="${ORBIT_SUBMISSION_IS_SAMPLE}" \
      -e ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS="${ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS}" \
      -e ORBIT_SUBMISSION_AOTI_EXAMPLE_INPUTS="${AOTI_EXAMPLE_INPUTS_CONTAINER}" \
      -e ORBIT_SUBMISSION_FAIL_ENABLED=0 \
      -e TORCHINDUCTOR_FREEZING=1 \
      -e ATEN_CPU_CAPABILITY="${ORBIT_SUBMISSION_CPU_CAPABILITY}" \
      -e ORBIT_SUBMISSION_CPU_MARCH="${ORBIT_SUBMISSION_CPU_MARCH}" \
      -e CXX=/build/cxx_cpu_target.sh \
      -e CFLAGS="-march=${ORBIT_SUBMISSION_CPU_MARCH}" \
      -e CXXFLAGS="-march=${ORBIT_SUBMISSION_CPU_MARCH}" \
      -e STAGING=/staging \
      -e CPP_BUILD_DIR=/build \
      -e PYTHONPATH=/staging \
      -e PYTHONDONTWRITEBYTECODE=1 \
      "${SOURCE_CKPT_DOCKER_ARGS[@]}" \
      -v "${STAGING}:/staging" \
      -v "${CPP_BUILD_DIR}:/build" \
      "${AOTI_EXAMPLE_INPUTS_DOCKER_ARGS[@]}" \
      -w /staging \
      "${KAGGLE_DOCKER_IMAGE}" \
      bash -lc "${INSTALL_CPU_TORCH_CMD} && python -u /build/build_optimized_submission_artifact.py"
    echo "Built optimized submission model artifact."
  fi
else
  [[ "${ORBIT_SUBMISSION_DYNAMIC_QUANTIZE}" == "0" && "${ORBIT_SUBMISSION_TORCHSCRIPT}" == "0" && "${ORBIT_SUBMISSION_AOTI}" == "0" ]] || {
    echo "Optimization flags require a checkpoint path." >&2
    exit 1
  }
  echo "No checkpoints passed and ORBIT_IMPALA_CHECKPOINT/2P/4P unset; archive has no weights." >&2
  echo "Place checkpoint_weights.pt or checkpoint_weights_2p.pt and checkpoint_weights_4p.pt beside submission.py on Kaggle." >&2
fi

rm -f "${STAGING}/kaggle_submission/debug_aoti_package.py"
rm -f "${STAGING}/kaggle_submission/aoti_policy_runner_cpp.cpp"
tar -I 'gzip -9' -cf "${OUT}" -C "${STAGING}" .

echo "Wrote ${OUT}"
