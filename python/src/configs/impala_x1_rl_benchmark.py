"""
x1 benchmark experiment: overrides on top of :func:`configs.base.default_training_config_dict`.

Paths under ``python/artifacts/`` (created at runtime as needed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# ``runpy.run_path`` loads this file as a top-level script (no package); relative imports fail.
_IMPALA_PY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_IMPALA_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPALA_PY_ROOT))

from src.configs.base import (
    ImpalaTrainingConfig,
    deep_merge_dicts,
    default_training_config_dict,
)
from src.configs.impala_x1_rl import IMPALA_X1_RL_OVERRIDES

def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


_POLICY_ENV_KEYS_FROM_MAIN = (
    "model",
    "unroll_length",
    "enable_orbit_estimated_power_early_stop",
    "orbit_actor_target_4p_sample_ratio",
    "enable_envs_validation",
    "enable_actor_wall_tree_profiler",
    "target_min_entropy",
    "entropy_floor_max_temperature",
    "entropy_floor_num_iters",
)


def _policy_env_overrides_from_main() -> dict:
    return {
        key: IMPALA_X1_RL_OVERRIDES[key]
        for key in _POLICY_ENV_KEYS_FROM_MAIN
    }

IMPALA_X1_RL_BENCHMARK_OVERRIDES: dict = {
    **_policy_env_overrides_from_main(),
    "enable_wandb": False,
    "enable_desync": False,

    "num_actors": 64,

    "batch_size": 8,
    "n_actor_envs": 1,

    "inference_batch_size": 4,


    "num_rl_vis_actors": 0,
    "vis_n_actor_envs": 1,

    "num_buffers": 0,
    "num_stats_buffers": 0,
    "num_inference_buffers_train": 64,

    "num_benchmark_inference_workers_cuda0": 8,
    "num_benchmark_inference_workers_cuda1": 8,

    "n_batch_prepare_processes": 0,
    "prepare_batches": 0,
}


def build_training_config() -> SimpleNamespace:
    merged = deep_merge_dicts(default_training_config_dict(), IMPALA_X1_RL_BENCHMARK_OVERRIDES)
    validated = ImpalaTrainingConfig.model_validate(merged)
    out = _ns(validated.model_dump(mode="python"))
    out.resume_checkpoint = validated.resume_checkpoint
    return out
