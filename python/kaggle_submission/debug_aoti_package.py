from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_CPU_THREADS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["TORCHINDUCTOR_FREEZING"] = "1"
_SUBMISSION_IS_SAMPLE_RAW = os.environ.get("ORBIT_SUBMISSION_IS_SAMPLE", "1").strip()
assert _SUBMISSION_IS_SAMPLE_RAW in ("0", "1"), _SUBMISSION_IS_SAMPLE_RAW
_SUBMISSION_IS_SAMPLE = _SUBMISSION_IS_SAMPLE_RAW == "1"
_SUBMISSION_SHUFFLE_IDENTITY_IDS_RAW = os.environ.get(
    "ORBIT_SUBMISSION_SHUFFLE_IDENTITY_IDS",
    "1",
).strip()
assert _SUBMISSION_SHUFFLE_IDENTITY_IDS_RAW in ("0", "1"), _SUBMISSION_SHUFFLE_IDENTITY_IDS_RAW
_SUBMISSION_SHUFFLE_IDENTITY_IDS = _SUBMISSION_SHUFFLE_IDENTITY_IDS_RAW == "1"

import torch
import torch._inductor
import torch.nn as nn
from torch.quantization import quantize_dynamic

_ALLOWED_CPUS = os.sched_getaffinity(0)
assert len(_ALLOWED_CPUS) > 0, _ALLOWED_CPUS
_CPU_CORE = min(_ALLOWED_CPUS)
os.sched_setaffinity(0, {_CPU_CORE})

_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from src.gym.obs_wrapper import (
    ORBIT_EDGE_FEATURES,
    ORBIT_MAX_PLANETS,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
)
from src.models.models import ImpalaOrbitModel, impala_orbit_model_init_kwargs_from_flags
from kaggle_submission.submission import OrbitSubmissionRunner

torch.set_num_threads(_CPU_THREADS)

_AOTI_EXAMPLE_INPUT_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
)
_AOTI_EXAMPLE_INPUT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float32,
    torch.float32,
    torch.float32,
    torch.float32,
    torch.float32,
    torch.int8,
)
_REFERENCE_ENV_PATH = (
    _PY_ROOT
    / "cpp"
    / "orbit_wars"
    / "reference_kaggle_upstream_github_no_edit"
    / "orbit_wars.py"
)
_RUNNER_COMPARE_INPUT_KEYS: tuple[str, ...] = _AOTI_EXAMPLE_INPUT_KEYS


def _log(message: str) -> None:
    print(message, flush=True)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


@dataclass(frozen=True)
class _OutputCompareStats:
    max_abs_finite: float
    max_rel_finite: float
    finite_values: int
    eager_nan: int
    aoti_nan: int
    eager_posinf: int
    aoti_posinf: int
    eager_neginf: int
    aoti_neginf: int
    nonfinite_mismatch: int
    first_nonfinite_mismatch_index: tuple[int, ...] | None
    first_nonfinite_mismatch_eager: float | None
    first_nonfinite_mismatch_aoti: float | None

    def summary(self) -> str:
        parts = [
            f"max_abs_finite={self.max_abs_finite:.9g}",
            f"max_rel_finite={self.max_rel_finite:.9g}",
            f"finite_values={self.finite_values}",
            f"eager_nan={self.eager_nan}",
            f"aoti_nan={self.aoti_nan}",
            f"eager_posinf={self.eager_posinf}",
            f"aoti_posinf={self.aoti_posinf}",
            f"eager_neginf={self.eager_neginf}",
            f"aoti_neginf={self.aoti_neginf}",
            f"nonfinite_mismatch={self.nonfinite_mismatch}",
        ]
        if self.first_nonfinite_mismatch_index is not None:
            parts.extend(
                [
                    f"first_nonfinite_mismatch_index={self.first_nonfinite_mismatch_index}",
                    f"first_nonfinite_mismatch_eager={self.first_nonfinite_mismatch_eager}",
                    f"first_nonfinite_mismatch_aoti={self.first_nonfinite_mismatch_aoti}",
                ]
            )
        return " ".join(parts)


def _submission_model_init_kwargs(model_config: Mapping[str, Any]) -> dict[str, Any]:
    policy_model_config = copy.deepcopy(model_config)
    assert isinstance(policy_model_config, dict), type(policy_model_config)
    orbit_impala_config = policy_model_config["orbit_impala"]
    assert isinstance(orbit_impala_config, dict), type(orbit_impala_config)
    orbit_impala_config["use_value_opponent_model_embedding"] = False
    flags = SimpleNamespace(
        model=policy_model_config,
        target_min_entropy={"spawn_fleet": 0.0},
        entropy_floor_max_temperature=10000.0,
        entropy_floor_num_iters=16,
    )
    kw = impala_orbit_model_init_kwargs_from_flags(flags)
    kw["entropy_floor_target"] = (0.0,)
    kw["entropy_floor_max_temperature"] = 10000.0
    kw["entropy_floor_num_iters"] = 16
    kw["include_rl_policy_value_heads"] = True
    kw["include_rl_value_head"] = False
    return kw


_TRAINING_ONLY_STATE_DICT_PREFIXES = (
    "_global_value_head.",
    "_global_value_head_production_delta.",
    "_value_opponent_model_encoder.",
    "_value_opponent_identity_model_fusion.",
)


def _strip_torch_compile_orig_mod_prefix(sd: dict[Any, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    keys = tuple(str(k) for k in sd.keys())
    prefixed = tuple(k.startswith(prefix) for k in keys)
    assert all(prefixed) or not any(prefixed), keys[:8]
    if not any(prefixed):
        return {str(k): v for k, v in sd.items()}
    return {str(k)[len(prefix) :]: v for k, v in sd.items()}


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
    unexpected_checkpoint_only = sorted(
        k
        for k in checkpoint_only
        if not k.startswith(_TRAINING_ONLY_STATE_DICT_PREFIXES)
    )
    assert not unexpected_checkpoint_only, unexpected_checkpoint_only
    load_sd = {k: v for k, v in sd.items() if k in model_sd}
    missing = sorted(k for k in model_sd if k not in load_sd)
    assert not missing, missing
    for k, v in load_sd.items():
        assert isinstance(v, torch.Tensor), (k, type(v))
        assert v.shape == model_sd[k].shape, (k, tuple(v.shape), tuple(model_sd[k].shape))
    model.load_state_dict(load_sd, strict=True)


def _configure_submission_model_runtime(model: ImpalaOrbitModel) -> None:
    zero_heads = torch.zeros(1, dtype=torch.float32)
    model.set_entropy_floor_targets(zero_heads)
    model.set_is_sample(_SUBMISSION_IS_SAMPLE)
    model.set_compile_friendly_sample(True)
    model.set_shuffle_identity_ids(_SUBMISSION_SHUFFLE_IDENTITY_IDS)


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


def _load_eager_wrapper(
    checkpoint_path: Path,
    *,
    dynamic_quantize: bool,
) -> _OrbitPolicyActionsTraceWrapper:
    model = _load_eager_model(checkpoint_path, dynamic_quantize=dynamic_quantize)
    wrapper = _OrbitPolicyActionsTraceWrapper(model).eval()
    return wrapper


def _load_eager_model(
    checkpoint_path: Path,
    *,
    dynamic_quantize: bool,
) -> nn.Module:
    ckpt = _load_checkpoint_payload(checkpoint_path)
    model = ImpalaOrbitModel(**_submission_model_init_kwargs(ckpt["model_config"]))
    _load_model_weights(model, ckpt)
    model.eval()
    _configure_submission_model_runtime(model)
    if bool(dynamic_quantize):
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)
        assert isinstance(model, ImpalaOrbitModel), type(model)
        model.eval()
        _configure_submission_model_runtime(model)
    return model


def _assert_example_input_contract(example_inputs: tuple[torch.Tensor, ...]) -> None:
    assert len(example_inputs) == len(_AOTI_EXAMPLE_INPUT_KEYS), len(example_inputs)
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
    for key, t, dtype in zip(
        _AOTI_EXAMPLE_INPUT_KEYS,
        example_inputs,
        _AOTI_EXAMPLE_INPUT_DTYPES,
        strict=True,
    ):
        assert t.dtype == dtype, (key, t.dtype, dtype)
        assert not t.is_cuda, (key, t.device)
        expected_stride = torch.empty(tuple(t.shape), dtype=t.dtype).stride()
        assert tuple(t.stride()) == tuple(expected_stride), (
            key,
            tuple(t.shape),
            tuple(t.stride()),
            tuple(expected_stride),
        )


def _canonicalize_aoti_inputs(sample: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    assert len(sample) == len(_AOTI_EXAMPLE_INPUT_KEYS), len(sample)
    out: list[torch.Tensor] = []
    for key, t, dtype in zip(
        _AOTI_EXAMPLE_INPUT_KEYS,
        sample,
        _AOTI_EXAMPLE_INPUT_DTYPES,
        strict=True,
    ):
        assert isinstance(t, torch.Tensor), (key, type(t))
        src = t.detach().cpu()
        dst = torch.empty(tuple(src.shape), dtype=dtype)
        dst.copy_(src.to(dtype=dtype))
        out.append(dst)
    canonical = tuple(out)
    _assert_example_input_contract(canonical)
    return canonical


def _load_aoti_capture(path: Path) -> tuple[tuple[torch.Tensor, ...], ...]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(payload, dict), type(payload)
    assert payload["format"] == "orbit_policy_logits_aoti_example_inputs_v1", payload["format"]
    assert tuple(payload["input_keys"]) == _AOTI_EXAMPLE_INPUT_KEYS, payload["input_keys"]
    assert int(payload["samples_per_step"]) == 4, payload["samples_per_step"]
    samples = payload["samples"]
    assert isinstance(samples, list), type(samples)
    assert len(samples) > 0, "AOTI example input capture contains no samples"
    out: list[tuple[torch.Tensor, ...]] = []
    for sample in samples:
        assert isinstance(sample, tuple), type(sample)
        assert len(sample) == len(_AOTI_EXAMPLE_INPUT_KEYS), len(sample)
        example_inputs = _canonicalize_aoti_inputs(sample)
        for key, t in zip(_AOTI_EXAMPLE_INPUT_KEYS, example_inputs, strict=True):
            assert isinstance(t, torch.Tensor), (key, type(t))
        out.append(example_inputs)
    return tuple(out)


def _eager_forward(
    wrapper: _OrbitPolicyActionsTraceWrapper,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    with torch.no_grad():
        out = wrapper(*example_inputs)
    assert isinstance(out, torch.Tensor), type(out)
    return out


def _timed_eager_forward(
    wrapper: _OrbitPolicyActionsTraceWrapper,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    out = _eager_forward(wrapper, example_inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, elapsed_ms


def _aoti_forward(
    aoti_model: Any,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out = aoti_model(*example_inputs)
    assert isinstance(out, torch.Tensor), type(out)
    return out


def _timed_aoti_forward(
    aoti_model: Any,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    out = _aoti_forward(aoti_model, example_inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, elapsed_ms


def _assert_outputs_close(
    *,
    eager_out: torch.Tensor,
    aoti_out: torch.Tensor,
    rtol: float,
    atol: float,
    label: str,
) -> _OutputCompareStats:
    assert eager_out.shape == aoti_out.shape, (label, eager_out.shape, aoti_out.shape)
    eager = eager_out.detach().cpu()
    aoti = aoti_out.detach().cpu()

    eager_finite = torch.isfinite(eager)
    aoti_finite = torch.isfinite(aoti)
    finite_both = eager_finite & aoti_finite
    finite_values = int(finite_both.sum().item())
    assert finite_values > 0, label

    finite_abs_diff = (eager[finite_both] - aoti[finite_both]).abs()
    max_abs_finite = float(finite_abs_diff.max().item())
    finite_denom = torch.maximum(eager[finite_both].abs(), torch.full_like(eager[finite_both], atol))
    max_rel_finite = float((finite_abs_diff / finite_denom).max().item())

    eager_nan_mask = torch.isnan(eager)
    aoti_nan_mask = torch.isnan(aoti)
    eager_posinf_mask = eager == torch.inf
    aoti_posinf_mask = aoti == torch.inf
    eager_neginf_mask = eager == -torch.inf
    aoti_neginf_mask = aoti == -torch.inf
    same_nonfinite = (
        (eager_nan_mask & aoti_nan_mask)
        | (eager_posinf_mask & aoti_posinf_mask)
        | (eager_neginf_mask & aoti_neginf_mask)
    )
    nonfinite_union = ~eager_finite | ~aoti_finite
    nonfinite_mismatch_mask = nonfinite_union & ~same_nonfinite
    nonfinite_mismatch = int(nonfinite_mismatch_mask.sum().item())

    first_nonfinite_mismatch_index: tuple[int, ...] | None = None
    first_nonfinite_mismatch_eager: float | None = None
    first_nonfinite_mismatch_aoti: float | None = None
    if nonfinite_mismatch > 0:
        flat_bad_idx = int(torch.argmax(nonfinite_mismatch_mask.reshape(-1).to(dtype=torch.int64)).item())
        idx_tensors = torch.unravel_index(torch.tensor(flat_bad_idx), eager.shape)
        first_nonfinite_mismatch_index = tuple(int(v.item()) for v in idx_tensors)
        first_nonfinite_mismatch_eager = float(eager[first_nonfinite_mismatch_index].item())
        first_nonfinite_mismatch_aoti = float(aoti[first_nonfinite_mismatch_index].item())

    stats = _OutputCompareStats(
        max_abs_finite=max_abs_finite,
        max_rel_finite=max_rel_finite,
        finite_values=finite_values,
        eager_nan=int(eager_nan_mask.sum().item()),
        aoti_nan=int(aoti_nan_mask.sum().item()),
        eager_posinf=int(eager_posinf_mask.sum().item()),
        aoti_posinf=int(aoti_posinf_mask.sum().item()),
        eager_neginf=int(eager_neginf_mask.sum().item()),
        aoti_neginf=int(aoti_neginf_mask.sum().item()),
        nonfinite_mismatch=nonfinite_mismatch,
        first_nonfinite_mismatch_index=first_nonfinite_mismatch_index,
        first_nonfinite_mismatch_eager=first_nonfinite_mismatch_eager,
        first_nonfinite_mismatch_aoti=first_nonfinite_mismatch_aoti,
    )
    torch.testing.assert_close(aoti, eager, rtol=rtol, atol=atol, msg=label)
    return stats


def _assert_outputs_same_contract(
    *,
    eager_out: torch.Tensor,
    aoti_out: torch.Tensor,
    label: str,
) -> None:
    assert eager_out.shape == aoti_out.shape, (label, eager_out.shape, aoti_out.shape)
    assert eager_out.dtype == aoti_out.dtype, (label, eager_out.dtype, aoti_out.dtype)


def _compile_aoti_package(
    *,
    wrapper: _OrbitPolicyActionsTraceWrapper,
    example_inputs: tuple[torch.Tensor, ...],
    package_path: Path,
) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    if package_path.exists():
        package_path.unlink()
    t0 = time.perf_counter()
    _log("export_start")
    exported = torch.export.export(wrapper, example_inputs)
    _log(f"export_done elapsed={_format_seconds(time.perf_counter() - t0)}")
    compile_t0 = time.perf_counter()
    _log("aoti_compile_start")
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(package_path),
        inductor_configs={
            "max_autotune": False,
        },
    )
    assert package_path.is_file(), package_path
    _log(f"aoti_compile_done elapsed={_format_seconds(time.perf_counter() - compile_t0)}")


def _load_reference_env_module() -> Any:
    assert _REFERENCE_ENV_PATH.is_file(), _REFERENCE_ENV_PATH
    spec = importlib.util.spec_from_file_location(
        "orbit_wars_reference_kaggle_upstream_github_no_edit_for_aoti_debug",
        _REFERENCE_ENV_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "specification")
    assert hasattr(module, "interpreter")
    assert hasattr(module, "renderer")
    assert hasattr(module, "html_renderer")
    return module


def _make_reference_orbit_wars_env(
    *,
    episode_steps: int,
    seed: int,
    debug: bool,
) -> Any:
    make = importlib.import_module("kaggle_environments").make
    module = _load_reference_env_module()
    local_env = {
        "specification": copy.deepcopy(module.specification),
        "interpreter": module.interpreter,
        "renderer": module.renderer,
        "html_renderer": module.html_renderer,
    }
    return make(
        local_env,
        configuration={
            "episodeSteps": int(episode_steps),
            "seed": int(seed),
        },
        debug=debug,
    )


def _factorized_logits_from_runner_output(model_output: dict[str, Any]) -> torch.Tensor:
    final_policy_logits = model_output["final_policy_logits_LEARN"]
    assert isinstance(final_policy_logits, dict), type(final_policy_logits)
    policy_logits = final_policy_logits["spawn_fleet"]
    assert isinstance(policy_logits, torch.Tensor), type(policy_logits)
    return policy_logits


def _aoti_inputs_from_batch_obs(batch_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    sample = tuple(batch_obs[key] for key in _AOTI_EXAMPLE_INPUT_KEYS)
    _assert_example_input_contract(sample)
    return sample


def _tensor_brief(t: torch.Tensor) -> str:
    assert isinstance(t, torch.Tensor), type(t)
    flat = t.detach().cpu().reshape(-1)
    assert flat.numel() > 0, tuple(t.shape)
    if t.dtype == torch.bool:
        numeric = flat.to(dtype=torch.int64)
        min_v = int(numeric.min().item())
        max_v = int(numeric.max().item())
    elif torch.is_floating_point(t):
        finite = torch.isfinite(flat)
        finite_count = int(finite.sum().item())
        if finite_count > 0:
            finite_flat = flat[finite]
            min_v = float(finite_flat.min().item())
            max_v = float(finite_flat.max().item())
        else:
            min_v = "no_finite"
            max_v = "no_finite"
    else:
        min_v = int(flat.min().item())
        max_v = int(flat.max().item())
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} stride={tuple(t.stride())} "
        f"contiguous={t.is_contiguous()} min={min_v} max={max_v}"
    )


def _tensor_max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert isinstance(a, torch.Tensor), type(a)
    assert isinstance(b, torch.Tensor), type(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype == torch.bool or b.dtype == torch.bool:
        diff = a.to(dtype=torch.int64) - b.to(dtype=torch.int64)
    else:
        diff = a.to(dtype=torch.float64) - b.to(dtype=torch.float64)
    return float(diff.abs().max().item())


def _sample_total_diff(
    env_inputs: tuple[torch.Tensor, ...],
    sample: tuple[torch.Tensor, ...],
) -> float:
    assert len(env_inputs) == len(_AOTI_EXAMPLE_INPUT_KEYS), len(env_inputs)
    assert len(sample) == len(_AOTI_EXAMPLE_INPUT_KEYS), len(sample)
    total = 0.0
    for env_t, sample_t in zip(env_inputs, sample, strict=True):
        total += _tensor_max_abs_diff(env_t, sample_t)
    return total


def _log_env_input_replay_diff(
    *,
    seat: int,
    call_index: int,
    env_inputs: tuple[torch.Tensor, ...],
    replay_samples: tuple[tuple[torch.Tensor, ...], ...],
) -> None:
    assert len(replay_samples) > 0, "replay samples required for env input diagnostics"
    best_index = 0
    best_total = _sample_total_diff(env_inputs, replay_samples[0])
    exact_index = -1
    for i, sample in enumerate(replay_samples):
        exact = all(torch.equal(a, b) for a, b in zip(env_inputs, sample, strict=True))
        if exact:
            exact_index = int(i)
            best_index = int(i)
            best_total = 0.0
            break
        if i > 0:
            total = _sample_total_diff(env_inputs, sample)
            if total < best_total:
                best_index = int(i)
                best_total = float(total)
    nearest = replay_samples[best_index]
    _log(
        f"kaggle_env_input_vs_replay seat={int(seat)} call={int(call_index)} "
        f"exact_sample_index={exact_index} nearest_sample_index={best_index} "
        f"nearest_total_max_abs_sum={best_total:.9g}"
    )
    for key, env_t, replay_t in zip(_AOTI_EXAMPLE_INPUT_KEYS, env_inputs, nearest, strict=True):
        max_abs = _tensor_max_abs_diff(env_t, replay_t)
        equal = torch.equal(env_t, replay_t)
        _log(
            f"kaggle_env_input_key key={key} equal_nearest={equal} "
            f"max_abs_vs_nearest={max_abs:.9g} env={_tensor_brief(env_t)} "
            f"nearest={_tensor_brief(replay_t)}"
        )


class _KaggleEnvAotiCompareAgent:
    def __init__(
        self,
        *,
        seat: int,
        eager_model: nn.Module,
        aoti_model: Any,
        replay_samples: tuple[tuple[torch.Tensor, ...], ...],
        rtol: float,
        atol: float,
    ) -> None:
        self._seat = int(seat)
        self._eager_runner = OrbitSubmissionRunner(
            model=eager_model,
            model_artifact_kind="impala_eager",
            device=torch.device("cpu"),
            emit_wall_profile_summary=False,
        )
        self._aoti_runner = OrbitSubmissionRunner(
            model=aoti_model,
            model_artifact_kind="policy_logits_aoti",
            device=torch.device("cpu"),
            emit_wall_profile_summary=False,
        )
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._replay_samples = replay_samples
        self._calls = 0

    def __call__(self, observation: Any, configuration: Any = None) -> list[Any]:
        call_i = int(self._calls)
        self._calls = call_i + 1
        eager_result = self._eager_runner.step(observation, configuration)
        env_aoti_inputs = _aoti_inputs_from_batch_obs(eager_result.batch_obs)
        _log_env_input_replay_diff(
            seat=self._seat,
            call_index=call_i,
            env_inputs=env_aoti_inputs,
            replay_samples=self._replay_samples,
        )
        aoti_result = self._aoti_runner.step(observation, configuration)

        for key in _RUNNER_COMPARE_INPUT_KEYS:
            eager_t = eager_result.batch_obs[key]
            aoti_t = aoti_result.batch_obs[key]
            assert isinstance(eager_t, torch.Tensor), (key, type(eager_t))
            assert isinstance(aoti_t, torch.Tensor), (key, type(aoti_t))
            assert eager_t.shape == aoti_t.shape, (
                self._seat,
                call_i,
                key,
                eager_t.shape,
                aoti_t.shape,
            )
            assert eager_t.dtype == aoti_t.dtype, (
                self._seat,
                call_i,
                key,
                eager_t.dtype,
                aoti_t.dtype,
            )
            assert torch.equal(eager_t, aoti_t), (self._seat, call_i, key)

        if not _SUBMISSION_IS_SAMPLE:
            assert torch.equal(eager_result.classes, aoti_result.classes), (
                self._seat,
                call_i,
                eager_result.classes,
                aoti_result.classes,
            )
            assert eager_result.moves == aoti_result.moves, (
                self._seat,
                call_i,
                eager_result.moves,
                aoti_result.moves,
            )
        if (call_i + 1) % 10 == 0:
            _log(
                f"kaggle_env_compare_progress seat={self._seat} calls={call_i + 1} "
                f"is_sample={int(_SUBMISSION_IS_SAMPLE)}"
            )
        return aoti_result.moves


def _validate_aoti_on_kaggle_env(
    *,
    eager_model: nn.Module,
    aoti_model: Any,
    replay_samples: tuple[tuple[torch.Tensor, ...], ...],
    episode_steps: int,
    num_agents: int,
    seed: int,
    rtol: float,
    atol: float,
) -> None:
    assert int(episode_steps) > 0, episode_steps
    assert int(num_agents) in (2, 4), num_agents
    _log(
        f"kaggle_env_compare_start agents={int(num_agents)} "
        f"episode_steps={int(episode_steps)} seed={int(seed)}"
    )
    env = _make_reference_orbit_wars_env(
        episode_steps=int(episode_steps),
        seed=int(seed),
        debug=True,
    )
    agents = [
        _KaggleEnvAotiCompareAgent(
            seat=seat,
            eager_model=eager_model,
            aoti_model=aoti_model,
            replay_samples=replay_samples,
            rtol=float(rtol),
            atol=float(atol),
        )
        for seat in range(int(num_agents))
    ]
    t0 = time.perf_counter()
    env.run(agents)
    _log(f"kaggle_env_compare_done elapsed={_format_seconds(time.perf_counter() - t0)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locally package ImpalaOrbitModel policy logits with AOTI and compare to eager CPU."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("example_inputs", type=Path)
    parser.add_argument(
        "--package-path",
        type=Path,
        default=Path("checkpoint_policy_logits_aoti_debug.pt2"),
    )
    parser.add_argument("--capture-step", type=int, default=50)
    parser.add_argument("--compare-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dynamic-quantize", action="store_true")
    parser.add_argument("--kaggle-env-steps", type=int, default=0)
    parser.add_argument("--kaggle-env-agents", type=int, default=4)
    parser.add_argument("--kaggle-env-seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    assert args.checkpoint.is_file(), args.checkpoint
    assert args.example_inputs.is_file(), args.example_inputs
    assert int(args.capture_step) >= 0, args.capture_step
    assert int(args.compare_samples) >= 0, args.compare_samples
    assert int(args.progress_every) > 0, args.progress_every
    assert int(args.kaggle_env_steps) >= 0, args.kaggle_env_steps
    assert int(args.kaggle_env_agents) in (2, 4), args.kaggle_env_agents
    assert float(args.rtol) >= 0.0, args.rtol
    assert float(args.atol) >= 0.0, args.atol
    return args


def main() -> None:
    args = _parse_args()
    script_t0 = time.perf_counter()
    _log("load_capture_start")
    samples = _load_aoti_capture(args.example_inputs)
    _log(f"load_capture_done elapsed={_format_seconds(time.perf_counter() - script_t0)}")
    sample_index = int(args.capture_step) * 4
    assert sample_index < len(samples), (args.capture_step, sample_index, len(samples))
    compile_inputs = samples[sample_index]

    _log(f"cpu_core={_CPU_CORE} torch_num_threads={torch.get_num_threads()}")
    _log(f"loaded_samples={len(samples)} compile_sample_index={sample_index}")

    load_model_t0 = time.perf_counter()
    _log(f"load_eager_model_start dynamic_quantize={bool(args.dynamic_quantize)}")
    eager_model = _load_eager_model(
        args.checkpoint,
        dynamic_quantize=bool(args.dynamic_quantize),
    )
    wrapper = _OrbitPolicyActionsTraceWrapper(eager_model).eval()
    _log(f"load_eager_model_done elapsed={_format_seconds(time.perf_counter() - load_model_t0)}")
    _log("eager_compile_sample_forward_start")
    eager_compile_out, eager_compile_sample_ms = _timed_eager_forward(wrapper, compile_inputs)
    _log(f"eager_compile_sample_forward_done inference_ms={eager_compile_sample_ms:.6f}")
    _compile_aoti_package(
        wrapper=wrapper,
        example_inputs=compile_inputs,
        package_path=args.package_path,
    )
    _log(f"wrote_aoti_package={args.package_path}")

    load_aoti_t0 = time.perf_counter()
    _log("load_aoti_package_start")
    aoti_model = torch._inductor.aoti_load_package(str(args.package_path))
    _log(f"load_aoti_package_done elapsed={_format_seconds(time.perf_counter() - load_aoti_t0)}")
    _log("aoti_compile_sample_forward_start")
    aoti_compile_out, aoti_compile_sample_ms = _timed_aoti_forward(aoti_model, compile_inputs)
    _log(f"aoti_compile_sample_forward_done inference_ms={aoti_compile_sample_ms:.6f}")
    _assert_outputs_same_contract(
        eager_out=eager_compile_out,
        aoti_out=aoti_compile_out,
        label=f"compile sample index {sample_index}",
    )
    compile_stats = None
    if not _SUBMISSION_IS_SAMPLE:
        compile_stats = _assert_outputs_close(
            eager_out=eager_compile_out,
            aoti_out=aoti_compile_out,
            rtol=float(args.rtol),
            atol=float(args.atol),
            label=f"compile sample index {sample_index}",
        )
    max_abs_finite = 0.0 if compile_stats is None else compile_stats.max_abs_finite
    max_rel_finite = 0.0 if compile_stats is None else compile_stats.max_rel_finite
    nonfinite_mismatch = 0 if compile_stats is None else compile_stats.nonfinite_mismatch
    _log(
        f"compile_sample_inference eager_ms={eager_compile_sample_ms:.6f} "
        f"aoti_ms={aoti_compile_sample_ms:.6f} "
        f"speedup={eager_compile_sample_ms / aoti_compile_sample_ms:.4f}x "
        f"max_abs_finite={max_abs_finite:.9g} "
        f"max_rel_finite={max_rel_finite:.9g} "
        f"nonfinite_mismatch={nonfinite_mismatch}"
    )
    if compile_stats is not None:
        _log(f"compile_sample_close {compile_stats.summary()}")

    compare_count = len(samples)
    if int(args.compare_samples) > 0:
        compare_count = int(args.compare_samples)
    assert compare_count <= len(samples), (compare_count, len(samples))
    _log(f"replay_compare_start samples={compare_count} progress_every={int(args.progress_every)}")
    replay_t0 = time.perf_counter()
    worst_abs = 0.0
    worst_rel = 0.0
    worst_nonfinite_mismatch = 0
    eager_replay_total_ms = 0.0
    aoti_replay_total_ms = 0.0
    for i in range(compare_count):
        iter_t0 = time.perf_counter()
        eager_out, eager_ms = _timed_eager_forward(wrapper, samples[i])
        aoti_out, aoti_ms = _timed_aoti_forward(aoti_model, samples[i])
        eager_replay_total_ms += eager_ms
        aoti_replay_total_ms += aoti_ms
        _assert_outputs_same_contract(
            eager_out=eager_out,
            aoti_out=aoti_out,
            label=f"replay sample index {i}",
        )
        if not _SUBMISSION_IS_SAMPLE:
            stats = _assert_outputs_close(
                eager_out=eager_out,
                aoti_out=aoti_out,
                rtol=float(args.rtol),
                atol=float(args.atol),
                label=f"replay sample index {i}",
            )
            worst_abs = max(worst_abs, stats.max_abs_finite)
            worst_rel = max(worst_rel, stats.max_rel_finite)
            worst_nonfinite_mismatch = max(worst_nonfinite_mismatch, stats.nonfinite_mismatch)
        if (i + 1) % int(args.progress_every) == 0 or i + 1 == compare_count:
            elapsed = time.perf_counter() - replay_t0
            if _SUBMISSION_IS_SAMPLE:
                _log(
                    f"replay_compare_progress done={i + 1}/{compare_count} "
                    f"last_eager_ms={eager_ms:.6f} "
                    f"last_aoti_ms={aoti_ms:.6f} "
                    f"avg_eager_ms={eager_replay_total_ms / float(i + 1):.6f} "
                    f"avg_aoti_ms={aoti_replay_total_ms / float(i + 1):.6f} "
                    f"last_sample_elapsed={_format_seconds(time.perf_counter() - iter_t0)} "
                    f"total_elapsed={_format_seconds(elapsed)}"
                )
            else:
                _log(
                    f"replay_compare_progress done={i + 1}/{compare_count} "
                    f"last_eager_ms={eager_ms:.6f} "
                    f"last_aoti_ms={aoti_ms:.6f} "
                    f"avg_eager_ms={eager_replay_total_ms / float(i + 1):.6f} "
                    f"avg_aoti_ms={aoti_replay_total_ms / float(i + 1):.6f} "
                    f"last_abs={stats.max_abs_finite:.9g} "
                    f"last_rel={stats.max_rel_finite:.9g} "
                    f"worst_abs={worst_abs:.9g} "
                    f"worst_rel={worst_rel:.9g} "
                    f"last_nonfinite_mismatch={stats.nonfinite_mismatch} "
                    f"worst_nonfinite_mismatch={worst_nonfinite_mismatch} "
                    f"last_sample_elapsed={_format_seconds(time.perf_counter() - iter_t0)} "
                    f"total_elapsed={_format_seconds(elapsed)}"
                )
    _log(
        f"replay_close samples={compare_count} worst_abs={worst_abs:.9g} "
        f"worst_rel={worst_rel:.9g} worst_nonfinite_mismatch={worst_nonfinite_mismatch}"
    )
    _log(
        f"replay_inference_avg samples={compare_count} "
        f"eager_ms={eager_replay_total_ms / float(compare_count):.6f} "
        f"aoti_ms={aoti_replay_total_ms / float(compare_count):.6f} "
        f"speedup={(eager_replay_total_ms / float(compare_count)) / (aoti_replay_total_ms / float(compare_count)):.4f}x"
    )

    if int(args.kaggle_env_steps) > 0:
        _validate_aoti_on_kaggle_env(
            eager_model=eager_model,
            aoti_model=aoti_model,
            replay_samples=samples,
            episode_steps=int(args.kaggle_env_steps),
            num_agents=int(args.kaggle_env_agents),
            seed=int(args.kaggle_env_seed),
            rtol=float(args.rtol),
            atol=float(args.atol),
        )

    _log(f"script_done elapsed={_format_seconds(time.perf_counter() - script_t0)}")


if __name__ == "__main__":
    main()
