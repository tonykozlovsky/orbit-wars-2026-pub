import copy
import os
import queue
import time
import torch
import logging
import yaml
from pathlib import Path
from typing import Any
from collections import deque
import multiprocessing as mp
from dataclasses import dataclass
from types import SimpleNamespace
from torch import nn

POLICY_ACTION_KEYS = ("spawn_fleet",)
ENTROPY_HEAD_KEYS = ("spawn_fleet",)
_ENABLE_MODEL_ASSERT = os.environ.get("ENABLE_MODEL_ASSERT", "0") == "1"
_ORBIT_PLAYER_AXIS_SLOTS = 4
_ORBIT_SELF_ENEMY_PLAYER_ORDER_4P = (
    (0, 1, 3, 2),
    (1, 3, 2, 0),
    (2, 0, 1, 3),
    (3, 2, 0, 1),
)
_ORBIT_SELF_ENEMY_POLICY_SLOT_ORDER_2P = (
    (0, 3, -1, -1),
    (-1, -1, -1, -1),
    (-1, -1, -1, -1),
    (3, 0, -1, -1),
)


def orbit_model_by_player_axis_from_seats(
    model_by_seat: torch.Tensor,
    game_num_players: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(model_by_seat, torch.Tensor)
    assert isinstance(game_num_players, torch.Tensor)
    assert model_by_seat.dtype == torch.int64, model_by_seat.dtype
    assert game_num_players.dtype == torch.int64, game_num_players.dtype
    assert model_by_seat.ndim == 2, tuple(model_by_seat.shape)
    e = int(model_by_seat.shape[0])
    assert tuple(model_by_seat.shape) == (e, _ORBIT_PLAYER_AXIS_SLOTS), tuple(model_by_seat.shape)
    assert tuple(game_num_players.shape) == (e,), tuple(game_num_players.shape)
    assert bool(torch.all((game_num_players == 2) | (game_num_players == 4)).item()), (
        game_num_players,
    )

    order_4p = torch.tensor(
        _ORBIT_SELF_ENEMY_PLAYER_ORDER_4P,
        device=model_by_seat.device,
        dtype=torch.long,
    )
    order_2p = torch.tensor(
        _ORBIT_SELF_ENEMY_POLICY_SLOT_ORDER_2P,
        device=model_by_seat.device,
        dtype=torch.long,
    )
    order = torch.where(
        (game_num_players == 2).view(e, 1, 1),
        order_2p.view(1, _ORBIT_PLAYER_AXIS_SLOTS, _ORBIT_PLAYER_AXIS_SLOTS),
        order_4p.view(1, _ORBIT_PLAYER_AXIS_SLOTS, _ORBIT_PLAYER_AXIS_SLOTS),
    )
    assert order.shape == (
        e,
        _ORBIT_PLAYER_AXIS_SLOTS,
        _ORBIT_PLAYER_AXIS_SLOTS,
    ), order.shape
    valid = order >= 0
    gathered = torch.gather(
        model_by_seat.unsqueeze(1).expand(
            e,
            _ORBIT_PLAYER_AXIS_SLOTS,
            _ORBIT_PLAYER_AXIS_SLOTS,
        ),
        dim=2,
        index=order.clamp_min(0),
    )
    out = torch.where(valid, gathered, torch.zeros_like(gathered))
    assert out.shape == (
        e,
        _ORBIT_PLAYER_AXIS_SLOTS,
        _ORBIT_PLAYER_AXIS_SLOTS,
    ), out.shape
    return out


def assert_spawn_fleet_actions_available(model_out: dict, model_in: dict) -> None:
    if not _ENABLE_MODEL_ASSERT:
        return
    assert "actions_LEARN" in model_out
    actions = model_out["actions_LEARN"]
    assert isinstance(actions, dict)
    assert "spawn_fleet" in actions
    spawn_actions = actions["spawn_fleet"]
    assert isinstance(spawn_actions, torch.Tensor)
    assert spawn_actions.dtype == torch.int64, spawn_actions.dtype
    assert spawn_actions.shape[-1] == 1, tuple(spawn_actions.shape)

    assert "obs_LEARN_INFER" in model_in
    obs = model_in["obs_LEARN_INFER"]
    assert isinstance(obs, dict)
    assert "available_action_mask" in obs
    available = obs["available_action_mask"]
    assert isinstance(available, torch.Tensor)
    assert tuple(available.shape[:-1]) == tuple(spawn_actions.shape[:-1]), (
        tuple(available.shape),
        tuple(spawn_actions.shape),
    )
    available_bool = available.to(device=spawn_actions.device, dtype=torch.bool)
    selected_available = torch.gather(available_bool, -1, spawn_actions).squeeze(-1)
    assert bool(selected_available.all().item()), (
        "policy selected unavailable spawn_fleet action",
        tuple(spawn_actions.shape),
    )


_CPU_THREAD_LIMIT_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def configure_process_cpu_thread_limits() -> None:
    """One CPU worker thread pool per process (BLAS/OpenMP/PyTorch). Call at process entry."""
    for key, value in _CPU_THREAD_LIMIT_ENV.items():
        os.environ[key] = value
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _no_weight_decay_module_types() -> tuple[type[nn.Module], ...]:
    """Modules whose parameters should not get L2 (affine/scale/bias-style or embedding table)."""
    types_list: list[type[nn.Module]] = [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LocalResponseNorm,
        nn.Embedding,
    ]
    if hasattr(nn, "SyncBatchNorm"):
        types_list.append(nn.SyncBatchNorm)
    if hasattr(nn, "RMSNorm"):
        types_list.append(nn.RMSNorm)
    return tuple(types_list)


def _global_value_head_trainable_parameter_ids(model: nn.Module) -> frozenset[int]:
    value_head = getattr(model, "_global_value_head", None)
    assert value_head is not None, (
        "optimizer_kwargs value_head_lr requires ImpalaOrbitModel._global_value_head"
    )
    ids = frozenset(id(par) for par in value_head.parameters() if par.requires_grad)
    assert ids, "_global_value_head has no trainable parameters"
    return ids


def _policy_head_trainable_parameter_ids(model: nn.Module) -> frozenset[int]:
    policy_head = getattr(model, "_policy_head", None)
    assert policy_head is not None, (
        "optimizer_kwargs policy_head_lr requires ImpalaOrbitModel._policy_head"
    )
    params = [par for par in policy_head.parameters() if par.requires_grad]
    ids = frozenset(id(par) for par in params)
    assert ids, "policy head has no trainable parameters"
    assert len(ids) == len(params), "policy head parameter set must be unique"
    return ids


def optimizer_param_groups_with_weight_decay(
    model: nn.Module,
    *,
    weight_decay: float,
    lr: float | None = None,
    value_head_lr: float | None = None,
    policy_head_lr: float | None = None,
) -> list[dict[str, Any]]:
    """AdamW/SGD-style: ``weight_decay`` only on weights; 0 on biases and norm/embedding params."""
    assert value_head_lr is None or lr is not None, "value_head_lr requires lr"
    assert policy_head_lr is None or lr is not None, "policy_head_lr requires lr"
    split_value_head = value_head_lr is not None
    split_policy_head = policy_head_lr is not None
    value_head_ids = (
        _global_value_head_trainable_parameter_ids(model) if split_value_head else None
    )
    policy_head_ids = (
        _policy_head_trainable_parameter_ids(model) if split_policy_head else None
    )
    if split_value_head and split_policy_head:
        assert value_head_ids is not None
        assert policy_head_ids is not None
        assert value_head_ids.isdisjoint(policy_head_ids), (
            "value_head_lr and policy_head_lr parameter groups must not overlap"
        )
    base_decay: list[nn.Parameter] = []
    base_no_decay: list[nn.Parameter] = []
    value_decay: list[nn.Parameter] = []
    value_no_decay: list[nn.Parameter] = []
    policy_decay: list[nn.Parameter] = []
    policy_no_decay: list[nn.Parameter] = []
    norm_types = _no_weight_decay_module_types()
    for mod in model.modules():
        for par_name, par in mod.named_parameters(recurse=False):
            if not par.requires_grad:
                continue
            is_value = split_value_head and id(par) in value_head_ids
            is_policy = split_policy_head and id(par) in policy_head_ids
            assert not (is_value and is_policy), "optimizer parameter groups must be disjoint"
            if par_name == "bias" or isinstance(mod, norm_types):
                if is_value:
                    value_no_decay.append(par)
                elif is_policy:
                    policy_no_decay.append(par)
                else:
                    base_no_decay.append(par)
            else:
                if is_value:
                    value_decay.append(par)
                elif is_policy:
                    policy_decay.append(par)
                else:
                    base_decay.append(par)
    wd = float(weight_decay)
    groups: list[dict[str, Any]] = []
    if base_decay:
        g: dict[str, Any] = {"params": base_decay, "weight_decay": wd}
        if split_value_head or split_policy_head:
            g["lr"] = float(lr)
        groups.append(g)
    if base_no_decay:
        g = {"params": base_no_decay, "weight_decay": 0.0}
        if split_value_head or split_policy_head:
            g["lr"] = float(lr)
        groups.append(g)
    if split_value_head:
        vlr = float(value_head_lr)
        assert vlr > 0.0, "value_head_lr must be > 0"
        if value_decay:
            groups.append({"params": value_decay, "weight_decay": wd, "lr": vlr})
        if value_no_decay:
            groups.append({"params": value_no_decay, "weight_decay": 0.0, "lr": vlr})
    if split_policy_head:
        plr = float(policy_head_lr)
        assert plr > 0.0, "policy_head_lr must be > 0"
        if policy_decay:
            groups.append({"params": policy_decay, "weight_decay": wd, "lr": plr})
        if policy_no_decay:
            groups.append({"params": policy_no_decay, "weight_decay": 0.0, "lr": plr})
    assert groups, "model has no trainable parameters"
    return groups


def optimizer_kwargs_to_dict(optimizer_kwargs) -> dict:
    """Config may use plain dicts (YAML) or :class:`~types.SimpleNamespace` (``build_training_config``)."""
    if isinstance(optimizer_kwargs, dict):
        return dict(optimizer_kwargs)
    return vars(optimizer_kwargs)


def build_optimizer_from_config(optimizer_config, model: nn.Module) -> list[torch.optim.Optimizer]:
    """Build optimizer with RL-style param groups (``weight_decay`` only on weight tensors)."""
    if isinstance(optimizer_config, dict):
        name = optimizer_config["optimizer_name"]
        kwargs = dict(optimizer_kwargs_to_dict(optimizer_config["optimizer_kwargs"]))
    else:
        name = optimizer_config.optimizer_name
        kwargs = dict(optimizer_kwargs_to_dict(optimizer_config.optimizer_kwargs))
    wd = float(kwargs.pop("weight_decay", 0.0))
    value_head_lr_raw = kwargs.pop("value_head_lr", None)
    value_head_lr = None if value_head_lr_raw is None else float(value_head_lr_raw)
    policy_head_lr_raw = kwargs.pop("policy_head_lr", None)
    policy_head_lr = None if policy_head_lr_raw is None else float(policy_head_lr_raw)
    base_lr = float(kwargs["lr"])
    param_groups = optimizer_param_groups_with_weight_decay(
        model,
        weight_decay=wd,
        lr=base_lr,
        value_head_lr=value_head_lr,
        policy_head_lr=policy_head_lr,
    )
    if value_head_lr is not None or policy_head_lr is not None:
        kwargs.pop("lr")
    if name == "AdamW":
        return [torch.optim.AdamW(param_groups, **kwargs)]
    if name == "Adam":
        return [torch.optim.Adam(param_groups, **kwargs)]
    if name == "SGD":
        return [torch.optim.SGD(param_groups, **kwargs)]
    if name == "RMSprop":
        return [torch.optim.RMSprop(param_groups, **kwargs)]
    raise ValueError(f"Unsupported optimizer_name: {name!r}")


def impala_compile_enabled() -> bool:
    return os.environ.get("IMPALA_COMPILE", "").strip() == "1"


def compile_impala_model_for_rl(module: nn.Module, *, dynamic: bool = False) -> nn.Module:
    if not impala_compile_enabled():
        return module
    return torch.compile(module, fullgraph=True, dynamic=dynamic)


def strip_torch_compile_orig_mod_prefix(sd: dict[Any, Any]) -> dict[str, Any]:
    """Map ``torch.compile`` checkpoint keys (``_orig_mod.*``) onto an unwrapped module."""
    prefix = "_orig_mod."
    keys = tuple(str(k) for k in sd.keys())
    prefixed = tuple(k.startswith(prefix) for k in keys)
    assert all(prefixed) or not any(prefixed), keys[:8]
    if not any(prefixed):
        return {str(k): v for k, v in sd.items()}
    return {str(k)[len(prefix) :]: v for k, v in sd.items()}


def _state_dict_for_model_keys(model_state: dict[str, Any], state_dict: dict[Any, Any]) -> dict[str, Any]:
    uncompiled_state_dict = strip_torch_compile_orig_mod_prefix(state_dict)
    model_keys = tuple(str(k) for k in model_state.keys())
    prefix = "_orig_mod."
    prefixed = tuple(k.startswith(prefix) for k in model_keys)
    assert all(prefixed) or not any(prefixed), model_keys[:8]
    if not any(prefixed):
        return uncompiled_state_dict
    return {
        f"{prefix}{key}": value
        for key, value in uncompiled_state_dict.items()
    }


_SUPERVISED_CHECKPOINT_ALLOWED_MISSING_STATE_DICT_PREFIXES = (
    "_global_value_head.",
    "_global_value_head_production_delta.",
)


def _checkpoint_config_is_supervised(cfg: Any) -> bool:
    root = _local_dirpath_from_torch_io(cfg.torch_io)
    return "supervised" in root.parts


def load_state_dict_strict(
    model: nn.Module,
    state_dict: dict,
    *,
    allowed_missing_prefixes: tuple[str, ...] = (),
):
    assert isinstance(state_dict, dict), "state_dict must be a dict"
    model_state = model.state_dict()
    state_dict = _state_dict_for_model_keys(model_state, state_dict)
    obsolete_final_keys = sorted(
        str(k) for k in state_dict if str(k).startswith("_policy_head._final_")
    )
    assert not obsolete_final_keys, obsolete_final_keys
    checkpoint_only_keys = sorted(str(k) for k in state_dict if k not in model_state)
    assert not checkpoint_only_keys, (
        "Checkpoint keys not present in model",
        checkpoint_only_keys,
    )
    model_not_loaded_keys = sorted(str(k) for k in model_state if k not in state_dict)
    unexpected_missing_keys = sorted(
        k
        for k in model_not_loaded_keys
        if not k.startswith(allowed_missing_prefixes)
    )
    assert not unexpected_missing_keys, (
        "Model keys not loaded from checkpoint",
        unexpected_missing_keys,
    )
    load_state_dict = dict(state_dict)
    for key in model_not_loaded_keys:
        assert key.startswith(allowed_missing_prefixes), key
        load_state_dict[key] = model_state[key]
    for key, value in load_state_dict.items():
        model_value = model_state[key]
        assert hasattr(value, "shape"), (key, type(value))
        assert hasattr(model_value, "shape"), (key, type(model_value))
        assert tuple(value.shape) == tuple(model_value.shape), (
            key,
            tuple(value.shape),
            tuple(model_value.shape),
        )
    load_result = model.load_state_dict(load_state_dict, strict=True)
    return load_result


def load_state_dict_as_much_as_possible(
    model: nn.Module,
    state_dict: dict,
):
    assert isinstance(state_dict, dict), "state_dict must be a dict"
    model_state = model.state_dict()
    state_dict = _state_dict_for_model_keys(model_state, state_dict)
    load_state_dict = dict(model_state)
    loaded_keys: list[str] = []
    checkpoint_only_keys: list[str] = []
    shape_mismatched_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, value in state_dict.items():
        if key not in model_state:
            checkpoint_only_keys.append(key)
            continue
        model_value = model_state[key]
        assert hasattr(value, "shape"), (key, type(value))
        assert hasattr(model_value, "shape"), (key, type(model_value))
        checkpoint_shape = tuple(value.shape)
        model_shape = tuple(model_value.shape)
        if checkpoint_shape != model_shape:
            shape_mismatched_keys.append((key, checkpoint_shape, model_shape))
            continue
        load_state_dict[key] = value
        loaded_keys.append(key)
    assert loaded_keys, (
        "No checkpoint keys matched the current model",
        sorted(checkpoint_only_keys),
        sorted(shape_mismatched_keys),
    )
    model_only_keys = sorted(str(k) for k in model_state if k not in state_dict)
    logging.warning(
        "Partial checkpoint load: loaded=%d checkpoint_only=%d model_only=%d shape_mismatch=%d",
        len(loaded_keys),
        len(checkpoint_only_keys),
        len(model_only_keys),
        len(shape_mismatched_keys),
    )
    logging.warning(
        "Partial checkpoint load skipped keys: checkpoint_only=%s model_only=%s shape_mismatch=%s",
        sorted(checkpoint_only_keys),
        model_only_keys,
        sorted(shape_mismatched_keys),
    )
    load_result = model.load_state_dict(load_state_dict, strict=True)
    return load_result



def state_dict_fully_matches_model(
    model: nn.Module,
    state_dict: dict,
) -> bool:
    assert isinstance(state_dict, dict), "state_dict must be a dict"
    model_state = model.state_dict()
    state_dict = _state_dict_for_model_keys(model_state, state_dict)
    if set(state_dict.keys()) != set(model_state.keys()):
        return False
    for key, value in state_dict.items():
        model_value = model_state[key]
        assert hasattr(value, "shape"), (key, type(value))
        assert hasattr(model_value, "shape"), (key, type(model_value))
        if tuple(value.shape) != tuple(model_value.shape):
            return False
    return True



def compute_lr_lambda(
    *,
    step: int,
    total_steps: float,
    lr_warmup_steps: int,
    initial_lr_lambda: float,
    final_lr_lambda: float,
) -> float:
    assert float(total_steps) > 0.0, "total_steps must be > 0"
    assert int(lr_warmup_steps) >= 0, "lr_warmup_steps must be >= 0"
    assert float(lr_warmup_steps) <= float(total_steps), "lr_warmup_steps must be <= total_steps"
    step_f = float(max(0, int(step)))
    total_f = float(total_steps)
    if step_f > total_f:
        step_f = total_f
    initial = float(initial_lr_lambda)
    final = float(final_lr_lambda)
    if int(lr_warmup_steps) > 0:
        warmup_f = float(lr_warmup_steps)
        if step_f < warmup_f:
            return initial * (step_f / warmup_f)
        decay_steps = total_f - warmup_f
        if decay_steps <= 0.0:
            return initial
        decay_progress = (step_f - warmup_f) / decay_steps
        return initial + (final - initial) * decay_progress
    decay_progress = step_f / total_f
    return initial + (final - initial) * decay_progress


def compute_target_entropy_combined(
    *,
    step: int,
    total_steps: float,
    warmup_steps: int,
    final_target: float,
    enable_decay: bool = True,
) -> float:
    """Schedule for the entropy target used in the policy entropy loss.

    If ``enable_decay`` is False: ``final_target`` for all steps (``warmup_steps`` is ignored).

    If ``enable_decay`` is True: ramp 0 → ``final_target`` over ``warmup_steps``, then linear decay to 0 by
    ``total_steps``. On ``[0, warmup_steps)``: ``final_target * (step / warmup_steps)``. On
    ``[warmup_steps, total_steps]``: ``final_target * (total_steps - step) / (total_steps - warmup_steps)``.
    If ``warmup_steps`` is 0, there is no ramp: linear decay ``final_target`` → 0 over ``total_steps``.
    If ``warmup_steps == total_steps``, the decay span is empty and the value stays at ``final_target``
    after the ramp (endpoint at ``step == total_steps`` is ``final_target``).
    """
    assert float(total_steps) > 0.0, "total_steps must be > 0"
    assert int(warmup_steps) >= 0, "warmup_steps must be >= 0"
    assert float(warmup_steps) <= float(total_steps), "warmup_steps must be <= total_steps"
    step_f = float(max(0, int(step)))
    total_f = float(total_steps)
    if step_f > total_f:
        step_f = total_f
    target = float(final_target)

    if not enable_decay:
        return target

    if int(warmup_steps) == 0:
        return target * (1.0 - step_f / total_f)

    warmup_f = float(warmup_steps)
    if step_f < warmup_f:
        return target * (step_f / warmup_f)

    decay_span = total_f - warmup_f
    if decay_span <= 0.0:
        return target
    return target * max(0.0, (total_f - step_f) / decay_span)


def _entropy_head_tuple_from_scalar(value: float) -> tuple[float, ...]:
    return tuple(float(value) for _ in ENTROPY_HEAD_KEYS)


def entropy_head_tuple_from_config(raw: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(raw, SimpleNamespace):
        raw = vars(raw)
    assert isinstance(raw, dict), f"{label} must be a dict keyed by entropy head"
    assert set(raw.keys()) == set(ENTROPY_HEAD_KEYS), (
        f"{label} must contain exactly {ENTROPY_HEAD_KEYS}, got {tuple(sorted(raw.keys()))}"
    )
    return tuple(float(raw[head]) for head in ENTROPY_HEAD_KEYS)


def _entropy_head_tuple_from_checkpoint(
    ckpt_file: dict[str, Any],
    key: str,
) -> tuple[float, ...]:
    assert key in ckpt_file, f"Resume checkpoint missing {key}"
    return entropy_head_tuple_from_config(ckpt_file[key], label=key)


def entropy_head_values_to_dict(values: Any) -> dict[str, float]:
    assert len(values) == len(ENTROPY_HEAD_KEYS), (
        f"entropy values length must match {ENTROPY_HEAD_KEYS}: {len(values)}"
    )
    return {
        head: float(values[i])
        for i, head in enumerate(ENTROPY_HEAD_KEYS)
    }


def initial_new_controller_temperature_threshold_from_flags(flags: Any) -> tuple[float, ...]:
    threshold = float(flags.new_controller_temperature_threshold)
    assert threshold > 1.0, threshold
    return _entropy_head_tuple_from_scalar(threshold)


def initial_entropy_ipc_from_flags(
    flags: Any,
    checkpoint_steps: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if bool(flags.classic_entropy) or bool(flags.ce_entropy):
        return (
            _entropy_head_tuple_from_scalar(0.0),
            _entropy_head_tuple_from_scalar(0.0),
            entropy_head_tuple_from_config(
                flags.target_mean_entropy,
                label="target_mean_entropy",
            ),
        )
    init_ema = entropy_head_tuple_from_config(
        flags.target_mean_entropy,
        label="target_mean_entropy",
    )
    max_te = entropy_head_tuple_from_config(
        flags.target_mean_entropy_max,
        label="target_mean_entropy_max",
    )
    if str(flags.target_mean_entropy_mode) == "ema_tracking":
        mult = entropy_head_tuple_from_config(
            flags.mean_entropy_ema_multiplier,
            label="mean_entropy_ema_multiplier",
        )
        assert all(float(v) > 0.0 for v in mult), (
            "mean_entropy_ema_multiplier values must be > 0"
        )
        init_te = tuple(
            min(float(init_ema[i]), float(max_te[i]))
            for i in range(len(ENTROPY_HEAD_KEYS))
        )
        init_ema = tuple(
            float(init_te[i]) / float(mult[i])
            for i in range(len(ENTROPY_HEAD_KEYS))
        )
    else:
        init_te = tuple(
            min(
                compute_target_entropy_combined(
                    step=int(checkpoint_steps),
                    total_steps=float(flags.total_steps),
                    warmup_steps=int(flags.target_entropy_warmup_steps),
                    final_target=float(init_ema[i]),
                    enable_decay=bool(flags.enable_entropy_decay),
                ),
                float(max_te[i]),
            )
            for i in range(len(ENTROPY_HEAD_KEYS))
        )
    if bool(getattr(flags, "use_new_controller", False)):
        init_sf_raw = entropy_head_tuple_from_config(
            flags.new_controller_target_min_entropy,
            label="new_controller_target_min_entropy",
        )
        init_sf_min = entropy_head_tuple_from_config(
            flags.new_controller_target_min_entropy_min_value,
            label="new_controller_target_min_entropy_min_value",
        )
        init_sf = tuple(
            max(float(init_sf_min[i]), min(float(init_sf_raw[i]), 1.0))
            for i in range(len(ENTROPY_HEAD_KEYS))
        )
    else:
        init_sf_raw = entropy_head_tuple_from_config(
            flags.target_min_entropy,
            label="target_min_entropy",
        )
        mn_sf = entropy_head_tuple_from_config(
            flags.shortfall_entropy_min,
            label="shortfall_entropy_min",
        )
        mx_sf = entropy_head_tuple_from_config(
            flags.shortfall_entropy_max,
            label="shortfall_entropy_max",
        )
        init_sf = tuple(
            max(float(mn_sf[i]), min(float(init_sf_raw[i]), float(mx_sf[i])))
            for i in range(len(ENTROPY_HEAD_KEYS))
        )
    return (
        tuple(float(v) for v in init_te),
        tuple(float(v) for v in init_sf),
        tuple(float(v) for v in init_ema),
    )


def new_controller_temperature_threshold_tuple_for_resume(
    ckpt_file: dict[str, Any],
    flags: Any,
) -> tuple[float, ...]:
    if "new_controller_temperature_threshold_by_head" not in ckpt_file:
        return initial_new_controller_temperature_threshold_from_flags(flags)
    threshold = _entropy_head_tuple_from_checkpoint(
        ckpt_file,
        "new_controller_temperature_threshold_by_head",
    )
    assert all(float(v) >= 1.0 for v in threshold), threshold
    return tuple(float(v) for v in threshold)


def entropy_ipc_tuple_for_resume(
    ckpt_file: dict[str, Any],
    flags: Any,
    checkpoint_steps: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if bool(flags.classic_entropy) or bool(flags.ce_entropy):
        _, _, ema = initial_entropy_ipc_from_flags(flags, int(checkpoint_steps))
        return (
            _entropy_head_tuple_from_scalar(0.0),
            _entropy_head_tuple_from_scalar(0.0),
            ema,
        )
    defaults_te, defaults_sf, defaults_ema = initial_entropy_ipc_from_flags(
        flags, int(checkpoint_steps)
    )
    if "shared_target_entropy_by_head" not in ckpt_file:
        return defaults_te, defaults_sf, defaults_ema
    raw_te = _entropy_head_tuple_from_checkpoint(ckpt_file, "shared_target_entropy_by_head")
    raw_ema = _entropy_head_tuple_from_checkpoint(
        ckpt_file, "shared_mean_entropy_ema_by_head"
    )
    max_te = entropy_head_tuple_from_config(
        flags.target_mean_entropy_max,
        label="target_mean_entropy_max",
    )
    te = tuple(
        min(float(raw_te[i]), float(max_te[i]))
        for i in range(len(ENTROPY_HEAD_KEYS))
    )
    ema = tuple(float(v) for v in raw_ema)
    raw_sf = _entropy_head_tuple_from_checkpoint(
        ckpt_file, "shared_shortfall_entropy_by_head"
    )
    mn_sf = entropy_head_tuple_from_config(
        flags.shortfall_entropy_min,
        label="shortfall_entropy_min",
    )
    mx_sf = entropy_head_tuple_from_config(
        flags.shortfall_entropy_max,
        label="shortfall_entropy_max",
    )
    sf = tuple(
        max(float(mn_sf[i]), min(float(raw_sf[i]), float(mx_sf[i])))
        for i in range(len(ENTROPY_HEAD_KEYS))
    )
    return (te, sf, ema)


def ema_alpha_compound_per_step(steps: int, alpha_per_step: float) -> float:
    """EMA blend over ``steps`` steps: ``1 - (1 - alpha_per_step) ** steps`` (same as PopArt / reward EMA)."""
    assert int(steps) > 0, "ema_alpha_compound_per_step: steps must be > 0"
    base = float(alpha_per_step)
    assert 0.0 <= base <= 1.0, "alpha_per_step must be in [0, 1]"
    return 1.0 - (1.0 - base) ** float(steps)


@dataclass
class ProcessWithOptions:
    process: mp.Process
    use_torch_compile: bool = False
    disable_gpu: bool = False


class StopRequested(Exception):
    pass


@dataclass
class _ImmediateWaitWindowState:
    events: deque[tuple[float, bool]]
    immediate_count: int
    next_log_time: float


class RollingImmediateWaitLogger:
    def __init__(
        self,
        *,
        metric_name: str,
        window_sec: float = 10.0 * 60.0,
        log_interval_sec: float = 60.0,
        immediate_threshold_sec: float = 0.002,
    ) -> None:
        assert len(str(metric_name)) > 0, "metric_name must be non-empty"
        assert float(window_sec) > 0.0, "window_sec must be > 0"
        assert float(log_interval_sec) > 0.0, "log_interval_sec must be > 0"
        assert float(immediate_threshold_sec) >= 0.0, "immediate_threshold_sec must be >= 0"
        self._metric_name = str(metric_name)
        self._window_sec = float(window_sec)
        self._log_interval_sec = float(log_interval_sec)
        self._immediate_threshold_sec = float(immediate_threshold_sec)
        self._states: dict[str, _ImmediateWaitWindowState] = {}

    def observe(
        self,
        wait_sec: float,
        *,
        stream_key: str,
        extra_context: str | None = None,
    ) -> None:
        assert float(wait_sec) >= 0.0, "wait_sec must be >= 0"
        assert len(str(stream_key)) > 0, "stream_key must be non-empty"

        now = time.monotonic()
        stream_key = str(stream_key)
        state = self._states.get(stream_key)
        if state is None:
            state = _ImmediateWaitWindowState(
                events=deque(),
                immediate_count=0,
                next_log_time=now + self._log_interval_sec,
            )
            self._states[stream_key] = state

        is_immediate = float(wait_sec) <= self._immediate_threshold_sec
        state.events.append((now, is_immediate))
        if is_immediate:
            state.immediate_count += 1

        window_start = now - self._window_sec
        while state.events and state.events[0][0] < window_start:
            _, was_immediate = state.events.popleft()
            if was_immediate:
                state.immediate_count -= 1

        if now < state.next_log_time:
            return
        state.next_log_time = now + self._log_interval_sec

        total_events = int(len(state.events))
        if total_events <= 0:
            return

        immediate_events = int(state.immediate_count)
        immediate_ratio_pct = 100.0 * float(immediate_events) / float(total_events)
        if extra_context is None:
            logging.info(
                "Immediate wait ratio [%s|%s]: %.2f%% over last %.1f min "
                "(immediate=%d total=%d threshold_ms=%.1f)",
                self._metric_name,
                stream_key,
                immediate_ratio_pct,
                self._window_sec / 60.0,
                immediate_events,
                total_events,
                self._immediate_threshold_sec * 1000.0,
            )
        else:
            logging.info(
                "Immediate wait ratio [%s|%s]: %.2f%% over last %.1f min "
                "(immediate=%d total=%d threshold_ms=%.1f, %s)",
                self._metric_name,
                stream_key,
                immediate_ratio_pct,
                self._window_sec / 60.0,
                immediate_events,
                total_events,
                self._immediate_threshold_sec * 1000.0,
                extra_context,
            )


def raise_if_stop_requested(stop_event) -> None:
    if stop_event.is_set():
        raise StopRequested()


def queue_get_or_stop(q, stop_event, *, timeout_sec: float):
    assert float(timeout_sec) > 0.0, "timeout_sec must be > 0"
    while True:
        raise_if_stop_requested(stop_event)
        try:
            item = q.get(timeout=timeout_sec)
        except queue.Empty:
            continue
        return item


def timed_queue_get_or_stop(
    q,
    stop_event,
    *,
    timeout_sec: float,
    wait_logger: RollingImmediateWaitLogger,
    stream_key: str,
    extra_context: str | None = None,
):
    started = time.monotonic()
    item = queue_get_or_stop(q, stop_event, timeout_sec=timeout_sec)
    wait_logger.observe(
        time.monotonic() - started,
        stream_key=stream_key,
        extra_context=extra_context,
    )
    return item


def queue_get_or_timeout(q, stop_event, *, timeout_sec: float):
    assert float(timeout_sec) > 0.0, "timeout_sec must be > 0"
    raise_if_stop_requested(stop_event)
    try:
        item = q.get(timeout=timeout_sec)
    except queue.Empty:
        return None, True
    raise_if_stop_requested(stop_event)
    return item, False


def queue_put_or_stop(q, item, stop_event, *, timeout_sec: float) -> None:
    assert float(timeout_sec) > 0.0, "timeout_sec must be > 0"
    while True:
        raise_if_stop_requested(stop_event)
        try:
            q.put(item, timeout=timeout_sec)
        except queue.Full:
            continue
        return


def event_wait_or_stop(event, stop_event, *, timeout_sec: float) -> bool:
    assert float(timeout_sec) > 0.0, "timeout_sec must be > 0"
    if event.wait(timeout=timeout_sec):
        return True
    raise_if_stop_requested(stop_event)
    return False


def stop_event_wait_or_raise(stop_event, *, timeout_sec: float) -> None:
    assert float(timeout_sec) > 0.0, "timeout_sec must be > 0"
    if stop_event.wait(timeout=timeout_sec):
        raise StopRequested()
    raise_if_stop_requested(stop_event)


def set_stop_event_with_reason(
    stop_event,
    *,
    process_name: str,
    reason: str,
) -> None:
    assert hasattr(stop_event, "_cond"), type(stop_event)
    assert hasattr(stop_event, "_flag"), type(stop_event)
    cond = stop_event._cond
    flag = stop_event._flag
    acquired = cond.acquire(False)
    if not acquired:
        logging.warning(
            "STOP_EVENT_SET_SKIPPED process=%s pid=%d reason=%s",
            process_name,
            int(os.getpid()),
            reason,
        )
        return
    try:
        already_set = bool(flag.acquire(False))
        if already_set:
            flag.release()
        logging.warning(
            "STOP_EVENT_SET process=%s pid=%d already_set=%s reason=%s",
            process_name,
            int(os.getpid()),
            already_set,
            reason,
        )
        if already_set:
            return
        flag.release()
        cond.notify_all()
    finally:
        cond.release()


@dataclass(frozen=True)
class CheckpointConfig:
    """Where to load checkpoints from. Supported: ``torch_io: {local: {dirpath: <path>}}``."""

    torch_io: dict[str, Any]
    name: str = "latest"


def checkpoint_config_from_checkpoint_file(path: str | Path) -> CheckpointConfig:
    """Build :class:`CheckpointConfig` from one ``.pt`` path: directory + filename."""
    p = Path(path).expanduser().resolve()
    assert p.is_file(), f"checkpoint file must exist: {p}"
    assert p.suffix == ".pt", f"checkpoint file must have .pt suffix: {p}"
    return CheckpointConfig(
        torch_io={"local": {"dirpath": str(p.parent)}},
        name=p.name,
    )


@dataclass(frozen=True)
class _LocalDirCheckpointReader:
    root: Path

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    def read_torch(self, name: str, *, map_location, weights_only: bool):
        path = self.root / name
        return torch.load(path, map_location=map_location, weights_only=weights_only)

    def read_text(self, name: str) -> str:
        return (self.root / name).read_text()


def _local_dirpath_from_torch_io(torch_io: Any) -> Path:
    """``torch_io`` is a dict (YAML) or :class:`~types.SimpleNamespace` (nested training flags)."""
    if isinstance(torch_io, dict):
        local = torch_io["local"]
    elif isinstance(torch_io, SimpleNamespace):
        local = torch_io.local
    else:
        raise TypeError(
            f"checkpoint.torch_io must be dict or SimpleNamespace, got {type(torch_io)}"
        )
    if isinstance(local, dict):
        assert "dirpath" in local, "checkpoint.torch_io.local must contain 'dirpath'"
        dirpath = local["dirpath"]
    elif isinstance(local, SimpleNamespace):
        dirpath = local.dirpath
    else:
        raise TypeError(
            f"checkpoint.torch_io.local must be dict or SimpleNamespace, got {type(local)}"
        )
    return Path(str(dirpath))


def _checkpoint_reader_from_cfg(cfg: CheckpointConfig | SimpleNamespace) -> _LocalDirCheckpointReader:
    root = _local_dirpath_from_torch_io(cfg.torch_io)
    assert root.is_dir(), f"checkpoint directory must exist: {root}"
    return _LocalDirCheckpointReader(root=root)


def parse_impala_run_checkpoint_step(name: str) -> int | None:
    """Step embedded in ``checkpoint_<steps>.pt`` under ``main_torch_io`` (see batch_and_learn)."""
    prefix = "checkpoint_"
    if not (name.startswith(prefix) and name.endswith(".pt")):
        return None
    body = name[len(prefix) : -len(".pt")]
    if not body.isdigit():
        return None
    return int(body)


def latest_checkpoint_file_at_or_before_step(
    reader: _LocalDirCheckpointReader,
    *,
    max_step: int,
) -> tuple[str, int] | None:
    """Latest on-disk run checkpoint with ``steps <= max_step`` (by filename step)."""
    assert int(max_step) >= 0, "max_step must be >= 0"
    best: tuple[str, int] | None = None
    for fname in reader.list_files():
        step = parse_impala_run_checkpoint_step(fname)
        if step is None:
            continue
        if step > int(max_step):
            continue
        if best is None or step > best[1]:
            best = (fname, step)
    return best


def get_checkpoint_file(reader, checkpoint: str) -> str:
    all_checkpoint_files = set(f for f in reader.list_files() if f.endswith('.pt'))
    if not all_checkpoint_files:
        raise ValueError(f'No checkpoint files found')

    if checkpoint == 'latest':
        model_checkpoint_files = set(
            f for f in all_checkpoint_files if f.startswith('checkpoint')
        )
        if not model_checkpoint_files:
            raise ValueError(
                "No model checkpoint files found for 'latest'. "
                "Expected files like checkpoint_*.pt. "
                f"Available .pt files: {sorted(all_checkpoint_files)}"
            )
        return max(model_checkpoint_files)

    for ckpt in (checkpoint, f'{checkpoint}.pt', f'{checkpoint}_weights.pt'):
        if ckpt in all_checkpoint_files:
            return ckpt

    raise ValueError(f'Checkpoint file {checkpoint} not found')


def load_model_from_checkpoint(
    model,
    cfg: CheckpointConfig | SimpleNamespace,
    *,
    state_dict_key: str = "model_state_dict",
    load_as_much_as_possible: bool = False,
):
    reader = _checkpoint_reader_from_cfg(cfg)
    checkpoint_file = get_checkpoint_file(reader, cfg.name)
    checkpoint = reader.read_torch(checkpoint_file, map_location='cpu', weights_only=False)
    if state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    allowed_missing_prefixes = (
        _SUPERVISED_CHECKPOINT_ALLOWED_MISSING_STATE_DICT_PREFIXES
        if _checkpoint_config_is_supervised(cfg)
        else ()
    )
    if load_as_much_as_possible:
        load_state_dict_as_much_as_possible(model, state_dict)
    else:
        load_state_dict_strict(
            model,
            state_dict,
            allowed_missing_prefixes=allowed_missing_prefixes,
        )


def checkpoint_model_config_from_checkpoint(
    cfg: CheckpointConfig | SimpleNamespace,
) -> dict[str, Any]:
    reader = _checkpoint_reader_from_cfg(cfg)
    checkpoint_file = get_checkpoint_file(reader, cfg.name)
    checkpoint = reader.read_torch(checkpoint_file, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict), (
        f"checkpoint must be a dict to contain model_config, got {type(checkpoint)}"
    )
    assert "model_config" in checkpoint, (
        f"checkpoint {checkpoint_file} is missing model_config"
    )
    model_config = checkpoint["model_config"]
    assert isinstance(model_config, dict), (
        f"checkpoint model_config must be a dict, got {type(model_config)}"
    )
    return model_config


def checkpoint_model_config_from_checkpoint_or_fallback(
    cfg: CheckpointConfig | SimpleNamespace,
    fallback_cfg: CheckpointConfig | SimpleNamespace | None,
    current_model_config: Any,
) -> Any:
    reader = _checkpoint_reader_from_cfg(cfg)
    checkpoint_file = get_checkpoint_file(reader, cfg.name)
    checkpoint = reader.read_torch(checkpoint_file, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict), (
        f"checkpoint must be a dict to contain model_config, got {type(checkpoint)}"
    )
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        assert isinstance(model_config, dict), (
            f"checkpoint model_config must be a dict, got {type(model_config)}"
        )
        return model_config
    if fallback_cfg is None:
        return copy.deepcopy(current_model_config)
    return checkpoint_model_config_from_checkpoint(fallback_cfg)


def load_model_from_resume_sources(
    model: nn.Module,
    *,
    resume_checkpoint: CheckpointConfig | None,
    load_as_much_as_possible: bool,
) -> nn.Module:
    if resume_checkpoint is not None:
        load_model_from_checkpoint(
            model,
            resume_checkpoint,
            load_as_much_as_possible=load_as_much_as_possible,
        )
    return model


def load_yaml(file_path: str) -> dict:
    path = Path(file_path)
    assert path.is_file(), f"load_yaml: expected a file path, got {path}"
    with path.open() as f:
        return yaml.safe_load(f)
