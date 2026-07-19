from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gym.obs_wrapper import (
    ORBIT_EDGE_BASE_FEATURES,
    ORBIT_EDGE_BASE_FEATURE_ATTACK_REDUNDANCY_SCORE,
    ORBIT_EDGE_BASE_FEATURE_DST_FINAL_OWNER_IS_SRC_OWNER_WITHOUT_ACTION,
    ORBIT_EDGE_BASE_FEATURE_DST_NEUTRAL,
    ORBIT_EDGE_BASE_FEATURE_IS_AVAILABLE_WITH_MAX_SEND,
    ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET,
    ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET_AVAILABLE,
    ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET_HIT_STEPS,
    ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET,
    ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_AVAILABLE,
    ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_HIT_STEPS,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET_AVAILABLE,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET_HIT_STEPS,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET_AVAILABLE,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET_HIT_STEPS,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET_AVAILABLE,
    ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET_HIT_STEPS,
    ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET,
    ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET_AVAILABLE,
    ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET_HIT_STEPS,
    ORBIT_EDGE_BASE_FEATURE_NEUTRALIZE_MARGIN_WITH_MAX_SEND,
    ORBIT_EDGE_BASE_FEATURE_SRC_NEUTRAL,
    ORBIT_EDGE_BASE_FEATURE_STABLE_MARGIN_WITH_MAX_SEND,
    ORBIT_EDGE_BASE_FEATURE_TAKEOVER_MARGIN_WITH_MAX_SEND,
    ORBIT_EDGE_BASE_FEATURE_TIME_TO_HIT_WITH_MAX_SEND,
    ORBIT_EDGE_FEATURES,
    ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
    ORBIT_EDGE_PLAYER_FEATURE_DST_OWNED,
    ORBIT_EDGE_PLAYER_FEATURE_SRC_OWNED,
    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_ENEMY_AXIS_SLOTS,
    ORBIT_HIT_CLASSES_PER_TARGET,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_BASE_FEATURE_COMET_TIME_BEFORE_DESPAWN,
    ORBIT_PLANET_BASE_FEATURE_IS_COMET,
    ORBIT_PLANET_BASE_FEATURE_IS_DYNAMIC,
    ORBIT_PLANET_BASE_FEATURE_IS_STATIC,
    ORBIT_PLANET_BASE_FEATURE_EPISODE_STEP,
    ORBIT_PLANET_BASE_FEATURE_NEUTRAL_SHIPS,
    ORBIT_PLANET_EPISODE_STEP_FEATURE_DIVISOR,
    ORBIT_PLANET_BASE_FEATURE_PLANET_PRODUCTION,
    ORBIT_PLANET_ARRIVAL_FEATURES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_BASE_FEATURES,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLANET_PLAYER_FEATURE_FLIP_TIME,
    ORBIT_PLANET_PLAYER_FEATURE_LAST_DECISIVE_BATTLE_STEP,
    ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
    ORBIT_PLANET_PLAYER_FEATURE_OWNER_SURVIVAL_MARGIN,
    ORBIT_PLANET_PLAYER_FEATURE_OWNER_CHURN,
    ORBIT_PLANET_PLAYER_FEATURE_POST_HORIZON_OWNER_MARGIN,
    ORBIT_PLANET_PLAYER_FEATURE_PRODUCTION,
    ORBIT_PLANET_PLAYER_FEATURE_SHIPS,
    ORBIT_PLANET_PLAYER_FEATURE_STABLE_FLIP_TIME,
    ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
)
from ..gym.wall_tree_profiler import WallTreeProfiler, wall_tree_cuda_model_block
from .orbit_obs_feature_layout import (
    logical_feature_normalization_keys,
    planet_edge_physical_to_logical_from_layout,
)
from .orbit_obs_feature_input_contract import (
    BcObsDiscreteEmbeddingTable,
    ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES,
    OrbitObsDiscreteEmbeddingChannelSpec,
    orbit_discrete_embedding_arrival_temporal_specs,
    orbit_discrete_embedding_count,
    orbit_discrete_embedding_edge_base_specs,
    orbit_discrete_embedding_edge_player_block_specs,
    orbit_discrete_embedding_planet_base_specs,
    orbit_discrete_embedding_planet_player_block_specs,
)

_ENABLE_MODEL_ASSERT = os.environ.get("ENABLE_MODEL_ASSERT", "0") == "1"
_ORBIT_FEATURE_SPIKE_EPSILON = 1e-9

_ORBIT_TEMPORAL_ARRIVAL_SHIPS = 0
_ORBIT_TEMPORAL_TAKEOVER_COST = 1
_ORBIT_TEMPORAL_RESOLUTION_OWNER = 2
_ORBIT_TEMPORAL_RESOLUTION_SHIPS = 3
_ORBIT_TEMPORAL_TIME_STEP = 4
_ORBIT_TEMPORAL_STABLE_TAKEOVER_COST = 5
_ORBIT_TEMPORAL_HOLD_COST = 6
_ORBIT_TEMPORAL_HOLD_VALID = 7
_ORBIT_TEMPORAL_NEUTRALIZATION_COST = 8
_ORBIT_TEMPORAL_NEUTRALIZATION_VALID = 9
_ORBIT_TEMPORAL_DENY_STABLE_ENEMY_COST = 10
_ORBIT_TEMPORAL_BATTLE_TIE_DISTANCE = 11
_ORBIT_TEMPORAL_BATTLE_TIE_VALID = 12
_ORBIT_TEMPORAL_PRODUCTION_SWING_PER_SHIP = 13
_ORBIT_TEMPORAL_ARRIVAL_LEVERAGE = 14

_ORBIT_DISCRETE_SHIP_MAX_INDEX = 1000
_ORBIT_DISCRETE_SHIP_COST_MAX_INDEX = 1000
_ORBIT_DISCRETE_PRODUCTION_MAX_INDEX = 5
_ORBIT_DISCRETE_SHIP_BUCKET_MAX_INDEX = ORBIT_HIT_CLASSES_PER_TARGET - 1
_ORBIT_DISCRETE_SIGNED_SHIP_MARGIN_ABS_MAX_INDEX = 1000
_ORBIT_DISCRETE_EMBED_DIM = 8
_ORBIT_DISCRETE_EPISODE_STEP_MAX_INDEX = ORBIT_PLANET_EPISODE_STEP_FEATURE_DIVISOR
_ORBIT_SELF_FEATURE_SLOT = 0
_ORBIT_IDENTITY_CLASSES = 5
_ORBIT_SELF_IDENTITY_CLASS = 0
_ORBIT_NEUTRAL_IDENTITY_CLASS = 4
_ORBIT_EDGE_RELATION_FEATURES = 12
_ORBIT_MODEL_DEFAULT_TEMPORAL_HORIZON = 16
_ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS = (_ORBIT_MODEL_DEFAULT_TEMPORAL_HORIZON, ORBIT_PLANET_ARRIVAL_HORIZON)


def _xavier_uniform_linear_(linear: nn.Linear, *, scale: float = 1.0) -> None:
    assert scale > 0.0, (scale,)
    nn.init.xavier_uniform_(linear.weight)
    with torch.no_grad():
        linear.weight.mul_(float(scale))
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _xavier_uniform_mha_(mha: nn.MultiheadAttention, *, out_proj_scale: float = 1.0) -> None:
    assert out_proj_scale > 0.0, (out_proj_scale,)
    assert bool(mha._qkv_same_embed_dim), "Orbit MHA init expects shared q/k/v embed_dim"
    assert isinstance(mha.in_proj_weight, torch.Tensor)
    assert mha.in_proj_weight.shape == (3 * int(mha.embed_dim), int(mha.embed_dim))
    nn.init.xavier_uniform_(mha.in_proj_weight)
    assert isinstance(mha.in_proj_bias, torch.Tensor)
    nn.init.zeros_(mha.in_proj_bias)
    _xavier_uniform_linear_(mha.out_proj, scale=out_proj_scale)


def _orbit_module_init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        _xavier_uniform_linear_(m)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        _xavier_uniform_mha_(m)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


def _orbit_activation_module(activation: str) -> nn.Module:
    name = str(activation)
    assert name in ("gelu", "silu", "leaky_relu", "relu", "elu", "hardswish"), name
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    assert name == "hardswish", name
    return nn.Hardswish()


def _orbit_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    n_layers: int,
    activation: str,
) -> nn.Sequential:
    nl = int(n_layers)
    assert nl >= 1, (n_layers,)
    layers: list[nn.Module] = []
    if nl == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)
    layers.append(nn.Linear(in_dim, hidden))
    layers.append(_orbit_activation_module(activation))
    for _ in range(nl - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(_orbit_activation_module(activation))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _cfg_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return vars(obj)


_IMPALA_ORBIT_MODEL_REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "hidden_dim",
    "use_token_batch_norm",
    "obs_feature_normalization",
    "encoder_num_layers",
    "arrivals_encoder_num_layers",
    "arrivals_fusion_num_layers",
    "decoder_num_layers",
    "transformer_num_layers",
    "transformer_num_heads",
    "edge_endpoint_update",
    "residual_dropout",
    "ffn_multiplier",
    "ffn_dropout",
    "use_obs_discrete_embeddings",
)
_IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS: dict[str, Any] = {
    "mlp_activation": "gelu",
    "transformer_activation": "gelu",
    "num_policy_actions": 2,
    "arrival_temporal_horizon": _ORBIT_MODEL_DEFAULT_TEMPORAL_HORIZON,
    "zeroed_obs_feature_inputs": (),
    "use_identity_embedding": False,
    "use_edge_relation_features": False,
    "use_arrival_attention_fusion": False,
    "arrival_attention_num_queries": 4,
    "use_arrival_attention_fusion_gate": False,
    "allow_self_edges": False,
    "policy_head_fp32": False,
    "value_head_fp32": False,
    "use_value_opponent_model_embedding": False,
    "value_opponent_model_count": 0,
}
_VALUE_OPPONENT_MODEL_ID_CAPACITY = 20


def _entropy_head_values_from_config(values: Any, *, label: str) -> tuple[float]:
    if isinstance(values, dict):
        assert tuple(values.keys()) == ("spawn_fleet",), (label, tuple(values.keys()))
        out = (float(values["spawn_fleet"]),)
    else:
        assert hasattr(values, "spawn_fleet"), (label, values)
        out = (float(values.spawn_fleet),)
    for v in out:
        assert 0.0 <= float(v) <= 1.0, (label, out)
    return out


def _masked_log_probs(
    logits: torch.Tensor,
    mask: torch.Tensor,
    wall_profiler: WallTreeProfiler | None = None,
) -> torch.Tensor:
    assert logits.shape == mask.shape, (tuple(logits.shape), tuple(mask.shape))
    assert logits.ndim >= 2, tuple(logits.shape)
    assert torch.is_floating_point(logits), logits.dtype
    assert mask.dtype == torch.bool, mask.dtype
    device = logits.device
    k_counts = mask.sum(dim=-1)
    active = k_counts > 0
    neg_large = torch.finfo(logits.dtype).min / 16.0
    with wall_tree_cuda_model_block(wall_profiler, device, "masked_log_softmax"):
        masked_logits = logits.masked_fill(~mask, neg_large)
        active_logits = torch.where(active.unsqueeze(-1), masked_logits, logits)
        log_probs = F.log_softmax(active_logits, dim=-1)
    return log_probs


def _masked_log_probs_and_normalized_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert logits.shape == mask.shape, (
        f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}"
    )
    assert logits.ndim >= 2, tuple(logits.shape)
    assert torch.is_floating_point(logits), logits.dtype
    assert mask.dtype == torch.bool, mask.dtype
    k_counts = mask.sum(dim=-1)
    log_probs = _masked_log_probs(logits, mask)
    valid_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(mask, log_probs.exp(), torch.zeros_like(log_probs))
    entropy = -(probs * valid_log_probs).sum(dim=-1)
    eligible = k_counts > 1
    log_k = torch.log(torch.clamp(k_counts.to(device=logits.device, dtype=logits.dtype), min=2))
    h_norm = torch.where(eligible, entropy / log_k, torch.zeros_like(entropy))
    return log_probs, h_norm, eligible, k_counts


def _choose_from_log_probs(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample: bool,
    compile_friendly_sample: bool,
    label: str,
) -> torch.Tensor:
    assert log_probs.ndim >= 2, tuple(log_probs.shape)
    assert log_probs.shape[-1] > 0, tuple(log_probs.shape)
    assert torch.is_floating_point(log_probs), log_probs.dtype
    assert mask.shape == log_probs.shape, (tuple(mask.shape), tuple(log_probs.shape))
    assert mask.dtype == torch.bool, mask.dtype
    row_has_available = mask.any(dim=-1)
    torch.ops.aten._assert_async.msg(
        torch.all(row_has_available),
        f"{label} action row has no available action",
    )
    available_finite = torch.where(
        mask,
        torch.isfinite(log_probs),
        torch.ones_like(mask, dtype=torch.bool),
    )
    torch.ops.aten._assert_async.msg(
        torch.all(available_finite),
        f"{label} logits/log-probs contain non-finite available action",
    )
    neg_large = torch.finfo(log_probs.dtype).min / 16.0
    masked_log_probs = log_probs.masked_fill(~mask, neg_large)
    if not bool(sample):
        actions = masked_log_probs.argmax(dim=-1).to(dtype=torch.long)
        _assert_chosen_actions_available(actions, mask, label=label)
        return actions

    if not bool(compile_friendly_sample):
        probs = masked_log_probs.exp()
        actions = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]),
            num_samples=1,
        ).reshape(probs.shape[:-1]).to(dtype=torch.long)
        _assert_chosen_actions_available(actions, mask, label=label)
        return actions

    with torch.amp.autocast(device_type=log_probs.device.type, enabled=False):
        sample_log_probs = masked_log_probs.to(dtype=torch.float32)
        exp_noise = torch.empty_like(sample_log_probs).exponential_(1.0).clamp_min(
            torch.finfo(sample_log_probs.dtype).tiny
        )
        sample_scores = sample_log_probs - exp_noise.log()
    actions = sample_scores.argmax(dim=-1).to(dtype=torch.long)
    _assert_chosen_actions_available(actions, mask, label=label)
    return actions


def _assert_chosen_actions_available(
    actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    label: str,
) -> None:
    assert actions.dtype == torch.long, (label, actions.dtype)
    assert mask.dtype == torch.bool, (label, mask.dtype)
    assert tuple(actions.shape) == tuple(mask.shape[:-1]), (
        label,
        tuple(actions.shape),
        tuple(mask.shape),
    )
    selected_available = torch.gather(mask, -1, actions.unsqueeze(-1)).squeeze(-1)
    assert selected_available.shape == actions.shape, (
        label,
        tuple(selected_available.shape),
        tuple(actions.shape),
    )
    torch.ops.aten._assert_async.msg(
        torch.all(selected_available),
        f"{label} selected unavailable action",
    )


def _apply_masked_entropy_floor_to_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_min_entropy: torch.Tensor,
    max_temperature: float,
    num_iters: int,
    wall_profiler: WallTreeProfiler | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert logits.shape == mask.shape, (
        f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}"
    )
    assert logits.ndim >= 2, tuple(logits.shape)
    assert torch.is_floating_point(logits), logits.dtype
    assert mask.dtype == torch.bool, mask.dtype
    if False:
        assert isinstance(target_min_entropy, torch.Tensor), type(target_min_entropy)
        assert target_min_entropy.ndim == 0, tuple(target_min_entropy.shape)
        target = target_min_entropy.to(device=logits.device, dtype=torch.float32)
        if _ENABLE_MODEL_ASSERT:
            _assert_tensor(
                torch.all((0.0 <= target) & (target <= 1.0)),
                "target_min_entropy must be in [0, 1]",
            )
        max_temp = float(max_temperature)
        assert max_temp >= 1.0, max_temperature
        iters = int(num_iters)
        assert iters >= 0, num_iters
        device = logits.device
        with torch.no_grad(), torch.amp.autocast(device_type=logits.device.type, enabled=False):
            with wall_tree_cuda_model_block(wall_profiler, device, "entropy_floor_initial_entropy"):
                logits_f = logits.float()
                _, h_norm, eligible, _ = _masked_log_probs_and_normalized_entropy(logits_f, mask)
                needs_floor = eligible & (h_norm < target)
                temperature = torch.ones_like(h_norm, dtype=torch.float32)
            if max_temp > 1.0 and iters > 0:
                lo = torch.ones_like(h_norm, dtype=torch.float32)
                hi = torch.full_like(h_norm, max_temp, dtype=torch.float32)
                for _ in range(iters):
                    with wall_tree_cuda_model_block(wall_profiler, device, "entropy_floor_search_iter"):
                        mid = (lo + hi) * 0.5
                        _, h_mid, _, _ = _masked_log_probs_and_normalized_entropy(
                            logits_f / mid.unsqueeze(-1),
                            mask,
                        )
                        mid_too_low = h_mid < target
                        lo = torch.where(needs_floor & mid_too_low, mid, lo)
                        hi = torch.where(needs_floor & mid_too_low, hi, mid)
                temperature = torch.where(needs_floor, hi, temperature)
            with wall_tree_cuda_model_block(wall_profiler, device, "entropy_floor_after_entropy"):
                scaled_logits_detached_f = logits_f / temperature.unsqueeze(-1)
                _, h_after, _, _ = _masked_log_probs_and_normalized_entropy(
                    scaled_logits_detached_f,
                    mask,
                )
                applied_f = needs_floor.to(dtype=logits.dtype)
                active_f = eligible.to(dtype=logits.dtype)
                miss_f = (needs_floor & (h_after < target)).to(dtype=logits.dtype)
                diagnostics = {
                    "temperature": temperature,
                    "applied": applied_f,
                    "active": active_f,
                    "miss": miss_f,
                }
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            with wall_tree_cuda_model_block(wall_profiler, device, "entropy_floor_scale_logits"):
                logits_f = logits.float()
                scaled_logits_f = logits_f / temperature.unsqueeze(-1)
                scaled_logits = scaled_logits_f.to(dtype=logits.dtype)
        return scaled_logits, diagnostics
    return logits, {}


def _disabled_entropy_floor_diagnostics(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert logits.shape == mask.shape, (
        f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}"
    )
    assert logits.ndim >= 2, tuple(logits.shape)
    assert torch.is_floating_point(logits), logits.dtype
    assert mask.dtype == torch.bool, mask.dtype
    shape = logits.shape[:-1]
    temperature = torch.ones(shape, device=logits.device, dtype=torch.float32)
    active = (mask.sum(dim=-1) > 1).to(dtype=logits.dtype)
    zero = torch.zeros(shape, device=logits.device, dtype=logits.dtype)
    return {
        "temperature": temperature,
        "applied": zero,
        "active": active,
        "miss": zero,
    }


def _entropy_floor_reduce_stats(
    diagnostics: dict[str, torch.Tensor],
    *,
    reduce_dims: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    temperature = diagnostics["temperature"]
    applied = diagnostics["applied"]
    active = diagnostics["active"]
    miss = diagnostics["miss"]
    assert temperature.shape == applied.shape == active.shape == miss.shape
    active_count = active.sum(dim=reduce_dims)
    active_denom = active_count.clamp(min=1.0)
    applied_count = applied.sum(dim=reduce_dims)
    miss_count = miss.sum(dim=reduce_dims)
    return {
        "temperature_mean": torch.where(
            active_count > 0.0,
            (temperature * active).sum(dim=reduce_dims) / active_denom,
            torch.ones_like(active_count),
        ),
        "temperature_max": torch.where(applied > 0.0, temperature, torch.ones_like(temperature)).amax(dim=reduce_dims),
        "active_frac": active.mean(dim=reduce_dims),
        "applied_frac": torch.where(
            active_count > 0.0,
            applied_count / active_denom,
            torch.zeros_like(active_count),
        ),
        "miss_frac": torch.where(
            active_count > 0.0,
            miss_count / active_denom,
            torch.zeros_like(active_count),
        ),
    }


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    assert isinstance(condition, torch.Tensor), type(condition)
    assert condition.ndim == 0, (tuple(condition.shape), message)
    if not _ENABLE_MODEL_ASSERT:
        return
    if torch.compiler.is_compiling():
        return
    assert bool(condition.detach().cpu().item()), message


def _assert_orbit_row_tensor_finite(
    *,
    head_name: str,
    reason: str,
    row_tensor: torch.Tensor,
    context: dict[str, torch.Tensor],
) -> None:
    if not _ENABLE_MODEL_ASSERT or torch.compiler.is_compiling():
        return
    assert row_tensor.ndim >= 3, (reason, tuple(row_tensor.shape))
    finite_by_row = torch.isfinite(row_tensor).flatten(start_dim=2).all(dim=-1)
    bad_rows = ~finite_by_row
    if bool(bad_rows.any().item()):
        bad_indices = torch.nonzero(bad_rows, as_tuple=False)
        assert bad_indices.ndim == 2 and int(bad_indices.shape[0]) > 0, bad_rows.shape
        bad_index = tuple(int(v.item()) for v in bad_indices[0])
        payload: dict[str, Any] = {
            "head": head_name,
            "reason": reason,
            "bad_index": bad_index,
            "row_tensor": row_tensor[bad_index].detach().cpu(),
        }
        for name, value in context.items():
            assert value.shape[:2] == row_tensor.shape[:2], (name, value.shape, row_tensor.shape)
            payload[name] = value[bad_index].detach().cpu()
        raise AssertionError(payload)


def _mha_sdpa_project(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert bool(mha.batch_first), "SDPA helper expects batch_first=True"
    assert bool(mha._qkv_same_embed_dim), "SDPA helper expects shared q/k/v embed_dim"
    assert mha.bias_k is None
    assert mha.bias_v is None
    assert not bool(mha.add_zero_attn)
    assert query.ndim == 3
    b, q_len, h = query.shape
    kv_b, kv_len, kv_h = key.shape
    assert value.shape == (kv_b, kv_len, kv_h)
    assert kv_b == b
    assert h == int(mha.embed_dim)
    assert kv_h == h
    nh = int(mha.num_heads)
    assert h % nh == 0, (h, nh)
    d = h // nh
    in_proj_weight = mha.in_proj_weight
    in_proj_bias = mha.in_proj_bias
    assert isinstance(in_proj_weight, torch.Tensor)
    assert isinstance(in_proj_bias, torch.Tensor)
    assert in_proj_weight.shape == (3 * h, h)
    assert in_proj_bias.shape == (3 * h,)
    q_proj = F.linear(query, in_proj_weight[:h], in_proj_bias[:h])
    k_proj = F.linear(key, in_proj_weight[h : 2 * h], in_proj_bias[h : 2 * h])
    v_proj = F.linear(value, in_proj_weight[2 * h :], in_proj_bias[2 * h :])
    q = q_proj.view(b, q_len, nh, d).transpose(1, 2)
    k = k_proj.view(b, kv_len, nh, d).transpose(1, 2)
    v = v_proj.view(b, kv_len, nh, d).transpose(1, 2)
    assert q.shape == (b, nh, q_len, d)
    assert k.shape == (b, nh, kv_len, d)
    assert v.shape == (b, nh, kv_len, d)
    return q, k, v


def _mha_sdpa_output(
    mha: nn.MultiheadAttention,
    attn: torch.Tensor,
    batch_size: int,
    query_len: int,
) -> torch.Tensor:
    h = int(mha.embed_dim)
    nh = int(mha.num_heads)
    assert h % nh == 0, (h, nh)
    d = h // nh
    assert attn.shape == (batch_size, nh, query_len, d)
    out = attn.transpose(1, 2).reshape(batch_size, query_len, h)
    assert out.shape == (batch_size, query_len, h)
    return mha.out_proj(out)


def _mha_sdpa_nomask(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert query.ndim == 3
    b, q_len, _h = query.shape
    q, k, v = _mha_sdpa_project(mha, query, key, value)
    dropout_p = float(mha.dropout) if bool(mha.training) else 0.0
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=False,
    )
    return _mha_sdpa_output(mha, attn, b, q_len)


def _mha_sdpa_key_padding(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    assert query.ndim == 3
    b, q_len, _h = query.shape
    kv_len = int(key.shape[1])
    assert key_padding_mask.shape == (b, kv_len)
    assert key_padding_mask.dtype == torch.bool
    q, k, v = _mha_sdpa_project(mha, query, key, value)
    allow_mask = (~key_padding_mask).view(b, 1, 1, kv_len).expand(b, 1, q_len, kv_len)
    assert allow_mask.shape == (b, 1, q_len, kv_len)
    dropout_p = float(mha.dropout) if bool(mha.training) else 0.0
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allow_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )
    return _mha_sdpa_output(mha, attn, b, q_len)


def _mha_sdpa_key_padding_attn_mask(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    assert query.ndim == 3
    b, q_len, _h = query.shape
    kv_len = int(key.shape[1])
    assert key_padding_mask.shape == (b, kv_len)
    assert key_padding_mask.dtype == torch.bool
    assert attn_mask.shape == (q_len, kv_len)
    assert attn_mask.dtype == torch.bool
    q, k, v = _mha_sdpa_project(mha, query, key, value)
    invalid_mask = key_padding_mask.view(b, 1, 1, kv_len) | attn_mask.view(1, 1, q_len, kv_len)
    allow_mask = ~invalid_mask
    assert allow_mask.shape == (b, 1, q_len, kv_len)
    dropout_p = float(mha.dropout) if bool(mha.training) else 0.0
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allow_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )
    return _mha_sdpa_output(mha, attn, b, q_len)


def _feature_norm_spec_mapping(spec: Any) -> dict[str, Any]:
    out = _cfg_mapping(spec)
    for key in (
        "mean",
        "std",
        "clip_down",
        "clip_up",
        "spike_values",
        "enabled",
        "norm_enabled",
        "clip_enabled",
    ):
        assert key in out, (key, sorted(out.keys()))
    std = float(out["std"])
    assert std > 0.0, out
    clip_down = float(out["clip_down"])
    clip_up = float(out["clip_up"])
    assert clip_down <= clip_up, out
    spike_values = out["spike_values"]
    assert isinstance(spike_values, (list, tuple)), (type(spike_values), out)
    assert isinstance(out["enabled"], bool), (type(out["enabled"]), out)
    assert isinstance(out["norm_enabled"], bool), (type(out["norm_enabled"]), out)
    assert isinstance(out["clip_enabled"], bool), (type(out["clip_enabled"]), out)
    return out


def _disabled_feature_norm_spec() -> dict[str, Any]:
    return {
        "mean": 0.0,
        "std": 1.0,
        "clip_down": 0.0,
        "clip_up": 0.0,
        "spike_values": (),
        "enabled": False,
        "norm_enabled": False,
        "clip_enabled": False,
    }


def _feature_norm_config_with_disabled_missing(
    feature_config: dict[str, Any],
    expected_feature_keys: set[str],
) -> dict[str, Any]:
    extra = set(feature_config) - expected_feature_keys
    assert not extra, sorted(extra)
    missing = sorted(expected_feature_keys - set(feature_config))
    if missing:
        warnings.warn(
            "Orbit obs feature normalization missing config; disabling features: "
            + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )
    out = dict(feature_config)
    for key in missing:
        out[key] = _disabled_feature_norm_spec()
    return out


def _feature_norm_spike_count(
    feature_config: dict[str, Any],
    keys: tuple[str, ...],
) -> int:
    total = 0
    for key in keys:
        assert key in feature_config, (key, sorted(feature_config.keys()))
        spec = _feature_norm_spec_mapping(feature_config[key])
        if bool(spec["enabled"]):
            total += len(spec["spike_values"])
    return total


def _feature_norm_clip_count(
    feature_config: dict[str, Any],
    keys: tuple[str, ...],
) -> int:
    total = 0
    for key in keys:
        assert key in feature_config, (key, sorted(feature_config.keys()))
        spec = _feature_norm_spec_mapping(feature_config[key])
        if bool(spec["enabled"]) and bool(spec["clip_enabled"]):
            total += 2
    return total


def _scatter_any_last_dim(
    flags: torch.Tensor,
    channel_indices: torch.Tensor,
    *,
    channel_count: int,
) -> torch.Tensor:
    assert flags.ndim >= 1, (flags.ndim, tuple(flags.shape))
    assert channel_indices.ndim == 1, (channel_indices.ndim, tuple(channel_indices.shape))
    assert int(flags.shape[-1]) == int(channel_indices.shape[0]), (
        tuple(flags.shape),
        tuple(channel_indices.shape),
    )
    c = int(channel_count)
    assert c > 0, c
    index = channel_indices.view((1,) * (flags.ndim - 1) + (-1,)).expand(flags.shape)
    accum = torch.zeros(flags.shape[:-1] + (c,), device=flags.device, dtype=torch.int32)
    accum.scatter_add_(-1, index, flags.to(dtype=accum.dtype))
    out = accum > 0
    assert out.shape == flags.shape[:-1] + (c,), (out.shape, flags.shape, c)
    return out


@dataclass(frozen=True)
class _ZeroedOrbitObsFeatureInput:
    kind: str
    domain: str
    feature_name: str
    player: str


def _zeroed_obs_feature_input_rows(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(raw.splitlines())
    assert isinstance(raw, (list, tuple)), (type(raw), raw)
    rows: list[str] = []
    for i, item in enumerate(raw):
        assert isinstance(item, str), (i, type(item))
        rows.extend(item.splitlines())
    return tuple(rows)


def _zeroed_obs_feature_inputs_from_config(raw: Any) -> tuple[_ZeroedOrbitObsFeatureInput, ...]:
    out: list[_ZeroedOrbitObsFeatureInput] = []
    seen: set[str] = set()
    for i, row in enumerate(_zeroed_obs_feature_input_rows(raw)):
        stripped = row.strip()
        if not stripped:
            continue
        token = stripped.split()[0]
        if token == "feature":
            continue
        assert token not in seen, token
        seen.add(token)
        parts = token.split(".")
        assert len(parts) == 3, token
        kind, domain, feature_token = parts
        assert kind in ("continuous", "embedding"), token
        assert domain in ("planet", "arrival", "edge"), token
        feature_player = feature_token.split("@")
        assert len(feature_player) in (1, 2), token
        feature_name = feature_player[0]
        assert feature_name, token
        player = ""
        if len(feature_player) == 2:
            player = feature_player[1]
            assert player in ("self", "enemy"), token
        out.append(
            _ZeroedOrbitObsFeatureInput(
                kind=kind,
                domain=domain,
                feature_name=feature_name,
                player=player,
            )
        )
    return tuple(out)


def _zeroed_self_enemy_block_channel_mask(
    zeroed_inputs: tuple[_ZeroedOrbitObsFeatureInput, ...],
    *,
    kind: str,
    domain: str,
    logical_feature_keys: tuple[str, ...],
    base_feature_count: int,
    player_feature_offset: int,
    player_features_per_player: int,
    num_physical_channels: int,
) -> torch.Tensor:
    assert kind in ("continuous", "embedding"), kind
    assert domain in ("planet", "edge"), domain
    assert len(logical_feature_keys) == base_feature_count + player_features_per_player, (
        len(logical_feature_keys),
        base_feature_count,
        player_features_per_player,
    )
    feature_key_prefix = f"continuous.{domain}."
    logical_names = tuple(k.removeprefix(feature_key_prefix) for k in logical_feature_keys)
    assert all(k.startswith(feature_key_prefix) for k in logical_feature_keys), logical_feature_keys
    base_names = logical_names[:base_feature_count]
    player_names = logical_names[base_feature_count:]
    mask = torch.zeros(num_physical_channels, dtype=torch.bool)
    for item in zeroed_inputs:
        if item.kind != kind or item.domain != domain:
            continue
        if item.feature_name in base_names:
            assert item.player == "", item
            ch = base_names.index(item.feature_name)
            mask[ch] = True
        else:
            assert item.feature_name in player_names, item
            assert item.player in ("self", "enemy"), item
            player_feature = player_names.index(item.feature_name)
            if item.player == "self":
                player_slots = (0,)
            else:
                player_slots = tuple(range(1, ORBIT_PLAYER_AXIS_SLOTS))
            for player_slot in player_slots:
                ch = (
                    int(player_feature_offset)
                    + player_slot * int(player_features_per_player)
                    + player_feature
                )
                assert 0 <= ch < num_physical_channels, (item, ch, num_physical_channels)
                mask[ch] = True
    return mask


def _zeroed_arrival_slot_channel_mask(
    zeroed_inputs: tuple[_ZeroedOrbitObsFeatureInput, ...],
    *,
    kind: str,
    feature_names: tuple[str, ...],
) -> torch.Tensor:
    assert kind in ("continuous", "embedding"), kind
    assert len(feature_names) == ORBIT_PLANET_TEMPORAL_FEATURES, (
        len(feature_names),
        ORBIT_PLANET_TEMPORAL_FEATURES,
    )
    mask = torch.zeros(
        (ORBIT_PLAYER_AXIS_SLOTS, ORBIT_PLANET_TEMPORAL_FEATURES),
        dtype=torch.bool,
    )
    for item in zeroed_inputs:
        if item.kind != kind or item.domain != "arrival":
            continue
        assert item.feature_name in feature_names, item
        assert item.player in ("self", "enemy"), item
        ch = feature_names.index(item.feature_name)
        if item.player == "self":
            player_slots = (0,)
        else:
            player_slots = tuple(range(1, ORBIT_PLAYER_AXIS_SLOTS))
        for player_slot in player_slots:
            mask[player_slot, ch] = True
    return mask


def _zero_last_dim_channels(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    assert channel_mask.ndim == 1, tuple(channel_mask.shape)
    assert int(channel_mask.shape[0]) == int(x.shape[-1]), (tuple(channel_mask.shape), tuple(x.shape))
    mask = channel_mask.to(device=x.device, dtype=torch.bool)
    return torch.where(
        mask.view((1,) * (x.ndim - 1) + (-1,)),
        torch.zeros_like(x),
        x,
    )


def _zero_arrival_slot_channels(x: torch.Tensor, slot_channel_mask: torch.Tensor) -> torch.Tensor:
    assert slot_channel_mask.ndim == 2, tuple(slot_channel_mask.shape)
    assert slot_channel_mask.shape == x.shape[-2:], (
        tuple(slot_channel_mask.shape),
        tuple(x.shape),
    )
    mask = slot_channel_mask.to(device=x.device, dtype=torch.bool)
    return torch.where(
        mask.view((1,) * (x.ndim - 2) + tuple(mask.shape)),
        torch.zeros_like(x),
        x,
    )


class _ConfiguredLogicalFeatureNorm(nn.Module):
    def __init__(
        self,
        *,
        num_physical_channels: int,
        logical_feature_keys: tuple[str, ...],
        physical_to_logical: torch.Tensor,
        feature_config: dict[str, Any],
    ) -> None:
        super().__init__()
        c = int(num_physical_channels)
        assert c > 0, (num_physical_channels,)
        idx = physical_to_logical.detach().long().cpu()
        assert idx.ndim == 1 and int(idx.shape[0]) == c, (tuple(idx.shape), c)
        l_count = len(logical_feature_keys)
        assert l_count > 0, logical_feature_keys
        assert int(idx.min().item()) >= 0, idx
        assert int(idx.max().item()) == l_count - 1, (int(idx.max().item()), l_count)
        assert int(torch.unique(idx).numel()) == l_count, (torch.unique(idx), l_count)

        mean_l: list[float] = []
        std_l: list[float] = []
        clip_down_l: list[float] = []
        clip_up_l: list[float] = []
        enabled_l: list[bool] = []
        norm_enabled_l: list[bool] = []
        clip_enabled_l: list[bool] = []
        spike_channel_indices: list[int] = []
        spike_values: list[float] = []
        extra_channel_indices: list[int] = []
        extra_spike_positions: list[int] = []
        extra_clip_positions: list[int] = []
        clip_extra_channel_indices: list[int] = []
        clip_extra_values: list[float] = []
        spec_l: list[dict[str, Any]] = []
        for key in logical_feature_keys:
            assert key in feature_config, (key, sorted(feature_config.keys()))
            spec = _feature_norm_spec_mapping(feature_config[key])
            spec_l.append(spec)
            mean_l.append(float(spec["mean"]))
            std_l.append(float(spec["std"]))
            clip_down = float(spec["clip_down"])
            clip_up = float(spec["clip_up"])
            clip_down_l.append(clip_down)
            clip_up_l.append(clip_up)
            enabled_l.append(bool(spec["enabled"]))
            norm_enabled_l.append(bool(spec["norm_enabled"]))
            clip_enabled_l.append(bool(spec["clip_enabled"]))
        for physical_ch, logical_idx in enumerate(idx.tolist()):
            spec = spec_l[int(logical_idx)]
            if not bool(spec["enabled"]):
                continue
            for spike_value in spec["spike_values"]:
                spike_channel_indices.append(int(physical_ch))
                spike_values.append(float(spike_value))
                extra_spike_positions.append(len(extra_channel_indices))
                extra_channel_indices.append(int(physical_ch))
            if bool(spec["clip_enabled"]):
                extra_clip_positions.append(len(extra_channel_indices))
                clip_extra_channel_indices.append(int(physical_ch))
                clip_extra_values.append(float(spec["clip_down"]))
                extra_channel_indices.append(int(physical_ch))
                extra_clip_positions.append(len(extra_channel_indices))
                clip_extra_channel_indices.append(int(physical_ch))
                clip_extra_values.append(float(spec["clip_up"]))
                extra_channel_indices.append(int(physical_ch))

        mean = torch.tensor(mean_l, dtype=torch.float32)[idx]
        std = torch.tensor(std_l, dtype=torch.float32)[idx]
        clip_down = torch.tensor(clip_down_l, dtype=torch.float32)[idx]
        clip_up = torch.tensor(clip_up_l, dtype=torch.float32)[idx]
        enabled = torch.tensor(enabled_l, dtype=torch.bool)[idx]
        norm_enabled = torch.tensor(norm_enabled_l, dtype=torch.bool)[idx]
        clip_enabled = torch.tensor(clip_enabled_l, dtype=torch.bool)[idx]
        assert (
            mean.shape
            == std.shape
            == clip_down.shape
            == clip_up.shape
            == enabled.shape
            == norm_enabled.shape
            == clip_enabled.shape
            == (c,)
        )
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        self.register_buffer("_clip_down", clip_down, persistent=False)
        self.register_buffer("_clip_up", clip_up, persistent=False)
        self.register_buffer("_enabled", enabled, persistent=False)
        self.register_buffer("_norm_enabled", norm_enabled, persistent=False)
        self.register_buffer("_clip_enabled", clip_enabled, persistent=False)
        spike_channels = torch.tensor(spike_channel_indices, dtype=torch.long)
        spike_values_t = torch.tensor(spike_values, dtype=torch.float32)
        self.register_buffer("_spike_channels", spike_channels, persistent=False)
        self.register_buffer("_spike_values", spike_values_t, persistent=False)
        extra_channels = torch.tensor(extra_channel_indices, dtype=torch.long)
        extra_spike_positions_t = torch.tensor(extra_spike_positions, dtype=torch.long)
        extra_clip_positions_t = torch.tensor(extra_clip_positions, dtype=torch.long)
        clip_extra_channels = torch.tensor(clip_extra_channel_indices, dtype=torch.long)
        clip_extra_values_t = torch.tensor(clip_extra_values, dtype=torch.float32)
        self.register_buffer("_extra_channels", extra_channels, persistent=False)
        self.register_buffer("_extra_spike_positions", extra_spike_positions_t, persistent=False)
        self.register_buffer("_extra_clip_positions", extra_clip_positions_t, persistent=False)
        self.register_buffer("_clip_extra_channels", clip_extra_channels, persistent=False)
        self.register_buffer("_clip_extra_values", clip_extra_values_t, persistent=False)
        self._spike_dim = len(spike_channel_indices)
        self._extra_dim = len(extra_channel_indices)
        self._clip_extra_dim = len(clip_extra_channel_indices)

    @property
    def extra_dim(self) -> int:
        return int(self._extra_dim)

    def extra_zero_mask_for_physical_channels(self, channel_mask: torch.Tensor) -> torch.Tensor:
        assert channel_mask.ndim == 1, tuple(channel_mask.shape)
        assert int(channel_mask.shape[0]) == int(self._mean.shape[0]), (
            tuple(channel_mask.shape),
            tuple(self._mean.shape),
        )
        extra_channels = self._extra_channels.detach().cpu().long()
        if int(extra_channels.numel()) == 0:
            return torch.zeros((0,), dtype=torch.bool)
        return channel_mask.detach().cpu().to(dtype=torch.bool)[extra_channels]

    def extra_source_channels(self) -> torch.Tensor:
        return self._extra_channels.detach().cpu().long()

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        feature_valid_mask: torch.Tensor | None = None,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim >= 2, (x.ndim, tuple(x.shape))
        assert valid_mask.shape == x.shape[:-1], (tuple(valid_mask.shape), tuple(x.shape))
        c = int(x.shape[-1])
        assert c == int(self._mean.shape[0]), (c, int(self._mean.shape[0]))
        if feature_valid_mask is not None:
            assert feature_valid_mask.shape == x.shape, (
                tuple(feature_valid_mask.shape),
                tuple(x.shape),
            )

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x
            calc_dtype = x.dtype
            with wall_tree_cuda_model_block(wall_profiler, x.device, "feature_norm_buffers"):
                mean = self._mean.to(device=x.device, dtype=calc_dtype)
                std = self._std.to(device=x.device, dtype=calc_dtype)
                clip_down = self._clip_down.to(device=x.device, dtype=calc_dtype)
                clip_up = self._clip_up.to(device=x.device, dtype=calc_dtype)
                enabled = self._enabled.to(device=x.device, dtype=torch.bool)
                norm_enabled = self._norm_enabled.to(device=x.device, dtype=torch.bool)
                clip_enabled = self._clip_enabled.to(device=x.device, dtype=torch.bool)
                valid = valid_mask > 0.5

            with wall_tree_cuda_model_block(wall_profiler, x.device, "feature_norm_spike_mask"):
                if self._spike_dim == 0:
                    spike_flags = torch.empty(
                        xf.shape[:-1] + (0,),
                        device=x.device,
                        dtype=torch.bool,
                    )
                    spike_zero_mask = torch.zeros_like(xf, dtype=torch.bool)
                else:
                    spike_channels = self._spike_channels.to(device=x.device, dtype=torch.long)
                    spike_values = self._spike_values.to(device=x.device, dtype=calc_dtype)
                    spike_x = torch.index_select(xf, -1, spike_channels)
                    spike_flags = torch.abs(
                        spike_x - spike_values.view((1,) * (xf.ndim - 1) + (-1,))
                    ) <= _ORBIT_FEATURE_SPIKE_EPSILON
                    spike_flags = spike_flags & valid.unsqueeze(-1)
                    if feature_valid_mask is not None:
                        spike_feature_valid = torch.index_select(
                            feature_valid_mask > 0.5,
                            -1,
                            spike_channels,
                        )
                        spike_flags = spike_flags & spike_feature_valid
                    spike_zero_mask = _scatter_any_last_dim(
                        spike_flags,
                        spike_channels,
                        channel_count=c,
                    )

            with wall_tree_cuda_model_block(wall_profiler, x.device, "feature_norm_clip"):
                clamped = torch.clamp(xf, min=clip_down, max=clip_up)
                clipped = torch.where(
                    clip_enabled.view((1,) * (xf.ndim - 1) + (-1,)),
                    clamped,
                    xf,
                )

            with wall_tree_cuda_model_block(wall_profiler, x.device, "feature_norm_extra_flags"):
                extra_channels = self._extra_channels.to(device=x.device, dtype=torch.long)
                extra_flags = torch.zeros(
                    xf.shape[:-1] + (self._extra_dim,),
                    device=x.device,
                    dtype=torch.bool,
                )
                extra_spike_positions = self._extra_spike_positions.to(
                    device=x.device,
                    dtype=torch.long,
                )
                extra_flags.scatter_(
                    -1,
                    extra_spike_positions.view((1,) * (xf.ndim - 1) + (-1,)).expand(
                        spike_flags.shape
                    ),
                    spike_flags,
                )
                extra_clip_positions = self._extra_clip_positions.to(
                    device=x.device,
                    dtype=torch.long,
                )
                clip_extra_channels = self._clip_extra_channels.to(
                    device=x.device,
                    dtype=torch.long,
                )
                clip_extra_values = self._clip_extra_values.to(device=x.device, dtype=calc_dtype)
                clip_x = torch.index_select(clipped, -1, clip_extra_channels)
                clip_flags = torch.abs(
                    clip_x - clip_extra_values.view((1,) * (xf.ndim - 1) + (-1,))
                ) <= torch.zeros_like(clip_extra_values).view((1,) * (xf.ndim - 1) + (-1,))
                clip_flags = clip_flags & valid.unsqueeze(-1)
                clip_flags = clip_flags & ~torch.index_select(
                    spike_zero_mask,
                    -1,
                    clip_extra_channels,
                )
                if feature_valid_mask is not None:
                    clip_feature_valid = torch.index_select(
                        feature_valid_mask > 0.5,
                        -1,
                        clip_extra_channels,
                    )
                    clip_flags = clip_flags & clip_feature_valid
                extra_flags.scatter_(
                    -1,
                    extra_clip_positions.view((1,) * (xf.ndim - 1) + (-1,)).expand(
                        clip_flags.shape
                    ),
                    clip_flags,
                )
                zero_mask = _scatter_any_last_dim(
                    extra_flags,
                    extra_channels,
                    channel_count=c,
                )

            with wall_tree_cuda_model_block(wall_profiler, x.device, "feature_norm_output"):
                normalized = (clipped - mean) / std
                out = torch.where(
                    norm_enabled.view((1,) * (xf.ndim - 1) + (-1,)),
                    normalized,
                    clipped,
                )
                out = torch.where(zero_mask, torch.zeros_like(out), out)
                out = torch.where(
                    enabled.view((1,) * (xf.ndim - 1) + (-1,)),
                    out,
                    torch.zeros_like(out),
                )
                extra = extra_flags.to(dtype=x.dtype)
            return out, extra


class _TokenNormIdentity(nn.Module):
    """Passes encoder activations through; symmetric API with :class:`_TokenLayerNormMasked`."""

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2
        assert token_mask.shape == x.shape[:-1]
        return x


class _TokenLayerNormMasked(nn.Module):
    """``LayerNorm`` on the channel dim; invalid tokens (mask false) are zeroed."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self._ln = nn.LayerNorm(int(num_features))

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2
        assert token_mask.shape == x.shape[:-1]
        y = self._ln(x)
        m = (token_mask > 0.5).unsqueeze(-1).to(dtype=y.dtype)
        return y * m


class _TokenEncoderStack(nn.Module):
    """Linear+activation layers with per-layer masked ``LayerNorm`` or identity."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        use_token_batch_norm: bool,
        activation: str,
    ) -> None:
        super().__init__()
        nl = int(n_layers)
        assert nl >= 1, (n_layers,)
        self._in_dim = int(in_dim)
        self._hidden = int(hidden_dim)
        self._activation = _orbit_activation_module(activation)
        self._linears = nn.ModuleList()
        self._norms = nn.ModuleList()
        cur = self._in_dim
        layer_dims = [2 * self._hidden] * (nl - 1) + [self._hidden]
        assert len(layer_dims) == nl, (len(layer_dims), nl)
        for out_dim in layer_dims:
            self._linears.append(nn.Linear(cur, out_dim))
            if bool(use_token_batch_norm):
                self._norms.append(_TokenLayerNormMasked(out_dim))
            else:
                self._norms.append(_TokenNormIdentity())
            cur = out_dim

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self._in_dim, (x.shape, self._in_dim)
        out = x
        for linear, norm in zip(self._linears, self._norms, strict=True):
            out = linear(out)
            out = self._activation(out)
            out = norm(out, token_mask)
        assert out.shape[-1] == self._hidden
        return out


class _EnemySetPooling(nn.Module):
    def __init__(self, *, hidden_dim: int, activation: str) -> None:
        super().__init__()
        h = int(hidden_dim)
        assert h > 0, (hidden_dim,)
        self._hidden = h
        self._sum_pool_projection = nn.Linear(h, h)
        self._max_pool_projection = nn.Linear(h, h)
        self._enemy_count_projection = nn.Linear(1, h)
        self._out_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)

    def forward(
        self,
        enemy_emb: torch.Tensor,
        enemy_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert enemy_emb.ndim >= 3, (enemy_emb.ndim, tuple(enemy_emb.shape))
        assert enemy_emb.shape[-1] == self._hidden, (enemy_emb.shape, self._hidden)
        assert enemy_token_mask.shape == enemy_emb.shape[:-1], (
            enemy_token_mask.shape,
            enemy_emb.shape,
        )
        assert int(enemy_emb.shape[-2]) == ORBIT_ENEMY_AXIS_SLOTS, (
            enemy_emb.shape,
            ORBIT_ENEMY_AXIS_SLOTS,
        )

        mask_bool = enemy_token_mask.to(device=enemy_emb.device, dtype=torch.bool)
        mask = mask_bool.unsqueeze(-1).to(dtype=enemy_emb.dtype)
        enemy_count = mask.sum(dim=-2)
        assert enemy_count.shape == enemy_emb.shape[:-2] + (1,), enemy_count.shape

        sum_pool = (enemy_emb * mask).sum(dim=-2)
        neg_large = torch.finfo(enemy_emb.dtype).min / 16.0
        max_raw = enemy_emb.masked_fill(~mask_bool.unsqueeze(-1), neg_large).max(dim=-2).values
        max_pool = torch.where(enemy_count > 0.0, max_raw, torch.zeros_like(max_raw))
        enemy_count_frac = enemy_count / float(ORBIT_ENEMY_AXIS_SLOTS)
        fused = (
            self._sum_pool_projection(sum_pool)
            + self._max_pool_projection(max_pool)
            + self._enemy_count_projection(enemy_count_frac)
        )
        assert fused.shape == enemy_emb.shape[:-2] + (self._hidden,), fused.shape
        out = self._out_mlp(fused)
        assert out.shape == enemy_emb.shape[:-2] + (self._hidden,), out.shape
        return out


class _SelfEnemyBlockTokenEncoderStack(nn.Module):
    def __init__(
        self,
        *,
        base_dim: int,
        player_block_dim: int,
        hidden_dim: int,
        n_layers: int,
        use_token_batch_norm: bool,
        activation: str,
    ) -> None:
        super().__init__()
        base = int(base_dim)
        block = int(player_block_dim)
        h = int(hidden_dim)
        assert base > 0 and block > 0 and h > 0, (base_dim, player_block_dim, hidden_dim)
        self._base_dim = base
        self._player_block_dim = block
        self._hidden = h
        self._base_encoder = _TokenEncoderStack(
            in_dim=base,
            hidden_dim=h,
            n_layers=n_layers,
            use_token_batch_norm=use_token_batch_norm,
            activation=activation,
        )
        self._self_encoder = _TokenEncoderStack(
            in_dim=block,
            hidden_dim=h,
            n_layers=n_layers,
            use_token_batch_norm=use_token_batch_norm,
            activation=activation,
        )
        self._enemy_encoder = _TokenEncoderStack(
            in_dim=block,
            hidden_dim=h,
            n_layers=n_layers,
            use_token_batch_norm=use_token_batch_norm,
            activation=activation,
        )
        self._enemy_set_pooling = _EnemySetPooling(hidden_dim=h, activation=activation)
        self._base_fusion_projection = nn.Linear(h, h)
        self._self_fusion_projection = nn.Linear(h, h)
        self._enemy_fusion_projection = nn.Linear(h, h)
        self._out_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        enemy_mask: torch.Tensor,
        player_identity_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.ndim >= 3, (x.ndim, tuple(x.shape))
        assert token_mask.shape == x.shape[:-1], (token_mask.shape, x.shape)
        b = int(x.shape[0])
        assert enemy_mask.shape == (b, ORBIT_ENEMY_AXIS_SLOTS), (enemy_mask.shape, b)
        if player_identity_emb is not None:
            assert player_identity_emb.shape == (b, ORBIT_PLAYER_AXIS_SLOTS, self._hidden), (
                player_identity_emb.shape,
                b,
                ORBIT_PLAYER_AXIS_SLOTS,
                self._hidden,
            )
        expected_dim = self._base_dim + ORBIT_PLAYER_AXIS_SLOTS * self._player_block_dim
        assert int(x.shape[-1]) == expected_dim, (tuple(x.shape), expected_dim)

        base = x[..., : self._base_dim]
        blocks = x[..., self._base_dim :].reshape(
            *x.shape[:-1],
            ORBIT_PLAYER_AXIS_SLOTS,
            self._player_block_dim,
        )
        self_block = blocks[..., _ORBIT_SELF_FEATURE_SLOT, :]
        enemy_blocks = blocks[..., 1:, :]
        assert enemy_blocks.shape == x.shape[:-1] + (
            ORBIT_ENEMY_AXIS_SLOTS,
            self._player_block_dim,
        ), enemy_blocks.shape

        base_emb = self._base_encoder(base, token_mask)
        self_emb = self._self_encoder(self_block, token_mask)

        enemy_valid_shape = (b,) + (1,) * (token_mask.ndim - 1) + (ORBIT_ENEMY_AXIS_SLOTS,)
        token_mask_bool = token_mask > 0.5
        enemy_token_mask = token_mask_bool.unsqueeze(-1) & enemy_mask.to(
            device=x.device,
            dtype=torch.bool,
        ).reshape(enemy_valid_shape)
        enemy_emb = self._enemy_encoder(enemy_blocks, enemy_token_mask)
        if player_identity_emb is not None:
            self_identity_shape = (b,) + (1,) * (self_emb.ndim - 2) + (self._hidden,)
            enemy_identity_shape = (b,) + (1,) * (enemy_emb.ndim - 3) + (
                ORBIT_ENEMY_AXIS_SLOTS,
                self._hidden,
            )
            self_emb = self_emb + player_identity_emb[:, 0, :].reshape(self_identity_shape)
            enemy_emb = enemy_emb + player_identity_emb[:, 1:, :].reshape(enemy_identity_shape)
            self_emb = self_emb * token_mask_bool.unsqueeze(-1).to(dtype=self_emb.dtype)
            enemy_emb = enemy_emb * enemy_token_mask.unsqueeze(-1).to(dtype=enemy_emb.dtype)
        enemy_set_emb = self._enemy_set_pooling(enemy_emb, enemy_token_mask)
        assert enemy_set_emb.shape == x.shape[:-1] + (self._hidden,), enemy_set_emb.shape

        fused = (
            self._base_fusion_projection(base_emb)
            + self._self_fusion_projection(self_emb)
            + self._enemy_fusion_projection(enemy_set_emb)
        )
        assert fused.shape == x.shape[:-1] + (self._hidden,), fused.shape
        out = self._out_mlp(fused)
        assert out.shape == x.shape[:-1] + (self._hidden,), out.shape
        return out * (token_mask > 0.5).unsqueeze(-1).to(dtype=out.dtype)


def _assert_integer_feature(x: torch.Tensor, *, label: str) -> torch.Tensor:
    idx = x.to(dtype=torch.int64)
    if _ENABLE_MODEL_ASSERT and torch.is_floating_point(x):
        roundtrip = idx.to(dtype=x.dtype)
        _assert_tensor(torch.all(x == roundtrip), f"{label} must be integer-valued")
    elif not torch.is_floating_point(x):
        integer_dtypes = (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
        assert x.dtype in integer_dtypes, (label, x.dtype)
    return idx


def _assert_nonnegative_integer_feature(
    x: torch.Tensor,
    *,
    max_index: int,
    label: str,
) -> torch.Tensor:
    idx = _assert_integer_feature(x, label=label)
    if _ENABLE_MODEL_ASSERT:
        _assert_tensor(torch.all(idx >= 0), f"{label} must be non-negative")
    return idx


def _orbit_model_temporal_step_max_index(arrival_temporal_horizon: int) -> int:
    h = int(arrival_temporal_horizon)
    assert h in _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS, (
        h,
        _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS,
    )
    assert h <= ORBIT_PLANET_ARRIVAL_HORIZON, (h, ORBIT_PLANET_ARRIVAL_HORIZON)
    return h + 1


def _orbit_available_action_mask_for_model(
    available_action_mask: torch.Tensor,
    *,
    num_policy_actions: int,
) -> torch.Tensor:
    assert available_action_mask.dtype == torch.int8, (
        f"available_action_mask dtype must be torch.int8, got {available_action_mask.dtype}"
    )
    assert available_action_mask.shape[-2:] == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), f"available_action_mask shape mismatch: {tuple(available_action_mask.shape)}"
    if _ENABLE_MODEL_ASSERT:
        _assert_tensor(
            torch.all((available_action_mask == 0) | (available_action_mask == 1)),
            "available_action_mask must contain only 0/1 entries",
        )
    nb = int(num_policy_actions)
    assert 1 <= nb <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        num_policy_actions,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    env_nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)
    n = int(ORBIT_PLANET_ACTION_SLOTS)
    mask_by_dst_action = available_action_mask.reshape(
        *available_action_mask.shape[:-1],
        n,
        env_nb,
    )
    model_mask = mask_by_dst_action[..., :nb].reshape(
        *available_action_mask.shape[:-1],
        n * nb,
    )
    return model_mask > 0


def _assert_orbit_discrete_obs_integer_contract(
    planet_input: torch.Tensor,
    arrival_input: torch.Tensor,
    edge_input: torch.Tensor,
    *,
    temporal_step_max_index: int,
) -> None:
    assert planet_input.shape[-1] == ORBIT_PLANET_FEATURES, planet_input.shape
    assert arrival_input.shape[-1] == ORBIT_PLANET_TEMPORAL_FEATURES, arrival_input.shape
    assert edge_input.shape[-1] == ORBIT_EDGE_FEATURES, edge_input.shape
    if not _ENABLE_MODEL_ASSERT:
        return
    if torch.is_floating_point(arrival_input):
        _assert_tensor(torch.all(torch.isfinite(arrival_input)), "arrival_input must be finite")
    _assert_integer_feature(
        planet_input[..., ORBIT_PLANET_BASE_FEATURE_NEUTRAL_SHIPS],
        label="planet_neutral_ships",
    )
    _assert_nonnegative_integer_feature(
        planet_input[..., ORBIT_PLANET_BASE_FEATURE_EPISODE_STEP],
        max_index=_ORBIT_DISCRETE_EPISODE_STEP_MAX_INDEX,
        label="planet_episode_step",
    )
    _assert_integer_feature(
        planet_input[..., ORBIT_PLANET_BASE_FEATURE_PLANET_PRODUCTION],
        label="planet_production",
    )
    _assert_nonnegative_integer_feature(
        planet_input[..., ORBIT_PLANET_BASE_FEATURE_COMET_TIME_BEFORE_DESPAWN],
        max_index=temporal_step_max_index,
        label="planet_comet_time_before_despawn",
    )
    for player_block in range(ORBIT_PLAYER_AXIS_SLOTS):
        base = (
            ORBIT_PLANET_PLAYER_FEATURE_OFFSET
            + player_block * ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_SHIPS],
            label=f"planet_player_{player_block}_ships_if_owned",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_PRODUCTION],
            label=f"planet_player_{player_block}_production_if_owned",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_OWNER_SURVIVAL_MARGIN],
            label=f"planet_player_{player_block}_owner_survival_margin",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_FLIP_TIME],
            label=f"planet_player_{player_block}_flip_time_by_player",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_STABLE_FLIP_TIME],
            label=f"planet_player_{player_block}_stable_flip_time_by_player",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_OWNER_CHURN],
            label=f"planet_player_{player_block}_owner_churn",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_LAST_DECISIVE_BATTLE_STEP],
            label=f"planet_player_{player_block}_last_decisive_battle_step",
        )
        _assert_integer_feature(
            planet_input[..., base + ORBIT_PLANET_PLAYER_FEATURE_POST_HORIZON_OWNER_MARGIN],
            label=f"planet_player_{player_block}_post_horizon_owner_margin",
        )
    for channel, label in (
        (ORBIT_PLANET_BASE_FEATURE_IS_STATIC, "planet_is_static"),
        (ORBIT_PLANET_BASE_FEATURE_IS_DYNAMIC, "planet_is_dynamic"),
        (ORBIT_PLANET_BASE_FEATURE_IS_COMET, "planet_is_comet"),
    ):
        _assert_nonnegative_integer_feature(
            planet_input[..., channel],
            max_index=1,
            label=label,
        )
    for channel, label in (
        (
            _ORBIT_TEMPORAL_ARRIVAL_SHIPS,
            "temporal_arrival_ships",
        ),
        (
            _ORBIT_TEMPORAL_TAKEOVER_COST,
            "temporal_takeover_cost",
        ),
        (
            _ORBIT_TEMPORAL_TIME_STEP,
            "temporal_time_step",
        ),
        (
            _ORBIT_TEMPORAL_RESOLUTION_SHIPS,
            "temporal_resolution_ships",
        ),
        (
            _ORBIT_TEMPORAL_STABLE_TAKEOVER_COST,
            "temporal_stable_takeover_cost",
        ),
        (
            _ORBIT_TEMPORAL_HOLD_COST,
            "temporal_hold_cost",
        ),
        (
            _ORBIT_TEMPORAL_HOLD_VALID,
            "temporal_hold_valid",
        ),
        (
            _ORBIT_TEMPORAL_NEUTRALIZATION_COST,
            "temporal_neutralization_cost",
        ),
        (
            _ORBIT_TEMPORAL_NEUTRALIZATION_VALID,
            "temporal_neutralization_valid",
        ),
        (
            _ORBIT_TEMPORAL_DENY_STABLE_ENEMY_COST,
            "temporal_deny_stable_enemy_cost",
        ),
        (
            _ORBIT_TEMPORAL_BATTLE_TIE_DISTANCE,
            "temporal_battle_tie_distance",
        ),
        (
            _ORBIT_TEMPORAL_BATTLE_TIE_VALID,
            "temporal_battle_tie_valid",
        ),
    ):
        _assert_integer_feature(
            arrival_input[..., channel],
            label=label,
        )
    _assert_integer_feature(
        arrival_input[..., _ORBIT_TEMPORAL_RESOLUTION_OWNER],
        label="temporal_resolution_owner",
    )
    for channel, label in (
        (ORBIT_EDGE_BASE_FEATURE_SRC_NEUTRAL, "edge_src_neutral"),
        (ORBIT_EDGE_BASE_FEATURE_DST_NEUTRAL, "edge_dst_neutral"),
        (ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET, "edge_min_takeover_bucket"),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET,
            "edge_min_stable_takeover_bucket",
        ),
        (ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET, "edge_min_neutralize_bucket"),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET_AVAILABLE,
            "edge_min_takeover_bucket_available",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET_HIT_STEPS,
            "edge_min_takeover_bucket_hit_steps",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET,
            "edge_min_time_takeover_bucket",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET_AVAILABLE,
            "edge_min_time_takeover_bucket_available",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET_HIT_STEPS,
            "edge_min_time_takeover_bucket_hit_steps",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_AVAILABLE,
            "edge_min_stable_takeover_bucket_available",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_HIT_STEPS,
            "edge_min_stable_takeover_bucket_hit_steps",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET,
            "edge_min_time_stable_takeover_bucket",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET_AVAILABLE,
            "edge_min_time_stable_takeover_bucket_available",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET_HIT_STEPS,
            "edge_min_time_stable_takeover_bucket_hit_steps",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET_AVAILABLE,
            "edge_min_neutralize_bucket_available",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET_HIT_STEPS,
            "edge_min_neutralize_bucket_hit_steps",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET,
            "edge_min_time_neutralize_bucket",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET_AVAILABLE,
            "edge_min_time_neutralize_bucket_available",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET_HIT_STEPS,
            "edge_min_time_neutralize_bucket_hit_steps",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_TAKEOVER_MARGIN_WITH_MAX_SEND,
            "edge_takeover_margin_with_max_send",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_STABLE_MARGIN_WITH_MAX_SEND,
            "edge_stable_margin_with_max_send",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_NEUTRALIZE_MARGIN_WITH_MAX_SEND,
            "edge_neutralize_margin_with_max_send",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_TIME_TO_HIT_WITH_MAX_SEND,
            "edge_time_to_hit_with_max_send",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_IS_AVAILABLE_WITH_MAX_SEND,
            "edge_is_available_with_max_send",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_DST_FINAL_OWNER_IS_SRC_OWNER_WITHOUT_ACTION,
            "edge_dst_final_owner_is_src_owner_without_action",
        ),
        (
            ORBIT_EDGE_BASE_FEATURE_ATTACK_REDUNDANCY_SCORE,
            "edge_attack_redundancy_score",
        ),
    ):
        _assert_integer_feature(edge_input[..., channel], label=label)
    for player_block in range(ORBIT_PLAYER_AXIS_SLOTS):
        base = ORBIT_EDGE_PLAYER_FEATURE_OFFSET + player_block * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
        _assert_integer_feature(
            edge_input[..., base + ORBIT_EDGE_PLAYER_FEATURE_SRC_OWNED],
            label=f"edge_player_{player_block}_src_owned",
        )
        _assert_integer_feature(
            edge_input[..., base + ORBIT_EDGE_PLAYER_FEATURE_DST_OWNED],
            label=f"edge_player_{player_block}_dst_owned",
        )


def _self_enemy_block_feature_valid_mask(
    x: torch.Tensor,
    *,
    player_feature_offset: int,
    player_features_per_player: int,
    enemy_mask: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim >= 3, (x.ndim, tuple(x.shape))
    off = int(player_feature_offset)
    width = int(player_features_per_player)
    assert off > 0 and width > 0, (off, width)
    assert int(x.shape[-1]) == off + ORBIT_PLAYER_AXIS_SLOTS * width, (
        tuple(x.shape),
        off,
        width,
        ORBIT_PLAYER_AXIS_SLOTS,
    )
    b = int(x.shape[0])
    assert enemy_mask.shape == (b, ORBIT_ENEMY_AXIS_SLOTS), (enemy_mask.shape, b)
    base_valid = torch.ones(x.shape[:-1] + (off,), device=x.device, dtype=torch.bool)
    player_valid = torch.cat(
        [
            torch.ones((b, 1), device=x.device, dtype=torch.bool),
            enemy_mask.to(device=x.device, dtype=torch.bool),
        ],
        dim=1,
    )
    mask_shape = (b,) + (1,) * (x.ndim - 2) + (ORBIT_PLAYER_AXIS_SLOTS, 1)
    player_valid = player_valid.reshape(mask_shape).expand(
        x.shape[:-1] + (ORBIT_PLAYER_AXIS_SLOTS, width)
    )
    player_valid = player_valid.reshape(x.shape[:-1] + (ORBIT_PLAYER_AXIS_SLOTS * width,))
    out = torch.cat([base_valid, player_valid], dim=-1)
    assert out.shape == x.shape, (out.shape, x.shape)
    return out


def _orbit_identity_lookup(
    batch_size: int,
    *,
    shuffle_identity_ids: bool,
    device: torch.device,
) -> torch.Tensor:
    b = int(batch_size)
    assert b > 0, (batch_size,)
    assert ORBIT_PLAYER_AXIS_SLOTS == ORBIT_ENEMY_AXIS_SLOTS + 1, (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_ENEMY_AXIS_SLOTS,
    )
    if shuffle_identity_ids:
        enemy_identity = torch.argsort(
            torch.rand((b, ORBIT_ENEMY_AXIS_SLOTS), device=device),
            dim=1,
        ) + 1
    else:
        enemy_identity = torch.arange(
            1,
            ORBIT_PLAYER_AXIS_SLOTS,
            device=device,
            dtype=torch.long,
        ).reshape(1, ORBIT_ENEMY_AXIS_SLOTS).expand(b, ORBIT_ENEMY_AXIS_SLOTS)
    lookup = torch.cat(
        [
            torch.full(
                (b, 1),
                _ORBIT_SELF_IDENTITY_CLASS,
                device=device,
                dtype=torch.long,
            ),
            enemy_identity.to(dtype=torch.long),
            torch.full(
                (b, 1),
                _ORBIT_NEUTRAL_IDENTITY_CLASS,
                device=device,
                dtype=torch.long,
            ),
        ],
        dim=1,
    )
    assert lookup.shape == (b, _ORBIT_IDENTITY_CLASSES), lookup.shape
    return lookup


def _orbit_edge_relation_features(
    edge_input_raw: torch.Tensor,
    pair_valid_bool: torch.Tensor,
) -> torch.Tensor:
    assert edge_input_raw.ndim == 4, edge_input_raw.shape
    b, n_src, n_dst, c = edge_input_raw.shape
    assert n_src == ORBIT_MAX_PLANETS and n_dst == ORBIT_MAX_PLANETS, edge_input_raw.shape
    assert c == ORBIT_EDGE_FEATURES, edge_input_raw.shape
    assert pair_valid_bool.shape == (b, n_src, n_dst), (
        pair_valid_bool.shape,
        edge_input_raw.shape,
    )
    assert pair_valid_bool.dtype == torch.bool, pair_valid_bool.dtype

    src_neutral = edge_input_raw[..., ORBIT_EDGE_BASE_FEATURE_SRC_NEUTRAL] > 0.5
    dst_neutral = edge_input_raw[..., ORBIT_EDGE_BASE_FEATURE_DST_NEUTRAL] > 0.5
    src_owned = torch.stack(
        [
            edge_input_raw[
                ...,
                ORBIT_EDGE_PLAYER_FEATURE_OFFSET
                + player_block * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
                + ORBIT_EDGE_PLAYER_FEATURE_SRC_OWNED,
            ]
            > 0.5
            for player_block in range(ORBIT_PLAYER_AXIS_SLOTS)
        ],
        dim=-1,
    )
    dst_owned = torch.stack(
        [
            edge_input_raw[
                ...,
                ORBIT_EDGE_PLAYER_FEATURE_OFFSET
                + player_block * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
                + ORBIT_EDGE_PLAYER_FEATURE_DST_OWNED,
            ]
            > 0.5
            for player_block in range(ORBIT_PLAYER_AXIS_SLOTS)
        ],
        dim=-1,
    )
    src_owner_count = src_owned.to(dtype=torch.int64).sum(dim=-1) + src_neutral.to(
        dtype=torch.int64
    )
    dst_owner_count = dst_owned.to(dtype=torch.int64).sum(dim=-1) + dst_neutral.to(
        dtype=torch.int64
    )
    if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
        _assert_tensor(
            torch.all(src_owner_count[pair_valid_bool] == 1),
            "valid edge relation src must have exactly one owner class",
        )
        _assert_tensor(
            torch.all(dst_owner_count[pair_valid_bool] == 1),
            "valid edge relation dst must have exactly one owner class",
        )

    src_self = src_owned[..., 0]
    dst_self = dst_owned[..., 0]
    src_enemy = src_owned[..., 1:].any(dim=-1)
    dst_enemy = dst_owned[..., 1:].any(dim=-1)
    same_player_owner = (src_owned & dst_owned).any(dim=-1)
    different_player_owner = src_owned.any(dim=-1) & dst_owned.any(dim=-1) & ~same_player_owner
    same_enemy_owner = (src_owned[..., 1:] & dst_owned[..., 1:]).any(dim=-1)
    different_enemy_owner = src_enemy & dst_enemy & ~same_enemy_owner
    relation = torch.stack(
        (
            same_player_owner,
            different_player_owner,
            src_self & dst_self,
            same_enemy_owner,
            different_enemy_owner,
            src_self & dst_enemy,
            src_enemy & dst_self,
            src_self & dst_neutral,
            src_neutral & dst_self,
            src_enemy & dst_neutral,
            src_neutral & dst_enemy,
            src_neutral & dst_neutral,
        ),
        dim=-1,
    ).to(dtype=edge_input_raw.dtype)
    assert relation.shape == (b, n_src, n_dst, _ORBIT_EDGE_RELATION_FEATURES), relation.shape
    return relation


class _OrbitDiscreteFeatureEmbeddings(nn.Module):
    def __init__(
        self,
        feature_config: dict[str, Any],
        zeroed_inputs: tuple[_ZeroedOrbitObsFeatureInput, ...],
        *,
        temporal_step_max_index: int,
    ) -> None:
        super().__init__()
        self._zeroed_inputs = zeroed_inputs
        self._temporal_step_max_index = int(temporal_step_max_index)
        e = _ORBIT_DISCRETE_EMBED_DIM
        self._ship_count_embedding = nn.Embedding(_ORBIT_DISCRETE_SHIP_MAX_INDEX + 1, e)
        self._ship_cost_embedding = nn.Embedding(_ORBIT_DISCRETE_SHIP_COST_MAX_INDEX + 1, e)
        self._production_embedding = nn.Embedding(_ORBIT_DISCRETE_PRODUCTION_MAX_INDEX + 1, e)
        self._ship_bucket_embedding = nn.Embedding(_ORBIT_DISCRETE_SHIP_BUCKET_MAX_INDEX + 1, e)
        self._temporal_step_embedding = nn.Embedding(
            self._temporal_step_max_index + 1, e
        )
        self._episode_step_embedding = nn.Embedding(
            _ORBIT_DISCRETE_EPISODE_STEP_MAX_INDEX + 1, e
        )
        self._signed_ship_margin_embedding = nn.Embedding(
            2 * _ORBIT_DISCRETE_SIGNED_SHIP_MARGIN_ABS_MAX_INDEX + 1,
            e,
        )
        self._planet_base_embed_specs = orbit_discrete_embedding_planet_base_specs()
        self._planet_player_embed_specs = orbit_discrete_embedding_planet_player_block_specs()
        self._arrival_embed_specs = orbit_discrete_embedding_arrival_temporal_specs()
        self._edge_base_embed_specs = orbit_discrete_embedding_edge_base_specs()
        self._edge_player_embed_specs = orbit_discrete_embedding_edge_player_block_specs()
        self._planet_base_embed_enabled = self._spec_enabled("planet", self._planet_base_embed_specs, feature_config)
        self._planet_player_embed_enabled = self._spec_enabled(
            "planet",
            self._planet_player_embed_specs,
            feature_config,
        )
        self._arrival_embed_enabled = self._spec_enabled(
            "arrival",
            self._arrival_embed_specs,
            feature_config,
        )
        self._edge_base_embed_enabled = self._spec_enabled("edge", self._edge_base_embed_specs, feature_config)
        self._edge_player_embed_enabled = self._spec_enabled("edge", self._edge_player_embed_specs, feature_config)
        self._validate_zeroed_embedding_inputs()
        self._planet_base_embed_zeroed = self._spec_zeroed(
            "planet",
            self._planet_base_embed_specs,
            player="",
        )
        self._planet_player_self_embed_zeroed = self._spec_zeroed(
            "planet",
            self._planet_player_embed_specs,
            player="self",
        )
        self._planet_player_enemy_embed_zeroed = self._spec_zeroed(
            "planet",
            self._planet_player_embed_specs,
            player="enemy",
        )
        self._arrival_enemy_embed_zeroed = self._spec_zeroed(
            "arrival",
            self._arrival_embed_specs,
            player="enemy",
        )
        self._edge_base_embed_zeroed = self._spec_zeroed(
            "edge",
            self._edge_base_embed_specs,
            player="",
        )

    @staticmethod
    def _spec_enabled(
        prefix: str,
        specs: tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...],
        feature_config: dict[str, Any],
    ) -> tuple[bool, ...]:
        out: list[bool] = []
        for spec in specs:
            key = f"continuous.{prefix}.{spec.label}"
            assert key in feature_config, (key, sorted(feature_config.keys()))
            out.append(bool(_feature_norm_spec_mapping(feature_config[key])["enabled"]))
        return tuple(out)

    def _spec_zeroed(
        self,
        prefix: str,
        specs: tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...],
        *,
        player: str,
    ) -> tuple[bool, ...]:
        assert prefix in ("planet", "arrival", "edge"), prefix
        assert player in ("", "self", "enemy"), player
        out: list[bool] = []
        for spec in specs:
            zeroed = False
            for item in self._zeroed_inputs:
                if item.kind != "embedding" or item.domain != prefix:
                    continue
                assert item.player in ("", "self", "enemy"), item
                if item.feature_name == spec.label and item.player == player:
                    zeroed = True
            out.append(zeroed)
        return tuple(out)

    def _validate_zeroed_embedding_inputs(self) -> None:
        valid: set[tuple[str, str, str]] = set()
        for spec in self._planet_base_embed_specs:
            valid.add(("planet", "", spec.label))
        for spec in self._planet_player_embed_specs:
            valid.add(("planet", "self", spec.label))
            valid.add(("planet", "enemy", spec.label))
        for spec in self._arrival_embed_specs:
            valid.add(("arrival", "self", spec.label))
            valid.add(("arrival", "enemy", spec.label))
        for spec in self._edge_base_embed_specs:
            valid.add(("edge", "", spec.label))
        for spec in self._edge_player_embed_specs:
            valid.add(("edge", "self", spec.label))
            valid.add(("edge", "enemy", spec.label))
        for item in self._zeroed_inputs:
            if item.kind != "embedding":
                continue
            key = (item.domain, item.player, item.feature_name)
            assert key in valid, item

    @staticmethod
    def _extra_dim(specs: tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...]) -> int:
        return orbit_discrete_embedding_count(specs) * _ORBIT_DISCRETE_EMBED_DIM

    @classmethod
    def planet_base_extra_dim(cls) -> int:
        return cls._extra_dim(orbit_discrete_embedding_planet_base_specs())

    @classmethod
    def planet_player_extra_dim(cls) -> int:
        return cls._extra_dim(orbit_discrete_embedding_planet_player_block_specs())

    @classmethod
    def arrival_extra_dim(cls) -> int:
        return cls._extra_dim(orbit_discrete_embedding_arrival_temporal_specs())

    @classmethod
    def edge_base_extra_dim(cls) -> int:
        return cls._extra_dim(orbit_discrete_embedding_edge_base_specs())

    @classmethod
    def edge_player_extra_dim(cls) -> int:
        return cls._extra_dim(orbit_discrete_embedding_edge_player_block_specs())

    def _ship_count_indices(self, x: torch.Tensor, *, label: str) -> torch.Tensor:
        idx = _assert_nonnegative_integer_feature(
            x,
            max_index=_ORBIT_DISCRETE_SHIP_MAX_INDEX,
            label=label,
        )
        return idx.clamp_max(_ORBIT_DISCRETE_SHIP_MAX_INDEX)

    def _ship_bucket_indices(self, x: torch.Tensor, *, label: str) -> torch.Tensor:
        idx = _assert_nonnegative_integer_feature(
            x,
            max_index=_ORBIT_DISCRETE_SHIP_BUCKET_MAX_INDEX,
            label=label,
        )
        return idx.clamp_max(_ORBIT_DISCRETE_SHIP_BUCKET_MAX_INDEX)

    def _temporal_step_indices(self, x: torch.Tensor, *, label: str) -> torch.Tensor:
        idx = _assert_nonnegative_integer_feature(
            x,
            max_index=self._temporal_step_max_index,
            label=label,
        )
        return idx.clamp_max(self._temporal_step_max_index)

    def _episode_step_indices(self, x: torch.Tensor, *, label: str) -> torch.Tensor:
        idx = _assert_nonnegative_integer_feature(
            x,
            max_index=_ORBIT_DISCRETE_EPISODE_STEP_MAX_INDEX,
            label=label,
        )
        return idx.clamp_max(_ORBIT_DISCRETE_EPISODE_STEP_MAX_INDEX)

    def _signed_ship_margin_indices(self, x: torch.Tensor, *, label: str) -> torch.Tensor:
        idx = _assert_integer_feature(x, label=label)
        m = _ORBIT_DISCRETE_SIGNED_SHIP_MARGIN_ABS_MAX_INDEX
        out = idx.clamp(min=-m, max=m) + m
        assert out.shape == idx.shape, (out.shape, idx.shape)
        return out

    def _discrete_embed(
        self,
        x: torch.Tensor,
        table: BcObsDiscreteEmbeddingTable,
        *,
        label: str,
    ) -> torch.Tensor:
        if table == "ship_count":
            return self._ship_count_embedding(self._ship_count_indices(x, label=label))
        if table == "ship_cost":
            idx = _assert_nonnegative_integer_feature(
                x,
                max_index=_ORBIT_DISCRETE_SHIP_COST_MAX_INDEX,
                label=label,
            )
            return self._ship_cost_embedding(
                idx.clamp_max(_ORBIT_DISCRETE_SHIP_COST_MAX_INDEX)
            )
        if table == "production":
            idx = _assert_nonnegative_integer_feature(
                x,
                max_index=_ORBIT_DISCRETE_PRODUCTION_MAX_INDEX,
                label=label,
            )
            return self._production_embedding(
                idx.clamp_max(_ORBIT_DISCRETE_PRODUCTION_MAX_INDEX)
            )
        if table == "ship_bucket":
            return self._ship_bucket_embedding(self._ship_bucket_indices(x, label=label))
        if table == "temporal_step":
            return self._temporal_step_embedding(self._temporal_step_indices(x, label=label))
        if table == "episode_step":
            return self._episode_step_embedding(self._episode_step_indices(x, label=label))
        if table == "signed_ship_margin":
            return self._signed_ship_margin_embedding(
                self._signed_ship_margin_indices(x, label=label)
            )
        raise AssertionError(f"unknown discrete embedding table: {table}")

    def _discrete_embedding_weight(self, table: BcObsDiscreteEmbeddingTable) -> torch.Tensor:
        if table == "ship_count":
            return self._ship_count_embedding.weight
        if table == "ship_cost":
            return self._ship_cost_embedding.weight
        if table == "production":
            return self._production_embedding.weight
        if table == "ship_bucket":
            return self._ship_bucket_embedding.weight
        if table == "temporal_step":
            return self._temporal_step_embedding.weight
        if table == "episode_step":
            return self._episode_step_embedding.weight
        if table == "signed_ship_margin":
            return self._signed_ship_margin_embedding.weight
        raise AssertionError(f"unknown discrete embedding table: {table}")

    def _embed_specs(
        self,
        values: torch.Tensor,
        specs: tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...],
        enabled: tuple[bool, ...],
        zeroed: tuple[bool, ...],
        *,
        channel_offset: int,
        label_suffix: str,
    ) -> list[torch.Tensor]:
        assert len(enabled) == len(specs), (len(enabled), len(specs))
        assert len(zeroed) == len(specs), (len(zeroed), len(specs))
        parts: list[torch.Tensor] = []
        for spec, spec_enabled, spec_zeroed in zip(specs, enabled, zeroed, strict=True):
            ch = int(spec.channel_index) + int(channel_offset)
            label = spec.label
            if label_suffix:
                label = f"{label}@{label_suffix}"
            raw_value = values[..., ch]
            emb = self._discrete_embed(raw_value, spec.table, label=label)
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                nonfinite_embedding = ~torch.isfinite(emb)
                if bool(nonfinite_embedding.any().item()):
                    bad_indices = torch.nonzero(
                        nonfinite_embedding.flatten(start_dim=-1).any(dim=-1),
                        as_tuple=False,
                    )
                    assert bad_indices.ndim == 2 and int(bad_indices.shape[0]) > 0, emb.shape
                    bad_index = tuple(int(v.item()) for v in bad_indices[0])
                    bad_raw_value = raw_value[bad_index]
                    bad_lookup_index = int(bad_raw_value.to(dtype=torch.int64).item())
                    embedding_weight = self._discrete_embedding_weight(spec.table)
                    raise AssertionError(
                        {
                            "head": "orbit_discrete_embedding",
                            "reason": "discrete embedding lookup produced non-finite values",
                            "label": label,
                            "table": spec.table,
                            "channel": ch,
                            "spec_enabled": bool(spec_enabled),
                            "spec_zeroed": bool(spec_zeroed),
                            "bad_index": bad_index,
                            "raw_value": bad_raw_value.detach().cpu(),
                            "lookup_index": bad_lookup_index,
                            "embedding_vector": emb[bad_index].detach().cpu(),
                            "embedding_weight_row": embedding_weight[bad_lookup_index].detach().cpu(),
                        }
                    )
            if not spec_enabled or spec_zeroed:
                emb = torch.zeros_like(emb)
            parts.append(emb)
        return parts

    def _planet_embeddings(
        self,
        planet_input: torch.Tensor,
        planet_mask: torch.Tensor,
        wall_profiler: WallTreeProfiler | None,
    ) -> torch.Tensor:
        assert planet_input.shape[-1] == ORBIT_PLANET_FEATURES, planet_input.shape
        assert planet_mask.shape == planet_input.shape[:-1], (planet_mask.shape, planet_input.shape)
        device = planet_input.device
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_planet_base"):
            base_parts = self._embed_specs(
                planet_input,
                self._planet_base_embed_specs,
                self._planet_base_embed_enabled,
                self._planet_base_embed_zeroed,
                channel_offset=0,
                label_suffix="",
            )
            base_extra = torch.cat(base_parts, dim=-1)
        assert base_extra.shape == planet_input.shape[:-1] + (
            self.planet_base_extra_dim(),
        ), base_extra.shape
        player_extras: list[torch.Tensor] = []
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_planet_player"):
            for player_block in range(ORBIT_PLAYER_AXIS_SLOTS):
                block_offset = (
                    ORBIT_PLANET_PLAYER_FEATURE_OFFSET
                    + player_block * ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
                )
                player_name = "self" if player_block == 0 else "enemy"
                player_zeroed = (
                    self._planet_player_self_embed_zeroed
                    if player_name == "self"
                    else self._planet_player_enemy_embed_zeroed
                )
                player_parts = self._embed_specs(
                    planet_input,
                    self._planet_player_embed_specs,
                    self._planet_player_embed_enabled,
                    player_zeroed,
                    channel_offset=block_offset,
                    label_suffix=f"player{player_block}",
                )
                player_extra = torch.cat(player_parts, dim=-1)
                assert player_extra.shape == planet_input.shape[:-1] + (
                    self.planet_player_extra_dim(),
                ), player_extra.shape
                player_extras.append(player_extra)
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_planet_cat_mask"):
            out = torch.cat([base_extra] + player_extras, dim=-1)
            out = out * (planet_mask > 0.5).unsqueeze(-1).to(dtype=out.dtype)
        expected_dim = self.planet_base_extra_dim() + (
            ORBIT_PLAYER_AXIS_SLOTS * self.planet_player_extra_dim()
        )
        assert out.shape == planet_input.shape[:-1] + (expected_dim,), out.shape
        return out

    def _arrival_embeddings(
        self,
        arrival_input: torch.Tensor,
        arrival_valid: torch.Tensor,
        wall_profiler: WallTreeProfiler | None,
    ) -> torch.Tensor:
        assert arrival_input.shape[-1] == ORBIT_PLANET_TEMPORAL_FEATURES, arrival_input.shape
        assert arrival_valid.shape == arrival_input.shape[:-1], (
            arrival_valid.shape,
            arrival_input.shape,
        )
        assert arrival_input.shape[-2] == ORBIT_PLAYER_AXIS_SLOTS, arrival_input.shape
        device = arrival_input.device
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_arrival_lookup_cat"):
            enemy_arrival_input = arrival_input[..., 1:, :]
            assert enemy_arrival_input.shape == arrival_input.shape[:-2] + (
                ORBIT_ENEMY_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ), enemy_arrival_input.shape
            parts = self._embed_specs(
                enemy_arrival_input,
                self._arrival_embed_specs,
                self._arrival_embed_enabled,
                self._arrival_enemy_embed_zeroed,
                channel_offset=0,
                label_suffix="",
            )
            enemy_out = torch.cat(parts, dim=-1)
            assert enemy_out.shape == arrival_input.shape[:-2] + (
                ORBIT_ENEMY_AXIS_SLOTS,
                self.arrival_extra_dim(),
            ), enemy_out.shape
            self_out = torch.zeros(
                arrival_input.shape[:-2] + (1, self.arrival_extra_dim()),
                device=device,
                dtype=enemy_out.dtype,
            )
            out = torch.cat([self_out, enemy_out], dim=-2)
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_arrival_mask"):
            out = out * arrival_valid.unsqueeze(-1).to(dtype=out.dtype)
        assert out.shape == arrival_input.shape[:-1] + (self.arrival_extra_dim(),), out.shape
        return out

    def _edge_embeddings(
        self,
        edge_input: torch.Tensor,
        pair_valid: torch.Tensor,
        wall_profiler: WallTreeProfiler | None,
    ) -> torch.Tensor:
        assert edge_input.shape[-1] == ORBIT_EDGE_FEATURES, edge_input.shape
        assert pair_valid.shape == edge_input.shape[:-1], (pair_valid.shape, edge_input.shape)
        device = edge_input.device
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_edge_lookup_cat"):
            base_parts = self._embed_specs(
                edge_input,
                self._edge_base_embed_specs,
                self._edge_base_embed_enabled,
                self._edge_base_embed_zeroed,
                channel_offset=0,
                label_suffix="",
            )
            base_extra = torch.cat(base_parts, dim=-1)
        assert base_extra.shape == edge_input.shape[:-1] + (
            self.edge_base_extra_dim(),
        ), base_extra.shape
        out = base_extra
        expected_dim = self.edge_base_extra_dim() + (
            ORBIT_PLAYER_AXIS_SLOTS * self.edge_player_extra_dim()
        )
        assert out.shape == edge_input.shape[:-1] + (expected_dim,), out.shape
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_edge_mask"):
            out = out * (pair_valid > 0.5).unsqueeze(-1).to(dtype=out.dtype)
        return out

    def forward(
        self,
        planet_input: torch.Tensor,
        arrival_input: torch.Tensor,
        edge_input: torch.Tensor,
        planet_mask: torch.Tensor,
        arrival_valid: torch.Tensor,
        pair_valid: torch.Tensor,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = planet_input.device
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_planet"):
            planet_extra = self._planet_embeddings(planet_input, planet_mask, wall_profiler)
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_arrival"):
            arrival_extra = self._arrival_embeddings(arrival_input, arrival_valid, wall_profiler)
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_discrete_edge"):
            edge_extra = self._edge_embeddings(edge_input, pair_valid, wall_profiler)
        expected_planet_dim = self.planet_base_extra_dim() + (
            ORBIT_PLAYER_AXIS_SLOTS * self.planet_player_extra_dim()
        )
        assert planet_extra.shape == planet_input.shape[:-1] + (
            expected_planet_dim,
        ), planet_extra.shape
        assert arrival_extra.shape == arrival_input.shape[:-1] + (
            self.arrival_extra_dim(),
        ), arrival_extra.shape
        expected_edge_dim = self.edge_base_extra_dim() + (
            ORBIT_PLAYER_AXIS_SLOTS * self.edge_player_extra_dim()
        )
        assert edge_extra.shape == edge_input.shape[:-1] + (expected_edge_dim,), edge_extra.shape
        assert edge_extra.shape[:-1] == pair_valid.shape, (
            edge_extra.shape,
            pair_valid.shape,
        )
        return planet_extra, arrival_extra, edge_extra


class _TemporalPlanetFeatureCnn(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        n_layers: int,
        recency_decay_horizon: int,
        activation: str,
    ) -> None:
        super().__init__()
        c = int(in_channels)
        h = int(hidden_dim)
        nl = int(n_layers)
        recency_horizon = int(recency_decay_horizon)
        assert c > 0, (in_channels,)
        assert h > 0, (hidden_dim,)
        assert nl >= 1, (n_layers,)
        assert recency_horizon in _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS, (
            recency_horizon,
            _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS,
        )
        self._in_channels = c
        self._hidden = h
        self._recency_decay_horizon = recency_horizon
        self._self_step_encoder = _orbit_mlp(c, 2 * h, h, n_layers=3, activation=activation)
        self._enemy_step_encoder = _orbit_mlp(c, 2 * h, h, n_layers=3, activation=activation)
        self._enemy_set_pooling = _EnemySetPooling(hidden_dim=h, activation=activation)
        self._self_step_fusion_projection = nn.Linear(h, h)
        self._enemy_step_fusion_projection = nn.Linear(h, h)
        self._step_projection = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        self._temporal_net = self._make_net(h, h, nl, activation=activation)
        self._mean_pool_projection = nn.Linear(h, h)
        self._max_pool_projection = nn.Linear(h, h)
        self._recency_pool_projection = nn.Linear(h, h)
        self._pool_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)

    @staticmethod
    def _make_net(
        in_channels: int,
        hidden_dim: int,
        n_layers: int,
        *,
        activation: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        cur = int(in_channels)
        h = int(hidden_dim)
        for layer_idx in range(int(n_layers)):
            dilation = 1 << layer_idx
            layers.append(
                nn.Conv1d(
                    cur,
                    h,
                    kernel_size=5,
                    padding=2 * dilation,
                    dilation=dilation,
                )
            )
            layers.append(_orbit_activation_module(activation))
            cur = h
        return nn.Sequential(*layers)

    def _encode_flat_sequences(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert x.ndim == 3, (x.ndim, tuple(x.shape))
        m, t, c = x.shape
        assert c == self._hidden, (c, self._hidden)
        y = x.transpose(1, 2).contiguous()
        assert y.shape == (m, self._hidden, t), y.shape
        y = self._temporal_net(y)
        assert y.shape == (m, self._hidden, t), y.shape
        mean_pool = y.mean(dim=2)
        max_pool = y.max(dim=2).values
        steps = torch.arange(t, device=y.device, dtype=torch.float32)
        recency_weights = torch.exp(-steps / float(self._recency_decay_horizon))
        recency_weights = recency_weights / recency_weights.sum()
        recency_pool = (y * recency_weights.to(dtype=y.dtype).view(1, 1, t)).sum(dim=2)
        pooled = (
            self._mean_pool_projection(mean_pool)
            + self._max_pool_projection(max_pool)
            + self._recency_pool_projection(recency_pool)
        )
        assert pooled.shape == (m, self._hidden), pooled.shape
        out = self._pool_mlp(pooled)
        assert out.shape == (m, self._hidden), out.shape
        return out

    def forward(
        self,
        x: torch.Tensor,
        enemy_mask: torch.Tensor,
        player_identity_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.ndim == 5, (x.ndim, tuple(x.shape))
        b, n, t, p, f = x.shape
        assert p == ORBIT_PLAYER_AXIS_SLOTS, (p, ORBIT_PLAYER_AXIS_SLOTS)
        assert f == self._in_channels, (f, self._in_channels)
        assert enemy_mask.shape == (b, ORBIT_ENEMY_AXIS_SLOTS), (
            enemy_mask.shape,
            b,
            ORBIT_ENEMY_AXIS_SLOTS,
        )
        if player_identity_emb is not None:
            assert player_identity_emb.shape == (b, ORBIT_PLAYER_AXIS_SLOTS, self._hidden), (
                player_identity_emb.shape,
                b,
                ORBIT_PLAYER_AXIS_SLOTS,
                self._hidden,
            )
        self_x = x[:, :, :, _ORBIT_SELF_FEATURE_SLOT, :]
        assert self_x.shape == (b, n, t, f), self_x.shape
        self_step_emb = self._self_step_encoder(self_x)
        assert self_step_emb.shape == (b, n, t, self._hidden), self_step_emb.shape

        enemy_x = x[:, :, :, 1:, :]
        assert enemy_x.shape == (b, n, t, ORBIT_ENEMY_AXIS_SLOTS, f), enemy_x.shape
        enemy_step_emb = self._enemy_step_encoder(enemy_x)
        assert enemy_step_emb.shape == (
            b,
            n,
            t,
            ORBIT_ENEMY_AXIS_SLOTS,
            self._hidden,
        ), enemy_step_emb.shape
        enemy_token_mask = enemy_mask.to(device=x.device, dtype=torch.bool).view(
            b,
            1,
            1,
            ORBIT_ENEMY_AXIS_SLOTS,
        )
        enemy_token_mask = enemy_token_mask.expand(b, n, t, ORBIT_ENEMY_AXIS_SLOTS)
        if player_identity_emb is not None:
            self_step_emb = self_step_emb + player_identity_emb[:, 0, :].reshape(
                b,
                1,
                1,
                self._hidden,
            )
            enemy_step_emb = enemy_step_emb + player_identity_emb[:, 1:, :].reshape(
                b,
                1,
                1,
                ORBIT_ENEMY_AXIS_SLOTS,
                self._hidden,
            )
            enemy_step_emb = enemy_step_emb * enemy_token_mask.unsqueeze(-1).to(
                dtype=enemy_step_emb.dtype
            )
        enemy_set_step_emb = self._enemy_set_pooling(enemy_step_emb, enemy_token_mask)
        assert enemy_set_step_emb.shape == (
            b,
            n,
            t,
            self._hidden,
        ), enemy_set_step_emb.shape

        step_fused = (
            self._self_step_fusion_projection(self_step_emb)
            + self._enemy_step_fusion_projection(enemy_set_step_emb)
        )
        assert step_fused.shape == (b, n, t, self._hidden), step_fused.shape
        step_emb = self._step_projection(step_fused)
        assert step_emb.shape == (b, n, t, self._hidden), step_emb.shape
        out = self._encode_flat_sequences(step_emb.reshape(b * n, t, self._hidden)).reshape(
            b,
            n,
            self._hidden,
        )
        assert out.shape == (b, n, self._hidden)
        return out


class _PlanetArrivalAttentionEncoder(nn.Module):
    def __init__(
        self,
        *,
        arrival_in_channels: int,
        arrival_temporal_horizon: int,
        hidden_dim: int,
        n_layers: int,
        num_heads: int,
        num_queries: int,
        activation: str,
    ) -> None:
        super().__init__()
        c = int(arrival_in_channels)
        t = int(arrival_temporal_horizon)
        h = int(hidden_dim)
        nl = int(n_layers)
        nh = int(num_heads)
        nq = int(num_queries)
        assert c > 0, arrival_in_channels
        assert t in _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS, (
            t,
            _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS,
        )
        assert h > 0 and nl >= 1 and nh >= 1 and nq >= 1, (
            hidden_dim,
            n_layers,
            num_heads,
            num_queries,
        )
        assert h % nh == 0, (h, nh)
        self._arrival_in_channels = c
        self._arrival_temporal_horizon = t
        self._hidden = h
        self._num_queries = nq
        self._arrival_token_encoder = _orbit_mlp(
            c,
            2 * h,
            h,
            n_layers=nl,
            activation=activation,
        )
        self._arrival_attention = nn.MultiheadAttention(
            h,
            nh,
            batch_first=True,
        )
        self._time_embedding = nn.Embedding(t, h)
        self._query_embedding = nn.Embedding(nq, h)
        self._out_mlp = _orbit_mlp(nq * h, h, h, n_layers=2, activation=activation)

    def forward(
        self,
        query: torch.Tensor,
        arrival_input: torch.Tensor,
        player_valid_mask: torch.Tensor,
        player_identity_emb: torch.Tensor | None = None,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> torch.Tensor:
        assert query.ndim == 3, (query.ndim, tuple(query.shape))
        assert arrival_input.ndim == 5, (arrival_input.ndim, tuple(arrival_input.shape))
        device = arrival_input.device
        b, n, h = query.shape
        assert h == self._hidden, (h, self._hidden)
        b_a, n_a, t, p, c = arrival_input.shape
        assert (b_a, n_a) == (b, n), (arrival_input.shape, query.shape)
        assert t == self._arrival_temporal_horizon, (t, self._arrival_temporal_horizon)
        assert p == ORBIT_PLAYER_AXIS_SLOTS, (p, ORBIT_PLAYER_AXIS_SLOTS)
        assert c == self._arrival_in_channels, (c, self._arrival_in_channels)
        assert player_valid_mask.shape == (b, ORBIT_PLAYER_AXIS_SLOTS), (
            player_valid_mask.shape,
            b,
            ORBIT_PLAYER_AXIS_SLOTS,
        )
        if player_identity_emb is not None:
            assert player_identity_emb.shape == (b, ORBIT_PLAYER_AXIS_SLOTS, self._hidden), (
                player_identity_emb.shape,
                b,
                ORBIT_PLAYER_AXIS_SLOTS,
                self._hidden,
            )

        valid_players = player_valid_mask.to(device=arrival_input.device, dtype=torch.bool)
        with wall_tree_cuda_model_block(
            wall_profiler,
            device,
            "orbit_arrival_attention_token_encoder",
        ):
            arrival_token = self._arrival_token_encoder(arrival_input)
            assert arrival_token.shape == (b, n, t, ORBIT_PLAYER_AXIS_SLOTS, self._hidden), (
                arrival_token.shape
            )
        with wall_tree_cuda_model_block(
            wall_profiler,
            device,
            "orbit_arrival_attention_time_identity",
        ):
            time_idx = torch.arange(t, device=arrival_input.device, dtype=torch.long)
            time_emb = self._time_embedding(time_idx).to(dtype=arrival_token.dtype)
            assert time_emb.shape == (t, self._hidden), time_emb.shape
            arrival_token = arrival_token + time_emb.view(1, 1, t, 1, self._hidden)
            if player_identity_emb is not None:
                arrival_token = arrival_token + player_identity_emb.to(
                    dtype=arrival_token.dtype
                ).view(
                    b,
                    1,
                    1,
                    ORBIT_PLAYER_AXIS_SLOTS,
                    self._hidden,
                )

        token_valid = valid_players.view(b, 1, 1, ORBIT_PLAYER_AXIS_SLOTS).expand(
            b,
            n,
            t,
            ORBIT_PLAYER_AXIS_SLOTS,
        )
        arrival_token = arrival_token * token_valid.unsqueeze(-1).to(dtype=arrival_token.dtype)
        with wall_tree_cuda_model_block(
            wall_profiler,
            device,
            "orbit_arrival_attention_mha",
        ):
            query_idx = torch.arange(self._num_queries, device=query.device, dtype=torch.long)
            query_emb = self._query_embedding(query_idx).to(dtype=query.dtype)
            assert query_emb.shape == (self._num_queries, self._hidden), query_emb.shape
            query_tokens = query.unsqueeze(2) + query_emb.view(
                1,
                1,
                self._num_queries,
                self._hidden,
            )
            assert query_tokens.shape == (b, n, self._num_queries, self._hidden), (
                query_tokens.shape
            )
            query_flat = query_tokens.reshape(b * n, self._num_queries, self._hidden)
            token_flat = arrival_token.reshape(b * n, t * ORBIT_PLAYER_AXIS_SLOTS, self._hidden)
            key_padding_mask = ~token_valid.reshape(b * n, t * ORBIT_PLAYER_AXIS_SLOTS)
            attn_out = self._arrival_attention(
                query_flat,
                token_flat,
                token_flat,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
            assert attn_out.shape == (b * n, self._num_queries, self._hidden), attn_out.shape
        with wall_tree_cuda_model_block(
            wall_profiler,
            device,
            "orbit_arrival_attention_out_mlp",
        ):
            out = self._out_mlp(attn_out.reshape(b * n, self._num_queries * self._hidden)).reshape(
                b,
                n,
                self._hidden,
            )
        assert out.shape == (b, n, self._hidden), out.shape
        return out


def _append_self_enemy_block_feature_extras(
    x: torch.Tensor,
    extra: torch.Tensor,
    *,
    base_dim: int,
    player_block_dim: int,
    base_extra_dim: int,
    player_extra_dim: int,
) -> torch.Tensor:
    assert x.ndim >= 3, (x.ndim, tuple(x.shape))
    assert extra.shape[:-1] == x.shape[:-1], (extra.shape, x.shape)
    base = int(base_dim)
    block = int(player_block_dim)
    base_extra = int(base_extra_dim)
    block_extra = int(player_extra_dim)
    assert base > 0 and block > 0 and base_extra >= 0, (base, block, base_extra)
    assert block_extra >= 0, block_extra
    assert int(x.shape[-1]) == base + ORBIT_PLAYER_AXIS_SLOTS * block, (
        x.shape,
        base,
        block,
    )
    assert int(extra.shape[-1]) == base_extra + ORBIT_PLAYER_AXIS_SLOTS * block_extra, (
        extra.shape,
        base_extra,
        block_extra,
    )

    x_base = x[..., :base]
    x_blocks = x[..., base:]
    extra_base = extra[..., :base_extra]
    if base_extra == 0 and block_extra == 0:
        return x
    if block_extra == 0:
        out = torch.cat([torch.cat([x_base, extra_base], dim=-1), x_blocks], dim=-1)
        expected_dim = base + base_extra + ORBIT_PLAYER_AXIS_SLOTS * block
        assert out.shape == x.shape[:-1] + (expected_dim,), out.shape
        return out

    x_blocks = x_blocks.reshape(*x.shape[:-1], ORBIT_PLAYER_AXIS_SLOTS, block)
    extra_blocks = extra[..., base_extra:].reshape(
        *x.shape[:-1],
        ORBIT_PLAYER_AXIS_SLOTS,
        block_extra,
    )
    aug_blocks = torch.cat([x_blocks, extra_blocks], dim=-1)
    assert aug_blocks.shape == x.shape[:-1] + (
        ORBIT_PLAYER_AXIS_SLOTS,
        block + block_extra,
    ), aug_blocks.shape
    out = torch.cat(
        [
            torch.cat([x_base, extra_base], dim=-1),
            aug_blocks.reshape(
                *x.shape[:-1],
                ORBIT_PLAYER_AXIS_SLOTS * (block + block_extra),
            ),
        ],
        dim=-1,
    )
    expected_dim = base + base_extra + ORBIT_PLAYER_AXIS_SLOTS * (block + block_extra)
    assert out.shape == x.shape[:-1] + (expected_dim,), out.shape
    return out


def _orbit_owner_player_weights_from_raw_planet_input(
    planet_input_raw: torch.Tensor,
    player_valid_mask: torch.Tensor,
) -> torch.Tensor:
    assert planet_input_raw.ndim == 3, (planet_input_raw.ndim, tuple(planet_input_raw.shape))
    b, n, f = planet_input_raw.shape
    assert f == ORBIT_PLANET_FEATURES, (f, ORBIT_PLANET_FEATURES)
    assert player_valid_mask.shape == (b, ORBIT_PLAYER_AXIS_SLOTS), (
        player_valid_mask.shape,
        b,
        ORBIT_PLAYER_AXIS_SLOTS,
    )
    player_blocks = planet_input_raw[..., ORBIT_PLANET_PLAYER_FEATURE_OFFSET:].reshape(
        b,
        n,
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    )
    owner_signal = player_blocks[..., ORBIT_PLANET_PLAYER_FEATURE_PRODUCTION].abs()
    weights = (owner_signal > 0.0).to(dtype=planet_input_raw.dtype)
    weights = weights * player_valid_mask.to(
        device=planet_input_raw.device,
        dtype=weights.dtype,
    ).view(
        b,
        1,
        ORBIT_PLAYER_AXIS_SLOTS,
    )
    if _ENABLE_MODEL_ASSERT:
        _assert_tensor(
            torch.all(weights.sum(dim=-1) <= 1.0),
            "planet owner features must identify at most one owner player block",
        )
    assert weights.shape == (b, n, ORBIT_PLAYER_AXIS_SLOTS), weights.shape
    return weights


def _orbit_planet_edge_triple_mlp(
    mlp: nn.Sequential,
    planet_emb: torch.Tensor,
    edge_emb: torch.Tensor,
) -> torch.Tensor:
    assert planet_emb.ndim == 3
    b, n, h = planet_emb.shape
    assert edge_emb.shape == (b, n, n, h), (edge_emb.shape, planet_emb.shape)
    first = mlp[0]
    assert isinstance(first, nn.Linear), type(first)
    assert first.weight.shape[1] == 3 * h, (first.weight.shape, h)
    src_weight, dst_weight, edge_weight = first.weight.split(h, dim=1)
    src_proj = F.linear(planet_emb, src_weight, first.bias)
    dst_proj = F.linear(planet_emb, dst_weight)
    edge_proj = F.linear(edge_emb, edge_weight)
    out = src_proj.unsqueeze(2) + dst_proj.unsqueeze(1) + edge_proj
    for layer in mlp[1:]:
        out = layer(out)
    return out


class _OrbitPlanetEdgeContextFusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        use_layer_norm: bool,
        activation: str,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        assert h > 0, (hidden_dim,)
        self._hidden = h
        self._src_projection = nn.Linear(h, h)
        self._dst_projection = nn.Linear(h, h)
        self._edge_projection = nn.Linear(h, h)
        self._out_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        self._norm = nn.LayerNorm(h) if bool(use_layer_norm) else nn.Identity()

    def forward(self, planet_emb: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        assert planet_emb.ndim == 3 and planet_emb.shape[-1] == self._hidden
        b, n, h = planet_emb.shape
        assert edge_emb.shape == (b, n, n, h), (edge_emb.shape, planet_emb.shape)
        src = self._src_projection(planet_emb).unsqueeze(2)
        dst = self._dst_projection(planet_emb).unsqueeze(1)
        edge = self._edge_projection(edge_emb)
        assert src.shape == (b, n, 1, h), src.shape
        assert dst.shape == (b, 1, n, h), dst.shape
        assert edge.shape == (b, n, n, h), edge.shape
        out = src + dst + edge
        assert out.shape == (b, n, n, h), out.shape
        out = self._out_mlp(out)
        assert out.shape == (b, n, n, h), out.shape
        out = self._norm(out)
        assert out.shape == (b, n, n, h), out.shape
        return out


class OrbitPlanetEdgeCrossAttentionBlock(nn.Module):
    """Experimental 3-phase block: planet self-attn, planet<-edges, edge<-endpoint-planets."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        residual_dropout: float,
        ffn_multiplier: int,
        ffn_dropout: float,
        edge_endpoint_update: str,
        allow_self_edges: bool,
        activation: str,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        nh = int(num_heads)
        endpoint_update = str(edge_endpoint_update)
        assert endpoint_update in (
            "mha",
            "mha_speedup",
            "mha_debug",
            "directed_fusion",
            "gated_fusion",
        ), endpoint_update
        assert h > 0 and nh > 0, (h, nh)
        assert h % nh == 0, (h, nh)
        assert 0.0 <= float(residual_dropout) < 1.0, (residual_dropout,)
        assert 0.0 <= float(ffn_dropout) < 1.0, (ffn_dropout,)
        ff_mult = int(ffn_multiplier)
        assert ff_mult >= 1, (ffn_multiplier,)
        ff_h = ff_mult * h
        self._hidden = h
        self._num_heads = nh
        self._edge_endpoint_update = endpoint_update
        self._allow_self_edges = bool(allow_self_edges)
        self._planet_self_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=float(residual_dropout),
            batch_first=True,
        )
        self._planet_from_edges_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=float(residual_dropout),
            batch_first=True,
        )
        self._edge_from_planets_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=float(residual_dropout),
            batch_first=True,
        )
        if endpoint_update in ("mha", "mha_speedup", "mha_debug"):
            self._edge_endpoint_role = nn.Parameter(torch.empty(2, h))
            nn.init.trunc_normal_(self._edge_endpoint_role, std=0.02, a=-0.04, b=0.04)
        self._planet_incident_edge_role = nn.Parameter(torch.empty(2, h))
        nn.init.trunc_normal_(self._planet_incident_edge_role, std=0.02, a=-0.04, b=0.04)
        if endpoint_update == "directed_fusion":
            self._edge_endpoint_edge_projection = nn.Linear(h, h)
            self._edge_endpoint_src_projection = nn.Linear(h, h)
            self._edge_endpoint_dst_projection = nn.Linear(h, h)
            self._edge_endpoint_fusion = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        elif endpoint_update == "gated_fusion":
            self._edge_endpoint_edge_projection = nn.Linear(h, h)
            self._edge_endpoint_src_projection = nn.Linear(h, h)
            self._edge_endpoint_dst_projection = nn.Linear(h, h)
            self._edge_endpoint_gate = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
            self._edge_endpoint_value = nn.Linear(h, h)
            self._edge_endpoint_out = nn.Linear(h, h)
        self._planet_ln_self = nn.LayerNorm(h)
        self._planet_ln_edges = nn.LayerNorm(h)
        self._planet_ln_edge_context = nn.LayerNorm(h)
        self._planet_ln_edge_endpoints = nn.LayerNorm(h)
        self._planet_ln_ffn = nn.LayerNorm(h)
        self._edge_ln_context = nn.LayerNorm(h)
        self._edge_ln_incident = nn.LayerNorm(h)
        self._edge_ln_planets = nn.LayerNorm(h)
        self._edge_ln_ffn = nn.LayerNorm(h)
        self._resid_dropout = nn.Dropout(float(residual_dropout))
        self._edge_context_mlp = _orbit_mlp(3 * h, h, h, n_layers=2, activation=activation)
        self._planet_ffn = nn.Sequential(
            nn.Linear(h, ff_h),
            _orbit_activation_module(activation),
            nn.Dropout(float(ffn_dropout)),
            nn.Linear(ff_h, h),
        )
        self._edge_ffn = nn.Sequential(
            nn.Linear(h, ff_h),
            _orbit_activation_module(activation),
            nn.Dropout(float(ffn_dropout)),
            nn.Linear(ff_h, h),
        )

    def forward(
        self,
        *,
        planet_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        planet_mask: torch.Tensor,
        pair_valid: torch.Tensor,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert planet_emb.ndim == 3 and int(planet_emb.shape[-1]) == self._hidden
        b, n, h = planet_emb.shape
        assert edge_emb.shape == (b, n, n, h), (
            tuple(edge_emb.shape),
            tuple(planet_emb.shape),
        )
        assert planet_mask.shape == (b, n), (
            tuple(planet_mask.shape),
            (b, n),
        )
        assert pair_valid.shape == (b, n, n), (
            tuple(pair_valid.shape),
            (b, n, n),
        )
        device = planet_emb.device
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_masks"):
            planet_mask_bool = planet_mask > 0.5
            pair_valid_bool = pair_valid > 0.5
            eye = torch.eye(n, device=planet_emb.device, dtype=torch.bool).unsqueeze(0)
            if self._allow_self_edges:
                edge_valid = pair_valid_bool
            else:
                edge_valid = pair_valid_bool & (~eye)

            # Phase 1: classic planet self-attention.
            self_attn_mask = eye.squeeze(0)
            if _ENABLE_MODEL_ASSERT:
                expected_pair_valid = planet_mask_bool.unsqueeze(2) & planet_mask_bool.unsqueeze(1)
                if not self._allow_self_edges:
                    expected_pair_valid = expected_pair_valid & (~eye)
                _assert_tensor(
                    torch.all(edge_valid == expected_pair_valid),
                    "edge_valid must match configured planet_mask outer-product",
                )
                valid_planet_keys = (
                    planet_mask_bool.unsqueeze(1).expand(b, n, n)
                    & (~self_attn_mask.unsqueeze(0))
                )
                _assert_tensor(
                    torch.all((~planet_mask_bool) | valid_planet_keys.any(dim=-1)),
                    "Each valid planet must have at least one non-self valid planet for self-attention",
                )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_planet_self_mha"):
            planet_norm = self._planet_ln_self(planet_emb)
            planet_self_out = _mha_sdpa_key_padding_attn_mask(
                self._planet_self_attn,
                planet_norm,
                planet_norm,
                planet_norm,
                key_padding_mask=~planet_mask_bool,
                attn_mask=self_attn_mask,
            )
            planet_after_self = planet_emb + self._resid_dropout(planet_self_out)
            planet_after_self = planet_after_self * planet_mask_bool.unsqueeze(-1).to(
                dtype=planet_after_self.dtype
            )

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_edge_context_incident"):
            # Phase 2: update each planet from outgoing+incoming contextualized edges.
            planet_context = self._planet_ln_edge_context(planet_after_self)
            assert planet_context.shape == (b, n, h), planet_context.shape
            edge_norm_context = self._edge_ln_context(edge_emb)
            assert edge_norm_context.shape == (b, n, n, h), edge_norm_context.shape
            edge_context_delta = _orbit_planet_edge_triple_mlp(
                self._edge_context_mlp,
                planet_context,
                edge_norm_context,
            )
            assert edge_context_delta.shape == (b, n, n, h), edge_context_delta.shape
            edge_ctx = edge_emb + self._resid_dropout(edge_context_delta)
            assert edge_ctx.shape == (b, n, n, h), edge_ctx.shape
            edge_ctx = edge_ctx * edge_valid.unsqueeze(-1).to(dtype=edge_ctx.dtype)
            incident_edge_role = self._planet_incident_edge_role.to(
                device=edge_ctx.device,
                dtype=edge_ctx.dtype,
            )
            assert incident_edge_role.shape == (2, h), incident_edge_role.shape
            incident_edge_ctx = self._edge_ln_incident(edge_ctx)
            assert incident_edge_ctx.shape == (b, n, n, h), incident_edge_ctx.shape
            outgoing = incident_edge_ctx + incident_edge_role[0].view(1, 1, 1, h)
            incoming = incident_edge_ctx.transpose(1, 2) + incident_edge_role[1].view(1, 1, 1, h)
            incident_edges = torch.cat([outgoing, incoming], dim=2)
            incident_valid = torch.cat(
                [edge_valid, edge_valid.transpose(1, 2)], dim=2
            )
            assert incident_edges.shape == (b, n, 2 * n, h), incident_edges.shape
            flat_planet_valid = planet_mask_bool.reshape(b * n)
            flat_incident_valid = incident_valid.reshape(b * n, 2 * n)
            if _ENABLE_MODEL_ASSERT:
                _assert_tensor(
                    torch.any(flat_planet_valid),
                    "Expected at least one valid planet token",
                )
                _assert_tensor(
                    torch.all((~flat_planet_valid) | flat_incident_valid.any(dim=-1)),
                    "Each valid planet must have at least one valid incident edge",
                )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_planet_from_edges_mha"):
            flat_planet_q = self._planet_ln_edges(planet_after_self).reshape(b * n, 1, h)
            flat_edge_kv = incident_edges.reshape(b * n, 2 * n, h)
            first_incident_key = torch.arange(2 * n, device=planet_emb.device) == 0
            edge_padding = torch.where(
                flat_planet_valid.unsqueeze(-1),
                ~flat_incident_valid,
                ~first_incident_key.view(1, 2 * n),
            )
            valid_planet_from_edges = _mha_sdpa_key_padding(
                self._planet_from_edges_attn,
                flat_planet_q,
                flat_edge_kv,
                flat_edge_kv,
                key_padding_mask=edge_padding,
            )
            planet_from_edges_flat = valid_planet_from_edges.squeeze(1).to(dtype=planet_emb.dtype)
            planet_from_edges_flat = planet_from_edges_flat * flat_planet_valid.unsqueeze(-1).to(
                dtype=planet_from_edges_flat.dtype,
            )
            planet_from_edges_out = planet_from_edges_flat.reshape(b, n, h)
            planet_after_edges = planet_after_self + self._resid_dropout(
                planet_from_edges_out
            )
            planet_after_edges = planet_after_edges * planet_mask_bool.unsqueeze(-1).to(
                dtype=planet_after_edges.dtype
            )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_planet_ffn"):
            planet_ffn_out = self._planet_ffn(self._planet_ln_ffn(planet_after_edges))
            planet_out = planet_after_edges + self._resid_dropout(planet_ffn_out)
            planet_out = planet_out * planet_mask_bool.unsqueeze(-1).to(dtype=planet_out.dtype)

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_edge_endpoints_flat"):
            # Phase 3: update each directed edge from its two endpoint planets.
            planet_endpoint = self._planet_ln_edge_endpoints(planet_out)
            assert planet_endpoint.shape == (b, n, h), planet_endpoint.shape
            src = planet_endpoint.unsqueeze(2).expand(b, n, n, h)
            dst = planet_endpoint.unsqueeze(1).expand(b, n, n, h)
            flat_edge_valid = edge_valid.reshape(b * n * n)
            if _ENABLE_MODEL_ASSERT:
                _assert_tensor(
                    torch.any(flat_edge_valid),
                    "Expected at least one valid directed edge",
                )
            edge_norm = self._edge_ln_planets(edge_ctx)
            assert edge_norm.shape == (b, n, n, h), edge_norm.shape
            if self._edge_endpoint_update in ("mha", "mha_speedup", "mha_debug"):
                flat_edge_q = edge_norm.reshape(b * n * n, 1, h)
                edge_count = int(flat_edge_q.shape[0])
                endpoint_role = self._edge_endpoint_role.to(
                    device=planet_out.device,
                    dtype=planet_out.dtype,
                )
                assert endpoint_role.shape == (2, h), endpoint_role.shape
                if self._edge_endpoint_update in ("mha", "mha_debug"):
                    src_endpoint = src + endpoint_role[0].view(1, 1, 1, h)
                    dst_endpoint = dst + endpoint_role[1].view(1, 1, 1, h)
                    edge_endpoints = torch.stack([src_endpoint, dst_endpoint], dim=3)
                    assert edge_endpoints.shape == (b, n, n, 2, h), edge_endpoints.shape
                    flat_endpoint_kv = edge_endpoints.reshape(b * n * n, 2, h)
        if self._edge_endpoint_update in ("mha", "mha_speedup", "mha_debug"):
            with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_edge_from_planets_mha"):
                if self._edge_endpoint_update in ("mha", "mha_debug"):
                    if self._edge_endpoint_update == "mha_debug":
                        assert (
                            not bool(self._edge_from_planets_attn.training)
                            or float(self._edge_from_planets_attn.dropout) == 0.0
                        ), "mha_debug requires edge attention dropout disabled while training"
                    # CUDA SDPA fast backends have a hard batch cap around 2^16-1.
                    # MHA internally expands over heads, so keep (chunk_size * num_heads) <= 65535.
                    edge_attn_chunk_cap = max(1, int(65535 // self._num_heads))
                    edge_attn_chunk_size = min(edge_count, edge_attn_chunk_cap)
                    assert edge_attn_chunk_size > 0
                    slow_edge_from_planets = torch.empty_like(flat_edge_q)
                    for chunk_start in range(0, edge_count, edge_attn_chunk_size):
                        chunk_end = min(chunk_start + edge_attn_chunk_size, edge_count)
                        chunk_q = flat_edge_q[chunk_start:chunk_end]
                        chunk_kv = flat_endpoint_kv[chunk_start:chunk_end]
                        chunk_out = _mha_sdpa_nomask(
                            self._edge_from_planets_attn,
                            chunk_q,
                            chunk_kv,
                            chunk_kv,
                        )
                        slow_edge_from_planets[chunk_start:chunk_end] = chunk_out
                if self._edge_endpoint_update == "mha":
                    valid_edge_from_planets = slow_edge_from_planets
                else:
                    edge_mha = self._edge_from_planets_attn
                    assert bool(edge_mha.batch_first), "edge endpoint MHA expects batch_first=True"
                    assert bool(edge_mha._qkv_same_embed_dim), (
                        "edge endpoint MHA expects shared q/k/v"
                    )
                    assert edge_mha.bias_k is None
                    assert edge_mha.bias_v is None
                    assert not bool(edge_mha.add_zero_attn)
                    assert int(edge_mha.embed_dim) == h
                    nh = int(edge_mha.num_heads)
                    assert nh == self._num_heads, (nh, self._num_heads)
                    assert h % nh == 0, (h, nh)
                    d = h // nh
                    in_proj_weight = edge_mha.in_proj_weight
                    in_proj_bias = edge_mha.in_proj_bias
                    assert isinstance(in_proj_weight, torch.Tensor)
                    assert isinstance(in_proj_bias, torch.Tensor)
                    assert in_proj_weight.shape == (3 * h, h)
                    assert in_proj_bias.shape == (3 * h,)
                    q_proj = F.linear(flat_edge_q, in_proj_weight[:h], in_proj_bias[:h])
                    planet_k_proj = F.linear(
                        planet_endpoint,
                        in_proj_weight[h : 2 * h],
                        in_proj_bias[h : 2 * h],
                    )
                    planet_v_proj = F.linear(
                        planet_endpoint,
                        in_proj_weight[2 * h :],
                        in_proj_bias[2 * h :],
                    )
                    endpoint_k_role = F.linear(endpoint_role, in_proj_weight[h : 2 * h])
                    endpoint_v_role = F.linear(endpoint_role, in_proj_weight[2 * h :])
                    assert q_proj.shape == (edge_count, 1, h), q_proj.shape
                    assert planet_k_proj.shape == (b, n, h), planet_k_proj.shape
                    assert planet_v_proj.shape == (b, n, h), planet_v_proj.shape
                    assert endpoint_k_role.shape == (2, h), endpoint_k_role.shape
                    assert endpoint_v_role.shape == (2, h), endpoint_v_role.shape
                    k_proj_stacked = torch.empty(
                        (b, n, n, 2, h),
                        device=planet_k_proj.device,
                        dtype=planet_k_proj.dtype,
                    )
                    v_proj_stacked = torch.empty(
                        (b, n, n, 2, h),
                        device=planet_v_proj.device,
                        dtype=planet_v_proj.dtype,
                    )
                    k_proj_stacked[:, :, :, 0, :] = (
                        planet_k_proj.unsqueeze(2) + endpoint_k_role[0].view(1, 1, 1, h)
                    )
                    k_proj_stacked[:, :, :, 1, :] = (
                        planet_k_proj.unsqueeze(1) + endpoint_k_role[1].view(1, 1, 1, h)
                    )
                    v_proj_stacked[:, :, :, 0, :] = (
                        planet_v_proj.unsqueeze(2) + endpoint_v_role[0].view(1, 1, 1, h)
                    )
                    v_proj_stacked[:, :, :, 1, :] = (
                        planet_v_proj.unsqueeze(1) + endpoint_v_role[1].view(1, 1, 1, h)
                    )
                    k_proj = k_proj_stacked.reshape(edge_count, 2, h)
                    v_proj = v_proj_stacked.reshape(edge_count, 2, h)
                    assert k_proj.shape == (edge_count, 2, h), k_proj.shape
                    assert v_proj.shape == (edge_count, 2, h), v_proj.shape
                    q = q_proj.view(edge_count, 1, nh, d).transpose(1, 2)
                    k = k_proj.view(edge_count, 2, nh, d).transpose(1, 2)
                    v = v_proj.view(edge_count, 2, nh, d).transpose(1, 2)
                    assert q.shape == (edge_count, nh, 1, d), q.shape
                    assert k.shape == (edge_count, nh, 2, d), k.shape
                    assert v.shape == (edge_count, nh, 2, d), v.shape
                    with torch.amp.autocast(device_type=planet_emb.device.type, enabled=False):
                        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (
                            float(d) ** -0.5
                        )
                        assert logits.shape == (edge_count, nh, 1, 2), logits.shape
                        weights = torch.softmax(logits, dim=-1)
                        assert weights.shape == (edge_count, nh, 1, 2), weights.shape
                        if bool(edge_mha.training):
                            weights = F.dropout(weights, p=float(edge_mha.dropout), training=True)
                        assert weights.shape == (edge_count, nh, 1, 2), weights.shape
                        attn = torch.matmul(weights, v.float())
                        assert attn.shape == (edge_count, nh, 1, d), attn.shape
                        fast_edge_from_planets = _mha_sdpa_output(edge_mha, attn, edge_count, 1)
                    assert fast_edge_from_planets.shape == (edge_count, 1, h), (
                        fast_edge_from_planets.shape,
                        (edge_count, 1, h),
                    )
                    if self._edge_endpoint_update == "mha_debug":
                        assert fast_edge_from_planets.shape == slow_edge_from_planets.shape, (
                            fast_edge_from_planets.shape,
                            slow_edge_from_planets.shape,
                        )
                        slow_q, slow_k, slow_v = _mha_sdpa_project(
                            edge_mha,
                            flat_edge_q,
                            flat_endpoint_kv,
                            flat_endpoint_kv,
                        )
                        assert slow_q.shape == q.shape, (slow_q.shape, q.shape)
                        assert slow_k.shape == k.shape, (slow_k.shape, k.shape)
                        assert slow_v.shape == v.shape, (slow_v.shape, v.shape)
                        slow_attn = torch.empty_like(attn)
                        for chunk_start in range(0, edge_count, edge_attn_chunk_size):
                            chunk_end = min(chunk_start + edge_attn_chunk_size, edge_count)
                            slow_attn[chunk_start:chunk_end] = F.scaled_dot_product_attention(
                                slow_q[chunk_start:chunk_end],
                                slow_k[chunk_start:chunk_end],
                                slow_v[chunk_start:chunk_end],
                                dropout_p=0.0,
                                is_causal=False,
                            )
                        assert slow_attn.shape == attn.shape, (slow_attn.shape, attn.shape)
                        q_diff = (q.float() - slow_q.float()).abs()
                        k_diff = (k.float() - slow_k.float()).abs()
                        v_diff = (v.float() - slow_v.float()).abs()
                        attn_diff = (attn.float() - slow_attn.float()).abs()
                        final_diff = (
                            fast_edge_from_planets.float() - slow_edge_from_planets.float()
                        ).abs()
                        flat_final_diff = final_diff.reshape(-1)
                        max_final_idx = flat_final_diff.argmax()
                        max_final_idx_int = int(max_final_idx.detach().cpu().item())
                        max_edge = max_final_idx_int // h
                        max_hidden = max_final_idx_int % h
                        max_b = max_edge // (n * n)
                        max_pair = max_edge % (n * n)
                        max_src = max_pair // n
                        max_dst = max_pair % n
                        print(
                            "mha_debug edge_from_planets diff "
                            f"shape={tuple(fast_edge_from_planets.shape)} "
                            f"fast_dtype={fast_edge_from_planets.dtype} "
                            f"slow_dtype={slow_edge_from_planets.dtype} "
                            f"q_max={float(q_diff.max().detach().cpu().item()):.9g} "
                            f"q_mean={float(q_diff.mean().detach().cpu().item()):.9g} "
                            f"k_max={float(k_diff.max().detach().cpu().item()):.9g} "
                            f"k_mean={float(k_diff.mean().detach().cpu().item()):.9g} "
                            f"v_max={float(v_diff.max().detach().cpu().item()):.9g} "
                            f"v_mean={float(v_diff.mean().detach().cpu().item()):.9g} "
                            f"attn_max={float(attn_diff.max().detach().cpu().item()):.9g} "
                            f"attn_mean={float(attn_diff.mean().detach().cpu().item()):.9g} "
                            f"final_max={float(final_diff.max().detach().cpu().item()):.9g} "
                            f"final_mean={float(final_diff.mean().detach().cpu().item()):.9g} "
                            f"max_at=(b={max_b},src={max_src},dst={max_dst},h={max_hidden}) "
                            f"fast={float(fast_edge_from_planets.reshape(-1)[max_final_idx_int].detach().cpu().item()):.9g} "
                            f"slow={float(slow_edge_from_planets.reshape(-1)[max_final_idx_int].detach().cpu().item()):.9g}",
                            flush=True,
                        )
                        if _ENABLE_MODEL_ASSERT:
                            close = torch.isclose(
                                fast_edge_from_planets.float(),
                                slow_edge_from_planets.float(),
                                rtol=1e-4,
                                atol=1e-4,
                            ).all()
                            _assert_tensor(close, "mha_speedup output must match slow edge MHA")
                    valid_edge_from_planets = fast_edge_from_planets
        elif self._edge_endpoint_update == "directed_fusion":
            assert self._edge_endpoint_update == "directed_fusion", self._edge_endpoint_update
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_attn_edge_from_planets_directed_fusion"
            ):
                edge_endpoint_fused = (
                    self._edge_endpoint_edge_projection(edge_norm)
                    + self._edge_endpoint_src_projection(src)
                    + self._edge_endpoint_dst_projection(dst)
                )
                assert edge_endpoint_fused.shape == (b, n, n, h), edge_endpoint_fused.shape
                valid_edge_from_planets = self._edge_endpoint_fusion(edge_endpoint_fused)
                assert valid_edge_from_planets.shape == (b, n, n, h), (
                    valid_edge_from_planets.shape,
                    (b, n, n, h),
                )
        else:
            assert self._edge_endpoint_update == "gated_fusion", self._edge_endpoint_update
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_attn_edge_from_planets_gated_fusion"
            ):
                edge_endpoint_fused = (
                    self._edge_endpoint_edge_projection(edge_norm)
                    + self._edge_endpoint_src_projection(src)
                    + self._edge_endpoint_dst_projection(dst)
                )
                assert edge_endpoint_fused.shape == (b, n, n, h), edge_endpoint_fused.shape
                endpoint_gate = torch.sigmoid(self._edge_endpoint_gate(edge_endpoint_fused))
                assert endpoint_gate.shape == (b, n, n, h), endpoint_gate.shape
                src_value = self._edge_endpoint_value(src)
                dst_value = self._edge_endpoint_value(dst)
                assert src_value.shape == (b, n, n, h), src_value.shape
                assert dst_value.shape == (b, n, n, h), dst_value.shape
                endpoint_mix = endpoint_gate * src_value + (1.0 - endpoint_gate) * dst_value
                assert endpoint_mix.shape == (b, n, n, h), endpoint_mix.shape
                valid_edge_from_planets = self._edge_endpoint_out(endpoint_mix)
                assert valid_edge_from_planets.shape == (b, n, n, h), (
                    valid_edge_from_planets.shape,
                    (b, n, n, h),
                )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_attn_edge_ffn"):
            if self._edge_endpoint_update in ("mha", "mha_speedup", "mha_debug"):
                edge_from_planets_flat = valid_edge_from_planets.squeeze(1).to(
                    dtype=planet_emb.dtype
                )
            else:
                edge_from_planets_flat = valid_edge_from_planets.reshape(b * n * n, h).to(
                    dtype=planet_emb.dtype
                )
            edge_from_planets_flat = edge_from_planets_flat * flat_edge_valid.unsqueeze(-1).to(
                dtype=edge_from_planets_flat.dtype,
            )
            edge_from_planets_out = edge_from_planets_flat.reshape(b, n, n, h)
            edge_after_planets = edge_ctx + self._resid_dropout(
                edge_from_planets_out
            )
            edge_after_planets = edge_after_planets * edge_valid.unsqueeze(-1).to(
                dtype=edge_after_planets.dtype
            )
            edge_ffn_out = self._edge_ffn(self._edge_ln_ffn(edge_after_planets))
            edge_out = edge_after_planets + self._resid_dropout(edge_ffn_out)
            edge_out = edge_out * edge_valid.unsqueeze(-1).to(dtype=edge_out.dtype)
        return planet_out, edge_out


class OrbitGlobalValuePoolHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        decoder_num_layers: int,
        pool_num_heads: int,
        pool_num_queries: int,
        pool_use_layer_norm: bool,
        activation: str,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        assert h > 0, (hidden_dim,)
        nh = int(pool_num_heads)
        assert nh > 0, (pool_num_heads,)
        nq = int(pool_num_queries)
        assert nq > 0, (pool_num_queries,)
        self._hidden = h
        self._value_context_dim = h
        self._triple_context_fusion = _OrbitPlanetEdgeContextFusion(
            hidden_dim=h,
            use_layer_norm=bool(pool_use_layer_norm),
            activation=activation,
        )
        self._value_score_mlp = _orbit_mlp(h, 3 * h, 1, n_layers=3, activation=activation)
        self._value_pool = _OrbitScoredSetPool(
            context_dim=h,
            hidden_dim=3 * h,
            num_queries=nq,
            activation=activation,
        )
        self._value_decoder = _orbit_mlp(
            self._value_pool.output_dim,
            3 * h,
            1,
            n_layers=decoder_num_layers,
            activation=activation,
        )

    def forward(
        self,
        planet_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        pair_valid: torch.Tensor,
        planet_slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert planet_emb.ndim == 3 and planet_emb.shape[-1] == self._hidden
        assert edge_emb.shape == planet_emb.shape[:2] + (planet_emb.shape[1], self._hidden)
        assert pair_valid.shape == edge_emb.shape[:3]
        assert planet_slot_mask.shape == planet_emb.shape[:2]
        device, dtype = planet_emb.device, planet_emb.dtype

        triple_h = self._triple_context_fusion(planet_emb, edge_emb)
        assert triple_h.shape == planet_emb.shape[:2] + (
            planet_emb.shape[1],
            self._value_context_dim,
        )
        value_scores = self._value_score_mlp(triple_h).squeeze(-1)
        assert value_scores.shape == pair_valid.shape
        b, n, _, c = triple_h.shape
        assert c == self._value_context_dim
        triple_tokens = triple_h.reshape(b, n * n, c)
        value_scores_flat = value_scores.reshape(b, n * n)
        triple_valid = (pair_valid > 0.5).reshape(b, n * n)

        pooled_features = self._value_pool(
            value_scores_flat,
            triple_tokens,
            triple_valid,
        )
        assert pooled_features.shape == (planet_emb.shape[0], self._value_pool.output_dim)
        baseline = self._value_decoder(pooled_features).squeeze(-1)
        assert baseline.shape == (planet_emb.shape[0],)
        return baseline


class _OrbitScoredSetPool(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        hidden_dim: int,
        num_queries: int,
        activation: str,
    ) -> None:
        super().__init__()
        c = int(context_dim)
        h = int(hidden_dim)
        nq = int(num_queries)
        assert c > 0, (context_dim,)
        assert h > 0, (hidden_dim,)
        assert nq > 0, (num_queries,)
        self._context_dim = c
        self._num_queries = nq
        self.output_dim = c * (1 + nq) + 4
        self._context_norm = nn.LayerNorm(c)
        self._attention_logits_mlp = _orbit_mlp(c, h, 1, n_layers=2, activation=activation)
        self._query_key_mlp = _orbit_mlp(c, h, c, n_layers=2, activation=activation)
        self._query_score_bias_mlp = _orbit_mlp(1, h, nq, n_layers=2, activation=activation)
        self._queries = nn.Parameter(torch.empty(nq, c))
        nn.init.trunc_normal_(self._queries, std=0.02, a=-0.04, b=0.04)

    def forward(
        self,
        candidate_scores: torch.Tensor,
        candidate_context: torch.Tensor,
        candidate_mask: torch.Tensor,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> torch.Tensor:
        assert candidate_scores.ndim >= 2, tuple(candidate_scores.shape)
        assert torch.is_floating_point(candidate_scores), candidate_scores.dtype
        assert candidate_context.shape == candidate_scores.shape + (self._context_dim,), (
            tuple(candidate_context.shape),
            tuple(candidate_scores.shape),
            self._context_dim,
        )
        assert torch.is_floating_point(candidate_context), candidate_context.dtype
        assert candidate_mask.shape == candidate_scores.shape, (
            tuple(candidate_mask.shape),
            tuple(candidate_scores.shape),
        )
        assert candidate_mask.dtype == torch.bool, candidate_mask.dtype
        device = candidate_scores.device
        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_context_norm"):
            candidate_context = self._context_norm(candidate_context)
            active = candidate_mask.any(dim=-1)
        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_logits"):
            attention_logits = self._attention_logits_mlp(candidate_context).squeeze(-1)
            assert attention_logits.shape == candidate_scores.shape, attention_logits.shape
            keys = self._query_key_mlp(candidate_context)
            assert keys.shape == candidate_context.shape
            score_bias = self._query_score_bias_mlp(candidate_scores.unsqueeze(-1))
            assert score_bias.shape == candidate_scores.shape + (self._num_queries,), score_bias.shape
            queries = self._queries.to(device=candidate_context.device, dtype=candidate_context.dtype)
            query_logits = torch.einsum("...kc,qc->...kq", keys, queries) * (self._context_dim**-0.5)
            assert query_logits.shape == candidate_scores.shape + (self._num_queries,), query_logits.shape
            attention_logits = attention_logits + candidate_scores
            query_logits = query_logits + score_bias
        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_score_stats"):
            neg_large = torch.finfo(candidate_scores.dtype).min / 16.0
            masked_scores = candidate_scores.masked_fill(~candidate_mask, neg_large)
            max_score = masked_scores.max(dim=-1).values
            logsumexp_score = torch.logsumexp(masked_scores, dim=-1)
            count = candidate_mask.sum(dim=-1).clamp(min=1).to(dtype=candidate_scores.dtype)
            mean_score = candidate_scores.masked_fill(~candidate_mask, 0.0).sum(dim=-1) / count
            zero_score = torch.zeros_like(max_score)
            max_score = torch.where(active, max_score, zero_score)
            logsumexp_score = torch.where(active, logsumexp_score, zero_score)
            mean_score = torch.where(active, mean_score, zero_score)

        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_attention_softmax"):
            attention_weights = F.softmax(
                attention_logits.masked_fill(~candidate_mask, neg_large),
                dim=-1,
                dtype=torch.float32,
            ).to(dtype=candidate_context.dtype)
            attention_weights = attention_weights * active.unsqueeze(-1).to(dtype=attention_weights.dtype)
        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_attention_pool"):
            pooled_context = (attention_weights.unsqueeze(-1) * candidate_context).sum(dim=-2)
            assert pooled_context.shape == candidate_scores.shape[:-1] + (self._context_dim,)

        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_query_softmax"):
            query_logits = query_logits.masked_fill(~candidate_mask.unsqueeze(-1), neg_large)
            query_weights = F.softmax(
                query_logits.transpose(-2, -1),
                dim=-1,
                dtype=torch.float32,
            ).to(dtype=candidate_context.dtype)
            assert query_weights.shape == candidate_scores.shape[:-1] + (
                self._num_queries,
                candidate_scores.shape[-1],
            )
            query_weights = query_weights * active.unsqueeze(-1).unsqueeze(-1).to(dtype=query_weights.dtype)
        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_query_pool"):
            query_pooled = torch.einsum("...qk,...kc->...qc", query_weights, candidate_context)
            assert query_pooled.shape == candidate_scores.shape[:-1] + (
                self._num_queries,
                self._context_dim,
            )
            query_pooled_flat = query_pooled.reshape(
                *candidate_scores.shape[:-1],
                self._num_queries * self._context_dim,
            )
            count_feature = count / float(candidate_scores.shape[-1])
            count_feature = torch.where(active, count_feature, torch.zeros_like(count_feature))

        with wall_tree_cuda_model_block(wall_profiler, device, "scored_pool_pack"):
            pooled_features = torch.cat(
                [
                    pooled_context,
                    query_pooled_flat,
                    logsumexp_score.unsqueeze(-1).to(dtype=candidate_context.dtype),
                    max_score.unsqueeze(-1).to(dtype=candidate_context.dtype),
                    mean_score.unsqueeze(-1).to(dtype=candidate_context.dtype),
                    count_feature.unsqueeze(-1).to(dtype=candidate_context.dtype),
                ],
                dim=-1,
            )
        assert pooled_features.shape == candidate_scores.shape[:-1] + (self.output_dim,)
        return pooled_features


def _masked_score_stats(
    candidate_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    assert candidate_scores.ndim >= 2, tuple(candidate_scores.shape)
    assert torch.is_floating_point(candidate_scores), candidate_scores.dtype
    assert candidate_mask.shape == candidate_scores.shape, (
        tuple(candidate_mask.shape),
        tuple(candidate_scores.shape),
    )
    assert candidate_mask.dtype == torch.bool, candidate_mask.dtype
    with torch.amp.autocast(device_type=candidate_scores.device.type, enabled=False):
        scores_f = candidate_scores.float()
        neg_large = torch.finfo(scores_f.dtype).min / 16.0
        masked_scores = scores_f.masked_fill(~candidate_mask, neg_large)
        active = candidate_mask.any(dim=-1)
        max_score = masked_scores.max(dim=-1).values
        logsumexp_score = torch.logsumexp(masked_scores, dim=-1)
        count = candidate_mask.sum(dim=-1).clamp(min=1).to(dtype=torch.float32)
        mean_score = scores_f.masked_fill(~candidate_mask, 0.0).sum(dim=-1) / count
        zero_score = torch.zeros_like(max_score)
        max_score = torch.where(active, max_score, zero_score)
        logsumexp_score = torch.where(active, logsumexp_score, zero_score)
        mean_score = torch.where(active, mean_score, zero_score)
        count_feature = count / float(candidate_scores.shape[-1])
        count_feature = torch.where(active, count_feature, torch.zeros_like(count_feature))
        stats = torch.stack(
            (logsumexp_score, max_score, mean_score, count_feature),
            dim=-1,
        )
    assert stats.shape == candidate_scores.shape[:-1] + (4,), stats.shape
    return stats.to(dtype=candidate_scores.dtype)


def _masked_context_score_pool(
    candidate_scores: torch.Tensor,
    candidate_context: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    assert candidate_scores.ndim >= 2, tuple(candidate_scores.shape)
    assert torch.is_floating_point(candidate_scores), candidate_scores.dtype
    assert candidate_context.shape == candidate_scores.shape + (candidate_context.shape[-1],), (
        tuple(candidate_context.shape),
        tuple(candidate_scores.shape),
    )
    assert torch.is_floating_point(candidate_context), candidate_context.dtype
    assert candidate_mask.shape == candidate_scores.shape, (
        tuple(candidate_mask.shape),
        tuple(candidate_scores.shape),
    )
    assert candidate_mask.dtype == torch.bool, candidate_mask.dtype
    stats = _masked_score_stats(candidate_scores, candidate_mask)
    active = candidate_mask.any(dim=-1)
    neg_large = torch.finfo(candidate_scores.dtype).min / 16.0
    weights = F.softmax(
        candidate_scores.masked_fill(~candidate_mask, neg_large),
        dim=-1,
        dtype=torch.float32,
    ).to(dtype=candidate_context.dtype)
    weights = weights * active.unsqueeze(-1).to(dtype=weights.dtype)
    pooled_context = (weights.unsqueeze(-1) * candidate_context).sum(dim=-2)
    assert pooled_context.shape == candidate_context.shape[:-2] + (candidate_context.shape[-1],)
    pooled = torch.cat([pooled_context, stats.to(dtype=candidate_context.dtype)], dim=-1)
    assert pooled.shape == candidate_scores.shape[:-1] + (candidate_context.shape[-1] + 4,)
    return pooled


class OrbitSimpleGlobalValuePoolHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        decoder_num_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        assert h > 0, (hidden_dim,)
        self._hidden = h
        self._planet_norm = nn.LayerNorm(h)
        self._edge_norm = nn.LayerNorm(h)
        self._planet_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        self._planet_score = nn.Linear(h, 1)
        self._edge_src_projection = nn.Linear(h, h)
        self._edge_dst_projection = nn.Linear(h, h)
        self._edge_projection = nn.Linear(h, h)
        self._edge_pair_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        self._edge_score = nn.Linear(h, 1)
        self._value_planet_projection = nn.Linear(h + 4, h)
        self._value_edge_projection = nn.Linear(h + 4, h)
        self._value_fusion_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        self._value_decoder = _orbit_mlp(
            h,
            3 * h,
            1,
            n_layers=decoder_num_layers,
            activation=activation,
        )

    def _edge_pair_context(
        self,
        planet_context: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        assert planet_context.ndim == 3 and planet_context.shape[-1] == self._hidden
        b, n, h = planet_context.shape
        assert edge_emb.shape == (b, n, n, h), (edge_emb.shape, planet_context.shape)
        src = self._edge_src_projection(planet_context).unsqueeze(2)
        dst = self._edge_dst_projection(planet_context).unsqueeze(1)
        edge = self._edge_projection(self._edge_norm(edge_emb))
        assert src.shape == (b, n, 1, h), src.shape
        assert dst.shape == (b, 1, n, h), dst.shape
        assert edge.shape == (b, n, n, h), edge.shape
        pair_context = self._edge_pair_mlp(src + dst + edge)
        assert pair_context.shape == (b, n, n, h), pair_context.shape
        return pair_context

    def forward(
        self,
        planet_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        pair_valid: torch.Tensor,
        planet_slot_mask: torch.Tensor,
        opponent_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert planet_emb.ndim == 3 and planet_emb.shape[-1] == self._hidden
        b, n, h = planet_emb.shape
        assert edge_emb.shape == (b, n, n, h), (edge_emb.shape, planet_emb.shape)
        assert pair_valid.shape == (b, n, n), pair_valid.shape
        assert planet_slot_mask.shape == (b, n), planet_slot_mask.shape
        if opponent_context is not None:
            assert opponent_context.shape == (b, h), opponent_context.shape

        planet_context = self._planet_mlp(self._planet_norm(planet_emb))
        assert planet_context.shape == (b, n, h), planet_context.shape
        planet_scores = self._planet_score(planet_context).squeeze(-1)
        assert planet_scores.shape == (b, n), planet_scores.shape
        planet_features = _masked_context_score_pool(
            planet_scores,
            planet_context,
            planet_slot_mask > 0.5,
        )
        assert planet_features.shape == (b, h + 4), planet_features.shape

        edge_context = self._edge_pair_context(planet_context, edge_emb)
        edge_scores = self._edge_score(edge_context).squeeze(-1)
        assert edge_scores.shape == (b, n, n), edge_scores.shape
        edge_features = _masked_context_score_pool(
            edge_scores.reshape(b, n * n),
            edge_context.reshape(b, n * n, h),
            (pair_valid > 0.5).reshape(b, n * n),
        )
        assert edge_features.shape == (b, h + 4), edge_features.shape

        value_planet = self._value_planet_projection(planet_features)
        value_edge = self._value_edge_projection(edge_features)
        assert value_planet.shape == (b, h), value_planet.shape
        assert value_edge.shape == (b, h), value_edge.shape
        value_fusion_input = value_planet + value_edge
        if opponent_context is not None:
            value_fusion_input = value_fusion_input + opponent_context
        value_features = self._value_fusion_mlp(value_fusion_input)
        assert value_features.shape == (b, h), value_features.shape
        baseline = self._value_decoder(value_features).squeeze(-1)
        assert baseline.shape == (b,), baseline.shape
        return baseline


class OrbitSimpleSpawnFleetPolicyHead(nn.Module):
    """Flat spawn-fleet policy over per-source move classes."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        decoder_num_layers: int,
        num_heads: int,
        num_policy_actions: int,
        entropy_floor_target: tuple[float],
        entropy_floor_max_temperature: float,
        entropy_floor_num_iters: int,
        activation: str,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        assert h > 0, (hidden_dim,)
        nh = int(num_heads)
        assert nh > 0 and h % nh == 0, (hidden_dim, num_heads)
        nb = int(num_policy_actions)
        assert 1 <= nb <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
            num_policy_actions,
            ORBIT_MOVE_CLASSES_PER_TARGET,
        )
        self._hidden = h
        self._nb = nb
        self._src_projection = nn.Linear(h, h)
        self._dst_projection = nn.Linear(h, h)
        self._edge_projection = nn.Linear(h, h)
        self._pair_mlp = _orbit_mlp(h, h, h, n_layers=2, activation=activation)
        self._dst_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=0.0,
            batch_first=True,
        )
        self._dst_attn_ln = nn.LayerNorm(h)
        self._dst_ffn_ln = nn.LayerNorm(h)
        self._dst_ffn = _orbit_mlp(h, 3 * h, h, n_layers=2, activation=activation)
        self._amount_logits = nn.Linear(h, nb)
        assert len(entropy_floor_target) == 1, entropy_floor_target
        self.register_buffer(
            "_entropy_floor_target",
            torch.zeros(1, dtype=torch.float32),
        )
        entropy_floor_target_tensor = torch.tensor(
            [float(v) for v in entropy_floor_target],
            dtype=torch.float32,
        )
        if _ENABLE_MODEL_ASSERT:
            assert bool(
                torch.all(
                    (0.0 <= entropy_floor_target_tensor)
                    & (entropy_floor_target_tensor <= 1.0)
                ).item()
            ), entropy_floor_target
        self._entropy_floor_target.copy_(entropy_floor_target_tensor)
        floor_max_temp = float(entropy_floor_max_temperature)
        assert floor_max_temp >= 1.0, entropy_floor_max_temperature
        floor_iters = int(entropy_floor_num_iters)
        assert floor_iters >= 0, entropy_floor_num_iters
        self._entropy_floor_max_temperature = floor_max_temp
        self._entropy_floor_num_iters = floor_iters
        self._entropy_floor_disabled = bool(
            torch.all(entropy_floor_target_tensor == 0.0).item()
        )

    def set_entropy_floor_targets(self, floor_target: torch.Tensor) -> None:
        assert isinstance(floor_target, torch.Tensor), type(floor_target)
        assert tuple(floor_target.shape) == (1,), tuple(floor_target.shape)
        assert torch.is_floating_point(floor_target), floor_target.dtype
        if _ENABLE_MODEL_ASSERT:
            assert bool(torch.all((0.0 <= floor_target) & (floor_target <= 1.0)).item()), floor_target
        self._entropy_floor_target.copy_(
            floor_target.to(device=self._entropy_floor_target.device, dtype=self._entropy_floor_target.dtype),
        )
        self._entropy_floor_disabled = bool(torch.all(floor_target == 0.0).item())

    def _pair_context(
        self,
        planet_emb: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        assert planet_emb.ndim == 3 and planet_emb.shape[-1] == self._hidden
        b, n, h = planet_emb.shape
        assert edge_emb.shape == (b, n, n, h), (edge_emb.shape, planet_emb.shape)
        src = self._src_projection(planet_emb).unsqueeze(2)
        dst = self._dst_projection(planet_emb).unsqueeze(1)
        edge = self._edge_projection(edge_emb)
        assert src.shape == (b, n, 1, h), src.shape
        assert dst.shape == (b, 1, n, h), dst.shape
        assert edge.shape == (b, n, n, h), edge.shape
        fused = src + dst + edge
        assert fused.shape == (b, n, n, h), fused.shape
        out = self._pair_mlp(fused)
        assert out.shape == (b, n, n, h), out.shape
        return out

    def _dst_attention_context(
        self,
        pair_context: torch.Tensor,
        pair_valid: torch.Tensor,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> torch.Tensor:
        assert pair_context.ndim == 4 and pair_context.shape[-1] == self._hidden
        b, n, n_dst, h = pair_context.shape
        assert n_dst == n, pair_context.shape
        assert pair_valid.shape == (b, n, n), pair_valid.shape
        device = pair_context.device
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_dst_attn_pack"):
            eye = torch.eye(n, device=device, dtype=torch.bool).view(1, n, n)
            dst_valid = (pair_valid > 0.5) | eye
            flat_dst_valid = dst_valid.reshape(b * n, n)
            flat_pair_context = pair_context.reshape(b * n, n, h)
            assert flat_pair_context.shape == (b * n, n, h), flat_pair_context.shape
            assert flat_dst_valid.shape == (b * n, n), flat_dst_valid.shape
            if _ENABLE_MODEL_ASSERT:
                _assert_tensor(
                    torch.all(flat_dst_valid.any(dim=-1)),
                    "Each policy source must have at least one destination token for dst attention",
                )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_dst_attn_mha"):
            flat_dst_norm = self._dst_attn_ln(flat_pair_context)
            flat_dst_delta = _mha_sdpa_key_padding(
                self._dst_attn,
                flat_dst_norm,
                flat_dst_norm,
                flat_dst_norm,
                key_padding_mask=~flat_dst_valid,
            )
            assert flat_dst_delta.shape == (b * n, n, h), flat_dst_delta.shape
            flat_pair_context = flat_pair_context + flat_dst_delta
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_dst_attn_ffn"):
            flat_dst_ffn = self._dst_ffn(self._dst_ffn_ln(flat_pair_context))
            assert flat_dst_ffn.shape == (b * n, n, h), flat_dst_ffn.shape
            flat_pair_context = flat_pair_context + flat_dst_ffn
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_dst_attn_unpack"):
            out = flat_pair_context.reshape(b, n, n, h)
            assert out.shape == pair_context.shape, (out.shape, pair_context.shape)
        return out

    def forward(
        self,
        planet_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        pair_valid: torch.Tensor,
        available_action_flat: torch.Tensor,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert planet_emb.ndim == 3 and planet_emb.shape[-1] == self._hidden
        n = int(planet_emb.shape[1])
        assert n == ORBIT_MAX_PLANETS
        assert edge_emb.shape == (planet_emb.shape[0], n, n, self._hidden)
        assert pair_valid.shape == (planet_emb.shape[0], n, n)
        assert available_action_flat.shape == (
            planet_emb.shape[0],
            n,
            n * self._nb,
        )
        assert available_action_flat.dtype == torch.bool

        b = int(planet_emb.shape[0])
        device = planet_emb.device
        nb = self._nb
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_action_masks"):
            available_by_dst_sn = available_action_flat.reshape(b, n, n, nb)
            assert available_by_dst_sn.shape == (b, n, n, nb)
            src_idx = torch.arange(n, device=device, dtype=torch.long)
            noop_available = available_by_dst_sn[:, src_idx, src_idx, 0]
            if _ENABLE_MODEL_ASSERT:
                _assert_tensor(
                    torch.all(noop_available),
                    "Each spawn_fleet source slot must expose self noop as an available action",
                )
            sn0_available = available_by_dst_sn[..., 0]
            if _ENABLE_MODEL_ASSERT:
                expected_sn0 = torch.eye(n, device=device, dtype=torch.bool).view(1, n, n).expand(
                    b,
                    n,
                    n,
                )
                _assert_tensor(
                    torch.all(sn0_available == expected_sn0),
                    "Only source-to-self ship_subindex=0 may be available for spawn_fleet",
                )
            if _ENABLE_MODEL_ASSERT:
                pair_valid_bool = pair_valid > 0.5
                non_noop_available = available_by_dst_sn[..., 1:].any(dim=-1)
                _assert_tensor(
                    torch.all(non_noop_available == (non_noop_available & pair_valid_bool)),
                    "Available non-noop destinations must be valid directed planet pairs",
                )

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_simple_pair_context"):
            pair_context = self._pair_context(planet_emb, edge_emb)
            assert pair_context.shape == (b, n, n, self._hidden)
        pair_context = self._dst_attention_context(
            pair_context,
            pair_valid,
            wall_profiler=wall_profiler,
        )
        assert pair_context.shape == (b, n, n, self._hidden)
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_action_logits"):
            action_logits_by_dst_class = self._amount_logits(pair_context)
            assert action_logits_by_dst_class.shape == (b, n, n, nb)
            policy_logits = action_logits_by_dst_class.reshape(
                b,
                n,
                n * nb,
            )
            assert policy_logits.shape == (b, n, n * nb)
            if _ENABLE_MODEL_ASSERT:
                _assert_tensor(
                    torch.all(torch.isfinite(policy_logits[available_action_flat])),
                    "Available spawn_fleet actions must have finite final logit",
                )
        entropy_floor_stats = {}
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_final_log_softmax"):
            with torch.amp.autocast(device_type=policy_logits.device.type, enabled=False):
                policy_logits_f = policy_logits.float()
                neg_large = torch.finfo(policy_logits_f.dtype).min / 16.0
                policy_log_probs = F.log_softmax(
                    policy_logits_f.masked_fill(~available_action_flat, neg_large),
                    dim=-1,
                ).masked_fill(
                    ~available_action_flat,
                    float("-inf"),
                )
            assert policy_log_probs.shape == (b, n, n * nb)
            if _ENABLE_MODEL_ASSERT:
                _assert_tensor(
                    torch.all(torch.isfinite(policy_log_probs[available_action_flat])),
                    "Available spawn_fleet actions must have finite final log-probability",
                )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_policy_output_pack"):
            policy_head_aux = {
                "action_logits": action_logits_by_dst_class,
                "action_mask": available_action_flat,
                "policy_logits": policy_logits,
                "entropy_floor_stats": entropy_floor_stats,
            }
        return policy_log_probs, policy_head_aux


class ImpalaOrbitModel(nn.Module):
    _PLANET_IN_DIM = ORBIT_PLANET_FEATURES
    _ARRIVAL_IN_DIM = ORBIT_PLANET_ARRIVAL_FEATURES
    _ARRIVAL_CNN_IN_CHANNELS = ORBIT_PLANET_TEMPORAL_FEATURES
    _RAW_EDGE_DIM = ORBIT_EDGE_FEATURES

    def __init__(
        self,
        *,
        hidden_dim: int,
        use_token_batch_norm: bool,
        obs_feature_normalization: dict[str, Any],
        encoder_num_layers: int,
        arrivals_encoder_num_layers: int,
        arrivals_fusion_num_layers: int,
        decoder_num_layers: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        edge_endpoint_update: str,
        mlp_activation: str,
        transformer_activation: str,
        residual_dropout: float,
        ffn_multiplier: int,
        ffn_dropout: float,
        use_obs_discrete_embeddings: bool,
        entropy_floor_target: tuple[float],
        entropy_floor_max_temperature: float,
        entropy_floor_num_iters: int,
        num_policy_actions: int = 2,
        arrival_temporal_horizon: int = _ORBIT_MODEL_DEFAULT_TEMPORAL_HORIZON,
        zeroed_obs_feature_inputs: tuple[str, ...] = (),
        use_identity_embedding: bool = False,
        use_edge_relation_features: bool = False,
        use_arrival_attention_fusion: bool = False,
        arrival_attention_num_queries: int = 4,
        use_arrival_attention_fusion_gate: bool = False,
        allow_self_edges: bool = False,
        policy_head_fp32: bool = False,
        value_head_fp32: bool = False,
        use_value_opponent_model_embedding: bool = False,
        value_opponent_model_count: int = 0,
        include_rl_policy_value_heads: bool = True,
        include_rl_value_head: bool = True,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        assert h > 0, (hidden_dim,)
        self._hidden = h
        self._include_rl_policy_value_heads = bool(include_rl_policy_value_heads)
        self._include_rl_value_head = bool(include_rl_value_head)
        self._is_sample = True
        self._compile_friendly_sample = False
        self._shuffle_identity_ids = True
        self._arrival_temporal_horizon = int(arrival_temporal_horizon)
        assert self._arrival_temporal_horizon in _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS, (
            self._arrival_temporal_horizon,
            _ORBIT_MODEL_SUPPORTED_TEMPORAL_HORIZONS,
        )
        assert self._arrival_temporal_horizon <= ORBIT_PLANET_ARRIVAL_HORIZON, (
            self._arrival_temporal_horizon,
            ORBIT_PLANET_ARRIVAL_HORIZON,
        )
        temporal_step_max_index = _orbit_model_temporal_step_max_index(
            self._arrival_temporal_horizon
        )
        assert not (
            self._include_rl_value_head and not self._include_rl_policy_value_heads
        ), "include_rl_value_head requires include_rl_policy_value_heads"
        self._use_obs_discrete_embeddings = bool(use_obs_discrete_embeddings)
        self._use_identity_embedding = bool(use_identity_embedding)
        self._use_edge_relation_features = bool(use_edge_relation_features)
        self._use_arrival_attention_fusion = bool(use_arrival_attention_fusion)
        self._use_arrival_attention_fusion_gate = bool(use_arrival_attention_fusion_gate)
        self._policy_head_fp32 = bool(policy_head_fp32)
        self._value_head_fp32 = bool(value_head_fp32)
        self._use_value_opponent_model_embedding = bool(use_value_opponent_model_embedding)
        self._value_opponent_model_count = int(value_opponent_model_count)
        if self._use_value_opponent_model_embedding:
            assert self._use_identity_embedding, (
                "use_value_opponent_model_embedding requires use_identity_embedding"
            )
            assert self._value_opponent_model_count > 0, self._value_opponent_model_count
        else:
            assert self._value_opponent_model_count == 0, self._value_opponent_model_count
        self._num_policy_actions = int(num_policy_actions)
        assert 1 <= self._num_policy_actions <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
            num_policy_actions,
            ORBIT_MOVE_CLASSES_PER_TARGET,
        )
        mlp_activation_name = str(mlp_activation)
        transformer_activation_name = str(transformer_activation)
        _orbit_activation_module(mlp_activation_name)
        _orbit_activation_module(transformer_activation_name)
        enc_nl = int(encoder_num_layers)
        arr_enc_nl = int(arrivals_encoder_num_layers)
        arr_fuse_nl = int(arrivals_fusion_num_layers)
        dec_nl = int(decoder_num_layers)
        assert enc_nl >= 1, (encoder_num_layers,)
        assert arr_enc_nl >= 1, (arrivals_encoder_num_layers,)
        assert arr_fuse_nl >= 1, (arrivals_fusion_num_layers,)
        assert arr_fuse_nl == 2, (arrivals_fusion_num_layers,)
        assert dec_nl >= 1, (decoder_num_layers,)
        nl = int(transformer_num_layers)
        nh_t = int(transformer_num_heads)
        arrival_attn_nq = int(arrival_attention_num_queries)
        assert arrival_attn_nq >= 1, arrival_attention_num_queries
        endpoint_update = str(edge_endpoint_update)
        assert endpoint_update in (
            "mha",
            "mha_speedup",
            "mha_debug",
            "directed_fusion",
            "gated_fusion",
        ), endpoint_update
        assert nh_t > 0, (transformer_num_heads,)
        assert nl >= 1, (transformer_num_layers,)
        assert h % nh_t == 0, (
            f"hidden_dim must divide transformer_num_heads evenly, got {h=} {nh_t=}"
        )
        assert ORBIT_PER_PLANET_MOVE_CLASSES == ORBIT_MAX_PLANETS * ORBIT_MOVE_CLASSES_PER_TARGET
        norm_config = _cfg_mapping(obs_feature_normalization)
        layout_map = _cfg_mapping(norm_config["layout"])
        feature_config = _cfg_mapping(norm_config["features"])
        zeroed_inputs = _zeroed_obs_feature_inputs_from_config(zeroed_obs_feature_inputs)
        p_idx, p_names, e_idx, e_names = planet_edge_physical_to_logical_from_layout(layout_map)
        p_feature_keys = logical_feature_normalization_keys("planet", p_names)
        e_feature_keys = logical_feature_normalization_keys("edge", e_names)
        arrival_feature_keys = logical_feature_normalization_keys(
            "arrival",
            ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES,
        )
        expected_feature_keys = set(p_feature_keys) | set(e_feature_keys) | set(arrival_feature_keys)
        feature_config = _feature_norm_config_with_disabled_missing(
            feature_config,
            expected_feature_keys,
        )
        self._planet_obs_feature_norm = _ConfiguredLogicalFeatureNorm(
            num_physical_channels=self._PLANET_IN_DIM,
            logical_feature_keys=p_feature_keys,
            physical_to_logical=p_idx,
            feature_config=feature_config,
        )
        self._edge_obs_feature_norm = _ConfiguredLogicalFeatureNorm(
            num_physical_channels=self._RAW_EDGE_DIM,
            logical_feature_keys=e_feature_keys,
            physical_to_logical=e_idx,
            feature_config=feature_config,
        )
        self._arrival_obs_feature_norm = _ConfiguredLogicalFeatureNorm(
            num_physical_channels=self._ARRIVAL_IN_DIM,
            logical_feature_keys=arrival_feature_keys,
            physical_to_logical=torch.arange(self._ARRIVAL_IN_DIM, dtype=torch.long),
            feature_config=feature_config,
        )
        zeroed_planet_continuous_channels = _zeroed_self_enemy_block_channel_mask(
            zeroed_inputs,
            kind="continuous",
            domain="planet",
            logical_feature_keys=p_feature_keys,
            base_feature_count=ORBIT_PLANET_BASE_FEATURES,
            player_feature_offset=ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
            player_features_per_player=ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
            num_physical_channels=self._PLANET_IN_DIM,
        )
        zeroed_edge_continuous_channels = _zeroed_self_enemy_block_channel_mask(
            zeroed_inputs,
            kind="continuous",
            domain="edge",
            logical_feature_keys=e_feature_keys,
            base_feature_count=ORBIT_EDGE_BASE_FEATURES,
            player_feature_offset=ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
            player_features_per_player=ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
            num_physical_channels=self._RAW_EDGE_DIM,
        )
        zeroed_arrival_continuous_slots = _zeroed_arrival_slot_channel_mask(
            zeroed_inputs,
            kind="continuous",
            feature_names=ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES,
        )
        zeroed_arrival_extra_source_channels = (
            self._arrival_obs_feature_norm.extra_source_channels()
        )
        if int(zeroed_arrival_extra_source_channels.numel()) == 0:
            zeroed_arrival_norm_extra_slots = torch.zeros(
                (ORBIT_PLAYER_AXIS_SLOTS, 0),
                dtype=torch.bool,
            )
        else:
            zeroed_arrival_norm_extra_slots = zeroed_arrival_continuous_slots[
                :, zeroed_arrival_extra_source_channels
            ]
        self.register_buffer(
            "_zeroed_planet_continuous_channels",
            zeroed_planet_continuous_channels,
            persistent=False,
        )
        self.register_buffer(
            "_zeroed_arrival_continuous_slots",
            zeroed_arrival_continuous_slots,
            persistent=False,
        )
        self.register_buffer(
            "_zeroed_edge_continuous_channels",
            zeroed_edge_continuous_channels,
            persistent=False,
        )
        self.register_buffer(
            "_zeroed_planet_norm_extra_channels",
            self._planet_obs_feature_norm.extra_zero_mask_for_physical_channels(
                zeroed_planet_continuous_channels
            ),
            persistent=False,
        )
        self.register_buffer(
            "_zeroed_arrival_norm_extra_slots",
            zeroed_arrival_norm_extra_slots,
            persistent=False,
        )
        self.register_buffer(
            "_zeroed_edge_norm_extra_channels",
            self._edge_obs_feature_norm.extra_zero_mask_for_physical_channels(
                zeroed_edge_continuous_channels
            ),
            persistent=False,
        )
        self._zero_planet_continuous_enabled = bool(zeroed_planet_continuous_channels.any().item())
        self._zero_arrival_continuous_enabled = bool(zeroed_arrival_continuous_slots.any().item())
        self._zero_edge_continuous_enabled = bool(zeroed_edge_continuous_channels.any().item())
        self._zero_planet_norm_extra_enabled = bool(
            self._zeroed_planet_norm_extra_channels.any().item()
        )
        self._zero_arrival_norm_extra_enabled = bool(
            zeroed_arrival_norm_extra_slots.any().item()
        )
        self._zero_edge_norm_extra_enabled = bool(
            self._zeroed_edge_norm_extra_channels.any().item()
        )
        planet_base_dim = ORBIT_PLANET_PLAYER_FEATURE_OFFSET
        planet_player_block_dim = ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
        arrival_in_channels = self._ARRIVAL_CNN_IN_CHANNELS
        edge_base_dim = ORBIT_EDGE_PLAYER_FEATURE_OFFSET
        edge_player_block_dim = ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
        self._planet_feature_norm_base_extra_dim = _feature_norm_spike_count(
            feature_config,
            p_feature_keys[:ORBIT_PLANET_BASE_FEATURES],
        ) + _feature_norm_clip_count(
            feature_config,
            p_feature_keys[:ORBIT_PLANET_BASE_FEATURES],
        )
        self._planet_feature_norm_player_extra_dim = _feature_norm_spike_count(
            feature_config,
            p_feature_keys[ORBIT_PLANET_BASE_FEATURES:],
        ) + _feature_norm_clip_count(
            feature_config,
            p_feature_keys[ORBIT_PLANET_BASE_FEATURES:],
        )
        self._edge_feature_norm_base_extra_dim = _feature_norm_spike_count(
            feature_config,
            e_feature_keys[:ORBIT_EDGE_BASE_FEATURES],
        ) + _feature_norm_clip_count(
            feature_config,
            e_feature_keys[:ORBIT_EDGE_BASE_FEATURES],
        )
        self._edge_feature_norm_player_extra_dim = _feature_norm_spike_count(
            feature_config,
            e_feature_keys[ORBIT_EDGE_BASE_FEATURES:],
        ) + _feature_norm_clip_count(
            feature_config,
            e_feature_keys[ORBIT_EDGE_BASE_FEATURES:],
        )
        self._arrival_feature_norm_extra_dim = _feature_norm_spike_count(
            feature_config,
            arrival_feature_keys,
        ) + _feature_norm_clip_count(
            feature_config,
            arrival_feature_keys,
        )
        assert self._planet_obs_feature_norm.extra_dim == (
            self._planet_feature_norm_base_extra_dim
            + ORBIT_PLAYER_AXIS_SLOTS * self._planet_feature_norm_player_extra_dim
        )
        assert self._edge_obs_feature_norm.extra_dim == (
            self._edge_feature_norm_base_extra_dim
            + ORBIT_PLAYER_AXIS_SLOTS * self._edge_feature_norm_player_extra_dim
        )
        assert self._arrival_obs_feature_norm.extra_dim == self._arrival_feature_norm_extra_dim
        planet_base_dim += self._planet_feature_norm_base_extra_dim
        planet_player_block_dim += self._planet_feature_norm_player_extra_dim
        arrival_in_channels += self._arrival_feature_norm_extra_dim
        edge_base_dim += self._edge_feature_norm_base_extra_dim
        edge_player_block_dim += self._edge_feature_norm_player_extra_dim
        if self._use_obs_discrete_embeddings:
            planet_base_dim += _OrbitDiscreteFeatureEmbeddings.planet_base_extra_dim()
            planet_player_block_dim += _OrbitDiscreteFeatureEmbeddings.planet_player_extra_dim()
            arrival_in_channels += _OrbitDiscreteFeatureEmbeddings.arrival_extra_dim()
            edge_base_dim += _OrbitDiscreteFeatureEmbeddings.edge_base_extra_dim()
            edge_player_block_dim += _OrbitDiscreteFeatureEmbeddings.edge_player_extra_dim()
        self._planet_base_dim = planet_base_dim
        self._planet_player_block_dim = planet_player_block_dim
        if self._use_arrival_attention_fusion:
            self._planet_encoder = _TokenEncoderStack(
                in_dim=planet_base_dim + planet_player_block_dim,
                hidden_dim=h,
                n_layers=enc_nl,
                use_token_batch_norm=bool(use_token_batch_norm),
                activation=mlp_activation_name,
            )
        else:
            self._planet_encoder = _SelfEnemyBlockTokenEncoderStack(
                base_dim=planet_base_dim,
                player_block_dim=planet_player_block_dim,
                hidden_dim=h,
                n_layers=enc_nl,
                use_token_batch_norm=bool(use_token_batch_norm),
                activation=mlp_activation_name,
            )
        if self._use_arrival_attention_fusion:
            self._arrival_encoder = _PlanetArrivalAttentionEncoder(
                arrival_in_channels=arrival_in_channels,
                arrival_temporal_horizon=self._arrival_temporal_horizon,
                hidden_dim=h,
                n_layers=arr_enc_nl,
                num_heads=nh_t,
                num_queries=arrival_attn_nq,
                activation=mlp_activation_name,
            )
        else:
            self._arrival_encoder = _TemporalPlanetFeatureCnn(
                in_channels=arrival_in_channels,
                hidden_dim=h,
                n_layers=arr_enc_nl,
                recency_decay_horizon=self._arrival_temporal_horizon,
                activation=mlp_activation_name,
            )
        self._arrival_planet_fusion_projection = nn.Linear(h, h)
        self._arrival_embedding_fusion_projection = nn.Linear(h, h)
        self._arrival_fusion = _orbit_mlp(
            h,
            h,
            h,
            n_layers=2,
            activation=mlp_activation_name,
        )
        if self._use_arrival_attention_fusion_gate:
            assert self._use_arrival_attention_fusion
            self._arrival_fusion_gate = _orbit_mlp(
                2 * h,
                h,
                h,
                n_layers=2,
                activation=mlp_activation_name,
            )
        self._edge_base_dim = edge_base_dim
        if self._use_arrival_attention_fusion:
            self._edge_encoder = _TokenEncoderStack(
                in_dim=edge_base_dim,
                hidden_dim=h,
                n_layers=enc_nl,
                use_token_batch_norm=bool(use_token_batch_norm),
                activation=mlp_activation_name,
            )
            self._edge_src_owner_identity_projection = nn.Linear(h, h)
            self._edge_dst_owner_identity_projection = nn.Linear(h, h)
        else:
            self._edge_encoder = _SelfEnemyBlockTokenEncoderStack(
                base_dim=edge_base_dim,
                player_block_dim=edge_player_block_dim,
                hidden_dim=h,
                n_layers=enc_nl,
                use_token_batch_norm=bool(use_token_batch_norm),
                activation=mlp_activation_name,
            )
        if self._use_identity_embedding:
            self._identity_encoder = _orbit_mlp(
                _ORBIT_IDENTITY_CLASSES,
                h,
                h,
                n_layers=2,
                activation=mlp_activation_name,
            )
        if self._use_value_opponent_model_embedding:
            self._value_opponent_model_encoder = nn.Embedding(
                self._value_opponent_model_count + 1,
                h,
            )
            self._value_opponent_identity_model_fusion = _orbit_mlp(
                2 * h,
                h,
                h,
                n_layers=2,
                activation=mlp_activation_name,
            )
        if self._use_edge_relation_features:
            self._edge_relation_encoder = _orbit_mlp(
                _ORBIT_EDGE_RELATION_FEATURES,
                h,
                h,
                n_layers=2,
                activation=mlp_activation_name,
            )
        if self._use_obs_discrete_embeddings:
            self._obs_discrete_embeddings = _OrbitDiscreteFeatureEmbeddings(
                feature_config,
                zeroed_inputs,
                temporal_step_max_index=temporal_step_max_index,
            )
        self._orbit_attn_layers = nn.ModuleList(
            OrbitPlanetEdgeCrossAttentionBlock(
                hidden_dim=h,
                num_heads=nh_t,
                residual_dropout=residual_dropout,
                ffn_multiplier=ffn_multiplier,
                ffn_dropout=ffn_dropout,
                edge_endpoint_update=endpoint_update,
                allow_self_edges=allow_self_edges,
                activation=transformer_activation_name,
            )
            for _ in range(nl)
        )
        if self._include_rl_policy_value_heads:
            self._policy_head = OrbitSimpleSpawnFleetPolicyHead(
                hidden_dim=h,
                decoder_num_layers=dec_nl,
                num_heads=nh_t,
                num_policy_actions=self._num_policy_actions,
                entropy_floor_target=entropy_floor_target,
                entropy_floor_max_temperature=float(entropy_floor_max_temperature),
                entropy_floor_num_iters=int(entropy_floor_num_iters),
                activation=mlp_activation_name,
            )
            if self._include_rl_value_head:
                self._global_value_head = OrbitSimpleGlobalValuePoolHead(
                    hidden_dim=h,
                    decoder_num_layers=dec_nl,
                    activation=mlp_activation_name,
                )
                self._global_value_head_production_delta = OrbitSimpleGlobalValuePoolHead(
                    hidden_dim=h,
                    decoder_num_layers=dec_nl,
                    activation=mlp_activation_name,
                )

        self.apply(_orbit_module_init_)
        residual_scale = (3.0 * float(nl)) ** -0.5
        for layer in self._orbit_attn_layers:
            assert isinstance(layer, OrbitPlanetEdgeCrossAttentionBlock)
            _xavier_uniform_mha_(layer._planet_self_attn, out_proj_scale=residual_scale)
            _xavier_uniform_mha_(layer._planet_from_edges_attn, out_proj_scale=residual_scale)
            _xavier_uniform_mha_(layer._edge_from_planets_attn, out_proj_scale=residual_scale)
            edge_context_out = layer._edge_context_mlp[-1]
            assert isinstance(edge_context_out, nn.Linear)
            _xavier_uniform_linear_(edge_context_out, scale=residual_scale)
            if layer._edge_endpoint_update == "directed_fusion":
                edge_endpoint_fusion_out = layer._edge_endpoint_fusion[-1]
                assert isinstance(edge_endpoint_fusion_out, nn.Linear)
                _xavier_uniform_linear_(edge_endpoint_fusion_out, scale=residual_scale)
            elif layer._edge_endpoint_update == "gated_fusion":
                _xavier_uniform_linear_(layer._edge_endpoint_out, scale=residual_scale)
            planet_ffn_out = layer._planet_ffn[-1]
            edge_ffn_out = layer._edge_ffn[-1]
            assert isinstance(planet_ffn_out, nn.Linear)
            assert isinstance(edge_ffn_out, nn.Linear)
            _xavier_uniform_linear_(planet_ffn_out, scale=residual_scale)
            _xavier_uniform_linear_(edge_ffn_out, scale=residual_scale)
        if self._include_rl_policy_value_heads:
            assert isinstance(self._policy_head, OrbitSimpleSpawnFleetPolicyHead)
            _xavier_uniform_mha_(self._policy_head._dst_attn, out_proj_scale=residual_scale)
            policy_dst_ffn_out = self._policy_head._dst_ffn[-1]
            assert isinstance(policy_dst_ffn_out, nn.Linear)
            _xavier_uniform_linear_(policy_dst_ffn_out, scale=residual_scale)
            policy_decoders = (
                self._policy_head._pair_mlp,
            )
            tail_decoders: tuple[nn.Sequential, ...] = policy_decoders
            if self._include_rl_value_head:
                assert isinstance(self._global_value_head, OrbitSimpleGlobalValuePoolHead)
                assert isinstance(
                    self._global_value_head_production_delta,
                    OrbitSimpleGlobalValuePoolHead,
                )
                baseline_value_decoders = (self._global_value_head._value_decoder,)
                production_delta_value_decoders = (
                    self._global_value_head_production_delta._value_decoder,
                )
                value_decoders = (*baseline_value_decoders, *production_delta_value_decoders)
                tail_decoders = (*value_decoders, *policy_decoders)
            for dec in tail_decoders:
                assert isinstance(dec, nn.Sequential)
                last_lin = dec[-1]
                assert isinstance(last_lin, nn.Linear)
                nn.init.trunc_normal_(last_lin.weight, std=1e-3, a=-2e-3, b=2e-3)

    def set_is_sample(self, enabled: bool) -> None:
        self._is_sample = bool(enabled)

    def set_compile_friendly_sample(self, enabled: bool) -> None:
        self._compile_friendly_sample = bool(enabled)

    def set_shuffle_identity_ids(self, enabled: bool) -> None:
        self._shuffle_identity_ids = bool(enabled)

    def _player_identity_embeddings(
        self,
        identity_lookup: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self._use_identity_embedding
        assert identity_lookup.ndim == 2, identity_lookup.shape
        b = int(identity_lookup.shape[0])
        assert identity_lookup.shape == (b, _ORBIT_IDENTITY_CLASSES), (
            identity_lookup.shape,
        )
        identity_class = identity_lookup[:, :ORBIT_PLAYER_AXIS_SLOTS]
        assert identity_class.shape == (b, ORBIT_PLAYER_AXIS_SLOTS), identity_class.shape
        identity_one_hot = F.one_hot(
            identity_class,
            num_classes=_ORBIT_IDENTITY_CLASSES,
        ).to(dtype=dtype)
        out = self._identity_encoder(identity_one_hot)
        assert out.shape == (b, ORBIT_PLAYER_AXIS_SLOTS, self._hidden), (
            out.shape,
            b,
            ORBIT_PLAYER_AXIS_SLOTS,
            self._hidden,
        )
        return out

    def _value_opponent_context(
        self,
        batch: dict,
        player_identity_emb: torch.Tensor,
        enemy_valid_bool: torch.Tensor,
        *,
        b_e: int,
        b_p: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert self._use_value_opponent_model_embedding
        assert player_identity_emb.shape == (
            int(b_e) * int(b_p),
            ORBIT_PLAYER_AXIS_SLOTS,
            self._hidden,
        ), player_identity_emb.shape
        assert enemy_valid_bool.shape == (
            int(b_e) * int(b_p),
            ORBIT_ENEMY_AXIS_SLOTS,
        ), enemy_valid_bool.shape
        assert "frozen_model_by_player_axis_LEARN" in batch, (
            "value opponent model embedding requires frozen_model_by_player_axis_LEARN"
        )
        model_by_axis = batch["frozen_model_by_player_axis_LEARN"]
        assert isinstance(model_by_axis, torch.Tensor)
        assert model_by_axis.dtype == torch.int64, model_by_axis.dtype
        assert tuple(model_by_axis.shape) == (
            int(b_e),
            int(b_p),
            ORBIT_PLAYER_AXIS_SLOTS,
        ), tuple(model_by_axis.shape)
        model_by_axis_flat = model_by_axis.reshape(
            int(b_e) * int(b_p),
            ORBIT_PLAYER_AXIS_SLOTS,
        ).to(device=device)
        _assert_tensor(
            torch.all(model_by_axis_flat >= 0),
            "frozen_model_by_player_axis_LEARN must contain nonnegative model ids",
        )
        _assert_tensor(
            torch.all(model_by_axis_flat <= self._value_opponent_model_count),
            "frozen_model_by_player_axis_LEARN model id exceeds configured value_opponent_model_count",
        )
        enemy_model_ids = model_by_axis_flat[:, 1:]
        assert enemy_model_ids.shape == (
            int(b_e) * int(b_p),
            ORBIT_ENEMY_AXIS_SLOTS,
        ), enemy_model_ids.shape
        opponent_model_emb = self._value_opponent_model_encoder(enemy_model_ids)
        opponent_model_emb = opponent_model_emb.to(dtype=dtype)
        enemy_identity_emb = player_identity_emb[:, 1:, :].to(dtype=dtype)
        assert opponent_model_emb.shape == enemy_identity_emb.shape, (
            opponent_model_emb.shape,
            enemy_identity_emb.shape,
        )
        enemy_pair_emb = self._value_opponent_identity_model_fusion(
            torch.cat([enemy_identity_emb, opponent_model_emb], dim=-1)
        )
        assert enemy_pair_emb.shape == (
            int(b_e) * int(b_p),
            ORBIT_ENEMY_AXIS_SLOTS,
            self._hidden,
        ), enemy_pair_emb.shape
        enemy_weight = enemy_valid_bool.to(device=device, dtype=dtype).unsqueeze(-1)
        enemy_count = enemy_weight.sum(dim=1)
        opponent_context = (enemy_pair_emb * enemy_weight).sum(dim=1) / enemy_count.clamp_min(1.0)
        assert opponent_context.shape == (
            int(b_e) * int(b_p),
            self._hidden,
        ), opponent_context.shape
        return opponent_context

    def forward(
        self,
        batch: dict,
        *,
        output_full_policy_log_probs: bool = True,
        include_policy_logits_pre_action_mask: bool = False,
        include_final_policy_logits: bool = False,
        include_value_head: bool = True,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> dict[str, Any]:
        assert "obs_LEARN_INFER" in batch, (
            f"ImpalaOrbitModel: expected obs_LEARN_INFER, got keys {sorted(batch.keys())}"
        )
        p = next(self.parameters())
        device, dtype = p.device, p.dtype
        with wall_tree_cuda_model_block(
            wall_profiler, device, "orbit_input_reshape_discrete"
        ):
            o = batch["obs_LEARN_INFER"]
            assert isinstance(o, dict)
            for k in (
                "orbit_planet_features",
                "orbit_planet_arrival_features",
                "orbit_enemy_mask",
                "orbit_planet_mask",
                "orbit_planet_pairwise_mask",
                "orbit_planet_pairwise_features",
                "available_action_mask",
            ):
                assert k in o, f"obs_LEARN_INFER missing {k!r}"
            planet_f = o["orbit_planet_features"]
            arrival_f = o["orbit_planet_arrival_features"]
            enemy_mask = o["orbit_enemy_mask"]
            mask = o["orbit_planet_mask"]
            pm = o["orbit_planet_pairwise_mask"]
            edge_f = o["orbit_planet_pairwise_features"]
            m_orbit = o["available_action_mask"]
            embedding_feature_keys = (
                "orbit_planet_embedding_features",
                "orbit_planet_arrival_embedding_features",
                "orbit_planet_pairwise_embedding_features",
            )
            has_embedding_features = tuple(k in o for k in embedding_feature_keys)
            if any(has_embedding_features):
                assert all(has_embedding_features), (
                    "embedding feature inputs must be provided as a complete planet/arrival/edge set",
                    embedding_feature_keys,
                    sorted(o.keys()),
                )
                planet_embedding_f = o["orbit_planet_embedding_features"]
                arrival_embedding_f = o["orbit_planet_arrival_embedding_features"]
                edge_embedding_f = o["orbit_planet_pairwise_embedding_features"]
            else:
                planet_embedding_f = planet_f
                arrival_embedding_f = arrival_f
                edge_embedding_f = edge_f
            b_e = int(planet_f.shape[0])
            b_p = int(planet_f.shape[1])
            b_tot = b_e * b_p
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                assert isinstance(planet_f, torch.Tensor)
                assert isinstance(arrival_f, torch.Tensor)
                assert isinstance(planet_embedding_f, torch.Tensor)
                assert isinstance(arrival_embedding_f, torch.Tensor)
                assert isinstance(enemy_mask, torch.Tensor)
                assert isinstance(mask, torch.Tensor)
                assert isinstance(pm, torch.Tensor)
                assert isinstance(edge_f, torch.Tensor)
                assert isinstance(edge_embedding_f, torch.Tensor)
                assert isinstance(m_orbit, torch.Tensor)
                assert planet_f.ndim == 4, (
                    "orbit_planet_features must be [batch, players, "
                    f"{ORBIT_MAX_PLANETS},{ORBIT_PLANET_FEATURES}], "
                    f"got {planet_f.ndim=} {tuple(planet_f.shape)}"
                )
                assert planet_f.shape == (
                    b_e,
                    b_p,
                    ORBIT_MAX_PLANETS,
                    ORBIT_PLANET_FEATURES,
                )
                assert arrival_f.shape == (
                    b_e,
                    b_p,
                    ORBIT_MAX_PLANETS,
                    ORBIT_PLANET_ARRIVAL_HORIZON,
                    ORBIT_PLAYER_AXIS_SLOTS,
                    ORBIT_PLANET_TEMPORAL_FEATURES,
                )
                assert enemy_mask.shape == (
                    b_e,
                    b_p,
                    ORBIT_ENEMY_AXIS_SLOTS,
                ), f"orbit_enemy_mask shape mismatch: {tuple(enemy_mask.shape)}"
                assert mask.shape == (b_e, b_p, ORBIT_MAX_PLANETS)
                assert pm.shape == (b_e, b_p, ORBIT_PLANET_PAIRWISE_COUNT)
                assert edge_f.shape == (
                    b_e,
                    b_p,
                    ORBIT_PLANET_PAIRWISE_COUNT,
                    ORBIT_EDGE_FEATURES,
                )
                assert m_orbit.dtype == torch.int8, (
                    f"available_action_mask dtype must be torch.int8, got {m_orbit.dtype}"
                )
                assert planet_embedding_f.shape == planet_f.shape, (
                    "orbit_planet_embedding_features shape mismatch: "
                    f"{tuple(planet_embedding_f.shape)} vs {tuple(planet_f.shape)}"
                )
                assert arrival_embedding_f.shape == arrival_f.shape, (
                    "orbit_planet_arrival_embedding_features shape mismatch: "
                    f"{tuple(arrival_embedding_f.shape)} vs {tuple(arrival_f.shape)}"
                )
                assert edge_embedding_f.shape == edge_f.shape, (
                    "orbit_planet_pairwise_embedding_features shape mismatch: "
                    f"{tuple(edge_embedding_f.shape)} vs {tuple(edge_f.shape)}"
                )
                assert m_orbit.shape == (
                    b_e,
                    b_p,
                    ORBIT_PLANET_ACTION_SLOTS,
                    ORBIT_PER_PLANET_MOVE_CLASSES,
                )
                assert ORBIT_PLANET_PAIRWISE_COUNT == ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS
                assert ORBIT_PLANET_ACTION_SLOTS == ORBIT_MAX_PLANETS

            n = ORBIT_MAX_PLANETS
        with wall_tree_cuda_model_block(
            wall_profiler, device, "orbit_input_reshape_discrete_assert"
        ):
            arrival_temporal_horizon = int(self._arrival_temporal_horizon)
            temporal_step_max_index = _orbit_model_temporal_step_max_index(
                arrival_temporal_horizon
            )
            planet_input_raw = planet_f.reshape(b_tot, n, ORBIT_PLANET_FEATURES).to(
                device=device
            )
            arrival_input_raw = arrival_f.reshape(
                b_tot,
                n,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ).to(device=device)
            assert arrival_input_raw.shape == (
                b_tot,
                n,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                self._ARRIVAL_IN_DIM,
            )
            arrival_input_raw = arrival_input_raw[
                :, :, :arrival_temporal_horizon, :, :
            ].contiguous()
            assert arrival_input_raw.shape == (
                b_tot,
                n,
                arrival_temporal_horizon,
                ORBIT_PLAYER_AXIS_SLOTS,
                self._ARRIVAL_IN_DIM,
            )
            edge_input_raw = edge_f.reshape(b_tot, n, n, ORBIT_EDGE_FEATURES).to(
                device=device
            )
            assert edge_input_raw.shape == (b_tot, n, n, self._RAW_EDGE_DIM)
            enemy_valid = enemy_mask.reshape(b_tot, ORBIT_ENEMY_AXIS_SLOTS).to(
                device=device,
                dtype=dtype,
            )
            planet_input = planet_input_raw.to(dtype=dtype)
            assert planet_input.shape == (b_tot, n, self._PLANET_IN_DIM)
            arrival_input = arrival_input_raw.to(dtype=dtype)
            assert enemy_valid.shape == (b_tot, ORBIT_ENEMY_AXIS_SLOTS), enemy_valid.shape
            planet_embedding_input_raw = planet_embedding_f.reshape(
                b_tot, n, ORBIT_PLANET_FEATURES
            ).to(device=device)
            arrival_embedding_input_raw = arrival_embedding_f.reshape(
                b_tot,
                n,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ).to(device=device)
            arrival_embedding_input_raw = arrival_embedding_input_raw[
                :, :, :arrival_temporal_horizon, :, :
            ].contiguous()
            edge_embedding_input_raw = edge_embedding_f.reshape(
                b_tot, n, n, ORBIT_EDGE_FEATURES
            ).to(device=device)
            assert planet_embedding_input_raw.shape == planet_input_raw.shape, (
                planet_embedding_input_raw.shape,
                planet_input_raw.shape,
            )
            assert arrival_embedding_input_raw.shape == arrival_input_raw.shape, (
                arrival_embedding_input_raw.shape,
                arrival_input_raw.shape,
            )
            assert edge_embedding_input_raw.shape == edge_input_raw.shape, (
                edge_embedding_input_raw.shape,
                edge_input_raw.shape,
            )
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                _assert_orbit_row_tensor_finite(
                    head_name="orbit_encoder",
                    reason="planet input contains non-finite values before obs normalization",
                    row_tensor=planet_input,
                    context={
                        "planet_input_raw_row": planet_input_raw,
                    },
                )
            _assert_orbit_discrete_obs_integer_contract(
                planet_embedding_input_raw,
                arrival_embedding_input_raw,
                edge_embedding_input_raw,
                temporal_step_max_index=temporal_step_max_index,
            )

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_masks_pairwise"):
            planet_slot_mask = mask.reshape(b_tot, n).to(device=device, dtype=dtype)
            planet_valid_bool = planet_slot_mask > 0.5
            planet_feature_valid_bool = planet_valid_bool
            pair_valid = pm.reshape(b_tot, n, n).to(device=device, dtype=dtype)
            if _ENABLE_MODEL_ASSERT:
                expected_pair_valid = planet_slot_mask.unsqueeze(2) * planet_slot_mask.unsqueeze(1)
                _assert_tensor(
                    torch.all((pair_valid > 0.5) == (expected_pair_valid > 0.5)),
                    "orbit_planet_pairwise_mask must match orbit_planet_mask outer-product",
                )
            pair_valid_bool = pair_valid > 0.5
            pair_feature_valid_bool = pair_valid_bool
            enemy_valid_bool = enemy_valid > 0.5
            player_feature_block_valid_bool = torch.cat(
                [
                    torch.ones((b_tot, 1), device=device, dtype=torch.bool),
                    enemy_valid_bool,
                ],
                dim=1,
            )
            assert player_feature_block_valid_bool.shape == (
                b_tot,
                ORBIT_PLAYER_AXIS_SLOTS,
            ), player_feature_block_valid_bool.shape
            owner_player_weights = _orbit_owner_player_weights_from_raw_planet_input(
                planet_input_raw,
                player_feature_block_valid_bool,
            ).to(dtype=dtype)
            planet_channel_valid_bool = _self_enemy_block_feature_valid_mask(
                planet_input,
                player_feature_offset=ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
                player_features_per_player=ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
                enemy_mask=enemy_valid_bool,
            )
            arrival_feature_valid_bool = planet_feature_valid_bool.view(b_tot, n, 1, 1).expand(
                b_tot, n, arrival_temporal_horizon, ORBIT_PLAYER_AXIS_SLOTS
            )
            arrival_feature_valid_bool = (
                arrival_feature_valid_bool
                & player_feature_block_valid_bool.view(
                    b_tot,
                    1,
                    1,
                    ORBIT_PLAYER_AXIS_SLOTS,
                )
            )
        if self._use_obs_discrete_embeddings:
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_discrete_embeddings"
            ):
                assert isinstance(self._obs_discrete_embeddings, _OrbitDiscreteFeatureEmbeddings)
                planet_discrete_extra, arrival_discrete_extra, edge_discrete_extra = (
                    self._obs_discrete_embeddings(
                        planet_embedding_input_raw,
                        arrival_embedding_input_raw,
                        edge_embedding_input_raw,
                        planet_slot_mask,
                        arrival_feature_valid_bool,
                        pair_valid,
                        wall_profiler,
                    )
                )
                planet_discrete_extra = planet_discrete_extra.to(dtype=dtype)
                arrival_discrete_extra = arrival_discrete_extra.to(dtype=dtype)
                edge_discrete_extra = edge_discrete_extra.to(dtype=dtype)
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_feature_norm"):
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_planet"
            ):
                planet_input, planet_norm_extra = self._planet_obs_feature_norm(
                    planet_input,
                    planet_feature_valid_bool,
                    feature_valid_mask=planet_channel_valid_bool,
                    wall_profiler=wall_profiler,
                )
                if self._zero_planet_continuous_enabled:
                    planet_input = _zero_last_dim_channels(
                        planet_input,
                        self._zeroed_planet_continuous_channels,
                    )
                if self._zero_planet_norm_extra_enabled:
                    planet_norm_extra = _zero_last_dim_channels(
                        planet_norm_extra,
                        self._zeroed_planet_norm_extra_channels,
                    )
                if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                    _assert_orbit_row_tensor_finite(
                        head_name="orbit_encoder",
                        reason="planet input contains non-finite values after obs normalization",
                        row_tensor=planet_input,
                        context={
                            "planet_norm_extra_row": planet_norm_extra,
                            "planet_slot_mask_row": planet_slot_mask,
                        },
                    )
                    _assert_orbit_row_tensor_finite(
                        head_name="orbit_encoder",
                        reason="planet norm extras contain non-finite values after obs normalization",
                        row_tensor=planet_norm_extra,
                        context={
                            "planet_input_row": planet_input,
                            "planet_slot_mask_row": planet_slot_mask,
                        },
                    )
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_arrival"
            ):
                arrival_input, arrival_norm_extra = self._arrival_obs_feature_norm(
                    arrival_input,
                    arrival_feature_valid_bool,
                    wall_profiler=wall_profiler,
                )
                if self._zero_arrival_continuous_enabled:
                    arrival_input = _zero_arrival_slot_channels(
                        arrival_input,
                        self._zeroed_arrival_continuous_slots,
                    )
                if self._zero_arrival_norm_extra_enabled:
                    arrival_norm_extra = _zero_arrival_slot_channels(
                        arrival_norm_extra,
                        self._zeroed_arrival_norm_extra_slots,
                    )
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_arrival_mask"
            ):
                arrival_input = arrival_input * arrival_feature_valid_bool.unsqueeze(-1).to(
                    dtype=arrival_input.dtype
                )
                arrival_norm_extra = arrival_norm_extra * arrival_feature_valid_bool.unsqueeze(
                    -1
                ).to(dtype=arrival_norm_extra.dtype)
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_planet_append"
            ):
                planet_input = _append_self_enemy_block_feature_extras(
                    planet_input,
                    planet_norm_extra,
                    base_dim=ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
                    player_block_dim=ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
                    base_extra_dim=self._planet_feature_norm_base_extra_dim,
                    player_extra_dim=self._planet_feature_norm_player_extra_dim,
                )
                if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                    _assert_orbit_row_tensor_finite(
                        head_name="orbit_encoder",
                        reason="planet input contains non-finite values after norm extras append",
                        row_tensor=planet_input,
                        context={
                            "planet_slot_mask_row": planet_slot_mask,
                        },
                    )
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_arrival_append"
            ):
                arrival_input = torch.cat([arrival_input, arrival_norm_extra], dim=-1)
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_edge_prepare"
            ):
                edge_input = edge_input_raw.to(dtype=dtype)
                assert edge_input.shape == (b_tot, n, n, self._RAW_EDGE_DIM)
                player_offset = ORBIT_EDGE_PLAYER_FEATURE_OFFSET
                player_width = ORBIT_PLAYER_AXIS_SLOTS * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
                assert ORBIT_EDGE_FEATURES == player_offset + player_width
                assert ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER == 2, (
                    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
                )
                edge_channel_valid_bool = _self_enemy_block_feature_valid_mask(
                    edge_input,
                    player_feature_offset=ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
                    player_features_per_player=ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
                    enemy_mask=enemy_valid_bool,
                )
            with wall_tree_cuda_model_block(wall_profiler, device, "orbit_obs_feature_norm_edge"):
                edge_input, edge_norm_extra = self._edge_obs_feature_norm(
                    edge_input,
                    pair_feature_valid_bool,
                    feature_valid_mask=edge_channel_valid_bool,
                    wall_profiler=wall_profiler,
                )
                if self._zero_edge_continuous_enabled:
                    edge_input = _zero_last_dim_channels(
                        edge_input,
                        self._zeroed_edge_continuous_channels,
                    )
                if self._zero_edge_norm_extra_enabled:
                    edge_norm_extra = _zero_last_dim_channels(
                        edge_norm_extra,
                        self._zeroed_edge_norm_extra_channels,
                    )
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_obs_feature_norm_edge_append"
            ):
                edge_input = _append_self_enemy_block_feature_extras(
                    edge_input,
                    edge_norm_extra,
                    base_dim=ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
                    player_block_dim=ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
                    base_extra_dim=self._edge_feature_norm_base_extra_dim,
                    player_extra_dim=self._edge_feature_norm_player_extra_dim,
                )
            if self._use_obs_discrete_embeddings:
                planet_input = _append_self_enemy_block_feature_extras(
                    planet_input,
                    planet_discrete_extra,
                    base_dim=(
                        ORBIT_PLANET_PLAYER_FEATURE_OFFSET
                        + self._planet_feature_norm_base_extra_dim
                    ),
                    player_block_dim=(
                        ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
                        + self._planet_feature_norm_player_extra_dim
                    ),
                    base_extra_dim=_OrbitDiscreteFeatureEmbeddings.planet_base_extra_dim(),
                    player_extra_dim=_OrbitDiscreteFeatureEmbeddings.planet_player_extra_dim(),
                )
                if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                    _assert_orbit_row_tensor_finite(
                        head_name="orbit_encoder",
                        reason="planet input contains non-finite values after discrete embedding append",
                        row_tensor=planet_input,
                        context={
                            "planet_discrete_extra_row": planet_discrete_extra,
                            "planet_slot_mask_row": planet_slot_mask,
                        },
                    )
                arrival_input = torch.cat([arrival_input, arrival_discrete_extra], dim=-1)
                assert arrival_input.shape[-1] == (
                    self._ARRIVAL_IN_DIM
                    + self._arrival_feature_norm_extra_dim
                    + _OrbitDiscreteFeatureEmbeddings.arrival_extra_dim()
                ), arrival_input.shape
                edge_input = _append_self_enemy_block_feature_extras(
                    edge_input,
                    edge_discrete_extra,
                    base_dim=(
                        ORBIT_EDGE_PLAYER_FEATURE_OFFSET
                        + self._edge_feature_norm_base_extra_dim
                    ),
                    player_block_dim=(
                        ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
                        + self._edge_feature_norm_player_extra_dim
                    ),
                    base_extra_dim=_OrbitDiscreteFeatureEmbeddings.edge_base_extra_dim(),
                    player_extra_dim=_OrbitDiscreteFeatureEmbeddings.edge_player_extra_dim(),
                )

        player_identity_emb: torch.Tensor | None = None
        if self._use_identity_embedding:
            with wall_tree_cuda_model_block(wall_profiler, device, "orbit_identity_embedding"):
                identity_lookup = _orbit_identity_lookup(
                    b_tot,
                    shuffle_identity_ids=self._shuffle_identity_ids,
                    device=device,
                )
                player_identity_emb = self._player_identity_embeddings(
                    identity_lookup,
                    dtype=dtype,
                )

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_planet_encoder"):
            if self._use_arrival_attention_fusion:
                assert isinstance(self._planet_encoder, _TokenEncoderStack)
                with wall_tree_cuda_model_block(
                    wall_profiler,
                    device,
                    "orbit_planet_owner_input",
                ):
                    planet_base_input = planet_input[..., : self._planet_base_dim]
                    planet_blocks = planet_input[..., self._planet_base_dim :].reshape(
                        b_tot,
                        n,
                        ORBIT_PLAYER_AXIS_SLOTS,
                        self._planet_player_block_dim,
                    )
                    owner_weights = owner_player_weights.to(
                        device=planet_input.device,
                        dtype=planet_input.dtype,
                    )
                    planet_owner_block = (planet_blocks * owner_weights.unsqueeze(-1)).sum(dim=-2)
                    assert planet_owner_block.shape == (
                        b_tot,
                        n,
                        self._planet_player_block_dim,
                    ), planet_owner_block.shape
                    planet_owner_input = torch.cat(
                        [planet_base_input, planet_owner_block],
                        dim=-1,
                    )
                with wall_tree_cuda_model_block(
                    wall_profiler,
                    device,
                    "orbit_planet_owner_encoder",
                ):
                    planet_emb = self._planet_encoder(planet_owner_input, planet_slot_mask)
                if player_identity_emb is not None:
                    with wall_tree_cuda_model_block(
                        wall_profiler,
                        device,
                        "orbit_planet_owner_identity",
                    ):
                        planet_owner_identity = (
                            player_identity_emb.to(dtype=planet_emb.dtype).unsqueeze(1)
                            * owner_player_weights.unsqueeze(-1).to(dtype=planet_emb.dtype)
                        ).sum(dim=-2)
                        assert planet_owner_identity.shape == (b_tot, n, self._hidden), (
                            planet_owner_identity.shape
                        )
                        planet_emb = planet_emb + planet_owner_identity
                        planet_emb = planet_emb * planet_valid_bool.unsqueeze(-1).to(
                            dtype=planet_emb.dtype
                        )
            else:
                assert isinstance(self._planet_encoder, _SelfEnemyBlockTokenEncoderStack)
                planet_emb = self._planet_encoder(
                    planet_input,
                    planet_slot_mask,
                    enemy_valid_bool,
                    player_identity_emb=player_identity_emb,
                )
            assert planet_emb.shape == (b_tot, n, self._hidden)
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                _assert_orbit_row_tensor_finite(
                    head_name="orbit_encoder",
                    reason="planet embedding contains non-finite values after planet encoder",
                    row_tensor=planet_emb,
                    context={
                        "planet_input_row": planet_input,
                        "planet_slot_mask_row": planet_slot_mask,
                    },
                )
        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_arrival_encoder_fusion"):
            if self._use_arrival_attention_fusion:
                assert isinstance(self._arrival_encoder, _PlanetArrivalAttentionEncoder)
                arrival_emb = self._arrival_encoder(
                    planet_emb,
                    arrival_input,
                    player_feature_block_valid_bool,
                    player_identity_emb=player_identity_emb,
                    wall_profiler=wall_profiler,
                )
            else:
                assert isinstance(self._arrival_encoder, _TemporalPlanetFeatureCnn)
                arrival_emb = self._arrival_encoder(
                    arrival_input,
                    enemy_valid,
                    player_identity_emb=player_identity_emb,
                )
            assert arrival_emb.shape == (b_tot, n, self._hidden)
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                _assert_orbit_row_tensor_finite(
                    head_name="orbit_encoder",
                    reason="arrival embedding contains non-finite values after arrival encoder",
                    row_tensor=arrival_emb,
                    context={
                        "arrival_input_row": arrival_input.flatten(start_dim=2),
                        "planet_emb_row": planet_emb,
                    },
                )
            arrival_fused = (
                self._arrival_planet_fusion_projection(planet_emb)
                + self._arrival_embedding_fusion_projection(arrival_emb)
            )
            assert arrival_fused.shape == (b_tot, n, self._hidden), arrival_fused.shape
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                _assert_orbit_row_tensor_finite(
                    head_name="orbit_encoder",
                    reason="arrival fused embedding contains non-finite values before arrival fusion MLP",
                    row_tensor=arrival_fused,
                    context={
                        "planet_emb_row": planet_emb,
                        "arrival_emb_row": arrival_emb,
                    },
                )
            arrival_delta = self._arrival_fusion(arrival_fused)
            assert arrival_delta.shape == (b_tot, n, self._hidden), arrival_delta.shape
            if self._use_arrival_attention_fusion_gate:
                with wall_tree_cuda_model_block(
                    wall_profiler,
                    device,
                    "orbit_arrival_attention_fusion_gate",
                ):
                    arrival_gate_input = torch.cat([planet_emb, arrival_emb], dim=-1)
                    assert arrival_gate_input.shape == (b_tot, n, 2 * self._hidden), (
                        arrival_gate_input.shape
                    )
                    arrival_gate = torch.sigmoid(self._arrival_fusion_gate(arrival_gate_input))
                    assert arrival_gate.shape == (b_tot, n, self._hidden), arrival_gate.shape
                    arrival_delta = arrival_delta * arrival_gate
            planet_emb = planet_emb + arrival_delta
            assert planet_emb.shape == (b_tot, n, self._hidden)
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                _assert_orbit_row_tensor_finite(
                    head_name="orbit_encoder",
                    reason="planet embedding contains non-finite values after arrival fusion",
                    row_tensor=planet_emb,
                    context={
                        "arrival_fused_row": arrival_fused,
                        "arrival_emb_row": arrival_emb,
                    },
                )

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_edge_encoder"):
            if self._use_arrival_attention_fusion:
                assert isinstance(self._edge_encoder, _TokenEncoderStack)
                with wall_tree_cuda_model_block(
                    wall_profiler,
                    device,
                    "orbit_edge_base_encoder",
                ):
                    edge_base_input = edge_input[..., : self._edge_base_dim]
                    edge_emb = self._edge_encoder(edge_base_input, pair_valid_bool)
                if player_identity_emb is not None:
                    with wall_tree_cuda_model_block(
                        wall_profiler,
                        device,
                        "orbit_edge_owner_identity",
                    ):
                        src_owner_identity = (
                            player_identity_emb.to(dtype=edge_emb.dtype).unsqueeze(1)
                            * owner_player_weights.unsqueeze(-1).to(dtype=edge_emb.dtype)
                        ).sum(dim=-2)
                        assert src_owner_identity.shape == (b_tot, n, self._hidden), (
                            src_owner_identity.shape
                        )
                        dst_owner_identity = src_owner_identity
                        edge_emb = (
                            edge_emb
                            + self._edge_src_owner_identity_projection(
                                src_owner_identity
                            ).unsqueeze(2)
                            + self._edge_dst_owner_identity_projection(
                                dst_owner_identity
                            ).unsqueeze(1)
                        )
                edge_emb = edge_emb * pair_valid_bool.unsqueeze(-1).to(dtype=edge_emb.dtype)
            else:
                assert isinstance(self._edge_encoder, _SelfEnemyBlockTokenEncoderStack)
                edge_emb = self._edge_encoder(
                    edge_input,
                    pair_valid_bool,
                    enemy_valid_bool,
                    player_identity_emb=player_identity_emb,
                )
            assert edge_emb.shape == (b_tot, n, n, self._hidden)
            if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                _assert_orbit_row_tensor_finite(
                    head_name="orbit_encoder",
                    reason="edge embedding contains non-finite values after edge encoder",
                    row_tensor=edge_emb,
                    context={
                        "edge_input_row": edge_input.flatten(start_dim=2),
                        "pair_valid_row": pair_valid,
                    },
                )
        if self._use_edge_relation_features:
            with wall_tree_cuda_model_block(wall_profiler, device, "orbit_edge_relation_features"):
                edge_relation_features = _orbit_edge_relation_features(
                    edge_input_raw,
                    pair_valid_bool,
                )
                edge_relation_emb = self._edge_relation_encoder(
                    edge_relation_features.to(dtype=edge_emb.dtype)
                )
                assert edge_relation_emb.shape == edge_emb.shape, (
                    edge_relation_emb.shape,
                    edge_emb.shape,
                )
                edge_emb = edge_emb + edge_relation_emb * pair_valid_bool.unsqueeze(-1).to(
                    dtype=edge_emb.dtype
                )

        with wall_tree_cuda_model_block(wall_profiler, device, "orbit_transformer_layers"):
            for i, layer in enumerate(self._orbit_attn_layers):
                with wall_tree_cuda_model_block(
                    wall_profiler, device, f"orbit_transformer_layer_{i}"
                ):
                    planet_emb, edge_emb = layer(
                        planet_emb=planet_emb,
                        edge_emb=edge_emb,
                        planet_mask=planet_slot_mask,
                        pair_valid=pair_valid,
                        wall_profiler=wall_profiler,
                    )
                    if _ENABLE_MODEL_ASSERT and not torch.compiler.is_compiling():
                        _assert_orbit_row_tensor_finite(
                            head_name="orbit_encoder",
                            reason=f"planet embedding contains non-finite values after transformer layer {i}",
                            row_tensor=planet_emb,
                            context={
                                "edge_emb_row": edge_emb.flatten(start_dim=2),
                                "planet_slot_mask_row": planet_slot_mask,
                            },
                        )
                        _assert_orbit_row_tensor_finite(
                            head_name="orbit_encoder",
                            reason=f"edge embedding contains non-finite values after transformer layer {i}",
                            row_tensor=edge_emb,
                            context={
                                "planet_emb_row": planet_emb,
                                "pair_valid_row": pair_valid,
                            },
                        )
            assert planet_emb.shape == (b_tot, n, self._hidden)

        entropy_floor_stats_learn: dict[str, torch.Tensor] = {}
        final_policy_logits_learn: dict[str, torch.Tensor] = {}
        if self._include_rl_policy_value_heads:
            with wall_tree_cuda_model_block(wall_profiler, device, "orbit_spawn_policy_head"):
                available_action_mask = _orbit_available_action_mask_for_model(
                    m_orbit.to(device=device),
                    num_policy_actions=self._num_policy_actions,
                )
                assert available_action_mask.shape == (
                    b_e,
                    b_p,
                    ORBIT_PLANET_ACTION_SLOTS,
                    ORBIT_PLANET_ACTION_SLOTS * self._num_policy_actions,
                ), available_action_mask.shape
                available_action_flat = available_action_mask.reshape(
                    b_tot,
                    ORBIT_PLANET_ACTION_SLOTS,
                    ORBIT_PLANET_ACTION_SLOTS * self._num_policy_actions,
                )
                _assert_tensor(
                    torch.all(available_action_flat.sum(dim=-1) > 0),
                    "Each action slot must contain at least one available action",
                )
                with (
                    torch.amp.autocast(device_type=planet_emb.device.type, enabled=False)
                    if self._policy_head_fp32
                    else nullcontext()
                ):
                    policy_planet_emb = planet_emb.float() if self._policy_head_fp32 else planet_emb
                    policy_edge_emb = edge_emb.float() if self._policy_head_fp32 else edge_emb
                    policy_pair_valid = pair_valid.float() if self._policy_head_fp32 else pair_valid
                    (
                        policy_log_probs,
                        policy_head_aux,
                    ) = self._policy_head(
                        policy_planet_emb,
                        policy_edge_emb,
                        policy_pair_valid,
                        available_action_flat,
                        wall_profiler=wall_profiler,
                    )
                policy_action_classes = ORBIT_PLANET_ACTION_SLOTS * self._num_policy_actions
                assert policy_log_probs.shape == (b_tot, n, policy_action_classes)
            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_policy_logits_optional_reshape"
            ):
                if bool(include_final_policy_logits) or bool(include_policy_logits_pre_action_mask):
                    final_policy_logits_learn = {
                        "spawn_fleet": policy_head_aux["policy_logits"].reshape(
                            b_e,
                            b_p,
                            n,
                            policy_action_classes,
                        ),
                    }
                if bool(output_full_policy_log_probs):
                    floor_stats = policy_head_aux["entropy_floor_stats"]
                    assert isinstance(floor_stats, dict)
                    assert len(floor_stats) == 0, tuple(floor_stats.keys())
                    entropy_floor_stats_learn = {}

            baseline_learn: dict[str, torch.Tensor] | None = None
            with wall_tree_cuda_model_block(wall_profiler, device, "orbit_global_value_pool"):
                if self._include_rl_value_head and bool(include_value_head):
                    with (
                        torch.amp.autocast(device_type=planet_emb.device.type, enabled=False)
                        if self._value_head_fp32
                        else nullcontext()
                    ):
                        value_planet_emb = planet_emb.float() if self._value_head_fp32 else planet_emb
                        value_edge_emb = edge_emb.float() if self._value_head_fp32 else edge_emb
                        value_pair_valid = pair_valid.float() if self._value_head_fp32 else pair_valid
                        value_planet_slot_mask = (
                            planet_slot_mask.float()
                            if self._value_head_fp32
                            else planet_slot_mask
                        )
                        value_opponent_context = None
                        if self._use_value_opponent_model_embedding:
                            assert player_identity_emb is not None
                            value_opponent_context = self._value_opponent_context(
                                batch,
                                player_identity_emb,
                                enemy_valid_bool,
                                b_e=b_e,
                                b_p=b_p,
                                dtype=value_planet_emb.dtype,
                                device=device,
                            )
                        baseline_all = self._global_value_head(
                            value_planet_emb,
                            value_edge_emb,
                            value_pair_valid,
                            value_planet_slot_mask,
                            value_opponent_context,
                        )
                        assert baseline_all.shape == (b_tot,)
                        production_delta_all = self._global_value_head_production_delta(
                            value_planet_emb,
                            value_edge_emb,
                            value_pair_valid,
                            value_planet_slot_mask,
                            value_opponent_context,
                        )
                    baseline_flat = baseline_all
                    assert baseline_flat.shape == (b_tot,)
                    assert production_delta_all.shape == (b_tot,)
                    production_delta_flat = production_delta_all
                    assert production_delta_flat.shape == (b_tot,)
                    baseline_learn = {
                        "baseline": baseline_flat.reshape(b_e, b_p),
                        "production_delta": production_delta_flat.reshape(b_e, b_p),
                    }

            with wall_tree_cuda_model_block(
                wall_profiler, device, "orbit_spawn_action_and_policy_outputs"
            ):
                actions_flat = _choose_from_log_probs(
                    policy_log_probs.reshape(-1, policy_action_classes),
                    available_action_flat.reshape(-1, policy_action_classes),
                    sample=self._is_sample,
                    compile_friendly_sample=self._compile_friendly_sample,
                    label="spawn_fleet",
                ).reshape(b_tot, ORBIT_PLANET_ACTION_SLOTS, 1).to(dtype=torch.int64)
                action_dst = actions_flat // int(self._num_policy_actions)
                action_subindex = actions_flat % int(self._num_policy_actions)
                actions_env_flat = (
                    action_dst * int(ORBIT_MOVE_CLASSES_PER_TARGET) + action_subindex
                )
                actions_grid = actions_env_flat.reshape(b_e, b_p, ORBIT_PLANET_ACTION_SLOTS, 1)
                ac_learn = {"spawn_fleet": actions_grid}
                out_learn: dict[str, Any] = {"actions_LEARN": ac_learn}
                if baseline_learn is not None:
                    out_learn["baseline_LEARN"] = baseline_learn
                if len(final_policy_logits_learn) > 0:
                    out_learn["final_policy_logits_LEARN"] = final_policy_logits_learn
                if len(entropy_floor_stats_learn) > 0:
                    out_learn["entropy_floor_stats_LEARN"] = entropy_floor_stats_learn
                full_lp = bool(output_full_policy_log_probs)
                if full_lp:
                    out_learn["available_action_mask_LEARN"] = {
                        "spawn_fleet": available_action_mask
                    }
                    policy_log_probs_out = policy_log_probs.reshape(
                        b_e,
                        b_p,
                        ORBIT_PLANET_ACTION_SLOTS,
                        policy_action_classes,
                    )
                    out_learn["policy_log_probs_LEARN"] = {"spawn_fleet": policy_log_probs_out}
                else:
                    idx = actions_flat.to(dtype=torch.long)
                    assert idx.shape == (b_tot, n, 1)
                    taken_lp = torch.gather(policy_log_probs, -1, idx)
                    assert taken_lp.shape == (b_tot, n, 1)
                    behavior_sum_flat = taken_lp.sum(dim=1).squeeze(-1)
                    assert behavior_sum_flat.shape == (b_tot,)
                    behavior_sum = behavior_sum_flat.reshape(b_e, b_p)
                    out_learn["behavior_log_prob_sum_LEARN"] = {"spawn_fleet": behavior_sum}

            return out_learn
        assert False, "include_rl_policy_value_heads must be True"

    def set_entropy_floor_targets(self, floor_target: torch.Tensor) -> None:
        assert self._include_rl_policy_value_heads
        assert isinstance(self._policy_head, OrbitSimpleSpawnFleetPolicyHead)
        self._policy_head.set_entropy_floor_targets(floor_target)

    def entropy_floor_logging_stats(self) -> dict[str, float]:
        assert self._include_rl_policy_value_heads
        assert isinstance(self._policy_head, OrbitSimpleSpawnFleetPolicyHead)
        if False:
            target = self._policy_head._entropy_floor_target.detach().cpu()
            return {
                "target_min_entropy_spawn_fleet": float(target[0].item()),
            }
        return {}


def impala_orbit_model_init_kwargs_from_flags(flags) -> dict[str, Any]:
    m = flags.model
    # Training entrypoint nests dicts via ``types.SimpleNamespace``; YAML configs keep plain dicts.
    m_map = _cfg_mapping(m)
    oc = m_map["orbit_impala"]
    oc_map = _cfg_mapping(oc)
    missing = sorted(k for k in _IMPALA_ORBIT_MODEL_REQUIRED_CONFIG_KEYS if k not in oc_map)
    assert not missing, ("orbit_impala missing required model config keys", missing)
    hidden_dim = int(oc_map["hidden_dim"])
    use_bn = bool(oc_map["use_token_batch_norm"])
    obs_feature_normalization = oc_map["obs_feature_normalization"]
    enc_nl = int(oc_map["encoder_num_layers"])
    arr_enc_nl = int(oc_map["arrivals_encoder_num_layers"])
    arr_fuse_nl = int(oc_map["arrivals_fusion_num_layers"])
    dec_nl = int(oc_map["decoder_num_layers"])
    nl = int(oc_map["transformer_num_layers"])
    nh = int(oc_map["transformer_num_heads"])
    edge_endpoint_update = str(oc_map["edge_endpoint_update"])
    mlp_activation = str(
        oc_map.get("mlp_activation", _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["mlp_activation"])
    )
    transformer_activation = str(
        oc_map.get(
            "transformer_activation",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["transformer_activation"],
        )
    )
    residual_dropout = float(oc_map["residual_dropout"])
    ffn_multiplier = int(oc_map["ffn_multiplier"])
    ffn_dropout = float(oc_map["ffn_dropout"])
    use_obs_discrete_embeddings = bool(oc_map["use_obs_discrete_embeddings"])
    use_identity_embedding = bool(
        oc_map.get(
            "use_identity_embedding",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["use_identity_embedding"],
        )
    )
    use_edge_relation_features = bool(
        oc_map.get(
            "use_edge_relation_features",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["use_edge_relation_features"],
        )
    )
    use_arrival_attention_fusion = bool(
        oc_map.get(
            "use_arrival_attention_fusion",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["use_arrival_attention_fusion"],
        )
    )
    use_arrival_attention_fusion_gate = bool(
        oc_map.get(
            "use_arrival_attention_fusion_gate",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["use_arrival_attention_fusion_gate"],
        )
    )
    allow_self_edges = bool(
        oc_map.get(
            "allow_self_edges",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["allow_self_edges"],
        )
    )
    arrival_attention_num_queries = int(
        oc_map.get(
            "arrival_attention_num_queries",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["arrival_attention_num_queries"],
        )
    )
    zeroed_obs_feature_inputs = oc_map.get(
        "zeroed_obs_feature_inputs",
        _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["zeroed_obs_feature_inputs"],
    )
    arrival_temporal_horizon = int(
        oc_map.get(
            "arrival_temporal_horizon",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["arrival_temporal_horizon"],
        )
    )
    policy_head_fp32 = bool(
        oc_map.get(
            "policy_head_fp32",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["policy_head_fp32"],
        )
    )
    value_head_fp32 = bool(
        oc_map.get(
            "value_head_fp32",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["value_head_fp32"],
        )
    )
    use_value_opponent_model_embedding = bool(
        oc_map.get(
            "use_value_opponent_model_embedding",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["use_value_opponent_model_embedding"],
        )
    )
    if use_value_opponent_model_embedding:
        frozen_cfg = getattr(flags, "frozen_opponent")
        frozen_checkpoints = getattr(frozen_cfg, "checkpoints")
        frozen_checkpoint_count = int(len(frozen_checkpoints))
        assert frozen_checkpoint_count <= _VALUE_OPPONENT_MODEL_ID_CAPACITY, (
            frozen_checkpoint_count,
            _VALUE_OPPONENT_MODEL_ID_CAPACITY,
        )
        value_opponent_model_count = _VALUE_OPPONENT_MODEL_ID_CAPACITY
    else:
        value_opponent_model_count = int(
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["value_opponent_model_count"]
        )
    num_policy_actions = int(
        oc_map.get(
            "num_policy_actions",
            _IMPALA_ORBIT_MODEL_CONFIG_DEFAULTS["num_policy_actions"],
        )
    )
    assert 1 <= num_policy_actions <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        num_policy_actions,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    assert hasattr(flags, "target_min_entropy"), "flags.target_min_entropy is required"
    entropy_floor_config = getattr(flags, "target_min_entropy")
    entropy_floor_target = _entropy_head_values_from_config(
        entropy_floor_config,
        label="target_min_entropy",
    )
    entropy_floor_max_temperature = float(
        getattr(flags, "entropy_floor_max_temperature", 64.0)
    )
    entropy_floor_num_iters = int(getattr(flags, "entropy_floor_num_iters", 10))
    assert hidden_dim > 0
    return {
        "hidden_dim": hidden_dim,
        "use_token_batch_norm": use_bn,
        "obs_feature_normalization": obs_feature_normalization,
        "encoder_num_layers": enc_nl,
        "arrivals_encoder_num_layers": arr_enc_nl,
        "arrivals_fusion_num_layers": arr_fuse_nl,
        "decoder_num_layers": dec_nl,
        "transformer_num_layers": nl,
        "transformer_num_heads": nh,
        "edge_endpoint_update": edge_endpoint_update,
        "mlp_activation": mlp_activation,
        "transformer_activation": transformer_activation,
        "residual_dropout": residual_dropout,
        "ffn_multiplier": ffn_multiplier,
        "ffn_dropout": ffn_dropout,
        "use_obs_discrete_embeddings": use_obs_discrete_embeddings,
        "use_identity_embedding": use_identity_embedding,
        "use_edge_relation_features": use_edge_relation_features,
        "use_arrival_attention_fusion": use_arrival_attention_fusion,
        "arrival_attention_num_queries": arrival_attention_num_queries,
        "use_arrival_attention_fusion_gate": use_arrival_attention_fusion_gate,
        "allow_self_edges": allow_self_edges,
        "zeroed_obs_feature_inputs": zeroed_obs_feature_inputs,
        "arrival_temporal_horizon": arrival_temporal_horizon,
        "policy_head_fp32": policy_head_fp32,
        "value_head_fp32": value_head_fp32,
        "use_value_opponent_model_embedding": use_value_opponent_model_embedding,
        "value_opponent_model_count": value_opponent_model_count,
        "num_policy_actions": num_policy_actions,
        "entropy_floor_target": entropy_floor_target,
        "entropy_floor_max_temperature": entropy_floor_max_temperature,
        "entropy_floor_num_iters": entropy_floor_num_iters,
    }


def create_impala_model(flags) -> nn.Module:
    return ImpalaOrbitModel(**impala_orbit_model_init_kwargs_from_flags(flags))


ImpalaOrbitStubModel = ImpalaOrbitModel
