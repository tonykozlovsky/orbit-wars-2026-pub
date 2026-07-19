import math
import os
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.amp
import torch.nn as nn
import torch.optim

from ...configs.base import teacher_kl_cost_at_step
from ...gym.obs_wrapper import (
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_assert_available_action_mask_contract,
)
from ...gym.wall_tree_profiler import (
    WallTreeProfiler,
    model_profile_enabled,
    profiler_span,
)
from .buffer_utils import buffers_apply
from .common import (
    ENTROPY_HEAD_KEYS,
    POLICY_ACTION_KEYS,
    entropy_head_tuple_from_config,
    impala_compile_enabled,
    queue_put_or_stop,
)
from .losses_func_selfplay import (
    REWARD_HEAD_KEYS,
    _popart_get_norm,
    _teacher_kl_cost_scalar,
    losses_func_selfplay,
    selfplay_entropy_loss,
    selfplay_teacher_baseline_loss,
    selfplay_teacher_kl_losses,
)
from .stats import RollingAverage
from .terminal_vtrace_dump import maybe_dump_terminal_selfplay_vtrace_and_exit


rolling_grad = RollingAverage(window_size=1000)
rolling_all_loss = RollingAverage(window_size=1000)

# When teacher KL cost is zero, teacher forward is only needed for the `teacher_kl_loss_orig`
# metric; run it every N learner batches (not raw env steps: ``local_step`` jumps by
# ``batch_size * unroll_length`` per batch in ``batch_and_learn``).
TEACHER_KL_METRIC_SUBSAMPLE_EVERY = 10

# EMA state for other reward heads (per learner process, per head).
_reward_ema_state: dict[str, dict[str, float]] = {}


def _model_stat_key(name: str) -> str:
    return name.replace(".", "/")


def _scalar_tensor_debug_value(tensor: torch.Tensor) -> float:
    assert tensor.numel() == 1, (tuple(tensor.shape), tensor.dtype)
    return float(tensor.detach().to(dtype=torch.float32).cpu().item())


def _loss_components_message(loss_components: dict[str, torch.Tensor]) -> str:
    parts = []
    for loss_name, loss_value in loss_components.items():
        assert isinstance(loss_value, torch.Tensor), (loss_name, type(loss_value))
        parts.append(f"{loss_name}={_scalar_tensor_debug_value(loss_value)}")
    return " losses={" + ", ".join(parts) + "}"


def _floating_tensors_to_float32(buffers):
    return buffers_apply(
        buffers,
        lambda t: t.to(dtype=torch.float32) if torch.is_floating_point(t) else t,
    )


def _nonfinite_tensor_message(
    kind: str,
    name: str,
    tensor: torch.Tensor,
    loss_components: dict[str, torch.Tensor],
) -> str:
    finite = torch.isfinite(tensor)
    bad_flat_indices = torch.nonzero(~finite.reshape(-1), as_tuple=False)
    assert bad_flat_indices.numel() > 0, (kind, name, tuple(tensor.shape), tensor.dtype)
    first_bad_flat_index = int(bad_flat_indices[0, 0].item())
    first_bad_value = tensor.detach().reshape(-1)[first_bad_flat_index].cpu().item()
    nonfinite_count = int((~finite).sum().item())
    return (
        f"model {kind} is not finite: {name} "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"first_nonfinite_flat_index={first_bad_flat_index} "
        f"first_nonfinite_value={first_bad_value} "
        f"nonfinite_count={nonfinite_count}"
        f"{_loss_components_message(loss_components)}"
    )


def _assert_model_weights_and_grads_finite(
    model: nn.Module,
    loss_components: dict[str, torch.Tensor],
) -> None:
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all().item(), _nonfinite_tensor_message(
            "parameter",
            name,
            param,
            loss_components,
        )
        if param.grad is not None:
            assert torch.isfinite(param.grad).all().item(), _nonfinite_tensor_message(
                "gradient",
                name,
                param.grad,
                loss_components,
            )


def _add_model_weight_grad_stats(
    model: nn.Module,
    *,
    add_stat,
) -> None:
    for name, param in model.named_parameters():
        stat_name = _model_stat_key(name)
        param_f32 = param.detach().to(dtype=torch.float32)
        add_stat(f"model_param_mean/{stat_name}", param_f32.mean())
        add_stat(f"model_param_maxabs/{stat_name}", param_f32.abs().max())
        if param.grad is not None:
            grad_f32 = param.grad.detach().to(dtype=torch.float32)
            add_stat(f"model_grad_mean/{stat_name}", grad_f32.mean())
            add_stat(f"model_grad_maxabs/{stat_name}", grad_f32.abs().max())


def get_reward_ema_state() -> dict[str, dict[str, float]]:
    return {
        head: {
            'mean': float(state['mean']),
            'm2': float(state['m2']),
        }
        for head, state in _reward_ema_state.items()
    }


def load_reward_ema_state(state: dict) -> None:
    if not isinstance(state, dict):
        raise TypeError("reward_ema_state must be a dict")
    for head, head_state in state.items():
        if not isinstance(head_state, dict):
            raise TypeError(f"reward_ema_state[{head}] must be a dict")
        if 'mean' not in head_state or 'm2' not in head_state:
            raise KeyError(f"reward_ema_state[{head}] must include 'mean' and 'm2'")
        _reward_ema_state[str(head)] = {
            'mean': float(head_state['mean']),
            'm2': float(head_state['m2']),
        }


def _policy_logit_l2_losses(
    learner_outputs: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    logit_limit: float,
    compute_raw: bool,
    compute_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    limit = float(logit_limit)
    assert limit >= 0.0, logit_limit
    assert bool(compute_raw) or bool(compute_centered)
    assert POLICY_ACTION_KEYS == ("spawn_fleet",), POLICY_ACTION_KEYS
    final_logits_by_action = learner_outputs["final_policy_logits_LEARN"]
    assert isinstance(final_logits_by_action, dict)
    assert tuple(final_logits_by_action.keys()) == POLICY_ACTION_KEYS, (
        tuple(final_logits_by_action.keys()),
        POLICY_ACTION_KEYS,
    )
    available_masks = learner_outputs["available_action_mask_LEARN"]
    assert isinstance(available_masks, dict)
    assert tuple(available_masks.keys()) == POLICY_ACTION_KEYS, (
        tuple(available_masks.keys()),
        POLICY_ACTION_KEYS,
    )
    raw_total = torch.zeros((), device=device, dtype=dtype)
    centered_total = torch.zeros((), device=device, dtype=dtype)
    diagnostics: dict[str, torch.Tensor] = {}
    for head in ENTROPY_HEAD_KEYS:
        assert head == "spawn_fleet", head
        logits = final_logits_by_action[head]
        mask = available_masks[head]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert logits.shape == mask.shape, (head, tuple(logits.shape), tuple(mask.shape))
        assert torch.is_floating_point(logits), logits.dtype
        assert mask.dtype == torch.bool, mask.dtype
        logits_f = logits
        valid_f = mask.to(dtype=logits.dtype)
        valid_count = valid_f.sum(dim=-1)
        eligible = valid_count > 1.0
        eligible_f = eligible.to(dtype=logits.dtype)
        valid_count_clamped = valid_count.clamp(min=1.0)
        denom = eligible_f.sum().clamp(min=1.0)
        if bool(compute_raw):
            raw_excess = torch.clamp(logits_f.abs() - limit, min=0.0)
            raw_row = (raw_excess.square() * valid_f).sum(dim=-1) / valid_count_clamped
            raw_head = (raw_row * eligible_f).sum() / denom
            raw_total = raw_total + raw_head.to(dtype=dtype)
            diagnostics[f"policy_logits_l2_{head}"] = raw_head
            diagnostics[f"policy_logits_l2_sample_count_{head}"] = eligible_f.sum()
        if bool(compute_centered):
            valid_mean = (logits_f * valid_f).sum(dim=-1) / valid_count_clamped
            centered = torch.where(
                mask,
                logits_f - valid_mean.unsqueeze(-1),
                torch.zeros_like(logits_f),
            )
            centered_excess = torch.clamp(centered.abs() - limit, min=0.0)
            centered_row = centered_excess.square().sum(dim=-1) / valid_count_clamped
            centered_head = (centered_row * eligible_f).sum() / denom
            centered_total = centered_total + centered_head.to(dtype=dtype)
            diagnostics[f"policy_centered_logits_l2_{head}"] = centered_head
    if bool(compute_raw):
        diagnostics["policy_logits_l2"] = raw_total
    if bool(compute_centered):
        diagnostics["policy_centered_logits_l2"] = centered_total
    return raw_total, centered_total, diagnostics


def _masked_log_probs_from_logits_for_kl(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    assert logits.shape == mask.shape, (tuple(logits.shape), tuple(mask.shape))
    assert torch.is_floating_point(logits), logits.dtype
    assert mask.dtype == torch.bool, mask.dtype
    neg_large = torch.finfo(logits.dtype).min / 16.0
    return torch.log_softmax(logits.masked_fill(~mask, neg_large), dim=-1).masked_fill(
        ~mask,
        float("-inf"),
    )


def _temperature_compensation_kl_losses(
    learner_outputs: dict,
    batch: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    temperature_compensation_kl_costs: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert len(temperature_compensation_kl_costs) == len(ENTROPY_HEAD_KEYS)
    assert not any(float(cost) != 0.0 for cost in temperature_compensation_kl_costs), (
        "temperature compensation KL was defined for split policy heads and is not valid for "
        "the single spawn_fleet action space",
        temperature_compensation_kl_costs,
    )
    total = torch.zeros((), device=device, dtype=dtype)
    diagnostics: dict[str, torch.Tensor] = {}
    diagnostics["temperature_compensation_kl_loss_raw"] = total
    return total, diagnostics


def _ema_alpha_for_steps(steps: int, base_alpha: float, *, label: str) -> float:
    assert int(steps) > 0, f"{label} EMA steps must be > 0"
    base = float(base_alpha)
    assert 0.0 <= base <= 1.0, f"{label} EMA alpha per step must be in [0, 1]"
    return 1.0 - (1.0 - base) ** float(steps)


def _reward_ema_update_and_get_norm(
    head: str,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    eps: float,
    device: torch.device,
    update: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) tensors for EMA-normalizing reward heads."""
    if head not in _reward_ema_state:
        _reward_ema_state[head] = {'mean': 0.0, 'm2': 1.0}
    state = _reward_ema_state[head]
    with torch.no_grad():
        m = mask.to(dtype=rewards.dtype, device=rewards.device)
        denom = m.sum()
        if denom.item() > 0:
            batch_mean = (rewards * m).sum() / denom
            batch_mean2 = ((rewards * rewards) * m).sum() / denom

            if update:
                batch_mean_scalar = float(batch_mean.detach().cpu().item())
                batch_mean2_scalar = float(batch_mean2.detach().cpu().item())
                alpha_scalar = float(alpha)
                state['mean'] = (1.0 - alpha_scalar) * float(state['mean']) + alpha_scalar * batch_mean_scalar
                state['m2'] = (1.0 - alpha_scalar) * float(state['m2']) + alpha_scalar * batch_mean2_scalar

        mean = torch.tensor(float(state['mean']), device=device, dtype=torch.float32)
        var = max(float(state['m2']) - float(state['mean']) ** 2, float(eps))
        std = torch.tensor(var ** 0.5, device=device, dtype=torch.float32)
        return mean, std


def learn(
    device,
    flags: SimpleNamespace,
    learner_model: nn.Module,
    batch: dict[str, torch.Tensor],
    wall_profiler: WallTreeProfiler,
    optimizers: Optional[list[torch.optim.Optimizer]],
    lr_schedulers: Optional[list[torch.optim.lr_scheduler._LRScheduler]],
    shared_target_entropy=None,
    shared_shortfall_entropy=None,
    stats_queue_learner=None,
    stop_event=None,
    teacher_models_by_num_players=None,
    teacher_kl_cost_multiplier: float = 1.0,
    train: bool = True,
    local_step: int = 0,
    stats_override: Optional[dict[str, float]] = None,
    separate_value_baseline_learn: Optional[dict[str, torch.Tensor]] = None,
):
    return _learn_impl(
        device=device,
        flags=flags,
        learner_model=learner_model,
        batch=batch,
        wall_profiler=wall_profiler,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        shared_target_entropy=shared_target_entropy,
        shared_shortfall_entropy=shared_shortfall_entropy,
        stats_queue_learner=stats_queue_learner,
        stop_event=stop_event,
        teacher_models_by_num_players=teacher_models_by_num_players,
        teacher_kl_cost_multiplier=teacher_kl_cost_multiplier,
        train=train,
        local_step=local_step,
        stats_override=stats_override,
        separate_value_baseline_learn=separate_value_baseline_learn,
    )


def _learn_impl(
    device,
    flags: SimpleNamespace,
    learner_model: nn.Module,
    batch: dict[str, torch.Tensor],
    wall_profiler: WallTreeProfiler,
    optimizers: Optional[list[torch.optim.Optimizer]],
    lr_schedulers: Optional[list[torch.optim.lr_scheduler._LRScheduler]],
    shared_target_entropy=None,
    shared_shortfall_entropy=None,
    stats_queue_learner=None,
    stop_event=None,
    teacher_models_by_num_players=None,
    teacher_kl_cost_multiplier: float = 1.0,
    train: bool = True,
    local_step: int = 0,
    stats_override: Optional[dict[str, float]] = None,
    separate_value_baseline_learn: Optional[dict[str, torch.Tensor]] = None,
):
    learner_forward_bf16 = bool(flags.learner_forward_bf16)
    learner_loss_bf16 = bool(flags.learner_loss_bf16)
    learner_backward_bf16 = bool(flags.learner_backward_bf16)

    def _run_model_on_rollout(model, rollout_batch, *, include_value_head: bool):
        T = int(flags.rollout_time_steps)
        B = int(flags.batch_size)
        assert "obs_LEARN_INFER" in rollout_batch
        obs_rollout = rollout_batch["obs_LEARN_INFER"]
        assert isinstance(obs_rollout, dict)
        assert "player_mask" in obs_rollout
        player_mask = obs_rollout["player_mask"]
        assert isinstance(player_mask, torch.Tensor)
        assert tuple(int(x) for x in player_mask.shape) == (T, B, 1), (
            tuple(player_mask.shape),
            T,
            B,
        )
        P = 1
        tb = T * B
        with profiler_span(wall_profiler, "rollout_obs_flatten"):
            flat_obs = buffers_apply(
                obs_rollout,
                lambda t: t.reshape(tb, P, *tuple(t.shape[3:])),
            )
            flat_batch = {"obs_LEARN_INFER": flat_obs}
            if "frozen_model_by_player_axis_LEARN" in rollout_batch:
                frozen_model_by_player_axis = rollout_batch[
                    "frozen_model_by_player_axis_LEARN"
                ]
                assert isinstance(frozen_model_by_player_axis, torch.Tensor)
                assert tuple(frozen_model_by_player_axis.shape) == (
                    T,
                    B,
                    P,
                    ORBIT_PLAYER_AXIS_SLOTS,
                ), tuple(frozen_model_by_player_axis.shape)
                flat_batch["frozen_model_by_player_axis_LEARN"] = (
                    frozen_model_by_player_axis.reshape(
                        tb,
                        P,
                        ORBIT_PLAYER_AXIS_SLOTS,
                    )
                )
        orbit_assert_available_action_mask_contract(
            flat_obs["available_action_mask"],
            label="learner",
        )
        if model_profile_enabled():
            with profiler_span(wall_profiler, "pre_forward_synchronize"):
                torch.cuda.synchronize(device)

        with profiler_span(wall_profiler, "rollout_model_forward"):
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=learner_forward_bf16,
            ):
                forward_out = model(
                    flat_batch,
                    output_full_policy_log_probs=True,
                    include_policy_logits_pre_action_mask=True,
                    include_value_head=bool(include_value_head),
                    wall_profiler=wall_profiler,
                )
        with profiler_span(wall_profiler, "rollout_outputs_reshape"):
            out_rollout = buffers_apply(
                forward_out,
                lambda t: t.reshape(T, B, P, *tuple(t.shape[2:])),
            )
        return out_rollout

    assert shared_shortfall_entropy is not None
    entropy_floor_target = torch.tensor(
        [float(shared_shortfall_entropy[i]) for i in range(len(ENTROPY_HEAD_KEYS))],
        device=device,
        dtype=torch.float32,
    )
    learner_model.set_entropy_floor_targets(
        entropy_floor_target,
    )
    expected_time = int(flags.unroll_length) + 1
    expected_batch = int(flags.batch_size)
    teacher_outputs_by_num_players = None
    fixed_teacher = bool(flags.teacher)
    teacher_present = fixed_teacher
    teacher_kl_cost: float | dict[str, float] = 0.0
    teacher_baseline_cost = 0.0
    if fixed_teacher:
        teacher_kl_cost = teacher_kl_cost_at_step(
            flags.teacher.kl_cost_initial,
            int(flags.teacher.kl_cost_decay_steps),
            local_step,
        )
        teacher_baseline_cost = float(flags.teacher.baseline_cost)
        assert teacher_baseline_cost >= 0.0, teacher_baseline_cost
        teacher_kl_cost_multiplier = float(teacher_kl_cost_multiplier)
        assert 0.0 <= teacher_kl_cost_multiplier <= 1.0, teacher_kl_cost_multiplier
        teacher_baseline_cost = teacher_baseline_cost * teacher_kl_cost_multiplier
        if isinstance(teacher_kl_cost, dict):
            teacher_kl_cost = {
                k: float(v) * teacher_kl_cost_multiplier
                for k, v in teacher_kl_cost.items()
            }
        else:
            teacher_kl_cost = float(teacher_kl_cost) * teacher_kl_cost_multiplier
    teacher_infer_this_step = False
    if teacher_present:
        kl_c = _teacher_kl_cost_scalar(teacher_kl_cost)
        batch_delta_steps = int(flags.batch_size) * int(flags.unroll_length)
        assert batch_delta_steps > 0
        learner_batch_index = int(local_step) // batch_delta_steps
        teacher_infer_this_step = (kl_c != 0.0) or (teacher_baseline_cost != 0.0) or (
            learner_batch_index % int(TEACHER_KL_METRIC_SUBSAMPLE_EVERY) == 0
        )
        if teacher_infer_this_step:
            assert isinstance(teacher_models_by_num_players, dict), (
                "Teacher models are not initialized"
            )
            teacher_outputs_by_num_players = {}
            with profiler_span(wall_profiler, "teacher_forward"):
                with torch.no_grad():
                    game_num_players = batch["game_num_players_LEARN"]
                    assert isinstance(game_num_players, torch.Tensor)
                    assert game_num_players.dtype == torch.int64, game_num_players.dtype
                    player_mask = batch["obs_LEARN_INFER"]["player_mask"]
                    assert isinstance(player_mask, torch.Tensor)
                    for teacher_scope, teacher_model in sorted(
                        teacher_models_by_num_players.items(),
                        key=lambda item: -1 if item[0] is None else int(item[0]),
                    ):
                        if teacher_scope is not None:
                            teacher_scope = int(teacher_scope)
                            assert teacher_scope in (2, 4), teacher_scope
                        teacher_model.set_entropy_floor_targets(
                            entropy_floor_target,
                        )
                        if teacher_scope is None:
                            teacher_player_mask = player_mask
                        else:
                            teacher_player_mask = player_mask * (
                                game_num_players == teacher_scope
                            ).to(
                                device=player_mask.device,
                                dtype=player_mask.dtype,
                            )
                        if not bool((teacher_player_mask > 0.5).any().item()):
                            continue
                        teacher_obs = dict(batch["obs_LEARN_INFER"])
                        teacher_obs["player_mask"] = teacher_player_mask
                        teacher_batch = {**batch, "obs_LEARN_INFER": teacher_obs}
                        teacher_output = _run_model_on_rollout(
                            teacher_model,
                            teacher_batch,
                            include_value_head=teacher_baseline_cost != 0.0,
                        )
                        if teacher_scope is None:
                            active_games = game_num_players[teacher_player_mask > 0.5]
                            assert int(active_games.numel()) > 0, tuple(
                                game_num_players.shape
                            )
                            for active_num_players in torch.unique(
                                active_games.detach().cpu(),
                                sorted=True,
                            ).tolist():
                                active_num_players = int(active_num_players)
                                assert active_num_players in (2, 4), active_num_players
                                assert active_num_players not in teacher_outputs_by_num_players, (
                                    active_num_players
                                )
                                teacher_outputs_by_num_players[active_num_players] = teacher_output
                        else:
                            assert teacher_scope not in teacher_outputs_by_num_players, (
                                teacher_scope
                            )
                            teacher_outputs_by_num_players[teacher_scope] = teacher_output
            for num_players, teacher_output in teacher_outputs_by_num_players.items():
                for action_space_key in POLICY_ACTION_KEYS:
                    assert action_space_key in teacher_output['policy_log_probs_LEARN'], (
                        "Teacher requires policy_log_probs_LEARN['"
                        f"{action_space_key}']"
                    )
                    teacher_lp = teacher_output['policy_log_probs_LEARN'][action_space_key]
                    assert isinstance(teacher_lp, torch.Tensor) and teacher_lp.ndim >= 2, (
                        "Teacher policy log_probs must be [T, B, ...], "
                        f"got {type(teacher_lp)} shape={tuple(teacher_lp.shape) if isinstance(teacher_lp, torch.Tensor) else None}"
                    )
                    assert teacher_lp.ndim == 5
                    assert int(teacher_lp.shape[2]) == 1, tuple(teacher_lp.shape)
                    assert int(teacher_lp.shape[3]) == int(ORBIT_PLANET_ACTION_SLOTS)
                    assert int(teacher_lp.shape[4]) % int(ORBIT_PLANET_ACTION_SLOTS) == 0, (
                        tuple(teacher_lp.shape),
                        ORBIT_PLANET_ACTION_SLOTS,
                    )
                    teacher_actions_per_target = int(teacher_lp.shape[4]) // int(ORBIT_PLANET_ACTION_SLOTS)
                    assert 1 <= teacher_actions_per_target <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
                        teacher_actions_per_target,
                        ORBIT_MOVE_CLASSES_PER_TARGET,
                    )
                    assert int(teacher_lp.shape[0]) == expected_time and int(teacher_lp.shape[1]) == expected_batch, (
                        "Teacher policy_log_probs_LEARN must be [T, B, ...] with "
                        f"T=unroll_length+1 ({expected_time}), B=batch_size ({expected_batch}); "
                        f"teacher_num_players={num_players} got shape={tuple(teacher_lp.shape)}"
                    )
                if teacher_baseline_cost != 0.0:
                    assert 'baseline_LEARN' in teacher_output, (
                        "Teacher requires baseline_LEARN when teacher.baseline_cost is non-zero"
                    )
                    teacher_baselines = teacher_output['baseline_LEARN']
                    assert isinstance(teacher_baselines, dict), type(teacher_baselines)
                    assert tuple(teacher_baselines.keys()) == REWARD_HEAD_KEYS, (
                        tuple(teacher_baselines.keys()),
                        REWARD_HEAD_KEYS,
                    )
                    teacher_baseline = teacher_baselines["baseline"]
                    assert isinstance(teacher_baseline, torch.Tensor)
                    assert tuple(int(x) for x in teacher_baseline.shape) == (
                        expected_time,
                        expected_batch,
                        1,
                    ), tuple(teacher_baseline.shape)
    with profiler_span(wall_profiler, "learner_forward"):
        learner_outputs = _run_model_on_rollout(learner_model, batch, include_value_head=True)
    assert 'policy_log_probs_LEARN' in learner_outputs, (
        "Learner requires policy_log_probs_LEARN in model outputs"
    )
    for action_space_key in POLICY_ACTION_KEYS:
        assert action_space_key in learner_outputs['policy_log_probs_LEARN'], (
            "Learner requires policy_log_probs_LEARN['"
            f"{action_space_key}']"
        )
        learner_lp = learner_outputs['policy_log_probs_LEARN'][action_space_key]
        assert isinstance(learner_lp, torch.Tensor) and learner_lp.ndim >= 2, (
            "Learner policy log_probs must be [T, B, ...], "
            f"got {type(learner_lp)} shape={tuple(learner_lp.shape) if isinstance(learner_lp, torch.Tensor) else None}"
        )
        assert learner_lp.ndim == 5
        assert int(learner_lp.shape[2]) == 1, tuple(learner_lp.shape)
        assert int(learner_lp.shape[3]) == int(ORBIT_PLANET_ACTION_SLOTS)
        assert int(learner_lp.shape[4]) % int(ORBIT_PLANET_ACTION_SLOTS) == 0, (
            tuple(learner_lp.shape),
            ORBIT_PLANET_ACTION_SLOTS,
        )
        learner_actions_per_target = int(learner_lp.shape[4]) // int(ORBIT_PLANET_ACTION_SLOTS)
        assert 1 <= learner_actions_per_target <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
            learner_actions_per_target,
            ORBIT_MOVE_CLASSES_PER_TARGET,
        )
        assert int(learner_lp.shape[0]) == expected_time and int(learner_lp.shape[1]) == expected_batch, (
            "Learner policy_log_probs_LEARN must be [T, B, ...] with "
            f"T=unroll_length+1 ({expected_time}), B=batch_size ({expected_batch}); "
            f"got shape={tuple(learner_lp.shape)}"
        )
    assert 'baseline_LEARN' in learner_outputs, (
        "Learner requires baseline_LEARN in model outputs"
    )
    if separate_value_baseline_learn is not None:
        learner_baseline_seq_by_head = learner_outputs['baseline_LEARN']
        assert isinstance(learner_baseline_seq_by_head, dict), (
            type(learner_baseline_seq_by_head)
        )
        assert tuple(learner_baseline_seq_by_head.keys()) == REWARD_HEAD_KEYS, (
            tuple(learner_baseline_seq_by_head.keys()),
            REWARD_HEAD_KEYS,
        )
        assert tuple(separate_value_baseline_learn.keys()) == REWARD_HEAD_KEYS, (
            tuple(separate_value_baseline_learn.keys()),
            REWARD_HEAD_KEYS,
        )
        for head in REWARD_HEAD_KEYS:
            learner_baseline_seq = learner_baseline_seq_by_head[head]
            separate_baseline_seq = separate_value_baseline_learn[head]
            assert isinstance(learner_baseline_seq, torch.Tensor)
            assert isinstance(separate_baseline_seq, torch.Tensor)
            assert tuple(separate_baseline_seq.shape) == tuple(learner_baseline_seq.shape), (
                head,
                tuple(separate_baseline_seq.shape),
                tuple(learner_baseline_seq.shape),
            )
        learner_outputs['baseline_LEARN'] = separate_value_baseline_learn
    baseline_seq_by_head = learner_outputs['baseline_LEARN']
    assert isinstance(baseline_seq_by_head, dict), type(baseline_seq_by_head)
    assert tuple(baseline_seq_by_head.keys()) == REWARD_HEAD_KEYS, (
        tuple(baseline_seq_by_head.keys()),
        REWARD_HEAD_KEYS,
    )
    for head, baseline_seq in baseline_seq_by_head.items():
        assert isinstance(baseline_seq, torch.Tensor) and baseline_seq.ndim >= 3, (
            "Learner requires baseline_LEARN head with time, batch, and player dims, "
            f"head={head!r} got {type(baseline_seq)} "
            f"shape={tuple(baseline_seq.shape) if isinstance(baseline_seq, torch.Tensor) else None}"
        )
        assert (
            int(baseline_seq.shape[0]) == expected_time
            and int(baseline_seq.shape[1]) == expected_batch
            and int(baseline_seq.shape[2]) == 1
        ), (
            "Learner baseline_LEARN head must be [T, B, 1] with "
            f"T=unroll_length+1 ({expected_time}), B=batch_size ({expected_batch}); "
            f"head={head!r} got shape={tuple(baseline_seq.shape)}"
        )
    with profiler_span(wall_profiler, "batch_align_bootstrap_reward"):
        pm_full = batch["obs_LEARN_INFER"]["player_mask"]
        assert isinstance(pm_full, torch.Tensor) and pm_full.ndim == 3
        assert int(pm_full.shape[0]) == expected_time
        assert int(pm_full.shape[1]) == expected_batch
        assert int(pm_full.shape[2]) == 1
        bootstrap_value = {
            head: values[-1] * pm_full[-1].to(device=values.device, dtype=values.dtype)
            for head, values in baseline_seq_by_head.items()
        }

        batch_aligned_with_learner = batch
        batch_aligned_with_learner = buffers_apply(
            batch_aligned_with_learner, lambda x: x[:-1]
        )

        learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])

        if teacher_present and teacher_outputs_by_num_players is not None:
            teacher_outputs_by_num_players = {
                int(num_players): buffers_apply(teacher_output, lambda x: x[:-1])
                for num_players, teacher_output in teacher_outputs_by_num_players.items()
            }

        # batch_aligned_with_learner: time t matches s_t, learner head outputs, and reward for the
        # transition s_t,a_t->s_{t+1} (see act._learn_payload: reward from env_after with obs_before).

        if bool(flags.enable_reward_ema_norm):
            reward_learn_stat = batch_aligned_with_learner['reward_LEARN_STAT']
            assert isinstance(reward_learn_stat, dict)
            assert tuple(reward_learn_stat.keys()) == REWARD_HEAD_KEYS, (
                tuple(reward_learn_stat.keys()),
                REWARD_HEAD_KEYS,
            )
            ema_eps = float(flags.reward_ema_eps)
            reward_mask = batch_aligned_with_learner["obs_LEARN_INFER"]["player_mask"]
            for head, reward in reward_learn_stat.items():
                assert isinstance(reward, torch.Tensor)
                steps_in_batch = int(reward.shape[0]) * int(reward.shape[1])
                ema_alpha = _ema_alpha_for_steps(
                    steps_in_batch,
                    float(flags.reward_ema_alpha),
                    label=f"reward_{head}",
                )
                assert reward_mask.shape == reward.shape, (
                    f"reward/mask shape mismatch for {head!r}: "
                    f"{tuple(reward.shape)} vs {tuple(reward_mask.shape)}"
                )
                reward_ema_mean_tensor, reward_ema_std_tensor = _reward_ema_update_and_get_norm(
                    head,
                    reward,
                    reward_mask,
                    alpha=ema_alpha,
                    eps=ema_eps,
                    device=torch.device(device),
                    update=bool(train) and (not bool(flags.lock_reward_ema)),
                )
                reward_learn_stat[head] = (
                    (reward - reward_ema_mean_tensor) / reward_ema_std_tensor
                ).to(dtype=reward.dtype)

    if train:
        assert optimizers is not None, "Optimizer must be provided when train=True"
        with profiler_span(wall_profiler, "zero_grad"):
            for optimizer in optimizers:
                optimizer.zero_grad()

    enable_clip_grad = bool(flags.enable_clip_grad)

    entropy_diagnostics: dict = {}

    assert "obs_LEARN_INFER" in batch_aligned_with_learner
    assert "available_action_mask" in batch_aligned_with_learner["obs_LEARN_INFER"]
    assert "player_mask" in batch_aligned_with_learner["obs_LEARN_INFER"]
    assert "behavior_log_prob_sum_LEARN" in batch_aligned_with_learner, (
        "Rollout batch must contain behavior_log_prob_sum_LEARN (actor policy log-prob sums)"
    )
    expected_transition_time = int(flags.unroll_length)
    for action_space_key in POLICY_ACTION_KEYS:
        assert action_space_key in batch_aligned_with_learner["behavior_log_prob_sum_LEARN"], (
            "behavior_log_prob_sum_LEARN must include '"
            f"{action_space_key}'"
        )
        bsum = batch_aligned_with_learner["behavior_log_prob_sum_LEARN"][action_space_key]
        assert isinstance(bsum, torch.Tensor) and bsum.ndim == 3, (
            "behavior_log_prob_sum_LEARN must be [T,B,1], "
            f"got {type(bsum)} shape={tuple(bsum.shape) if isinstance(bsum, torch.Tensor) else None}"
        )
        assert int(bsum.shape[0]) == expected_transition_time and int(bsum.shape[1]) == expected_batch, (
            "behavior_log_prob_sum_LEARN must match [unroll_length, batch_size, 1]; "
            f"got {tuple(bsum.shape)} expected T={expected_transition_time} B={expected_batch}"
        )
        assert int(bsum.shape[2]) == 1, tuple(bsum.shape)
    assert "action_taken_index_LEARN_STAT" in batch_aligned_with_learner, (
        "Learner batch must contain action_taken_index_LEARN_STAT"
    )
    batch_for_vtrace = dict(batch_aligned_with_learner)
    if not learner_loss_bf16:
        learner_outputs = _floating_tensors_to_float32(learner_outputs)
        bootstrap_value = _floating_tensors_to_float32(bootstrap_value)
        batch_for_vtrace = _floating_tensors_to_float32(batch_for_vtrace)
        if teacher_outputs_by_num_players is not None:
            teacher_outputs_by_num_players = _floating_tensors_to_float32(
                teacher_outputs_by_num_players
            )

    maybe_dump_terminal_selfplay_vtrace_and_exit(
        flags=flags,
        batch=batch_for_vtrace,
        learner_outputs=learner_outputs,
        bootstrap_value=bootstrap_value,
        player_mask_time_major=pm_full,
        local_step=int(local_step),
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=learner_loss_bf16):
        with profiler_span(wall_profiler, "vtrace_selfplay_losses"):
            (
                vtrace_pg_loss,
                upgo_pg_loss,
                upgo_original_pg_loss,
                baseline_loss,
                selfplay_diagnostics,
                baseline_loss_terminal,
            ) = losses_func_selfplay(
                rank=device,
                flags=flags,
                batch=batch_for_vtrace,
                learner_outputs=learner_outputs,
                bootstrap_value=bootstrap_value,
                player_mask_time_major=pm_full,
                collect_diagnostics=True,
                loss_weight=None,
                update_popart=bool(train) and (not bool(flags.lock_popart)),
            )

    batch_for_policy_aux = dict(batch_for_vtrace)

    teacher_kl_loss = None
    teacher_kl_loss_orig = None
    teacher_kl_loss_orig_2p = None
    teacher_kl_loss_orig_4p = None
    teacher_baseline_loss = None
    teacher_baseline_loss_orig = None
    teacher_baseline_explained_variance = None
    diagnostics_detached = {}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=learner_loss_bf16):
        with profiler_span(wall_profiler, "entropy_teacher_warmup_total"):
            for key, value in selfplay_diagnostics.items():
                detached_value = value.detach() if torch.is_tensor(value) else value
                if (
                    (key.startswith("vtrace") or key.startswith("upgo") or key.startswith("td"))
                    and ("loss" not in key)
                ):
                    diagnostics_detached[f"Diagnostic_{key}"] = detached_value
                else:
                    diagnostics_detached[key] = detached_value

            if bool(flags.classic_entropy) or bool(flags.ce_entropy):
                entropy_loss, entropy_diagnostics = selfplay_entropy_loss(
                    flags=flags,
                    batch=batch_for_policy_aux,
                    learner_outputs=learner_outputs,
                    shared_shortfall_entropy=shared_shortfall_entropy,
                )
            else:
                with torch.no_grad():
                    entropy_loss, entropy_diagnostics = selfplay_entropy_loss(
                        flags=flags,
                        batch=batch_for_policy_aux,
                        learner_outputs=learner_outputs,
                        shared_shortfall_entropy=shared_shortfall_entropy,
                    )
            policy_logits_l2_cost = float(flags.policy_logits_l2_cost)
            policy_centered_logits_l2_cost = float(flags.policy_centered_logits_l2_cost)
            policy_logits_l2_active = policy_logits_l2_cost != 0.0
            policy_centered_logits_l2_active = policy_centered_logits_l2_cost != 0.0
            policy_logits_l2_loss = torch.zeros_like(baseline_loss)
            policy_centered_logits_l2_loss = torch.zeros_like(baseline_loss)
            policy_logit_l2_diagnostics: dict[str, torch.Tensor] = {}
            if policy_logits_l2_active or policy_centered_logits_l2_active:
                policy_logits_l2_raw, policy_centered_logits_l2_raw, policy_logit_l2_diagnostics = (
                    _policy_logit_l2_losses(
                        learner_outputs,
                        device=device,
                        dtype=baseline_loss.dtype,
                        logit_limit=float(flags.logit_limit),
                        compute_raw=policy_logits_l2_active,
                        compute_centered=policy_centered_logits_l2_active,
                    )
                )
                if policy_logits_l2_active:
                    policy_logits_l2_loss = policy_logits_l2_cost * policy_logits_l2_raw
                if policy_centered_logits_l2_active:
                    policy_centered_logits_l2_loss = (
                        policy_centered_logits_l2_cost * policy_centered_logits_l2_raw
                    )
            temperature_compensation_kl_costs = entropy_head_tuple_from_config(
                flags.temperature_compensation_kl_cost,
                label="temperature_compensation_kl_cost",
            )
            temperature_compensation_active = any(
                float(cost) != 0.0 for cost in temperature_compensation_kl_costs
            )
            temperature_compensation_kl_loss = torch.zeros_like(baseline_loss)
            temperature_compensation_kl_diagnostics: dict[str, torch.Tensor] = {}
            if temperature_compensation_active:
                (
                    temperature_compensation_kl_raw,
                    temperature_compensation_kl_diagnostics,
                ) = _temperature_compensation_kl_losses(
                    learner_outputs,
                    batch_for_policy_aux,
                    device=device,
                    dtype=baseline_loss.dtype,
                    temperature_compensation_kl_costs=temperature_compensation_kl_costs,
                )
                for head_idx, head in enumerate(ENTROPY_HEAD_KEYS):
                    head_cost = float(temperature_compensation_kl_costs[head_idx])
                    if head_cost == 0.0:
                        continue
                    head_raw = temperature_compensation_kl_diagnostics[
                        f"temperature_compensation_kl_loss_{head}"
                    ]
                    temperature_compensation_kl_loss = temperature_compensation_kl_loss + (
                        head_cost * head_raw
                    )

            if teacher_present:
                if teacher_outputs_by_num_players is not None:
                    teacher_kl_components = selfplay_teacher_kl_losses(
                        flags=flags,
                        batch=batch_for_policy_aux,
                        learner_outputs=learner_outputs,
                        teacher_outputs_by_num_players=teacher_outputs_by_num_players,
                        teacher_kl_cost=teacher_kl_cost,
                    )
                    teacher_kl_loss = teacher_kl_components['teacher_kl_loss']
                    teacher_kl_loss_orig = teacher_kl_components['teacher_kl_loss_orig']
                    teacher_kl_loss_orig_2p = teacher_kl_components['teacher_kl_loss_orig_2p']
                    teacher_kl_loss_orig_4p = teacher_kl_components['teacher_kl_loss_orig_4p']
                    if teacher_baseline_cost != 0.0:
                        teacher_baseline_components = selfplay_teacher_baseline_loss(
                            flags=flags,
                            batch=batch_for_policy_aux,
                            learner_outputs=learner_outputs,
                            teacher_outputs_by_num_players=teacher_outputs_by_num_players,
                        )
                        teacher_baseline_loss = teacher_baseline_components['teacher_baseline_loss']
                        teacher_baseline_loss_orig = teacher_baseline_components[
                            'teacher_baseline_loss_orig'
                        ]
                        teacher_baseline_explained_variance = teacher_baseline_components[
                            'teacher_baseline_explained_variance'
                        ]
                else:
                    assert _teacher_kl_cost_scalar(teacher_kl_cost) == 0.0
                    assert teacher_baseline_cost == 0.0
                    teacher_kl_loss = torch.zeros_like(baseline_loss)
                    teacher_kl_loss_orig = None

            warmup_active = (int(flags.warmup_steps) > 0) and (
                int(local_step) < int(flags.warmup_steps)
            )
            warmup_scale = 0.0 if warmup_active else 1.0
            warmup_multiplier = torch.tensor(
                warmup_scale, device=device, dtype=baseline_loss.dtype
            )
            vtrace_pg_loss = vtrace_pg_loss * warmup_multiplier
            upgo_pg_loss = upgo_pg_loss * warmup_multiplier
            upgo_original_pg_loss = upgo_original_pg_loss * warmup_multiplier
            baseline_loss_terminal = baseline_loss_terminal * warmup_multiplier
            entropy_loss = entropy_loss * warmup_multiplier
            policy_logits_l2_loss = policy_logits_l2_loss * warmup_multiplier
            policy_centered_logits_l2_loss = policy_centered_logits_l2_loss * warmup_multiplier
            temperature_compensation_kl_loss = temperature_compensation_kl_loss * warmup_multiplier
            teacher_kl_term = torch.zeros_like(baseline_loss)
            teacher_baseline_term = torch.zeros_like(baseline_loss)
            if teacher_present:
                assert teacher_kl_loss is not None
                teacher_kl_term = teacher_kl_loss
                if teacher_baseline_cost != 0.0:
                    assert teacher_baseline_loss is not None
                    teacher_baseline_term = teacher_baseline_loss
            # Single scalar for backward(); RL partials from losses_func_selfplay, aux terms below.
            backward_loss_components = {
                "baseline_loss": baseline_loss,
                "baseline_loss_terminal": baseline_loss_terminal,
                "teacher_kl_term": teacher_kl_term,
                "teacher_baseline_term": teacher_baseline_term,
                "vtrace_pg_loss": vtrace_pg_loss,
                "upgo_pg_loss": upgo_pg_loss,
                "upgo_original_pg_loss": upgo_original_pg_loss,
                "entropy_loss": entropy_loss,
                "policy_logits_l2_loss": policy_logits_l2_loss,
                "policy_centered_logits_l2_loss": policy_centered_logits_l2_loss,
                "temperature_compensation_kl_loss": temperature_compensation_kl_loss,
            }
            total_loss = torch.zeros_like(baseline_loss)
            for loss_component_value in backward_loss_components.values():
                total_loss = total_loss + loss_component_value
            loss_components_for_backward = {
                "total_loss": total_loss,
                **backward_loss_components,
            }
    if train:
        #with profiler_span(wall_profiler, "before_backward_sync"):
        #    torch.cuda.synchronize(device)
        with profiler_span(wall_profiler, "finite_model_check_before"):
            _assert_model_weights_and_grads_finite(
                learner_model,
                loss_components_for_backward,
            )

        if total_loss.requires_grad:
            with profiler_span(wall_profiler, "backward"):
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=learner_backward_bf16,
                ):
                    total_loss.backward()

            with profiler_span(wall_profiler, "finite_model_check_after"):
                _assert_model_weights_and_grads_finite(
                    learner_model,
                    loss_components_for_backward,
                )
        #with profiler_span(wall_profiler, "after_backward_sync"):
        #    torch.cuda.synchronize(device)

    clip_grad_value = 0.0
    post_clip_grad_norm_scalar = 0.0
    with torch.no_grad():
        if train:
            with profiler_span(wall_profiler, "grad_clip"):
                if enable_clip_grad:
                    clip_grad_value = min(rolling_grad.average() * 1.5, 10000.0)
                    if clip_grad_value == 0.0:
                        clip_grad_value = 10.0
                    #clip_grad_value = 50.0
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        learner_model.parameters(), clip_grad_value
                    )
                    post_clip_grad_norm_scalar = total_norm.item()
                    if not math.isnan(post_clip_grad_norm_scalar):
                        rolling_grad.add(post_clip_grad_norm_scalar)
            with profiler_span(wall_profiler, "optimizer_step"):
                for optimizer in optimizers:
                    optimizer.step()

        with profiler_span(wall_profiler, "stats_enqueue"):
            stats = {}

            def add_stat(stat_key, stat_value):
                if stat_value is None:
                    return
                if torch.is_tensor(stat_value):
                    stats[stat_key] = stat_value.detach().cpu().item()
                else:
                    stats[stat_key] = float(stat_value)

            model_stats_frequency = int(flags.logging_config.model_stats_frequency)
            assert model_stats_frequency >= 0, model_stats_frequency
            model_stats_batch_delta_steps = int(flags.batch_size) * int(flags.unroll_length)
            assert model_stats_batch_delta_steps > 0
            model_stats_batch_index = int(local_step) // model_stats_batch_delta_steps
            should_add_model_stats = (
                bool(train)
                and model_stats_frequency > 0
                and (model_stats_batch_index % model_stats_frequency == 0)
            )
            if should_add_model_stats:
                _add_model_weight_grad_stats(learner_model, add_stat=add_stat)

            if flags.enable_popart:
                for reward_head in REWARD_HEAD_KEYS:
                    mean_head, std_head = _popart_get_norm(reward_head, device)
                    add_stat(f'popart_mean_{reward_head}', mean_head)
                    add_stat(f'popart_std_{reward_head}', std_head)

            if bool(flags.enable_reward_ema_norm):
                ema_eps = float(flags.reward_ema_eps)
                for reward_head in REWARD_HEAD_KEYS:
                    state = _reward_ema_state.get(reward_head)
                    if state is not None:
                        reward_ema_mean = float(state['mean'])
                        reward_ema_var = max(
                            float(state['m2']) - reward_ema_mean ** 2,
                            ema_eps,
                        )
                        add_stat(f'reward_ema_mean_{reward_head}', reward_ema_mean)
                        add_stat(f'reward_ema_std_{reward_head}', reward_ema_var ** 0.5)

            add_stat('teacher_kl_loss', teacher_kl_loss)
            add_stat('teacher_kl_loss_orig', teacher_kl_loss_orig)
            add_stat('teacher_kl_loss_orig_2p', teacher_kl_loss_orig_2p)
            add_stat('teacher_kl_loss_orig_4p', teacher_kl_loss_orig_4p)
            add_stat('teacher_baseline_loss', teacher_baseline_loss)
            add_stat('teacher_baseline_loss_orig', teacher_baseline_loss_orig)
            add_stat(
                'teacher_baseline_explained_variance',
                teacher_baseline_explained_variance,
            )
            if policy_logits_l2_active:
                add_stat('policy_logits_l2_loss', policy_logits_l2_loss)
                add_stat('policy_logits_l2_raw', policy_logits_l2_raw)
            if policy_centered_logits_l2_active:
                add_stat('policy_centered_logits_l2_loss', policy_centered_logits_l2_loss)
                add_stat('policy_centered_logits_l2_raw', policy_centered_logits_l2_raw)
            if temperature_compensation_active:
                add_stat('temperature_compensation_kl_loss', temperature_compensation_kl_loss)
                add_stat(
                    'temperature_compensation_kl_loss_raw',
                    temperature_compensation_kl_raw,
                )
            add_stat('total_loss', total_loss)
            game_num_players = batch_aligned_with_learner["game_num_players_LEARN"]
            assert isinstance(game_num_players, torch.Tensor)
            assert game_num_players.dtype == torch.int64, game_num_players.dtype
            assert tuple(int(x) for x in game_num_players.shape) == (
                int(flags.unroll_length),
                int(flags.batch_size),
                1,
            ), tuple(game_num_players.shape)
            learner_2p_samples = (game_num_players == 2).sum()
            learner_4p_samples = (game_num_players == 4).sum()
            learner_total_game_samples = learner_2p_samples + learner_4p_samples
            learner_4p_sample_ratio = (
                learner_4p_samples.to(dtype=torch.float32)
                / learner_total_game_samples.to(dtype=torch.float32)
            )
            add_stat('learner_4p_sample_ratio', learner_4p_sample_ratio)

            stats['learning_rate'] = (
                lr_schedulers[0].get_last_lr()[0]
                if (lr_schedulers is not None and train)
                else 0.0
            )

            for diag_key, diag_value in diagnostics_detached.items():
                diag_stat_value = diag_value
                if (
                    diag_key == 'vtrace_pg_loss'
                    or diag_key == 'upgo_pg_loss'
                    or diag_key == 'upgo_original_pg_loss'
                    or diag_key == 'baseline_loss_terminal'
                    or diag_key.startswith('vtrace_pg_loss_')
                    or diag_key.startswith('upgo_pg_loss_')
                    or diag_key.startswith('upgo_original_pg_loss_')
                    or diag_key.startswith('baseline_loss_terminal_')
                ):
                    diag_stat_value = diag_value * warmup_multiplier
                add_stat(diag_key, diag_stat_value)
            assert entropy_diagnostics is not None
            for head in ENTROPY_HEAD_KEYS:
                add_stat(
                    f'entropy_loss_{head}',
                    entropy_diagnostics[f'entropy_loss_{head}'] * warmup_multiplier,
                )
                add_stat(
                    f'entropy_sample_count_{head}',
                    entropy_diagnostics[f'entropy_sample_count_{head}'],
                )
                add_stat(f'mean_entropy_{head}', entropy_diagnostics[f'mean_entropy_{head}'])
            add_stat('mean_entropy', entropy_diagnostics['mean_entropy'])
            for diag_key, diag_value in entropy_diagnostics.items():
                if (
                    diag_key.startswith('Entropy.class_freq_')
                    or diag_key.startswith('Action.class_freq_')
                ):
                    add_stat(diag_key, diag_value)
            for l2_k, l2_v in policy_logit_l2_diagnostics.items():
                add_stat(l2_k, l2_v)
            for comp_k, comp_v in temperature_compensation_kl_diagnostics.items():
                add_stat(comp_k, comp_v)
            entropy_floor_stats_fn = getattr(learner_model, "entropy_floor_logging_stats", None)
            if callable(entropy_floor_stats_fn):
                for floor_k, floor_v in entropy_floor_stats_fn().items():
                    add_stat(floor_k, floor_v)
            floor_stats = learner_outputs.get("entropy_floor_stats_LEARN", {})
            assert isinstance(floor_stats, dict)

            def add_floor_temperature_stat(floor_stat_key, floor_stat_value, floor_active_frac):
                assert isinstance(floor_stat_value, torch.Tensor)
                assert isinstance(floor_active_frac, torch.Tensor)
                assert floor_stat_value.shape == floor_active_frac.shape, (
                    floor_stat_key,
                    tuple(floor_stat_value.shape),
                    tuple(floor_active_frac.shape),
                )
                active_samples = floor_active_frac > 0.0
                if bool(active_samples.any().item()):
                    if floor_stat_key.endswith("_temperature_max"):
                        add_stat(f"entropy_floor_{floor_stat_key}", floor_stat_value[active_samples].max())
                    else:
                        add_stat(f"entropy_floor_{floor_stat_key}", floor_stat_value[active_samples].mean())
                else:
                    add_stat(f"entropy_floor_{floor_stat_key}", 1.0)

            for floor_k, floor_v in floor_stats.items():
                assert isinstance(floor_v, torch.Tensor)
                primary_temperature_stat = False
                for head in ENTROPY_HEAD_KEYS:
                    if floor_k in (
                        f"{head}_temperature_mean",
                        f"{head}_temperature_max",
                    ):
                        active_key = f"{head}_active_frac"
                        assert active_key in floor_stats, (
                            f"learner entropy floor stats must include {active_key!r}"
                        )
                        add_floor_temperature_stat(floor_k, floor_v, floor_stats[active_key])
                        primary_temperature_stat = True
                if primary_temperature_stat:
                    continue
                if floor_k.endswith("_temperature_max"):
                    add_stat(f"entropy_floor_{floor_k}", floor_v.max())
                else:
                    add_stat(f"entropy_floor_{floor_k}", floor_v.mean())
            if stats_override is not None:
                for stat_key, stat_value in stats_override.items():
                    assert stat_key in stats or stat_key.startswith("value_"), stat_key
                    stats[stat_key] = float(stat_value)
            payload_losses = stats
            if not train:
                payload_losses = {f"{k}_val": v for k, v in stats.items()}

            payload = {
                'losses': payload_losses
            }
            if train and enable_clip_grad:
                payload['clip_grad'] = {
                    'clip_grad_value': clip_grad_value,
                    'total_norm': post_clip_grad_norm_scalar,
                }

            algo = 'rl'
            phase = 'train' if train else 'val'
            assert stop_event is not None, "stop_event is required for stats_queue_learner"
            queue_put_or_stop(
                stats_queue_learner,
                (payload, f'{algo}_{phase}'),
                stop_event,
                timeout_sec=0.1,
            )

        if train and flags.enable_lr_scheduler and lr_schedulers is not None:
            with profiler_span(wall_profiler, "lr_scheduler_step"):
                for scheduler in lr_schedulers:
                    scheduler.step()

    if model_profile_enabled():
        with profiler_span(wall_profiler, "after_learn_sync"):
            torch.cuda.synchronize(device)

