import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.amp
import torch.nn as nn
import torch.optim

from ...gym.obs_wrapper import ORBIT_PLAYER_AXIS_SLOTS, orbit_assert_available_action_mask_contract
from ...gym.wall_tree_profiler import (
    WallTreeProfiler,
    model_profile_enabled,
    profiler_span,
)
from . import vtrace
from .buffer_utils import buffers_apply
from .learn import (
    _assert_model_weights_and_grads_finite,
    _ema_alpha_for_steps,
    _floating_tensors_to_float32,
    _reward_ema_update_and_get_norm,
)
from .losses import compute_baseline_loss, masked_explained_variance
from .losses_func_selfplay import (
    REWARD_HEAD_KEYS,
    _popart_get_norm,
    _reward_head_metric_key,
    _reward_head_bool_config,
    _reward_head_float_config,
    _tensor_moments,
)
from .stats import RollingAverage


rolling_value_grad = RollingAverage(window_size=1000)


def learn_value(
    device,
    flags: SimpleNamespace,
    value_model: nn.Module,
    batch: dict[str, torch.Tensor],
    wall_profiler: WallTreeProfiler,
    optimizers: Optional[list[torch.optim.Optimizer]],
    lr_schedulers: Optional[list[torch.optim.lr_scheduler._LRScheduler]],
    train: bool = True,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, float],
    dict[str, torch.Tensor],
]:
    value_forward_bf16 = bool(flags.learner_forward_bf16)
    value_loss_bf16 = bool(flags.learner_loss_bf16)
    value_backward_bf16 = bool(flags.learner_backward_bf16)
    expected_time = int(flags.unroll_length) + 1
    expected_batch = int(flags.batch_size)
    value_stats_tensors: dict[str, torch.Tensor] = {}

    def _run_value_model_on_rollout(model, rollout_batch):
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
        with profiler_span(wall_profiler, "value_rollout_obs_flatten"):
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
            label="value",
        )
        if model_profile_enabled():
            with profiler_span(wall_profiler, "value_pre_forward_synchronize"):
                torch.cuda.synchronize(device)

        with profiler_span(wall_profiler, "value_rollout_model_forward"):
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=value_forward_bf16,
            ):
                forward_out = model(
                    flat_batch,
                    output_full_policy_log_probs=False,
                    include_policy_logits_pre_action_mask=False,
                    include_value_head=True,
                    wall_profiler=wall_profiler,
                )
        with profiler_span(wall_profiler, "value_rollout_outputs_reshape"):
            out_rollout = buffers_apply(
                forward_out,
                lambda t: t.reshape(T, B, P, *tuple(t.shape[2:])),
            )
        return out_rollout

    with profiler_span(wall_profiler, "value_forward"):
        value_outputs = _run_value_model_on_rollout(value_model, batch)
    assert 'baseline_LEARN' in value_outputs, (
        "Value learner requires baseline_LEARN in model outputs"
    )
    baseline_seq_by_head = value_outputs['baseline_LEARN']
    assert isinstance(baseline_seq_by_head, dict), type(baseline_seq_by_head)
    assert tuple(baseline_seq_by_head.keys()) == REWARD_HEAD_KEYS, (
        tuple(baseline_seq_by_head.keys()),
        REWARD_HEAD_KEYS,
    )
    for head, baseline_seq in baseline_seq_by_head.items():
        assert isinstance(baseline_seq, torch.Tensor) and baseline_seq.ndim >= 3, (
            "Value learner requires baseline_LEARN head with time, batch, and player dims, "
            f"head={head!r} got {type(baseline_seq)} "
            f"shape={tuple(baseline_seq.shape) if isinstance(baseline_seq, torch.Tensor) else None}"
        )
        assert (
            int(baseline_seq.shape[0]) == expected_time
            and int(baseline_seq.shape[1]) == expected_batch
            and int(baseline_seq.shape[2]) == 1
        ), (
            "Value baseline_LEARN head must be [T, B, 1] with "
            f"T=unroll_length+1 ({expected_time}), B=batch_size ({expected_batch}); "
            f"head={head!r} got shape={tuple(baseline_seq.shape)}"
        )

    with profiler_span(wall_profiler, "value_batch_align_bootstrap_reward"):
        pm_full = batch["obs_LEARN_INFER"]["player_mask"]
        assert isinstance(pm_full, torch.Tensor) and pm_full.ndim == 3
        assert int(pm_full.shape[0]) == expected_time
        assert int(pm_full.shape[1]) == expected_batch
        assert int(pm_full.shape[2]) == 1
        bootstrap_value = {
            head: values[-1] * pm_full[-1].to(device=values.device, dtype=values.dtype)
            for head, values in baseline_seq_by_head.items()
        }
        batch_aligned_with_value = buffers_apply(batch, lambda x: x[:-1])
        value_outputs = buffers_apply(value_outputs, lambda x: x[:-1])

        if bool(flags.enable_reward_ema_norm):
            reward_learn_stat = batch_aligned_with_value['reward_LEARN_STAT']
            assert isinstance(reward_learn_stat, dict)
            assert tuple(reward_learn_stat.keys()) == REWARD_HEAD_KEYS, (
                tuple(reward_learn_stat.keys()),
                REWARD_HEAD_KEYS,
            )
            ema_eps = float(flags.reward_ema_eps)
            reward_mask = batch_aligned_with_value["obs_LEARN_INFER"]["player_mask"]
            for head, reward in reward_learn_stat.items():
                assert isinstance(reward, torch.Tensor)
                steps_in_batch = int(reward.shape[0]) * int(reward.shape[1])
                ema_alpha = _ema_alpha_for_steps(
                    steps_in_batch,
                    float(flags.reward_ema_alpha),
                    label=f"value_reward_{head}",
                )
                assert reward_mask.shape == reward.shape, (
                    f"value reward/mask shape mismatch for {head!r}: "
                    f"{tuple(reward.shape)} vs {tuple(reward_mask.shape)}"
                )
                reward_ema_mean_tensor, reward_ema_std_tensor = _reward_ema_update_and_get_norm(
                    head,
                    reward,
                    reward_mask,
                    alpha=ema_alpha,
                    eps=ema_eps,
                    device=torch.device(device),
                    update=False,
                )
                value_stats_tensors[f'reward_ema_mean_{head}'] = reward_ema_mean_tensor
                value_stats_tensors[f'reward_ema_std_{head}'] = reward_ema_std_tensor
                reward_learn_stat[head] = (
                    (reward - reward_ema_mean_tensor) / reward_ema_std_tensor
                ).to(dtype=reward.dtype)

    if not value_loss_bf16:
        value_outputs = _floating_tensors_to_float32(value_outputs)
        bootstrap_value = _floating_tensors_to_float32(bootstrap_value)
        batch_for_value = _floating_tensors_to_float32(batch_aligned_with_value)
    else:
        batch_for_value = batch_aligned_with_value

    if train:
        assert optimizers is not None, "Value optimizer must be provided when train=True"
        with profiler_span(wall_profiler, "value_zero_grad"):
            for optimizer in optimizers:
                optimizer.zero_grad()

    values_by_head = value_outputs['baseline_LEARN']
    reward_learn_stat = batch_for_value['reward_LEARN_STAT']
    assert isinstance(reward_learn_stat, dict)
    assert tuple(reward_learn_stat.keys()) == REWARD_HEAD_KEYS, (
        tuple(reward_learn_stat.keys()),
        REWARD_HEAD_KEYS,
    )
    done_raw = batch_for_value["done_LEARN_STAT"]
    assert isinstance(done_raw, torch.Tensor) and done_raw.ndim == 3
    pm_s = batch_for_value["obs_LEARN_INFER"]["player_mask"]
    assert isinstance(pm_s, torch.Tensor) and pm_s.ndim == 3
    assert tuple(int(x) for x in pm_s.shape) == (
        int(flags.unroll_length),
        int(flags.batch_size),
        1,
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=value_loss_bf16):
        value_targets: dict[str, torch.Tensor] = {}
        first_head = REWARD_HEAD_KEYS[0]
        total_loss = torch.zeros(
            (),
            device=values_by_head[first_head].device,
            dtype=values_by_head[first_head].dtype,
        )
        for reward_head in REWARD_HEAD_KEYS:
            baseline_cost = _reward_head_float_config(flags, "baseline_cost", reward_head)
            discounting = _reward_head_float_config(flags, "discounting", reward_head)
            trace_lambda = _reward_head_float_config(flags, "lmb", reward_head)
            value_bound_min = _reward_head_float_config(
                flags,
                "value_target_bound_min",
                reward_head,
            )
            value_bound_max = _reward_head_float_config(
                flags,
                "value_target_bound_max",
                reward_head,
            )
            baseline_loss_use_mse = _reward_head_bool_config(
                flags,
                "baseline_loss_use_mse",
                reward_head,
            )
            baseline_smooth_l1_beta = _reward_head_float_config(
                flags,
                "baseline_smooth_l1_beta",
                reward_head,
            )
            assert value_bound_min < value_bound_max, (
                reward_head,
                value_bound_min,
                value_bound_max,
            )
            values = values_by_head[reward_head]
            assert isinstance(values, torch.Tensor)
            assert values.ndim == 3
            assert int(values.shape[2]) == 1, tuple(values.shape)
            rewards_for_value_full = reward_learn_stat[reward_head]
            assert isinstance(rewards_for_value_full, torch.Tensor)
            assert tuple(int(x) for x in rewards_for_value_full.shape) == tuple(
                int(x) for x in values.shape
            )
            rewards_for_value = rewards_for_value_full.to(
                dtype=values.dtype,
                device=rewards_for_value_full.device,
            )
            assert tuple(int(x) for x in done_raw.shape) == tuple(
                int(x) for x in rewards_for_value.shape
            )
            done_fp = done_raw.to(dtype=values.dtype, device=rewards_for_value.device)
            next_step_valid = 1.0 - done_fp
            assert int(pm_full.shape[0]) == int(values.shape[0]) + 1
            assert tuple(int(x) for x in pm_full.shape[1:]) == tuple(
                int(x) for x in values.shape[1:]
            )
            pm_next = pm_full[1:].to(dtype=values.dtype, device=rewards_for_value.device)
            pm_s_head = pm_full[:-1].to(dtype=values.dtype, device=rewards_for_value.device)
            assert pm_next.shape == values.shape
            assert pm_s_head.shape == values.shape
            discounts = next_step_valid * discounting * pm_next
            values_for_returns = torch.clamp(values, min=value_bound_min, max=value_bound_max)
            bootstrap_value_head = bootstrap_value[reward_head]
            assert isinstance(bootstrap_value_head, torch.Tensor)
            bootstrap_value_for_returns = torch.clamp(
                bootstrap_value_head,
                min=value_bound_min,
                max=value_bound_max,
            )
            td_lambda_returns = vtrace.from_importance_weights(
                log_rhos=torch.zeros_like(values_for_returns),
                discounts=discounts,
                rewards=rewards_for_value,
                values=values_for_returns,
                bootstrap_value=bootstrap_value_for_returns,
                trace_lambda=trace_lambda,
                clip_rho_threshold=None,
                clip_pg_rho_threshold=None,
            )
            value_target = torch.clamp(
                td_lambda_returns.vs,
                min=value_bound_min,
                max=value_bound_max,
            )
            value_targets[reward_head] = value_target.detach()
            with torch.no_grad():
                td_err = _tensor_moments((td_lambda_returns.vs - values).detach())
                value_stats_tensors.update(
                    {
                        f"Diagnostic_{_reward_head_metric_key('td_error_mean', reward_head)}": (
                            td_err['mean']
                        ),
                        f"Diagnostic_{_reward_head_metric_key('td_error_std', reward_head)}": (
                            td_err['std']
                        ),
                        f"Diagnostic_{_reward_head_metric_key('td_error_abs_mean', reward_head)}": (
                            td_err['abs_mean']
                        ),
                        f"Diagnostic_{_reward_head_metric_key('td_error_abs_max', reward_head)}": (
                            td_err['abs_max']
                        ),
                    }
                )
            if flags.enable_popart:
                mean, std = _popart_get_norm(reward_head, device=device)
                value_stats_tensors[f'popart_mean_{reward_head}'] = mean
                value_stats_tensors[f'popart_std_{reward_head}'] = std
                mean = mean.to(dtype=values.dtype, device=values.device)
                std = std.to(dtype=values.dtype, device=values.device)
                values_norm = (values - mean) / std
                targets_norm = (value_target - mean) / std
            else:
                values_norm = values
                targets_norm = value_target
            if baseline_cost != 0.0:
                baseline_loss = baseline_cost * compute_baseline_loss(
                    values_norm,
                    targets_norm,
                    reduction=flags.reduction,
                    mask=pm_s_head,
                    use_mse=baseline_loss_use_mse,
                    smooth_l1_beta=baseline_smooth_l1_beta,
                )
            else:
                baseline_loss = torch.zeros(
                    (),
                    device=values_norm.device,
                    dtype=values_norm.dtype,
                )
            total_loss = total_loss + baseline_loss
            with torch.no_grad():
                ev = masked_explained_variance(
                    values_norm.detach(),
                    targets_norm.detach(),
                    pm_s_head.detach(),
                )
            value_stats_tensors[
                _reward_head_metric_key("baseline_explained_variance", reward_head)
            ] = ev.detach().to(dtype=torch.float32)
            value_stats_tensors[
                _reward_head_metric_key("baseline_loss", reward_head)
            ] = baseline_loss.detach()

        loss_components_for_backward = {"value_total_loss": total_loss}

    if train:
        with profiler_span(wall_profiler, "value_finite_model_check_before"):
            _assert_model_weights_and_grads_finite(
                value_model,
                loss_components_for_backward,
            )
        if total_loss.requires_grad:
            with profiler_span(wall_profiler, "value_backward"):
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=value_backward_bf16,
                ):
                    total_loss.backward()
            with profiler_span(wall_profiler, "value_finite_model_check_after"):
                _assert_model_weights_and_grads_finite(
                    value_model,
                    loss_components_for_backward,
                )
        with torch.no_grad():
            with profiler_span(wall_profiler, "value_grad_clip"):
                if bool(flags.enable_clip_grad):
                    clip_grad_value = min(rolling_value_grad.average() * 1.5, 10000.0)
                    if clip_grad_value == 0.0:
                        clip_grad_value = 10.0
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        value_model.parameters(),
                        clip_grad_value,
                    )
                    post_clip_grad_norm_scalar = total_norm.item()
                    value_stats_tensors['value_clip_grad_value'] = torch.tensor(
                        clip_grad_value,
                        device=total_norm.device,
                        dtype=torch.float32,
                    )
                    value_stats_tensors['value_total_norm'] = total_norm.detach().to(
                        dtype=torch.float32
                    )
                    if not math.isnan(post_clip_grad_norm_scalar):
                        rolling_value_grad.add(post_clip_grad_norm_scalar)
            with profiler_span(wall_profiler, "value_optimizer_step"):
                for optimizer in optimizers:
                    optimizer.step()
            if flags.enable_lr_scheduler and lr_schedulers is not None:
                with profiler_span(wall_profiler, "value_lr_scheduler_step"):
                    for scheduler in lr_schedulers:
                        scheduler.step()

    detached_bootstrap = {
        head: value.detach()
        for head, value in bootstrap_value.items()
    }
    detached_baseline_seq = {
        head: value.detach()
        for head, value in baseline_seq_by_head.items()
    }
    value_stats = {
        key: value.detach().cpu().item()
        for key, value in value_stats_tensors.items()
    }
    return value_targets, detached_bootstrap, value_stats, detached_baseline_seq
