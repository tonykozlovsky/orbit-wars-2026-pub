import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ...gym.obs_wrapper import (
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_MOVE_CLASS_FREQ_NAMES,
)
from . import upgo, vtrace
from .common import ENTROPY_HEAD_KEYS, POLICY_ACTION_KEYS


def _available_action_mask_from_model_outputs(outputs: dict, act_space: str) -> torch.Tensor:
    assert "available_action_mask_LEARN" in outputs
    masks = outputs["available_action_mask_LEARN"]
    assert isinstance(masks, dict)
    assert act_space in masks, (act_space, tuple(masks.keys()))
    mask = masks[act_space]
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool, mask.dtype
    return mask


def _player_mask_from_batch(batch: dict) -> torch.Tensor:
    o = batch["obs_LEARN_INFER"]
    assert isinstance(o, dict) and "player_mask" in o
    m = o["player_mask"]
    assert isinstance(m, torch.Tensor) and m.ndim == 3
    assert int(m.shape[2]) == 1, tuple(m.shape)
    return m


from .losses import (
    combine_policy_entropy_per_planet,
    combine_policy_log_probs_for_taken_actions,
    compute_baseline_loss,
    compute_policy_gradient_loss,
    compute_teacher_kl_from_log_probs,
    masked_explained_variance,
    reduce,
)


_popart_state = {}
_popart_shared_dict = None  # Optional multiprocessing.Manager().dict() mapping head->(mean, m2)
_popart_lock = None  # Optional multiprocessing.Manager().RLock for coordinating updates


def _tensor_moments(x: torch.Tensor, *, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    mean = x.mean()
    mean2 = (x * x).mean()
    var = torch.clamp(mean2 - mean * mean, min=eps)
    std = torch.sqrt(var)
    abs_x = torch.abs(x)
    abs_mean = abs_x.mean()
    abs_max = abs_x.max()
    return {
        'mean': mean,
        'std': std,
        'abs_mean': abs_mean,
        'abs_max': abs_max,
    }


def _popart_get_state(head_name: str):
    if head_name not in _popart_state:
        if _popart_shared_dict is not None and head_name in _popart_shared_dict:
            mean, m2 = _popart_shared_dict[head_name]
            _popart_state[head_name] = {'mean': float(mean), 'm2': float(m2)}
        else:
            _popart_state[head_name] = {'mean': 0.0, 'm2': 1.0}
    return _popart_state[head_name]


def _popart_alpha_for_steps(steps: int, base_alpha: float) -> float:
    assert int(steps) > 0, "popart steps must be > 0"
    base = float(base_alpha)
    assert 0.0 <= base <= 1.0, "popart_alpha_per_step must be in [0, 1]"
    return 1.0 - (1.0 - base) ** float(steps)


def _popart_update(
    head_name: str,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    alpha: float,
):
    state = _popart_get_state(head_name)
    with torch.no_grad():
        if mask is not None:
            m = mask.to(dtype=targets.dtype, device=targets.device)
            denom = m.sum()
            if denom.item() <= 0:
                return
            batch_mean = (targets * m).sum() / denom
            batch_mean2 = ((targets * targets) * m).sum() / denom
            batch_mean = batch_mean.detach().cpu().item()
            batch_mean2 = batch_mean2.detach().cpu().item()
        else:
            batch_mean = targets.mean().detach().cpu().item()
            batch_mean2 = (targets * targets).mean().detach().cpu().item()

        # Simple serialized update with mutex: each learner updates EMA sequentially
        if _popart_lock is not None and _popart_shared_dict is not None:
            with _popart_lock:
                cur_mean, cur_m2 = _popart_shared_dict.get(head_name, (state['mean'], state['m2']))
                new_mean = (1.0 - alpha) * float(cur_mean) + alpha * float(batch_mean)
                new_m2 = (1.0 - alpha) * float(cur_m2) + alpha * float(batch_mean2)
                _popart_shared_dict[head_name] = (float(new_mean), float(new_m2))
                state['mean'] = float(new_mean)
                state['m2'] = float(new_m2)
        else:
            new_mean = (1.0 - alpha) * state['mean'] + alpha * float(batch_mean)
            new_m2 = (1.0 - alpha) * state['m2'] + alpha * float(batch_mean2)
            state['mean'] = new_mean
            state['m2'] = new_m2


def _popart_get_norm(head_name: str, device: torch.device):
    state = _popart_get_state(head_name)
    mean = torch.tensor(state['mean'], device=device, dtype=torch.float32)
    var = max(state['m2'] - state['mean'] * state['mean'], 1e-6)
    std = torch.tensor(var ** 0.5, device=device, dtype=torch.float32)
    return mean, std


def get_popart_state() -> dict:
    if _popart_shared_dict is not None and len(_popart_shared_dict) > 0:
        return {k: {'mean': float(v[0]), 'm2': float(v[1])} for k, v in dict(_popart_shared_dict).items()}
    return copy.deepcopy(_popart_state)


def load_popart_state(state: dict) -> None:
    global _popart_state
    _popart_state = copy.deepcopy(state)
    if _popart_shared_dict is not None:
        for k, v in state.items():
            _popart_shared_dict[k] = (float(v.get('mean', 0.0)), float(v.get('m2', 1.0)))


def set_popart_shared_dict(shared_dict, lock=None) -> None:
    global _popart_shared_dict
    global _popart_lock
    _popart_shared_dict = shared_dict
    _popart_lock = lock


def _reward_head_config_mapping(values: dict[str, object] | SimpleNamespace, name: str) -> dict[str, object]:
    if isinstance(values, SimpleNamespace):
        mapping = vars(values)
    else:
        assert isinstance(values, dict), (name, type(values))
        mapping = values
    assert tuple(mapping.keys()) == REWARD_HEAD_KEYS, (name, tuple(mapping.keys()), REWARD_HEAD_KEYS)
    return mapping


def _reward_head_float_config(flags, name: str, reward_head: str) -> float:
    values = getattr(flags, name)
    mapping = _reward_head_config_mapping(values, name)
    return float(mapping[reward_head])


def _reward_head_bool_config(flags, name: str, reward_head: str) -> bool:
    values = getattr(flags, name)
    mapping = _reward_head_config_mapping(values, name)
    return bool(mapping[reward_head])


def _entropy_head_config_mapping(values: dict[str, object] | SimpleNamespace, name: str) -> dict[str, object]:
    if isinstance(values, SimpleNamespace):
        mapping = vars(values)
    else:
        assert isinstance(values, dict), (name, type(values))
        mapping = values
    assert tuple(mapping.keys()) == ENTROPY_HEAD_KEYS, (name, tuple(mapping.keys()), ENTROPY_HEAD_KEYS)
    return mapping


def _entropy_head_float_config(flags, name: str, entropy_head: str) -> float:
    values = getattr(flags, name)
    mapping = _entropy_head_config_mapping(values, name)
    return float(mapping[entropy_head])


def _teacher_kl_cost_scalar(teacher_kl_cost: float | dict[str, float]) -> float:
    if isinstance(teacher_kl_cost, dict):
        vals = [float(v) for v in teacher_kl_cost.values()]
        assert len(vals) >= 1
        return sum(vals) / float(len(vals))
    return float(teacher_kl_cost)


def _masked_log_probs_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    assert logits.shape == mask.shape, (
        f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}"
    )
    assert mask.dtype == torch.bool, f"Expected bool mask, got {mask.dtype}"
    neg_large = torch.finfo(logits.dtype).min / 16.0
    return F.log_softmax(logits.masked_fill(~mask, neg_large), dim=-1)


def _normalized_entropy_and_slot_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
    player_slot_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert logits.shape == mask.shape, (
        f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}"
    )
    log_probs = _masked_log_probs_from_logits(logits, mask)
    valid = mask.to(device=log_probs.device, dtype=torch.bool)
    p = torch.where(valid, torch.exp(log_probs), torch.zeros_like(log_probs))
    log_probs_finite = torch.where(valid, log_probs, torch.zeros_like(log_probs))
    contrib = p * log_probs_finite
    entropy = -contrib.sum(dim=-1)
    k_counts = valid.sum(dim=-1)
    eligible = k_counts > 1
    log_k = torch.log(torch.clamp(k_counts.to(dtype=entropy.dtype), min=2))
    h_norm = torch.where(eligible, entropy / log_k, torch.zeros_like(entropy))
    assert player_slot_mask.shape == h_norm.shape, (
        f"player_slot_mask must match entropy shape: {tuple(player_slot_mask.shape)} vs "
        f"{tuple(h_norm.shape)}"
    )
    return entropy, h_norm, player_slot_mask.to(
        device=h_norm.device,
        dtype=h_norm.dtype,
    ) * eligible.to(device=h_norm.device, dtype=h_norm.dtype)


def _orbit_amount_class_frequency_diagnostics(
    *,
    action_taken_index: torch.Tensor,
    available_action_mask: torch.Tensor,
    player_mask: torch.Tensor,
    eligible_source_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert action_taken_index.ndim == 5, tuple(action_taken_index.shape)
    assert int(action_taken_index.shape[-1]) == 1, tuple(action_taken_index.shape)
    assert available_action_mask.ndim == 5, tuple(available_action_mask.shape)
    assert available_action_mask.dtype == torch.bool, available_action_mask.dtype
    assert player_mask.ndim == 3, tuple(player_mask.shape)
    assert eligible_source_mask.ndim == 4, tuple(eligible_source_mask.shape)
    assert tuple(action_taken_index.shape[:-1]) == tuple(available_action_mask.shape[:-1]), (
        tuple(action_taken_index.shape),
        tuple(available_action_mask.shape),
    )
    assert tuple(player_mask.shape) == tuple(available_action_mask.shape[:3]), (
        tuple(player_mask.shape),
        tuple(available_action_mask.shape),
    )
    assert tuple(eligible_source_mask.shape) == tuple(available_action_mask.shape[:-1]), (
        tuple(eligible_source_mask.shape),
        tuple(available_action_mask.shape),
    )
    n_slots = int(available_action_mask.shape[-2])
    action_width = int(available_action_mask.shape[-1])
    assert action_width % n_slots == 0, (tuple(available_action_mask.shape), n_slots)
    model_actions_per_target = action_width // n_slots
    assert model_actions_per_target == int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        model_actions_per_target,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    available_by_dst_class = available_action_mask.reshape(
        *available_action_mask.shape[:-1],
        n_slots,
        model_actions_per_target,
    )
    assert available_by_dst_class.shape == (
        *available_action_mask.shape[:-1],
        n_slots,
        model_actions_per_target,
    )
    taken_env = action_taken_index[..., 0].to(
        device=available_action_mask.device,
        dtype=torch.int64,
    )
    taken_subindex = taken_env % int(ORBIT_MOVE_CLASSES_PER_TARGET)
    active_source_mask = (
        player_mask.unsqueeze(-1).to(device=available_action_mask.device, dtype=torch.bool)
        & eligible_source_mask.to(device=available_action_mask.device, dtype=torch.bool)
    )
    out: dict[str, torch.Tensor] = {}
    assert len(ORBIT_MOVE_CLASS_FREQ_NAMES) == int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        ORBIT_MOVE_CLASS_FREQ_NAMES,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    for class_idx, class_name in enumerate(ORBIT_MOVE_CLASS_FREQ_NAMES):
        class_available = available_by_dst_class[..., class_idx].any(dim=-1)
        class_denominator_mask = active_source_mask & class_available
        class_denominator = class_denominator_mask.to(dtype=torch.float32).sum()
        class_selected = class_denominator_mask & (taken_subindex == int(class_idx))
        class_numerator = class_selected.to(dtype=torch.float32).sum()
        class_freq = (
            class_numerator / class_denominator.clamp(min=1.0)
        ).detach()
        out[f"Entropy.class_freq_{class_name}"] = class_freq
        out[f"Action.class_freq_{class_name}"] = class_freq
    return out


def selfplay_entropy_loss(
    flags,
    batch,
    learner_outputs,
    shared_shortfall_entropy,
):
    assert "action_taken_index_LEARN_STAT" in batch
    avail_mask = _available_action_mask_from_model_outputs(learner_outputs, "spawn_fleet")
    player_mask = _player_mask_from_batch(batch)
    assert player_mask.ndim == 3
    assert POLICY_ACTION_KEYS == ("spawn_fleet",), POLICY_ACTION_KEYS
    final_policy_log_probs = learner_outputs['policy_log_probs_LEARN']["spawn_fleet"]
    assert final_policy_log_probs.ndim == 5, (
        f"final policy log_probs must be [T,B,P,N,A], got {final_policy_log_probs.ndim=} "
        f"{tuple(final_policy_log_probs.shape)}"
    )
    assert avail_mask.shape == final_policy_log_probs.shape, (
        f"final availability must match log_probs: {tuple(avail_mask.shape)} vs "
        f"{tuple(final_policy_log_probs.shape)}"
    )
    assert ENTROPY_HEAD_KEYS == ("spawn_fleet",), ENTROPY_HEAD_KEYS
    policy_math_dtype = final_policy_log_probs.dtype
    entropy_loss = torch.tensor(
        0.0,
        device=final_policy_log_probs.device,
        dtype=policy_math_dtype,
    )
    diagnostics: dict = {}

    final_planet_entropy = combine_policy_entropy_per_planet(
        final_policy_log_probs,
        avail_mask,
    )
    final_k_counts = avail_mask.sum(dim=-1)
    final_eligible = final_k_counts > 1
    final_log_k = torch.log(
        torch.clamp(final_k_counts.to(device=final_planet_entropy.device, dtype=final_planet_entropy.dtype), min=2)
    )
    final_h_norm = torch.where(
        final_eligible,
        final_planet_entropy / final_log_k,
        torch.zeros_like(final_planet_entropy),
    )
    final_slot_mask = player_mask.unsqueeze(-1).to(
        device=final_h_norm.device,
        dtype=final_h_norm.dtype,
    ) * final_eligible.to(device=final_h_norm.device, dtype=final_h_norm.dtype)
    diagnostics['mean_entropy'] = (
        (final_h_norm.detach() * final_slot_mask).sum()
        / final_slot_mask.sum().clamp(min=1.0)
    )
    diagnostics.update(
        _orbit_amount_class_frequency_diagnostics(
            action_taken_index=batch["action_taken_index_LEARN_STAT"],
            available_action_mask=avail_mask,
            player_mask=player_mask,
            eligible_source_mask=final_eligible,
        )
    )
    entropy_loss_head = torch.zeros((), device=entropy_loss.device, dtype=entropy_loss.dtype)
    assert not (bool(flags.classic_entropy) and bool(flags.ce_entropy))
    if bool(flags.classic_entropy):
        entropy_cost = _entropy_head_float_config(flags, "entropy_cost", "spawn_fleet")
        if entropy_cost != 0.0:
            classic_joint_entropy = final_planet_entropy.sum(dim=-1)
            classic_player_mask = player_mask.expand_as(classic_joint_entropy).to(
                device=final_planet_entropy.device,
                dtype=final_planet_entropy.dtype,
            )
            entropy_loss_head = entropy_cost * reduce(
                -classic_joint_entropy,
                reduction=flags.reduction,
                mask=classic_player_mask,
            )
    if bool(flags.ce_entropy):
        entropy_cost = _entropy_head_float_config(flags, "entropy_cost", "spawn_fleet")
        if entropy_cost != 0.0:
            valid_log_probs = torch.where(
                avail_mask,
                final_policy_log_probs,
                torch.zeros_like(final_policy_log_probs),
            )
            ce_per_planet = -valid_log_probs.sum(dim=-1) / final_k_counts.to(
                device=final_policy_log_probs.device,
                dtype=final_policy_log_probs.dtype,
            )
            ce_per_planet = torch.where(
                final_eligible,
                ce_per_planet,
                torch.zeros_like(ce_per_planet),
            )
            ce_joint_entropy = ce_per_planet.sum(dim=-1)
            ce_player_mask = player_mask.expand_as(ce_joint_entropy).to(
                device=ce_joint_entropy.device,
                dtype=ce_joint_entropy.dtype,
            )
            entropy_loss_head = entropy_cost * reduce(
                ce_joint_entropy,
                reduction=flags.reduction,
                mask=ce_player_mask,
            )
    entropy_loss = entropy_loss + entropy_loss_head
    diagnostics["entropy_loss_spawn_fleet"] = entropy_loss_head
    diagnostics["entropy_sample_count_spawn_fleet"] = final_slot_mask.detach().sum()
    diagnostics["mean_entropy_spawn_fleet"] = diagnostics["mean_entropy"]

    return entropy_loss, diagnostics


def selfplay_teacher_kl_losses(
    flags,
    batch,
    learner_outputs,
    teacher_outputs_by_num_players,
    teacher_kl_cost: float | dict[str, float],
):
    assert isinstance(teacher_outputs_by_num_players, dict)
    player_mask = _player_mask_from_batch(batch)
    assert "game_num_players_LEARN" in batch
    game_num_players = batch["game_num_players_LEARN"]
    assert isinstance(game_num_players, torch.Tensor)
    assert game_num_players.dtype == torch.int64, game_num_players.dtype
    assert tuple(game_num_players.shape) == tuple(player_mask.shape), (
        tuple(game_num_players.shape),
        tuple(player_mask.shape),
    )
    active_games = game_num_players[player_mask > 0.5]
    assert int(active_games.numel()) > 0, tuple(game_num_players.shape)
    assert bool(((active_games == 2) | (active_games == 4)).all().item()), active_games
    teacher_num_players = {
        int(num_players)
        for num_players in teacher_outputs_by_num_players.keys()
    }
    assert teacher_num_players.issubset({2, 4}), teacher_num_players
    active_num_players = {
        int(num_players)
        for num_players in torch.unique(active_games.detach().cpu(), sorted=True).tolist()
    }
    assert active_num_players.issubset(teacher_num_players), (
        active_num_players,
        teacher_num_players,
    )
    sample_lp = learner_outputs['policy_log_probs_LEARN'][
        next(iter(learner_outputs['policy_log_probs_LEARN']))
    ]
    z = torch.zeros((), device=sample_lp.device, dtype=torch.float32)

    teacher_kl_by_sample = torch.zeros_like(player_mask, dtype=torch.float32)
    teacher_kl_loss_orig_by_num_players = {
        2: z,
        4: z,
    }

    for act_space in POLICY_ACTION_KEYS:
        assert act_space in batch['actions_LEARN'], (
            f"batch['actions_LEARN'] must contain '{act_space}' for teacher KL"
        )
        learner_policy_log_probs = learner_outputs['policy_log_probs_LEARN'][act_space]
        assert learner_policy_log_probs.ndim == 5, (
            f"learner policy log_probs must be [T,B,P,…,…], got {learner_policy_log_probs.ndim=} "
            f"{tuple(learner_policy_log_probs.shape)}"
        )
        learner_avail_mask = _available_action_mask_from_model_outputs(learner_outputs, act_space)
        assert learner_avail_mask.shape == learner_policy_log_probs.shape, (
            tuple(learner_avail_mask.shape),
            tuple(learner_policy_log_probs.shape),
        )
        for num_players, teacher_output in sorted(teacher_outputs_by_num_players.items()):
            num_players = int(num_players)
            teacher_mask = player_mask * (game_num_players == num_players).to(
                device=player_mask.device,
                dtype=player_mask.dtype,
            )
            if not bool((teacher_mask > 0.5).any().item()):
                continue
            teacher_policy_log_probs = teacher_output['policy_log_probs_LEARN'][act_space]
            assert teacher_policy_log_probs.ndim == 5, (
                f"teacher policy log_probs must be [T,B,P,…,…], got {teacher_policy_log_probs.ndim=} "
                f"{tuple(teacher_policy_log_probs.shape)}"
            )
            teacher_avail_mask = _available_action_mask_from_model_outputs(teacher_output, act_space)
            assert teacher_avail_mask.shape == teacher_policy_log_probs.shape, (
                tuple(teacher_avail_mask.shape),
                tuple(teacher_policy_log_probs.shape),
            )
            kl = compute_teacher_kl_from_log_probs(
                learner_policy_log_probs,
                teacher_policy_log_probs,
                available_action_mask=learner_avail_mask,
                teacher_available_action_mask=teacher_avail_mask,
                zero_missing_policy_actions=bool(flags.teacher.zero_missing_policy_actions),
            )
            assert kl.shape == player_mask.shape, (
                f"teacher KL must reduce to [T,B,P]: {tuple(kl.shape)} vs {tuple(player_mask.shape)}"
            )
            teacher_kl_by_sample = teacher_kl_by_sample + (
                kl * teacher_mask.to(device=kl.device, dtype=kl.dtype)
            )
            kl_reduced = reduce(
                kl, reduction=flags.reduction, mask=teacher_mask
            )
            teacher_kl_loss_orig_by_num_players[num_players] = (
                teacher_kl_loss_orig_by_num_players[num_players] + kl_reduced
            )

    teacher_kl_loss_orig = reduce(
        teacher_kl_by_sample,
        reduction=flags.reduction,
        mask=player_mask,
    )

    c = _teacher_kl_cost_scalar(teacher_kl_cost)
    if c != 0.0:
        teacher_kl_loss = c * teacher_kl_loss_orig
    else:
        teacher_kl_loss = z

    return {
        'teacher_kl_loss': teacher_kl_loss,
        'teacher_kl_loss_orig': teacher_kl_loss_orig,
        'teacher_kl_loss_orig_2p': teacher_kl_loss_orig_by_num_players[2],
        'teacher_kl_loss_orig_4p': teacher_kl_loss_orig_by_num_players[4],
    }


REWARD_HEAD_KEYS = ("baseline", "production_delta")


def _reward_head_metric_key(base_key: str, reward_head: str) -> str:
    assert reward_head in REWARD_HEAD_KEYS, reward_head
    if reward_head == "baseline":
        return base_key
    return f"{base_key}_{reward_head}"


def selfplay_teacher_baseline_loss(
    flags,
    batch,
    learner_outputs,
    teacher_outputs_by_num_players,
):
    assert isinstance(teacher_outputs_by_num_players, dict)
    cost = float(flags.teacher.baseline_cost)
    assert cost > 0.0, cost
    player_mask = _player_mask_from_batch(batch)
    assert "game_num_players_LEARN" in batch
    game_num_players = batch["game_num_players_LEARN"]
    assert isinstance(game_num_players, torch.Tensor)
    assert game_num_players.dtype == torch.int64, game_num_players.dtype
    assert tuple(game_num_players.shape) == tuple(player_mask.shape), (
        tuple(game_num_players.shape),
        tuple(player_mask.shape),
    )
    learner_values = learner_outputs['baseline_LEARN']["baseline"]
    assert isinstance(learner_values, torch.Tensor)
    assert tuple(learner_values.shape) == tuple(player_mask.shape), (
        tuple(learner_values.shape),
        tuple(player_mask.shape),
    )
    value_bound_min = _reward_head_float_config(flags, "value_target_bound_min", "baseline")
    value_bound_max = _reward_head_float_config(flags, "value_target_bound_max", "baseline")
    assert value_bound_min < value_bound_max, (value_bound_min, value_bound_max)
    teacher_targets = torch.zeros_like(learner_values)
    for num_players, teacher_output in sorted(teacher_outputs_by_num_players.items()):
        num_players = int(num_players)
        teacher_mask = (game_num_players == num_players) & (player_mask > 0.5)
        if not bool(teacher_mask.any().item()):
            continue
        teacher_values = teacher_output['baseline_LEARN']["baseline"]
        assert isinstance(teacher_values, torch.Tensor)
        assert tuple(teacher_values.shape) == tuple(learner_values.shape), (
            tuple(teacher_values.shape),
            tuple(learner_values.shape),
        )
        teacher_values = teacher_values.to(
            device=learner_values.device,
            dtype=learner_values.dtype,
        )
        teacher_values = torch.clamp(
            teacher_values,
            min=value_bound_min,
            max=value_bound_max,
        )
        teacher_targets = torch.where(
            teacher_mask.to(device=learner_values.device),
            teacher_values,
            teacher_targets,
        )

    teacher_baseline_loss_orig = compute_baseline_loss(
        learner_values,
        teacher_targets,
        reduction=flags.reduction,
        mask=player_mask,
        use_mse=_reward_head_bool_config(flags, "baseline_loss_use_mse", "baseline"),
        smooth_l1_beta=_reward_head_float_config(
            flags,
            "baseline_smooth_l1_beta",
            "baseline",
        ),
    )
    teacher_baseline_loss = cost * teacher_baseline_loss_orig
    with torch.no_grad():
        teacher_baseline_explained_variance = masked_explained_variance(
            learner_values.detach(),
            teacher_targets.detach(),
            player_mask.detach(),
        )
    return {
        'teacher_baseline_loss': teacher_baseline_loss,
        'teacher_baseline_loss_orig': teacher_baseline_loss_orig,
        'teacher_baseline_explained_variance': teacher_baseline_explained_variance,
    }


def losses_func_selfplay(
    rank,
    flags,
    batch,
    learner_outputs,
    bootstrap_value,
    player_mask_time_major: torch.Tensor,
    collect_diagnostics: bool = True,
    loss_weight: float | None = None,
    update_popart: bool = True,
):
    """RL losses: V-trace(λ) / off-policy UPGO / baseline over all policy slots.

    Returns partial losses only; the scalar passed to ``backward()`` is built in ``learn.learn``.
    """
    assert isinstance(learner_outputs['policy_log_probs_LEARN'], dict), (
        "Expected learner_outputs['policy_log_probs_LEARN'] dict"
    )
    assert len(learner_outputs['policy_log_probs_LEARN']) > 0, (
        "Expected non-empty learner policy log_probs dict"
    )
    first_act_space = next(iter(learner_outputs['policy_log_probs_LEARN']))
    sample_lp = learner_outputs['policy_log_probs_LEARN'][first_act_space]
    assert isinstance(sample_lp, torch.Tensor), (
        f"Expected tensor log_probs for act_space='{first_act_space}'"
    )
    assert torch.is_floating_point(sample_lp), (
        f"Expected floating point log_probs for act_space='{first_act_space}', got {sample_lp.dtype}"
    )
    policy_math_dtype = sample_lp.dtype
    n_players = int(sample_lp.shape[2])
    assert n_players == 1, tuple(sample_lp.shape)
    combined_behavior_action_log_probs = torch.zeros(
        (int(sample_lp.shape[0]), int(sample_lp.shape[1]), n_players),
        device=sample_lp.device,
        dtype=policy_math_dtype,
    )
    combined_learner_action_log_probs = torch.zeros_like(combined_behavior_action_log_probs)

    assert "action_taken_index_LEARN_STAT" in batch
    taken_index = batch["action_taken_index_LEARN_STAT"]
    assert isinstance(taken_index, torch.Tensor)
    assert "obs_LEARN_INFER" in batch
    sample_actions = batch["actions_LEARN"][first_act_space]
    assert taken_index.shape == sample_actions.shape
    for act_space in batch['actions_LEARN'].keys():
        actions = batch['actions_LEARN'][act_space]
        assert actions.ndim == 5, (
            f"buffer actions must be [T,B,P,…,…], got {actions.ndim=} "
            f"{tuple(actions.shape)}"
        )

        assert act_space in batch['behavior_log_prob_sum_LEARN'], (
            f"batch['behavior_log_prob_sum_LEARN'] must contain '{act_space}'"
        )
        behavior_sum = batch['behavior_log_prob_sum_LEARN'][act_space]
        assert behavior_sum.ndim == 3, (
            f"behavior_log_prob_sum must be [T,B,P], got {behavior_sum.ndim=} "
            f"{tuple(behavior_sum.shape)}"
        )
        assert behavior_sum.shape == combined_behavior_action_log_probs.shape, (
            "behavior_log_prob_sum_LEARN must match [T,B,P] of learner policy tensors: "
            f"{tuple(behavior_sum.shape)} vs {tuple(combined_behavior_action_log_probs.shape)}"
        )
        assert bool(torch.isfinite(behavior_sum).all().item()), (
            "behavior_log_prob_sum_LEARN must contain finite actor log-prob sums"
        )

        learner_policy_log_probs = learner_outputs['policy_log_probs_LEARN'][act_space]
        avail_mask = _available_action_mask_from_model_outputs(learner_outputs, act_space)
        assert learner_policy_log_probs.ndim == 5, (
            f"learner policy log_probs must be [T,B,P,…,…], got {learner_policy_log_probs.ndim=} "
            f"{tuple(learner_policy_log_probs.shape)}"
        )
        taken_act = taken_index
        assert taken_act.shape == actions.shape, (
            f"taken index must match actions: {tuple(taken_act.shape)} vs {tuple(actions.shape)}"
        )
        behavior_action_log_probs = behavior_sum.to(
            device=combined_behavior_action_log_probs.device,
            dtype=combined_behavior_action_log_probs.dtype,
        )
        assert behavior_action_log_probs.shape == combined_behavior_action_log_probs.shape, (
            "behavior_action_log_probs shape mismatch for selfplay accumulation: "
            f"{tuple(behavior_action_log_probs.shape)} vs "
            f"{tuple(combined_behavior_action_log_probs.shape)}"
        )
        combined_behavior_action_log_probs = (
            combined_behavior_action_log_probs + behavior_action_log_probs
        )
        learner_action_log_probs = combine_policy_log_probs_for_taken_actions(
            learner_policy_log_probs,
            actions,
            taken_act,
            avail_mask,
        )
        assert learner_action_log_probs.shape == combined_learner_action_log_probs.shape, (
            "learner_action_log_probs shape mismatch for selfplay accumulation: "
            f"{tuple(learner_action_log_probs.shape)} vs {tuple(combined_learner_action_log_probs.shape)}"
        )
        combined_learner_action_log_probs = (
            combined_learner_action_log_probs + learner_action_log_probs
        )

    values_by_head = learner_outputs['baseline_LEARN']
    assert isinstance(values_by_head, dict), type(values_by_head)
    assert tuple(values_by_head.keys()) == REWARD_HEAD_KEYS, (
        tuple(values_by_head.keys()),
        REWARD_HEAD_KEYS,
    )
    reward_learn_stat = batch['reward_LEARN_STAT']
    assert isinstance(reward_learn_stat, dict)
    assert tuple(reward_learn_stat.keys()) == REWARD_HEAD_KEYS, (
        tuple(reward_learn_stat.keys()),
        REWARD_HEAD_KEYS,
    )
    assert isinstance(bootstrap_value, dict), type(bootstrap_value)
    assert tuple(bootstrap_value.keys()) == REWARD_HEAD_KEYS, (
        tuple(bootstrap_value.keys()),
        REWARD_HEAD_KEYS,
    )
    assert "done_LEARN_STAT" in batch
    done_raw = batch["done_LEARN_STAT"]
    assert isinstance(done_raw, torch.Tensor)
    assert done_raw.ndim == 3

    compute_policy_gradient_loss_func = compute_policy_gradient_loss
    loss_zero = torch.zeros(
        (),
        device=combined_learner_action_log_probs.device,
        dtype=combined_learner_action_log_probs.dtype,
    )

    summed_vtrace_pg_loss = torch.zeros_like(loss_zero)
    summed_upgo_pg_loss = torch.zeros_like(loss_zero)
    summed_upgo_original_pg_loss = torch.zeros_like(loss_zero)
    summed_baseline_loss = torch.zeros_like(loss_zero)
    summed_baseline_loss_terminal = torch.zeros_like(loss_zero)
    diagnostics = {}

    for reward_head in REWARD_HEAD_KEYS:
        vtrace_cost = _reward_head_float_config(flags, "vtrace_cost", reward_head)
        upgo_cost = _reward_head_float_config(flags, "upgo_cost", reward_head)
        upgo_original_cost = _reward_head_float_config(
            flags,
            "upgo_original_cost",
            reward_head,
        )
        baseline_cost = _reward_head_float_config(flags, "baseline_cost", reward_head)
        term_cost = _reward_head_float_config(
            flags,
            "baseline_terminal_loss_cost",
            reward_head,
        )
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
        assert int(values.shape[-1]) == n_players, (
            "Expected learner baseline slot dim to match policy log_probs: "
            f"head={reward_head!r} baseline={tuple(values.shape)} n_players={n_players}"
        )
        value_dtype = values.dtype
        rewards_for_rl_full = reward_learn_stat[reward_head]
        assert isinstance(rewards_for_rl_full, torch.Tensor)
        assert int(rewards_for_rl_full.shape[-1]) == n_players, (
            "Expected reward_LEARN_STAT slot dim to match policy log_probs: "
            f"head={reward_head!r} rewards={tuple(rewards_for_rl_full.shape)} n_players={n_players}"
        )
        rewards_for_rl = rewards_for_rl_full.to(
            dtype=value_dtype, device=rewards_for_rl_full.device
        )
        assert tuple(int(x) for x in done_raw.shape) == tuple(
            int(x) for x in rewards_for_rl.shape
        )
        done_fp = done_raw.to(dtype=value_dtype, device=rewards_for_rl.device)
        next_step_valid = 1.0 - done_fp
        assert int(player_mask_time_major.shape[0]) == int(values.shape[0]) + 1
        assert tuple(int(x) for x in player_mask_time_major.shape[1:]) == tuple(
            int(x) for x in values.shape[1:]
        )
        pm_next = player_mask_time_major[1:].to(
            dtype=value_dtype, device=rewards_for_rl.device
        )
        assert pm_next.shape == values.shape
        pm_s = player_mask_time_major[:-1].to(
            dtype=value_dtype, device=rewards_for_rl.device
        )
        assert pm_s.shape == values.shape
        assert int(done_raw.shape[0]) == int(pm_s.shape[0])
        discounts = next_step_valid * discounting * pm_next
        values_for_returns = torch.clamp(values, min=value_bound_min, max=value_bound_max)
        bootstrap_value_head = bootstrap_value[reward_head]
        assert isinstance(bootstrap_value_head, torch.Tensor)
        bootstrap_value_for_returns = torch.clamp(
            bootstrap_value_head,
            min=value_bound_min,
            max=value_bound_max,
        )

        vtrace_returns = vtrace.from_action_log_probs(
            combined_behavior_action_log_probs,
            combined_learner_action_log_probs,
            discounts,
            rewards_for_rl,
            values_for_returns,
            bootstrap_value_for_returns,
            trace_lambda=trace_lambda,
            clip_rho_threshold=1.0,
            clip_pg_rho_threshold=1.0,
        )
        td_lambda_returns = vtrace.from_importance_weights(
            log_rhos=torch.zeros_like(combined_learner_action_log_probs),
            discounts=discounts,
            rewards=rewards_for_rl,
            values=values_for_returns,
            bootstrap_value=bootstrap_value_for_returns,
            trace_lambda=trace_lambda,
            clip_rho_threshold=None,
            clip_pg_rho_threshold=None,
        )

        upgo_returns = upgo.upgo(
            rewards=rewards_for_rl,
            values=values_for_returns,
            bootstrap_value=bootstrap_value_for_returns,
            discounts=discounts,
            log_rhos=vtrace_returns.log_rhos,
            lmb=trace_lambda,
            clip_rho_threshold=1.0,
        )
        upgo_original_returns = upgo.upgo_original(
            rewards=rewards_for_rl,
            values=values_for_returns,
            bootstrap_value=bootstrap_value_for_returns,
            discounts=discounts,
            log_rhos=vtrace_returns.log_rhos,
            clip_rho_threshold=1.0,
        )
        value_targets = torch.clamp(
            td_lambda_returns.vs,
            min=value_bound_min,
            max=value_bound_max,
        )

        if collect_diagnostics:
            with torch.no_grad():
                vt = _tensor_moments(vtrace_returns.pg_advantages.detach())
                ug = _tensor_moments(upgo_returns.advantages.detach())
                ugo = _tensor_moments(upgo_original_returns.advantages.detach())
                td_err = _tensor_moments((td_lambda_returns.vs - values).detach())
                diagnostics.update({
                    _reward_head_metric_key('vtrace_pg_adv_mean', reward_head): vt['mean'],
                    _reward_head_metric_key('vtrace_pg_adv_std', reward_head): vt['std'],
                    _reward_head_metric_key('vtrace_pg_adv_abs_mean', reward_head): vt['abs_mean'],
                    _reward_head_metric_key('vtrace_pg_adv_abs_max', reward_head): vt['abs_max'],
                    _reward_head_metric_key('upgo_adv_mean', reward_head): ug['mean'],
                    _reward_head_metric_key('upgo_adv_std', reward_head): ug['std'],
                    _reward_head_metric_key('upgo_adv_abs_mean', reward_head): ug['abs_mean'],
                    _reward_head_metric_key('upgo_adv_abs_max', reward_head): ug['abs_max'],
                    _reward_head_metric_key('upgo_original_adv_mean', reward_head): ugo['mean'],
                    _reward_head_metric_key('upgo_original_adv_std', reward_head): ugo['std'],
                    _reward_head_metric_key('upgo_original_adv_abs_mean', reward_head): ugo['abs_mean'],
                    _reward_head_metric_key('upgo_original_adv_abs_max', reward_head): ugo['abs_max'],
                    _reward_head_metric_key('td_error_mean', reward_head): td_err['mean'],
                    _reward_head_metric_key('td_error_std', reward_head): td_err['std'],
                    _reward_head_metric_key('td_error_abs_mean', reward_head): td_err['abs_mean'],
                    _reward_head_metric_key('td_error_abs_max', reward_head): td_err['abs_max'],
                })

                rhos = vtrace_returns.log_rhos.detach().exp()
                rho_stats = _tensor_moments(rhos)
                log_rho_stats = _tensor_moments(vtrace_returns.log_rhos.detach())
                rho_clip_frac = (rhos > 1.0).to(dtype=torch.float32).mean()
                diagnostics.update({
                    _reward_head_metric_key('vtrace_rho_mean', reward_head): rho_stats['mean'],
                    _reward_head_metric_key('vtrace_rho_std', reward_head): rho_stats['std'],
                    _reward_head_metric_key('vtrace_rho_abs_max', reward_head): rho_stats['abs_max'],
                    _reward_head_metric_key('vtrace_log_rho_mean', reward_head): log_rho_stats['mean'],
                    _reward_head_metric_key('vtrace_log_rho_std', reward_head): log_rho_stats['std'],
                    _reward_head_metric_key('vtrace_rho_clip_frac', reward_head): rho_clip_frac,
                })

        if vtrace_cost != 0.0:
            vtrace_pg_loss = vtrace_cost * compute_policy_gradient_loss_func(
                combined_learner_action_log_probs,
                vtrace_returns.pg_advantages,
                reduction=flags.reduction,
                mask=pm_s,
            )
        else:
            vtrace_pg_loss = torch.zeros_like(loss_zero)

        if upgo_cost != 0.0:
            upgo_pg_loss = upgo_cost * compute_policy_gradient_loss_func(
                combined_learner_action_log_probs,
                upgo_returns.advantages,
                reduction=flags.reduction,
                mask=pm_s,
            )
        else:
            upgo_pg_loss = torch.zeros_like(loss_zero)

        if upgo_original_cost != 0.0:
            upgo_original_pg_loss = upgo_original_cost * compute_policy_gradient_loss_func(
                combined_learner_action_log_probs,
                upgo_original_returns.advantages,
                reduction=flags.reduction,
                mask=pm_s,
            )
        else:
            upgo_original_pg_loss = torch.zeros_like(loss_zero)

        if flags.enable_popart:
            steps_in_batch = int(values.shape[0]) * int(values.shape[1])
            popart_alpha = _popart_alpha_for_steps(
                steps_in_batch,
                float(flags.popart_alpha_per_step),
            )
            if update_popart:
                _popart_update(
                    reward_head,
                    value_targets,
                    mask=pm_s,
                    alpha=popart_alpha,
                )
            mean, std = _popart_get_norm(reward_head, device=rank)
            mean = mean.to(dtype=values.dtype, device=values.device)
            std = std.to(dtype=values.dtype, device=values.device)
            values_norm = (values - mean) / std
            targets_norm = (value_targets - mean) / std
        else:
            values_norm = values
            targets_norm = value_targets

        if baseline_cost != 0.0:
            baseline_loss = baseline_cost * compute_baseline_loss(
                values_norm,
                targets_norm,
                reduction=flags.reduction,
                mask=pm_s,
                use_mse=baseline_loss_use_mse,
                smooth_l1_beta=baseline_smooth_l1_beta,
            )
        else:
            baseline_loss = torch.zeros((), device=values_norm.device, dtype=values_norm.dtype)

        with torch.no_grad():
            ev = masked_explained_variance(
                values_norm.detach(),
                targets_norm.detach(),
                pm_s.detach(),
            )
        diagnostics[_reward_head_metric_key("baseline_explained_variance", reward_head)] = (
            ev.detach().to(dtype=torch.float32)
        )

        baseline_loss_terminal = torch.zeros(
            (),
            device=baseline_loss.device,
            dtype=baseline_loss.dtype,
        )
        if term_cost != 0.0:
            T_done, B_done, P_done = (
                int(done_raw.shape[0]),
                int(done_raw.shape[1]),
                int(done_raw.shape[2]),
            )
            T_val, B_val, P_val = int(values.shape[0]), int(values.shape[1]), int(values.shape[2])
            assert T_done == T_val and B_done == B_val and P_done == P_val, (
                "done/reward time-batch-player must match values: "
                f"head={reward_head!r} done {(T_done, B_done, P_done)} "
                f"vs values {(T_val, B_val, P_val)}"
            )
            done_bool = done_raw > 0.5
            player_terminal_bool = done_bool | ((pm_s > 0.5) & (pm_next <= 0.5))
            t_ix = torch.arange(
                T_done,
                device=player_terminal_bool.device,
                dtype=torch.long,
            ).view(T_done, 1, 1).expand(T_done, B_done, P_done)
            sentinel = torch.full_like(t_ix, T_done)
            first_done_t = torch.where(player_terminal_bool, t_ix, sentinel).min(dim=0).values
            in_prefix = (first_done_t < T_done).unsqueeze(0) & (
                torch.arange(T_done, device=done_bool.device, dtype=torch.long).view(T_done, 1, 1)
                <= first_done_t.view(1, B_done, P_done)
            )
            terminal_active = in_prefix & (pm_s > 0.5)
            terminal_mask = terminal_active.to(dtype=values_norm.dtype, device=values_norm.device)
            terminal_targets = targets_norm.detach().to(
                dtype=values_norm.dtype,
                device=values_norm.device,
            )
            if baseline_loss_use_mse:
                terminal_loss_raw = F.mse_loss(values_norm, terminal_targets, reduction="none")
            else:
                terminal_loss_raw = F.smooth_l1_loss(
                    values_norm,
                    terminal_targets,
                    reduction="none",
                    beta=baseline_smooth_l1_beta,
                )
            terminal_loss_masked = terminal_loss_raw * terminal_mask
            terminal_time_reduction = "mean"
            if terminal_time_reduction == "mean":
                terminal_lane_loss = terminal_loss_masked.sum(dim=0) / terminal_mask.sum(
                    dim=0
                ).clamp(min=1.0)
            elif terminal_time_reduction == "sum":
                terminal_lane_loss = terminal_loss_masked.sum(dim=0)
            else:
                raise ValueError(
                    "terminal_time_reduction must be one of 'mean' or 'sum', "
                    f"was: {terminal_time_reduction}"
                )
            if flags.reduction == "mean":
                terminal_lane_active = terminal_mask.sum(dim=0) > 0.5
                per_batch_loss = terminal_lane_loss.sum(dim=-1) / terminal_lane_active.to(
                    dtype=terminal_loss_masked.dtype,
                    device=terminal_loss_masked.device,
                ).sum(dim=-1).clamp(min=1.0)
                baseline_loss_terminal = term_cost * per_batch_loss.mean()
            elif flags.reduction == "sum":
                baseline_loss_terminal = term_cost * terminal_lane_loss.sum()
            else:
                raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {flags.reduction}")

        if loss_weight is not None:
            weight = torch.tensor(
                float(loss_weight),
                device=baseline_loss.device,
                dtype=baseline_loss.dtype,
            )
            vtrace_pg_loss = vtrace_pg_loss * weight
            upgo_pg_loss = upgo_pg_loss * weight
            upgo_original_pg_loss = upgo_original_pg_loss * weight
            baseline_loss = baseline_loss * weight
            baseline_loss_terminal = baseline_loss_terminal * weight

        assert not torch.isnan(vtrace_pg_loss).any(), f'vtrace_pg_loss is NaN for {reward_head}'
        assert not torch.isnan(upgo_pg_loss).any(), f'upgo_pg_loss is NaN for {reward_head}'
        assert not torch.isnan(upgo_original_pg_loss).any(), (
            f'upgo_original_pg_loss is NaN for {reward_head}'
        )
        assert not torch.isnan(baseline_loss).any(), f'baseline_loss is NaN for {reward_head}'
        assert not torch.isnan(baseline_loss_terminal).any(), (
            f'baseline_loss_terminal is NaN for {reward_head}'
        )

        diagnostics[_reward_head_metric_key("vtrace_pg_loss", reward_head)] = (
            vtrace_pg_loss.detach()
        )
        diagnostics[_reward_head_metric_key("upgo_pg_loss", reward_head)] = upgo_pg_loss.detach()
        diagnostics[_reward_head_metric_key("upgo_original_pg_loss", reward_head)] = (
            upgo_original_pg_loss.detach()
        )
        diagnostics[_reward_head_metric_key("baseline_loss", reward_head)] = baseline_loss.detach()
        if term_cost != 0.0:
            diagnostics[_reward_head_metric_key("baseline_loss_terminal", reward_head)] = (
                baseline_loss_terminal.detach()
            )

        summed_vtrace_pg_loss = summed_vtrace_pg_loss + vtrace_pg_loss
        summed_upgo_pg_loss = summed_upgo_pg_loss + upgo_pg_loss
        summed_upgo_original_pg_loss = summed_upgo_original_pg_loss + upgo_original_pg_loss
        summed_baseline_loss = summed_baseline_loss + baseline_loss
        summed_baseline_loss_terminal = (
            summed_baseline_loss_terminal + baseline_loss_terminal
        )

    return (
        summed_vtrace_pg_loss,
        summed_upgo_pg_loss,
        summed_upgo_original_pg_loss,
        summed_baseline_loss,
        diagnostics,
        summed_baseline_loss_terminal,
    )
