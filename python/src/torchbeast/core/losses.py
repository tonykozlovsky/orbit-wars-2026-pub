import torch
import torch.nn.functional as F

from ...gym.obs_wrapper import ORBIT_MOVE_CLASSES_PER_TARGET


def _assert_masked_log_probs_are_normalized(
    log_probs: torch.Tensor,
    available_mask: torch.Tensor,
) -> None:
    assert log_probs.shape == available_mask.shape, (
        f"log_probs/available_mask shape mismatch: {tuple(log_probs.shape)} vs "
        f"{tuple(available_mask.shape)}"
    )
    assert torch.is_floating_point(log_probs), (
        f"Expected floating point log_probs, got dtype={log_probs.dtype}"
    )
    assert available_mask.dtype == torch.bool, (
        f"Expected bool available_mask, got dtype={available_mask.dtype}"
    )
    has_available = available_mask.any(dim=-1)
    assert bool(torch.all(has_available).item()), (
        "Every policy slot must expose at least one available action"
    )
    invalid_finite_count = (torch.isfinite(log_probs) & (~available_mask)).sum()
    assert int(invalid_finite_count.item()) == 0, (
        "policy_log_probs must be -inf where available_action_mask is 0"
    )
    log_z = torch.logsumexp(log_probs.to(dtype=torch.float32), dim=-1)
    assert torch.allclose(log_z, torch.zeros_like(log_z), atol=1e-1, rtol=0.0), (
        "policy_log_probs must be normalized over available_action_mask"
    )


def combine_policy_log_probs_for_taken_actions(
    policy_log_probs: torch.Tensor,
    actions: torch.Tensor,
    actions_taken_index: torch.Tensor,
    available_action_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Joint log-prob per timestep from per-slot ``log_softmax`` outputs (invalid actions are ``-inf``).

    policy_log_probs shape:       time, batch, players, n_units, n_actions
    actions shape:                time, batch, players, n_units, 1 in env action indexing
    actions_taken_index shape:    time, batch, players, n_units, 1 in env action indexing
    available_action_mask shape:  time, batch, players, n_units, n_actions
    Returned shape: time, batch, players — sum of log-probs over slots.
    """
    assert actions.shape[-1] == 1, f"Expected single action per sample, got shape {actions.shape}"

    assert torch.is_floating_point(policy_log_probs), (
        f"Expected floating point log_probs, got dtype={policy_log_probs.dtype}"
    )
    compute_dtype = torch.float32
    log_probs_all = policy_log_probs.to(dtype=compute_dtype)
    assert actions_taken_index.shape == actions.shape, (
        f"actions_taken_index must match actions: {tuple(actions_taken_index.shape)} vs "
        f"{tuple(actions.shape)}"
    )
    assert available_action_mask.shape == policy_log_probs.shape, (
        f"available_action_mask must match policy_log_probs: {tuple(available_action_mask.shape)} vs "
        f"{tuple(policy_log_probs.shape)}"
    )
    available_mask = available_action_mask.to(device=log_probs_all.device, dtype=torch.bool)
    _assert_masked_log_probs_are_normalized(log_probs_all, available_mask)
    n_actions = int(policy_log_probs.shape[-1])
    n_slots = int(policy_log_probs.shape[-2])
    assert n_actions % n_slots == 0, (tuple(policy_log_probs.shape), n_slots)
    model_actions_per_target = n_actions // n_slots
    assert 1 <= model_actions_per_target <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        model_actions_per_target,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    taken_env = actions_taken_index.to(device=log_probs_all.device, dtype=torch.int64)
    assert bool(torch.all(taken_env >= 0).item()), (
        "actions_taken_index must be non-negative"
    )
    action_idx = actions.to(device=log_probs_all.device, dtype=torch.int64)
    assert torch.equal(action_idx, taken_env), (
        "actions must match action_taken_index"
    )
    taken_dst = taken_env // int(ORBIT_MOVE_CLASSES_PER_TARGET)
    taken_subindex = taken_env % int(ORBIT_MOVE_CLASSES_PER_TARGET)
    assert bool(torch.all(taken_dst < n_slots).item()), (
        "actions_taken_index destination must be in action-slot range"
    )
    assert bool(torch.all(taken_subindex < model_actions_per_target).item()), (
        "actions_taken_index subaction must exist in the model policy head"
    )
    taken = taken_dst * model_actions_per_target + taken_subindex
    assert bool(torch.all(taken < n_actions).item()), (
        "actions_taken_index must map into model action range"
    )
    selected_available = torch.gather(available_mask, -1, taken)
    assert bool(torch.all(selected_available).item()), (
        "action_taken_index must select available actions"
    )
    selected_log_probs = torch.gather(log_probs_all, -1, taken)
    assert bool((~torch.isneginf(selected_log_probs)).all().item()), (
        "taken actions must have finite log-prob under stored policy_log_probs"
    )
    return torch.flatten(selected_log_probs, start_dim=-2, end_dim=-1).sum(dim=-1)


def combine_policy_entropy_per_planet(
    policy_log_probs: torch.Tensor,
    available_action_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Entropy from stored ``log_softmax`` outputs (invalid actions are ``-inf``).

    ``[..., n_planets, n_actions]`` → ``[..., n_planets]``.
    """
    compute_dtype = policy_log_probs.dtype
    assert torch.is_floating_point(policy_log_probs), (
        f"Expected floating point log_probs, got dtype={policy_log_probs.dtype}"
    )
    assert available_action_mask.shape == policy_log_probs.shape, (
        f"available_action_mask must match policy_log_probs: {tuple(available_action_mask.shape)} vs "
        f"{tuple(policy_log_probs.shape)}"
    )
    log_p = policy_log_probs.to(dtype=compute_dtype)
    valid = available_action_mask.to(device=log_p.device, dtype=torch.bool)
    _assert_masked_log_probs_are_normalized(log_p, valid)
    log_p_valid = torch.where(valid, log_p, torch.zeros_like(log_p))
    p = torch.where(valid, torch.exp(log_p_valid), torch.zeros_like(log_p))
    contrib = p * log_p_valid
    entropies = -contrib.sum(dim=-1)
    return entropies


def compute_teacher_kl_from_log_probs(
    learner_policy_log_probs: torch.Tensor,
    teacher_policy_log_probs: torch.Tensor,
    *,
    available_action_mask: torch.Tensor,
    teacher_available_action_mask: torch.Tensor,
    zero_missing_policy_actions: bool,
) -> torch.Tensor:
    compute_dtype = torch.float32
    assert torch.is_floating_point(learner_policy_log_probs), (
        f"Expected floating point learner log_probs, got dtype={learner_policy_log_probs.dtype}"
    )
    assert torch.is_floating_point(teacher_policy_log_probs), (
        f"Expected floating point teacher log_probs, got dtype={teacher_policy_log_probs.dtype}"
    )
    learner_lp = learner_policy_log_probs.to(dtype=compute_dtype)
    teacher_lp = teacher_policy_log_probs.to(dtype=compute_dtype)
    learner_mask = available_action_mask.to(
        device=learner_lp.device, dtype=torch.bool
    )
    teacher_mask = teacher_available_action_mask.to(
        device=learner_lp.device, dtype=torch.bool
    )
    assert learner_mask.shape == learner_lp.shape, (
        "available_action_mask must match learner log_probs: "
        f"mask={tuple(learner_mask.shape)} "
        f"log_probs={tuple(learner_lp.shape)}"
    )
    assert teacher_mask.shape == teacher_lp.shape, (
        "teacher_available_action_mask must match teacher log_probs: "
        f"mask={tuple(teacher_mask.shape)} "
        f"log_probs={tuple(teacher_lp.shape)}"
    )
    assert learner_lp.ndim == teacher_lp.ndim, (
        tuple(learner_lp.shape),
        tuple(teacher_lp.shape),
    )
    assert learner_lp.shape[:-1] == teacher_lp.shape[:-1], (
        tuple(learner_lp.shape),
        tuple(teacher_lp.shape),
    )
    n = int(learner_lp.shape[-2])
    assert n > 0, tuple(learner_lp.shape)
    assert int(teacher_lp.shape[-2]) == n, (tuple(teacher_lp.shape), n)
    learner_action_width = int(learner_lp.shape[-1])
    teacher_action_width = int(teacher_lp.shape[-1])
    assert learner_action_width % n == 0, (tuple(learner_lp.shape), n)
    assert teacher_action_width % n == 0, (tuple(teacher_lp.shape), n)
    learner_actions_per_target = learner_action_width // n
    teacher_actions_per_target = teacher_action_width // n
    common_actions_per_target = min(learner_actions_per_target, teacher_actions_per_target)
    assert common_actions_per_target >= 1, (
        learner_actions_per_target,
        teacher_actions_per_target,
    )

    learner_lp_by_target = learner_lp.reshape(
        *learner_lp.shape[:-1],
        n,
        learner_actions_per_target,
    )
    teacher_lp_by_target = teacher_lp.reshape(
        *teacher_lp.shape[:-1],
        n,
        teacher_actions_per_target,
    )
    learner_mask_by_target = learner_mask.reshape(
        *learner_mask.shape[:-1],
        n,
        learner_actions_per_target,
    )
    teacher_mask_by_target = teacher_mask.reshape(
        *teacher_mask.shape[:-1],
        n,
        teacher_actions_per_target,
    )
    neg_large = torch.finfo(learner_lp.dtype).min / 16.0
    _assert_masked_log_probs_are_normalized(learner_lp, learner_mask)
    _assert_masked_log_probs_are_normalized(teacher_lp, teacher_mask)

    if (
        bool(zero_missing_policy_actions)
        and learner_actions_per_target >= teacher_actions_per_target
    ):
        teacher_width = teacher_actions_per_target
        target_width = n * teacher_width
        learner_lp_for_teacher = learner_lp_by_target[..., :teacher_width].reshape(
            *learner_lp.shape[:-1],
            target_width,
        )
        teacher_lp_for_teacher = teacher_lp_by_target.reshape(
            *teacher_lp.shape[:-1],
            target_width,
        )
        learner_mask_for_teacher = learner_mask_by_target[..., :teacher_width].reshape(
            *learner_mask.shape[:-1],
            target_width,
        )
        teacher_mask_for_teacher = teacher_mask_by_target.reshape(
            *teacher_mask.shape[:-1],
            target_width,
        )
        target_mask = learner_mask_for_teacher & teacher_mask_for_teacher
        assert bool(torch.all(target_mask.any(dim=-1)).item()), (
            "teacher KL target action support must contain at least one action per source slot",
            tuple(target_mask.shape),
        )
        teacher_lp_target = torch.log_softmax(
            teacher_lp_for_teacher.masked_fill(~target_mask, neg_large),
            dim=-1,
        ).masked_fill(~target_mask, float("-inf"))
        _assert_masked_log_probs_are_normalized(teacher_lp_target, target_mask)
        has_valid_action = target_mask.any(dim=-1)
        t_lp = teacher_lp_target.detach()
        learner_lp_valid = torch.where(
            target_mask,
            learner_lp_for_teacher,
            torch.zeros_like(learner_lp_for_teacher),
        )
        teacher_lp_valid = torch.where(target_mask, t_lp, torch.zeros_like(t_lp))
        teacher_p = torch.where(target_mask, torch.exp(teacher_lp_valid), torch.zeros_like(t_lp))
    else:
        common_width = n * common_actions_per_target
        learner_lp = learner_lp_by_target[..., :common_actions_per_target].reshape(
            *learner_lp.shape[:-1],
            common_width,
        )
        teacher_lp = teacher_lp_by_target[..., :common_actions_per_target].reshape(
            *teacher_lp.shape[:-1],
            common_width,
        )
        learner_mask = learner_mask_by_target[..., :common_actions_per_target].reshape(
            *learner_mask.shape[:-1],
            common_width,
        )
        teacher_mask = teacher_mask_by_target[..., :common_actions_per_target].reshape(
            *teacher_mask.shape[:-1],
            common_width,
        )
        target_mask = learner_mask & teacher_mask
        assert bool(torch.all(target_mask.any(dim=-1)).item()), (
            "teacher KL common action support must contain at least one action per source slot",
            tuple(target_mask.shape),
        )
        learner_lp = torch.log_softmax(
            learner_lp.masked_fill(~target_mask, neg_large),
            dim=-1,
        ).masked_fill(~target_mask, float("-inf"))
        teacher_lp = torch.log_softmax(
            teacher_lp.masked_fill(~target_mask, neg_large),
            dim=-1,
        ).masked_fill(~target_mask, float("-inf"))
        _assert_masked_log_probs_are_normalized(learner_lp, target_mask)
        _assert_masked_log_probs_are_normalized(teacher_lp, target_mask)
        has_valid_action = target_mask.any(dim=-1)
        t_lp = teacher_lp.detach()
        learner_lp_valid = torch.where(target_mask, learner_lp, torch.zeros_like(learner_lp))
        teacher_lp_valid = torch.where(target_mask, t_lp, torch.zeros_like(t_lp))
        teacher_p = torch.where(target_mask, torch.exp(teacher_lp_valid), torch.zeros_like(t_lp))
    log_prob_delta = teacher_lp_valid - learner_lp_valid
    kl_terms = teacher_p * log_prob_delta
    kl_div = kl_terms.sum(dim=-1)
    kl_div = kl_div * has_valid_action.to(dtype=kl_div.dtype, device=kl_div.device)
    return kl_div.mean(dim=-1)


def reduce(losses: torch.Tensor, reduction: str, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        assert losses.shape == mask.shape, (
            f"losses/mask shape mismatch: losses={tuple(losses.shape)} mask={tuple(mask.shape)}"
        )
        mask = mask.to(dtype=losses.dtype, device=losses.device)
        losses = losses * mask

    if reduction == 'mean':
        if mask is not None:
            if losses.ndim < 2:
                return losses.sum() / mask.sum().clamp(min=1.0)
            assert losses.ndim >= 2, "Expected batch dimension at index 1 for mean reduction"
            dims = tuple(i for i in range(losses.ndim) if i != 1)
            per_batch_sum = losses.sum(dim=dims)
            per_batch_count = mask.sum(dim=dims).clamp(min=1.0)
            per_batch_mean = per_batch_sum / per_batch_count
            return per_batch_mean.mean()
        else:
            if losses.ndim < 2:
                return losses.mean()
            assert losses.ndim >= 2, "Expected batch dimension at index 1 for mean reduction"
            dims = tuple(i for i in range(losses.ndim) if i != 1)
            per_batch_mean = losses.mean(dim=dims)
            return per_batch_mean.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def masked_explained_variance(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``1 - Var_w(y - pred) / Var_w(y)`` with the same nonnegative weights as baseline ``mask``."""
    assert pred.shape == target.shape == mask.shape, (
        f"pred/target/mask shape mismatch: {tuple(pred.shape)} {tuple(target.shape)} {tuple(mask.shape)}"
    )
    w = mask.to(dtype=pred.dtype, device=pred.device)
    denom = w.sum().clamp(min=1.0)
    mean_y = (target * w).sum() / denom
    y_c = target - mean_y
    var_y = (y_c.pow(2) * w).sum() / denom
    diff = target - pred
    mean_diff = (diff * w).sum() / denom
    d_c = diff - mean_diff
    var_err = (d_c.pow(2) * w).sum() / denom
    return 1.0 - var_err / var_y.clamp(min=eps)


def compute_baseline_loss(
    values: torch.Tensor,
    value_targets: torch.Tensor,
    reduction: str,
    mask: torch.Tensor = None,
    *,
    use_mse: bool = False,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    targets = value_targets.detach().to(dtype=values.dtype, device=values.device)
    assert float(smooth_l1_beta) > 0.0, smooth_l1_beta
    if use_mse:
        baseline_loss = F.mse_loss(values, targets, reduction="none")
    else:
        baseline_loss = F.smooth_l1_loss(
            values,
            targets,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
    return reduce(baseline_loss, reduction=reduction, mask=mask)


def compute_policy_gradient_loss(
    action_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    reduction: str,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    cross_entropy = -action_log_probs.view_as(advantages)
    advantages_detached = advantages.detach().to(
        dtype=action_log_probs.dtype, device=action_log_probs.device
    )
    return reduce(cross_entropy * advantages_detached, reduction, mask=mask)
