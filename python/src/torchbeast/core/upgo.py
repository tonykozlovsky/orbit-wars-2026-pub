import collections

import torch

UPGOReturns = collections.namedtuple('UPGOReturns', 'vs advantages')


@torch.no_grad()
def upgo(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discounts: torch.Tensor,
    log_rhos: torch.Tensor,
    lmb: float,
    clip_rho_threshold: float = 1.0,
) -> UPGOReturns:
    assert rewards.shape == values.shape == discounts.shape == log_rhos.shape, (
        f"rewards/values/discounts/log_rhos shape mismatch: "
        f"{tuple(rewards.shape)} {tuple(values.shape)} {tuple(discounts.shape)} {tuple(log_rhos.shape)}"
    )
    lmb_f = float(lmb)
    assert 0.0 <= lmb_f <= 1.0, f"lmb must be in [0, 1], got {lmb_f}"
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=float(clip_rho_threshold))
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    target_values = [bootstrap_value]
    for t in range(discounts.shape[0] - 1, -1, -1):
        optimistic_continuation = torch.max(
            values_t_plus_1[t],
            (1.0 - lmb_f) * values_t_plus_1[t] + lmb_f * target_values[-1],
        )
        raw_target = rewards[t] + discounts[t] * optimistic_continuation
        corrected_target = values[t] + clipped_rhos[t] * (raw_target - values[t])
        target_values.append(
            corrected_target
        )
    target_values.reverse()
    # Remove bootstrap value from end of target_values list
    target_values = torch.stack(target_values[:-1], dim=0)

    return UPGOReturns(vs=target_values, advantages=target_values - values)


@torch.no_grad()
def upgo_original(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discounts: torch.Tensor,
    log_rhos: torch.Tensor,
    clip_rho_threshold: float = 1.0,
) -> UPGOReturns:
    assert rewards.shape == values.shape == discounts.shape == log_rhos.shape, (
        f"rewards/values/discounts/log_rhos shape mismatch: "
        f"{tuple(rewards.shape)} {tuple(values.shape)} {tuple(discounts.shape)} {tuple(log_rhos.shape)}"
    )
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=float(clip_rho_threshold))
    values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    one_step_q = rewards + discounts * values_t_plus_1
    upgoing = one_step_q >= values
    continue_trace = torch.cat([upgoing[1:], torch.ones_like(upgoing[-1:])], dim=0)

    target_values = [bootstrap_value]
    for t in range(discounts.shape[0] - 1, -1, -1):
        continuation = torch.where(continue_trace[t], target_values[-1], values_t_plus_1[t])
        target_values.append(rewards[t] + discounts[t] * continuation)
    target_values.reverse()
    target_values = torch.stack(target_values[:-1], dim=0)

    advantages = clipped_rhos * (target_values - values)
    corrected_values = values + advantages
    return UPGOReturns(vs=corrected_values, advantages=advantages)
