import collections

import torch
import torch.nn.functional as F

VTraceFromLogitsReturns = collections.namedtuple(
    'VTraceFromLogitsReturns',
    [
        'vs',
        'pg_advantages',
        'log_rhos',
        'behavior_action_log_probs',
        'target_action_log_probs',
    ],
)

VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(policy_logits.view(-1, policy_logits.shape[-1]), dim=-1),
        torch.flatten(actions),
        reduction='none',
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    trace_lambda,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    return from_action_log_probs(
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        trace_lambda=trace_lambda,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )


def from_action_log_probs(
    behavior_action_log_probs,
    target_action_log_probs,
    discounts,
    rewards,
    values,
    bootstrap_value,
    trace_lambda,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        trace_lambda=trace_lambda,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    trace_lambda,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        lmb = float(trace_lambda)
        assert 0.0 <= lmb <= 1.0, f"trace_lambda must be in [0, 1], got {lmb}"
        cs = lmb * torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


@torch.no_grad()
def from_importance_weights_detailed(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    trace_lambda,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """Same math as ``from_importance_weights``, plus intermediate tensors for debugging."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        lmb = float(trace_lambda)
        assert 0.0 <= lmb <= 1.0, f"trace_lambda must be in [0, 1], got {lmb}"
        cs = lmb * torch.clamp(rhos, max=1.0)
        values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        vs = torch.add(vs_minus_v_xs, values)

        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        detail = {
            "log_rhos": log_rhos,
            "rhos": rhos,
            "clipped_rhos": clipped_rhos,
            "cs": cs,
            "values_t_plus_1": values_t_plus_1,
            "deltas": deltas,
            "vs_minus_v_xs": vs_minus_v_xs,
            "vs": vs,
            "vs_t_plus_1": vs_t_plus_1,
            "clipped_pg_rhos": clipped_pg_rhos,
            "pg_advantages": pg_advantages,
        }
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages), detail
