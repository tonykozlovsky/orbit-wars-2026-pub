"""One-shot V-trace text dump (debug).

Requires ``flags.enable_terminal_vtrace_dump`` and non-empty env
``IMPALA_TERMINAL_VTRACE_DUMP_FILE`` (path). When disabled, :func:`maybe_dump_terminal_selfplay_vtrace_and_exit`
returns immediately without reading the environment.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any

import torch

from ...gym.obs_wrapper import ORBIT_PER_PLANET_MOVE_CLASSES, ORBIT_PLANET_ACTION_SLOTS
from . import upgo, vtrace
from .losses import combine_policy_log_probs_for_taken_actions


_ENV_DUMP_PATH = "IMPALA_TERMINAL_VTRACE_DUMP_FILE"
_REWARD_HEAD_KEYS = ("baseline", "production_delta")


def _reward_head_config_mapping(values: dict[str, Any] | SimpleNamespace, name: str) -> dict[str, Any]:
    if isinstance(values, SimpleNamespace):
        mapping = vars(values)
    else:
        assert isinstance(values, dict), (name, type(values))
        mapping = values
    assert tuple(mapping.keys()) == _REWARD_HEAD_KEYS, (name, tuple(mapping.keys()), _REWARD_HEAD_KEYS)
    return mapping


def _baseline_float_config_value(values: dict[str, Any] | SimpleNamespace, name: str) -> float:
    mapping = _reward_head_config_mapping(values, name)
    return float(mapping["baseline"])


def _find_first_terminal(done: torch.Tensor) -> tuple[int, int, int] | None:
    assert isinstance(done, torch.Tensor) and done.ndim == 3
    t_max, b_max, p_max = int(done.shape[0]), int(done.shape[1]), int(done.shape[2])
    for t in range(t_max):
        for b in range(b_max):
            for p in range(p_max):
                if float(done[t, b, p].item()) > 0.5:
                    return t, b, p
    return None


def _cpu_f32(x: torch.Tensor) -> torch.Tensor:
    assert isinstance(x, torch.Tensor)
    return x.detach().float().cpu()


def _cpu_i64(x: torch.Tensor) -> torch.Tensor:
    assert isinstance(x, torch.Tensor)
    return x.detach().to(dtype=torch.int64).cpu()


def _f(x: float) -> str:
    return format(float(x), ".10g")


def _class_indices_over_threshold(row: torch.Tensor, *, thr: float = 0.5) -> list[int]:
    assert isinstance(row, torch.Tensor) and row.ndim == 1
    m = row > thr
    ix = torch.nonzero(m, as_tuple=False).reshape(-1).tolist()
    return [int(i) for i in ix]


def _append_terminal_mask_dump(
    lines: list[str],
    *,
    t_done: int,
    b_done: int,
    n_players: int,
    taken_index: torch.Tensor,
    avail_mask: torch.Tensor,
) -> None:
    lines.append("")
    lines.append(
        f"=== terminal step t={t_done} b={b_done}: action_taken_index & available_action_mask ==="
    )
    lines.append(
        "Per planet_slot: taken move-class index and available move-class indices "
        f"(valid class range 0 .. {int(ORBIT_PER_PLANET_MOVE_CLASSES) - 1})."
    )
    taken_tb = _cpu_i64(taken_index[t_done, b_done])
    avail_tb = _cpu_f32(avail_mask[t_done, b_done])
    assert taken_tb.shape == (n_players, ORBIT_PLANET_ACTION_SLOTS, 1), (
        taken_tb.shape,
        n_players,
        ORBIT_PLANET_ACTION_SLOTS,
    )
    assert avail_tb.shape == (n_players, ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_MOVE_CLASSES), (
        avail_tb.shape,
    )
    for p in range(n_players):
        lines.append(f"--- player p={p} ---")
        lines.append("  action_taken_index (taken move-class index per planet_slot):")
        for s in range(int(ORBIT_PLANET_ACTION_SLOTS)):
            idx = int(taken_tb[p, s, 0].item())
            lines.append(f"    planet_slot {s}: {idx}")
        lines.append("  available_action_mask (allowed move-class indices per planet_slot):")
        for s in range(int(ORBIT_PLANET_ACTION_SLOTS)):
            idxs = _class_indices_over_threshold(avail_tb[p, s])
            lines.append(f"    planet_slot {s}: {idxs!r}")


def maybe_dump_terminal_selfplay_vtrace_and_exit(
    *,
    flags: Any,
    batch: dict[str, Any],
    learner_outputs: dict[str, Any],
    bootstrap_value: dict[str, torch.Tensor],
    player_mask_time_major: torch.Tensor,
    local_step: int,
) -> None:
    if not bool(flags.enable_terminal_vtrace_dump):
        return
    out_path = os.environ.get(_ENV_DUMP_PATH, "").strip()
    if out_path == "":
        return

    assert "done_LEARN_STAT" in batch
    done_raw = batch["done_LEARN_STAT"]
    assert isinstance(done_raw, torch.Tensor)
    hit = _find_first_terminal(done_raw)
    if hit is None:
        return

    t_done, b_done, p_done = hit
    sample_lp = learner_outputs["policy_log_probs_LEARN"][
        next(iter(learner_outputs["policy_log_probs_LEARN"]))
    ]
    assert isinstance(sample_lp, torch.Tensor)
    assert "available_action_mask_LEARN" in learner_outputs
    available_masks = learner_outputs["available_action_mask_LEARN"]
    assert isinstance(available_masks, dict)
    n_players = int(sample_lp.shape[2])
    policy_math_dtype = sample_lp.dtype
    device = sample_lp.device

    combined_behavior = torch.zeros(
        (int(sample_lp.shape[0]), int(sample_lp.shape[1]), n_players),
        device=device,
        dtype=policy_math_dtype,
    )
    combined_learner = torch.zeros_like(combined_behavior)

    assert "action_taken_index_LEARN_STAT" in batch
    taken_index = batch["action_taken_index_LEARN_STAT"]
    assert isinstance(taken_index, torch.Tensor)
    sample_actions = batch["actions_LEARN"][next(iter(batch["actions_LEARN"]))]
    assert isinstance(sample_actions, torch.Tensor)
    assert taken_index.shape == sample_actions.shape

    for act_space in batch["actions_LEARN"].keys():
        actions = batch["actions_LEARN"][act_space]
        assert act_space in batch["behavior_log_prob_sum_LEARN"]
        behavior_sum = batch["behavior_log_prob_sum_LEARN"][act_space]
        assert behavior_sum.ndim == 3, behavior_sum.shape
        assert behavior_sum.shape == combined_behavior.shape, (
            tuple(behavior_sum.shape),
            tuple(combined_behavior.shape),
        )
        learner_policy_log_probs = learner_outputs["policy_log_probs_LEARN"][act_space]
        assert act_space in available_masks, (act_space, tuple(available_masks.keys()))
        avail_mask = available_masks[act_space]
        assert isinstance(avail_mask, torch.Tensor)
        assert avail_mask.shape == learner_policy_log_probs.shape
        taken_act = taken_index
        combined_behavior = combined_behavior + behavior_sum.to(
            device=combined_behavior.device,
            dtype=combined_behavior.dtype,
        )
        learner_action_log_probs = combine_policy_log_probs_for_taken_actions(
            learner_policy_log_probs,
            actions,
            taken_act,
            avail_mask,
        )
        combined_learner = combined_learner + learner_action_log_probs

    baseline_learn = learner_outputs["baseline_LEARN"]
    assert isinstance(baseline_learn, dict)
    assert "baseline" in baseline_learn, sorted(baseline_learn.keys())
    values = baseline_learn["baseline"]
    assert isinstance(values, torch.Tensor)
    reward_learn_stat = batch["reward_LEARN_STAT"]
    assert isinstance(reward_learn_stat, dict)
    assert "baseline" in reward_learn_stat, sorted(reward_learn_stat.keys())
    rewards_full = reward_learn_stat["baseline"]
    assert isinstance(rewards_full, torch.Tensor)
    assert isinstance(bootstrap_value, dict)
    assert "baseline" in bootstrap_value, sorted(bootstrap_value.keys())
    bootstrap_value = bootstrap_value["baseline"]
    assert isinstance(bootstrap_value, torch.Tensor)
    value_dtype = values.dtype
    rewards = rewards_full.to(dtype=value_dtype, device=rewards_full.device)

    done_fp = done_raw.to(dtype=value_dtype, device=rewards.device)
    next_step_valid = 1.0 - done_fp
    pm_next = player_mask_time_major[1:].to(dtype=value_dtype, device=rewards.device)
    pm_s = player_mask_time_major[:-1].to(dtype=value_dtype, device=rewards.device)
    discounting = _baseline_float_config_value(flags.discounting, "discounting")
    trace_lambda = _baseline_float_config_value(flags.lmb, "lmb")
    discounts = next_step_valid * discounting * pm_next

    log_rhos = combined_learner - combined_behavior

    v_ref = vtrace.from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        trace_lambda=trace_lambda,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    )
    v_det, detail = vtrace.from_importance_weights_detailed(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        trace_lambda=trace_lambda,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    )
    assert torch.allclose(v_ref.vs, v_det.vs, atol=1e-5, rtol=1e-5), (
        "from_importance_weights vs from_importance_weights_detailed mismatch"
    )
    assert torch.allclose(v_ref.pg_advantages, v_det.pg_advantages, atol=1e-5, rtol=1e-5), (
        "pg_advantages mismatch between vtrace paths"
    )

    upgo_vs = upgo.upgo(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        log_rhos=log_rhos,
        lmb=trace_lambda,
        clip_rho_threshold=1.0,
    ).vs

    lines: list[str] = []
    lines.append("IMPALA learner terminal V-trace dump (one batch lane, prefix of unroll)")
    lines.append(f"output_file={out_path!r}")
    lines.append(f"local_step={int(local_step)}")
    lines.append(f"enable_terminal_vtrace_dump={bool(flags.enable_terminal_vtrace_dump)}")
    lines.append(
        f"terminal_found: time_index t_done={t_done} batch_index b_done={b_done} player_index p_done={p_done}"
    )
    lines.append(f"flags.discounting.baseline={discounting} flags.lmb.baseline={trace_lambda}")
    lines.append(
        f"flags.enable_reward_ema_norm={bool(flags.enable_reward_ema_norm)} "
        f"enable_popart={bool(flags.enable_popart)}"
    )
    lines.append("")
    lines.append("Legend: behavior = rollout policy log_probs in batch; target = learner on same obs.")
    lines.append(
        "discounts = (1-done) * flags.discounting[reward_head] * player_mask[t+1] "
        "(see losses_func_selfplay)."
    )
    lines.append("")

    b = b_done
    lines.append(f"=== bootstrap_value (last rollout row, masked) for b={b} ===")
    boot = _cpu_f32(bootstrap_value[b])
    for p in range(int(boot.numel())):
        lines.append(f"  player {p}: {_f(float(boot[p].item()))}")
    lines.append("")

    for t in range(t_done + 1):
        lines.append(f"========== t={t}  b={b} ==========")
        lines.append("--- masks and discounts (per player) ---")
        pmn_row = _cpu_f32(pm_next[t, b])
        pms_row = _cpu_f32(pm_s[t, b])
        done_row = _cpu_f32(done_fp[t, b])
        nsv_row = _cpu_f32(next_step_valid[t, b])
        disc_row = _cpu_f32(discounts[t, b])
        for p in range(n_players):
            lines.append(
                f"  p={p}: done_fp={_f(float(done_row[p].item()))} "
                f"next_step_valid={_f(float(nsv_row[p].item()))} "
                f"pm_next={_f(float(pmn_row[p].item()))} "
                f"pm_s={_f(float(pms_row[p].item()))} "
                f"discount={_f(float(disc_row[p].item()))}"
            )

        lines.append("--- reward, baseline V (learner), targets ---")
        r_row = _cpu_f32(rewards[t, b])
        v_row = _cpu_f32(values[t, b])
        vs_row = _cpu_f32(detail["vs"][t, b])
        for p in range(n_players):
            extra = ""
            extra += f" offpolicy_upgo.vs={_f(float(_cpu_f32(upgo_vs[t, b, p]).item()))}"
            lines.append(
                f"  p={p}: reward={_f(float(r_row[p].item()))} "
                f"V_baseline={_f(float(v_row[p].item()))} "
                f"vtrace.vs={_f(float(vs_row[p].item()))}{extra}"
            )

        lines.append("--- policy log-probs (summed over action spaces in batch) ---")
        lb = _cpu_f32(combined_behavior[t, b])
        ll = _cpu_f32(combined_learner[t, b])
        lr = _cpu_f32(log_rhos[t, b])
        for p in range(n_players):
            lines.append(
                f"  p={p}: log_pi_behavior={_f(float(lb[p].item()))} "
                f"log_pi_target={_f(float(ll[p].item()))} "
                f"log_rho={_f(float(lr[p].item()))}"
            )

        lines.append("--- V-trace intermediates ---")
        rho = _cpu_f32(detail["rhos"][t, b])
        cr = _cpu_f32(detail["clipped_rhos"][t, b])
        cs = _cpu_f32(detail["cs"][t, b])
        vtp1 = _cpu_f32(detail["values_t_plus_1"][t, b])
        delt = _cpu_f32(detail["deltas"][t, b])
        vmv = _cpu_f32(detail["vs_minus_v_xs"][t, b])
        vst1 = _cpu_f32(detail["vs_t_plus_1"][t, b])
        cpg = _cpu_f32(detail["clipped_pg_rhos"][t, b])
        pga = _cpu_f32(detail["pg_advantages"][t, b])
        for p in range(n_players):
            lines.append(
                f"  p={p}: rho={_f(float(rho[p].item()))} clipped_rho={_f(float(cr[p].item()))} "
                f"c={_f(float(cs[p].item()))} "
                f"V_tp1={_f(float(vtp1[p].item()))} "
                f"delta={_f(float(delt[p].item()))} "
                f"vs_minus_V={_f(float(vmv[p].item()))} "
                f"vs_tp1={_f(float(vst1[p].item()))} "
                f"clipped_pg_rho={_f(float(cpg[p].item()))} "
                f"pg_advantage={_f(float(pga[p].item()))}"
            )
        lines.append("")

    _append_terminal_mask_dump(
        lines,
        t_done=t_done,
        b_done=b,
        n_players=n_players,
        taken_index=taken_index,
        avail_mask=avail_mask,
    )

    text = "\n".join(lines) + "\n"
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent != "":
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


__all__ = ["maybe_dump_terminal_selfplay_vtrace_and_exit"]
