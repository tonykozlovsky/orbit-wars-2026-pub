import secrets
from typing import Any

import torch

from .dict_io_contract import (
    maybe_validate_dict_io_contract_step_input,
    validated_dict_io_contract_output,
)
from .obs_wrapper import ORBIT_PLAYER_AXIS_SLOTS, orbit_active_policy_slots
from .wall_tree_profiler import WallTreeProfiler, profiler_span

_ORBIT_FLEET_DELTA_WEIGHT = 0.0
_ORBIT_FLEET_DELTA_VS_BEST_OPPONENT_WEIGHT = 0.0

_ENV_STEP_WEIGHT = 0.0
_PLANETS_DELTA_WEIGHT = 0.0

_PRODUCTION_DELTA_WEIGHT = 0.0
_PRODUCTION_DELTA_VS_BEST_OPPONENT_WEIGHT = 0.0

_GAME_RESULT_WIN_REWARD = 1.1
_GAME_RESULT_LOSS_REWARD = -1.1

_EARLY_GAME_RESULT_WIN_WEIGHT = 0.0
_EARLY_GAME_RESULT_LOSS_WEIGHT = 0.0

_EARLY_GAME_RESULT_STEP_CAP = 500
_ESTIMATED_POWER_HORIZON = 50
_ESTIMATED_POWER_SMOOTH_STEPS = 50
_ESTIMATED_POWER_EARLY_STOP_MIN_STEP = 50
_ESTIMATED_POWER_EARLY_STOP_TOP_RATIO = 2
_BASELINE_REWARD_HEAD = "baseline"
_PRODUCTION_DELTA_REWARD_HEAD = "production_delta"


def _orbit_episode_steps_cap_from_padding_inner(padding_env: Any) -> int:
    """``OrbitPaddingWrapper`` wraps ``OrbitWarsEnv``; use its resolved ``episodeSteps`` (not ``flags``)."""
    assert hasattr(padding_env, "env"), type(padding_env)
    orbit = padding_env.env
    cap = int(orbit.episode_step_limit)
    assert cap >= 1, cap
    return cap


def _episode_step_scalar_from_out(out: dict[str, Any]) -> int:
    raw = out["obs_raw"]
    assert isinstance(raw, dict), type(raw)
    es = raw["episode_step"]
    assert isinstance(es, torch.Tensor), type(es)
    assert es.ndim == 1 and int(es.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS, es.shape
    return int(es[0].item())


def total_reward_from_metrics(metrics: dict[str, Any]) -> torch.Tensor:
    fleet = metrics["orbit_fleet_delta"]
    fleet_vs_best = metrics["orbit_fleet_delta_vs_best_opponent"]
    env_step = metrics["env_step"]
    planets_delta = metrics["planets_delta"]
    production_delta = metrics["production_delta"]
    production_vs_best = metrics["production_delta_vs_best_opponent"]
    game_result = metrics["game_result"]
    early_game_result = metrics["early_game_result"]
    assert isinstance(fleet, torch.Tensor) and isinstance(env_step, torch.Tensor)
    assert isinstance(fleet_vs_best, torch.Tensor)
    assert isinstance(planets_delta, torch.Tensor)
    assert isinstance(production_delta, torch.Tensor)
    assert isinstance(production_vs_best, torch.Tensor)
    assert isinstance(game_result, torch.Tensor)
    assert isinstance(early_game_result, torch.Tensor)
    assert tuple(fleet.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(fleet_vs_best.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(env_step.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(planets_delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(production_delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(production_vs_best.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(early_game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    win_result = torch.clamp(game_result, min=0.0)
    loss_result = torch.clamp(-game_result, min=0.0)
    early_win_result = early_game_result * win_result
    early_loss_result = early_game_result * loss_result
    return (
        _ORBIT_FLEET_DELTA_WEIGHT * fleet
        + _ORBIT_FLEET_DELTA_VS_BEST_OPPONENT_WEIGHT * fleet_vs_best
        + _ENV_STEP_WEIGHT * env_step
        + _PLANETS_DELTA_WEIGHT * planets_delta
        + _PRODUCTION_DELTA_WEIGHT * production_delta
        + _PRODUCTION_DELTA_VS_BEST_OPPONENT_WEIGHT * production_vs_best
        + _GAME_RESULT_WIN_REWARD * win_result
        + _GAME_RESULT_LOSS_REWARD * loss_result
        + _EARLY_GAME_RESULT_WIN_WEIGHT * early_win_result
        + _EARLY_GAME_RESULT_LOSS_WEIGHT * early_loss_result
    )


def rewards_from_metrics(metrics: dict[str, Any]) -> dict[str, torch.Tensor]:
    baseline_reward = total_reward_from_metrics(metrics)
    production_delta_reward = production_delta_reward_from_metrics(metrics)
    assert isinstance(baseline_reward, torch.Tensor)
    assert tuple(baseline_reward.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert isinstance(production_delta_reward, torch.Tensor)
    assert tuple(production_delta_reward.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    return {
        _BASELINE_REWARD_HEAD: baseline_reward,
        _PRODUCTION_DELTA_REWARD_HEAD: production_delta_reward,
    }


def production_delta_reward_from_metrics(metrics: dict[str, Any]) -> torch.Tensor:
    production_delta = metrics["production_delta"]
    assert isinstance(production_delta, torch.Tensor)
    assert tuple(production_delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert production_delta.dtype == torch.float32, production_delta.dtype
    return production_delta


def early_game_result_from_step(game_result: torch.Tensor, episode_step: int) -> torch.Tensor:
    assert tuple(game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert 0 <= int(episode_step) <= _EARLY_GAME_RESULT_STEP_CAP, episode_step
    assert bool(
        torch.all((game_result == 0.0) | (game_result == 1.0) | (game_result == -1.0)).item()
    ), game_result
    step_fraction = float(int(episode_step)) / float(_EARLY_GAME_RESULT_STEP_CAP)
    remaining_fraction = float(_EARLY_GAME_RESULT_STEP_CAP - int(episode_step)) / float(
        _EARLY_GAME_RESULT_STEP_CAP
    )
    win_result = torch.clamp(game_result, min=0.0)
    loss_result = torch.clamp(-game_result, min=0.0)
    return win_result * remaining_fraction + loss_result * step_fraction


def delta_vs_best_opponent_by_strength(
    delta: torch.Tensor,
    strength: torch.Tensor,
    *,
    num_agents: int,
) -> torch.Tensor:
    assert isinstance(delta, torch.Tensor) and isinstance(strength, torch.Tensor)
    assert tuple(delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(strength.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert delta.dtype == torch.float32, delta.dtype
    assert strength.dtype == torch.float32, strength.dtype
    assert delta.device == strength.device, (delta.device, strength.device)
    active_slots = orbit_active_policy_slots(int(num_agents))
    out = torch.zeros_like(delta)
    for slot in active_slots:
        opponent_slots = tuple(s for s in active_slots if s != slot)
        assert len(opponent_slots) == int(num_agents) - 1, (slot, opponent_slots, num_agents)
        opponent_strength = strength[list(opponent_slots)]
        best_opponent = opponent_slots[int(torch.argmax(opponent_strength).item())]
        out[slot] = delta[slot] - delta[best_opponent]
    return out


def estimated_power_from_raw(
    raw: dict[str, Any],
    *,
    episode_step: int,
    episode_cap: int,
) -> torch.Tensor:
    fleet_total = raw["orbit_fleet_total"]
    production_total = raw["production_total"]
    assert isinstance(fleet_total, torch.Tensor) and isinstance(production_total, torch.Tensor)
    assert tuple(fleet_total.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(production_total.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert fleet_total.dtype == torch.float32, fleet_total.dtype
    assert production_total.dtype == torch.float32, production_total.dtype
    assert fleet_total.device == production_total.device, (fleet_total.device, production_total.device)
    remaining_steps = int(episode_cap) - int(episode_step)
    assert 0 <= remaining_steps <= int(episode_cap), (episode_step, episode_cap)
    estimate_horizon = min(int(_ESTIMATED_POWER_HORIZON), remaining_steps)
    return fleet_total + production_total * float(estimate_horizon)


def estimated_power_early_stop_triggered(
    estimated_power_smooth: torch.Tensor,
    *,
    already_credited_by_slot: list[bool],
    current_game_result: torch.Tensor,
    active_slots: tuple[int, ...],
    episode_step: int,
    env_step: torch.Tensor,
) -> bool:
    assert isinstance(estimated_power_smooth, torch.Tensor)
    assert isinstance(current_game_result, torch.Tensor)
    assert isinstance(env_step, torch.Tensor)
    assert tuple(estimated_power_smooth.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(current_game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(env_step.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert estimated_power_smooth.dtype == torch.float32, estimated_power_smooth.dtype
    assert current_game_result.dtype == torch.float32, current_game_result.dtype
    assert env_step.dtype == torch.float32, env_step.dtype
    assert len(already_credited_by_slot) == ORBIT_PLAYER_AXIS_SLOTS
    if int(episode_step) < int(_ESTIMATED_POWER_EARLY_STOP_MIN_STEP):
        return False
    if float(env_step[active_slots[0]].item()) != 1.0:
        return False
    eligible_slots = tuple(
        slot
        for slot in active_slots
        if not bool(already_credited_by_slot[slot])
        and abs(float(current_game_result[slot].item()) + 1.0) >= 1e-5
    )
    if len(eligible_slots) < 2:
        return False
    eligible_power = estimated_power_smooth[list(eligible_slots)]
    assert bool(torch.all(eligible_power >= 0.0).item()), eligible_power
    top_two = torch.topk(eligible_power, k=2, largest=True, sorted=True)
    ratio = top_two.values[0] / top_two.values[1]
    return bool((ratio >= float(_ESTIMATED_POWER_EARLY_STOP_TOP_RATIO)).item())


def early_stop_game_result_from_estimated_power(
    estimated_power_smooth: torch.Tensor,
    *,
    already_credited_by_slot: list[bool],
    current_game_result: torch.Tensor,
    active_slots: tuple[int, ...],
) -> torch.Tensor:
    assert isinstance(estimated_power_smooth, torch.Tensor)
    assert isinstance(current_game_result, torch.Tensor)
    assert tuple(estimated_power_smooth.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(current_game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert estimated_power_smooth.dtype == torch.float32, estimated_power_smooth.dtype
    assert current_game_result.dtype == torch.float32, current_game_result.dtype
    assert len(already_credited_by_slot) == ORBIT_PLAYER_AXIS_SLOTS
    out = torch.zeros_like(current_game_result)
    eligible_slots = tuple(
        slot
        for slot in active_slots
        if not bool(already_credited_by_slot[slot])
        and abs(float(current_game_result[slot].item()) + 1.0) >= 1e-5
    )
    assert len(eligible_slots) >= 2, (eligible_slots, already_credited_by_slot, current_game_result)
    eligible_power = estimated_power_smooth[list(eligible_slots)]
    assert bool(torch.all(eligible_power >= 0.0).item()), eligible_power
    top_power = torch.max(eligible_power)
    winners = tuple(
        slot for slot in eligible_slots if bool((estimated_power_smooth[slot] == top_power).item())
    )
    assert len(winners) >= 1, (eligible_slots, estimated_power_smooth)
    for slot in active_slots:
        if abs(float(current_game_result[slot].item()) + 1.0) < 1e-5:
            out[slot] = -1.0
        elif bool(already_credited_by_slot[slot]):
            out[slot] = 0.0
        elif slot in winners:
            out[slot] = 1.0
        else:
            out[slot] = -1.0
    return out


class RewardWrapper:
    """Adds ``reward``, ``done``, and ``desync_done``.

    ``reward`` is a dict of named ``(ORBIT_PLAYER_AXIS_SLOTS,)`` reward heads. ``reward["baseline"]``
    and ``metrics["weighted_reward"]`` are the current weighted sum of ``orbit_fleet_delta``,
    ``env_step``, ``planets_delta``, ``production_delta`` and ``game_result`` (see module-level
    weights); inactive slots are zero. ``done`` uses slot 0 of length-``ORBIT_PLAYER_AXIS_SLOTS``
    ``orbit_episode_done`` (all slots equal; local terminal: episode end or game-over such as a
    single remaining planet owner in orbit_wars).

    If ``flags.enable_desync``, after each ``reset()`` (until the first desync has fired) a new target
    ``T`` is drawn uniformly from ``1 .. episodeSteps-1`` (``episodeSteps`` from the inner
    ``OrbitWarsEnv``). The first time ``obs_raw["episode_step"]`` equals ``T`` on a returned state,
    ``desync_done`` is true once; later resets do not resample and ``desync_done`` stays false.
    (``T >= 1`` so reset with ``episode_step == 0`` never desyncs.)
    """

    def __init__(self, env: Any, flags: Any, wall_profiler: WallTreeProfiler | None = None) -> None:
        self.env = env
        self.flags = flags
        self._wall_prof = wall_profiler
        self._desync_fired = False
        self._desync_target_episode_step = 0
        self._desync_episode_cap = 0
        self._episode_cap = 0
        self._estimated_power_history: list[torch.Tensor] = []
        self._game_result_credited_by_slot = [False] * ORBIT_PLAYER_AXIS_SLOTS
        cap = _orbit_episode_steps_cap_from_padding_inner(self.env)
        assert cap == _EARLY_GAME_RESULT_STEP_CAP, (cap, _EARLY_GAME_RESULT_STEP_CAP)
        self._episode_cap = int(cap)
        if bool(self.flags.enable_desync):
            assert cap >= 2, (
                "enable_desync requires resolved episodeSteps >= 2 so desync targets lie on post-step states",
                cap,
            )
            self._desync_episode_cap = int(cap)

    def _wall_span(self, name: str):
        return profiler_span(self._wall_prof, name)

    def _resample_desync_target(self) -> None:
        if not bool(self.flags.enable_desync) or self._desync_fired:
            return
        cap = int(self._desync_episode_cap)
        assert cap >= 2, cap
        # OS RNG (not Python ``random`` / torch): independent draws across forked actor processes.
        self._desync_target_episode_step = int(secrets.randbelow(cap - 1)) + 1

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        out = self.env.reset(**kwargs)
        assert _episode_step_scalar_from_out(out) == 0
        self._estimated_power_history.clear()
        self._game_result_credited_by_slot = [False] * ORBIT_PLAYER_AXIS_SLOTS
        self._resample_desync_target()
        return self._with_reward_done(out)

    def step(self, actions: Any) -> dict[str, Any]:
        with self._wall_span("wrap_reward_validate_input"):
            maybe_validate_dict_io_contract_step_input(
                self.flags, actions, "orbit_wars_env_step_actions"
            )
        with self._wall_span("wrap_reward_inner"):
            out = self.env.step(actions)
        with self._wall_span("wrap_reward_merge"):
            return self._with_reward_done(out)

    def _estimated_power_smooth(self, estimated_power: torch.Tensor) -> torch.Tensor:
        assert isinstance(estimated_power, torch.Tensor)
        assert tuple(estimated_power.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
        assert estimated_power.dtype == torch.float32, estimated_power.dtype
        self._estimated_power_history.append(estimated_power.clone())
        if len(self._estimated_power_history) > int(_ESTIMATED_POWER_SMOOTH_STEPS):
            self._estimated_power_history.pop(0)
        assert 1 <= len(self._estimated_power_history) <= int(_ESTIMATED_POWER_SMOOTH_STEPS)
        history = torch.stack(self._estimated_power_history, dim=0)
        assert tuple(history.shape) == (
            len(self._estimated_power_history),
            ORBIT_PLAYER_AXIS_SLOTS,
        )
        return torch.mean(history, dim=0)

    def _with_reward_done(self, out: dict[str, Any]) -> dict[str, Any]:
        m = out["metrics"]
        raw = out["obs_raw"]
        assert isinstance(raw, dict), type(raw)
        game_result = m["game_result"]
        assert isinstance(game_result, torch.Tensor)
        na = int(self.flags.orbit_num_agents)
        assert na in (2, 4), na
        active_slots = orbit_active_policy_slots(na)
        episode_step = _episode_step_scalar_from_out(out)
        estimated_power = estimated_power_from_raw(
            raw,
            episode_step=episode_step,
            episode_cap=int(self._episode_cap),
        )
        estimated_power_smooth = self._estimated_power_smooth(estimated_power)
        od = out["orbit_episode_done"]
        assert isinstance(od, torch.Tensor)
        assert od.ndim == 1 and int(od.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS
        assert bool(torch.all(od[list(active_slots)] == od[0]).item()), od
        inactive = torch.ones((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.bool, device=od.device)
        inactive[list(active_slots)] = False
        assert bool(torch.all(od[inactive] == 0).item()), od
        done_b = bool(float(od[0].item()) == 1.0)
        early_stop_b = False
        if not done_b and bool(self.flags.enable_orbit_estimated_power_early_stop):
            early_stop_b = estimated_power_early_stop_triggered(
                estimated_power_smooth,
                already_credited_by_slot=self._game_result_credited_by_slot,
                current_game_result=game_result,
                active_slots=active_slots,
                episode_step=episode_step,
                env_step=m["env_step"],
            )
        if early_stop_b:
            game_result = early_stop_game_result_from_estimated_power(
                estimated_power_smooth,
                already_credited_by_slot=self._game_result_credited_by_slot,
                current_game_result=game_result,
                active_slots=active_slots,
            )
            od = torch.zeros_like(od)
            od[list(active_slots)] = 1.0
            done_b = True
        estimated_power_metric = estimated_power if done_b else torch.zeros_like(estimated_power)
        metrics = {
            **m,
            "game_result": game_result,
            "orbit_fleet_delta_vs_best_opponent": delta_vs_best_opponent_by_strength(
                m["orbit_fleet_delta"],
                raw["orbit_fleet_total"],
                num_agents=na,
            ),
            "production_delta_vs_best_opponent": delta_vs_best_opponent_by_strength(
                m["production_delta"],
                raw["production_total"],
                num_agents=na,
            ),
            "early_game_result": early_game_result_from_step(
                game_result,
                episode_step,
            ),
            "estimated_power": estimated_power_metric,
        }
        reward = rewards_from_metrics(metrics)
        assert _BASELINE_REWARD_HEAD in reward, sorted(reward.keys())
        baseline_reward = reward[_BASELINE_REWARD_HEAD]
        assert isinstance(baseline_reward, torch.Tensor)
        assert tuple(baseline_reward.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
        original_player_mask = torch.zeros(
            (ORBIT_PLAYER_AXIS_SLOTS,),
            dtype=torch.float32,
            device=baseline_reward.device,
        )
        original_player_mask[list(active_slots)] = 1.0
        metrics = {**metrics, "weighted_reward": baseline_reward}
        for slot in active_slots:
            if abs(float(game_result[slot].item()) + 1.0) < 1e-5:
                self._game_result_credited_by_slot[slot] = True
        done = torch.tensor(done_b, dtype=torch.bool)
        desync_b = False
        if bool(self.flags.enable_desync) and not self._desync_fired:
            if episode_step == int(self._desync_target_episode_step):
                desync_b = True
                self._desync_fired = True
        desync_done = torch.tensor(desync_b, dtype=torch.bool)
        merged = {
            **out,
            "orbit_episode_done": od,
            "metrics": metrics,
            "reward": reward,
            "done": done,
            "desync_done": desync_done,
            "original_player_mask_STAT": original_player_mask,
        }
        return validated_dict_io_contract_output(self.flags, merged, "reward_wrapper_output")
