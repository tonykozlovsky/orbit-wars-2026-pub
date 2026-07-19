import logging
import multiprocessing as mp
import os
import queue
import random
import time
from multiprocessing.queues import Queue as MPQueue
from types import SimpleNamespace
from typing import Any

import setproctitle
import torch

from ...configs import ImpalaTrainingConfig
from ...gym.create_env import create_env
from ...gym.dict_io_contract import maybe_validate_dict_io_contract
from ...gym.obs_wrapper import (
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
)
from ...gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from .buffer_utils import (
    copy_buffers,
    copy_matching_tree_into,
    fill_buffers_inplace_2,
    get_buffers_with_tag,
)
from .common import (
    configure_process_cpu_thread_limits,
    event_wait_or_stop,
    orbit_model_by_player_axis_from_seats,
    queue_get_or_stop,
    queue_put_or_stop,
    raise_if_stop_requested,
    set_stop_event_with_reason,
    StopRequested,
)


_BASELINE_REWARD_HEAD = "baseline"
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
    return float(mapping[_BASELINE_REWARD_HEAD])


def _baseline_reward_from_learn_stat(reward_learn_stat: dict[str, Any]) -> torch.Tensor:
    assert isinstance(reward_learn_stat, dict)
    assert _BASELINE_REWARD_HEAD in reward_learn_stat, sorted(reward_learn_stat.keys())
    reward = reward_learn_stat[_BASELINE_REWARD_HEAD]
    assert isinstance(reward, torch.Tensor)
    return reward


def _baseline_value_from_learn_output(baseline_learn: dict[str, Any]) -> torch.Tensor:
    assert isinstance(baseline_learn, dict)
    assert _BASELINE_REWARD_HEAD in baseline_learn, sorted(baseline_learn.keys())
    value = baseline_learn[_BASELINE_REWARD_HEAD]
    assert isinstance(value, torch.Tensor)
    return value


def _apply_desync_warmup_flags(mask: list[bool] | None, dd: torch.Tensor) -> None:
    if mask is None:
        return
    assert isinstance(dd, torch.Tensor) and dd.dtype == torch.bool
    v = dd.reshape(-1)
    assert tuple(v.shape) == (len(mask),), (tuple(v.shape), len(mask))
    for i in range(len(mask)):
        if bool(v[i].item()):
            mask[i] = True


def _run_model_from_inputs(
    inference_inputs: dict,
    model_by_seat: torch.Tensor,
    infer_req_queue: MPQueue,
    infer_res_queue: MPQueue,
    infer_req_free_queue: MPQueue,
    infer_request_buffers: list,
    infer_result_buffers: list,
    actor_index_for_buffers: int,
    stop_event,
    flags: ImpalaTrainingConfig,
) -> dict:
    req_idx = queue_get_or_stop(infer_req_free_queue, stop_event, timeout_sec=0.1)

    copy_matching_tree_into(infer_request_buffers[req_idx], inference_inputs)
    maybe_validate_dict_io_contract(
        flags, infer_request_buffers[req_idx], "infer_request_buffer_template"
    )
    queue_put_or_stop(
        infer_req_queue,
        (actor_index_for_buffers, req_idx, model_by_seat),
        stop_event,
        timeout_sec=0.1,
    )
    res_idx = queue_get_or_stop(infer_res_queue, stop_event, timeout_sec=0.1)
    assert int(res_idx) == int(req_idx), "inference must return the same buffer index as the request"
    out = infer_result_buffers[res_idx]
    assert isinstance(out, dict)
    maybe_validate_dict_io_contract(flags, out, "infer_result_buffer_template")
    agent_out = copy_buffers(out)
    queue_put_or_stop(infer_req_free_queue, req_idx, stop_event, timeout_sec=0.1)
    return agent_out


def _env_for_learner_player_mask(
    env_out: dict[str, Any],
    learner_player_mask: torch.Tensor,
) -> dict[str, Any]:
    obs = env_out["obs_LEARN_INFER"]
    assert isinstance(obs, dict)
    player_mask = obs["player_mask"]
    assert isinstance(player_mask, torch.Tensor)
    assert tuple(player_mask.shape) == tuple(learner_player_mask.shape), (
        tuple(player_mask.shape),
        tuple(learner_player_mask.shape),
    )
    learn_obs = dict(obs)
    learn_obs["player_mask"] = player_mask * learner_player_mask.to(
        dtype=player_mask.dtype,
        device=player_mask.device,
    )
    return {**env_out, "obs_LEARN_INFER": learn_obs}


def _learn_payload(
    env_before: dict[str, Any],
    agent_out: dict[str, torch.Tensor],
    env_after: dict[str, Any],
    game_num_players_learn: torch.Tensor,
    model_by_player_axis_learn: torch.Tensor,
    flags: ImpalaTrainingConfig,
) -> dict[str, Any]:
    # Transition (s_t, a_t, r_{t+1}, s_{t+1}): policy and ``obs_LEARN_INFER`` are on s_t (``env_before``).
    # Post-step taken action is the remapped top-level ``action_taken_index_LEARN_STAT``.
    obs_before = env_before["obs_LEARN_INFER"]
    assert isinstance(obs_before, dict)
    taken = env_after["action_taken_index_LEARN_STAT"]
    assert isinstance(taken, torch.Tensor)
    reward_learn_stat = env_after["reward_LEARN_STAT"]
    reward = _baseline_reward_from_learn_stat(reward_learn_stat)
    done = env_after["done_LEARN_STAT"]
    assert isinstance(done, torch.Tensor) and done.dtype == torch.bool
    assert reward.ndim == 2
    assert done.shape == reward.shape[:1], (done.shape, reward.shape)
    done_per_player = done.unsqueeze(-1).expand_as(reward)
    assert isinstance(model_by_player_axis_learn, torch.Tensor)
    assert model_by_player_axis_learn.dtype == torch.int64
    assert tuple(model_by_player_axis_learn.shape) == (
        int(reward.shape[0]),
        int(reward.shape[1]),
        ORBIT_PLAYER_AXIS_SLOTS,
    ), (tuple(model_by_player_axis_learn.shape), tuple(reward.shape))
    policy_agent_out = dict(agent_out)
    assert "baseline_LEARN" in policy_agent_out
    del policy_agent_out["baseline_LEARN"]
    merged: dict[str, Any] = {
        **policy_agent_out,
        "reward_LEARN_STAT": reward_learn_stat,
        "done_LEARN_STAT": done_per_player,
        "obs_LEARN_INFER": obs_before,
        "action_taken_index_LEARN_STAT": taken,
        "game_num_players_LEARN": game_num_players_learn,
        "frozen_model_by_player_axis_LEARN": model_by_player_axis_learn,
    }
    out = get_buffers_with_tag(
        merged,
        device=None,
        tag="LEARN",
    )
    assert out is not None
    return out


def _bootstrap_learn_payload(
    env_terminal: dict[str, Any],
    policy_actions_from_last_step: dict[str, torch.Tensor],
    reward_done_source: dict[str, Any],
    game_num_players_learn: torch.Tensor,
    model_by_player_axis_learn: torch.Tensor,
    flags: ImpalaTrainingConfig,
) -> dict[str, Any]:
    """Last time index: ``obs``/availability from ``env_terminal``; r/d from ``reward_done_source`` (last ``env_next``).

    ``action_taken_index_LEARN_STAT`` comes from the remapped env dict.
    """
    obs_terminal = env_terminal["obs_LEARN_INFER"]
    assert isinstance(obs_terminal, dict)
    taken = reward_done_source["action_taken_index_LEARN_STAT"]
    assert isinstance(taken, torch.Tensor)
    reward_learn_stat = reward_done_source["reward_LEARN_STAT"]
    reward = _baseline_reward_from_learn_stat(reward_learn_stat)
    done = reward_done_source["done_LEARN_STAT"]
    assert isinstance(done, torch.Tensor) and done.dtype == torch.bool
    assert reward.ndim == 2
    assert done.shape == reward.shape[:1], (done.shape, reward.shape)
    done_per_player = done.unsqueeze(-1).expand_as(reward)
    assert isinstance(model_by_player_axis_learn, torch.Tensor)
    assert model_by_player_axis_learn.dtype == torch.int64
    assert tuple(model_by_player_axis_learn.shape) == (
        int(reward.shape[0]),
        int(reward.shape[1]),
        ORBIT_PLAYER_AXIS_SLOTS,
    ), (tuple(model_by_player_axis_learn.shape), tuple(reward.shape))
    policy_actions = dict(policy_actions_from_last_step)
    assert "baseline_LEARN" in policy_actions
    del policy_actions["baseline_LEARN"]
    merged = {
        **policy_actions,
        "reward_LEARN_STAT": reward_learn_stat,
        "done_LEARN_STAT": done_per_player,
        "obs_LEARN_INFER": obs_terminal,
        "action_taken_index_LEARN_STAT": taken,
        "game_num_players_LEARN": game_num_players_learn,
        "frozen_model_by_player_axis_LEARN": model_by_player_axis_learn,
    }
    out = get_buffers_with_tag(
        merged,
        device=None,
        tag="LEARN",
    )
    assert out is not None
    return out


def _game_num_players_learn_tensor(game_num_players_by_env: torch.Tensor) -> torch.Tensor:
    assert isinstance(game_num_players_by_env, torch.Tensor)
    assert game_num_players_by_env.dtype == torch.int64
    assert game_num_players_by_env.ndim == 1, tuple(game_num_players_by_env.shape)
    out = torch.zeros(
        (int(game_num_players_by_env.shape[0]), ORBIT_PLAYER_AXIS_SLOTS),
        dtype=torch.int64,
    )
    for env_i in range(int(game_num_players_by_env.shape[0])):
        na = int(game_num_players_by_env[env_i].item())
        assert na in (2, 4), na
        for slot in orbit_active_policy_slots(na):
            out[env_i, int(slot)] = na
    return out


def _sample_game_num_players_by_env(
    rng: random.Random,
    probability_4p: float,
    n_actor_envs: int,
) -> torch.Tensor:
    p4 = float(probability_4p)
    assert 0.0 <= p4 <= 1.0, p4
    out = torch.empty((int(n_actor_envs),), dtype=torch.int64)
    for env_i in range(int(n_actor_envs)):
        out[env_i] = 4 if rng.random() < p4 else 2
    return out


def _reset_probability_4p_from_target_sample_ratio(target_4p_sample_ratio: float) -> float:
    ratio = float(target_4p_sample_ratio)
    assert 0.0 <= ratio <= 1.0, ratio
    return ratio / (2.0 - ratio)


def _stats_payload(
    env_after: dict[str, Any],
    model_by_seat: torch.Tensor,
    critic_mc_stat: dict[str, torch.Tensor],
) -> dict[str, Any]:
    out = get_buffers_with_tag(env_after, device=None, tag="STAT")
    assert out is not None
    assert isinstance(model_by_seat, torch.Tensor)
    reward = _baseline_reward_from_learn_stat(env_after["reward_LEARN_STAT"])
    assert tuple(model_by_seat.shape) == (
        int(reward.shape[0]),
        ORBIT_PLAYER_AXIS_SLOTS,
    ), tuple(model_by_seat.shape)
    out["frozen_model_by_seat_STAT"] = model_by_seat
    out["critic_mc_STAT"] = critic_mc_stat
    return out


def _empty_critic_mc_stat(n_actor_envs: int) -> dict[str, torch.Tensor]:
    shape = (int(n_actor_envs), ORBIT_PLAYER_AXIS_SLOTS)
    return {
        "valid": torch.zeros(shape, dtype=torch.bool),
        "count": torch.zeros(shape, dtype=torch.float32),
        "sqerr_sum": torch.zeros(shape, dtype=torch.float32),
        "abserr_sum": torch.zeros(shape, dtype=torch.float32),
        "err_sum": torch.zeros(shape, dtype=torch.float32),
        "return_sum": torch.zeros(shape, dtype=torch.float32),
        "return_sq_sum": torch.zeros(shape, dtype=torch.float32),
        "value_sum": torch.zeros(shape, dtype=torch.float32),
        "value_sq_sum": torch.zeros(shape, dtype=torch.float32),
        "value_return_sum": torch.zeros(shape, dtype=torch.float32),
        "first_error": torch.zeros(shape, dtype=torch.float32),
    }


class _CriticMcEpisodeTracker:
    def __init__(self, *, n_actor_envs: int, discounting: float) -> None:
        self._n_actor_envs = int(n_actor_envs)
        self._discounting = float(discounting)
        self._values: list[list[torch.Tensor]] = [[] for _ in range(self._n_actor_envs)]
        self._rewards: list[list[torch.Tensor]] = [[] for _ in range(self._n_actor_envs)]
        self._discounts: list[list[torch.Tensor]] = [[] for _ in range(self._n_actor_envs)]
        self._learn_masks: list[list[torch.Tensor]] = [[] for _ in range(self._n_actor_envs)]

    def update(
        self,
        *,
        baseline: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        done_reset: torch.Tensor,
        learner_player_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert isinstance(baseline, torch.Tensor)
        assert tuple(baseline.shape) == (
            self._n_actor_envs,
            ORBIT_PLAYER_AXIS_SLOTS,
        ), tuple(baseline.shape)
        assert isinstance(reward, torch.Tensor)
        assert tuple(reward.shape) == tuple(baseline.shape), (
            tuple(reward.shape),
            tuple(baseline.shape),
        )
        assert isinstance(done, torch.Tensor) and done.dtype == torch.bool
        assert tuple(done.shape) == (self._n_actor_envs,), tuple(done.shape)
        assert isinstance(done_reset, torch.Tensor) and done_reset.dtype == torch.bool
        assert tuple(done_reset.shape) == tuple(done.shape), (
            tuple(done_reset.shape),
            tuple(done.shape),
        )
        assert isinstance(learner_player_mask, torch.Tensor)
        assert tuple(learner_player_mask.shape) == tuple(baseline.shape), (
            tuple(learner_player_mask.shape),
            tuple(baseline.shape),
        )

        baseline_cpu = baseline.detach().to(device="cpu", dtype=torch.float32)
        reward_cpu = reward.detach().to(device="cpu", dtype=torch.float32)
        done_cpu = done.detach().to(device="cpu")
        done_reset_cpu = done_reset.detach().to(device="cpu")
        learn_mask_cpu = (
            learner_player_mask.detach().to(device="cpu", dtype=torch.float32) > 0.5
        )
        assert torch.isfinite(baseline_cpu).all().item(), baseline_cpu
        assert torch.isfinite(reward_cpu).all().item(), reward_cpu

        discount_cpu = torch.full_like(reward_cpu, self._discounting)
        discount_cpu[done_cpu] = 0.0

        out = _empty_critic_mc_stat(self._n_actor_envs)
        for env_i in range(self._n_actor_envs):
            self._values[env_i].append(baseline_cpu[env_i].clone())
            self._rewards[env_i].append(reward_cpu[env_i].clone())
            self._discounts[env_i].append(discount_cpu[env_i].clone())
            self._learn_masks[env_i].append(learn_mask_cpu[env_i].clone())

            if bool(done_cpu[env_i].item()):
                self._write_terminal_stats(out, env_i)

            if bool(done_reset_cpu[env_i].item()):
                self._values[env_i].clear()
                self._rewards[env_i].clear()
                self._discounts[env_i].clear()
                self._learn_masks[env_i].clear()

        return out

    def _write_terminal_stats(
        self,
        out: dict[str, torch.Tensor],
        env_i: int,
    ) -> None:
        values = torch.stack(self._values[int(env_i)], dim=0)
        rewards = torch.stack(self._rewards[int(env_i)], dim=0)
        discounts = torch.stack(self._discounts[int(env_i)], dim=0)
        learn_masks = torch.stack(self._learn_masks[int(env_i)], dim=0)
        assert values.ndim == 2 and int(values.shape[1]) == ORBIT_PLAYER_AXIS_SLOTS
        assert tuple(rewards.shape) == tuple(values.shape), (
            tuple(rewards.shape),
            tuple(values.shape),
        )
        assert tuple(discounts.shape) == tuple(values.shape), (
            tuple(discounts.shape),
            tuple(values.shape),
        )
        assert tuple(learn_masks.shape) == tuple(values.shape), (
            tuple(learn_masks.shape),
            tuple(values.shape),
        )

        returns = torch.empty_like(rewards)
        running = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32)
        for t in range(int(rewards.shape[0]) - 1, -1, -1):
            running = rewards[t] + discounts[t] * running
            returns[t] = running

        assert torch.isfinite(returns).all().item(), returns
        mask_f = learn_masks.to(dtype=torch.float32)
        count = mask_f.sum(dim=0)
        err = values - returns
        out["valid"][int(env_i)] = count > 0.0
        out["count"][int(env_i)] = count
        out["sqerr_sum"][int(env_i)] = ((err * err) * mask_f).sum(dim=0)
        out["abserr_sum"][int(env_i)] = (err.abs() * mask_f).sum(dim=0)
        out["err_sum"][int(env_i)] = (err * mask_f).sum(dim=0)
        out["return_sum"][int(env_i)] = (returns * mask_f).sum(dim=0)
        out["return_sq_sum"][int(env_i)] = ((returns * returns) * mask_f).sum(dim=0)
        out["value_sum"][int(env_i)] = (values * mask_f).sum(dim=0)
        out["value_sq_sum"][int(env_i)] = ((values * values) * mask_f).sum(dim=0)
        out["value_return_sum"][int(env_i)] = ((values * returns) * mask_f).sum(dim=0)
        out["first_error"][int(env_i)] = err[0] * learn_masks[0].to(dtype=torch.float32)


def act_rollout_func(
    flags: ImpalaTrainingConfig,
    actor_index_for_env: int,
    actor_index_for_buffers: int,
    free_queue: MPQueue,
    full_queue: MPQueue,
    full_queue_lock: mp.Lock,
    buffers,
    stats_free_queue: MPQueue,
    stats_full_queue: MPQueue,
    stats_buffers,
    infer_req_queue: MPQueue,
    infer_res_queue: MPQueue,
    infer_req_free_queue: MPQueue,
    infer_request_buffers,
    infer_result_buffers,
    name: str,
    start_event,
    stop_event,
    visualization_queue: MPQueue | None,
    actor_desync_done_env_count: Any,
    set_process_title_on_start: bool = True,
) -> None:
    try:
        if bool(set_process_title_on_start):
            setproctitle.setproctitle(name)
            configure_process_cpu_thread_limits()
        logging.info(
            "%s start env=%s buffer_client=%s orbit_num_agents=%s",
            name,
            actor_index_for_env,
            actor_index_for_buffers,
            int(flags.orbit_num_agents),
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        while not event_wait_or_stop(start_event, stop_event, timeout_sec=0.1):
            pass

        record_tape = (
            visualization_queue is not None and bool(flags.enable_rl_actor_visualization)
        )
        wall_prof = (
            WallTreeProfiler()
            if bool(flags.enable_actor_wall_tree_profiler) and int(actor_index_for_env) == 0
            else None
        )
        wall_prof_rollout_ix = 0
        _envs_val = bool(flags.enable_envs_validation)
        env = create_env(
            flags,
            device="cpu",
            visualize=record_tape,
            visualize_sim_env=record_tape,
            visualization_queue=visualization_queue if record_tape else None,
            record_tape=record_tape,
            cpp_env_obs_full=not _envs_val,
            cpp_env_obs_validate=_envs_val,
            wall_profiler=wall_prof,
        )

        desync_warmup_by_env: list[bool] | None
        if bool(flags.enable_desync):
            eb = int(flags.n_actor_envs)
            assert eb >= 1, eb
            desync_warmup_by_env = [False] * eb
        else:
            desync_warmup_by_env = None

        unroll = int(flags.unroll_length)
        bootstrap_time = unroll

        def _take_stats_buffer_idx() -> int | None:
            try:
                return stats_free_queue.get_nowait()
            except queue.Empty:
                return None

        need_initial_reset = True
        frozen_cfg = flags.frozen_opponent
        frozen_probability = float(frozen_cfg.probability)
        frozen_checkpoint_global_count = len(frozen_cfg.checkpoints)
        frozen_checkpoint_ids_by_num_players = {
            num_players: tuple(
                checkpoint_i + 1
                for checkpoint_i, checkpoint in enumerate(frozen_cfg.checkpoints)
                if checkpoint.num_players is None
                or int(checkpoint.num_players) == int(num_players)
            )
            for num_players in (2, 4)
        }
        frozen_rng = random.Random(
            int(frozen_cfg.seed) + int(actor_index_for_env) * 100003
        )
        game_size_rng = random.Random(
            int(frozen_cfg.seed) + 7919 + int(actor_index_for_env) * 100003
        )
        reset_probability_4p = _reset_probability_4p_from_target_sample_ratio(
            float(flags.orbit_actor_target_4p_sample_ratio)
        )
        current_game_num_players = torch.full(
            (int(flags.n_actor_envs),),
            int(flags.orbit_num_agents),
            dtype=torch.int64,
        )
        model_by_seat = torch.zeros(
            int(flags.n_actor_envs),
            ORBIT_PLAYER_AXIS_SLOTS,
            dtype=torch.int64,
        )
        learner_player_mask = torch.zeros(
            int(flags.n_actor_envs),
            ORBIT_PLAYER_AXIS_SLOTS,
            dtype=torch.float32,
        )
        critic_mc_tracker = _CriticMcEpisodeTracker(
            n_actor_envs=int(flags.n_actor_envs),
            discounting=_baseline_float_config_value(flags.discounting, "discounting"),
        )

        def _assign_selfplay_env(env_i: int) -> None:
            active_slots = tuple(
                orbit_active_policy_slots(int(current_game_num_players[int(env_i)].item()))
            )
            model_by_seat[int(env_i)].zero_()
            learner_player_mask[int(env_i)].zero_()
            for slot in active_slots:
                learner_player_mask[int(env_i), int(slot)] = 1.0

        def _assign_frozen_env(env_i: int) -> None:
            assert frozen_checkpoint_global_count > 0
            game_num_players = int(current_game_num_players[int(env_i)].item())
            assert game_num_players in (2, 4), game_num_players
            frozen_checkpoint_ids = frozen_checkpoint_ids_by_num_players[game_num_players]
            assert len(frozen_checkpoint_ids) > 0, (
                "frozen_opponent.checkpoints must include checkpoints for "
                f"{game_num_players}p games"
            )
            active_slots = tuple(
                orbit_active_policy_slots(game_num_players)
            )
            seats = list(active_slots)
            frozen_rng.shuffle(seats)
            if bool(frozen_cfg.no_selfplay):
                frozen_count = len(seats)
            elif len(seats) == 2:
                frozen_count = 1
            else:
                assert len(seats) == 4, seats
                if bool(frozen_cfg.all_frozen):
                    frozen_count = 3
                else:
                    frozen_count = frozen_rng.randint(1, 3)
            model_by_seat[int(env_i)].zero_()
            learner_player_mask[int(env_i)].zero_()
            frozen_slots = set(seats[:frozen_count])
            for slot in seats:
                if int(slot) in frozen_slots:
                    model_by_seat[int(env_i), int(slot)] = frozen_rng.choice(
                        frozen_checkpoint_ids
                    )
                    if bool(frozen_cfg.learn_on_frozen):
                        learner_player_mask[int(env_i), int(slot)] = 1.0
                else:
                    learner_player_mask[int(env_i), int(slot)] = 1.0

        def _assign_env(env_i: int) -> None:
            if frozen_rng.random() < frozen_probability:
                _assign_frozen_env(env_i)
            else:
                _assign_selfplay_env(env_i)

        def _assign_all_envs() -> None:
            for env_i in range(int(flags.n_actor_envs)):
                _assign_env(env_i)

        def _assign_reset_envs(
            done_reset: torch.Tensor,
            next_game_num_players: torch.Tensor,
        ) -> None:
            assert isinstance(done_reset, torch.Tensor) and done_reset.dtype == torch.bool
            dr = done_reset.reshape(-1)
            assert tuple(dr.shape) == (int(flags.n_actor_envs),), (
                tuple(dr.shape),
                int(flags.n_actor_envs),
            )
            assert isinstance(next_game_num_players, torch.Tensor)
            assert next_game_num_players.dtype == torch.int64
            ngr = next_game_num_players.reshape(-1)
            assert tuple(ngr.shape) == (int(flags.n_actor_envs),), (
                tuple(ngr.shape),
                int(flags.n_actor_envs),
            )
            for env_i in range(int(flags.n_actor_envs)):
                if bool(dr[env_i].item()):
                    na = int(ngr[env_i].item())
                    assert na in (2, 4), na
                    current_game_num_players[int(env_i)] = na
                    _assign_env(env_i)

        while True:
            raise_if_stop_requested(stop_event)

            with profiler_span(wall_prof, "rollout"):
                with profiler_span(wall_prof, "take_buffer"):
                    buffer_idx = queue_get_or_stop(
                        free_queue, stop_event, timeout_sec=0.1
                    )
                    stats_buffer_idx = _take_stats_buffer_idx()

                if need_initial_reset:
                    with profiler_span(wall_prof, "initial_reset"):
                        current_game_num_players = _sample_game_num_players_by_env(
                            game_size_rng,
                            reset_probability_4p,
                            int(flags.n_actor_envs),
                        )
                        env_out = env.reset(
                            orbit_num_agents_by_env=current_game_num_players,
                        )
                        _assign_all_envs()
                    need_initial_reset = False

                # Buffers: time dim ``unroll_length + 1`` — ``unroll_length`` env steps, then one row with
                # terminal obs for learner bootstrap (no actor infer); r/d on that row are dummies.
                last_env_next: dict[str, Any] | None = None
                agent_out: dict[str, torch.Tensor] | None = None
                eb_roll = int(flags.n_actor_envs)
                rollout_contains_desync = False
                for t in range(unroll):
                    raise_if_stop_requested(stop_event)
                    with profiler_span(wall_prof, "step"):
                        with profiler_span(wall_prof, "infer_input"):
                            infer_in = get_buffers_with_tag(env_out, device=None, tag="INFER")
                            assert infer_in is not None
                            infer_in["frozen_model_by_player_axis_LEARN"] = (
                                orbit_model_by_player_axis_from_seats(
                                    model_by_seat,
                                    current_game_num_players,
                                )
                            )
                        with profiler_span(wall_prof, "infer"):
                            agent_out = _run_model_from_inputs(
                                infer_in,
                                model_by_seat,
                                infer_req_queue,
                                infer_res_queue,
                                infer_req_free_queue,
                                infer_request_buffers,
                                infer_result_buffers,
                                actor_index_for_buffers,
                                stop_event,
                                flags,
                            )
                        with profiler_span(wall_prof, "action_to_cpu"):
                            bl_cpu = (
                                _baseline_value_from_learn_output(
                                    agent_out["baseline_LEARN"]
                                ).detach().cpu()
                                if record_tape and "baseline_LEARN" in agent_out
                                else None
                            )
                            ac_orbit = (
                                agent_out["actions_LEARN"]["spawn_fleet"].detach().cpu()
                            )

                        assert isinstance(ac_orbit, torch.Tensor) and ac_orbit.ndim == 4
                        assert int(ac_orbit.shape[1]) == int(ORBIT_PLAYER_AXIS_SLOTS), (
                            tuple(ac_orbit.shape),
                            ORBIT_PLAYER_AXIS_SLOTS,
                        )
                        assert int(ac_orbit.shape[2]) == int(ORBIT_PLANET_ACTION_SLOTS), (
                            tuple(ac_orbit.shape),
                            ORBIT_PLANET_ACTION_SLOTS,
                        )
                        assert int(ac_orbit.shape[3]) == 1, tuple(ac_orbit.shape)

                        with profiler_span(wall_prof, "env_step"):
                            env_next = env.step(
                                ac_orbit,
                                tape_baseline_learn=bl_cpu,
                            )

                        with profiler_span(wall_prof, "desync"):
                            dd = env_next["desync_done"]
                            assert isinstance(dd, torch.Tensor) and dd.dtype == torch.bool
                            ddr = dd.reshape(-1)
                            assert tuple(ddr.shape) == (eb_roll,), (
                                tuple(ddr.shape),
                                eb_roll,
                            )
                            if desync_warmup_by_env is not None and bool(ddr.any().item()):
                                rollout_contains_desync = True
                            _apply_desync_warmup_flags(desync_warmup_by_env, dd)

                        with profiler_span(wall_prof, "learn_payload"):
                            learn_env_out = _env_for_learner_player_mask(
                                env_out,
                                learner_player_mask,
                            )
                            learn = _learn_payload(
                                learn_env_out,
                                agent_out,
                                env_next,
                                _game_num_players_learn_tensor(current_game_num_players),
                                orbit_model_by_player_axis_from_seats(
                                    model_by_seat,
                                    current_game_num_players,
                                ),
                                flags,
                            )
                            fill_buffers_inplace_2(buffers[buffer_idx], learn, t)
                        with profiler_span(wall_prof, "critic_mc_stats"):
                            done_reset = env_next["done_LEARN_STAT"] | dd
                            critic_mc_stat = critic_mc_tracker.update(
                                baseline=_baseline_value_from_learn_output(
                                    agent_out["baseline_LEARN"]
                                ),
                                reward=_baseline_reward_from_learn_stat(
                                    env_next["reward_LEARN_STAT"]
                                ),
                                done=env_next["done_LEARN_STAT"],
                                done_reset=done_reset,
                                learner_player_mask=learner_player_mask,
                            )
                        if stats_buffer_idx is not None:
                            with profiler_span(wall_prof, "stats_payload"):
                                stats = _stats_payload(
                                    env_next,
                                    model_by_seat,
                                    critic_mc_stat,
                                )
                                fill_buffers_inplace_2(
                                    stats_buffers[stats_buffer_idx], stats, t
                                )
                        last_env_next = env_next
                        with profiler_span(wall_prof, "reset_where_done"):
                            dr = done_reset.reshape(-1)
                            assert tuple(dr.shape) == (eb_roll,), (
                                tuple(dr.shape),
                                eb_roll,
                            )
                            next_game_num_players = _sample_game_num_players_by_env(
                                game_size_rng,
                                reset_probability_4p,
                                int(flags.n_actor_envs),
                            )
                            env_out = env.reset_where_done(
                                env_next,
                                done_reset,
                                orbit_num_agents_by_env=next_game_num_players,
                            )
                            _assign_reset_envs(done_reset, next_game_num_players)

                assert last_env_next is not None
                assert agent_out is not None
                raise_if_stop_requested(stop_event)
                with profiler_span(wall_prof, "bootstrap_payload"):
                    learn_boot = _bootstrap_learn_payload(
                        _env_for_learner_player_mask(env_out, learner_player_mask),
                        agent_out,
                        last_env_next,
                        _game_num_players_learn_tensor(current_game_num_players),
                        orbit_model_by_player_axis_from_seats(
                            model_by_seat,
                            current_game_num_players,
                        ),
                        flags,
                    )
                    fill_buffers_inplace_2(buffers[buffer_idx], learn_boot, bootstrap_time)

                with profiler_span(wall_prof, "validate_actor_buffer"):
                    maybe_validate_dict_io_contract(
                        flags, buffers[buffer_idx], "actor_learn_buffer_template"
                    )

                with profiler_span(wall_prof, "desync_status"):
                    if desync_warmup_by_env is not None:
                        n_desynced = sum(1 for x in desync_warmup_by_env if x)
                    else:
                        n_desynced = 0
                    actor_desync_done_env_count[int(actor_index_for_env)] = int(n_desynced)

                learner_ready = desync_warmup_by_env is None or (
                    all(desync_warmup_by_env) and not rollout_contains_desync
                )
                if learner_ready:
                    rollout_player_mask = buffers[buffer_idx]["obs_LEARN_INFER"]["player_mask"]
                    assert isinstance(rollout_player_mask, torch.Tensor)
                    assert tuple(rollout_player_mask.shape) == (
                        int(flags.rollout_time_steps),
                        int(flags.n_actor_envs),
                        ORBIT_PLAYER_AXIS_SLOTS,
                    ), tuple(rollout_player_mask.shape)
                    rollout_transition_player_mask = rollout_player_mask[:-1]
                    assert int(rollout_transition_player_mask.shape[0]) == int(flags.unroll_length), (
                        tuple(rollout_transition_player_mask.shape),
                        int(flags.unroll_length),
                    )
                    learner_ready = bool(
                        (rollout_transition_player_mask > 0.5).any().item()
                    )
                with profiler_span(wall_prof, "enqueue_rollout"):
                    if learner_ready:
                        with full_queue_lock:
                            queue_put_or_stop(
                                full_queue,
                                (time.time(), buffer_idx),
                                stop_event,
                                timeout_sec=0.1,
                            )
                    else:
                        queue_put_or_stop(
                            free_queue, buffer_idx, stop_event, timeout_sec=0.1
                        )
                if stats_buffer_idx is not None:
                    with profiler_span(wall_prof, "enqueue_stats"):
                        if learner_ready:
                            queue_put_or_stop(
                                stats_full_queue,
                                stats_buffer_idx,
                                stop_event,
                                timeout_sec=0.1,
                            )
                        else:
                            queue_put_or_stop(
                                stats_free_queue,
                                stats_buffer_idx,
                                stop_event,
                                timeout_sec=0.1,
                            )

            if wall_prof is not None:
                wall_prof.summary(f"{name} rollout {wall_prof_rollout_ix}")
                wall_prof.clear()
                wall_prof_rollout_ix += 1

    except KeyboardInterrupt:
        pass
    except StopRequested:
        logging.info("%s received StopRequested, exiting", name)
    except Exception:
        logging.exception("%s failed in act_rollout_func", name)
    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name=name,
            reason="act_rollout_func finally",
        )
        os._exit(0)
