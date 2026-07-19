import logging
import os
import random
import time
from typing import Any

import setproctitle
import torch

from ...gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from ...gym.create_env import create_env
from ...gym.dict_io_contract import maybe_validate_dict_io_contract
from ...gym.obs_wrapper import (
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
)
from .buffer_utils import copy_buffers, copy_matching_tree_into, get_buffers_with_tag
from .act import (
    _reset_probability_4p_from_target_sample_ratio,
    _sample_game_num_players_by_env,
)
from .common import (
    StopRequested,
    queue_get_or_stop,
    queue_put_or_stop,
    raise_if_stop_requested,
    set_stop_event_with_reason,
)


def _run_benchmark_model_from_inputs(
    inference_inputs: dict,
    model_by_seat: torch.Tensor,
    infer_req_queue,
    infer_res_queue,
    infer_req_free_queue,
    infer_request_buffers: list,
    infer_result_buffers: list,
    actor_index_for_buffers: int,
    stop_event,
    flags: Any,
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


def _enqueue_decisive_benchmark_results(
    *,
    env_next: dict[str, Any],
    seat_mapping: torch.Tensor,
    game_num_players: torch.Tensor,
    result_queue,
    stop_event,
) -> None:
    done = env_next["done_LEARN_STAT"]
    assert isinstance(done, torch.Tensor) and done.dtype == torch.bool
    done_flat = done.reshape(-1)

    info_stat = env_next["info_STAT"]
    assert isinstance(info_stat, dict), type(info_stat)
    gr_cum = info_stat["LOGGING_STAT_game_result_cumsum"]
    assert isinstance(gr_cum, torch.Tensor)
    assert gr_cum.ndim == 2, tuple(gr_cum.shape)
    assert int(gr_cum.shape[0]) == int(done_flat.shape[0]), (
        tuple(gr_cum.shape),
        tuple(done_flat.shape),
    )
    assert int(gr_cum.shape[1]) == ORBIT_PLAYER_AXIS_SLOTS, tuple(gr_cum.shape)

    assert isinstance(seat_mapping, torch.Tensor)
    assert seat_mapping.shape == (
        int(done_flat.shape[0]),
        ORBIT_PLAYER_AXIS_SLOTS,
    ), tuple(seat_mapping.shape)
    assert isinstance(game_num_players, torch.Tensor)
    assert game_num_players.dtype == torch.int64, game_num_players.dtype
    game_num_players_flat = game_num_players.reshape(-1)
    assert tuple(game_num_players_flat.shape) == tuple(done_flat.shape), (
        tuple(game_num_players_flat.shape),
        tuple(done_flat.shape),
    )

    for env_i in range(int(done_flat.shape[0])):
        if not bool(done_flat[env_i].item()):
            continue
        row = seat_mapping[env_i].detach().cpu().to(dtype=torch.int64)
        assert int(row.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS, tuple(row.shape)
        active_slots = tuple(
            orbit_active_policy_slots(int(game_num_players_flat[env_i].item()))
        )
        active_row = row[list(active_slots)]
        a_seats = int((active_row == 0).sum().item())
        b_seats = int((active_row == 1).sum().item())
        assert 1 <= a_seats < len(active_slots), row
        assert 1 <= b_seats < len(active_slots), row
        assert a_seats + b_seats == len(active_slots), row
        gc_row = gr_cum[env_i].detach().cpu().to(dtype=torch.float32)
        gc_active = gc_row[list(active_slots)]
        mx = gc_active.max()
        at_mx = gc_active == mx
        if int(at_mx.sum().item()) != 1:
            queue_put_or_stop(
                result_queue,
                {
                    "winner": -1,
                    "a_win_events": 0,
                    "b_win_events": 0,
                },
                stop_event,
                timeout_sec=0.1,
            )
            continue
        win_seat = int(active_slots[int(torch.argmax(gc_active).item())])
        winner = int(row[win_seat].item())
        opponent_seats = b_seats if winner == 0 else a_seats
        assert opponent_seats >= 1, (winner, row)
        queue_put_or_stop(
            result_queue,
            {
                "winner": int(winner),
                "a_win_events": int(opponent_seats if winner == 0 else 0),
                "b_win_events": int(opponent_seats if winner == 1 else 0),
            },
            stop_event,
            timeout_sec=0.1,
        )


def _sample_benchmark_game_num_players_by_env(
    *,
    flags: Any,
    rng: random.Random,
    probability_4p: float,
    n_actor_envs: int,
) -> torch.Tensor:
    mode = str(flags.benchmark_game_size)
    if mode == "mixed":
        return _sample_game_num_players_by_env(rng, probability_4p, n_actor_envs)
    if mode == "2p":
        return torch.full((int(n_actor_envs),), 2, dtype=torch.int64)
    if mode == "4p":
        return torch.full((int(n_actor_envs),), 4, dtype=torch.int64)
    raise AssertionError(f"unsupported benchmark_game_size: {mode}")


def _bench_act_func_impl(
    flags: Any,
    actor_index_for_env: int,
    actor_index_for_buffers: int,
    infer_req_queue,
    infer_res_queue,
    infer_req_free_queue,
    infer_request_buffers,
    infer_result_buffers,
    benchmark_result_queue,
    benchmark_env_steps_total,
    name: str,
    start_event,
    stop_event,
) -> None:
    setproctitle.setproctitle(name)
    logging.info("%s start env=%s buffer_client=%s", name, actor_index_for_env, actor_index_for_buffers)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert not bool(flags.enable_desync), "benchmark requires full episodes; enable_desync must be false"
    assert hasattr(flags, "benchmark_game_size"), "benchmark_game_size is required"
    assert str(flags.benchmark_game_size) in ("mixed", "2p", "4p"), flags.benchmark_game_size

    rng = random.Random()
    seed = getattr(flags, "benchmark_position_seed", None)
    if seed is not None:
        rng.seed(int(seed) + int(actor_index_for_env) * 100003)
    game_size_rng = random.Random()
    if seed is not None:
        game_size_rng.seed(int(seed) + 7919 + int(actor_index_for_env) * 100003)
    reset_probability_4p = _reset_probability_4p_from_target_sample_ratio(
        float(flags.orbit_actor_target_4p_sample_ratio)
    )

    e_roll = int(flags.n_actor_envs)
    current_game_num_players = torch.full(
        (e_roll,),
        int(flags.orbit_num_agents),
        dtype=torch.int64,
    )
    benchmark_model_by_seat = torch.zeros(
        e_roll,
        ORBIT_PLAYER_AXIS_SLOTS,
        dtype=torch.int64,
    )

    def _assign_seats_for_env(env_i: int) -> None:
        active_slots = tuple(
            orbit_active_policy_slots(int(current_game_num_players[int(env_i)].item()))
        )
        assert len(active_slots) in (2, 4), active_slots
        a_seat_count = rng.randint(1, len(active_slots) - 1)
        row = [0] * a_seat_count + [1] * (len(active_slots) - a_seat_count)
        rng.shuffle(row)
        benchmark_model_by_seat[int(env_i)].zero_()
        for i, slot in enumerate(active_slots):
            benchmark_model_by_seat[int(env_i), int(slot)] = int(row[i])

    def _fill_all_seat_rows() -> None:
        for env_i in range(e_roll):
            _assign_seats_for_env(env_i)

    def _resample_done_games(
        done_reset: torch.Tensor,
        next_game_num_players: torch.Tensor,
    ) -> None:
        assert isinstance(done_reset, torch.Tensor) and done_reset.dtype == torch.bool
        flat = done_reset.reshape(-1)
        assert int(flat.shape[0]) == e_roll, (tuple(flat.shape), e_roll)
        assert isinstance(next_game_num_players, torch.Tensor)
        assert next_game_num_players.dtype == torch.int64, next_game_num_players.dtype
        next_flat = next_game_num_players.reshape(-1)
        assert int(next_flat.shape[0]) == e_roll, (tuple(next_flat.shape), e_roll)
        for env_i in range(e_roll):
            if bool(flat[env_i].item()):
                na = int(next_flat[env_i].item())
                assert na in (2, 4), na
                current_game_num_players[int(env_i)] = na
                _assign_seats_for_env(env_i)

    while not start_event.wait(timeout=0.1):
        raise_if_stop_requested(stop_event)

    wall_prof = WallTreeProfiler() if bool(flags.enable_actor_wall_tree_profiler) else None
    _envs_val = bool(flags.enable_envs_validation)
    env = create_env(
        flags,
        device="cpu",
        visualize=False,
        visualize_sim_env=False,
        visualization_queue=None,
        record_tape=False,
        cpp_env_obs_full=not _envs_val,
        cpp_env_obs_validate=_envs_val,
        wall_profiler=wall_prof,
    )

    current_game_num_players = _sample_benchmark_game_num_players_by_env(
        flags=flags,
        rng=game_size_rng,
        probability_4p=reset_probability_4p,
        n_actor_envs=e_roll,
    )
    env_out = env.reset(orbit_num_agents_by_env=current_game_num_players)
    _fill_all_seat_rows()
    while True:
        raise_if_stop_requested(stop_event)
        t_rollout0 = time.perf_counter()
        with profiler_span(wall_prof, "rollout"):
            infer_in = get_buffers_with_tag(env_out, device=None, tag="INFER")
            assert infer_in is not None
            with profiler_span(wall_prof, "infer"):
                agent_out = _run_benchmark_model_from_inputs(
                    infer_in,
                    benchmark_model_by_seat.clone(),
                    infer_req_queue,
                    infer_res_queue,
                    infer_req_free_queue,
                    infer_request_buffers,
                    infer_result_buffers,
                    actor_index_for_buffers,
                    stop_event,
                    flags,
                )
            ac_orbit = agent_out["actions_LEARN"]["spawn_fleet"].detach().cpu()
            assert isinstance(ac_orbit, torch.Tensor) and ac_orbit.ndim == 4
            assert int(ac_orbit.shape[1]) == ORBIT_PLAYER_AXIS_SLOTS, tuple(ac_orbit.shape)
            assert int(ac_orbit.shape[2]) == ORBIT_PLANET_ACTION_SLOTS, tuple(ac_orbit.shape)
            assert int(ac_orbit.shape[3]) == 1, tuple(ac_orbit.shape)

            with profiler_span(wall_prof, "env_step"):
                env_next = env.step(ac_orbit)
            with benchmark_env_steps_total.get_lock():
                benchmark_env_steps_total.value += int(e_roll)
            with profiler_span(wall_prof, "enqueue_results"):
                _enqueue_decisive_benchmark_results(
                    env_next=env_next,
                    seat_mapping=benchmark_model_by_seat,
                    game_num_players=current_game_num_players,
                    result_queue=benchmark_result_queue,
                    stop_event=stop_event,
                )
            done_reset = env_next["done_LEARN_STAT"]
            assert isinstance(done_reset, torch.Tensor) and done_reset.dtype == torch.bool
            with profiler_span(wall_prof, "reset_where_done"):
                next_game_num_players = _sample_benchmark_game_num_players_by_env(
                    flags=flags,
                    rng=game_size_rng,
                    probability_4p=reset_probability_4p,
                    n_actor_envs=e_roll,
                )
                env_out = env.reset_where_done(
                    env_next,
                    done_reset,
                    orbit_num_agents_by_env=next_game_num_players,
                )
            _resample_done_games(done_reset, next_game_num_players)
        rollout_wall_ms = (time.perf_counter() - t_rollout0) * 1000.0
        if wall_prof is not None:
            wall_prof.summary_stdout(name, iteration_wall_ms=rollout_wall_ms)


def bench_act_func(
    flags: Any,
    actor_index_for_env: int,
    actor_index_for_buffers: int,
    infer_req_queue,
    infer_res_queue,
    infer_req_free_queue,
    infer_request_buffers,
    infer_result_buffers,
    benchmark_result_queue,
    benchmark_env_steps_total,
    name: str,
    start_event,
    stop_event,
) -> None:
    try:
        _bench_act_func_impl(
            flags,
            actor_index_for_env,
            actor_index_for_buffers,
            infer_req_queue,
            infer_res_queue,
            infer_req_free_queue,
            infer_request_buffers,
            infer_result_buffers,
            benchmark_result_queue,
            benchmark_env_steps_total,
            name,
            start_event,
            stop_event,
        )
    except KeyboardInterrupt:
        pass
    except StopRequested:
        logging.info("%s received StopRequested, exiting", name)
    except Exception:
        logging.exception("%s failed in bench_act_func", name)
    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name=name,
            reason="bench_act_func finally",
        )
        os._exit(0)
