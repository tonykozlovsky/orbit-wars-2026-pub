"""Build orbit debug tapes from Kaggle-format replays (same frame schema as RL tape, no multiprocessing queue)."""
from __future__ import annotations

import sys
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .debug_viewer import append_frames_to_tape
from .orbit_kaggle_cpp_cache import (
    OrbitKaggleCppObservationCache,
    _honest_mask_requests_from_rows,
)
from .orbit_kaggle_replay_fleet_gt import (
    assert_replay_frame_contract,
    replay_observation_to_plain,
)
from .orbit_wars_env import (
    batched_planet_geometry_only_from_plain_seats,
    orbit_tape_frame_dict_from_obs_all,
)
from .obs_wrapper import (
    orbit_active_policy_slots,
    orbit_policy_slot_for_compact_agent,
)
from .orbit_tape_feature_pack import (
    orbit_policy_obs_action_edges_from_plain_and_policy_obs,
    orbit_policy_obs_feature_pack_from_plain_and_policy_obs,
)
from .wall_tree_profiler import WallTreeProfiler, profiler_span

def kaggle_replay_obs_all_from_plain(plain: dict[str, Any], num_agents: int) -> list[dict[str, Any]]:
    na = int(num_agents)
    assert na >= 1
    shared = dict(plain)
    planets = shared["planets"]
    fleets = shared["fleets"]
    out: list[dict[str, Any]] = []
    for s in range(na):
        d = dict(shared)
        d["player"] = int(s)
        d["planets"] = planets
        d["fleets"] = fleets
        out.append(d)
    assert int(out[0]["player"]) == 0
    return out


def _replay_frame_episode_done(steps_si: list[dict[str, Any]]) -> bool:
    for st in steps_si:
        assert isinstance(st, dict)
        if st["status"] == "ACTIVE":
            return False
    return True


def _policy_obs_feature_pack_from_training_snapshot(
    *,
    plain: dict[str, Any],
    policy_obs: dict[str, torch.Tensor],
    num_agents: int,
    hit_kind: torch.Tensor,
    intercept_fail_reason: torch.Tensor,
) -> dict[str, Any]:
    na = int(num_agents)
    assert na in (2, 4), na
    return orbit_policy_obs_feature_pack_from_plain_and_policy_obs(
        plain=plain,
        policy_obs=policy_obs,
        num_agents=na,
        hit_kind=hit_kind,
        intercept_fail_reason=intercept_fail_reason,
    )


def write_kaggle_replay_orbit_tape(
    ep: Mapping[str, Any],
    *,
    tape_root: Path,
    tape_name: str,
    hit_horizon: int,
    player_supervised_heads_by_frame: Sequence[dict[str, dict[str, Any]] | None] | None = None,
    profile: bool = False,
    wall_profiler: WallTreeProfiler | None = None,
    profile_summary_every_steps: int = 1,
) -> Path:
    assert profile_summary_every_steps >= 1, profile_summary_every_steps
    steps = ep["steps"]
    assert isinstance(steps, list)
    cfg = ep["configuration"]
    assert isinstance(cfg, Mapping)
    row0 = steps[0]
    assert isinstance(row0, list)
    n_agents = len(row0)
    assert_replay_frame_contract(ep)
    if player_supervised_heads_by_frame is not None:
        assert len(player_supervised_heads_by_frame) == len(steps), (
            len(player_supervised_heads_by_frame),
            len(steps),
        )
    cache = OrbitKaggleCppObservationCache(configuration=cfg, num_agents=n_agents)
    frames: list[dict[str, Any]] = []
    hz = int(hit_horizon)
    assert hz >= 1
    if profile:
        os.environ["ORBIT_HONEST_MASK_CPP_PROFILE"] = "1"
    for si in range(len(steps)):
        step_prof = wall_profiler if wall_profiler is not None else WallTreeProfiler() if profile else None
        row = steps[si]
        assert isinstance(row, list) and len(row) == n_agents
        cell0 = row[0]
        assert isinstance(cell0, dict)
        obs = cell0["observation"]
        assert isinstance(obs, dict)
        with profiler_span(step_prof, "tape_step"):
            with profiler_span(step_prof, "prep_plain"):
                plain = replay_observation_to_plain(obs, replay_step=si)
            with profiler_span(step_prof, "cache_clock"):
                if si == 0:
                    cache.reset_from_kaggle_plain(
                        plain=plain,
                        num_agents=n_agents,
                    )
                else:
                    cache.step_noop_and_update_comets_from_kaggle_plain(
                        plain=plain,
                        num_agents=n_agents,
                    )
            with profiler_span(step_prof, "obs_batched"):
                obs_all = kaggle_replay_obs_all_from_plain(plain, n_agents)
                dev = torch.device("cpu")
                active_policy_slots = orbit_active_policy_slots(n_agents)
            if profile and (si + 1) % profile_summary_every_steps == 0:
                batched = batched_planet_geometry_only_from_plain_seats(
                    seats_plain=obs_all,
                    device=dev,
                )
                rows_t = batched["planet_rows"]
                cnt_t = batched["planet_count"]
                cpu = torch.device("cpu")
                n_req_by_seat: list[int] = []
                n_bucket_checks_by_seat: list[int] = []
                for seat, seat_plain in enumerate(obs_all):
                    policy_slot = orbit_policy_slot_for_compact_agent(seat, n_agents)
                    planet_rows = rows_t[policy_slot].to(dtype=torch.float32, device=cpu).contiguous()
                    planet_count = int(cnt_t[policy_slot].item())
                    req_t = _honest_mask_requests_from_rows(
                        planet_rows=planet_rows,
                        planet_count=planet_count,
                        player_id=int(seat_plain["player"]),
                    )
                    n_req_by_seat.append(int(req_t.shape[0]))
                    n_bucket_checks_by_seat.append(int(req_t[:, 2].sum().item()))
                print(
                    f"WALL_TREE_TAPE honest_mask_requests step={si} episode_step={int(plain['step'])} "
                    f"n_requests_per_seat={n_req_by_seat} sum_requests={sum(n_req_by_seat)} "
                    f"n_bucket_checks_per_seat={n_bucket_checks_by_seat} "
                    f"sum_bucket_checks={sum(n_bucket_checks_by_seat)}",
                    file=sys.stderr,
                    flush=True,
                )
            with profiler_span(step_prof, "policy_obs_snapshot"):
                policy_obs = cache.snapshot_policy_obs_from_plain_seats_cpu(
                    plain=plain,
                    seats_plain=obs_all,
                    policy_slots=active_policy_slots,
                    ship_speed=float(cfg["shipSpeed"]),
                )
            with profiler_span(step_prof, "action_edges"):
                action_edges = orbit_policy_obs_action_edges_from_plain_and_policy_obs(
                    plain=plain,
                    policy_obs=policy_obs,
                    num_agents=n_agents,
                )
            with profiler_span(step_prof, "fleet_traces"):
                traces = cache.fleet_hit_traces_from_plain_fleets(
                    fleets=plain["fleets"],
                    horizon=hz,
                )
            with profiler_span(step_prof, "policy_obs_feature_pack"):
                orbit_planet_feature_pack = _policy_obs_feature_pack_from_training_snapshot(
                    plain=plain,
                    policy_obs=policy_obs,
                    num_agents=n_agents,
                    hit_kind=cache.honest_shared_hit_kind_last(),
                    intercept_fail_reason=cache.honest_shared_intercept_fail_reason_last(),
                )
            with profiler_span(step_prof, "frame_dict"):
                frames.append(
                    orbit_tape_frame_dict_from_obs_all(
                        obs_all,
                        num_agents=n_agents,
                        episode_step=int(plain["step"]),
                        episode_done=_replay_frame_episode_done(row),
                        player_value_baseline=None,
                        player_supervised_heads=(
                            None
                            if player_supervised_heads_by_frame is None
                            else player_supervised_heads_by_frame[si]
                        ),
                        action_edges=action_edges,
                        fleet_arrival_traces=traces,
                        orbit_planet_feature_pack=orbit_planet_feature_pack,
                    )
                )
        if profile and (si + 1) % profile_summary_every_steps == 0:
            assert step_prof is not None
            step_prof.summary_stdout(
                f"orbit_kaggle_replay_tape step={si} episode_step={int(plain['step'])}",
                line_prefix="WALL_TREE_TAPE ",
                file=sys.stderr,
            )
            step_prof.clear()
    root = Path(tape_root).expanduser().resolve()
    viewer_path = root / tape_name / "frames.jsonl"
    if viewer_path.is_file():
        viewer_path.unlink()
    return append_frames_to_tape(root, tape_name, frames)


def default_replay_tape_hit_horizon(configuration: Any) -> int:
    return OrbitKaggleCppObservationCache.configuration_episode_steps(configuration)
