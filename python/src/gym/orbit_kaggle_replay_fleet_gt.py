"""Replay ground-truth (from_planet_id, target_planet_id, ships) from newly spawned fleets + C++ hit traces."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TextIO

from .orbit_kaggle_cpp_cache import OrbitKaggleCppObservationCache
from .obs_wrapper import ORBIT_PLANET_ARRIVAL_HORIZON
from .orbit_wars_env import orbit_observation_to_plain

def replay_observation_to_plain(obs: Mapping[str, Any], *, replay_step: int) -> dict[str, Any]:
    rs = int(replay_step)
    assert rs >= 0, rs
    plain = orbit_observation_to_plain(obs)
    out: dict[str, Any] = {}
    for k, v in plain.items():
        if k == "planets":
            out[k] = [list(row) for row in v] if isinstance(v, list) else v
        elif k == "fleets":
            out[k] = [list(row) for row in v] if isinstance(v, list) else v
        elif isinstance(v, list):
            out[k] = [x for x in v]
        else:
            out[k] = v
    out["step"] = rs
    return out


def assert_replay_frame_contract(ep: Mapping[str, Any]) -> None:
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 1
    row0 = steps[0]
    assert isinstance(row0, list)
    n_agents = len(row0)
    assert n_agents in (2, 4), n_agents
    for i, row in enumerate(steps):
        assert isinstance(row, list) and len(row) == n_agents, (i, len(row), n_agents)
        for seat, st in enumerate(row):
            assert isinstance(st, dict), (i, seat, type(st))
            obs = st["observation"]
            assert isinstance(obs, dict), (i, seat, type(obs))

def _fleet_ids(fleets: list[Any]) -> set[int]:
    out: set[int] = set()
    for row in fleets:
        assert isinstance(row, (list, tuple)) and len(row) >= 1
        out.add(int(row[0]))
    return out


def _new_spawn_fleet_rows(prev_fleets: list[Any], next_fleets: list[Any]) -> list[list[Any]]:
    prev_ids = _fleet_ids(prev_fleets)
    new_rows: list[list[Any]] = []
    for row in next_fleets:
        assert isinstance(row, (list, tuple)) and len(row) == 7
        if int(row[0]) not in prev_ids:
            new_rows.append(list(row))
    new_rows.sort(key=lambda r: int(r[0]))
    return new_rows


def print_replay_fleet_spawn_ground_truth(
    ep: Mapping[str, Any],
    *,
    horizon: int,
    out: TextIO,
) -> None:
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 2
    n_agents = len(steps[0])
    cfg = ep["configuration"]
    assert isinstance(cfg, Mapping)
    cache = OrbitKaggleCppObservationCache(configuration=cfg, num_agents=n_agents)
    assert_replay_frame_contract(ep)
    row0 = steps[0]
    assert isinstance(row0, list) and len(row0) >= 1
    st00 = row0[0]
    assert isinstance(st00, dict)
    obs00 = st00["observation"]
    assert isinstance(obs00, dict)
    cache.reset_from_kaggle_plain(
        plain=replay_observation_to_plain(obs00, replay_step=0),
        num_agents=n_agents,
    )
    for next_idx in range(1, len(steps)):
        row_next0 = steps[next_idx][0]
        assert isinstance(row_next0, dict)
        obs_next = row_next0["observation"]
        assert isinstance(obs_next, dict)
        plain_next = replay_observation_to_plain(obs_next, replay_step=next_idx)
        row_prev0 = steps[next_idx - 1][0]
        assert isinstance(row_prev0, dict)
        obs_prev = row_prev0["observation"]
        assert isinstance(obs_prev, dict)
        fleets_prev = obs_prev["fleets"]
        fleets_next = obs_next["fleets"]
        assert isinstance(fleets_prev, list) and isinstance(fleets_next, list)
        cache.step_noop_and_update_comets_from_kaggle_plain(
            plain=plain_next,
            num_agents=n_agents,
        )
        new_rows = _new_spawn_fleet_rows(fleets_prev, fleets_next)
        parts = [
            f"frame={next_idx}",
            f"obs_step={int(plain_next['step'])}",
        ]
        for s in range(n_agents):
            st = steps[next_idx][s]
            assert isinstance(st, dict)
            act = st["action"]
            if act is None:
                act = []
            else:
                assert isinstance(act, list)
            parts.append(f"p{s}_kaggle={act!r}")
        out.write(" ".join(parts) + "\n")
        for frow in new_rows:
            owner = int(frow[1])
            traces = cache.fleet_hit_traces_from_plain_fleets(
                fleets=[frow],
                horizon=int(horizon),
            )
            from_pid = int(frow[5])
            ships = float(frow[6])
            if len(traces) == 0:
                out.write(
                    f"  spawn_gt player={owner} fleet_id={int(frow[0])} "
                    f"from_planet_id={from_pid} ships={ships} target=NO_HIT\n"
                )
                continue
            assert len(traces) == 1, (next_idx, frow, traces)
            tr = traces[0]
            hit_pid = int(tr["hit_planet_id"])
            hit_steps = int(tr["hit_steps"])
            out.write(
                f"  spawn_gt player={owner} fleet_id={int(frow[0])} "
                f"from_planet_id={from_pid} ships={ships} "
                f"target_planet_id={hit_pid} hit_steps={hit_steps}\n"
            )


def default_hit_horizon() -> int:
    return int(ORBIT_PLANET_ARRIVAL_HORIZON)
