"""Shared Orbit C++ env wiring: one reset-without-comets contract + external comet payloads.

Orbit C++ episode/Kaggle step counters advance only via ``step()`` after ``reset``.
``update_comets_from_state`` loads comet geometry into ``planets_`` and refreshes the noop comet slots using
the **current** ``episode_step()`` — callers must apply ``orbit_cpp_env_apply_comet_sync_update_one`` only when
``cpp_env.episode_step()`` already equals ``upd[\"episode_step\"]`` (same outer stepping loop as ``step()``).
"""

from __future__ import annotations

from typing import Any

import torch

from .obs_wrapper import (
    ORBIT_EDGE_FEATURES,
    ORBIT_MAX_PLANETS,
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLAYER_AXIS_SLOTS,
)

def _planet_ships_by_id_from_plain_planets(planets: Any) -> dict[int, float]:
    assert isinstance(planets, list)
    out: dict[int, float] = {}
    for row in planets:
        assert isinstance(row, (list, tuple)) and len(row) >= 6
        pid = int(row[0])
        assert pid not in out, (pid, len(out))
        out[pid] = float(row[5])
    return out


def orbit_comet_path_by_planet_id(
    comets: Any,
    planets: Any,
) -> dict[int, tuple[int, tuple[tuple[float, float], ...], float]]:
    assert isinstance(comets, list)
    ships_by_id = _planet_ships_by_id_from_plain_planets(planets)
    out: dict[int, tuple[int, tuple[tuple[float, float], ...], float]] = {}
    for group in comets:
        assert isinstance(group, dict)
        planet_ids = group["planet_ids"]
        paths = group["paths"]
        path_index = int(group["path_index"])
        assert isinstance(planet_ids, list)
        assert isinstance(paths, list)
        assert len(planet_ids) == len(paths)
        for pid_raw, path_raw in zip(planet_ids, paths, strict=True):
            pid = int(pid_raw)
            assert isinstance(path_raw, list)
            assert pid in ships_by_id, (
                "comet planet_id missing from plain planets (no ships row)",
                pid,
                len(ships_by_id),
            )
            ships = ships_by_id[pid]
            path_xy: list[tuple[float, float]] = []
            for pt in path_raw:
                assert isinstance(pt, (list, tuple)) and len(pt) == 2
                path_xy.append((float(pt[0]), float(pt[1])))
            out[pid] = (path_index, tuple(path_xy), ships)
    return out


def orbit_cpp_env_minimal_workspace_tensors_cpu() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.zeros(
            (ORBIT_PLAYER_AXIS_SLOTS, ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES),
            dtype=torch.float32,
        ),
        torch.zeros(
            (ORBIT_PLAYER_AXIS_SLOTS, ORBIT_MAX_PLANETS),
            dtype=torch.float32,
        ),
        torch.zeros(
            (ORBIT_PLAYER_AXIS_SLOTS, ORBIT_PLANET_PAIRWISE_COUNT),
            dtype=torch.float32,
        ),
        torch.zeros(
            (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_PAIRWISE_COUNT,
                ORBIT_EDGE_FEATURES,
            ),
            dtype=torch.float32,
        ),
        torch.zeros(
            (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_MAX_PLANETS,
                ORBIT_PER_PLANET_HIT_CLASSES,
            ),
            dtype=torch.int8,
        ),
        torch.zeros((ORBIT_PLAYER_AXIS_SLOTS, ORBIT_MAX_PLANETS, 1), dtype=torch.int32),
        torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32),
    )


def orbit_cpp_env_reset_no_comets(
    cpp_env: Any,
    *,
    angular_velocity: float,
    planet_rows: torch.Tensor,
    planet_count: int,
    orbit_planet_features: torch.Tensor,
    orbit_planet_mask: torch.Tensor,
    orbit_planet_pairwise_mask: torch.Tensor,
    orbit_planet_pairwise_features: torch.Tensor,
    action_taken_index: torch.Tensor,
    player_mask: torch.Tensor,
) -> None:
    cpp_env.reset(
        float(angular_velocity),
        planet_rows,
        int(planet_count),
        orbit_planet_features,
        orbit_planet_mask,
        orbit_planet_pairwise_mask,
        orbit_planet_pairwise_features,
        action_taken_index,
        player_mask,
    )


def orbit_cpp_env_apply_comet_sync_update_one(cpp_env: Any, upd: Any) -> None:
    target = int(upd["episode_step"])
    cur = int(cpp_env.episode_step())
    assert cur == target, (cur, target, upd)
    c_ids = upd["comet_planet_ids"]
    assert isinstance(c_ids, list), (target, c_ids)
    assert len(c_ids) in (0, 4), (target, c_ids)
    groups = upd["comets_groups"]
    assert isinstance(groups, list), (target, groups)
    if len(c_ids) == 0:
        assert len(groups) == 0, (target, groups)
        return
    assert len(groups) == 1, (target, groups)
    planets = upd["planets"]
    assert isinstance(planets, list)
    path_d = orbit_comet_path_by_planet_id(groups, planets)
    cpp_env.update_comets_from_state(c_ids, path_d)
