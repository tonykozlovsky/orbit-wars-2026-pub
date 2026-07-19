"""Kaggle Orbit observation sync for ``orbit_wars_cpp`` (honest geometry, fleet arrivals)."""
from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Mapping
from typing import Any

import torch

from .orbit_cpp_plain_sync import (
    orbit_comet_path_by_planet_id,
    orbit_cpp_env_minimal_workspace_tensors_cpu,
)
from .orbit_wars_env import (
    _orbit_available_action_mask_from_send_all_hit_mask_for_player,
    batched_planet_geometry_only_from_plain_seats,
)
from .obs_wrapper import (
    ORBIT_HIT_CLASSES_PER_TARGET,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_MOVE_SEND_SUBINDICES,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_place_compact_agent_axis,
    orbit_policy_slot_for_compact_agent,
    orbit_self_enemy_mask,
)
from .wall_tree_profiler import WallTreeProfiler, profiler_span

_ORBIT_COMET_SPAWN_UPDATE_STEPS = frozenset((50, 150, 250, 350, 450))


def planet_rows_tensor_from_plain_planets_float64(
    planets: list[Any],
) -> tuple[torch.Tensor, int]:
    assert isinstance(planets, list)
    n = len(planets)
    assert n <= ORBIT_PLANET_ACTION_SLOTS, (n, ORBIT_PLANET_ACTION_SLOTS)
    rows = torch.zeros((ORBIT_PLANET_ACTION_SLOTS, 7), dtype=torch.float64)
    for i in range(n):
        row = planets[i]
        assert isinstance(row, (list, tuple)) and len(row) == 7
        for j in range(7):
            rows[i, j] = float(row[j])
    return rows.contiguous(), n


def fleet_rows_tensor_from_plain_fleets_float64(fleets: list[Any]) -> torch.Tensor:
    assert isinstance(fleets, list)
    rows = torch.zeros((len(fleets), 7), dtype=torch.float64)
    for i, row in enumerate(fleets):
        assert isinstance(row, (list, tuple)) and len(row) == 7
        for j in range(7):
            rows[i, j] = float(row[j])
    return rows.contiguous()


def _kaggle_configuration_field(configuration: Any, name: str) -> Any:
    if isinstance(configuration, Mapping):
        return configuration[name]
    return getattr(configuration, name)


_PLANET_ROW_FIELD_NAMES = ("id", "owner", "x", "y", "radius", "ships", "production")
_PLANET_GEOMETRY_ABS_EPS = 1e-9


def _planet_row_as_dict(row: Any) -> dict[str, float]:
    assert isinstance(row, (list, tuple)) and len(row) == 7, (type(row), row)
    return {name: float(row[j]) for j, name in enumerate(_PLANET_ROW_FIELD_NAMES)}


def _comet_groups_brief(comets: Any) -> str:
    if not isinstance(comets, list):
        return f"comets_type={type(comets).__name__}"
    parts: list[str] = [f"n_groups={len(comets)}"]
    for gi, g in enumerate(comets):
        if not isinstance(g, Mapping):
            parts.append(f"g{gi}_type={type(g).__name__}")
            continue
        pids = g.get("planet_ids")
        pi = g.get("path_index")
        parts.append(f"g{gi}:ids={pids!r} path_index={pi!r}")
    return " | ".join(parts)


def _fleets_brief(fleets: Any, *, max_rows: int = 6) -> str:
    if not isinstance(fleets, list):
        return f"fleets_type={type(fleets).__name__}"
    out = [f"n_fleets={len(fleets)}"]
    for fi, row in enumerate(fleets[:max_rows]):
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            out.append(f"fleet[{fi}]_bad={row!r}")
            continue
        out.append(
            f"fleet[{fi}]: id={int(row[0])} owner={int(row[1])} "
            f"x={float(row[2]):.4g} y={float(row[3]):.4g} ships={float(row[6]):.4g}"
        )
    if len(fleets) > max_rows:
        out.append(f"... ({len(fleets) - max_rows} more)")
    return "\n    ".join(out)


def _plain_sync_debug_lines(plain: Mapping[str, Any] | None) -> list[str]:
    if plain is None:
        return ["plain_snapshot: None"]
    lines: list[str] = []
    if "step" in plain:
        lines.append(f"plain[step]={int(plain['step'])}")
    lines.append(f"plain[angular_velocity]={plain.get('angular_velocity')!r}")
    lines.append(f"plain[next_fleet_id]={plain.get('next_fleet_id')!r}")
    lines.append(f"plain[remainingOverageTime]={plain.get('remainingOverageTime')!r}")
    cids = plain.get("comet_planet_ids")
    lines.append(f"plain[comet_planet_ids]={cids!r}")
    lines.append(f"plain[comets]: {_comet_groups_brief(plain.get('comets'))}")
    ip = plain.get("initial_planets")
    if isinstance(ip, list):
        lines.append(f"plain[initial_planets]: n={len(ip)}")
    else:
        lines.append(f"plain[initial_planets]: {type(ip).__name__}")
    lines.append("plain[fleets]:")
    lines.append(f"    {_fleets_brief(plain.get('fleets'))}")
    return lines


def _stderr_dump_planet_row_mismatch(
    *,
    field_names: tuple[str, ...],
    replay_planets: list[Any],
    cache_planets: list[Any],
    row_index: int,
    field_index: int,
    cache_v: float,
    replay_v: float,
    obs_step: int,
    cpp_episode_step: int,
    action_summary: str,
    plain: Mapping[str, Any] | None,
    sync_context: Mapping[str, Any] | None,
) -> None:
    n = len(replay_planets)
    neigh = [row_index + d for d in (-1, 0, 1) if 0 <= row_index + d < n]
    geometry_field_indices = (0, 2, 3, 4)
    lines: list[str] = [
        "ORBIT_KAGGLE_CPP_CACHE planet row mismatch (detailed dump)",
        f"  replay_obs_step={int(obs_step)} cpp_episode_step={int(cpp_episode_step)}",
        f"  first_mismatch: row_index={row_index} field={field_names[field_index]} "
        f"cache={cache_v} replay={replay_v}",
        f"  {action_summary}",
    ]
    if sync_context:
        lines.append("  sync_context:")
        for k in sorted(sync_context.keys()):
            lines.append(f"    {k}={sync_context[k]!r}")
    lines.append("  neighbor rows (same index: cache vs replay):")
    for idx in neigh:
        cr = cache_planets[idx]
        rr = replay_planets[idx]
        lines.append(f"    row {idx}:")
        lines.append(f"      cache={_planet_row_as_dict(cr)}")
        lines.append(f"      replay={_planet_row_as_dict(rr)}")
        if isinstance(cr, (list, tuple)) and isinstance(rr, (list, tuple)):
            diffs = []
            for j in geometry_field_indices:
                if len(cr) <= j or len(rr) <= j:
                    continue
                if abs(float(cr[j]) - float(rr[j])) > _PLANET_GEOMETRY_ABS_EPS:
                    diffs.append(field_names[j])
            lines.append(f"      geometry_fields_differing={diffs!r}")
    lines.append("  plain snapshot:")
    for ln in _plain_sync_debug_lines(plain):
        lines.append(f"    {ln}")
    print("\n".join(lines), file=sys.stderr, flush=True)


def _stderr_dump_planet_length_mismatch(
    *,
    replay_planets: list[Any],
    cache_planets: list[Any],
    obs_step: int,
    cpp_episode_step: int,
    action_summary: str,
    plain: Mapping[str, Any] | None,
    sync_context: Mapping[str, Any] | None,
) -> None:
    lines: list[str] = [
        "ORBIT_KAGGLE_CPP_CACHE planet list length mismatch (detailed dump)",
        f"  replay_obs_step={int(obs_step)} cpp_episode_step={int(cpp_episode_step)}",
        f"  len_cache={len(cache_planets)} len_replay={len(replay_planets)}",
        f"  {action_summary}",
    ]
    if sync_context:
        lines.append("  sync_context:")
        for k in sorted(sync_context.keys()):
            lines.append(f"    {k}={sync_context[k]!r}")
    lines.append("  plain snapshot:")
    for ln in _plain_sync_debug_lines(plain):
        lines.append(f"    {ln}")
    print("\n".join(lines), file=sys.stderr, flush=True)


def _assert_planets_rows_match_exact(
    *,
    replay_planets: list[Any],
    cache_planets: list[Any],
    obs_step: int,
    cpp_episode_step: int,
    row_actions: list[dict[str, Any]] | None,
    plain: Mapping[str, Any] | None = None,
    sync_context: Mapping[str, Any] | None = None,
) -> None:
    field_names = _PLANET_ROW_FIELD_NAMES
    action_summary: str
    if row_actions is None:
        action_summary = "none"
    else:
        move_counts: list[int] = []
        for st in row_actions:
            assert isinstance(st, dict)
            moves = st["action"]
            assert isinstance(moves, list)
            move_counts.append(len(moves))
        action_summary = f"moves_per_seat={move_counts!r}"
    assert isinstance(replay_planets, list)
    assert isinstance(cache_planets, list)
    if len(cache_planets) != len(replay_planets):
        _stderr_dump_planet_length_mismatch(
            replay_planets=replay_planets,
            cache_planets=cache_planets,
            obs_step=int(obs_step),
            cpp_episode_step=int(cpp_episode_step),
            action_summary=action_summary,
            plain=plain,
            sync_context=sync_context,
        )
    assert len(cache_planets) == len(replay_planets), (
        "orbit_kaggle_cpp_cache planets length mismatch",
        f"replay_obs_step={int(obs_step)}",
        f"cpp_episode_step={int(cpp_episode_step)}",
        len(cache_planets),
        len(replay_planets),
        action_summary,
    )
    geometry_field_indices = (0, 2, 3, 4)
    for i, (cache_row, replay_row) in enumerate(zip(cache_planets, replay_planets, strict=True)):
        assert isinstance(cache_row, (list, tuple)) and len(cache_row) == 7, (i, type(cache_row))
        assert isinstance(replay_row, (list, tuple)) and len(replay_row) == 7, (i, type(replay_row))
        for j in geometry_field_indices:
            cache_v = float(cache_row[j])
            replay_v = float(replay_row[j])
            geometry_diff = abs(cache_v - replay_v)
            if geometry_diff > _PLANET_GEOMETRY_ABS_EPS:
                _stderr_dump_planet_row_mismatch(
                    field_names=field_names,
                    replay_planets=replay_planets,
                    cache_planets=cache_planets,
                    row_index=i,
                    field_index=j,
                    cache_v=cache_v,
                    replay_v=replay_v,
                    obs_step=int(obs_step),
                    cpp_episode_step=int(cpp_episode_step),
                    action_summary=action_summary,
                    plain=plain,
                    sync_context=sync_context,
                )
            assert geometry_diff <= _PLANET_GEOMETRY_ABS_EPS, (
                "orbit_kaggle_cpp_cache planet geometry mismatch",
                f"replay_obs_step={int(obs_step)}",
                f"cpp_episode_step={int(cpp_episode_step)}",
                f"planet_row_index={i}",
                f"planet_id_cache={int(float(cache_row[0]))}",
                f"planet_id_replay={int(float(replay_row[0]))}",
                f"field={field_names[j]}",
                f"cache_value={cache_v}",
                f"replay_value={replay_v}",
                f"abs_diff={geometry_diff}",
                f"abs_eps={_PLANET_GEOMETRY_ABS_EPS}",
                action_summary,
            )


class OrbitKaggleCppObservationCache:
    __slots__ = (
        "_static_v2",
        "_num_agents",
        "_initialized",
        "_honest_limited_mask_buf",
        "_honest_send_all_mask_buf",
        "_ws_orbit_planet_features",
        "_ws_orbit_planet_mask",
        "_ws_orbit_planet_pairwise_mask",
        "_ws_orbit_planet_pairwise_features",
        "_ws_available_hit_mask",
        "_ws_action_taken_index",
        "_ws_player_mask",
        "_ws_available_action_mask",
        "_last_obs_step",
        "_episode_steps",
    )

    @staticmethod
    def configuration_episode_steps(configuration: Any) -> int:
        return int(_kaggle_configuration_field(configuration, "episodeSteps"))

    def __init__(self, *, configuration: Any, num_agents: int) -> None:
        na = int(num_agents)
        assert na in (2, 4), na
        episode_steps = int(_kaggle_configuration_field(configuration, "episodeSteps"))
        native = importlib.import_module("src.gym.orbit_wars_cpp_ext").orbit_wars_cpp
        ship_speed = float(_kaggle_configuration_field(configuration, "shipSpeed"))
        comet_speed = float(_kaggle_configuration_field(configuration, "cometSpeed"))
        self._static_v2 = native.CppEnvStaticCacheV2(
            na,
            0,
            ship_speed,
            episode_steps,
            comet_speed,
        )
        self._episode_steps = episode_steps
        self._honest_limited_mask_buf = torch.zeros(
            (ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_HIT_CLASSES),
            dtype=torch.int8,
            device=torch.device("cpu"),
        )
        self._honest_send_all_mask_buf = torch.zeros_like(self._honest_limited_mask_buf)
        ws = orbit_cpp_env_minimal_workspace_tensors_cpu()
        self._ws_orbit_planet_features = ws[0]
        self._ws_orbit_planet_mask = ws[1]
        self._ws_orbit_planet_pairwise_mask = ws[2]
        self._ws_orbit_planet_pairwise_features = ws[3]
        self._ws_available_hit_mask = ws[4]
        self._ws_action_taken_index = ws[5]
        self._ws_player_mask = ws[6]
        self._ws_available_action_mask = torch.zeros(
            (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_ACTION_SLOTS,
                ORBIT_PER_PLANET_MOVE_CLASSES,
            ),
            dtype=torch.int8,
            device=torch.device("cpu"),
        )
        self._num_agents = na
        self._initialized = False
        self._last_obs_step = 0

    @property
    def episode_steps(self) -> int:
        return int(self._episode_steps)

    def _assert_plain_step_matches_cache_clock(self, *, plain: dict[str, Any], context: str) -> None:
        obs_step = int(plain["step"])
        assert obs_step == int(self._last_obs_step), (
            context,
            obs_step,
            int(self._last_obs_step),
        )

    def honest_shared_action_mask_limited(
        self,
        requests: torch.Tensor,
        out_action_mask: torch.Tensor,
    ) -> None:
        assert self._initialized
        episode_step = int(self._last_obs_step)
        self._static_v2.honest_shared_action_mask_limited(
            episode_step,
            requests,
            out_action_mask,
        )

    def honest_shared_action_mask_full_cache_warmup_one(
        self,
        *,
        plain: dict[str, Any],
        wall_profiler: WallTreeProfiler | None = None,
    ) -> tuple[int, bool]:
        with profiler_span(wall_profiler, "warmup_one_wrapper_contract"):
            assert self._initialized
            self._assert_plain_step_matches_cache_clock(
                plain=plain,
                context="honest_shared_action_mask_full_cache_warmup_one",
            )
        with profiler_span(wall_profiler, "warmup_one_plain_planets_to_tensor"):
            planet_rows, planet_count = planet_rows_tensor_from_plain_planets_float64(
                plain["planets"],
            )
        return self.honest_shared_action_mask_full_cache_warmup_one_from_rows(
            planet_rows=planet_rows,
            planet_count=planet_count,
            wall_profiler=wall_profiler,
        )

    def honest_shared_action_mask_full_cache_warmup_one_from_rows(
        self,
        *,
        planet_rows: torch.Tensor,
        planet_count: int,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> tuple[int, bool]:
        with profiler_span(wall_profiler, "warmup_one_rows_contract"):
            assert self._initialized
            assert isinstance(planet_rows, torch.Tensor)
            assert tuple(planet_rows.shape) == (ORBIT_PLANET_ACTION_SLOTS, 7), planet_rows.shape
            assert planet_rows.dtype == torch.float64, planet_rows.dtype
            assert not planet_rows.is_cuda
            assert planet_rows.is_contiguous()
            assert 0 < int(planet_count) <= ORBIT_PLANET_ACTION_SLOTS, planet_count
        with profiler_span(wall_profiler, "warmup_one_cpp_profile_enable"):
            profile_cpp = wall_profiler is not None
            self._static_v2.set_wall_profile_enabled(profile_cpp)
        with profiler_span(wall_profiler, "cpp_full_cache_warmup_one"):
            warmup_result = self._static_v2.honest_shared_action_mask_full_cache_warmup_one(
                int(self._last_obs_step) + 1,
                planet_rows,
                int(planet_count),
            )
            if profile_cpp:
                wall_profiler.add_subtree_rows(self._static_v2.wall_profile_rows())
        with profiler_span(wall_profiler, "warmup_one_cpp_profile_disable"):
            if profile_cpp:
                self._static_v2.set_wall_profile_enabled(False)
        with profiler_span(wall_profiler, "warmup_one_return_contract"):
            assert isinstance(warmup_result, tuple), type(warmup_result)
            assert len(warmup_result) == 2, warmup_result
            warmed, finished = warmup_result
            assert isinstance(warmed, int), type(warmed)
            assert isinstance(finished, bool), type(finished)
            assert warmed >= 0, warmed
        return warmed, finished

    def honest_shared_action_mask_full_cache_warmup_stats(self) -> tuple[int, int, int, int, int, int]:
        assert self._initialized
        stats = self._static_v2.honest_shared_action_mask_full_cache_warmup_stats()
        assert isinstance(stats, tuple)
        assert len(stats) == 6
        (
            complete_before_step,
            cursor_done_src_count,
            cursor_src_total,
            entry_count,
            extra_sn,
            last_lookahead_steps,
        ) = stats
        out = (
            int(complete_before_step),
            int(cursor_done_src_count),
            int(cursor_src_total),
            int(entry_count),
            int(extra_sn),
            int(last_lookahead_steps),
        )
        assert 0 <= out[1] <= out[2], out
        assert out[3] >= 0, out
        assert out[4] >= 0, out
        assert out[5] >= 0, out
        return out

    def honest_shared_action_mask_full_cache_prune_before_current_future(self) -> int:
        assert self._initialized
        pruned = self._static_v2.honest_shared_action_mask_full_cache_prune_before(
            int(self._last_obs_step) + 1,
        )
        assert isinstance(pruned, int), type(pruned)
        assert pruned >= 0, pruned
        return pruned

    def send_all_from_external(
        self,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        out_action_mask: torch.Tensor,
    ) -> None:
        assert self._initialized
        episode_step = int(self._last_obs_step)
        self._static_v2.send_all_from_external(
            episode_step,
            fleet_rows,
            planet_rows,
            int(planet_count),
            out_action_mask,
        )

    def honest_shared_hit_kind_last(self) -> torch.Tensor:
        hit_kind = self._static_v2.honest_shared_hit_kind_last()
        assert isinstance(hit_kind, torch.Tensor)
        return hit_kind

    def honest_shared_hit_slot_last(self) -> torch.Tensor:
        hit_slot = self._static_v2.honest_shared_hit_slot_last()
        assert isinstance(hit_slot, torch.Tensor)
        return hit_slot

    def honest_shared_hit_steps_last(self) -> torch.Tensor:
        hit_steps = self._static_v2.honest_shared_hit_steps_last()
        assert isinstance(hit_steps, torch.Tensor)
        return hit_steps

    def honest_shared_intercept_fail_reason_last(self) -> torch.Tensor:
        fail_reason = self._static_v2.honest_shared_intercept_fail_reason_last()
        assert isinstance(fail_reason, torch.Tensor)
        return fail_reason

    def honest_shared_send_ships_last(self) -> torch.Tensor:
        send_ships = self._static_v2.honest_shared_send_ships_last()
        assert isinstance(send_ships, torch.Tensor)
        return send_ships

    def cpp_honest_shared_dir_last(self) -> tuple[torch.Tensor, torch.Tensor]:
        xy = self._static_v2.honest_shared_dir_last()
        assert isinstance(xy, tuple)
        assert len(xy) == 2
        dir_x, dir_y = xy
        assert isinstance(dir_x, torch.Tensor)
        assert isinstance(dir_y, torch.Tensor)
        return dir_x, dir_y

    def cpp_honest_shared_angle_for_action(
        self,
        *,
        src_slot: int,
        dst_slot: int,
        ship_count: int,
        action_class: int,
    ) -> float:
        dir_x, dir_y = self.cpp_honest_shared_dir_last()
        src = int(src_slot)
        cls = int(action_class)
        dst = int(dst_slot)
        _ = int(ship_count)
        assert dst == cls // int(ORBIT_MOVE_CLASSES_PER_TARGET), (dst, cls)
        move_subindex = cls % int(ORBIT_MOVE_CLASSES_PER_TARGET)
        assert move_subindex in ORBIT_MOVE_SEND_SUBINDICES, (cls, move_subindex)
        hit_cls = dst * int(ORBIT_HIT_CLASSES_PER_TARGET) + int(move_subindex)
        cache_x = float(dir_x[src, hit_cls].item())
        cache_y = float(dir_y[src, hit_cls].item())
        assert math.isfinite(cache_x) and math.isfinite(cache_y), (
            src,
            cls,
            cache_x,
            cache_y,
        )
        return math.atan2(cache_y, cache_x)

    def reset_from_kaggle_plain(
        self,
        *,
        plain: dict[str, Any],
        num_agents: int,
    ) -> None:
        na = int(num_agents)
        assert na in (2, 4), na
        obs_step = int(plain["step"])
        assert obs_step == 0, obs_step
        assert not self._initialized
        planet_rows, planet_count = planet_rows_tensor_from_plain_planets_float64(
            plain["planets"],
        )
        comet_ids = plain["comet_planet_ids"]
        assert isinstance(comet_ids, list)
        assert len(comet_ids) == 0, comet_ids
        angular_velocity = float(plain["angular_velocity"])
        self._static_v2.reset(angular_velocity, planet_rows, int(planet_count))
        self._num_agents = na
        self._initialized = True
        self._last_obs_step = obs_step

    def step_noop_and_update_comets_from_kaggle_plain(
        self,
        *,
        plain: dict[str, Any],
        num_agents: int,
    ) -> None:
        na = int(num_agents)
        assert na in (2, 4), na
        assert na == int(self._num_agents), (na, int(self._num_agents))
        assert self._initialized
        obs_step = int(plain["step"])
        assert obs_step == self._last_obs_step + 1, (self._last_obs_step, obs_step)
        if obs_step in _ORBIT_COMET_SPAWN_UPDATE_STEPS:
            comet_ids = plain["comet_planet_ids"]
            comet_groups = plain["comets"]
            assert isinstance(comet_ids, list), (obs_step, comet_ids)
            assert isinstance(comet_groups, list), (obs_step, comet_groups)
            assert len(comet_ids) in (0, 4), (obs_step, comet_ids)
            if len(comet_ids) == 0:
                assert len(comet_groups) == 0, (obs_step, comet_groups)
                self._last_obs_step = obs_step
                return
            path_by_pid = orbit_comet_path_by_planet_id(comet_groups, plain["planets"])
            ids = sorted(int(pid) for pid in comet_ids)
            path_index = int(path_by_pid[ids[0]][0])
            assert all(int(path_by_pid[pid][0]) == path_index for pid in ids), obs_step
            self._static_v2.update_comet_in_noop_cache(
                obs_step,
                path_index,
                obs_step - path_index,
                ids,
                [list(path_by_pid[pid][1]) for pid in ids],
            )
        self._last_obs_step = obs_step

    def _cpp_wall_profile_begin(self, wall_profiler: WallTreeProfiler | None) -> bool:
        profile_cpp = wall_profiler is not None
        self._static_v2.set_wall_profile_enabled(profile_cpp)
        return profile_cpp

    def _cpp_wall_profile_add_and_disable(self, wall_profiler: WallTreeProfiler, profile_cpp: bool) -> None:
        assert profile_cpp
        wall_profiler.add_subtree_rows(self._static_v2.wall_profile_rows())
        self._static_v2.set_wall_profile_enabled(False)

    def honest_available_mask_stacked_for_seats(
        self,
        *,
        batched: dict[str, Any],
        seats_plain: list[dict[str, Any]],
        wall_profiler: WallTreeProfiler | None = None,
    ) -> torch.Tensor:
        assert len(seats_plain) >= 1
        rows = batched["planet_rows"]
        counts = batched["planet_count"]
        assert isinstance(rows, torch.Tensor)
        assert isinstance(counts, torch.Tensor)
        na = len(seats_plain)
        assert na in (2, 4), na
        assert int(rows.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS, (rows.shape[0], na)
        cpu = torch.device("cpu")
        slot0 = orbit_policy_slot_for_compact_agent(0, na)
        planet_rows0 = rows[slot0].to(dtype=torch.float32, device=cpu).contiguous()
        planet_count = int(counts[slot0].item())
        for compact_idx in range(1, na):
            slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
            assert int(counts[slot].item()) == planet_count, (slot, planet_count)
            pr = rows[slot].to(dtype=torch.float32, device=cpu).contiguous()
            assert torch.equal(pr, planet_rows0), slot
        with profiler_span(wall_profiler, "cpp_send_all_world"):
            fleet_rows = fleet_rows_tensor_from_plain_fleets_float64(seats_plain[0]["fleets"])
            profile_cpp = self._cpp_wall_profile_begin(wall_profiler)
            self._static_v2.fill_available_action_mask_from_rows(
                int(self._last_obs_step),
                fleet_rows,
                planet_rows0,
                int(planet_count),
                self._ws_available_action_mask,
            )
            if profile_cpp:
                assert wall_profiler is not None
                self._cpp_wall_profile_add_and_disable(wall_profiler, profile_cpp)
        return self._ws_available_action_mask

    def honest_hit_mask_stacked_for_seats(
        self,
        *,
        batched: dict[str, Any],
        seats_plain: list[dict[str, Any]],
        wall_profiler: WallTreeProfiler | None = None,
    ) -> torch.Tensor:
        assert len(seats_plain) >= 1
        rows = batched["planet_rows"]
        counts = batched["planet_count"]
        assert isinstance(rows, torch.Tensor)
        assert isinstance(counts, torch.Tensor)
        na = len(seats_plain)
        assert na in (2, 4), na
        assert int(rows.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS, (rows.shape[0], na)
        cpu = torch.device("cpu")
        slot0 = orbit_policy_slot_for_compact_agent(0, na)
        planet_rows0 = rows[slot0].to(dtype=torch.float32, device=cpu).contiguous()
        planet_count = int(counts[slot0].item())
        for compact_idx in range(1, na):
            slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
            assert int(counts[slot].item()) == planet_count, (slot, planet_count)
            pr = rows[slot].to(dtype=torch.float32, device=cpu).contiguous()
            assert torch.equal(pr, planet_rows0), slot
        return torch.zeros(
            (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_ACTION_SLOTS,
                ORBIT_PER_PLANET_HIT_CLASSES,
            ),
            dtype=torch.int8,
            device=cpu,
        )

    def available_mask_for_player(
        self,
        *,
        batched: dict[str, torch.Tensor],
        fleet_rows: torch.Tensor,
        seat_index: int,
        player_id: int,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> torch.Tensor:
        rows = batched["planet_rows"]
        counts = batched["planet_count"]
        assert isinstance(rows, torch.Tensor)
        assert isinstance(counts, torch.Tensor)
        with profiler_span(wall_profiler, "tensor_slice"):
            planet_rows = rows[int(seat_index)].to(
                dtype=torch.float32,
                device=torch.device("cpu"),
            ).contiguous()
            planet_count = int(counts[int(seat_index)].item())
        with profiler_span(wall_profiler, "cpp_send_all_world"):
            profile_cpp = self._cpp_wall_profile_begin(wall_profiler)
            self.send_all_from_external(
                fleet_rows,
                planet_rows,
                planet_count,
                self._honest_send_all_mask_buf,
            )
            if profile_cpp:
                assert wall_profiler is not None
                self._cpp_wall_profile_add_and_disable(wall_profiler, profile_cpp)
        with profiler_span(wall_profiler, "action_mask_from_send_all"):
            mask = _orbit_available_action_mask_from_send_all_hit_mask_for_player(
                send_all_hit_mask=self._honest_send_all_mask_buf,
                planet_rows=planet_rows,
                planet_count=planet_count,
                player_id=player_id,
                wall_profiler=wall_profiler,
            )
        assert isinstance(mask, torch.Tensor)
        assert tuple(mask.shape) == (
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PER_PLANET_MOVE_CLASSES,
        ), mask.shape
        assert mask.dtype == torch.int8
        assert not mask.is_cuda
        return mask

    def fleet_arrivals_from_plain(self, *, plain: dict[str, Any], horizon: int) -> torch.Tensor:
        self._assert_plain_step_matches_cache_clock(
            plain=plain,
            context="fleet_arrivals_from_plain",
        )
        fleet_rows = fleet_rows_tensor_from_plain_fleets_float64(plain["fleets"])
        arrivals = self._static_v2.fleet_arrivals_from_rows(
            int(self._last_obs_step),
            fleet_rows,
            int(horizon),
        )
        assert isinstance(arrivals, torch.Tensor)
        assert tuple(arrivals.shape) == (
            int(horizon),
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PLAYER_AXIS_SLOTS,
        ), arrivals.shape
        assert arrivals.dtype == torch.float32
        assert not arrivals.is_cuda
        return arrivals

    def snapshot_policy_obs_from_plain_seats_cpu(
        self,
        *,
        plain: dict[str, Any],
        seats_plain: list[dict[str, Any]],
        policy_slots: tuple[int, ...],
        ship_speed: float,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> dict[str, torch.Tensor]:
        with profiler_span(wall_profiler, "snapshot_contract"):
            assert self._initialized
            self._assert_plain_step_matches_cache_clock(
                plain=plain,
                context="snapshot_policy_obs_from_plain_seats_cpu",
            )
            na = int(self._num_agents)
            assert na in (2, 4), na
            assert len(seats_plain) == na, (len(seats_plain), na)
            assert len(policy_slots) >= 1, policy_slots
            assert float(ship_speed) > 0.0, ship_speed
            for seat, seat_plain in enumerate(seats_plain):
                assert int(seat_plain["player"]) == int(seat), (seat, seat_plain["player"])
                assert seat_plain["planets"] == plain["planets"]
                assert seat_plain["fleets"] == plain["fleets"]
                assert int(seat_plain["step"]) == int(plain["step"])
            cpu = torch.device("cpu")
        with profiler_span(wall_profiler, "batched_planet_geometry"):
            batched = batched_planet_geometry_only_from_plain_seats(
                seats_plain=seats_plain,
                device=cpu,
            )
        with profiler_span(wall_profiler, "rows_from_batched"):
            slot0 = orbit_policy_slot_for_compact_agent(0, na)
            planet_rows = batched["planet_rows"][slot0].to(dtype=torch.float32, device=cpu).contiguous()
            planet_count = int(batched["planet_count"][slot0].item())
            fleet_rows = fleet_rows_tensor_from_plain_fleets_float64(plain["fleets"])
        with profiler_span(wall_profiler, "cpp_fill_policy_obs"):
            profile_cpp = self._cpp_wall_profile_begin(wall_profiler)
            self._static_v2.fill_policy_obs_from_rows(
                int(self._last_obs_step),
                fleet_rows,
                planet_rows,
                int(planet_count),
                self._ws_orbit_planet_features,
                self._ws_orbit_planet_mask,
                self._ws_orbit_planet_pairwise_mask,
                self._ws_orbit_planet_pairwise_features,
                self._ws_action_taken_index,
                self._ws_player_mask,
            )
            if profile_cpp:
                assert wall_profiler is not None
                self._cpp_wall_profile_add_and_disable(wall_profiler, profile_cpp)
        with profiler_span(wall_profiler, "available_action_mask"):
            available_action_mask = self.honest_available_mask_stacked_for_seats(
                batched=batched,
                seats_plain=seats_plain,
                wall_profiler=wall_profiler,
            )
        with profiler_span(wall_profiler, "fleet_arrival_features"):
            arrival_features = torch.zeros(
                (
                    ORBIT_PLAYER_AXIS_SLOTS,
                    ORBIT_MAX_PLANETS,
                    ORBIT_PLANET_ARRIVAL_HORIZON,
                    ORBIT_PLAYER_AXIS_SLOTS,
                    ORBIT_PLANET_TEMPORAL_FEATURES,
                ),
                dtype=torch.float32,
                device=cpu,
            )
            assert isinstance(arrival_features, torch.Tensor)
            assert tuple(arrival_features.shape) == (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_MAX_PLANETS,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ), arrival_features.shape
            assert arrival_features.dtype == torch.float32
            assert not arrival_features.is_cuda
            profile_cpp = self._cpp_wall_profile_begin(wall_profiler)
            self._static_v2.fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
                int(self._last_obs_step),
                fleet_rows,
                planet_rows,
                int(planet_count),
                int(ORBIT_PLANET_ARRIVAL_HORIZON),
                self._ws_orbit_planet_features,
                self._ws_orbit_planet_pairwise_features,
                self._ws_available_hit_mask,
                arrival_features,
            )
            if profile_cpp:
                assert wall_profiler is not None
                self._cpp_wall_profile_add_and_disable(wall_profiler, profile_cpp)
            arrival_features[:, planet_count:] = 0.0
        with profiler_span(wall_profiler, "policy_obs_dict"):
            policy_obs = {
                "orbit_planet_features": self._ws_orbit_planet_features,
                "orbit_planet_arrival_features": arrival_features,
                "orbit_enemy_mask": orbit_self_enemy_mask(na, device=cpu),
                "orbit_planet_mask": self._ws_orbit_planet_mask,
                "orbit_planet_pairwise_mask": self._ws_orbit_planet_pairwise_mask,
                "orbit_planet_pairwise_features": self._ws_orbit_planet_pairwise_features,
                "available_action_mask": available_action_mask,
                "action_taken_index": self._ws_action_taken_index,
                "player_mask": self._ws_player_mask,
            }
        with profiler_span(wall_profiler, "select_policy_slots"):
            idx = torch.tensor(tuple(int(slot) for slot in policy_slots), dtype=torch.long, device=cpu)
            return {k: v.index_select(0, idx).clone().contiguous() for k, v in policy_obs.items()}

    def fleet_arrival_features_from_plain(self, *, plain: dict[str, Any], horizon: int) -> torch.Tensor:
        self._assert_plain_step_matches_cache_clock(
            plain=plain,
            context="fleet_arrival_features_from_plain",
        )
        planets = plain["planets"]
        assert isinstance(planets, list)
        planet_rows, planet_count = planet_rows_tensor_from_plain_planets_float64(planets)
        assert 0 <= planet_count <= ORBIT_MAX_PLANETS, planet_count
        fleet_rows = fleet_rows_tensor_from_plain_fleets_float64(plain["fleets"])
        features = self._static_v2.fleet_arrival_features_from_rows(
            int(self._last_obs_step),
            fleet_rows,
            planet_rows,
            int(planet_count),
            int(horizon),
        )
        assert isinstance(features, torch.Tensor)
        assert tuple(features.shape) == (
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_ACTION_SLOTS,
            int(horizon),
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_TEMPORAL_FEATURES,
        ), features.shape
        assert features.dtype == torch.float32
        assert not features.is_cuda
        features[:, planet_count:] = 0.0
        return features

    def fleet_arrivals_resolution_from_plain(
        self,
        *,
        plain: dict[str, Any],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_plain_step_matches_cache_clock(
            plain=plain,
            context="fleet_arrivals_resolution_from_plain",
        )
        planets = plain["planets"]
        assert isinstance(planets, list)
        planet_rows, planet_count = planet_rows_tensor_from_plain_planets_float64(planets)
        fleet_rows = fleet_rows_tensor_from_plain_fleets_float64(plain["fleets"])
        resolution = self._static_v2.fleet_arrivals_resolution_from_rows(
            int(self._last_obs_step),
            fleet_rows,
            planet_rows,
            int(planet_count),
            int(horizon),
        )
        assert isinstance(resolution, tuple) and len(resolution) == 2, type(resolution)
        owners, ships = resolution
        assert isinstance(owners, torch.Tensor)
        assert isinstance(ships, torch.Tensor)
        assert tuple(owners.shape) == (int(horizon), ORBIT_PLANET_ACTION_SLOTS), owners.shape
        assert tuple(ships.shape) == (int(horizon), ORBIT_PLANET_ACTION_SLOTS), ships.shape
        assert owners.dtype == torch.int32
        assert ships.dtype == torch.float32
        assert not owners.is_cuda
        assert not ships.is_cuda
        return owners, ships

    def fleet_hit_traces_from_plain_fleets(
        self, *, fleets: list[Any], horizon: int
    ) -> list[dict[str, Any]]:
        fleet_rows = fleet_rows_tensor_from_plain_fleets_float64(fleets)
        traces = self._static_v2.fleet_hit_traces_from_rows(
            int(self._last_obs_step),
            fleet_rows,
            int(horizon),
        )
        assert isinstance(traces, list)
        out: list[dict[str, Any]] = []
        for row in traces:
            assert isinstance(row, dict), type(row)
            out.append(dict(row))
        return out

