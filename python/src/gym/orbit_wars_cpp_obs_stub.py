"""Orbit policy obs buffers filled by the ``orbit_wars_cpp`` PyTorch extension (in-place)."""

from __future__ import annotations

import math
from typing import Any

import torch

from .dict_io_contract import dict_io_contract_validation_enabled
from .obs_wrapper import (
    ORBIT_EDGE_FEATURES,
    ORBIT_HIT_CLASSES_PER_TARGET,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_MOVE_SEND_SUBINDICES,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_ENEMY_AXIS_SLOTS,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
    orbit_self_enemy_mask,
)
from .orbit_cpp_plain_sync import (
    orbit_comet_path_by_planet_id,
    orbit_cpp_env_apply_comet_sync_update_one,
    orbit_cpp_env_reset_no_comets,
)
from .orbit_wars_cpp_ext import orbit_wars_cpp

_ORBIT_PLANET_ROW_LEN = 7
_ORBIT_FLEET_ROW_LEN = 7


def _planet_rows_tensor_from_plain(planets: list[Any]) -> tuple[torch.Tensor, int]:
    assert isinstance(planets, list)
    n = len(planets)
    assert n <= ORBIT_MAX_PLANETS, (n, ORBIT_MAX_PLANETS)
    rows = torch.zeros((ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), dtype=torch.float64)
    for i in range(n):
        row = planets[i]
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        for j in range(_ORBIT_PLANET_ROW_LEN):
            rows[i, j] = float(row[j])
    return rows.contiguous(), n


class OrbitWarsCppObsStub:
    """Buffers sized for max orbit seats; ``reset`` / ``step`` produce ``*_CPP`` tensors."""

    def __init__(self, *, orbit_env: Any, device: torch.device) -> None:
        self._orbit_env = orbit_env
        self._device = device
        assert hasattr(orbit_env, "_orbit_instance_id")
        assert hasattr(orbit_env, "_env")
        self._cpp_env = orbit_wars_cpp.CppEnvLiveV2(
            int(orbit_env.num_agents),
            int(orbit_env._orbit_instance_id),
            float(orbit_env._env.configuration.shipSpeed),
            int(orbit_env._env.configuration.episodeSteps),
            float(orbit_env._env.configuration.cometSpeed),
            bool(orbit_env._cpp_env_obs_validate),
        )
        self._static_cache = orbit_wars_cpp.CppEnvStaticCacheV2(
            int(orbit_env.num_agents),
            int(orbit_env._orbit_instance_id),
            float(orbit_env._env.configuration.shipSpeed),
            int(orbit_env._env.configuration.episodeSteps),
            float(orbit_env._env.configuration.cometSpeed),
        )
        self._last_cpp_reset_trace = ""
        self._last_cpp_step_trace = ""
        self._honest_shared_action_mask_buf = torch.zeros(
            (ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_HIT_CLASSES),
            dtype=torch.int8,
            device=torch.device("cpu"),
        )
        self._available_hit_mask_buf = torch.zeros(
            (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_ACTION_SLOTS,
                ORBIT_PER_PLANET_HIT_CLASSES,
            ),
            dtype=torch.int8,
            device=torch.device("cpu"),
        )
        self._orbit_enemy_mask_cpp = orbit_self_enemy_mask(
            int(self._orbit_env.num_agents),
            device=self._device,
        ).contiguous()
        self._comet_sync_schedule: tuple[tuple[int, Any], ...] = ()
        self._comet_sync_schedule_idx: int = 0

    def _new_cpp_output(self) -> dict[str, torch.Tensor]:
        p = int(ORBIT_PLAYER_AXIS_SLOTS)
        return {
            "orbit_planet_features_CPP": torch.zeros(
                (p, ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES),
                dtype=torch.float32,
                device=self._device,
            ),
            "orbit_planet_arrival_features_CPP": torch.zeros(
                (
                    p,
                    ORBIT_MAX_PLANETS,
                    ORBIT_PLANET_ARRIVAL_HORIZON,
                    p,
                    ORBIT_PLANET_TEMPORAL_FEATURES,
                ),
                dtype=torch.float32,
                device=self._device,
            ),
            "orbit_enemy_mask_CPP": self._orbit_enemy_mask_cpp,
            "orbit_planet_mask_CPP": torch.zeros(
                (p, ORBIT_MAX_PLANETS),
                dtype=torch.float32,
                device=self._device,
            ),
            "orbit_planet_pairwise_mask_CPP": torch.zeros(
                (p, ORBIT_PLANET_PAIRWISE_COUNT),
                dtype=torch.float32,
                device=self._device,
            ),
            "orbit_planet_pairwise_features_CPP": torch.zeros(
                (p, ORBIT_PLANET_PAIRWISE_COUNT, ORBIT_EDGE_FEATURES),
                dtype=torch.float32,
                device=self._device,
            ),
            "available_action_mask_CPP": torch.zeros(
                (p, ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_MOVE_CLASSES),
                dtype=torch.int8,
                device=self._device,
            ),
            "action_taken_index_CPP": torch.zeros(
                (p, ORBIT_PLANET_ACTION_SLOTS, 1),
                dtype=torch.int32,
                device=self._device,
            ),
            "player_mask_CPP": torch.zeros((p,), dtype=torch.float32, device=self._device),
        }

    @property
    def last_cpp_reset_trace(self) -> str:
        return self._last_cpp_reset_trace

    @property
    def last_cpp_step_trace(self) -> str:
        return self._last_cpp_step_trace

    def cpp_orbit_episode_terminal(self) -> bool:
        return bool(self._cpp_env.orbit_episode_terminal())

    def cpp_fleet_ship_total_int_for_owner(self, owner: int) -> int:
        return int(self._cpp_env.fleet_ship_total_int_for_owner(int(owner)))

    def cpp_player_alive_for_owner(self, owner: int) -> bool:
        return bool(self._cpp_env.player_alive_for_owner(int(owner)))

    def cpp_planet_count_int_for_owner(self, owner: int) -> int:
        return int(self._cpp_env.planet_count_int_for_owner(int(owner)))

    def cpp_production_sum_for_owner(self, owner: int) -> float:
        return float(self._cpp_env.production_sum_for_owner(int(owner)))

    def cpp_game_result_for_owner(self, owner: int) -> float:
        return float(self._cpp_env.game_result_for_owner(int(owner)))

    def cpp_fleet_delta_for_owner(self, owner: int) -> float:
        return float(self._cpp_env.fleet_delta_for_owner(int(owner)))

    def cpp_planets_delta_for_owner(self, owner: int) -> float:
        return float(self._cpp_env.planets_delta_for_owner(int(owner)))

    def cpp_production_delta_for_owner(self, owner: int) -> float:
        return float(self._cpp_env.production_delta_for_owner(int(owner)))

    def cpp_step_metric_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self._cpp_env.step_metric_tensors()
        assert isinstance(result, tuple), type(result)
        assert len(result) == 4, len(result)
        fleet_delta, planets_delta, production_delta, fleet_total = result
        for t in (fleet_delta, planets_delta, production_delta):
            assert isinstance(t, torch.Tensor)
            assert tuple(t.shape) == (int(self._orbit_env.num_agents),), t.shape
            assert t.dtype == torch.float32, t.dtype
            assert not t.is_cuda
        assert isinstance(fleet_total, torch.Tensor)
        assert tuple(fleet_total.shape) == (int(self._orbit_env.num_agents),), fleet_total.shape
        assert fleet_total.dtype == torch.int32, fleet_total.dtype
        assert not fleet_total.is_cuda
        return fleet_delta, planets_delta, production_delta, fleet_total

    def cpp_assert_planets_match_noop_cache(self, step_index: int) -> None:
        self._cpp_env.assert_planets_match_noop_cache(int(step_index))

    def cpp_noop_trajectory_planets_tensor(self) -> torch.Tensor:
        return self._cpp_env.noop_trajectory_planets_tensor()

    def cpp_noop_trajectory_planets_row_tensor(self, step_index: int) -> torch.Tensor:
        return self._cpp_env.noop_trajectory_planets_row_tensor(int(step_index))

    def cpp_honest_shared_hit_kind_last(self) -> torch.Tensor:
        hit_kind = self._cpp_env.honest_shared_hit_kind_last()
        assert isinstance(hit_kind, torch.Tensor)
        return hit_kind

    def cpp_honest_shared_hit_slot_last(self) -> torch.Tensor:
        hit_slot = self._cpp_env.honest_shared_hit_slot_last()
        assert isinstance(hit_slot, torch.Tensor)
        return hit_slot

    def cpp_honest_shared_hit_steps_last(self) -> torch.Tensor:
        hit_steps = self._cpp_env.honest_shared_hit_steps_last()
        assert isinstance(hit_steps, torch.Tensor)
        return hit_steps

    def cpp_honest_shared_intercept_fail_reason_last(self) -> torch.Tensor:
        fail_reason = self._cpp_env.honest_shared_intercept_fail_reason_last()
        assert isinstance(fail_reason, torch.Tensor)
        return fail_reason

    def cpp_honest_shared_send_ships_last(self) -> torch.Tensor:
        send_ships = self._cpp_env.honest_shared_send_ships_last()
        assert isinstance(send_ships, torch.Tensor)
        return send_ships

    def cpp_honest_shared_intercept_trace(
        self, *, src_slot: int, dst_slot: int, ship_subindex: int
    ) -> torch.Tensor:
        trace = self._cpp_env.honest_shared_intercept_trace(
            int(src_slot),
            int(dst_slot),
            int(ship_subindex),
        )
        assert isinstance(trace, torch.Tensor)
        return trace

    def cpp_honest_shared_dir_last(self) -> tuple[torch.Tensor, torch.Tensor]:
        xy = self._cpp_env.honest_shared_dir_last()
        assert isinstance(xy, tuple)
        assert len(xy) == 2
        dir_x, dir_y = xy
        assert isinstance(dir_x, torch.Tensor)
        assert isinstance(dir_y, torch.Tensor)
        return dir_x, dir_y

    def cpp_honest_shared_angle(self, *, src_slot: int, dst_slot: int, ship_count: int) -> float:
        return float(
            self._cpp_env.honest_shared_angle(
                int(src_slot),
                int(dst_slot),
                int(ship_count),
            )
        )

    def cpp_honest_shared_angle_or_nan(
        self, *, src_slot: int, dst_slot: int, ship_count: int
    ) -> float:
        return float(
            self._cpp_env.honest_shared_angle_or_nan(
                int(src_slot),
                int(dst_slot),
                int(ship_count),
            )
        )

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
        dst = int(dst_slot)
        _ = int(ship_count)
        cls = int(action_class)
        assert dst == cls // int(ORBIT_MOVE_CLASSES_PER_TARGET), (dst, cls)
        move_subindex = cls % int(ORBIT_MOVE_CLASSES_PER_TARGET)
        assert move_subindex in ORBIT_MOVE_SEND_SUBINDICES, (cls, move_subindex)
        hit_cls = dst * int(ORBIT_HIT_CLASSES_PER_TARGET) + int(move_subindex)
        cache_x = float(dir_x[src, hit_cls].item())
        cache_y = float(dir_y[src, hit_cls].item())
        assert math.isfinite(cache_x) and math.isfinite(cache_y), (
            src,
            cls,
            hit_cls,
            cache_x,
            cache_y,
        )
        return math.atan2(cache_y, cache_x)

    def cpp_static_honest_shared_action_mask_limited(
        self,
        *,
        requests: torch.Tensor,
    ) -> torch.Tensor:
        self._static_cache.honest_shared_action_mask_limited(
            int(self._cpp_env.episode_step()),
            requests,
            self._honest_shared_action_mask_buf,
        )
        return self._honest_shared_action_mask_buf

    def cpp_static_honest_shared_action_mask_all_geometry(self) -> torch.Tensor:
        self._static_cache.honest_shared_action_mask_all_geometry(
            int(self._cpp_env.episode_step()),
            self._honest_shared_action_mask_buf,
        )
        return self._honest_shared_action_mask_buf

    def cpp_static_send_all_from_external(
        self,
        *,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
    ) -> torch.Tensor:
        assert isinstance(fleet_rows, torch.Tensor)
        assert fleet_rows.dtype == torch.float64, fleet_rows.dtype
        assert not fleet_rows.is_cuda
        assert isinstance(planet_rows, torch.Tensor)
        assert tuple(planet_rows.shape) == (ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), planet_rows.shape
        assert planet_rows.dtype == torch.float32, planet_rows.dtype
        assert not planet_rows.is_cuda
        self._static_cache.send_all_from_external(
            int(self._cpp_env.episode_step()),
            fleet_rows,
            planet_rows,
            int(planet_count),
            self._honest_shared_action_mask_buf,
        )
        return self._honest_shared_action_mask_buf

    def _fill_available_masks_from_state(self, out: dict[str, torch.Tensor]) -> None:
        self._cpp_env.fill_available_action_mask(out["available_action_mask_CPP"])

    def cpp_static_honest_shared_dir_last(self) -> tuple[torch.Tensor, torch.Tensor]:
        xy = self._static_cache.honest_shared_dir_last()
        assert isinstance(xy, tuple)
        assert len(xy) == 2
        dir_x, dir_y = xy
        assert isinstance(dir_x, torch.Tensor)
        assert isinstance(dir_y, torch.Tensor)
        return dir_x, dir_y

    @staticmethod
    def _assert_fleet_arrivals(arrivals: torch.Tensor, horizon: int) -> torch.Tensor:
        assert isinstance(arrivals, torch.Tensor)
        assert tuple(arrivals.shape) == (
            int(horizon),
            ORBIT_MAX_PLANETS,
            ORBIT_PLAYER_AXIS_SLOTS,
        ), arrivals.shape
        assert arrivals.dtype == torch.float32
        assert not arrivals.is_cuda
        return arrivals

    def cpp_fleet_arrivals_from_state(self, *, horizon: int) -> torch.Tensor:
        arrivals = self._cpp_env.fleet_arrivals_from_state(int(horizon))
        return self._assert_fleet_arrivals(arrivals, int(horizon))

    @staticmethod
    def _assert_player_centric_temporal_planet_features(
        features: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        assert isinstance(features, torch.Tensor)
        assert tuple(features.shape) == (
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_MAX_PLANETS,
            int(horizon),
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_TEMPORAL_FEATURES,
        ), features.shape
        assert features.dtype == torch.float32
        assert not features.is_cuda
        return features

    def cpp_fleet_arrival_features_from_state(self, *, horizon: int) -> torch.Tensor:
        features = self._cpp_env.fleet_arrival_features_from_state(int(horizon))
        return self._assert_player_centric_temporal_planet_features(features, int(horizon))

    def cpp_fleet_arrival_features_from_rows(
        self,
        *,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
    ) -> torch.Tensor:
        assert isinstance(fleet_rows, torch.Tensor)
        assert fleet_rows.ndim == 2 and int(fleet_rows.shape[1]) == _ORBIT_FLEET_ROW_LEN, (
            tuple(fleet_rows.shape),
        )
        assert not fleet_rows.is_cuda
        assert fleet_rows.dtype in (torch.float32, torch.float64), fleet_rows.dtype
        assert isinstance(planet_rows, torch.Tensor)
        assert planet_rows.shape == (ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), planet_rows.shape
        assert not planet_rows.is_cuda
        assert planet_rows.dtype in (torch.float32, torch.float64), planet_rows.dtype
        assert 0 <= int(planet_count) <= ORBIT_MAX_PLANETS, planet_count
        features = self._cpp_env.fleet_arrival_features_from_rows(
            fleet_rows.contiguous(),
            planet_rows.contiguous(),
            int(planet_count),
            int(horizon),
        )
        return self._assert_player_centric_temporal_planet_features(features, int(horizon))

    def cpp_fleet_takeover_cost_features_from_state(self, *, horizon: int) -> torch.Tensor:
        features = self._cpp_env.fleet_takeover_cost_features_from_state(int(horizon))
        assert isinstance(features, torch.Tensor)
        assert tuple(features.shape) == (
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_MAX_PLANETS,
            int(horizon),
            ORBIT_PLAYER_AXIS_SLOTS,
        ), features.shape
        assert features.dtype == torch.float32
        assert not features.is_cuda
        return features

    def cpp_fleet_arrivals_from_rows(
        self,
        *,
        fleet_rows: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        assert isinstance(fleet_rows, torch.Tensor)
        assert fleet_rows.ndim == 2 and int(fleet_rows.shape[1]) == _ORBIT_FLEET_ROW_LEN, (
            tuple(fleet_rows.shape),
        )
        assert not fleet_rows.is_cuda
        assert fleet_rows.dtype in (torch.float32, torch.float64), fleet_rows.dtype
        arrivals = self._cpp_env.fleet_arrivals_from_rows(
            fleet_rows.contiguous(),
            int(horizon),
        )
        return self._assert_fleet_arrivals(arrivals, int(horizon))

    @staticmethod
    def _assert_fleet_arrivals_resolution(
        resolution: tuple[torch.Tensor, torch.Tensor],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(resolution, tuple) and len(resolution) == 2, type(resolution)
        owners, ships = resolution
        assert isinstance(owners, torch.Tensor)
        assert isinstance(ships, torch.Tensor)
        assert tuple(owners.shape) == (int(horizon), ORBIT_MAX_PLANETS), owners.shape
        assert tuple(ships.shape) == (int(horizon), ORBIT_MAX_PLANETS), ships.shape
        assert owners.dtype == torch.int32
        assert ships.dtype == torch.float32
        assert not owners.is_cuda
        assert not ships.is_cuda
        return owners, ships

    def cpp_fleet_arrivals_resolution(
        self,
        *,
        arrivals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arrivals = self._assert_fleet_arrivals(arrivals, int(arrivals.shape[0]))
        resolution = self._cpp_env.fleet_arrivals_resolution(arrivals.contiguous())
        return self._assert_fleet_arrivals_resolution(resolution, int(arrivals.shape[0]))

    def cpp_fleet_arrivals_resolution_from_state(
        self,
        *,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolution = self._cpp_env.fleet_arrivals_resolution_from_state(int(horizon))
        return self._assert_fleet_arrivals_resolution(resolution, int(horizon))

    def cpp_fleet_arrivals_resolution_from_rows(
        self,
        *,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(fleet_rows, torch.Tensor)
        assert fleet_rows.ndim == 2 and int(fleet_rows.shape[1]) == _ORBIT_FLEET_ROW_LEN, (
            tuple(fleet_rows.shape),
        )
        assert not fleet_rows.is_cuda
        assert fleet_rows.dtype in (torch.float32, torch.float64), fleet_rows.dtype
        assert isinstance(planet_rows, torch.Tensor)
        assert planet_rows.shape == (ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), planet_rows.shape
        assert not planet_rows.is_cuda
        assert planet_rows.dtype in (torch.float32, torch.float64), planet_rows.dtype
        assert 0 <= int(planet_count) <= ORBIT_MAX_PLANETS, planet_count
        resolution = self._cpp_env.fleet_arrivals_resolution_from_rows(
            fleet_rows.contiguous(),
            planet_rows.contiguous(),
            int(planet_count),
            int(horizon),
        )
        return self._assert_fleet_arrivals_resolution(resolution, int(horizon))

    def cpp_fleet_hit_traces_from_state(self, *, horizon: int) -> list[dict[str, Any]]:
        traces = self._cpp_env.fleet_hit_traces_from_state(int(horizon))
        assert isinstance(traces, list)
        out: list[dict[str, Any]] = []
        for row in traces:
            assert isinstance(row, dict), type(row)
            out_row = dict(row)
            for key in (
                "fleet_id",
                "owner",
                "ships",
                "x0",
                "y0",
                "x1",
                "y1",
                "hit_slot",
                "hit_planet_id",
                "hit_steps",
                "object_x",
                "object_y",
                "object_radius",
            ):
                assert key in out_row, (key, out_row.keys())
            out.append(out_row)
        return out

    def kaggle_plain_planets_and_fleets_for_tape(self) -> tuple[list[Any], list[Any]]:
        plist = list(self._cpp_env.tape_kaggle_planets_rows())
        flist = list(self._cpp_env.tape_kaggle_fleets_rows())
        return plist, flist

    def cpp_tape_kaggle_planet_rows(self) -> list[Any]:
        rows = self._cpp_env.tape_kaggle_planets_rows()
        assert isinstance(rows, list)
        return rows

    def cpp_tape_kaggle_fleet_rows(self) -> list[Any]:
        rows = self._cpp_env.tape_kaggle_fleets_rows()
        assert isinstance(rows, list)
        return rows

    def _assert_cpu_workspace(self) -> None:
        assert self._device.type == "cpu", self._device

    def _assert_cpp_output(self, out: dict[str, torch.Tensor]) -> None:
        assert isinstance(out, dict)
        for _k, t in out.items():
            assert isinstance(t, torch.Tensor)
            assert not t.is_cuda
            assert t.is_contiguous(), (_k,)

    def apply_comet_sync_scheduled_for_current_cpp_clock(self, *, include_static_cache: bool = True) -> None:
        cpp_ep = int(self._cpp_env.episode_step())
        sched = self._comet_sync_schedule
        idx = self._comet_sync_schedule_idx
        while idx < len(sched) and sched[idx][0] == cpp_ep:
            orbit_cpp_env_apply_comet_sync_update_one(self._cpp_env, sched[idx][1])
            if include_static_cache:
                self._static_cache_apply_comet_sync_update_one(sched[idx][1])
            idx += 1
        self._comet_sync_schedule_idx = idx
        if idx < len(sched):
            assert sched[idx][0] > cpp_ep, (cpp_ep, sched[idx][0], sched[idx][1])

    def update_static_cache_comets_from_kaggle_plain(
        self,
        *,
        plain: dict[str, Any],
        num_agents: int,
    ) -> None:
        na = int(num_agents)
        assert na in (2, 4), na
        assert na == int(self._orbit_env.num_agents), (na, int(self._orbit_env.num_agents))
        cur = int(self._cpp_env.episode_step())
        obs_step = cur
        comet_ids = plain["comet_planet_ids"]
        comet_groups = plain["comets"]
        assert isinstance(comet_ids, list), (obs_step, comet_ids)
        assert isinstance(comet_groups, list), (obs_step, comet_groups)
        assert len(comet_ids) in (0, 4), (obs_step, comet_ids)
        if len(comet_ids) == 0:
            assert len(comet_groups) == 0, (obs_step, comet_groups)
            return
        path_by_pid = orbit_comet_path_by_planet_id(comet_groups, plain["planets"])
        ids = sorted(int(pid) for pid in comet_ids)
        path_index = int(path_by_pid[ids[0]][0])
        assert all(int(path_by_pid[pid][0]) == path_index for pid in ids), (obs_step, ids)
        self._static_cache.update_comet_in_noop_cache(
            obs_step,
            path_index,
            obs_step - path_index,
            ids,
            [list(path_by_pid[pid][1]) for pid in ids],
        )

    def _static_cache_apply_comet_sync_update_one(self, upd: Any) -> None:
        target = int(upd["episode_step"])
        cur = int(self._cpp_env.episode_step())
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
        path_by_pid = orbit_comet_path_by_planet_id(groups, planets)
        ids = sorted(int(pid) for pid in c_ids)
        path_index = int(path_by_pid[ids[0]][0])
        assert all(int(path_by_pid[pid][0]) == path_index for pid in ids), (target, ids)
        self._static_cache.update_comet_in_noop_cache(
            target,
            path_index,
            target - path_index,
            ids,
            [list(path_by_pid[pid][1]) for pid in ids],
        )

    def reset_from_external_random_state(
        self,
        *,
        random_state: dict[str, Any],
        plain_state: dict[str, Any],
        omit_reference_comet_sync_schedule: bool = False,
    ) -> dict[str, torch.Tensor]:
        na = int(self._orbit_env.num_agents)
        assert 1 <= na <= ORBIT_PLAYER_AXIS_SLOTS
        self._assert_cpu_workspace()
        assert int(random_state["num_agents"]) == na
        assert "comet_sync_updates" in random_state
        assert "angular_velocity" in plain_state
        av_plain_f32 = torch.tensor(plain_state["angular_velocity"], dtype=torch.float32)
        av_ref_f32 = torch.tensor(random_state["angular_velocity"], dtype=torch.float32)
        assert bool(torch.isfinite(av_plain_f32).item())
        assert bool(torch.isfinite(av_ref_f32).item())
        assert torch.equal(av_plain_f32, av_ref_f32), (
            av_plain_f32.item(),
            av_ref_f32.item(),
        )
        planet_rows, planet_count = _planet_rows_tensor_from_plain(plain_state["planets"])
        out = self._new_cpp_output()
        self._assert_cpp_output(out)
        orbit_cpp_env_reset_no_comets(
            self._cpp_env,
            angular_velocity=float(plain_state["angular_velocity"]),
            planet_rows=planet_rows,
            planet_count=int(planet_count),
            orbit_planet_features=out["orbit_planet_features_CPP"],
            orbit_planet_mask=out["orbit_planet_mask_CPP"],
            orbit_planet_pairwise_mask=out["orbit_planet_pairwise_mask_CPP"],
            orbit_planet_pairwise_features=out["orbit_planet_pairwise_features_CPP"],
            action_taken_index=out["action_taken_index_CPP"],
            player_mask=out["player_mask_CPP"],
        )
        self._static_cache.reset(
            float(plain_state["angular_velocity"]),
            planet_rows,
            int(planet_count),
        )
        raw: list[Any] = (
            [] if omit_reference_comet_sync_schedule else list(random_state["comet_sync_updates"])
        )
        raw.sort(key=lambda u: int(u["episode_step"]))
        for u in raw:
            cg = u["comets_groups"]
            assert isinstance(cg, list)
            if len(cg) > 0:
                assert "planets" in u, sorted(u.keys())
        self._comet_sync_schedule = tuple((int(u["episode_step"]), u) for u in raw)
        self._comet_sync_schedule_idx = 0
        self.apply_comet_sync_scheduled_for_current_cpp_clock()
        self._fill_available_masks_from_state(out)
        self._last_cpp_reset_trace = str(self._cpp_env.reset_trace_get())
        if dict_io_contract_validation_enabled():
            self._assert_inactive_rows_zero(out, na)
        return out

    def step(
        self,
        *,
        action_classes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        w = self._orbit_env._wall_span
        with w("stub_step_contract"):
            na = int(self._orbit_env.num_agents)
            assert 1 <= na <= ORBIT_PLAYER_AXIS_SLOTS
            self._assert_cpu_workspace()
            assert isinstance(action_classes, torch.Tensor)
            assert tuple(action_classes.shape) == (na, ORBIT_PLANET_ACTION_SLOTS)
            assert action_classes.dtype == torch.int32
            assert not action_classes.is_cuda
            wall_prof = self._orbit_env._wall_prof
            profile_cpp = wall_prof is not None
            out = self._new_cpp_output()
            if dict_io_contract_validation_enabled():
                self._assert_cpp_output(out)
        with w("stub_step_set_profile"):
            self._cpp_env.set_wall_profile_enabled(profile_cpp)
        with w("stub_step_action_contiguous"):
            action_classes_c = action_classes.contiguous()
        with w("stub_step_cpp_call"):
            self._cpp_env.step(
                action_classes_c,
                out["orbit_planet_features_CPP"],
                out["orbit_planet_mask_CPP"],
                out["orbit_planet_pairwise_mask_CPP"],
                out["orbit_planet_pairwise_features_CPP"],
                out["action_taken_index_CPP"],
                out["player_mask_CPP"],
            )
            if profile_cpp:
                wall_prof.add_subtree_rows(self._cpp_env.wall_profile_rows())
        with w("stub_step_available_masks"):
            self._fill_available_masks_from_state(out)
            if profile_cpp:
                wall_prof.add_subtree_rows(self._cpp_env.wall_profile_rows())
        with w("stub_step_clock_assert"):
            assert int(self._cpp_env.episode_step()) == int(self._orbit_env._episode_step), (
                int(self._cpp_env.episode_step()),
                int(self._orbit_env._episode_step),
            )
        with w("stub_step_trace_get"):
            self._last_cpp_step_trace = str(self._cpp_env.step_trace_get())
        with w("stub_step_inactive_assert"):
            if dict_io_contract_validation_enabled():
                self._assert_inactive_rows_zero(out, na)
        with w("stub_step_return_outputs"):
            return out

    def _fill_arrival_features_from_state(
        self,
        na: int,
        out: dict[str, torch.Tensor],
    ) -> None:
        w = self._orbit_env._wall_span
        with w("refill_arrival_features_contract"):
            assert int(na) in (2, 4), na
            self._assert_cpp_output(out)
            assert tuple(self._available_hit_mask_buf.shape) == (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_ACTION_SLOTS,
                ORBIT_PER_PLANET_HIT_CLASSES,
            ), self._available_hit_mask_buf.shape
            assert self._available_hit_mask_buf.dtype == torch.int8
            assert not self._available_hit_mask_buf.is_cuda
            assert tuple(out["orbit_planet_arrival_features_CPP"].shape) == (
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_MAX_PLANETS,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            )
        with w("refill_arrival_features_cpp"):
            self._cpp_env.fleet_arrival_features_and_fill_future_resolution_planet_features_from_state(
                int(ORBIT_PLANET_ARRIVAL_HORIZON),
                out["orbit_planet_features_CPP"],
                out["orbit_planet_pairwise_features_CPP"],
                self._available_hit_mask_buf,
                out["orbit_planet_arrival_features_CPP"],
            )

    def refill_arrival_features_from_state(
        self,
        *,
        out: dict[str, torch.Tensor],
    ) -> None:
        w = self._orbit_env._wall_span
        with w("refill_arrival_features_input"):
            na = int(self._orbit_env.num_agents)
        self._fill_arrival_features_from_state(na, out)

    def _assert_inactive_rows_zero(self, out: dict[str, torch.Tensor], na: int) -> None:
        p = int(ORBIT_PLAYER_AXIS_SLOTS)
        active_slots = frozenset(orbit_active_policy_slots(int(na)))
        inactive_slots = [i for i in range(p) if i not in active_slots]
        assert torch.all(out["orbit_planet_features_CPP"][inactive_slots] == 0)
        assert torch.all(out["orbit_planet_arrival_features_CPP"][inactive_slots] == 0)
        assert torch.all(out["orbit_planet_mask_CPP"][inactive_slots] == 0)
        assert torch.all(out["orbit_planet_pairwise_mask_CPP"][inactive_slots] == 0)
        assert torch.all(out["orbit_planet_pairwise_features_CPP"][inactive_slots] == 0)
        assert torch.all(out["available_action_mask_CPP"][inactive_slots] == 0)
        assert torch.all(out["action_taken_index_CPP"][inactive_slots] == 0)
        assert torch.all(out["player_mask_CPP"][inactive_slots] == 0)
        assert tuple(out["orbit_enemy_mask_CPP"].shape) == (p, ORBIT_ENEMY_AXIS_SLOTS)
