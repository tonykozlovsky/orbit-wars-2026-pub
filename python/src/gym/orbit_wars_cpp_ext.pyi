from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch


class CppEnvStaticCacheV2:
    def __init__(
        self,
        num_agents: int,
        orbit_instance_id: int,
        ship_speed: float,
        episode_steps: int,
        comet_speed: float,
    ) -> None: ...
    def reset(
        self,
        angular_velocity: float,
        planet_rows: torch.Tensor,
        planet_count: int,
    ) -> None: ...
    def update_comet_in_noop_cache(
        self,
        current_episode_step: int,
        path_index: int,
        comet_internal_id: int,
        planet_ids: Any,
        paths_kaggle_yx: Any,
    ) -> None: ...
    def noop_trajectory_length(self) -> int: ...
    def noop_trajectory_planets_tensor(self) -> torch.Tensor: ...
    def noop_trajectory_planets_row_tensor(
        self,
        step_index: int,
    ) -> torch.Tensor: ...
    def honest_shared_action_mask_limited(
        self,
        episode_step: int,
        requests: torch.Tensor,
        out_action_mask: torch.Tensor,
    ) -> None: ...
    def honest_shared_action_mask_all_geometry(
        self,
        episode_step: int,
        out_action_mask: torch.Tensor,
    ) -> None: ...
    def send_all_from_external(
        self,
        episode_step: int,
        planet_rows: torch.Tensor,
        planet_count: int,
        out_action_mask: torch.Tensor,
    ) -> None: ...
    def fill_policy_obs_from_rows(
        self,
        episode_step: int,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        orbit_planet_features: torch.Tensor,
        orbit_planet_mask: torch.Tensor,
        orbit_planet_pairwise_mask: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        action_taken_index: torch.Tensor,
        player_mask: torch.Tensor,
    ) -> None: ...
    def fleet_arrivals_from_rows(
        self,
        episode_step: int,
        fleet_rows: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor: ...
    def fleet_arrival_features_from_rows(
        self,
        episode_step: int,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
    ) -> torch.Tensor: ...
    def fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
        self,
        episode_step: int,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
        orbit_planet_features: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        available_hit_mask: torch.Tensor,
        orbit_planet_arrival_features: torch.Tensor,
    ) -> None: ...
    def fleet_arrivals_resolution_from_rows(
        self,
        episode_step: int,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def fleet_hit_traces_from_rows(
        self,
        episode_step: int,
        fleet_rows: torch.Tensor,
        horizon: int,
    ) -> list[Any]: ...
    def honest_shared_hit_kind_last(self) -> torch.Tensor: ...
    def honest_shared_hit_slot_last(self) -> torch.Tensor: ...
    def honest_shared_hit_steps_last(self) -> torch.Tensor: ...
    def honest_shared_intercept_fail_reason_last(self) -> torch.Tensor: ...
    def honest_shared_dir_last(self) -> tuple[torch.Tensor, torch.Tensor]: ...


class CppEnvLiveV2:
    def __init__(
        self,
        num_agents: int,
        orbit_instance_id: int,
        ship_speed: float,
        episode_steps: int,
        comet_speed: float,
        enable_state_trace: bool = True,
    ) -> None: ...
    def reset(
        self,
        angular_velocity: float,
        planet_rows: torch.Tensor,
        planet_count: int,
        orbit_planet_features: torch.Tensor,
        orbit_planet_mask: torch.Tensor,
        orbit_planet_pairwise_mask: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        action_taken_index: torch.Tensor,
        player_mask: torch.Tensor,
    ) -> None: ...
    def update_comets_from_state(
        self,
        comet_planet_ids: Any,
        comet_path_by_planet_id: Any,
    ) -> None: ...
    def step(
        self,
        action_classes: torch.Tensor,
        orbit_planet_features: torch.Tensor,
        orbit_planet_mask: torch.Tensor,
        orbit_planet_pairwise_mask: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        action_taken_index: torch.Tensor,
        player_mask: torch.Tensor,
    ) -> None: ...
    def reset_trace_get(self) -> str: ...
    def step_trace_get(self) -> str: ...
    def set_wall_profile_enabled(self, enabled: bool) -> None: ...
    def wall_profile_rows(self) -> list[Any]: ...
    def orbit_episode_terminal(self) -> bool: ...
    def fleet_ship_total_int_for_owner(self, owner: int) -> int: ...
    def player_alive_for_owner(self, owner: int) -> bool: ...
    def planet_count_int_for_owner(self, owner: int) -> int: ...
    def production_sum_for_owner(self, owner: int) -> float: ...
    def game_result_for_owner(self, owner: int) -> float: ...
    def fleet_delta_for_owner(self, owner: int) -> float: ...
    def planets_delta_for_owner(self, owner: int) -> float: ...
    def production_delta_for_owner(self, owner: int) -> float: ...
    def step_metric_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def tape_kaggle_planets_rows(self) -> list[Any]: ...
    def tape_kaggle_fleets_rows(self) -> list[Any]: ...
    def angular_velocity(self) -> float: ...
    def ship_speed(self) -> float: ...
    def episode_step(self) -> int: ...
    def kaggle_observation_step(self) -> int: ...
    def honest_shared_send_all_hit_mask(
        self,
        out_hit_mask: torch.Tensor,
    ) -> None: ...
    def merge_dataset_available_hit_mask(
        self,
        honest_geometry_mask: torch.Tensor,
        out_hit_mask: torch.Tensor,
    ) -> None: ...
    def honest_shared_hit_kind_last(self) -> torch.Tensor: ...
    def honest_shared_hit_slot_last(self) -> torch.Tensor: ...
    def honest_shared_hit_steps_last(self) -> torch.Tensor: ...
    def honest_shared_intercept_fail_reason_last(self) -> torch.Tensor: ...
    def honest_shared_dir_last(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def honest_shared_angle(
        self,
        src_slot: int,
        dst_slot: int,
        ship_count: int,
    ) -> float: ...
    def honest_shared_angle_or_nan(
        self,
        src_slot: int,
        dst_slot: int,
        ship_count: int,
    ) -> float: ...
    def honest_shared_intercept_trace(
        self,
        src_slot: int,
        dst_slot: int,
        ship_subindex: int,
    ) -> torch.Tensor: ...
    def fleet_arrivals_from_state(self, horizon: int) -> torch.Tensor: ...
    def fleet_arrivals_from_rows(
        self,
        fleet_rows: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor: ...
    def fleet_arrival_features_from_state(self, horizon: int) -> torch.Tensor: ...
    def fleet_arrival_features_from_rows(
        self,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
    ) -> torch.Tensor: ...
    def fleet_arrival_features_and_fill_future_resolution_planet_features_from_state(
        self,
        horizon: int,
        orbit_planet_features: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        available_hit_mask: torch.Tensor,
        orbit_planet_arrival_features: torch.Tensor,
    ) -> None: ...
    def fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
        self,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
        orbit_planet_features: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        available_hit_mask: torch.Tensor,
        orbit_planet_arrival_features: torch.Tensor,
    ) -> None: ...
    def fill_policy_obs_from_rows(
        self,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        orbit_planet_features: torch.Tensor,
        orbit_planet_mask: torch.Tensor,
        orbit_planet_pairwise_mask: torch.Tensor,
        orbit_planet_pairwise_features: torch.Tensor,
        action_taken_index: torch.Tensor,
        player_mask: torch.Tensor,
    ) -> None: ...
    def fill_future_resolution_planet_features_from_state(
        self,
        horizon: int,
        orbit_planet_features: torch.Tensor,
    ) -> None: ...
    def fill_future_resolution_planet_features_from_rows(
        self,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
        orbit_planet_features: torch.Tensor,
    ) -> None: ...
    def fleet_takeover_cost_features_from_state(self, horizon: int) -> torch.Tensor: ...
    def fleet_arrivals_resolution(
        self,
        arrivals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def fleet_arrivals_resolution_from_state(
        self,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def fleet_arrivals_resolution_from_rows(
        self,
        fleet_rows: torch.Tensor,
        planet_rows: torch.Tensor,
        planet_count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def fleet_hit_traces_from_state(self, horizon: int) -> list[Any]: ...
    def fleet_hit_traces_from_rows(
        self,
        fleet_rows: torch.Tensor,
        horizon: int,
    ) -> list[Any]: ...
    def comet_mask_inputs_py(self) -> tuple[Any, ...]: ...
    def noop_trajectory_length(self) -> int: ...
    def noop_trajectory_planets_tensor(self) -> torch.Tensor: ...
    def noop_trajectory_planets_row_tensor(
        self,
        step_index: int,
    ) -> torch.Tensor: ...
    def assert_planets_match_noop_cache(self, step_index: int) -> None: ...


class _OrbitWarsCppNative(Protocol):
    CppEnvStaticCacheV2: type[CppEnvStaticCacheV2]
    CppEnvLiveV2: type[CppEnvLiveV2]
    orbit_wars_format_double_for_reset_trace: Callable[[float], str]
    orbit_wars_format_double_two_decimals_for_reset_trace: Callable[[float], str]
    orbit_wars_policy_obs_edge_distance: Callable[[float, float, float, float], float]
    orbit_wars_policy_obs_pairwise_distance_matrix: Callable[
        [torch.Tensor], torch.Tensor
    ]
    orbit_wars_action_taken_index_from_classes: Callable[[torch.Tensor], torch.Tensor]
    orbit_wars_ship_count_for_grid_subindex: Callable[[int], int]
    orbit_wars_fleet_speed: Callable[[float, float], float]
    orbit_wars_available_action_mask_from_planet_rows: Callable[
        [torch.Tensor, int, int, float, float, Any, Any], torch.Tensor
    ]
    def __getattr__(self, name: str) -> Any: ...


orbit_wars_cpp: _OrbitWarsCppNative
