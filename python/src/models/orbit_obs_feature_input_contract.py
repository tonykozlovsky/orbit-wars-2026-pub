"""Observation feature contract: continuous vs discrete embedding per logical channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..configs.impala_orbit_model_hyperparams import ORBIT_IMPALA_OBS_FEATURE_LAYOUT
from ..gym.obs_wrapper import (
    ORBIT_EDGE_BASE_FEATURES,
    ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLANET_BASE_FEATURES,
    ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
    ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
)

BcObsFeatureInputKind = Literal["continuous", "embedding"]
BcObsDiscreteEmbeddingTable = Literal[
    "ship_count",
    "ship_cost",
    "production",
    "ship_bucket",
    "temporal_step",
    "episode_step",
    "signed_ship_margin",
]


@dataclass(frozen=True)
class OrbitObsFeatureInputSpec:
    """``embedding_table is None`` → continuous encoder only; else raw + embedding lookup."""

    embedding_table: BcObsDiscreteEmbeddingTable | None = None

    def importance_inputs(self) -> tuple[BcObsFeatureInputKind, ...]:
        if self.embedding_table is None:
            return ("continuous",)
        return ("continuous", "embedding")


@dataclass(frozen=True)
class OrbitObsDiscreteEmbeddingChannelSpec:
    channel_index: int
    table: BcObsDiscreteEmbeddingTable
    label: str


def _continuous() -> OrbitObsFeatureInputSpec:
    return OrbitObsFeatureInputSpec()


def _both(table: BcObsDiscreteEmbeddingTable) -> OrbitObsFeatureInputSpec:
    return OrbitObsFeatureInputSpec(embedding_table=table)


_ORBIT_PLANET_BASE_FEATURE_INPUTS: dict[str, OrbitObsFeatureInputSpec] = {
    "planet_x": _continuous(), # VERIFIED
    "planet_y": _continuous(), # VERIFIED
    "planet_neutral_ships": _both("ship_count"), # VERIFIED
    "planet_episode_step": _both("episode_step"), # VERIFIED
    "planet_is_static": _continuous(), # VERIFIED
    "planet_is_dynamic": _continuous(), # VERIFIED
    "planet_is_comet": _continuous(), # VERIFIED
    "planet_comet_time_before_despawn": _both("temporal_step"), # ?
    "planet_radius": _continuous(), # VERIFIED
    "planet_production": _both("production"), # VERIFIED
    "planet_orbit_radius": _continuous(), # VERIFIED
    "planet_angular_velocity": _continuous(), # VERIFIED
    "planet_sun_angle": _continuous(), # VERIFIED
}

_ORBIT_PLANET_PLAYER_BLOCK_FEATURE_INPUTS: dict[str, OrbitObsFeatureInputSpec] = {
    "planet_player_ships_if_owned": _both("ship_count"), # VERIFIED
    "planet_player_total_fleet_frac": _continuous(), # VERIFIED
    "planet_player_production_if_owned": _both("production"), # VERIFIED
    "planet_player_owner_survival_margin": _both("signed_ship_margin"), # ?
    "planet_player_flip_time_by_player": _both("temporal_step"), # ?
    "planet_player_stable_flip_time_by_player": _both("temporal_step"), # ?
    "planet_player_owner_churn": _both("temporal_step"), # ?
    "planet_player_last_decisive_battle_step": _both("temporal_step"), # ?
    "planet_player_post_horizon_owner_margin": _both("signed_ship_margin"), # ?
}

_ORBIT_EDGE_BASE_FEATURE_INPUTS: dict[str, OrbitObsFeatureInputSpec] = {
    "edge_distance": _continuous(), # VERIFIED
    "edge_src_neutral": _continuous(), # VERIFIED
    "edge_dst_neutral": _continuous(), # VERIFIED

    "edge_min_takeover_bucket": _both("ship_bucket"), # VERIFIED
    "edge_min_takeover_ships": _both("ship_count"), # VERIFIED
    "edge_min_takeover_bucket_available": _continuous(), # VERIFIED
    "edge_min_takeover_bucket_hit_steps": _both("temporal_step"), # VERIFIED

    "edge_min_time_takeover_bucket": _both("ship_bucket"), # VERIFIED
    "edge_min_time_takeover_ships": _both("ship_count"), # VERIFIED
    "edge_min_time_takeover_bucket_available": _continuous(), # VERIFIED
    "edge_min_time_takeover_bucket_hit_steps": _both("temporal_step"), # VERIFIED

    "edge_min_stable_takeover_bucket": _both("ship_bucket"), # VERIFIED
    "edge_min_stable_takeover_ships": _both("ship_count"), # VERIFIED
    "edge_min_stable_takeover_bucket_available": _continuous(),  # VERIFIED
    "edge_min_stable_takeover_bucket_hit_steps": _both("temporal_step"), # VERIFIED

    "edge_min_time_stable_takeover_bucket": _both("ship_bucket"), # VERIFIED
    "edge_min_time_stable_takeover_ships": _both("ship_count"), # VERIFIED
    "edge_min_time_stable_takeover_bucket_available": _continuous(), # VERIFIED
    "edge_min_time_stable_takeover_bucket_hit_steps": _both("temporal_step"), # VERIFIED
    
    "edge_min_neutralize_bucket": _both("ship_bucket"), # VERIFIED
    "edge_min_neutralize_ships": _both("ship_count"), # VERIFIED
    "edge_min_neutralize_bucket_available": _continuous(), # VERIFIED
    "edge_min_neutralize_bucket_hit_steps": _both("temporal_step"), # VERIFIED

    "edge_min_time_neutralize_bucket": _both("ship_bucket"), # VERIFIED
    "edge_min_time_neutralize_ships": _both("ship_count"), # VERIFIED
    "edge_min_time_neutralize_bucket_available": _continuous(), # VERIFIED
    "edge_min_time_neutralize_bucket_hit_steps": _both("temporal_step"), # VERIFIED

    "edge_takeover_margin_with_max_send": _both("signed_ship_margin"), # VERIFIED
    "edge_stable_margin_with_max_send": _both("signed_ship_margin"), # VERIFIED
    "edge_neutralize_margin_with_max_send": _both("signed_ship_margin"), # VERIFIED
    "edge_time_to_hit_with_max_send": _both("temporal_step"), # VERIFIED
    "edge_is_available_with_max_send": _continuous(), # VERIFIED
    "edge_dst_motion_angle_to_src_dst": _continuous(), # VERIFIED
    "edge_velocity_dx": _continuous(), # VERIFIED
    "edge_velocity_dy": _continuous(), # VERIFIED
    "edge_closing_speed": _continuous(), # VERIFIED

    "edge_min_stable_takeover_bucket_roi": _continuous(), # ?
    "edge_max_send_stable_roi": _continuous(), # ?
    "edge_source_stable_hold_margin_after_min_takeover": _continuous(), # ?
    "edge_source_stable_hold_margin_after_min_stable_takeover": _continuous(), # ?
    "edge_capture_deadline_slack": _continuous(), # ?
    "edge_arrival_tactical_pressure": _continuous(), # ?
    "edge_snipe_score_at_min_takeover_time": _continuous(), # ?
    "edge_overkill_with_min_stable_bucket": _continuous(), # ?
    "edge_stable_capture_vs_current_owner_value": _continuous(), # ?
    "edge_dst_final_owner_is_src_owner_without_action": _continuous(), # ?
    "edge_attack_redundancy_score": _continuous(), # ?
}

_ORBIT_EDGE_PLAYER_BLOCK_FEATURE_INPUTS: dict[str, OrbitObsFeatureInputSpec] = {
    "edge_player_src_owned": _continuous(), # VERIFIED
    "edge_player_dst_owned": _continuous(), # VERIFIED
}

ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "arrival_ships",
    "takeover_cost",
    "resolution_owner",
    "resolution_ships",
    "time_step",
    "stable_takeover_cost",
    "hold_cost",
    "hold_valid",
    "neutralization_cost",
    "neutralization_valid",
    "deny_stable_enemy_cost",
    "battle_tie_distance",
    "battle_tie_valid",
    "production_swing_per_ship",
    "arrival_leverage",
)

_ORBIT_ARRIVAL_TEMPORAL_FEATURE_INPUTS: dict[str, OrbitObsFeatureInputSpec] = {
    "arrival_ships": _continuous(), # VERIFIED
    "takeover_cost": _both("ship_cost"), # VERIFIED
    "resolution_owner": _continuous(), # VERIFIED
    "resolution_ships": _continuous(), # VERIFIED
    "time_step": _continuous(), # VERIFIED
    "stable_takeover_cost": _both("ship_cost"), # VERIFIED
    "hold_cost": _continuous(), # VERIFIED
    "hold_valid": _continuous(), # VERIFIED
    "neutralization_cost": _continuous(), # VERIFIED
    "neutralization_valid": _continuous(), # VERIFIED
    "deny_stable_enemy_cost": _continuous(), # VERIFIED
    "battle_tie_distance": _continuous(), # VERIFIED
    "battle_tie_valid": _continuous(), # VERIFIED

    "production_swing_per_ship": _continuous(), # ?
    "arrival_leverage": _continuous(), # ?
}


def _layout_names(key: str) -> tuple[str, ...]:
    raw = ORBIT_IMPALA_OBS_FEATURE_LAYOUT[key]
    assert isinstance(raw, (list, tuple)), (key, type(raw))
    return tuple(raw)


def _channel_index_in_layout(key: str, logical_name: str) -> int:
    names = _layout_names(key)
    idx = names.index(logical_name)
    assert names[idx] == logical_name, (key, logical_name)
    return idx


def _embedding_specs_from_layout(
    layout_key: str,
    specs: dict[str, OrbitObsFeatureInputSpec],
) -> tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...]:
    out: list[OrbitObsDiscreteEmbeddingChannelSpec] = []
    for logical_name in _layout_names(layout_key):
        spec = specs[logical_name]
        if spec.embedding_table is None:
            continue
        out.append(
            OrbitObsDiscreteEmbeddingChannelSpec(
                channel_index=_channel_index_in_layout(layout_key, logical_name),
                table=spec.embedding_table,
                label=logical_name,
            )
        )
    return tuple(out)


def orbit_discrete_embedding_planet_base_specs() -> tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...]:
    return _embedding_specs_from_layout("planet_base_feature_names", _ORBIT_PLANET_BASE_FEATURE_INPUTS)


def orbit_discrete_embedding_planet_player_block_specs() -> tuple[
    OrbitObsDiscreteEmbeddingChannelSpec, ...
]:
    return _embedding_specs_from_layout(
        "planet_player_block_feature_names",
        _ORBIT_PLANET_PLAYER_BLOCK_FEATURE_INPUTS,
    )


def orbit_discrete_embedding_edge_base_specs() -> tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...]:
    return _embedding_specs_from_layout("edge_base_feature_names", _ORBIT_EDGE_BASE_FEATURE_INPUTS)


def orbit_discrete_embedding_edge_player_block_specs() -> tuple[
    OrbitObsDiscreteEmbeddingChannelSpec, ...
]:
    return _embedding_specs_from_layout(
        "edge_player_block_feature_names",
        _ORBIT_EDGE_PLAYER_BLOCK_FEATURE_INPUTS,
    )


def orbit_discrete_embedding_arrival_temporal_specs() -> tuple[
    OrbitObsDiscreteEmbeddingChannelSpec, ...
]:
    out: list[OrbitObsDiscreteEmbeddingChannelSpec] = []
    for channel_index, logical_name in enumerate(ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES):
        spec = _ORBIT_ARRIVAL_TEMPORAL_FEATURE_INPUTS[logical_name]
        if spec.embedding_table is None:
            continue
        out.append(
            OrbitObsDiscreteEmbeddingChannelSpec(
                channel_index=channel_index,
                table=spec.embedding_table,
                label=logical_name,
            )
        )
    return tuple(out)


def orbit_discrete_embedding_count(specs: tuple[OrbitObsDiscreteEmbeddingChannelSpec, ...]) -> int:
    return len(specs)


def _assert_contract_matches_layout() -> None:
    layout = ORBIT_IMPALA_OBS_FEATURE_LAYOUT
    assert set(_ORBIT_PLANET_BASE_FEATURE_INPUTS) == set(layout["planet_base_feature_names"]), (
        set(_ORBIT_PLANET_BASE_FEATURE_INPUTS),
        set(layout["planet_base_feature_names"]),
    )
    assert set(_ORBIT_PLANET_PLAYER_BLOCK_FEATURE_INPUTS) == set(
        layout["planet_player_block_feature_names"]
    ), (
        set(_ORBIT_PLANET_PLAYER_BLOCK_FEATURE_INPUTS),
        set(layout["planet_player_block_feature_names"]),
    )
    assert set(_ORBIT_EDGE_BASE_FEATURE_INPUTS) == set(layout["edge_base_feature_names"]), (
        set(_ORBIT_EDGE_BASE_FEATURE_INPUTS),
        set(layout["edge_base_feature_names"]),
    )
    assert set(_ORBIT_EDGE_PLAYER_BLOCK_FEATURE_INPUTS) == set(
        layout["edge_player_block_feature_names"]
    ), (
        set(_ORBIT_EDGE_PLAYER_BLOCK_FEATURE_INPUTS),
        set(layout["edge_player_block_feature_names"]),
    )
    assert set(ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES) == set(_ORBIT_ARRIVAL_TEMPORAL_FEATURE_INPUTS), (
        ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES,
        sorted(_ORBIT_ARRIVAL_TEMPORAL_FEATURE_INPUTS),
    )
    assert len(ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES) == int(ORBIT_PLANET_TEMPORAL_FEATURES), (
        len(ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES),
        ORBIT_PLANET_TEMPORAL_FEATURES,
    )
    assert len(_layout_names("planet_base_feature_names")) == int(ORBIT_PLANET_BASE_FEATURES), (
        len(_layout_names("planet_base_feature_names")),
        ORBIT_PLANET_BASE_FEATURES,
    )
    assert len(_layout_names("planet_player_block_feature_names")) == int(
        ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
    ), (
        len(_layout_names("planet_player_block_feature_names")),
        ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    )
    assert len(_layout_names("edge_base_feature_names")) == int(ORBIT_EDGE_BASE_FEATURES), (
        len(_layout_names("edge_base_feature_names")),
        ORBIT_EDGE_BASE_FEATURES,
    )
    assert len(_layout_names("edge_player_block_feature_names")) == int(
        ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
    ), (
        len(_layout_names("edge_player_block_feature_names")),
        ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    )


_assert_contract_matches_layout()


def orbit_obs_planet_feature_importance_inputs(logical_name: str) -> tuple[str, ...]:
    assert logical_name in _ORBIT_PLANET_BASE_FEATURE_INPUTS, logical_name
    return _ORBIT_PLANET_BASE_FEATURE_INPUTS[logical_name].importance_inputs()


def orbit_obs_planet_player_feature_importance_inputs(logical_name: str) -> tuple[str, ...]:
    assert logical_name in _ORBIT_PLANET_PLAYER_BLOCK_FEATURE_INPUTS, logical_name
    return _ORBIT_PLANET_PLAYER_BLOCK_FEATURE_INPUTS[logical_name].importance_inputs()


def orbit_obs_edge_feature_importance_inputs(logical_name: str) -> tuple[str, ...]:
    assert logical_name in _ORBIT_EDGE_BASE_FEATURE_INPUTS, logical_name
    return _ORBIT_EDGE_BASE_FEATURE_INPUTS[logical_name].importance_inputs()


def orbit_obs_edge_player_feature_importance_inputs(logical_name: str) -> tuple[str, ...]:
    assert logical_name in _ORBIT_EDGE_PLAYER_BLOCK_FEATURE_INPUTS, logical_name
    return _ORBIT_EDGE_PLAYER_BLOCK_FEATURE_INPUTS[logical_name].importance_inputs()


def orbit_obs_arrival_temporal_feature_importance_inputs(logical_name: str) -> tuple[str, ...]:
    assert logical_name in _ORBIT_ARRIVAL_TEMPORAL_FEATURE_INPUTS, logical_name
    return _ORBIT_ARRIVAL_TEMPORAL_FEATURE_INPUTS[logical_name].importance_inputs()
