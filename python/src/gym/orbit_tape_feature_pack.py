from __future__ import annotations

from typing import Any

import torch

from ..configs.impala_orbit_model_hyperparams import ORBIT_IMPALA_OBS_FEATURE_LAYOUT
from .obs_wrapper import (
    ORBIT_EDGE_BASE_FEATURES,
    ORBIT_EDGE_FEATURES,
    ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_HIT_CLASSES_PER_TARGET,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX,
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
    ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_place_compact_agent_axis,
    orbit_policy_slot_for_compact_agent,
    orbit_self_enemy_mask,
)

_ORBIT_TAPE_TEMPORAL_POLICY_FEATURE_NAMES: tuple[str, ...] = (
    "temporal_arrival_ships",
    "temporal_takeover_cost",
    "temporal_resolution_owner",
    "temporal_resolution_ships",
    "temporal_time_step",
    "temporal_stable_takeover_cost",
    "temporal_hold_cost",
    "temporal_hold_valid",
    "temporal_neutralization_cost",
    "temporal_neutralization_valid",
    "temporal_deny_stable_enemy_cost",
    "temporal_battle_tie_distance",
    "temporal_battle_tie_valid",
    "temporal_production_swing_per_ship",
    "temporal_arrival_leverage",
)


def _feature_entry(
    *,
    name: str,
    kind: str,
    values: Any,
    unit: str = "value",
) -> dict[str, Any]:
    return {
        "name": str(name),
        "kind": str(kind),
        "dtype": "float32",
        "unit": str(unit),
        "values": values,
    }


_INTERCEPT_FAIL_REASON_UNIT = (
    "intercept_fail_reason_digit_by_sn "
    "0=ok 1=static_zero_norm 2=static_bad_turns 3=dynamic_seed_invalid "
    "4=dynamic_solver_no_converge 5=dynamic_nonfinite_aim 6=dynamic_zero_norm "
    "7=dynamic_bad_turns"
)

_HIT_TYPE_UNIT = (
    "hit_kind_digit_by_sn "
    "0=none 1=target 2=static 3=dynamic 4=sun 5=out_of_board 6=timeout "
    "7=end_of_game 8=interception_failed 9=verified_timeout"
)


def orbit_policy_obs_action_edges_from_plain_and_policy_obs(
    *,
    plain: dict[str, Any],
    policy_obs: dict[str, torch.Tensor],
    num_agents: int,
) -> list[dict[str, Any]]:
    na = int(num_agents)
    assert na in (2, 4), na
    planets = plain["planets"]
    assert isinstance(planets, list)
    assert len(planets) <= ORBIT_MAX_PLANETS, (len(planets), ORBIT_MAX_PLANETS)
    available_action_mask = orbit_place_compact_agent_axis(
        policy_obs["available_action_mask"],
        num_agents=na,
    ).detach().cpu()
    assert tuple(available_action_mask.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_MAX_PLANETS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), available_action_mask.shape
    assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
    action_edges: list[dict[str, Any]] = []
    for src_slot, src_row in enumerate(planets):
        assert isinstance(src_row, (list, tuple)) and len(src_row) == 7
        owner = int(src_row[1])
        if not (0 <= owner < na):
            continue
        policy_slot = orbit_policy_slot_for_compact_agent(owner, na)
        source_id = int(src_row[0])
        x0 = float(src_row[2])
        y0 = float(src_row[3])
        for dst_slot, dst_row in enumerate(planets):
            assert isinstance(dst_row, (list, tuple)) and len(dst_row) == 7
            if dst_slot == src_slot:
                continue
            action_cls = (
                int(dst_slot) * int(ORBIT_MOVE_CLASSES_PER_TARGET)
                + int(ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX)
            )
            action_edges.append(
                {
                    "source_id": source_id,
                    "target_id": int(dst_row[0]),
                    "x0": x0,
                    "y0": y0,
                    "x1": float(dst_row[2]),
                    "y1": float(dst_row[3]),
                    "available": bool(
                        int(available_action_mask[policy_slot, src_slot, action_cls].item()) > 0
                    ),
                }
            )
    return action_edges


def _max_send_debug_scalar_matrices_from_tensors(
    *,
    plain: dict[str, Any],
    policy_obs: dict[str, torch.Tensor],
    num_agents: int,
    hit_kind: torch.Tensor,
    intercept_fail_reason: torch.Tensor,
) -> tuple[list[list[float]], list[list[float]]]:
    na = int(num_agents)
    assert na in (2, 4), na
    n = int(ORBIT_MAX_PLANETS)
    send_all_sn = int(ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX)
    hit_kind = hit_kind.detach().cpu()
    intercept_fail_reason = intercept_fail_reason.detach().cpu()
    assert tuple(hit_kind.shape) == (ORBIT_MAX_PLANETS, ORBIT_PER_PLANET_HIT_CLASSES)
    assert intercept_fail_reason.shape == hit_kind.shape, (
        intercept_fail_reason.shape,
        hit_kind.shape,
    )
    pwm = orbit_place_compact_agent_axis(
        policy_obs["orbit_planet_pairwise_mask"],
        num_agents=na,
    )[0].detach().cpu()
    planets = plain["planets"]
    assert isinstance(planets, list)
    hit_values: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    fail_values: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for src in range(n):
        if src >= len(planets):
            continue
        src_row = planets[src]
        assert isinstance(src_row, (list, tuple)) and len(src_row) == 7
        if int(src_row[1]) < 0:
            continue
        for dst in range(n):
            if src == dst:
                continue
            eidx = src * n + dst
            if float(pwm[eidx].item()) <= 0.5:
                continue
            cls = int(dst) * int(ORBIT_HIT_CLASSES_PER_TARGET) + send_all_sn
            kind = int(hit_kind[src, cls].item())
            reason = int(intercept_fail_reason[src, cls].item())
            assert 0 <= kind <= 9, (src, dst, kind)
            assert 0 <= reason <= 9, (src, dst, reason)
            hit_values[src][dst] = float(kind)
            fail_values[src][dst] = float(reason)
    return hit_values, fail_values


def orbit_policy_obs_feature_pack_from_plain_and_policy_obs(
    *,
    plain: dict[str, Any],
    policy_obs: dict[str, torch.Tensor],
    num_agents: int,
    hit_kind: torch.Tensor | None = None,
    intercept_fail_reason: torch.Tensor | None = None,
) -> dict[str, Any]:
    na = int(num_agents)
    assert na in (2, 4), na
    planets = plain["planets"]
    assert isinstance(planets, list)
    planet_ids = [int(row[0]) for row in planets]
    p = int(ORBIT_PLAYER_AXIS_SLOTS)
    n = int(ORBIT_MAX_PLANETS)
    hz = int(ORBIT_PLANET_ARRIVAL_HORIZON)
    planet_f = orbit_place_compact_agent_axis(
        policy_obs["orbit_planet_features"],
        num_agents=na,
    ).detach().cpu().to(dtype=torch.float32)
    arrival_f = orbit_place_compact_agent_axis(
        policy_obs["orbit_planet_arrival_features"],
        num_agents=na,
    ).detach().cpu().to(dtype=torch.float32)
    enemy_mask = orbit_self_enemy_mask(na).detach().cpu()
    edge_f = orbit_place_compact_agent_axis(
        policy_obs["orbit_planet_pairwise_features"],
        num_agents=na,
    ).detach().cpu().to(dtype=torch.float32)
    assert planet_f.shape == (p, n, ORBIT_PLANET_FEATURES), planet_f.shape
    assert arrival_f.shape[0] == p and arrival_f.shape[1] == n, arrival_f.shape
    assert arrival_f.shape[2] == hz and arrival_f.shape[3] == p, arrival_f.shape
    assert arrival_f.shape[4] == len(_ORBIT_TAPE_TEMPORAL_POLICY_FEATURE_NAMES), arrival_f.shape
    assert edge_f.shape == (p, n * n, ORBIT_EDGE_FEATURES), edge_f.shape
    assert enemy_mask.shape == (p, p - 1), enemy_mask.shape
    assert bool(torch.isfinite(planet_f).all().item())
    assert bool(torch.isfinite(arrival_f).all().item())
    assert bool(torch.isfinite(edge_f).all().item())

    layout = ORBIT_IMPALA_OBS_FEATURE_LAYOUT
    planet_base_names = tuple(str(x) for x in layout["planet_base_feature_names"])
    planet_player_names = tuple(str(x) for x in layout["planet_player_block_feature_names"])
    edge_base_names = tuple(str(x) for x in layout["edge_base_feature_names"])
    edge_player_names = tuple(str(x) for x in layout["edge_player_block_feature_names"])
    assert len(planet_player_names) == ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
    assert len(edge_base_names) == ORBIT_EDGE_BASE_FEATURES
    assert len(edge_player_names) == ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER

    def owner_policy_slot(owner: int) -> int:
        o = int(owner)
        if o >= na:
            return -1
        return orbit_policy_slot_for_compact_agent(o, na)

    features: list[dict[str, Any]] = []
    for c, name in enumerate(planet_base_names):
        values = []
        for slot in range(n):
            v = float(planet_f[0, slot, c].item())
            values.append([v for _owner in range(p)])
        features.append(_feature_entry(name=name, kind="player_scalar", values=values))
    for c, name in enumerate(_ORBIT_TAPE_TEMPORAL_POLICY_FEATURE_NAMES):
        values = []
        for slot in range(n):
            owner_values = []
            for owner in range(p):
                ps = owner_policy_slot(owner)
                if ps < 0:
                    owner_values.append([0.0 for _t in range(hz)])
                else:
                    owner_values.append(
                        [float(x) for x in arrival_f[ps, slot, :, 0, c].tolist()]
                    )
            values.append(owner_values)
        features.append(_feature_entry(name=name, kind="player_temporal", values=values))
    for c, name in enumerate(planet_player_names):
        channel = ORBIT_PLANET_PLAYER_FEATURE_OFFSET + c
        values = []
        for slot in range(n):
            owner_values = []
            for owner in range(p):
                ps = owner_policy_slot(owner)
                owner_values.append(0.0 if ps < 0 else float(planet_f[ps, slot, channel].item()))
            values.append(owner_values)
        features.append(_feature_entry(name=name, kind="player_scalar", values=values))

    edge_features: list[dict[str, Any]] = []
    for c, name in enumerate(edge_base_names):
        values = []
        for src in range(n):
            row = []
            for dst in range(n):
                row.append(float(edge_f[0, src * n + dst, c].item()))
            values.append(row)
        edge_features.append(_feature_entry(name=name, kind="edge_scalar", values=values))
    for c, name in enumerate(edge_player_names):
        channel = ORBIT_EDGE_PLAYER_FEATURE_OFFSET + c
        values = []
        for src in range(n):
            dst_values = []
            for dst in range(n):
                owner_values = []
                eidx = src * n + dst
                for owner in range(p):
                    ps = owner_policy_slot(owner)
                    owner_values.append(0.0 if ps < 0 else float(edge_f[ps, eidx, channel].item()))
                dst_values.append(owner_values)
            values.append(dst_values)
        edge_features.append(
            _feature_entry(name=name, kind="edge_player_scalar", values=values)
        )
    if hit_kind is not None or intercept_fail_reason is not None:
        assert hit_kind is not None
        assert intercept_fail_reason is not None
        max_send_hit_type, max_send_fail_reason = _max_send_debug_scalar_matrices_from_tensors(
            plain=plain,
            policy_obs=policy_obs,
            num_agents=na,
            hit_kind=hit_kind,
            intercept_fail_reason=intercept_fail_reason,
        )
        edge_features.append(
            _feature_entry(
                name="max_send_hit_type",
                kind="edge_scalar",
                values=max_send_hit_type,
                unit=_HIT_TYPE_UNIT,
            )
        )
        edge_features.append(
            _feature_entry(
                name="max_send_interception_fail_reason",
                kind="edge_scalar",
                values=max_send_fail_reason,
                unit=_INTERCEPT_FAIL_REASON_UNIT,
            )
        )

    return {
        "version": 4,
        "horizon": hz,
        "player_axis_slots": p,
        "num_agents": na,
        "planet_ids": planet_ids,
        "features": features,
        "edge_features": edge_features,
    }
