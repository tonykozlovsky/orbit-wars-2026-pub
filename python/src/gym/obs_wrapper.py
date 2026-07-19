from __future__ import annotations

from typing import Any

import torch

from .dict_io_contract import (
    dict_io_contract_validation_enabled,
    validated_dict_io_contract_output,
)
from .orbit_wars_cpp_ext import orbit_wars_cpp
from .wall_tree_profiler import WallTreeProfiler, profiler_span

# Planet rows: ``[id, owner, x, y, radius, ships, production]`` (``orbit_wars.json``); max planets per seat.
ORBIT_MAX_PLANETS = 44
ORBIT_PLANET_ROW_LEN = 7
ORBIT_PLANET_PAIRWISE_COUNT = ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS
ORBIT_PLANET_ACTION_SLOTS = ORBIT_MAX_PLANETS
# Per-target class ``c = j * ORBIT_MOVE_CLASSES_PER_TARGET + sn``.
ORBIT_MOVE_CLASS_NOOP_SUBINDEX = 0
ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX = 1
ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX = 2
ORBIT_MOVE_CLASS_SEND_TAKEOVER_SUBINDEX = 3
ORBIT_MOVE_CLASS_SEND_STABLE_TAKEOVER_SUBINDEX = 4
ORBIT_MOVE_CLASSES_PER_TARGET = 5
ORBIT_MOVE_SEND_SUBINDICES: tuple[int, ...] = (
    ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_TAKEOVER_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_STABLE_TAKEOVER_SUBINDEX,
)
ORBIT_MOVE_CLASS_FREQ_NAMES: tuple[str, ...] = (
    "noop",
    "send_all",
    "send_half",
    "send_takeover",
    "send_stable_takeover",
)
ORBIT_PER_PLANET_MOVE_CLASSES = ORBIT_MAX_PLANETS * ORBIT_MOVE_CLASSES_PER_TARGET
ORBIT_HIT_CLASSES_PER_TARGET = 102
ORBIT_PER_PLANET_HIT_CLASSES = ORBIT_MAX_PLANETS * ORBIT_HIT_CLASSES_PER_TARGET
ORBIT_PLAYER_AXIS_SLOTS = 4
ORBIT_ENEMY_AXIS_SLOTS = ORBIT_PLAYER_AXIS_SLOTS - 1
assert ORBIT_MOVE_CLASSES_PER_TARGET == 5
assert len(ORBIT_MOVE_CLASS_FREQ_NAMES) == ORBIT_MOVE_CLASSES_PER_TARGET


def orbit_move_ship_count(source_ships: int, move_subindex: int) -> int:
    ships = int(source_ships)
    sn = int(move_subindex)
    assert ships > 0, ships
    if sn == ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX:
        return ships
    if sn == ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX:
        return (ships + 1) // 2
    assert False, sn

ORBIT_PLANET_ARRIVAL_HORIZON = 40 # KEK

ORBIT_PLANET_TEMPORAL_FEATURES = 15
ORBIT_PLANET_ARRIVAL_FEATURES = ORBIT_PLANET_TEMPORAL_FEATURES
ORBIT_PLANET_EPISODE_STEP_FEATURE_DIVISOR = 500
ORBIT_PLANET_BASE_FEATURES = 13
ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER = 9
ORBIT_PLANET_FEATURES = (
    ORBIT_PLANET_BASE_FEATURES + ORBIT_PLAYER_AXIS_SLOTS * ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
)
ORBIT_EDGE_BASE_FEATURES = 47
ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER = 2
ORBIT_EDGE_FEATURES = (
    ORBIT_EDGE_BASE_FEATURES + ORBIT_PLAYER_AXIS_SLOTS * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
)
ORBIT_PLANET_BASE_FEATURE_X = 0
ORBIT_PLANET_BASE_FEATURE_Y = 1
ORBIT_PLANET_BASE_FEATURE_NEUTRAL_SHIPS = 2
ORBIT_PLANET_BASE_FEATURE_EPISODE_STEP = 3
ORBIT_PLANET_BASE_FEATURE_IS_STATIC = 4
ORBIT_PLANET_BASE_FEATURE_IS_DYNAMIC = 5
ORBIT_PLANET_BASE_FEATURE_IS_COMET = 6
ORBIT_PLANET_BASE_FEATURE_COMET_TIME_BEFORE_DESPAWN = 7
ORBIT_PLANET_BASE_FEATURE_RADIUS = 8
ORBIT_PLANET_BASE_FEATURE_PLANET_PRODUCTION = 9
ORBIT_PLANET_BASE_FEATURE_ORBIT_RADIUS = 10
ORBIT_PLANET_BASE_FEATURE_ANGULAR_VELOCITY = 11
ORBIT_PLANET_BASE_FEATURE_SUN_ANGLE = 12
ORBIT_PLANET_PLAYER_FEATURE_OFFSET = ORBIT_PLANET_BASE_FEATURES
ORBIT_PLANET_PLAYER_FLEET_NORMALIZER = 1000.0
ORBIT_PLANET_PLAYER_FEATURE_SHIPS = 0
ORBIT_PLANET_PLAYER_FEATURE_TOTAL_FLEET_FRAC = 1
ORBIT_PLANET_PLAYER_FEATURE_PRODUCTION = 2
ORBIT_PLANET_PLAYER_FEATURE_OWNER_SURVIVAL_MARGIN = 3
ORBIT_PLANET_PLAYER_FEATURE_FLIP_TIME = 4
ORBIT_PLANET_PLAYER_FEATURE_STABLE_FLIP_TIME = 5
ORBIT_PLANET_PLAYER_FEATURE_OWNER_CHURN = 6
ORBIT_PLANET_PLAYER_FEATURE_LAST_DECISIVE_BATTLE_STEP = 7
ORBIT_PLANET_PLAYER_FEATURE_POST_HORIZON_OWNER_MARGIN = 8
ORBIT_EDGE_BASE_FEATURE_DISTANCE = 0
ORBIT_EDGE_BASE_FEATURE_SRC_NEUTRAL = 1
ORBIT_EDGE_BASE_FEATURE_DST_NEUTRAL = 2
ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET = 3
ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_SHIPS = 4
ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET_AVAILABLE = 5
ORBIT_EDGE_BASE_FEATURE_MIN_TAKEOVER_BUCKET_HIT_STEPS = 6
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET = 7
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_SHIPS = 8
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET_AVAILABLE = 9
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_TAKEOVER_BUCKET_HIT_STEPS = 10
ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET = 11
ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_SHIPS = 12
ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_AVAILABLE = 13
ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_HIT_STEPS = 14
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET = 15
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_SHIPS = 16
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET_AVAILABLE = 17
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_STABLE_TAKEOVER_BUCKET_HIT_STEPS = 18
ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET = 19
ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_SHIPS = 20
ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET_AVAILABLE = 21
ORBIT_EDGE_BASE_FEATURE_MIN_NEUTRALIZE_BUCKET_HIT_STEPS = 22
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET = 23
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_SHIPS = 24
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET_AVAILABLE = 25
ORBIT_EDGE_BASE_FEATURE_MIN_TIME_NEUTRALIZE_BUCKET_HIT_STEPS = 26
ORBIT_EDGE_BASE_FEATURE_TAKEOVER_MARGIN_WITH_MAX_SEND = 27
ORBIT_EDGE_BASE_FEATURE_STABLE_MARGIN_WITH_MAX_SEND = 28
ORBIT_EDGE_BASE_FEATURE_NEUTRALIZE_MARGIN_WITH_MAX_SEND = 29
ORBIT_EDGE_BASE_FEATURE_TIME_TO_HIT_WITH_MAX_SEND = 30
ORBIT_EDGE_BASE_FEATURE_IS_AVAILABLE_WITH_MAX_SEND = 31
ORBIT_EDGE_BASE_FEATURE_DST_MOTION_ANGLE_TO_SRC_DST = 32
ORBIT_EDGE_BASE_FEATURE_VELOCITY_DX = 33
ORBIT_EDGE_BASE_FEATURE_VELOCITY_DY = 34
ORBIT_EDGE_BASE_FEATURE_CLOSING_SPEED = 35
ORBIT_EDGE_BASE_FEATURE_MIN_STABLE_TAKEOVER_BUCKET_ROI = 36
ORBIT_EDGE_BASE_FEATURE_MAX_SEND_STABLE_ROI = 37
ORBIT_EDGE_BASE_FEATURE_SOURCE_STABLE_HOLD_MARGIN_AFTER_MIN_TAKEOVER = 38
ORBIT_EDGE_BASE_FEATURE_SOURCE_STABLE_HOLD_MARGIN_AFTER_MIN_STABLE_TAKEOVER = 39
ORBIT_EDGE_BASE_FEATURE_CAPTURE_DEADLINE_SLACK = 40
ORBIT_EDGE_BASE_FEATURE_ARRIVAL_TACTICAL_PRESSURE = 41
ORBIT_EDGE_BASE_FEATURE_SNIPE_SCORE_AT_MIN_TAKEOVER_TIME = 42
ORBIT_EDGE_BASE_FEATURE_OVERKILL_WITH_MIN_STABLE_BUCKET = 43
ORBIT_EDGE_BASE_FEATURE_STABLE_CAPTURE_VS_CURRENT_OWNER_VALUE = 44
ORBIT_EDGE_BASE_FEATURE_DST_FINAL_OWNER_IS_SRC_OWNER_WITHOUT_ACTION = 45
ORBIT_EDGE_BASE_FEATURE_ATTACK_REDUNDANCY_SCORE = 46
ORBIT_EDGE_PLAYER_FEATURE_OFFSET = ORBIT_EDGE_BASE_FEATURES
ORBIT_EDGE_PLAYER_FEATURE_SRC_OWNED = 0
ORBIT_EDGE_PLAYER_FEATURE_DST_OWNED = 1
ORBIT_POLICY_SLOT_BY_COMPACT_AGENT_2P: tuple[int, int] = (0, 3)
ORBIT_SELF_ENEMY_PLAYER_ORDER_4P: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 3, 2),
    (1, 3, 2, 0),
    (2, 0, 1, 3),
    (3, 2, 0, 1),
)
ORBIT_SELF_ENEMY_PLAYER_ORDER_2P_BY_POLICY_SLOT: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, -1, -1),
    (-1, -1, -1, -1),
    (-1, -1, -1, -1),
    (1, 0, -1, -1),
)
assert ORBIT_POLICY_SLOT_BY_COMPACT_AGENT_2P == (0, 3)


def orbit_policy_slot_for_compact_agent(agent_idx: int, num_agents: int) -> int:
    a = int(agent_idx)
    na = int(num_agents)
    assert na in (2, 4), na
    assert 0 <= a < na, (a, na)
    if na == 4:
        return a
    return int(ORBIT_POLICY_SLOT_BY_COMPACT_AGENT_2P[a])


def orbit_active_policy_slots(num_agents: int) -> tuple[int, ...]:
    na = int(num_agents)
    assert na in (2, 4), na
    if na == 4:
        return (0, 1, 2, 3)
    return ORBIT_POLICY_SLOT_BY_COMPACT_AGENT_2P


def orbit_place_compact_agent_axis(t: torch.Tensor, *, num_agents: int) -> torch.Tensor:
    na = int(num_agents)
    p = int(ORBIT_PLAYER_AXIS_SLOTS)
    assert na in (2, 4), na
    assert isinstance(t, torch.Tensor)
    assert t.ndim >= 1, (t.ndim, tuple(t.shape))
    assert int(t.shape[0]) in (na, p), (tuple(t.shape), na, p)
    if int(t.shape[0]) == p:
        return t
    out = torch.zeros((p,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    for compact_idx in range(na):
        out[orbit_policy_slot_for_compact_agent(compact_idx, na)].copy_(t[compact_idx])
    return out


def _orbit_unravel_flat_index(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = int(flat_idx)
    rev: list[int] = []
    for dim in reversed(shape):
        d = int(dim)
        rev.append(idx % d)
        idx //= d
    return tuple(reversed(rev))


def _orbit_planet_feature_channel_name(ch: int) -> str:
    c = int(ch)
    assert 0 <= c < ORBIT_PLANET_FEATURES, (c, ORBIT_PLANET_FEATURES)
    if c < ORBIT_PLANET_BASE_FEATURES:
        base_names = (
            "x",
            "y",
            "neutral_ships",
            "episode_step",
            "is_static",
            "is_dynamic",
            "is_comet",
            "comet_time_before_despawn",
            "radius",
            "planet_production",
            "orbit_radius",
            "angular_velocity",
            "sun_angle",
        )
        return f"base.{base_names[c]}"
    off = c - ORBIT_PLANET_PLAYER_FEATURE_OFFSET
    player_block = off // ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
    sub = off % ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
    sub_names = (
        "ships_if_owned",
        "total_fleet_frac",
        "production_if_owned",
        "owner_survival_margin",
        "flip_time_by_player",
        "stable_flip_time_by_player",
        "owner_churn",
        "last_decisive_battle_step",
        "post_horizon_owner_margin",
    )
    assert 0 <= player_block < ORBIT_PLAYER_AXIS_SLOTS, (c, player_block)
    assert 0 <= sub < ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER, (c, sub)
    slot_names = ("self", "enemy0", "enemy1", "enemy2")
    return f"{slot_names[player_block]}.{sub_names[sub]}"


def _orbit_edge_feature_channel_name(ch: int) -> str:
    c = int(ch)
    assert 0 <= c < ORBIT_EDGE_FEATURES, (c, ORBIT_EDGE_FEATURES)
    if c < ORBIT_EDGE_BASE_FEATURES:
        base_names = (
            "distance",
            "src_neutral",
            "dst_neutral",
            "min_takeover_bucket",
            "min_takeover_ships",
            "min_takeover_bucket_available",
            "min_takeover_bucket_hit_steps",
            "min_time_takeover_bucket",
            "min_time_takeover_ships",
            "min_time_takeover_bucket_available",
            "min_time_takeover_bucket_hit_steps",
            "min_stable_takeover_bucket",
            "min_stable_takeover_ships",
            "min_stable_takeover_bucket_available",
            "min_stable_takeover_bucket_hit_steps",
            "min_time_stable_takeover_bucket",
            "min_time_stable_takeover_ships",
            "min_time_stable_takeover_bucket_available",
            "min_time_stable_takeover_bucket_hit_steps",
            "min_neutralize_bucket",
            "min_neutralize_ships",
            "min_neutralize_bucket_available",
            "min_neutralize_bucket_hit_steps",
            "min_time_neutralize_bucket",
            "min_time_neutralize_ships",
            "min_time_neutralize_bucket_available",
            "min_time_neutralize_bucket_hit_steps",
            "takeover_margin_with_max_send",
            "stable_margin_with_max_send",
            "neutralize_margin_with_max_send",
            "time_to_hit_with_max_send",
            "is_available_with_max_send",
            "dst_motion_angle_to_src_dst",
            "velocity_dx",
            "velocity_dy",
            "closing_speed",
            "min_stable_takeover_bucket_roi",
            "max_send_stable_roi",
            "source_stable_hold_margin_after_min_takeover",
            "source_stable_hold_margin_after_min_stable_takeover",
            "capture_deadline_slack",
            "arrival_tactical_pressure",
            "snipe_score_at_min_takeover_time",
            "overkill_with_min_stable_bucket",
            "stable_capture_vs_current_owner_value",
            "dst_final_owner_is_src_owner_without_action",
            "attack_redundancy_score",
        )
        assert len(base_names) == ORBIT_EDGE_BASE_FEATURES, (
            len(base_names),
            ORBIT_EDGE_BASE_FEATURES,
        )
        return f"base.{base_names[c]}"
    off = c - ORBIT_EDGE_PLAYER_FEATURE_OFFSET
    player_block = off // ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
    sub = off % ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
    sub_names = (
        "src_owned",
        "dst_owned",
    )
    assert len(sub_names) == ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER, (
        len(sub_names),
        ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    )
    assert 0 <= player_block < ORBIT_PLAYER_AXIS_SLOTS, (c, player_block)
    assert 0 <= sub < ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER, (c, sub)
    slot_names = ("self", "enemy0", "enemy1", "enemy2")
    return f"{slot_names[player_block]}.{sub_names[sub]}"


def _orbit_tensor_equal_detail(cpp_t: torch.Tensor, obs_t: torch.Tensor, *, key: str) -> str:
    d = (cpp_t.detach().cpu().to(dtype=torch.float64) - obs_t.detach().cpu().to(dtype=torch.float64)).abs()
    flat = d.reshape(-1)
    max_v = float(flat.max().item())
    if max_v == 0.0:
        return ""
    idx = int(flat.argmax().item())
    unraveled = _orbit_unravel_flat_index(idx, tuple(int(x) for x in d.shape))
    cpp_v = float(cpp_t.detach().cpu().reshape(-1)[idx].item())
    obs_v = float(obs_t.detach().cpu().reshape(-1)[idx].item())
    parts = [
        f"max_abs_diff={max_v}",
        f"idx_flat={idx}",
        f"idx_unraveled={unraveled}",
        f"cpp={cpp_v}",
        f"py={obs_v}",
    ]
    if key == "orbit_planet_features" and len(unraveled) == 3:
        ai, pi, fi = int(unraveled[0]), int(unraveled[1]), int(unraveled[2])
        parts.append(f"feature_channel={fi}({_orbit_planet_feature_channel_name(fi)})")
        parts.append(f"agent_axis={ai}_planet_slot={pi}")
    if key == "orbit_planet_pairwise_features" and len(unraveled) == 3:
        ai, ei, fi = int(unraveled[0]), int(unraveled[1]), int(unraveled[2])
        parts.append(f"edge_channel={fi}({_orbit_edge_feature_channel_name(fi)})")
        parts.append(f"agent_axis={ai}_edge_flat={ei}")
    return "; ".join(parts)


ORBIT_BOARD_CENTER = 50.0
ORBIT_ROTATION_RADIUS_LIMIT = 50.0

ORBIT_POLICY_OBS_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
    "action_taken_index",
    "player_mask",
)


def orbit_assert_available_action_mask_contract(
    available_action_mask: torch.Tensor,
    *,
    label: str,
) -> None:
    assert isinstance(available_action_mask, torch.Tensor), (label, type(available_action_mask))
    n = int(ORBIT_PLANET_ACTION_SLOTS)
    nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)
    pc = int(ORBIT_PER_PLANET_MOVE_CLASSES)
    assert tuple(available_action_mask.shape[-2:]) == (n, pc), (
        label,
        tuple(available_action_mask.shape),
    )
    assert available_action_mask.dtype == torch.int8, (label, available_action_mask.dtype)
    assert bool(torch.all(available_action_mask >= 0).item()), (
        label,
        "available_action_mask must not contain negative entries",
    )
    available = (available_action_mask > 0)
    by_dst_class = available.reshape(*tuple(available.shape[:-1]), n, nb)
    src_idx = torch.arange(n, device=available.device, dtype=torch.long)
    noop_available = by_dst_class[..., src_idx, src_idx, 0]
    assert bool(torch.all(noop_available).item()), (
        label,
        "Each source row must expose self noop as an available action",
    )
    sn0_available = by_dst_class[..., 0]
    expected_sn0 = torch.eye(n, device=available.device, dtype=torch.bool).expand(
        *tuple(available.shape[:-2]),
        n,
        n,
    )
    assert bool(torch.equal(sn0_available, expected_sn0)), (
        label,
        "Only source-to-self ship_subindex=0 may be available",
    )
    self_amount_available = by_dst_class[..., src_idx, src_idx, 1:].any(dim=-1)
    assert bool(torch.all(~self_amount_available).item()), (
        label,
        "Source-to-self positive ship_subindex actions must be unavailable",
    )
    amount_available = by_dst_class.clone()
    amount_available[..., 0] = False
    dst_available = amount_available.any(dim=-1)
    send_available = dst_available.any(dim=-1)
    send_rows_have_dst = torch.where(
        send_available,
        dst_available.any(dim=-1),
        torch.ones_like(send_available, dtype=torch.bool),
    )
    assert bool(torch.all(send_rows_have_dst).item()), (
        label,
        "Send-available source row must have at least one available destination",
    )
    dst_rows_have_amount = torch.where(
        dst_available,
        amount_available.any(dim=-1),
        torch.ones_like(dst_available, dtype=torch.bool),
    )
    assert bool(torch.all(dst_rows_have_amount).item()), (
        label,
        "Available destination row must have at least one amount action",
    )
    send_rows_have_amount = torch.where(
        send_available,
        amount_available.any(dim=(-1, -2)),
        torch.ones_like(send_available, dtype=torch.bool),
    )
    assert bool(torch.all(send_rows_have_amount).item()), (
        label,
        "Send-available source row must have at least one amount action",
    )


def orbit_self_enemy_mask(num_agents: int, *, device: torch.device | None = None) -> torch.Tensor:
    na = int(num_agents)
    assert na in (2, 4), na
    out = torch.zeros(
        (ORBIT_PLAYER_AXIS_SLOTS, ORBIT_ENEMY_AXIS_SLOTS),
        dtype=torch.float32,
        device=device,
    )
    if na == 2:
        out[0, 0] = 1.0
        out[3, 0] = 1.0
        return out
    out[:, :] = 1.0
    return out


def orbit_assert_policy_obs_2p_self_enemy_blocks(obs: dict[str, torch.Tensor]) -> None:
    active_slots = ORBIT_POLICY_SLOT_BY_COMPACT_AGENT_2P
    inactive_slots = (1, 2)
    player_mask = obs["player_mask"]
    enemy_mask = obs["orbit_enemy_mask"]
    planet_features = obs["orbit_planet_features"]
    arrivals = obs["orbit_planet_arrival_features"]
    edge_features = obs["orbit_planet_pairwise_features"]
    planet_mask = obs["orbit_planet_mask"]
    pairwise_mask = obs["orbit_planet_pairwise_mask"]
    available_action_mask = obs["available_action_mask"]
    action_taken_index = obs["action_taken_index"]
    assert tuple(player_mask.shape) == (ORBIT_PLAYER_AXIS_SLOTS,), player_mask.shape
    assert tuple(enemy_mask.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_ENEMY_AXIS_SLOTS,
    ), enemy_mask.shape
    assert torch.equal(
        enemy_mask,
        orbit_self_enemy_mask(2, device=enemy_mask.device),
    ), enemy_mask
    assert tuple(planet_features.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_MAX_PLANETS,
        ORBIT_PLANET_FEATURES,
    ), planet_features.shape
    assert tuple(arrivals.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_MAX_PLANETS,
        ORBIT_PLANET_ARRIVAL_HORIZON,
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_TEMPORAL_FEATURES,
    ), arrivals.shape
    assert tuple(edge_features.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_PAIRWISE_COUNT,
        ORBIT_EDGE_FEATURES,
    ), edge_features.shape
    assert torch.all(player_mask[list(inactive_slots)] == 0), player_mask
    assert torch.all(planet_mask[list(inactive_slots)] == 0), planet_mask[list(inactive_slots)]
    assert torch.all(pairwise_mask[list(inactive_slots)] == 0), pairwise_mask[list(inactive_slots)]
    expected_inactive_available = torch.zeros_like(available_action_mask[list(inactive_slots)])
    rows = torch.arange(
        ORBIT_PLANET_ACTION_SLOTS,
        device=available_action_mask.device,
        dtype=torch.int64,
    )
    noop_idx = rows * int(ORBIT_MOVE_CLASSES_PER_TARGET)
    expected_inactive_available[:, rows, noop_idx] = True
    assert torch.equal(
        available_action_mask[list(inactive_slots)],
        expected_inactive_available,
    ), available_action_mask[list(inactive_slots)]
    expected_inactive_taken = noop_idx.view(ORBIT_PLANET_ACTION_SLOTS, 1).expand(
        len(inactive_slots),
        ORBIT_PLANET_ACTION_SLOTS,
        1,
    )
    assert torch.equal(
        action_taken_index[list(inactive_slots)],
        expected_inactive_taken,
    ), action_taken_index[list(inactive_slots)]
    for player_block in (2, 3):
        p0 = (
            ORBIT_PLANET_PLAYER_FEATURE_OFFSET
            + player_block * ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
        )
        p1 = p0 + ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
        e0 = (
            ORBIT_EDGE_PLAYER_FEATURE_OFFSET
            + player_block * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
        )
        e1 = e0 + ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
        assert torch.all(planet_features[list(active_slots), :, p0:p1] == 0), (
            "2p inactive enemy block must be empty; if this comes from orbit_wars_cpp, rebuild the C++ extension",
            player_block,
            planet_features[list(active_slots), :, p0:p1],
        )
        assert torch.all(edge_features[list(active_slots), :, e0:e1] == 0), (
            "2p inactive enemy edge block must be empty; if this comes from orbit_wars_cpp, rebuild the C++ extension",
            player_block,
            edge_features[list(active_slots), :, e0:e1],
        )
        assert torch.all(arrivals[list(active_slots), :, :, player_block, :] == 0), (
            "2p inactive enemy arrival block must be empty",
            player_block,
            arrivals[list(active_slots), :, :, player_block, :],
        )

# C++ vs Python path can differ slightly in float32 edge distances and reductions.
ORBIT_PLANET_FEATURES_CPP_OBS_RTOL = 1e-5
ORBIT_PLANET_FEATURES_CPP_OBS_ATOL = 1e-5
ORBIT_PAIRWISE_FEATURES_CPP_OBS_RTOL = 1e-5
ORBIT_PAIRWISE_FEATURES_CPP_OBS_ATOL = 1e-5
ORBIT_ARRIVAL_FEATURES_CPP_OBS_RTOL = 1e-5
ORBIT_ARRIVAL_FEATURES_CPP_OBS_ATOL = 1e-5


def orbit_inner_cpp_env_obs_full(env: Any) -> bool:
    e: Any = env
    for _ in range(32):
        if hasattr(e, "_cpp_env_obs_full"):
            return bool(e._cpp_env_obs_full)
        ne = getattr(e, "env", None)
        if ne is None:
            return False
        e = ne
    return False


def orbit_inner_cpp_env_obs_validate(env: Any) -> bool:
    e: Any = env
    for _ in range(32):
        if hasattr(e, "_cpp_env_obs_validate"):
            return bool(e._cpp_env_obs_validate)
        ne = getattr(e, "env", None)
        if ne is None:
            return False
        e = ne
    return False


def _orbit_unwrap_env_with_cpp_obs_stub(env: Any) -> tuple[Any, Any] | None:
    """If ``env`` wraps ``OrbitWarsEnv`` with ``_cpp_obs_stub``, return ``(orbit_env, cpp_env)``."""
    e: Any = env
    for _ in range(32):
        stub = getattr(e, "_cpp_obs_stub", None)
        if stub is not None:
            cpp_env = getattr(stub, "_cpp_env", None)
            if cpp_env is None:
                return None
            return e, cpp_env
        ne = getattr(e, "env", None)
        if ne is None:
            return None
        e = ne
    return None


ORBIT_SIM_STATE_TRACE_PREFIXES: tuple[str, ...] = (
    "01\t",
    "02\t",
    "10\t",
    "11\t",
    "30\t",
    "31\t",
)


def _orbit_policy_obs_digest_lines_from_cpp_trace_text(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(ORBIT_SIM_STATE_TRACE_PREFIXES)]


def _orbit_state_fields_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "planet":
        return ("id", "owner", "x", "y", "radius", "ships", "production")
    if kind == "fleet":
        return ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")
    return ()


def _orbit_token_is_number(token: str) -> bool:
    if len(token) == 0:
        return False
    has_digit = False
    for ch in token:
        if "0" <= ch <= "9":
            has_digit = True
            continue
        if ch in ("-", "+", ".", "e", "E"):
            continue
        return False
    return has_digit


def _orbit_trace_context(lines: list[str], center: int, radius: int = 2) -> list[str]:
    lo = max(0, int(center) - int(radius))
    hi = min(len(lines), int(center) + int(radius) + 1)
    return lines[lo:hi]


def _orbit_policy_obs_digest_mismatch_detail(cpp_lines: list[str], py_lines: list[str]) -> str:
    out: list[str] = []
    cpp_n = len(cpp_lines)
    py_n = len(py_lines)
    out.append(f"line count: cpp={cpp_n} py={py_n}")
    n = min(cpp_n, py_n)
    mismatch_i = -1
    for i in range(n):
        if cpp_lines[i] != py_lines[i]:
            mismatch_i = i
            break
    if mismatch_i < 0 and cpp_n == py_n:
        out.append(
            "all compared lines match"
        )
        return "\n".join(out)
    if mismatch_i < 0:
        mismatch_i = n
        out.append(
            "all state trace lines in shared prefix match; one side has extra tail lines"
        )
    else:
        out.append(f"first mismatch at state trace line index {mismatch_i} (0-based)")
    cpp_line = cpp_lines[mismatch_i] if mismatch_i < cpp_n else "<cpp eof>"
    py_line = py_lines[mismatch_i] if mismatch_i < py_n else "<py eof>"
    out.append(f"cpp line: {cpp_line}")
    out.append(f"py line: {py_line}")
    tc = cpp_line.split("\t") if mismatch_i < cpp_n else []
    tp = py_line.split("\t") if mismatch_i < py_n else []
    if len(tc) >= 3 and len(tp) >= 3:
        out.append(
            f"block: cpp stage={tc[0]} name={tc[1]} kind={tc[2]} | "
            f"py stage={tp[0]} name={tp[1]} kind={tp[2]}"
        )
        if mismatch_i + 1 < cpp_n and cpp_lines[mismatch_i + 1] == py_line:
            out.append("alignment hint: extra line on cpp side at mismatch index")
        if mismatch_i + 1 < py_n and py_lines[mismatch_i + 1] == cpp_line:
            out.append("alignment hint: extra line on python side at mismatch index")
        if tc[0] != tp[0] or tc[1] != tp[1] or tc[2] != tp[2]:
            out.append("mismatch type: different stage/name/kind")
        else:
            names = _orbit_state_fields_for_kind(tc[2])
            ncols = min(len(tc), len(tp))
            found_field_diff = False
            for j in range(3, ncols):
                if tc[j] == tp[j]:
                    continue
                found_field_diff = True
                field_idx = j - 3
                field_name = names[field_idx] if field_idx < len(names) else f"field_{field_idx}"
                if _orbit_token_is_number(tc[j]) and _orbit_token_is_number(tp[j]):
                    c_num = float(tc[j])
                    p_num = float(tp[j])
                    out.append(
                        f"field {field_name}: cpp={tc[j]} py={tp[j]} abs_diff={abs(c_num - p_num)}"
                    )
                else:
                    out.append(f"field {field_name}: cpp={tc[j]} py={tp[j]}")
                break
            if not found_field_diff and len(tc) != len(tp):
                out.append(f"value column count differs: cpp_cols={len(tc)} py_cols={len(tp)}")
    out.append(f"py_context={_orbit_trace_context(py_lines, mismatch_i)}")
    out.append(f"cpp_context={_orbit_trace_context(cpp_lines, mismatch_i)}")
    return "\n".join(out)


def orbit_assert_cpp_py_policy_obs_trace_matches(env: Any) -> None:
    if not orbit_inner_cpp_env_obs_validate(env):
        return
    pair = _orbit_unwrap_env_with_cpp_obs_stub(env)
    if pair is None:
        return
    orbit_env, _cpp_api = pair
    stub = getattr(orbit_env, "_cpp_obs_stub", None)
    assert stub is not None
    if bool(getattr(orbit_env, "_orbit_cpp_obs_validate_transition_was_reset", False)):
        cpp_text = stub.last_cpp_reset_trace
    else:
        cpp_text = stub.last_cpp_step_trace
    cpp_lines = _orbit_policy_obs_digest_lines_from_cpp_trace_text(cpp_text)
    py_lines = getattr(orbit_env, "_orbit_cpp_obs_validate_last_py_digest_lines", None)
    assert isinstance(py_lines, list), type(py_lines)
    assert len(cpp_lines) > 0, "cpp state trace lines are empty"
    if cpp_lines != py_lines:
        raise AssertionError(
            "orbit sim state trace mismatch cpp vs python:\n"
            + _orbit_policy_obs_digest_mismatch_detail(cpp_lines, py_lines)
        )


def orbit_force_self_noop_available(available_action_mask: torch.Tensor) -> torch.Tensor:
    """Set self noop (0 ships to same planet slot) to 1 for every source row / seat.

    Class index per source planet row ``i`` is ``i * ORBIT_MOVE_CLASSES_PER_TARGET`` (``sn == 0``).
    Accepts ``[slots, classes]`` or ``[agents, slots, classes]`` (in-place).
    """
    nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)
    n = int(ORBIT_PLANET_ACTION_SLOTS)
    pc = int(ORBIT_PER_PLANET_MOVE_CLASSES)
    idx = torch.arange(0, n * nb, nb, device=available_action_mask.device, dtype=torch.int64)
    rows = torch.arange(0, n, device=available_action_mask.device, dtype=torch.int64)
    if available_action_mask.ndim == 2:
        assert tuple(available_action_mask.shape) == (n, pc), available_action_mask.shape
        available_action_mask[rows, idx] = 1
        return available_action_mask
    if available_action_mask.ndim == 3:
        assert tuple(available_action_mask.shape[1:]) == (n, pc), available_action_mask.shape
        available_action_mask[:, rows, idx] = 1
        return available_action_mask
    raise ValueError(
        "orbit_force_self_noop_available: expected ndim 2 or 3, "
        f"got {available_action_mask.ndim=} shape={tuple(available_action_mask.shape)}"
    )


ORBIT_POLICY_OBS_FEATURE_CONTRACT = """
Orbit policy observation: planet / pairwise feature layout
============================================================

Per-seat tensors (one policy row before batch / time broadcasting):
  orbit_planet_features:
      [ORBIT_PLAYER_AXIS_SLOTS, ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES]
  orbit_planet_mask:
      [ORBIT_PLAYER_AXIS_SLOTS, ORBIT_MAX_PLANETS]
  orbit_planet_pairwise_features:
      [ORBIT_PLAYER_AXIS_SLOTS, ORBIT_PLANET_PAIRWISE_COUNT, ORBIT_EDGE_FEATURES]
      with ORBIT_PLANET_PAIRWISE_COUNT = ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS
      i.e. flattened (src, dst) row-major over planet indices (same layout as C++ env).
  orbit_planet_pairwise_mask:
      [ORBIT_PLAYER_AXIS_SLOTS, ORBIT_PLANET_PAIRWISE_COUNT]
  orbit_planet_arrival_features:
      [ORBIT_PLAYER_AXIS_SLOTS, ORBIT_MAX_PLANETS, ORBIT_PLANET_ARRIVAL_HORIZON,
       ORBIT_PLAYER_AXIS_SLOTS, ORBIT_PLANET_TEMPORAL_FEATURES]
  orbit_enemy_mask:
      [ORBIT_PLAYER_AXIS_SLOTS, ORBIT_ENEMY_AXIS_SLOTS]

Leading batch / time dims are prepended by the training pipeline as usual.

Player-dependent feature layout is always [self, enemy0, enemy1, enemy2]. Enemy slots are an
unordered set: models may aggregate or attend over active enemies, but must not assign gameplay
meaning to the enemy slot index.


Planet channels (ORBIT_PLANET_FEATURES)
---------------------------------------

Base block — indices 0 .. ORBIT_PLANET_BASE_FEATURES - 1 (player-independent):

  x, y — board coordinates (same units as Kaggle planet rows; masked to 0 on invalid slots in Python path)
  neutral_ships — ships on neutral planets (owner == -1); 0 when owned
  episode_step — integer episode step in [0, ORBIT_PLANET_EPISODE_STEP_FEATURE_DIVISOR] (same on all valid planet slots)
  is_static — 1 iff planet is not a comet and does not orbit under the rotation cutoff (same rule as C++ ``planet_is_rotating_for_mask`` negated); else 0
  is_dynamic — 1 iff planet is not a comet and orbits inside the rotation radius limit; else 0
  is_comet — 1 iff planet id is in Kaggle ``comet_planet_ids``; else 0
  comet_time_before_despawn — integer remaining comet path steps before removal; 0 on non-comets and inactive slots
  radius — planet radius (from row / sim)
  planet_production — planet production scalar independent of ownership (same as row production column)
  orbit_radius — distance from board center (50, 50) to planet center
  angular_velocity — Kaggle ``angularVelocity`` on dynamic planets only (``is_dynamic == 1``); 0 on static and comet slots
  sun_angle — ``atan2(y - center, x - center)`` radians, center = ORBIT_BOARD_CENTER

C++ path: geometry and production from sim state; ``is_*`` from comet ids and ``planet_is_rotating_for_mask``; angular velocity from ``angular_velocity_`` only when ``!is_comet && rotates``.

Player blocks — repeated ORBIT_PLAYER_AXIS_SLOTS times in [self, enemy0, enemy1, enemy2]
order.  Within each block, indices are
ORBIT_PLANET_PLAYER_FEATURE_OFFSET + block * ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER + k:

  ships_if_owned — planet ships when owner matches this self/enemy owner id
  total_fleet_frac — player total fleet ships / ORBIT_PLANET_PLAYER_FLEET_NORMALIZER when owner matches this block; else 0 (same gating as ships_if_owned)
  production_if_owned — planet production when owner matches this self/enemy owner id


Edge / pairwise channels (ORBIT_EDGE_FEATURES)
----------------------------------------------

Base block — indices 0 .. ORBIT_EDGE_BASE_FEATURES - 1 (one vector per flattened edge slot):

  distance — pairwise src→dst distance; 0 where pair invalid / masked out
  src_neutral — 1 iff src planet neutral (owner == -1), else 0 (invalid pairs masked to 0 in Python path)
  dst_neutral — 1 iff dst planet neutral (owner == -1), else 0 (idem)

Player blocks — same self/enemy order as planets; within each block,
ORBIT_EDGE_PLAYER_FEATURE_OFFSET + block * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER + k:

  src_owned — 1 iff src planet owned by this self/enemy owner id
  dst_owned — 1 iff dst planet owned by this self/enemy owner id


Fleet arrivals (orbit_planet_arrival_features)
----------------------------------------------

The tensor keeps time, player block, and temporal-feature axes separate. For player block ``k``
and temporal step ``t``, values describe in-flight fleets owned by self/enemy ``k`` whose first
simulated hit on a planet slot occurs after exactly ``t + 1`` steps from the current frame
(same ray / noop planet motion model as ``orbit_wars_cpp`` ``fleet_arrivals_from_state`` /
``fleet_arrivals_from_rows``; C++ stores the source state as ``[horizon, planets, owners]``
before mapping owners to self/enemy blocks).
Effective lookahead is capped to the remaining noop-cache frames as in C++ (``min(horizon, remaining)``).

Python raw policy feature generation is unsupported; policy obs must come from C++ ``*_CPP``
tensors. In 2p, active policy seats are slots 0 and 3, and each active row has exactly one
active enemy slot.


Evolution rules
---------------

Add new player-independent scalars only in the planet or edge *base* block.
Add new player-dependent scalars only inside the per-player block, in the same
order inside every block.  Do not add absolute player-id scalars.

When ORBIT_*_FEATURES or block widths change, update model config
orbit_impala.obs_feature_normalization (see configs.impala_orbit_model_hyperparams)
and src.models.orbit_obs_feature_layout together.
Changing ``ORBIT_PLANET_ARRIVAL_HORIZON`` changes the temporal axis length consumed by the
model arrivals stack; keep configs and ``src.models`` in sync.
"""


def _self_enemy_player_order(player_slot: int, num_players: int) -> tuple[int, int, int, int]:
    p = int(player_slot)
    np = int(num_players)
    s = int(ORBIT_PLAYER_AXIS_SLOTS)
    assert 0 <= p < s, (p, s)
    assert np in (2, 4), np
    if np == 2:
        return ORBIT_SELF_ENEMY_PLAYER_ORDER_2P_BY_POLICY_SLOT[p]
    return ORBIT_SELF_ENEMY_PLAYER_ORDER_4P[p]


def _assert_self_enemy_player_order_contract() -> None:
    assert (
        tuple(_self_enemy_player_order(p, 4) for p in range(4))
        == ORBIT_SELF_ENEMY_PLAYER_ORDER_4P
    )
    assert (
        tuple(_self_enemy_player_order(p, 2) for p in range(4))
        == ORBIT_SELF_ENEMY_PLAYER_ORDER_2P_BY_POLICY_SLOT
    )


_assert_self_enemy_player_order_contract()


def orbit_obs_policy_from_cpp_contract(out_top: dict[str, Any]) -> dict[str, Any]:
    """Maps env top-level ``*_CPP`` tensor outputs to ``obs`` keys (no Python raw obs pathway)."""
    merged: dict[str, torch.Tensor] = {}
    for k in ORBIT_POLICY_OBS_KEYS:
        cpp_key = f"{k}_CPP"
        assert cpp_key in out_top
        v = out_top[cpp_key]
        assert isinstance(v, torch.Tensor)
        merged[k] = v
    return merged


def orbit_obs_policy_from_raw(obs_raw: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError(
        "Python policy obs feature generation is unsupported; use C++ *_CPP policy obs"
    )


class ObsWrapper:
    """Adds policy ``obs``.

    When the inner env has ``_cpp_env_obs_full``, ``obs`` is renamed from top-level ``*_CPP`` tensors.
    Otherwise ``obs`` is built from ``obs_raw`` (Python Kaggle path).
    """

    def __init__(self, env: Any, flags: Any, wall_profiler: WallTreeProfiler | None = None) -> None:
        self.env = env
        self.flags = flags
        self._wall_prof = wall_profiler

    def _wall_span(self, name: str):
        return profiler_span(self._wall_prof, name)

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        out = self.env.reset(**kwargs)
        return self._with_obs(out)

    def step(self, actions: Any) -> dict[str, Any]:
        with self._wall_span("wrap_obs_inner"):
            out = self.env.step(actions)
        with self._wall_span("wrap_obs_build"):
            return self._with_obs(out)

    def _with_obs(self, out: dict[str, Any]) -> dict[str, Any]:
        if orbit_inner_cpp_env_obs_full(self.env) or orbit_inner_cpp_env_obs_validate(self.env):
            obs = orbit_obs_policy_from_cpp_contract(out)
        else:
            obs = self._obs_from_raw(out["obs_raw"])
        if dict_io_contract_validation_enabled() and int(self.flags.orbit_num_agents) == 2:
            orbit_assert_policy_obs_2p_self_enemy_blocks(obs)
        merged = {**out, "obs": obs}
        return validated_dict_io_contract_output(self.flags, merged, "obs_wrapper_output")

    def _obs_from_raw(self, obs_raw: dict[str, Any]) -> dict[str, Any]:
        return orbit_obs_policy_from_raw(obs_raw)
