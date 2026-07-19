from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
import multiprocessing as mp
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

# OpenSpiel optional envs log a long INFO block on ``import kaggle_environments``; not needed for orbit_wars.
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").disabled = True

from kaggle_environments import make

from .dict_io_contract import (
    maybe_validate_dict_io_contract_step_input,
    validated_dict_io_contract_output,
)
from .obs_wrapper import (
    ORBIT_ARRIVAL_FEATURES_CPP_OBS_ATOL,
    ORBIT_ARRIVAL_FEATURES_CPP_OBS_RTOL,
    _orbit_edge_feature_channel_name,
    _orbit_planet_feature_channel_name,
    ORBIT_HIT_CLASSES_PER_TARGET,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_MOVE_CLASS_SEND_STABLE_TAKEOVER_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_TAKEOVER_SUBINDEX,
    ORBIT_MOVE_SEND_SUBINDICES,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_BASE_FEATURE_ANGULAR_VELOCITY,
    ORBIT_PLANET_BASE_FEATURE_COMET_TIME_BEFORE_DESPAWN,
    ORBIT_PLANET_BASE_FEATURE_EPISODE_STEP,
    ORBIT_PLANET_BASE_FEATURE_IS_COMET,
    ORBIT_PLANET_BASE_FEATURE_IS_DYNAMIC,
    ORBIT_PLANET_BASE_FEATURE_IS_STATIC,
    ORBIT_PLANET_BASE_FEATURE_NEUTRAL_SHIPS,
    ORBIT_PLANET_BASE_FEATURE_ORBIT_RADIUS,
    ORBIT_PLANET_BASE_FEATURE_PLANET_PRODUCTION,
    ORBIT_PLANET_BASE_FEATURE_RADIUS,
    ORBIT_PLANET_BASE_FEATURE_SUN_ANGLE,
    ORBIT_PLANET_BASE_FEATURE_X,
    ORBIT_PLANET_BASE_FEATURE_Y,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
    ORBIT_PAIRWISE_FEATURES_CPP_OBS_ATOL,
    ORBIT_PAIRWISE_FEATURES_CPP_OBS_RTOL,
    ORBIT_PLANET_FEATURES_CPP_OBS_ATOL,
    ORBIT_PLANET_FEATURES_CPP_OBS_RTOL,
    ORBIT_POLICY_OBS_KEYS,
    orbit_active_policy_slots,
    orbit_force_self_noop_available,
    orbit_obs_policy_from_raw,
    orbit_place_compact_agent_axis,
    orbit_policy_slot_for_compact_agent,
    orbit_self_enemy_mask,
)
from .orbit_cpp_plain_sync import orbit_comet_path_by_planet_id as _orbit_comet_path_by_planet_id
from .orbit_reference_upstream_random_state import orbit_reference_upstream_random_derived_dict
from .orbit_tape_feature_pack import (
    orbit_policy_obs_action_edges_from_plain_and_policy_obs,
    orbit_policy_obs_feature_pack_from_plain_and_policy_obs,
)
from .orbit_wars_cpp_obs_stub import OrbitWarsCppObsStub
from .orbit_wars_cpp_ext import orbit_wars_cpp
from .wall_tree_profiler import WallTreeProfiler, profiler_span

ORBIT_WARS_ENV_NAME = "orbit_wars"
_VALID_NUM_AGENTS = frozenset({2, 4})
_ORBIT_WARS_LOCAL_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "cpp"
    / "orbit_wars"
    / "reference_kaggle_upstream_github_no_edit"
    / "orbit_wars.py"
)
_ORBIT_WARS_LOCAL_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "orbit_wars_reference_kaggle_upstream_github_no_edit",
    _ORBIT_WARS_LOCAL_REFERENCE_PATH,
)
assert _ORBIT_WARS_LOCAL_REFERENCE_SPEC is not None
assert _ORBIT_WARS_LOCAL_REFERENCE_SPEC.loader is not None
_ORBIT_WARS_LOCAL_REFERENCE = importlib.util.module_from_spec(
    _ORBIT_WARS_LOCAL_REFERENCE_SPEC
)
sys.modules[_ORBIT_WARS_LOCAL_REFERENCE_SPEC.name] = _ORBIT_WARS_LOCAL_REFERENCE
_ORBIT_WARS_LOCAL_REFERENCE_SPEC.loader.exec_module(_ORBIT_WARS_LOCAL_REFERENCE)
assert hasattr(_ORBIT_WARS_LOCAL_REFERENCE, "specification")
assert hasattr(_ORBIT_WARS_LOCAL_REFERENCE, "interpreter")
assert hasattr(_ORBIT_WARS_LOCAL_REFERENCE, "renderer")
assert hasattr(_ORBIT_WARS_LOCAL_REFERENCE, "html_renderer")
_ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_reset_trace_fmt_d = (
    orbit_wars_cpp.orbit_wars_format_double_for_reset_trace
)
_ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_trace_format_double = (
    orbit_wars_cpp.orbit_wars_format_double_for_reset_trace
)


def _orbit_wars_sim_detail_to_reset_trace(line: str) -> None:
    _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_reset_trace.append(line)


_ORBIT_PLANET_ROW_LEN = 7
_ORBIT_FLEET_ROW_LEN = 7
_ORBIT_PLANET_STATIC_FIELD_INDICES = (0, 2, 3, 4, 6)
_ORBIT_BOARD_CENTER = 50.0
_ORBIT_ROTATION_RADIUS_LIMIT = 50.0


def _orbit_planet_position_hash_sha256_from_rows(planet_rows: torch.Tensor) -> str:
    assert isinstance(planet_rows, torch.Tensor)
    assert tuple(planet_rows.shape) == (ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), (
        planet_rows.shape
    )
    assert not planet_rows.is_cuda
    position_columns = torch.tensor([0, 2, 3], dtype=torch.int64)
    position_rows = torch.index_select(
        planet_rows,
        dim=1,
        index=position_columns,
    ).to(dtype=torch.float32).contiguous()
    h = hashlib.sha256()
    h.update(b"orbit_planet_position_hash_sha256_v1")
    h.update(bytes(str(tuple(position_rows.shape)), "ascii"))
    h.update(position_rows.numpy().tobytes())
    return h.hexdigest()


def _orbit_planet_rows_tensor_from_plain_planets(planets: list[Any]) -> tuple[torch.Tensor, int]:
    assert isinstance(planets, list)
    n = len(planets)
    assert n <= ORBIT_MAX_PLANETS, (n, ORBIT_MAX_PLANETS)
    rows = torch.zeros((ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), dtype=torch.float32)
    for i in range(n):
        row = planets[i]
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        for j in range(_ORBIT_PLANET_ROW_LEN):
            rows[i, j] = float(row[j])
    return rows.contiguous(), n


def _orbit_fleet_rows_tensor_from_plain_fleets(fleets: list[Any]) -> torch.Tensor:
    assert isinstance(fleets, list)
    rows = torch.zeros((len(fleets), _ORBIT_FLEET_ROW_LEN), dtype=torch.float64)
    for i, row in enumerate(fleets):
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_FLEET_ROW_LEN
        for j in range(_ORBIT_FLEET_ROW_LEN):
            rows[i, j] = float(row[j])
    return rows.contiguous()


def _orbit_planet_position_hash_sha256_from_plain_planets(planets: list[Any]) -> str:
    rows, _n = _orbit_planet_rows_tensor_from_plain_planets(planets)
    return _orbit_planet_position_hash_sha256_from_rows(rows)


def _orbit_policy_obs_symmetry_mismatch_detail(
    *,
    key: str,
    ref_slot: int,
    other_slot: int,
    ref: torch.Tensor,
    other: torch.Tensor,
) -> str:
    assert isinstance(key, str) and len(key) > 0, key
    assert isinstance(ref_slot, int), type(ref_slot)
    assert isinstance(other_slot, int), type(other_slot)
    assert isinstance(ref, torch.Tensor)
    assert isinstance(other, torch.Tensor)
    assert tuple(ref.shape) == tuple(other.shape), (key, tuple(ref.shape), tuple(other.shape))
    ref_cpu = ref.detach().cpu()
    other_cpu = other.detach().cpu()
    diff = ref_cpu != other_cpu
    if ref_cpu.is_floating_point() or other_cpu.is_floating_point():
        diff_value = (
            ref_cpu.to(dtype=torch.float64) - other_cpu.to(dtype=torch.float64)
        ).abs()
        diff = diff_value > 0.0
        flat_diff = diff_value.reshape(-1)
        max_idx = int(flat_diff.argmax().item())
        max_abs_diff = float(flat_diff[max_idx].item())
        idx = tuple(
            int(x)
            for x in torch.unravel_index(
                torch.tensor(max_idx, dtype=torch.int64),
                diff_value.shape,
            )
        )
    else:
        diff_value = diff.to(dtype=torch.int64)
        nz = diff.nonzero(as_tuple=False)
        assert int(nz.shape[0]) > 0, key
        idx = tuple(int(x.item()) for x in nz[0])
        max_abs_diff = float(diff_value.reshape(-1).max().item())
    ref_v = ref_cpu[idx].item()
    other_v = other_cpu[idx].item()
    parts = [
        f"key={key}",
        f"ref_slot={ref_slot}",
        f"other_slot={other_slot}",
        f"shape={tuple(ref.shape)}",
        f"idx={idx}",
        f"ref={ref_v}",
        f"other={other_v}",
        f"max_abs_diff={max_abs_diff}",
    ]
    if key == "orbit_planet_features":
        assert len(idx) == 2, (key, idx)
        planet_slot, feature_channel = idx
        parts.append(f"planet_slot={planet_slot}")
        parts.append(
            "feature_channel="
            + str(feature_channel)
            + "("
            + _orbit_planet_feature_channel_name(feature_channel)
            + ")"
        )
    elif key == "orbit_planet_pairwise_features":
        assert len(idx) == 2, (key, idx)
        edge_flat, edge_channel = idx
        parts.append(f"edge_flat={edge_flat}")
        parts.append(f"src_slot={edge_flat // ORBIT_MAX_PLANETS}")
        parts.append(f"dst_slot={edge_flat % ORBIT_MAX_PLANETS}")
        parts.append(
            "edge_channel="
            + str(edge_channel)
            + "("
            + _orbit_edge_feature_channel_name(edge_channel)
            + ")"
        )
    elif key == "orbit_planet_arrival_features":
        assert len(idx) == 4, (key, idx)
        planet_slot, horizon_idx, player_block, temporal_channel = idx
        parts.append(f"planet_slot={planet_slot}")
        parts.append(f"horizon_idx={horizon_idx}")
        parts.append(f"player_block={player_block}")
        parts.append(f"temporal_channel={temporal_channel}")
    elif key == "available_action_mask":
        assert len(idx) == 2, (key, idx)
        src_slot, action_class = idx
        parts.append(f"src_slot={src_slot}")
        parts.append(f"dst_slot={action_class // ORBIT_MOVE_CLASSES_PER_TARGET}")
        parts.append(f"amount_class={action_class % ORBIT_MOVE_CLASSES_PER_TARGET}")
        parts.append(f"action_class={action_class}")
    elif key == "orbit_planet_pairwise_mask":
        assert len(idx) == 1, (key, idx)
        edge_flat = idx[0]
        parts.append(f"edge_flat={edge_flat}")
        parts.append(f"src_slot={edge_flat // ORBIT_MAX_PLANETS}")
        parts.append(f"dst_slot={edge_flat % ORBIT_MAX_PLANETS}")
    elif key == "orbit_planet_mask":
        assert len(idx) == 1, (key, idx)
        parts.append(f"planet_slot={idx[0]}")
    elif key == "action_taken_index":
        assert len(idx) == 2, (key, idx)
        planet_slot, trailing_dim = idx
        assert trailing_dim == 0, (key, idx)
        action_class = int(ref_v)
        parts.append(f"planet_slot={planet_slot}")
        parts.append(f"dst_slot={action_class // ORBIT_MOVE_CLASSES_PER_TARGET}")
        parts.append(f"ship_subindex={action_class % ORBIT_MOVE_CLASSES_PER_TARGET}")
        parts.append(f"action_class={action_class}")
    elif key == "orbit_enemy_mask":
        assert len(idx) == 1, (key, idx)
        parts.append(f"enemy_slot={idx[0]}")
    return "; ".join(parts)


_ORBIT_POLICY_OBS_CANONICAL_PLANET_SORT_CHANNELS = (
    ORBIT_PLANET_BASE_FEATURE_X,
    ORBIT_PLANET_BASE_FEATURE_Y,
    ORBIT_PLANET_BASE_FEATURE_RADIUS,
    ORBIT_PLANET_BASE_FEATURE_PLANET_PRODUCTION,
    ORBIT_PLANET_BASE_FEATURE_NEUTRAL_SHIPS,
    ORBIT_PLANET_BASE_FEATURE_EPISODE_STEP,
    ORBIT_PLANET_BASE_FEATURE_IS_STATIC,
    ORBIT_PLANET_BASE_FEATURE_IS_DYNAMIC,
    ORBIT_PLANET_BASE_FEATURE_IS_COMET,
    ORBIT_PLANET_BASE_FEATURE_COMET_TIME_BEFORE_DESPAWN,
    ORBIT_PLANET_BASE_FEATURE_ORBIT_RADIUS,
    ORBIT_PLANET_BASE_FEATURE_ANGULAR_VELOCITY,
    ORBIT_PLANET_BASE_FEATURE_SUN_ANGLE,
)
_ORBIT_POLICY_OBS_CANONICAL_PLANET_SORT_DECIMALS = 4


def _orbit_policy_obs_canonical_planet_perm_for_slot(
    *,
    policy_obs: dict[str, torch.Tensor],
    slot: int,
) -> torch.Tensor:
    assert isinstance(policy_obs, dict)
    s = int(slot)
    assert 0 <= s < ORBIT_PLAYER_AXIS_SLOTS, s
    planet_features = policy_obs["orbit_planet_features"]
    planet_mask = policy_obs["orbit_planet_mask"]
    assert isinstance(planet_features, torch.Tensor)
    assert isinstance(planet_mask, torch.Tensor)
    assert tuple(planet_features.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_MAX_PLANETS,
        ORBIT_PLANET_FEATURES,
    ), planet_features.shape
    assert tuple(planet_mask.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_MAX_PLANETS,
    ), planet_mask.shape
    pf = planet_features.detach().cpu()
    pm = planet_mask.detach().cpu()
    valid: list[tuple[tuple[float, ...], int]] = []
    invalid: list[int] = []
    for planet_slot in range(ORBIT_MAX_PLANETS):
        if float(pm[s, planet_slot].item()) > 0.5:
            key = tuple(
                round(
                    float(pf[s, planet_slot, ch].item()),
                    _ORBIT_POLICY_OBS_CANONICAL_PLANET_SORT_DECIMALS,
                )
                for ch in _ORBIT_POLICY_OBS_CANONICAL_PLANET_SORT_CHANNELS
            )
            valid.append((key, planet_slot))
        else:
            invalid.append(planet_slot)
    valid.sort(key=lambda row: row[0])
    order = [planet_slot for _key, planet_slot in valid] + invalid
    assert len(order) == ORBIT_MAX_PLANETS, order
    return torch.tensor(order, dtype=torch.int64, device=planet_features.device)


def _orbit_policy_obs_canonicalize_slot_tensor(
    *,
    key: str,
    slot_tensor: torch.Tensor,
    planet_perm: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(key, str) and len(key) > 0, key
    assert isinstance(slot_tensor, torch.Tensor)
    assert isinstance(planet_perm, torch.Tensor)
    assert tuple(planet_perm.shape) == (ORBIT_MAX_PLANETS,), planet_perm.shape
    assert planet_perm.dtype == torch.int64, planet_perm.dtype
    if key in ("orbit_planet_features", "orbit_planet_mask"):
        assert int(slot_tensor.shape[0]) == ORBIT_MAX_PLANETS, (key, tuple(slot_tensor.shape))
        return slot_tensor.index_select(0, planet_perm)
    if key == "action_taken_index":
        assert tuple(slot_tensor.shape) == (ORBIT_PLANET_ACTION_SLOTS, 1), (
            key,
            tuple(slot_tensor.shape),
        )
        canonical_rows = slot_tensor.index_select(0, planet_perm)
        inverse_perm = torch.empty_like(planet_perm)
        inverse_perm[planet_perm] = torch.arange(
            ORBIT_PLANET_ACTION_SLOTS,
            dtype=torch.int64,
            device=planet_perm.device,
        )
        old_class = canonical_rows[:, 0].to(dtype=torch.int64)
        old_dst = old_class // ORBIT_MOVE_CLASSES_PER_TARGET
        ship_subindex = old_class % ORBIT_MOVE_CLASSES_PER_TARGET
        canonical_dst = inverse_perm.index_select(0, old_dst)
        canonical_class = canonical_dst * ORBIT_MOVE_CLASSES_PER_TARGET + ship_subindex
        return canonical_class.reshape(ORBIT_PLANET_ACTION_SLOTS, 1).to(dtype=slot_tensor.dtype)
    if key == "orbit_planet_arrival_features":
        assert tuple(slot_tensor.shape) == (
            ORBIT_MAX_PLANETS,
            ORBIT_PLANET_ARRIVAL_HORIZON,
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_TEMPORAL_FEATURES,
        ), (key, tuple(slot_tensor.shape))
        return slot_tensor.index_select(0, planet_perm)
    if key == "orbit_planet_pairwise_mask":
        assert tuple(slot_tensor.shape) == (
            ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS,
        ), (key, tuple(slot_tensor.shape))
        pair = slot_tensor.reshape(ORBIT_MAX_PLANETS, ORBIT_MAX_PLANETS)
        pair = pair.index_select(0, planet_perm).index_select(1, planet_perm)
        return pair.reshape(ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS)
    if key == "orbit_planet_pairwise_features":
        assert int(slot_tensor.shape[0]) == ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS, (
            key,
            tuple(slot_tensor.shape),
        )
        pair = slot_tensor.reshape(ORBIT_MAX_PLANETS, ORBIT_MAX_PLANETS, int(slot_tensor.shape[1]))
        pair = pair.index_select(0, planet_perm).index_select(1, planet_perm)
        return pair.reshape(ORBIT_MAX_PLANETS * ORBIT_MAX_PLANETS, int(slot_tensor.shape[1]))
    if key == "available_action_mask":
        assert tuple(slot_tensor.shape) == (
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PER_PLANET_MOVE_CLASSES,
        ), (key, tuple(slot_tensor.shape))
        by_dst_ship = slot_tensor.reshape(
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_MOVE_CLASSES_PER_TARGET,
        )
        by_dst_ship = by_dst_ship.index_select(0, planet_perm).index_select(1, planet_perm)
        return by_dst_ship.reshape(ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_MOVE_CLASSES)
    if key in ("orbit_enemy_mask", "player_mask"):
        return slot_tensor
    raise AssertionError(("unexpected orbit policy obs key", key))


def _orbit_policy_obs_canonical_tensors_equal(
    *,
    key: str,
    ref: torch.Tensor,
    other: torch.Tensor,
) -> bool:
    assert isinstance(key, str) and len(key) > 0, key
    assert isinstance(ref, torch.Tensor)
    assert isinstance(other, torch.Tensor)
    assert tuple(ref.shape) == tuple(other.shape), (key, tuple(ref.shape), tuple(other.shape))
    if key == "orbit_planet_features":
        return bool(
            torch.allclose(
                ref,
                other,
                rtol=ORBIT_PLANET_FEATURES_CPP_OBS_RTOL,
                atol=ORBIT_PLANET_FEATURES_CPP_OBS_ATOL,
            )
        )
    if key == "orbit_planet_pairwise_features":
        return bool(
            torch.allclose(
                ref,
                other,
                rtol=ORBIT_PAIRWISE_FEATURES_CPP_OBS_RTOL,
                atol=ORBIT_PAIRWISE_FEATURES_CPP_OBS_ATOL,
            )
        )
    if key == "orbit_planet_arrival_features":
        return bool(
            torch.allclose(
                ref,
                other,
                rtol=ORBIT_ARRIVAL_FEATURES_CPP_OBS_RTOL,
                atol=ORBIT_ARRIVAL_FEATURES_CPP_OBS_ATOL,
            )
        )
    return bool(torch.equal(ref, other))


def _orbit_policy_obs_invariance_mismatch_across_active_players(
    *,
    policy_obs: dict[str, torch.Tensor],
    num_agents: int,
    episode_step: int,
    source: str,
) -> str | None:
    na = int(num_agents)
    assert na in _VALID_NUM_AGENTS, na
    assert int(episode_step) >= 0, episode_step
    active_slots = orbit_active_policy_slots(na)
    assert len(active_slots) >= 2, active_slots
    ref_slot = int(active_slots[0])
    planet_perm_by_slot = {
        int(slot): _orbit_policy_obs_canonical_planet_perm_for_slot(
            policy_obs=policy_obs,
            slot=int(slot),
        )
        for slot in active_slots
    }
    for key in ORBIT_POLICY_OBS_KEYS:
        if key == "action_taken_index":
            continue
        assert key in policy_obs, (key, sorted(policy_obs.keys()))
        tensor = policy_obs[key]
        assert isinstance(tensor, torch.Tensor), (key, type(tensor))
        assert int(tensor.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS, (key, tuple(tensor.shape))
        ref = _orbit_policy_obs_canonicalize_slot_tensor(
            key=key,
            slot_tensor=tensor[ref_slot],
            planet_perm=planet_perm_by_slot[ref_slot],
        )
        for slot in active_slots[1:]:
            other_slot = int(slot)
            other = _orbit_policy_obs_canonicalize_slot_tensor(
                key=key,
                slot_tensor=tensor[other_slot],
                planet_perm=planet_perm_by_slot[other_slot],
            )
            if _orbit_policy_obs_canonical_tensors_equal(
                key=key,
                ref=ref,
                other=other,
            ):
                continue
            detail = _orbit_policy_obs_symmetry_mismatch_detail(
                key=key,
                ref_slot=ref_slot,
                other_slot=other_slot,
                ref=ref,
                other=other,
            )
            return (
                "orbit policy obs invariance mismatch across active players\n"
                f"source={source}\n"
                f"episode_step={int(episode_step)}\n"
                f"num_agents={na}\n"
                f"active_slots={active_slots}\n"
                "planet_order=canonical_sort_by_player_relative_planet_features\n"
                + detail
            )
    return None


def _orbit_assert_policy_obs_invariant_across_active_players(
    *,
    policy_obs: dict[str, torch.Tensor],
    num_agents: int,
    episode_step: int,
    source: str,
) -> None:
    mismatch = _orbit_policy_obs_invariance_mismatch_across_active_players(
        policy_obs=policy_obs,
        num_agents=num_agents,
        episode_step=episode_step,
        source=source,
    )
    if mismatch is not None:
        raise AssertionError(mismatch)


def _orbit_assert_two_way_kaggle_row_lists_equal(
    *,
    kind: str,
    py_rows: list[Any],
    cpp_rows: list[Any],
    episode_step: int,
    row_len: int,
) -> None:
    assert isinstance(py_rows, list)
    assert isinstance(cpp_rows, list)
    n_py = len(py_rows)
    n_cpp = len(cpp_rows)
    assert n_py == n_cpp, (
        "orbit validate row list length mismatch",
        kind,
        int(episode_step),
        n_py,
        n_cpp,
    )
    for i in range(n_py):
        pr = py_rows[i]
        cr = cpp_rows[i]
        assert isinstance(pr, (list, tuple)) and len(pr) == row_len, (kind, i, type(pr), len(pr))
        assert isinstance(cr, (list, tuple)) and len(cr) == row_len, (kind, i, type(cr), len(cr))
        for j in range(row_len):
            pv = float(pr[j])
            cv = float(cr[j])
            assert pv == cv, (
                "orbit validate row field mismatch",
                kind,
                int(episode_step),
                i,
                j,
                pv,
                cv,
            )


def _orbit_assert_planet_static_rows_equal(
    *,
    kind: str,
    rows_by_source: dict[str, list[Any] | torch.Tensor],
    episode_step: int,
) -> None:
    assert len(rows_by_source) >= 2, rows_by_source.keys()
    lengths: dict[str, int] = {}
    normalized: dict[str, list[Any] | torch.Tensor] = {}
    for source, rows in rows_by_source.items():
        assert isinstance(source, str) and source, source
        if isinstance(rows, torch.Tensor):
            assert rows.ndim == 2, (source, rows.shape)
            assert int(rows.shape[1]) == _ORBIT_PLANET_ROW_LEN, (source, rows.shape)
            assert not rows.is_cuda
            normalized[source] = rows
            lengths[source] = int(rows.shape[0])
        else:
            assert isinstance(rows, list), (source, type(rows))
            normalized[source] = rows
            lengths[source] = len(rows)
    n_set = set(lengths.values())
    assert len(n_set) == 1, (
        "orbit validate planet static row length mismatch",
        kind,
        int(episode_step),
        lengths,
    )
    n = next(iter(n_set))

    def value_at(rows: list[Any] | torch.Tensor, row_index: int, field_index: int) -> float:
        if isinstance(rows, torch.Tensor):
            return float(rows[row_index, field_index].item())
        row = rows[row_index]
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN, (
            kind,
            row_index,
            type(row),
        )
        return float(row[field_index])

    sources = tuple(normalized.keys())
    ref_source = sources[0]
    ref_rows = normalized[ref_source]
    for i in range(n):
        for j in _ORBIT_PLANET_STATIC_FIELD_INDICES:
            ref_v = value_at(ref_rows, i, j)
            values = {ref_source: ref_v}
            for source in sources[1:]:
                v = value_at(normalized[source], i, j)
                values[source] = v
                assert v == ref_v, (
                    "orbit validate planet static field mismatch",
                    kind,
                    int(episode_step),
                    i,
                    j,
                    values,
                )


def _orbit_plain_minimal_for_cpp_reset_from_reference_random_state(
    *, reference_random_state: dict[str, Any]
) -> dict[str, Any]:
    """``planets`` / ``angular_velocity`` from upstream reference RNG (``orbit_reference_upstream_random_derived_dict``)."""
    planets = reference_random_state["planets"]
    assert isinstance(planets, list), type(planets)
    av = reference_random_state["angular_velocity"]
    assert isinstance(av, (int, float)), type(av)
    return {
        "planets": planets,
        "angular_velocity": float(av),
    }


def _orbit_trace_context(lines: list[str], center: int, radius: int = 2) -> list[str]:
    lo = max(0, int(center) - int(radius))
    hi = min(len(lines), int(center) + int(radius) + 1)
    return lines[lo:hi]


def _orbit_trace_token_is_number(token: str) -> bool:
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


def _orbit_trace_mismatch_message(prefix: str, py_trace: list[str], cpp_lines: list[str], idx: int) -> str:
    py_len = len(py_trace)
    cpp_len = len(cpp_lines)
    py_line = py_trace[idx] if idx < py_len else "<py eof>"
    cpp_line = cpp_lines[idx] if idx < cpp_len else "<cpp eof>"
    out: list[str] = [
        f"{prefix}",
        f"- line: {idx} (0-based)",
        f"- line_count: py={py_len} cpp={cpp_len}",
        "- mismatch:",
        f"  py : {py_line}",
        f"  cpp: {cpp_line}",
    ]
    tp = py_line.split("\t") if idx < py_len else []
    tc = cpp_line.split("\t") if idx < cpp_len else []
    if len(tp) >= 3 and len(tc) >= 3:
        out.append(
            f"- block: py=({tp[0]}, {tp[1]}, {tp[2]}) cpp=({tc[0]}, {tc[1]}, {tc[2]})"
        )
        if idx + 1 < cpp_len and cpp_lines[idx + 1] == py_line:
            out.append("- alignment_hint: extra line on cpp side")
        if idx + 1 < py_len and py_trace[idx + 1] == cpp_line:
            out.append("- alignment_hint: extra line on python side")
        if tp[0] == tc[0] and tp[1] == tc[1] and tp[2] == tc[2]:
            ncols = min(len(tp), len(tc))
            for j in range(3, ncols):
                if tp[j] == tc[j]:
                    continue
                field_idx = j - 3
                field_name = f"field_{field_idx}"
                if _orbit_trace_token_is_number(tp[j]) and _orbit_trace_token_is_number(tc[j]):
                    py_v = float(tp[j])
                    cpp_v = float(tc[j])
                    out.append(
                        f"- value_mismatch: {field_name} py={tp[j]} cpp={tc[j]} abs_diff={abs(py_v - cpp_v)}"
                    )
                else:
                    out.append(f"- value_mismatch: {field_name} py={tp[j]} cpp={tc[j]}")
                break
    py_ctx = _orbit_trace_context(py_trace, idx)
    cpp_ctx = _orbit_trace_context(cpp_lines, idx)
    out.append("- py_context:")
    for line in py_ctx:
        out.append(f"  {line}")
    out.append("- cpp_context:")
    for line in cpp_ctx:
        out.append(f"  {line}")
    return "\n".join(out)


def _orbit_wars_reset_trace_report_mismatch(py_trace: list[str], cpp_text: str) -> None:
    cpp_lines = cpp_text.splitlines()
    n = max(len(py_trace), len(cpp_lines))
    for i in range(n):
        py_line = py_trace[i] if i < len(py_trace) else "<py eof>"
        cpp_line = cpp_lines[i] if i < len(cpp_lines) else "<cpp eof>"
        if py_line != cpp_line:
            raise AssertionError(
                _orbit_trace_mismatch_message(
                    "orbit_wars reset trace mismatch",
                    py_trace,
                    cpp_lines,
                    i,
                )
            )


def _orbit_wars_reset_trace_append_policy_digest_from_obs_raw(
    obs_raw: dict[str, Any], num_agents: int
) -> None:
    ref = _ORBIT_WARS_LOCAL_REFERENCE
    fmt_d = ref._orbit_wars_reset_trace_fmt_d
    pol = orbit_obs_policy_from_raw(obs_raw)
    pf = pol["orbit_planet_features"].detach().cpu()
    na = int(num_agents)
    fi_plane = int(ORBIT_PLANET_BASE_FEATURE_ORBIT_RADIUS)
    for seat in range(na):
        pf_s = pf[seat]
        assert tuple(pf_s.shape) == (ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES)
        sum_pf = [0.0] * int(ORBIT_PLANET_FEATURES)
        for i in range(int(ORBIT_MAX_PLANETS)):
            for f in range(int(ORBIT_PLANET_FEATURES)):
                sum_pf[f] += float(pf_s[i, f].item())
        parts7 = [fmt_d(sum_pf[f]) for f in range(int(ORBIT_PLANET_FEATURES))]
        ref._orbit_wars_reset_trace.append(
            "07\tplanet_feat_chan_sum\t" + str(seat) + "\t" + "\t".join(parts7)
        )
    for seat in range(na):
        pf_s = pf[seat]
        parts8 = [fmt_d(float(pf_s[i, fi_plane].item())) for i in range(int(ORBIT_MAX_PLANETS))]
        ref._orbit_wars_reset_trace.append(
            "08\tplanet_feat_plane\t"
            + str(seat)
            + "\t"
            + str(fi_plane)
            + "\t"
            + "\t".join(parts8)
        )


_ORBIT_WARS_STEP_TRACE: list[str] = []


def _orbit_wars_step_trace_append(line: str) -> None:
    _ORBIT_WARS_STEP_TRACE.append(line)


_ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_sim_subtrace_append = (
    _orbit_wars_step_trace_append
)


def _orbit_wars_step_trace_report_mismatch(py_trace: list[str], cpp_text: str) -> None:
    cpp_lines = cpp_text.splitlines()
    n = max(len(py_trace), len(cpp_lines))
    for i in range(n):
        py_line = py_trace[i] if i < len(py_trace) else "<py eof>"
        cpp_line = cpp_lines[i] if i < len(cpp_lines) else "<cpp eof>"
        if py_line != cpp_line:
            raise AssertionError(
                _orbit_trace_mismatch_message(
                    "orbit_wars step trace mismatch",
                    py_trace,
                    cpp_lines,
                    i,
                )
            )


def _orbit_wars_step_trace_emit_planets(stage: str, planets: list) -> None:
    ref = _ORBIT_WARS_LOCAL_REFERENCE
    for row in sorted(planets, key=lambda p: int(p[0])):
        _orbit_wars_step_trace_append(
            f"{stage}\tplanet\t"
            + "\t".join(
                (
                    str(int(row[0])),
                    str(int(row[1])),
                    str(ref._orbit_wars_reset_trace_fmt_d(float(row[2]))),
                    str(ref._orbit_wars_reset_trace_fmt_d(float(row[3]))),
                    str(ref._orbit_wars_reset_trace_fmt_d(float(row[4]))),
                    str(ref._orbit_wars_reset_trace_fmt_d(float(row[5]))),
                    str(ref._orbit_wars_reset_trace_fmt_d(float(row[6]))),
                )
            )
        )


def _orbit_wars_step_trace_append_policy_digest_from_obs_raw(
    obs_raw: dict[str, Any], num_agents: int
) -> None:
    ref = _ORBIT_WARS_LOCAL_REFERENCE
    fmt_d = ref._orbit_wars_reset_trace_fmt_d
    pol = orbit_obs_policy_from_raw(obs_raw)
    pf = pol["orbit_planet_features"].detach().cpu()
    na = int(num_agents)
    fi_plane = int(ORBIT_PLANET_BASE_FEATURE_ORBIT_RADIUS)
    for seat in range(na):
        pf_s = pf[seat]
        assert tuple(pf_s.shape) == (ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES)
        sum_pf = [0.0] * int(ORBIT_PLANET_FEATURES)
        for i in range(int(ORBIT_MAX_PLANETS)):
            for f in range(int(ORBIT_PLANET_FEATURES)):
                sum_pf[f] += float(pf_s[i, f].item())
        parts7 = [fmt_d(sum_pf[f]) for f in range(int(ORBIT_PLANET_FEATURES))]
        _orbit_wars_step_trace_append(
            "07\tplanet_feat_chan_sum\t" + str(seat) + "\t" + "\t".join(parts7)
        )
    for seat in range(na):
        pf_s = pf[seat]
        parts8 = [fmt_d(float(pf_s[i, fi_plane].item())) for i in range(int(ORBIT_MAX_PLANETS))]
        _orbit_wars_step_trace_append(
            "08\tplanet_feat_plane\t"
            + str(seat)
            + "\t"
            + str(fi_plane)
            + "\t"
            + "\t".join(parts8)
        )


# Matches ``orbit_wars.json`` observation schema.
_ORBIT_OBSERVATION_KEYS: tuple[str, ...] = (
    "planets",
    "fleets",
    "player",
    "angular_velocity",
    "initial_planets",
    "next_fleet_id",
    "comets",
    "comet_planet_ids",
    "remainingOverageTime",
)

# Kaggle plain copy keys plus ``action_taken_index`` (tensor).
ORBIT_PLAIN_SEAT_OBS_KEYS: frozenset[str] = frozenset(_ORBIT_OBSERVATION_KEYS) | frozenset(("action_taken_index",))

# ``OrbitWarsEnv.step`` payload (one env): per-seat per-planet class (noop or send ships to a target planet).
ORBIT_STEP_KEY_ORBIT_PAIRWISE_CLASSES = "orbit_pairwise_classes"
ORBIT_STEP_KEY_TAPE_BASELINE_LEARN = "tape_baseline_LEARN"
ORBIT_STEP_KEY_TAPE_SUPERVISED_PRED = "tape_supervised_pred"
ORBIT_STEP_KEY_TAPE_SUPERVISED_TARGET = "tape_supervised_target"
ORBIT_STEP_KEY_TAPE_SUPERVISED_VALID = "tape_supervised_valid"

_ORBIT_TAPE_ENVELOPE_VERSION = 1
_ORBIT_TAPE_FRAME_VERSION = 2


def _orbit_wars_prng_seed(*, instance_id: int, episode_index: int) -> int:
    a = int(instance_id)
    b = int(episode_index)
    assert 0 <= a, (a,)
    assert 0 <= b < (1 << 32), (a, b)
    return (a << 32) | b


def _orbit_tape_hex_to_rgba_list(hex_rgb: int) -> list[int]:
    c = int(hex_rgb) & 0xFFFFFF
    r = (c >> 16) & 0xFF
    g = (c >> 8) & 0xFF
    b = c & 0xFF
    return [r, g, b, 255]


def _orbit_tape_owner_rgba(owner: int, num_agents: int) -> list[int]:
    o = int(owner)
    assert o == -1 or (0 <= o < int(num_agents)), (o, num_agents)
    if o == -1:
        return _orbit_tape_hex_to_rgba_list(0x64748B)
    palette = (0xDC2626, 0x2563EB, 0x16A34A, 0xD97706)
    return _orbit_tape_hex_to_rgba_list(palette[o])


def _orbit_tape_fill_frame(
    *,
    world: dict[str, Any],
    num_agents: int,
    episode_step: int,
    episode_done: bool,
    lines: list[dict[str, Any]],
    points: list[dict[str, Any]],
    texts: list[dict[str, Any]],
) -> None:
    planets = world["planets"]
    assert isinstance(planets, list)
    fleets = world["fleets"]
    assert isinstance(fleets, list)

    xs: list[float] = []
    ys: list[float] = []

    for row in planets:
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        owner = int(row[1])
        px = float(row[2])
        py = float(row[3])
        radius = float(row[4])
        ships = int(row[5])
        planet_id = int(row[0])
        rgba = _orbit_tape_owner_rgba(owner, num_agents)
        r_m = max(radius, 0.06)
        xs.append(px)
        ys.append(py)
        points.append(
            {
                "planet_id": int(planet_id),
                "x": px,
                "y": py,
                "r_m": float(r_m),
                "color": rgba,
                "layer": "orbit_planet",
            }
        )
        texts.append(
            {
                "x": px,
                "y": float(py + r_m + 0.14),
                "text": f"#{planet_id} {ships}",
                "size_px": 10.0,
                "color": rgba,
                "layer": "orbit_planet_label",
            }
        )

    for row in fleets:
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_FLEET_ROW_LEN
        owner = int(row[1])
        fx = float(row[2])
        fy = float(row[3])
        ang = float(row[4])
        nships = int(row[6])
        rgba = _orbit_tape_owner_rgba(owner, num_agents)
        xs.append(fx)
        ys.append(fy)
        lines.append(
            {
                "x0": fx,
                "y0": fy,
                "x1": float(fx + 0.35 * math.cos(ang)),
                "y1": float(fy + 0.35 * math.sin(ang)),
                "w_m": 0.04,
                "color": rgba,
                "layer": "orbit_fleet_heading",
            }
        )
        points.append(
            {
                "fleet_id": int(row[0]),
                "x": fx,
                "y": fy,
                "r_m": 0.09,
                "color": rgba,
                "layer": "orbit_fleet",
            }
        )
        texts.append(
            {
                "x": float(fx + 0.14),
                "y": fy,
                "text": str(nships),
                "size_px": 10.0,
                "color": rgba,
                "layer": "orbit_fleet_label",
            }
        )

    if len(xs) > 0:
        hud_x = float(min(xs) - 0.9)
        hud_y = float(max(ys) + 0.65)
    else:
        hud_x = 0.0
        hud_y = 0.0

    totals = "\n".join(
        f"P{i} {fleet_ship_count_for_player(planets, fleets, i)}" for i in range(int(num_agents))
    )
    hud = f"step {int(episode_step)}\ndone {int(bool(episode_done))}\n{totals}"
    texts.append(
        {
            "x": hud_x,
            "y": hud_y,
            "text": hud,
            "size_px": 13.0,
            "color": [248, 250, 252, 255],
            "layer": "orbit_hud",
        }
    )


def _orbit_tape_world0_from_obs_all(obs_all: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(obs_all) >= 1
    w0 = obs_all[0]
    assert int(w0["player"]) == 0
    return w0


def _orbit_tape_supervised_pred_cell_mean_scalar(cell: object) -> float:
    if isinstance(cell, list):
        if len(cell) == 0:
            return 0.0
        first = cell[0]
        assert isinstance(first, list) and len(first) == 2, first
        return float(first[1])
    assert isinstance(cell, (int, float)), type(cell)
    return float(cell)


def _orbit_tape_format_move_action_class(action_class: int) -> str:
    c = int(action_class)
    assert 0 <= c < int(ORBIT_PER_PLANET_MOVE_CLASSES), c
    dst_slot = c // int(ORBIT_MOVE_CLASSES_PER_TARGET)
    amount_class = c % int(ORBIT_MOVE_CLASSES_PER_TARGET)
    if amount_class == 0:
        amount_name = "noop"
    elif amount_class == int(ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX):
        amount_name = "send_all"
    elif amount_class == int(ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX):
        amount_name = "send_half"
    elif amount_class == int(ORBIT_MOVE_CLASS_SEND_TAKEOVER_SUBINDEX):
        amount_name = "send_takeover"
    elif amount_class == int(ORBIT_MOVE_CLASS_SEND_STABLE_TAKEOVER_SUBINDEX):
        amount_name = "send_stable_takeover"
    else:
        amount_name = "reserved"
    return f"dst={dst_slot} cls={amount_name}({amount_class})"


def _orbit_tape_supervised_format_pred_cell(*, head_name: str, cell: object) -> str:
    if isinstance(cell, list):
        if len(cell) == 0:
            return "—"
        first = cell[0]
        assert isinstance(first, list) and len(first) == 2, (head_name, cell)
        parts: list[str] = []
        for pair in cell:
            assert isinstance(pair, list) and len(pair) == 2, pair
            action_class = int(pair[0])
            prob = float(pair[1])
            if head_name == "final_policy":
                parts.append(f"{_orbit_tape_format_move_action_class(action_class)} p={prob:.3f}")
            else:
                parts.append(f"{action_class}:{prob:.3f}")
        return " ".join(parts)
    assert isinstance(cell, (int, float)), (head_name, type(cell))
    return f"p={float(cell):.3f}"


def _orbit_tape_supervised_format_target_cell(*, head_name: str, target: float) -> str:
    if head_name == "final_policy":
        return _orbit_tape_format_move_action_class(int(target))
    return f"{int(float(target))}"


def _orbit_tape_append_planet_supervised_texts_for_agents(
    *,
    world: dict[str, Any],
    texts: list[dict[str, Any]],
    player_supervised_heads: dict[str, dict[str, Any]],
    num_agents: int,
) -> None:
    planets = world["planets"]
    assert isinstance(planets, list)
    na = int(num_agents)
    head_names = sorted(player_supervised_heads.keys())
    for slot_idx, row in enumerate(planets):
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        owner = int(row[1])
        if not (0 <= owner < na):
            continue
        assert 0 <= slot_idx < ORBIT_MAX_PLANETS, (slot_idx, ORBIT_MAX_PLANETS)
        px = float(row[2])
        py = float(row[3])
        radius = float(row[4])
        r_m = max(radius, 0.06)
        color = _orbit_tape_owner_rgba(owner, na)
        text_color = [int(color[0]), int(color[1]), int(color[2]), 235]
        line_index = 0
        for head_name in head_names:
            head_payload = player_supervised_heads[head_name]
            pred_all = head_payload["prediction"]
            tgt_all = head_payload["target"]
            valid_all = head_payload["valid"]
            assert isinstance(pred_all, list) and isinstance(tgt_all, list) and isinstance(valid_all, list), (
                head_name,
                type(pred_all),
                type(tgt_all),
                type(valid_all),
            )
            assert len(pred_all) == na and len(tgt_all) == na and len(valid_all) == na, (
                head_name,
                len(pred_all),
                len(tgt_all),
                len(valid_all),
                na,
            )
            pred_row = pred_all[owner]
            tgt_row = tgt_all[owner]
            valid_row = valid_all[owner]
            assert len(pred_row) == ORBIT_MAX_PLANETS
            assert len(tgt_row) == ORBIT_MAX_PLANETS
            assert len(valid_row) == ORBIT_MAX_PLANETS
            if not bool(valid_row[slot_idx]):
                continue
            pred_cell = pred_row[slot_idx]
            tgt_v = float(tgt_row[slot_idx])
            pred_str = _orbit_tape_supervised_format_pred_cell(head_name=head_name, cell=pred_cell)
            tgt_str = _orbit_tape_supervised_format_target_cell(head_name=head_name, target=tgt_v)
            texts.append(
                {
                    "x": float(px + r_m + 0.12),
                    "y": float(py + r_m + 0.30 + 0.14 * line_index),
                    "text": f"{head_name}: PRED {pred_str} GT {tgt_str}",
                    "size_px": 8.0,
                    "color": text_color,
                    "layer": "orbit_planet_supervised_label",
                }
            )
            line_index += 1


def orbit_tape_frame_dict_from_obs_all(
    obs_all: list[dict[str, Any]],
    *,
    num_agents: int,
    episode_step: int,
    episode_done: bool,
    player_value_baseline: list[float] | None,
    player_supervised_heads: dict[str, dict[str, Any]] | None,
    action_edges: list[dict[str, Any]],
    fleet_arrival_traces: list[dict[str, Any]],
    fleet_arrival_resolution: list[dict[str, Any]] | None = None,
    orbit_planet_feature_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    na = int(num_agents)
    assert len(obs_all) == na, (len(obs_all), na)
    lines: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    world = _orbit_tape_world0_from_obs_all(obs_all)
    _orbit_tape_fill_frame(
        world=world,
        num_agents=na,
        episode_step=int(episode_step),
        episode_done=bool(episode_done),
        lines=lines,
        points=points,
        texts=texts,
    )
    out: dict[str, Any] = {
        "version": int(_ORBIT_TAPE_FRAME_VERSION),
        "lines": lines,
        "points": points,
        "texts": texts,
        "action_edges": action_edges,
        "fleet_arrival_traces": fleet_arrival_traces,
    }
    if fleet_arrival_resolution is not None:
        out["fleet_arrival_resolution"] = fleet_arrival_resolution
    if orbit_planet_feature_pack is not None:
        out["orbit_planet_feature_pack"] = orbit_planet_feature_pack
    if player_value_baseline is not None:
        assert len(player_value_baseline) == na, (
            len(player_value_baseline),
            na,
        )
        out["player_value_baseline"] = [float(player_value_baseline[i]) for i in range(na)]
        out["player_value_baseline_eliminated"] = [
            not player_alive_seat(obs_all, i) for i in range(na)
        ]
    if player_supervised_heads is not None:
        out_heads: dict[str, dict[str, list[float] | list[bool]]] = {}
        for head_name, head_payload in player_supervised_heads.items():
            assert isinstance(head_payload, dict), (head_name, type(head_payload))
            pred = head_payload["prediction"]
            tgt = head_payload["target"]
            valid = head_payload["valid"]
            assert isinstance(pred, list) and isinstance(tgt, list) and isinstance(valid, list), (
                head_name,
                type(pred),
                type(tgt),
                type(valid),
            )
            assert len(pred) == na and len(tgt) == na and len(valid) == na, (
                head_name,
                len(pred),
                len(tgt),
                len(valid),
                na,
            )
            pred_mean: list[float] = []
            tgt_mean: list[float] = []
            for i in range(na):
                pred_row = pred[i]
                tgt_row = tgt[i]
                valid_row = valid[i]
                assert isinstance(pred_row, list) and isinstance(tgt_row, list) and isinstance(valid_row, list), (
                    head_name,
                    i,
                    type(pred_row),
                    type(tgt_row),
                    type(valid_row),
                )
                assert len(pred_row) == ORBIT_MAX_PLANETS
                assert len(tgt_row) == ORBIT_MAX_PLANETS
                assert len(valid_row) == ORBIT_MAX_PLANETS
                denom = sum(1 for j in range(ORBIT_MAX_PLANETS) if bool(valid_row[j]))
                if denom == 0:
                    pred_mean.append(0.0)
                    tgt_mean.append(0.0)
                else:
                    pred_mean.append(
                        sum(
                            _orbit_tape_supervised_pred_cell_mean_scalar(pred_row[j])
                            for j in range(ORBIT_MAX_PLANETS)
                            if bool(valid_row[j])
                        )
                        / float(denom)
                    )
                    tgt_mean.append(
                        sum(float(tgt_row[j]) for j in range(ORBIT_MAX_PLANETS) if bool(valid_row[j]))
                        / float(denom)
                    )
            out_heads[str(head_name)] = {
                "prediction": pred_mean,
                "target": tgt_mean,
                "eliminated": [not player_alive_seat(obs_all, i) for i in range(na)],
            }
        _orbit_tape_append_planet_supervised_texts_for_agents(
            world=world,
            texts=texts,
            player_supervised_heads=player_supervised_heads,
            num_agents=na,
        )
        out["player_supervised_heads"] = out_heads
    return out


def make_orbit_wars_env(
    *,
    configuration: Mapping[str, Any] | None = None,
    debug: bool = False,
) -> Any:
    cfg = dict(configuration) if configuration is not None else {}
    local_env = {
        "specification": copy.deepcopy(_ORBIT_WARS_LOCAL_REFERENCE.specification),
        "interpreter": _ORBIT_WARS_LOCAL_REFERENCE.interpreter,
        "renderer": _ORBIT_WARS_LOCAL_REFERENCE.renderer,
        "html_renderer": _ORBIT_WARS_LOCAL_REFERENCE.html_renderer,
    }
    return make(local_env, configuration=cfg, debug=debug)


def _observation_get(obs: Any, key: str) -> Any:
    if isinstance(obs, dict):
        assert key in obs, f"missing observation key {key!r}"
        return obs[key]
    assert hasattr(obs, key), f"missing observation attribute {key!r}"
    return getattr(obs, key)


def orbit_observation_to_plain(obs: Any) -> dict[str, Any]:
    """Map Kaggle observation to a new flat ``dict`` (one snapshot). Values are references: do not mutate them."""
    return {k: _observation_get(obs, k) for k in _ORBIT_OBSERVATION_KEYS}


def assert_orbit_action(action: Any) -> list[Any]:
    """Kaggle action: list of moves ``[from_planet_id, direction_angle, num_ships]`` (may be empty)."""
    assert isinstance(action, list)
    for m in action:
        assert isinstance(m, list) and len(m) == 3
        pid, ang, nships = m
        assert isinstance(pid, int)
        assert isinstance(ang, (int, float))
        assert isinstance(nships, int)
    return action


def fleet_ship_count_for_player(planets: list[Any], fleets: list[Any], player_id: int) -> int:
    """Total ships owned by ``player_id``: on planets plus in flight (orbit row layouts)."""
    pid = int(player_id)
    n = 0
    for p in planets:
        assert isinstance(p, (list, tuple)) and len(p) == _ORBIT_PLANET_ROW_LEN
        if int(p[1]) == pid:
            n += int(p[5])
    for f in fleets:
        assert isinstance(f, (list, tuple)) and len(f) == _ORBIT_FLEET_ROW_LEN
        if int(f[1]) == pid:
            n += int(f[6])
    return n


def fleet_total_seat(obs_all: list[dict[str, Any]], seat: int) -> int:
    """Ship count for seat ``seat`` using that seat's observation (``player`` must match ``seat``)."""
    obs = obs_all[int(seat)]
    assert int(obs["player"]) == int(seat)
    return fleet_ship_count_for_player(obs["planets"], obs["fleets"], int(seat))


def player_alive_for_player(planets: list[Any], fleets: list[Any], player_id: int) -> bool:
    """Original Orbit Wars alive contract: owns a planet or has total ships > 0."""
    pid = int(player_id)
    return (
        planet_count_for_player(planets, pid) > 0
        or fleet_ship_count_for_player(planets, fleets, pid) > 0
    )


def player_alive_seat(obs_all: list[dict[str, Any]], seat: int) -> bool:
    """Alive contract for a seat observation: owns a planet or has total ships > 0."""
    obs = obs_all[int(seat)]
    assert int(obs["player"]) == int(seat)
    return player_alive_for_player(obs["planets"], obs["fleets"], int(seat))


def planet_count_for_player(planets: list[Any], player_id: int) -> int:
    pid = int(player_id)
    n = 0
    for p in planets:
        assert isinstance(p, (list, tuple)) and len(p) == _ORBIT_PLANET_ROW_LEN
        if int(p[1]) == pid:
            n += 1
    return n


def production_sum_for_player(planets: list[Any], player_id: int) -> float:
    pid = int(player_id)
    total = 0.0
    for p in planets:
        assert isinstance(p, (list, tuple)) and len(p) == _ORBIT_PLANET_ROW_LEN
        if int(p[1]) == pid:
            total += float(p[6])
    return float(total)


def planet_count_seat(obs_all: list[dict[str, Any]], seat: int) -> int:
    obs = obs_all[int(seat)]
    assert int(obs["player"]) == int(seat)
    return planet_count_for_player(obs["planets"], int(seat))


def production_sum_seat(obs_all: list[dict[str, Any]], seat: int) -> float:
    obs = obs_all[int(seat)]
    assert int(obs["player"]) == int(seat)
    return production_sum_for_player(obs["planets"], int(seat))


def _observations_all_plain(env: Any) -> list[dict[str, Any]]:
    assert len(env.state) >= 1
    return [orbit_observation_to_plain(s.observation) for s in env.state]

def tensor_action_taken_index_from_per_planet_classes(
    *,
    classes: torch.Tensor,
) -> torch.Tensor:
    """``(ORBIT_PLANET_ACTION_SLOTS, 1)``: class index selected for each planet slot."""
    assert isinstance(classes, torch.Tensor)
    idx = classes.reshape(ORBIT_PLANET_ACTION_SLOTS, 1).to(dtype=torch.int32)
    assert idx.shape == (ORBIT_PLANET_ACTION_SLOTS, 1)
    assert not idx.is_cuda
    lo = int(idx.min().item())
    hi = int(idx.max().item())
    assert 0 <= lo and hi < ORBIT_PER_PLANET_MOVE_CLASSES, (lo, hi, ORBIT_PER_PLANET_MOVE_CLASSES)
    return idx.clone()


def zeros_action_taken_index_per_seat() -> torch.Tensor:
    return torch.zeros(
        (ORBIT_PLANET_ACTION_SLOTS, 1),
        dtype=torch.int32,
    )


def attach_zeros_action_taken_index_on_seats(seats_plain: list[dict[str, Any]]) -> None:
    for plain_dst in seats_plain:
        plain_dst["action_taken_index"] = zeros_action_taken_index_per_seat()


def attach_action_taken_index_from_per_planet_classes(
    seats_plain: list[dict[str, Any]],
    classes_per_seat: Sequence[torch.Tensor],
) -> None:
    assert len(seats_plain) == len(classes_per_seat)
    for plain_dst, cls in zip(seats_plain, classes_per_seat, strict=True):
        plain_dst["action_taken_index"] = tensor_action_taken_index_from_per_planet_classes(
            classes=cls
        )


def _planet_decode_tensors_from_plain(
    planets: list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(planets, list)
    n = len(planets)
    assert n <= ORBIT_MAX_PLANETS, (n, ORBIT_MAX_PLANETS)
    xy = torch.zeros((ORBIT_MAX_PLANETS, 2), dtype=torch.float32)
    mask = torch.zeros((ORBIT_MAX_PLANETS,), dtype=torch.float32)
    ids = torch.full((ORBIT_MAX_PLANETS,), -1, dtype=torch.int64)
    owners = torch.full((ORBIT_MAX_PLANETS,), -1, dtype=torch.int64)
    ships = torch.zeros((ORBIT_MAX_PLANETS,), dtype=torch.float32)
    radii = torch.zeros((ORBIT_MAX_PLANETS,), dtype=torch.float32)
    for i in range(n):
        row = planets[i]
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        xy[i, 0] = float(row[2])
        xy[i, 1] = float(row[3])
        mask[i] = 1.0
        ids[i] = int(row[0])
        owners[i] = int(row[1])
        ships[i] = float(row[5])
        radii[i] = float(row[4])
    return xy, mask, ids, owners, ships, radii


class _OrbitCppHonestAngleDualValidateSource:
    __slots__ = ("_reference_cpp", "_kaggle_cache_cpp")

    def __init__(self, reference_cpp: Any, kaggle_cache_cpp: Any) -> None:
        self._reference_cpp = reference_cpp
        self._kaggle_cache_cpp = kaggle_cache_cpp

    def cpp_honest_shared_angle(self, *, src_slot: int, dst_slot: int, ship_count: int) -> float:
        a = float(
            self._reference_cpp.cpp_honest_shared_angle(
                src_slot=src_slot,
                dst_slot=dst_slot,
                ship_count=ship_count,
            )
        )
        b = float(
            self._kaggle_cache_cpp.cpp_honest_shared_angle(
                src_slot=src_slot,
                dst_slot=dst_slot,
                ship_count=ship_count,
            )
        )
        assert abs(a - b) <= 1e-12, (
            "orbit C++ honest angle mismatch between reference and kaggle-cache stubs",
            src_slot,
            dst_slot,
            ship_count,
            a,
            b,
        )
        return a

    def cpp_honest_shared_angle_or_nan(self, *, src_slot: int, dst_slot: int, ship_count: int) -> float:
        a = float(
            self._reference_cpp.cpp_honest_shared_angle_or_nan(
                src_slot=src_slot,
                dst_slot=dst_slot,
                ship_count=ship_count,
            )
        )
        b = float(
            self._kaggle_cache_cpp.cpp_honest_shared_angle_or_nan(
                src_slot=src_slot,
                dst_slot=dst_slot,
                ship_count=ship_count,
            )
        )
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), (
                "orbit C++ honest angle validity mismatch between reference and kaggle-cache stubs",
                src_slot,
                dst_slot,
                ship_count,
                a,
                b,
            )
            return a
        assert abs(a - b) <= 1e-12, (
            "orbit C++ honest angle mismatch between reference and kaggle-cache stubs",
            src_slot,
            dst_slot,
            ship_count,
            a,
            b,
        )
        return a

    def cpp_honest_shared_dir_last(self) -> tuple[torch.Tensor, torch.Tensor]:
        ax, ay = self._reference_cpp.cpp_honest_shared_dir_last()
        bx, by = self._kaggle_cache_cpp.cpp_honest_shared_dir_last()
        assert torch.allclose(ax, bx, equal_nan=True), (
            "orbit C++ honest dir_x mismatch between reference and kaggle-cache stubs",
        )
        assert torch.allclose(ay, by, equal_nan=True), (
            "orbit C++ honest dir_y mismatch between reference and kaggle-cache stubs",
        )
        return ax, ay

    def cpp_honest_shared_angle_for_action(
        self,
        *,
        src_slot: int,
        dst_slot: int,
        ship_count: int,
        action_class: int,
    ) -> float:
        live_angle = float(
            self._reference_cpp.cpp_honest_shared_angle_for_action(
                src_slot=int(src_slot),
                dst_slot=int(dst_slot),
                ship_count=int(ship_count),
                action_class=int(action_class),
            )
        )
        dir_x, dir_y = self._kaggle_cache_cpp.cpp_static_honest_shared_dir_last()
        cls = int(action_class)
        dst = cls // int(ORBIT_MOVE_CLASSES_PER_TARGET)
        move_subindex = cls % int(ORBIT_MOVE_CLASSES_PER_TARGET)
        hit_cls = dst * int(ORBIT_HIT_CLASSES_PER_TARGET) + move_subindex
        cache_x = float(dir_x[int(src_slot), hit_cls].item())
        cache_y = float(dir_y[int(src_slot), hit_cls].item())
        if math.isnan(live_angle) or not (math.isfinite(cache_x) and math.isfinite(cache_y)):
            assert math.isnan(live_angle) and math.isnan(cache_x) and math.isnan(cache_y), (
                "orbit C++ honest action angle validity mismatch between live and static-cache dir_last",
                int(src_slot),
                int(dst_slot),
                int(ship_count),
                cls,
                hit_cls,
                live_angle,
                cache_x,
                cache_y,
            )
            return live_angle
        cache_angle = math.atan2(cache_y, cache_x)
        assert abs(live_angle - cache_angle) <= 1e-6, (
            "orbit C++ honest action angle mismatch between live and static-cache dir_last",
            int(src_slot),
            int(dst_slot),
            int(ship_count),
            cls,
            hit_cls,
            live_angle,
            cache_angle,
            cache_x,
            cache_y,
        )
        return live_angle


def orbit_per_planet_classes_to_kaggle_moves(
    *,
    planet_xy: torch.Tensor,
    planet_ids: torch.Tensor,
    planet_owners: torch.Tensor,
    planet_ships: torch.Tensor,
    planet_radii: torch.Tensor,
    classes_m: torch.Tensor,
    planet_count: int,
    player_id: int,
    ship_speed: float,
    honest_available_action_mask: torch.Tensor,
    honest_send_ships: torch.Tensor,
    honest_angle_source: Any,
) -> list[list[Any]]:
    assert planet_xy.shape == (ORBIT_MAX_PLANETS, 2)
    assert planet_ids.shape == (ORBIT_MAX_PLANETS,)
    assert planet_owners.shape == (ORBIT_MAX_PLANETS,)
    assert planet_ships.shape == (ORBIT_MAX_PLANETS,)
    assert planet_radii.shape == (ORBIT_MAX_PLANETS,)
    assert classes_m.shape == (ORBIT_PLANET_ACTION_SLOTS,)
    dev = planet_xy.device
    xy = planet_xy.to(device=dev, dtype=torch.float32)
    ids = planet_ids.to(device=dev, dtype=torch.int64)
    owners = planet_owners.to(device=dev, dtype=torch.int64)
    ships = planet_ships.to(device=dev, dtype=torch.float32)
    radii = planet_radii.to(device=dev, dtype=torch.float32)
    cls = classes_m.to(device=dev, dtype=torch.int64).reshape(-1)
    assert int(cls.shape[0]) == int(ORBIT_PLANET_ACTION_SLOTS)
    assert hasattr(honest_angle_source, "cpp_honest_shared_angle_for_action")
    n = int(planet_count)
    assert 0 <= n <= int(ORBIT_MAX_PLANETS)
    pid = int(player_id)
    m = int(ORBIT_MAX_PLANETS)
    assert isinstance(honest_available_action_mask, torch.Tensor)
    assert tuple(honest_available_action_mask.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), honest_available_action_mask.shape
    assert honest_available_action_mask.dtype == torch.int8
    assert not honest_available_action_mask.is_cuda
    assert isinstance(honest_send_ships, torch.Tensor)
    assert tuple(honest_send_ships.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), honest_send_ships.shape
    assert honest_send_ships.dtype == torch.int32
    assert not honest_send_ships.is_cuda
    ss = float(ship_speed)
    assert ss >= 1.0, ss
    moves: list[list[Any]] = []
    ids_l = [int(ids[i].item()) for i in range(m)]
    for i in range(m):
        if i >= n:
            continue
        c = int(cls[i].item())
        assert 0 <= c < int(ORBIT_PER_PLANET_MOVE_CLASSES), c
        assert int(honest_available_action_mask[i, c].item()) > 0, (i, c)
        j = c // ORBIT_MOVE_CLASSES_PER_TARGET
        sn = c % ORBIT_MOVE_CLASSES_PER_TARGET
        if sn == 0 and j == i:
            continue
        assert int(sn) in ORBIT_MOVE_SEND_SUBINDICES and j < n and j != i, (i, j, sn, c)
        from_id = int(ids_l[i])
        assert from_id >= 0, (i, from_id)
        assert int(owners[i].item()) == pid, (i, int(owners[i].item()), pid)
        rem_i = float(ships[i].item())
        assert rem_i > 0.0 and float(int(rem_i)) == rem_i, (i, rem_i)
        available_ships = int(rem_i)
        hit_cls = int(j) * int(ORBIT_HIT_CLASSES_PER_TARGET) + int(sn)
        n_send = int(honest_send_ships[i, hit_cls].item())
        assert n_send > 0, (i, j, sn, n_send)
        assert n_send <= available_ships, (i, j, sn, n_send, available_ships)
        ang = float(
            honest_angle_source.cpp_honest_shared_angle_for_action(
                src_slot=i,
                dst_slot=j,
                ship_count=n_send,
                action_class=c,
            )
        )
        assert math.isfinite(ang), (i, j, sn, n_send, ang)
        moves.append([from_id, ang, n_send])
    return moves


def kaggle_moves_for_seat_from_classes_honest_angle(
    *,
    seat_plain: dict[str, Any],
    classes: torch.Tensor,
    ship_speed: float,
    honest_available_action_mask: torch.Tensor,
    honest_send_ships: torch.Tensor,
    honest_angle_source: Any,
) -> list[Any]:
    planets = seat_plain["planets"]
    n = len(planets)
    player_id = int(seat_plain["player"])
    xy, _mask, ids, owners, ships, radii = _planet_decode_tensors_from_plain(planets)
    return orbit_per_planet_classes_to_kaggle_moves(
        planet_xy=xy,
        planet_ids=ids,
        planet_owners=owners,
        planet_ships=ships,
        planet_radii=radii,
        classes_m=classes,
        planet_count=n,
        player_id=player_id,
        ship_speed=ship_speed,
        honest_available_action_mask=honest_available_action_mask,
        honest_send_ships=honest_send_ships,
        honest_angle_source=honest_angle_source,
    )


def _merge_dicts_stack_agents(leaves: list[Any]) -> Any:
    v0 = leaves[0]
    if isinstance(v0, dict):
        keys = v0.keys()
        assert all(isinstance(d, dict) and set(d.keys()) == set(keys) for d in leaves)
        return {k: _merge_dicts_stack_agents([d[k] for d in leaves]) for k in keys}
    assert isinstance(v0, torch.Tensor)
    return torch.stack(list(leaves), dim=0)


def _merge_dicts_stack_policy_slots(leaves: list[Any], *, num_agents: int) -> Any:
    stacked = _merge_dicts_stack_agents(leaves)
    if isinstance(stacked, dict):
        return {
            k: orbit_place_compact_agent_axis(v, num_agents=int(num_agents))
            for k, v in stacked.items()
        }
    return orbit_place_compact_agent_axis(stacked, num_agents=int(num_agents))


def _stack_agent_scalar_metrics(
    scalars: list[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    assert isinstance(scalars, list)
    n = len(scalars)
    assert n >= 1
    for t in scalars:
        assert isinstance(t, torch.Tensor) and int(t.numel()) == 1
    return torch.stack([t.to(dtype=dtype, device=device) for t in scalars], dim=0)


def _stack_policy_slot_scalar_metrics(
    scalars: list[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
    num_agents: int,
) -> torch.Tensor:
    return orbit_place_compact_agent_axis(
        _stack_agent_scalar_metrics(scalars, dtype=dtype, device=device),
        num_agents=int(num_agents),
    )


def _orbit_fill_output_inactive_action_noops(out: dict[str, Any], *, num_agents: int) -> None:
    na = int(num_agents)
    assert na in _VALID_NUM_AGENTS, na
    active_slots = set(orbit_policy_slot_for_compact_agent(i, na) for i in range(na))
    available_action_mask = out["available_action_mask_CPP"]
    action_taken_index = out["action_taken_index_CPP"]
    assert isinstance(available_action_mask, torch.Tensor)
    assert isinstance(action_taken_index, torch.Tensor)
    assert tuple(available_action_mask.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), available_action_mask.shape
    assert tuple(action_taken_index.shape) == (
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_ACTION_SLOTS,
        1,
    ), action_taken_index.shape
    rows = torch.arange(
        ORBIT_PLANET_ACTION_SLOTS,
        device=available_action_mask.device,
        dtype=torch.int64,
    )
    noop_idx = rows * int(ORBIT_MOVE_CLASSES_PER_TARGET)
    for slot in range(ORBIT_PLAYER_AXIS_SLOTS):
        if slot in active_slots:
            continue
        available_action_mask[slot].zero_()
        available_action_mask[slot, rows, noop_idx] = 1
        action_taken_index[slot, :, 0] = noop_idx.to(
            device=action_taken_index.device,
            dtype=action_taken_index.dtype,
        )
def seat_obs_tensors_from_plain(
    *,
    plain: dict[str, Any],
    device: torch.device,
    ship_speed: float,
) -> dict[str, torch.Tensor]:
    """One agent: padded ``planet_rows``, ``planet_count``, ``comet_planet_slot_mask``,
    ``angular_velocity``, ``available_action_mask`` placeholder, ``action_taken_index``, ``player_mask``."""
    planets = plain["planets"]
    assert isinstance(planets, list)
    n = len(planets)
    assert n <= ORBIT_MAX_PLANETS, (n, ORBIT_MAX_PLANETS)
    planet_rows = torch.zeros((ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), dtype=torch.float32, device=device)
    if n > 0:
        for row in planets:
            assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        pr = torch.as_tensor(planets, dtype=torch.float32, device=device)
        assert int(pr.shape[0]) == n
        assert int(pr.shape[1]) == int(_ORBIT_PLANET_ROW_LEN)
        planet_rows[:n].copy_(pr)
    planet_count = torch.tensor(n, dtype=torch.int64, device=device)
    assert "action_taken_index" in plain
    taken = plain["action_taken_index"]
    assert isinstance(taken, torch.Tensor)
    action_taken_index = taken.to(device=device, dtype=torch.int32)
    assert tuple(action_taken_index.shape) == (ORBIT_PLANET_ACTION_SLOTS, 1)
    pid = int(plain["player"])
    player_alive = player_alive_for_player(plain["planets"], plain["fleets"], pid)
    player_mask = torch.tensor(1.0 if player_alive else 0.0, dtype=torch.float32, device=device)
    available_action_mask = torch.zeros(
        (ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_MOVE_CLASSES),
        dtype=torch.int8,
        device=device,
    )
    ap = int(ORBIT_PLAYER_AXIS_SLOTS)
    fleet_abs = torch.zeros((ap,), dtype=torch.float32, device=device)
    for aid in range(ap):
        fleet_abs[aid] = float(
            fleet_ship_count_for_player(plain["planets"], plain["fleets"], aid)
        )
    comet_ids_raw = plain["comet_planet_ids"]
    assert isinstance(comet_ids_raw, list)
    comet_set = frozenset(int(x) for x in comet_ids_raw)
    comet_planet_slot_mask = torch.zeros(
        (ORBIT_MAX_PLANETS,), dtype=torch.float32, device=device
    )
    if n > 0:
        for i in range(n):
            row = planets[i]
            assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
            pid = int(row[0])
            if pid in comet_set:
                comet_planet_slot_mask[i] = 1.0
    angular_velocity_t = torch.tensor(
        float(plain["angular_velocity"]), dtype=torch.float32, device=device
    )
    return {
        "planet_rows": planet_rows,
        "planet_count": planet_count,
        "fleet_arrival_features": torch.zeros(
            (
                ORBIT_MAX_PLANETS,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ),
            dtype=torch.float32,
            device=device,
        ),
        "comet_planet_slot_mask": comet_planet_slot_mask,
        "angular_velocity": angular_velocity_t,
        "fleet_abs_total_ships": fleet_abs,
        "available_action_mask": available_action_mask,
        "action_taken_index": action_taken_index,
        "player_mask": player_mask,
    }


def batched_player_obs_from_plain_seats(
    *,
    seats_plain: list[dict[str, Any]],
    device: torch.device,
    ship_speed: float,
) -> dict[str, Any]:
    """Stack compact Kaggle seats into policy slots; 2p uses slots 0 and 3."""
    assert isinstance(seats_plain, list)
    k = len(seats_plain)
    assert k in _VALID_NUM_AGENTS
    blocks = [
        seat_obs_tensors_from_plain(
            plain=o,
            device=device,
            ship_speed=ship_speed,
        )
        for o in seats_plain
    ]
    out = _merge_dicts_stack_policy_slots(blocks, num_agents=k)
    out["enemy_mask"] = orbit_self_enemy_mask(k, device=device)
    return out


def batched_planet_geometry_only_from_plain_seats(
    *,
    seats_plain: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Policy-slot ``planet_rows`` and ``planet_count`` only."""
    assert isinstance(seats_plain, list)
    k = len(seats_plain)
    assert k in _VALID_NUM_AGENTS
    planet_rows_list: list[torch.Tensor] = []
    planet_count_list: list[torch.Tensor] = []
    for plain in seats_plain:
        planets = plain["planets"]
        assert isinstance(planets, list)
        n = len(planets)
        assert n <= ORBIT_MAX_PLANETS, (n, ORBIT_MAX_PLANETS)
        planet_rows = torch.zeros(
            (ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN),
            dtype=torch.float32,
            device=device,
        )
        if n > 0:
            for row in planets:
                assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
            pr = torch.as_tensor(planets, dtype=torch.float32, device=device)
            assert int(pr.shape[0]) == n
            assert int(pr.shape[1]) == int(_ORBIT_PLANET_ROW_LEN)
            planet_rows[:n].copy_(pr)
        planet_count = torch.tensor(n, dtype=torch.int64, device=device)
        planet_rows_list.append(planet_rows)
        planet_count_list.append(planet_count)
    return {
        "planet_rows": orbit_place_compact_agent_axis(
            torch.stack(planet_rows_list, dim=0),
            num_agents=k,
        ),
        "planet_count": orbit_place_compact_agent_axis(
            torch.stack(planet_count_list, dim=0),
            num_agents=k,
        ),
    }


def _orbit_noop_action_classes_for_agents(num_agents: int) -> torch.Tensor:
    na = int(num_agents)
    assert na in _VALID_NUM_AGENTS, na
    classes = torch.zeros((na, ORBIT_PLANET_ACTION_SLOTS), dtype=torch.int32)
    for src in range(ORBIT_PLANET_ACTION_SLOTS):
        classes[:, src] = int(src * ORBIT_MOVE_CLASSES_PER_TARGET)
    return classes.contiguous()


def _orbit_honest_send_all_hit_mask_from_cpp_env(cpp_env: Any) -> torch.Tensor:
    out = torch.zeros(
        (ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_HIT_CLASSES),
        dtype=torch.int8,
        device=torch.device("cpu"),
    )
    cpp_env.honest_shared_send_all_hit_mask(out)
    return out


def _orbit_honest_send_ships_from_cpp_env(cpp_env: Any) -> torch.Tensor:
    out = cpp_env.honest_shared_send_ships_last()
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), out.shape
    assert out.dtype == torch.int32, out.dtype
    assert not out.is_cuda
    return out


def _orbit_assert_send_all_action_masks_equal(
    *,
    live_send_all_mask: torch.Tensor,
    external_send_all_mask: torch.Tensor,
    source: str,
    episode_step: int,
) -> None:
    assert isinstance(live_send_all_mask, torch.Tensor)
    assert isinstance(external_send_all_mask, torch.Tensor)
    assert tuple(live_send_all_mask.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), live_send_all_mask.shape
    assert tuple(external_send_all_mask.shape) == tuple(live_send_all_mask.shape), (
        external_send_all_mask.shape,
        live_send_all_mask.shape,
    )
    assert live_send_all_mask.dtype == torch.int8, live_send_all_mask.dtype
    assert external_send_all_mask.dtype == torch.int8, external_send_all_mask.dtype
    assert not live_send_all_mask.is_cuda
    assert not external_send_all_mask.is_cuda
    if torch.equal(live_send_all_mask, external_send_all_mask):
        return
    diff_ix = (live_send_all_mask != external_send_all_mask).nonzero(as_tuple=False)
    row = diff_ix[0]
    src = int(row[0].item())
    cls = int(row[1].item())
    raise AssertionError(
        (
            "orbit send_all action mask mismatch between live C++ state and static-cache external state",
            source,
            int(episode_step),
            int(diff_ix.shape[0]),
            src,
            cls // int(ORBIT_HIT_CLASSES_PER_TARGET),
            cls % int(ORBIT_HIT_CLASSES_PER_TARGET),
            cls,
            int(live_send_all_mask[src, cls].item()),
            int(external_send_all_mask[src, cls].item()),
        )
    )


def _orbit_available_action_mask_from_send_all_hit_mask_for_player(
    *,
    send_all_hit_mask: torch.Tensor,
    planet_rows: torch.Tensor,
    planet_count: int,
    player_id: int,
    wall_profiler: WallTreeProfiler | None,
) -> torch.Tensor:
    with profiler_span(wall_profiler, "player_contract"):
        assert isinstance(send_all_hit_mask, torch.Tensor)
        assert tuple(send_all_hit_mask.shape) == (
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PER_PLANET_HIT_CLASSES,
        ), send_all_hit_mask.shape
        assert send_all_hit_mask.dtype == torch.int8, send_all_hit_mask.dtype
        assert not send_all_hit_mask.is_cuda
        assert isinstance(planet_rows, torch.Tensor)
        assert tuple(planet_rows.shape) == (ORBIT_MAX_PLANETS, _ORBIT_PLANET_ROW_LEN), planet_rows.shape
        assert planet_rows.dtype == torch.float32
        assert not planet_rows.is_cuda
        n = int(planet_count)
        assert 0 <= n <= ORBIT_MAX_PLANETS, n
        pid = int(player_id)
    with profiler_span(wall_profiler, "init_noop_actions"):
        out = torch.zeros(
            (ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_MOVE_CLASSES),
            dtype=torch.int8,
            device=torch.device("cpu"),
        )
        rows = torch.arange(ORBIT_PLANET_ACTION_SLOTS, dtype=torch.int64)
        out[rows, rows * int(ORBIT_MOVE_CLASSES_PER_TARGET)] = 1
    with profiler_span(wall_profiler, "source_owner_ship_items"):
        active_srcs: list[int] = []
        for src in range(n):
            owner = int(planet_rows[src, 1].item())
            ships = int(planet_rows[src, 5].item())
            if owner == pid and ships > 0:
                active_srcs.append(src)
    with profiler_span(wall_profiler, "send_subindex_loop"):
        for send_subindex in ORBIT_MOVE_SEND_SUBINDICES:
            with profiler_span(wall_profiler, f"send_subindex_{int(send_subindex)}"):
                for src in active_srcs:
                    for dst in range(n):
                        if dst == src:
                            continue
                        hit_cls = int(dst) * int(ORBIT_HIT_CLASSES_PER_TARGET) + int(send_subindex)
                        action_cls = int(dst) * int(ORBIT_MOVE_CLASSES_PER_TARGET) + int(send_subindex)
                        out[src, action_cls] = (
                            1 if int(send_all_hit_mask[src, hit_cls].item()) > 0 else 0
                        )
    return out


def _orbit_available_action_mask_from_send_all_hit_mask_for_plain_seats(
    *,
    send_all_hit_mask: torch.Tensor,
    planet_rows: torch.Tensor,
    planet_count: int,
    obs_all: list[dict[str, Any]],
    num_agents: int,
    device: torch.device,
    wall_profiler: WallTreeProfiler | None,
) -> torch.Tensor:
    with profiler_span(wall_profiler, "plain_seats_contract"):
        na = int(num_agents)
        assert na in _VALID_NUM_AGENTS, na
        assert len(obs_all) == na, (len(obs_all), na)
    masks: list[torch.Tensor] = []
    with profiler_span(wall_profiler, "player_masks"):
        for seat in range(na):
            with profiler_span(wall_profiler, "player_mask"):
                masks.append(
                    _orbit_available_action_mask_from_send_all_hit_mask_for_player(
                        send_all_hit_mask=send_all_hit_mask,
                        planet_rows=planet_rows,
                        planet_count=int(planet_count),
                        player_id=int(obs_all[seat]["player"]),
                        wall_profiler=wall_profiler,
                    )
                )
    with profiler_span(wall_profiler, "stack_and_place_compact_axis"):
        return orbit_place_compact_agent_axis(
            torch.stack(masks, dim=0).to(device=device, dtype=torch.int8),
            num_agents=na,
        )


def _orbit_fleet_arrival_features_from_plain_seats(
    *,
    seats_plain: list[dict[str, Any]],
    stub: OrbitWarsCppObsStub,
    device: torch.device,
) -> torch.Tensor:
    na = len(seats_plain)
    assert na in _VALID_NUM_AGENTS, na
    features = stub.cpp_fleet_arrival_features_from_state(
        horizon=ORBIT_PLANET_ARRIVAL_HORIZON,
    )
    return features.to(device=device, dtype=torch.float32)


class OrbitWarsEnv:
    """Joint control (2 or 4 seats): Kaggle ``reset`` + internal noop ``step``, then ``step(actions)``.

    Returns ``obs_raw``, ``metrics``, and top-level ``orbit_episode_done`` for ``OrbitPaddingWrapper``.

    ``obs_raw["episode_step"]`` (int64) and top-level ``orbit_episode_done`` (float32) are length-``num_agents``
    vectors with the same scalar repeated per seat (global step and global Kaggle ``env.done``).

    Reward-related metrics are length-``num_agents`` ``float32`` vectors: slot ``i`` is
    ``orbit_fleet_delta`` = change in ship count for seat ``i`` since the previous returned state;
    ``env_step`` is ``1.0`` on each ``step`` output and ``0.0`` on ``reset`` output for each seat.

    ``step`` expects a dict with ``ORBIT_STEP_KEY_ORBIT_PAIRWISE_CLASSES``: a list of length
    ``num_agents`` of ``Tensor[ORBIT_PLANET_ACTION_SLOTS]`` int64 CPU (class per planet slot).
    Optionally ``ORBIT_STEP_KEY_TAPE_BASELINE_LEARN``: ``Tensor[num_agents]`` float32 CPU (model
    ``baseline_LEARN`` for the pre-transition state; RL debug tape HUD). Kaggle move lists are derived
    inside ``step`` from the pre-step seat observation and those classes.
    ``action_taken_index`` is zeros on ``reset`` and the per-planet class index after each
    ``step``.
    """

    def __init__(
        self,
        *,
        num_agents: int,
        configuration: Mapping[str, Any] | None = None,
        orbit_instance_id: int,
        debug: bool = False,
        visualize: bool = False,
        visualize_sim_env: bool = False,
        visualization_queue: mp.Queue | None = None,
        record_tape: bool = False,
        cpp_env_obs_full: bool = False,
        cpp_env_obs_validate: bool = False,
        flags: Any | None = None,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> None:
        assert num_agents in _VALID_NUM_AGENTS
        self._num_agents = int(num_agents)
        self._orbit_instance_id = int(orbit_instance_id)
        self._orbit_episode_index = 0
        self._flags = flags
        self._visualize = bool(visualize)
        self._visualize_sim_env = bool(visualize_sim_env)
        self._visualization_queue = visualization_queue
        self._record_tape = bool(record_tape)
        self._cpp_env_obs_full = bool(cpp_env_obs_full)
        self._cpp_env_obs_validate = bool(cpp_env_obs_validate)
        assert not (self._cpp_env_obs_full and self._cpp_env_obs_validate), (
            "cpp_env_obs_full and cpp_env_obs_validate cannot both be true"
        )
        kaggle_debug = bool(debug) or self._visualize_sim_env
        self._env = make_orbit_wars_env(configuration=configuration, debug=kaggle_debug)
        self._episode_step = 0
        self._prev_fleet_by_seat: list[int] = [0] * self._num_agents
        self._prev_alive_by_seat: list[bool] = [False] * self._num_agents
        self._orbit_elimination_game_result_credited: list[bool] = [False] * self._num_agents
        self._prev_planets_by_seat: list[int] = [0] * self._num_agents
        self._prev_production_by_seat: list[float] = [0.0] * self._num_agents
        self._orbit_tape_value_hold: list[float] = [0.0] * self._num_agents
        self._orbit_tape_supervised_hold: dict[str, dict[str, Any]] = {}
        self._orbit_tape_snapshots: list[list[Any]] = []
        self._wall_prof = wall_profiler
        self._cpp_obs_stub = OrbitWarsCppObsStub(orbit_env=self, device=self._device())
        self._cpp_kaggle_cache_stub = (
            OrbitWarsCppObsStub(orbit_env=self, device=self._device())
            if self._cpp_env_obs_validate
            else None
        )
        self._orbit_cpp_obs_validate_transition_was_reset = True
        self._orbit_cpp_obs_validate_last_py_digest_lines: list[str] = []
        self._orbit_policy_obs_invariance_check_disabled = False

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def visualize(self) -> bool:
        return self._visualize

    @property
    def visualize_sim_env(self) -> bool:
        return self._visualize_sim_env

    @property
    def visualization_queue(self) -> mp.Queue | None:
        return self._visualization_queue

    @property
    def record_tape(self) -> bool:
        return self._record_tape

    @property
    def cpp_env_obs_full(self) -> bool:
        return self._cpp_env_obs_full

    @property
    def cpp_env_obs_validate(self) -> bool:
        return self._cpp_env_obs_validate

    def _sample_orbit_episode_seed_on_reset(self) -> int:
        if self._visualize:
            assert self._flags is not None, "visualization env requires flags"
            fixed_episode_seed = int(self._flags.rl_vis_episode_seed)
            assert fixed_episode_seed >= -1, fixed_episode_seed
            if fixed_episode_seed >= 0:
                return fixed_episode_seed
        episode_seed = secrets.randbelow(1 << 31)
        assert 0 <= int(episode_seed) < (1 << 31), episode_seed
        return int(episode_seed)

    def _cpp_clock_step_for_noop_and_dataset(self) -> int:
        """``CppEnvLiveV2.episode_step()`` — clock used to index noop rows and honest-mask shards.

        The C++ clock advances only via ``stub.step()`` in the outer rollout loop. At reset each stub
        freezes ``comet_sync_updates`` into an immutable schedule; each payload is applied once when
        ``stub._cpp_env.episode_step()`` matches ``upd[\"episode_step\"]`` (see
        ``apply_comet_sync_scheduled_for_current_cpp_clock``). In ``cpp_env_obs_validate`` mode both
        C++ stubs share the same schedule from ``orbit_reference_upstream_random_derived_dict``.
        """
        assert self._cpp_obs_stub is not None
        return int(self._cpp_obs_stub._cpp_env.episode_step())

    def _orbit_assert_cpp_terminal_and_fleet_parity_vs_python(
        self,
        obs_all: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        assert self._cpp_env_obs_validate
        assert not self._cpp_env_obs_full
        stub = self._cpp_obs_stub
        assert stub is not None
        py_done = bool(self._env.done)
        cpp_done = stub.cpp_orbit_episode_terminal()
        assert py_done is cpp_done, (py_done, cpp_done)
        py_fleet_by_seat = [fleet_total_seat(obs_all, i) for i in range(self._num_agents)]
        py_fleet_delta = metrics["orbit_fleet_delta"]
        py_planets_delta = metrics["planets_delta"]
        py_production_delta = metrics["production_delta"]
        py_game_result = metrics["game_result"]
        assert isinstance(py_fleet_delta, torch.Tensor)
        assert isinstance(py_planets_delta, torch.Tensor)
        assert isinstance(py_production_delta, torch.Tensor)
        assert isinstance(py_game_result, torch.Tensor)
        assert tuple(py_fleet_delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
        assert tuple(py_planets_delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
        assert tuple(py_production_delta.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
        assert tuple(py_game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
        for i in range(self._num_agents):
            policy_slot = orbit_policy_slot_for_compact_agent(i, self._num_agents)
            py_f = py_fleet_by_seat[i]
            cpp_f = stub.cpp_fleet_ship_total_int_for_owner(i)
            assert py_f == cpp_f, (i, py_f, cpp_f)
            py_p = planet_count_seat(obs_all, i)
            cpp_p = stub.cpp_planet_count_int_for_owner(i)
            assert py_p == cpp_p, (i, py_p, cpp_p)
            py_prod = production_sum_seat(obs_all, i)
            cpp_prod = stub.cpp_production_sum_for_owner(i)
            assert abs(py_prod - cpp_prod) < 1e-6, (i, py_prod, cpp_prod)
            cpp_res = stub.cpp_game_result_for_owner(i)
            py_g = float(py_game_result[policy_slot].item())
            if not py_done:
                assert abs(cpp_res) < 1e-6, (i, cpp_res)
                assert abs(py_g) < 1e-6 or abs(py_g + 1.0) < 1e-5, (i, py_g)
            else:
                if self._orbit_elimination_game_result_credited[i]:
                    assert abs(py_g) < 1e-5 or abs(py_g + 1.0) < 1e-5, (i, py_g, cpp_res)
                else:
                    assert abs(py_g - cpp_res) < 1e-5, (i, py_g, cpp_res)
            py_f_delta_i = float(py_fleet_delta[policy_slot].item())
            cpp_f_delta_i = float(stub.cpp_fleet_delta_for_owner(i))
            assert abs(py_f_delta_i - cpp_f_delta_i) < 1e-6, (i, py_f_delta_i, cpp_f_delta_i)
            py_p_delta_i = float(py_planets_delta[policy_slot].item())
            cpp_p_delta_i = float(stub.cpp_planets_delta_for_owner(i))
            assert abs(py_p_delta_i - cpp_p_delta_i) < 1e-6, (i, py_p_delta_i, cpp_p_delta_i)
            py_prod_delta_i = float(py_production_delta[policy_slot].item())
            cpp_prod_delta_i = float(stub.cpp_production_delta_for_owner(i))
            assert abs(py_prod_delta_i - cpp_prod_delta_i) < 1e-6, (
                i,
                py_prod_delta_i,
                cpp_prod_delta_i,
            )

    def _orbit_assert_validate_live_and_noop_cache_rows_vs_python(
        self,
        obs_all: list[dict[str, Any]],
        *,
        source: str,
    ) -> None:
        assert self._cpp_env_obs_validate
        assert not self._cpp_env_obs_full
        assert len(obs_all) == self._num_agents
        stub = self._cpp_obs_stub
        kg = self._cpp_kaggle_cache_stub
        assert stub is not None
        assert kg is not None
        plain0 = obs_all[0]
        py_planets = plain0["planets"]
        py_fleets = plain0["fleets"]
        assert isinstance(py_planets, list)
        assert isinstance(py_fleets, list)
        ref_planets = stub.cpp_tape_kaggle_planet_rows()
        ref_fleets = stub.cpp_tape_kaggle_fleet_rows()
        kg_planets = kg.cpp_tape_kaggle_planet_rows()
        step_i = int(self._episode_step)
        _orbit_assert_two_way_kaggle_row_lists_equal(
            kind=f"{source}:planet_rows_python_vs_live_cpp",
            py_rows=py_planets,
            cpp_rows=ref_planets,
            episode_step=step_i,
            row_len=_ORBIT_PLANET_ROW_LEN,
        )
        _orbit_assert_two_way_kaggle_row_lists_equal(
            kind=f"{source}:fleet_rows_python_vs_live_cpp",
            py_rows=py_fleets,
            cpp_rows=ref_fleets,
            episode_step=step_i,
            row_len=_ORBIT_FLEET_ROW_LEN,
        )
        live_noop_row = stub.cpp_noop_trajectory_planets_row_tensor(step_i)
        kg_noop_row = kg.cpp_noop_trajectory_planets_row_tensor(step_i)
        assert isinstance(live_noop_row, torch.Tensor)
        assert isinstance(kg_noop_row, torch.Tensor)
        assert tuple(live_noop_row.shape) == (
            ORBIT_MAX_PLANETS,
            _ORBIT_PLANET_ROW_LEN,
        ), live_noop_row.shape
        assert tuple(kg_noop_row.shape) == (
            ORBIT_MAX_PLANETS,
            _ORBIT_PLANET_ROW_LEN,
        ), kg_noop_row.shape
        n_planets = len(py_planets)
        _orbit_assert_planet_static_rows_equal(
            kind=f"{source}:planet_static_rows_python_live_cpp_noop_cache",
            rows_by_source={
                "python": py_planets,
                "live_cpp": ref_planets,
                "noop_cache_cpp": kg_planets,
                "live_cpp_noop_trajectory": live_noop_row[:n_planets],
                "noop_cache_cpp_noop_trajectory": kg_noop_row[:n_planets],
            },
            episode_step=step_i,
        )

    def _cpp_noop_cache_position_hash(self, stub: OrbitWarsCppObsStub, step_index: int) -> str:
        idx = int(step_index)
        row = stub.cpp_noop_trajectory_planets_row_tensor(idx)
        assert tuple(row.shape) == (
            ORBIT_MAX_PLANETS,
            _ORBIT_PLANET_ROW_LEN,
        ), row.shape
        row = row.to(dtype=torch.float32, device=torch.device("cpu")).contiguous()
        return _orbit_planet_position_hash_sha256_from_rows(row)

    def _assert_cpp_kaggle_cache_position_parity(
        self,
        obs_all: list[dict[str, Any]],
        *,
        source: str,
    ) -> None:
        assert self._cpp_env_obs_validate
        assert len(obs_all) == self._num_agents
        assert self._cpp_obs_stub is not None
        assert self._cpp_kaggle_cache_stub is not None
        step_idx = int(self._episode_step)
        py_hash = _orbit_planet_position_hash_sha256_from_plain_planets(obs_all[0]["planets"])
        cpp_hash = self._cpp_noop_cache_position_hash(self._cpp_obs_stub, step_idx)
        kg_hash = self._cpp_noop_cache_position_hash(self._cpp_kaggle_cache_stub, step_idx)
        assert py_hash == cpp_hash == kg_hash, (
            "orbit C++ Kaggle-cache position parity mismatch",
            source,
            step_idx,
            py_hash,
            cpp_hash,
            kg_hash,
        )

    def _assert_cpp_kaggle_validator_cpp_stub_angular_velocity_parity(self) -> None:
        assert self._cpp_env_obs_validate
        assert self._cpp_obs_stub is not None
        assert self._cpp_kaggle_cache_stub is not None
        cpp_av = float(self._cpp_obs_stub._cpp_env.angular_velocity())
        kg_av = float(self._cpp_kaggle_cache_stub._cpp_env.angular_velocity())
        assert cpp_av == kg_av, (
            "orbit C++ Kaggle-validator angular velocity mismatch",
            cpp_av,
            kg_av,
        )

    def _external_cpp_policy_obs_from_plain_seats(
        self,
        *,
        obs_all: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        assert self._cpp_env_obs_validate
        assert self._cpp_kaggle_cache_stub is not None
        na = int(self._num_agents)
        assert na in _VALID_NUM_AGENTS, na
        assert len(obs_all) == na, (len(obs_all), na)
        plain0 = obs_all[0]
        for seat_plain in obs_all:
            assert seat_plain["planets"] == plain0["planets"]
            assert seat_plain["fleets"] == plain0["fleets"]

        kg = self._cpp_kaggle_cache_stub
        cpu = torch.device("cpu")
        batched = batched_planet_geometry_only_from_plain_seats(
            seats_plain=obs_all,
            device=cpu,
        )
        slot0 = orbit_policy_slot_for_compact_agent(0, na)
        planet_rows = batched["planet_rows"][slot0].to(dtype=torch.float32, device=cpu).contiguous()
        planet_count = int(batched["planet_count"][slot0].item())
        for compact_idx in range(1, na):
            slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
            assert int(batched["planet_count"][slot].item()) == planet_count, (
                slot,
                planet_count,
            )
            row = batched["planet_rows"][slot].to(dtype=torch.float32, device=cpu).contiguous()
            assert torch.equal(row, planet_rows), slot

        fleet_rows = _orbit_fleet_rows_tensor_from_plain_fleets(plain0["fleets"])
        cpp_out = kg._new_cpp_output()
        kg._cpp_env.fill_policy_obs_from_rows(
            fleet_rows,
            planet_rows,
            int(planet_count),
            cpp_out["orbit_planet_features_CPP"],
            cpp_out["orbit_planet_mask_CPP"],
            cpp_out["orbit_planet_pairwise_mask_CPP"],
            cpp_out["orbit_planet_pairwise_features_CPP"],
            cpp_out["action_taken_index_CPP"],
            cpp_out["player_mask_CPP"],
        )

        static_send_all_mask = kg.cpp_static_send_all_from_external(
            fleet_rows=fleet_rows,
            planet_rows=planet_rows,
            planet_count=planet_count,
        ).clone()

        available_action_mask = _orbit_available_action_mask_from_send_all_hit_mask_for_plain_seats(
            send_all_hit_mask=static_send_all_mask,
            planet_rows=planet_rows,
            planet_count=planet_count,
            obs_all=obs_all,
            num_agents=na,
            device=cpu,
            wall_profiler=None,
        )

        arrival_features = cpp_out["orbit_planet_arrival_features_CPP"]
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

        action_taken_rows: list[torch.Tensor] = []
        for seat_plain in obs_all:
            action_taken = seat_plain["action_taken_index"]
            assert isinstance(action_taken, torch.Tensor)
            assert tuple(action_taken.shape) == (ORBIT_PLANET_ACTION_SLOTS, 1), action_taken.shape
            assert action_taken.dtype == torch.int32
            assert not action_taken.is_cuda
            action_taken_rows.append(action_taken)

        out = {
            "orbit_planet_features": cpp_out["orbit_planet_features_CPP"].clone(),
            "orbit_planet_arrival_features": arrival_features.clone(),
            "orbit_enemy_mask": orbit_self_enemy_mask(
                na,
                device=arrival_features.device,
            ).clone(),
            "orbit_planet_mask": cpp_out["orbit_planet_mask_CPP"].clone(),
            "orbit_planet_pairwise_mask": cpp_out["orbit_planet_pairwise_mask_CPP"].clone(),
            "orbit_planet_pairwise_features": cpp_out[
                "orbit_planet_pairwise_features_CPP"
            ].clone(),
            "available_action_mask": available_action_mask.clone(),
            "action_taken_index": orbit_place_compact_agent_axis(
                torch.stack(action_taken_rows, dim=0),
                num_agents=na,
            ).clone(),
            "player_mask": cpp_out["player_mask_CPP"].clone(),
        }
        inactive_fill = {
            "available_action_mask_CPP": out["available_action_mask"],
            "action_taken_index_CPP": out["action_taken_index"],
        }
        _orbit_fill_output_inactive_action_noops(inactive_fill, num_agents=na)
        return out

    def _assert_live_cpp_policy_obs_matches_external_plain_cpp(
        self,
        *,
        live_cpp_extra: dict[str, torch.Tensor],
        obs_all: list[dict[str, Any]],
        source: str,
    ) -> None:
        assert self._cpp_env_obs_validate
        external = self._external_cpp_policy_obs_from_plain_seats(obs_all=obs_all)
        for key in ORBIT_POLICY_OBS_KEYS:
            cpp_key = f"{key}_CPP"
            assert cpp_key in live_cpp_extra, cpp_key
            live_t = live_cpp_extra[cpp_key]
            external_t = external[key]
            assert isinstance(live_t, torch.Tensor)
            assert isinstance(external_t, torch.Tensor)
            external_t = external_t.to(device=live_t.device, dtype=live_t.dtype)
            assert tuple(live_t.shape) == tuple(external_t.shape), (
                "orbit live-vs-external C++ policy obs shape mismatch",
                source,
                key,
                tuple(live_t.shape),
                tuple(external_t.shape),
            )
            if key in (
                "orbit_planet_features",
                "orbit_planet_pairwise_features",
                "orbit_planet_arrival_features",
            ):
                assert torch.allclose(live_t, external_t, rtol=1e-5, atol=1e-5), (
                    "orbit live-vs-external C++ policy obs float mismatch",
                    source,
                    key,
                    tuple(live_t.shape),
                    float((live_t - external_t).abs().max().item()),
                )
            else:
                assert torch.equal(live_t, external_t), (
                    "orbit live-vs-external C++ policy obs exact mismatch",
                    source,
                    key,
                    tuple(live_t.shape),
                    int((live_t != external_t).sum().item()),
                )

    def _reset_cpp_kaggle_cache_validator(
        self,
        obs_all: list[dict[str, Any]],
        *,
        reference_random_state: dict[str, Any],
    ) -> None:
        assert self._cpp_env_obs_validate
        assert self._cpp_kaggle_cache_stub is not None
        _ = self._cpp_kaggle_cache_stub.reset_from_external_random_state(
            random_state=reference_random_state,
            plain_state=obs_all[0],
        )
        self._assert_cpp_kaggle_validator_cpp_stub_angular_velocity_parity()
        self._assert_cpp_kaggle_cache_position_parity(obs_all, source="reset")

    def _update_cpp_kaggle_cache_validator_if_needed(self, obs_all: list[dict[str, Any]]) -> None:
        assert self._cpp_env_obs_validate
        assert self._cpp_kaggle_cache_stub is not None
        episode_step = int(self._episode_step)
        cur_step = int(self._cpp_kaggle_cache_stub._cpp_env.episode_step())
        assert cur_step == episode_step, (cur_step, episode_step)
        self._assert_cpp_kaggle_cache_position_parity(obs_all, source="step")
        self._orbit_assert_validate_live_and_noop_cache_rows_vs_python(
            obs_all,
            source="step",
        )

    def _orbit_plain_obs_all_for_tape_from_cpp_stub(
        self,
        stub: OrbitWarsCppObsStub,
    ) -> list[dict[str, Any]]:
        assert self._cpp_env_obs_full
        planets_list, fleets_list = stub.kaggle_plain_planets_and_fleets_for_tape()
        assert isinstance(planets_list, list)
        assert isinstance(fleets_list, list)
        out: list[dict[str, Any]] = []
        for seat in range(self._num_agents):
            out.append(
                {
                    "player": int(seat),
                    "planets": planets_list,
                    "fleets": fleets_list,
                }
            )
        return out

    def _orbit_episode_done_tensor_from_cpp_terminal(self, done_b: bool) -> torch.Tensor:
        v = 1.0 if bool(done_b) else 0.0
        done = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32, device=self._device())
        for compact_idx in range(self._num_agents):
            done[orbit_policy_slot_for_compact_agent(compact_idx, self._num_agents)] = v
        return done

    def _orbit_game_result_from_fleet_transition(
        self,
        prev_alive_by_seat: list[bool],
        new_alive_by_seat: list[bool],
        new_fleet_by_seat: list[int],
        terminal: bool,
    ) -> list[float]:
        """``game_result`` metric: -1 when alive ``true->false``.

        On elimination ``terminal``, the sole alive seat gets +1 and others -1. On
        step-limit ``terminal``, remaining seats get +1/-1 by final ship score; any tie for
        first is 0 for terminal scoring. Seats already given elimination -1 get 0 here so
        the outcome is not double-counted in ``reward``.
        """
        n = self._num_agents
        assert len(prev_alive_by_seat) == n
        assert len(new_alive_by_seat) == n
        assert len(new_fleet_by_seat) == n
        out = [0.0] * n
        for i in range(n):
            if (
                bool(prev_alive_by_seat[i])
                and not bool(new_alive_by_seat[i])
                and not bool(self._orbit_elimination_game_result_credited[i])
            ):
                out[i] = -1.0
                self._orbit_elimination_game_result_credited[i] = True
        if not terminal:
            return out
        alive_count = sum(1 for v in new_alive_by_seat if bool(v))
        assert 0 <= alive_count <= n, (alive_count, new_alive_by_seat)
        if alive_count <= 1:
            if alive_count == 0:
                terminal_vals = [-1.0] * n
            else:
                terminal_vals = [1.0 if bool(new_alive_by_seat[i]) else -1.0 for i in range(n)]
        else:
            max_fleet = max(int(v) for v in new_fleet_by_seat)
            winners = [int(v) == max_fleet for v in new_fleet_by_seat]
            winner_count = sum(1 for v in winners if v)
            assert 1 <= winner_count <= n, (winner_count, winners, new_fleet_by_seat)
            if max_fleet == 0 or winner_count != 1:
                terminal_vals = [0.0] * n
            else:
                terminal_vals = [1.0 if winners[i] else -1.0 for i in range(n)]
        merged = [0.0] * n
        for i in range(n):
            if abs(out[i] + 1.0) < 1e-5:
                merged[i] = out[i]
            elif self._orbit_elimination_game_result_credited[i]:
                merged[i] = 0.0
            else:
                merged[i] = terminal_vals[i]
        return merged

    def _policy_obs_cpp_extra_from_python(
        self, player_obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        _ = player_obs
        raise AssertionError(
            "Python policy obs feature generation is unsupported; use C++ *_CPP policy obs"
        )

    def _orbit_validate_policy_obs_invariance_or_disable(
        self,
        *,
        policy_obs: dict[str, torch.Tensor],
        source: str,
    ) -> None:
        if self._orbit_policy_obs_invariance_check_disabled:
            return
        mismatch = _orbit_policy_obs_invariance_mismatch_across_active_players(
            policy_obs=policy_obs,
            num_agents=self._num_agents,
            episode_step=int(self._episode_step),
            source=source,
        )
        if mismatch is not None:
            self._orbit_policy_obs_invariance_check_disabled = True
            logging.info(
                "Disabling orbit policy obs invariance check for the rest of this episode:\n%s",
                mismatch,
            )

    @property
    def episode_step_limit(self) -> int:
        return int(self._env.configuration.episodeSteps)

    @property
    def kaggle_steps_recorded(self) -> int:
        return len(self._env.steps)

    def _device(self) -> torch.device:
        return torch.device("cpu")

    def _split_cpp_extra_available_masks(
        self,
        cpp_extra: dict[str, torch.Tensor],
        *,
        planet_rows: torch.Tensor,
        planet_count: int,
    ) -> None:
        assert self._cpp_obs_stub is not None
        assert tuple(planet_rows.shape) == (
            ORBIT_PLANET_ACTION_SLOTS,
            _ORBIT_PLANET_ROW_LEN,
        ), planet_rows.shape
        assert 0 <= int(planet_count) <= ORBIT_PLANET_ACTION_SLOTS, planet_count
        available_action_mask = cpp_extra["available_action_mask_CPP"]
        assert isinstance(available_action_mask, torch.Tensor)
        assert tuple(available_action_mask.shape) == (
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PER_PLANET_MOVE_CLASSES,
        ), available_action_mask.shape
        assert available_action_mask.dtype == torch.int8, available_action_mask.dtype

    def _orbit_tape_world0(self, obs_all: list[dict[str, Any]]) -> dict[str, Any]:
        assert len(obs_all) == self._num_agents
        return _orbit_tape_world0_from_obs_all(obs_all)

    def _orbit_policy_obs_from_cpp_extra(
        self,
        cpp_extra: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {k: cpp_extra[f"{k}_CPP"] for k in ORBIT_POLICY_OBS_KEYS}

    def _orbit_tape_feature_pack_from_policy_obs(
        self,
        obs_all: list[dict[str, Any]],
        policy_obs: dict[str, torch.Tensor],
        hit_kind: torch.Tensor,
        intercept_fail_reason: torch.Tensor,
    ) -> dict[str, Any]:
        return orbit_policy_obs_feature_pack_from_plain_and_policy_obs(
            plain=self._orbit_tape_world0(obs_all),
            policy_obs=policy_obs,
            num_agents=self._num_agents,
            hit_kind=hit_kind,
            intercept_fail_reason=intercept_fail_reason,
        )

    def _orbit_tape_action_edges_from_policy_obs(
        self,
        obs_all: list[dict[str, Any]],
        policy_obs: dict[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        return orbit_policy_obs_action_edges_from_plain_and_policy_obs(
            plain=self._orbit_tape_world0(obs_all),
            policy_obs=policy_obs,
            num_agents=self._num_agents,
        )

    def _orbit_tape_fleet_hit_trace_horizon(self) -> int:
        return int(self._env.configuration.episodeSteps)

    def _orbit_tape_append_planet_supervised_texts(
        self,
        *,
        world: dict[str, Any],
        texts: list[dict[str, Any]],
        player_supervised_heads: dict[str, dict[str, Any]],
    ) -> None:
        _orbit_tape_append_planet_supervised_texts_for_agents(
            world=world,
            texts=texts,
            player_supervised_heads=player_supervised_heads,
            num_agents=int(self._num_agents),
        )

    def _orbit_tape_frame_payload(
        self,
        obs_all: list[dict[str, Any]],
        *,
        episode_step: int,
        episode_done: bool,
        player_value_baseline: list[float] | None,
        player_supervised_heads: dict[str, dict[str, Any]] | None,
        action_edges: list[dict[str, Any]],
        fleet_arrival_traces: list[dict[str, Any]],
        orbit_planet_feature_pack: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return orbit_tape_frame_dict_from_obs_all(
            obs_all,
            num_agents=int(self._num_agents),
            episode_step=int(episode_step),
            episode_done=bool(episode_done),
            player_value_baseline=player_value_baseline,
            player_supervised_heads=player_supervised_heads,
            action_edges=action_edges,
            fleet_arrival_traces=fleet_arrival_traces,
            orbit_planet_feature_pack=orbit_planet_feature_pack,
        )

    def _orbit_tape_recording_enabled(self) -> bool:
        return (
            bool(self._record_tape)
            and bool(self._visualize)
            and bool(self._visualize_sim_env)
            and self._visualization_queue is not None
        )

    def _flush_accumulated_orbit_tape(self) -> None:
        if not self._orbit_tape_snapshots:
            return
        if self._visualization_queue is not None and bool(self._visualize):
            frames: list[dict[str, Any]] = []
            for snap in self._orbit_tape_snapshots:
                assert len(snap) == 8
                obs_all = snap[0]
                ep_step = int(snap[1])
                ep_done = bool(snap[2])
                pvb = snap[3]
                psh = snap[4]
                action_edges = snap[5]
                fleet_arrival_traces = snap[6]
                orbit_planet_feature_pack = snap[7]
                assert pvb is None or isinstance(pvb, list)
                assert psh is None or isinstance(psh, dict)
                assert isinstance(action_edges, list)
                assert isinstance(fleet_arrival_traces, list)
                assert orbit_planet_feature_pack is None or isinstance(orbit_planet_feature_pack, dict)
                frames.append(
                    self._orbit_tape_frame_payload(
                        obs_all,
                        episode_step=ep_step,
                        episode_done=ep_done,
                        player_value_baseline=pvb,
                        player_supervised_heads=psh,
                        action_edges=action_edges,
                        fleet_arrival_traces=fleet_arrival_traces,
                        orbit_planet_feature_pack=orbit_planet_feature_pack,
                    )
                )
            envelope = {
                "version": int(_ORBIT_TAPE_ENVELOPE_VERSION),
                "frames": frames,
            }
            self._visualization_queue.put(
                json.dumps(envelope, separators=(",", ":"), sort_keys=False)
            )
        self._orbit_tape_snapshots.clear()

    def _maybe_record_orbit_tape_snapshot(
        self,
        obs_all: list[dict[str, Any]],
        *,
        episode_step: int,
        episode_done: bool,
        fleet_arrival_traces: list[dict[str, Any]],
        action_edges_fn: Callable[[], list[dict[str, Any]]] | None,
        orbit_planet_feature_pack_fn: Callable[[], dict[str, Any]] | None,
    ) -> None:
        w = self._wall_span
        with w("tape_record_enabled_check"):
            if not self._orbit_tape_recording_enabled():
                return
        with w("tape_action_edges"):
            action_edges = None if action_edges_fn is None else action_edges_fn()
        with w("tape_feature_pack"):
            orbit_planet_feature_pack = (
                None if orbit_planet_feature_pack_fn is None else orbit_planet_feature_pack_fn()
            )
        with w("tape_snapshot_append"):
            self._orbit_tape_snapshots.append(
                [
                    copy.deepcopy(obs_all),
                    int(episode_step),
                    bool(episode_done),
                    None,
                    None,
                    [] if action_edges is None else copy.deepcopy(action_edges),
                    copy.deepcopy(fleet_arrival_traces),
                    copy.deepcopy(orbit_planet_feature_pack),
                ]
            )

    def _orbit_episode_done_tensor(self) -> torch.Tensor:
        done_b = bool(self._env.done)
        v = 1.0 if done_b else 0.0
        done = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32, device=self._device())
        for compact_idx in range(self._num_agents):
            done[orbit_policy_slot_for_compact_agent(compact_idx, self._num_agents)] = v
        return done

    def _wall_span(self, name: str):
        p = self._wall_prof
        if p is None:
            return nullcontext()
        return p(name)

    def _reset_cpp_env_obs_full(self) -> dict[str, Any]:
        assert self._cpp_obs_stub is not None
        w = self._wall_span
        with w("cpp_preflight"):
            self._flush_accumulated_orbit_tape()
            episode_seed = self._sample_orbit_episode_seed_on_reset()
            reference_random_state = orbit_reference_upstream_random_derived_dict(
                seed=episode_seed,
                num_agents=self._num_agents,
                comet_speed=float(self._env.configuration.cometSpeed),
            )
            self._episode_step = 0
            self._orbit_policy_obs_invariance_check_disabled = False
            self._orbit_tape_value_hold = [0.0] * self._num_agents
        stub = self._cpp_obs_stub
        with w("cpp_stub_reset"):
            cpp_extra = stub.reset_from_external_random_state(
                random_state=reference_random_state,
                plain_state=_orbit_plain_minimal_for_cpp_reset_from_reference_random_state(
                    reference_random_state=reference_random_state,
                ),
            )
            if self._cpp_env_obs_validate:
                assert stub.last_cpp_reset_trace.splitlines()[0] == "01\treset"
            self._episode_step = self._cpp_clock_step_for_noop_and_dataset()
            if self._cpp_env_obs_validate:
                stub.cpp_assert_planets_match_noop_cache(int(self._episode_step))
            planet_rows_for_masks = stub.cpp_noop_trajectory_planets_row_tensor(
                int(self._episode_step)
            )
            planet_count_for_masks = int(cpp_extra["orbit_planet_mask_CPP"][0].sum().item())
            self._split_cpp_extra_available_masks(
                cpp_extra,
                planet_rows=planet_rows_for_masks,
                planet_count=planet_count_for_masks,
            )
            stub.refill_arrival_features_from_state(
                out=cpp_extra,
            )
        with w("cpp_prev_state"):
            self._orbit_episode_index = int(self._orbit_episode_index) + 1
            self._prev_fleet_by_seat = [
                stub.cpp_fleet_ship_total_int_for_owner(i) for i in range(self._num_agents)
            ]
            self._prev_alive_by_seat = [
                stub.cpp_player_alive_for_owner(i) for i in range(self._num_agents)
            ]
            self._orbit_elimination_game_result_credited = [False] * self._num_agents
            self._prev_planets_by_seat = [
                stub.cpp_planet_count_int_for_owner(i) for i in range(self._num_agents)
            ]
            self._prev_production_by_seat = [
                stub.cpp_production_sum_for_owner(i) for i in range(self._num_agents)
            ]
        with w("cpp_build_obs_metrics"):
            dev = self._device()
            z = torch.tensor(0.0, dtype=torch.float32, device=dev)
            fleet_zeros = [z.clone() for _ in range(self._num_agents)]
            step_zeros = [z.clone() for _ in range(self._num_agents)]
            planet_zeros = [z.clone() for _ in range(self._num_agents)]
            production_zeros = [z.clone() for _ in range(self._num_agents)]
            game_result_zeros = [z.clone() for _ in range(self._num_agents)]
            fleet_total_tensors = [
                torch.tensor(self._prev_fleet_by_seat[i], dtype=torch.float32, device=dev)
                for i in range(self._num_agents)
            ]
            production_total_tensors = [
                torch.tensor(self._prev_production_by_seat[i], dtype=torch.float32, device=dev)
                for i in range(self._num_agents)
            ]
            obs_raw: dict[str, Any] = {
                "episode_step": torch.full(
                    (ORBIT_PLAYER_AXIS_SLOTS,),
                    int(self._episode_step),
                    dtype=torch.int64,
                    device=dev,
                ),
                "orbit_fleet_total": _stack_policy_slot_scalar_metrics(
                    fleet_total_tensors, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "production_total": _stack_policy_slot_scalar_metrics(
                    production_total_tensors,
                    dtype=torch.float32,
                    device=dev,
                    num_agents=self._num_agents,
                ),
            }
            metrics: dict[str, Any] = {
                "orbit_fleet_delta": _stack_policy_slot_scalar_metrics(
                    fleet_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "env_step": _stack_policy_slot_scalar_metrics(
                    step_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "planets_delta": _stack_policy_slot_scalar_metrics(
                    planet_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "production_delta": _stack_policy_slot_scalar_metrics(
                    production_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "game_result": _stack_policy_slot_scalar_metrics(
                    game_result_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
            }
            terminal_b = stub.cpp_orbit_episode_terminal()
        with w("cpp_tape"):
            with w("cpp_tape_enabled_check"):
                tape_enabled = self._orbit_tape_recording_enabled()
            if tape_enabled:
                with w("cpp_tape_plain_obs"):
                    obs_all_tape = self._orbit_plain_obs_all_for_tape_from_cpp_stub(stub)
                with w("cpp_tape_fleet_hit_traces"):
                    fleet_arrival_traces = stub.cpp_fleet_hit_traces_from_state(
                        horizon=self._orbit_tape_fleet_hit_trace_horizon()
                    )
                with w("cpp_tape_record_snapshot"):
                    self._maybe_record_orbit_tape_snapshot(
                        obs_all_tape,
                        episode_step=0,
                        episode_done=terminal_b,
                        fleet_arrival_traces=fleet_arrival_traces,
                        action_edges_fn=lambda: self._orbit_tape_action_edges_from_policy_obs(
                            obs_all_tape,
                            self._orbit_policy_obs_from_cpp_extra(cpp_extra),
                        ),
                        orbit_planet_feature_pack_fn=lambda: self._orbit_tape_feature_pack_from_policy_obs(
                            obs_all_tape,
                            self._orbit_policy_obs_from_cpp_extra(cpp_extra),
                            stub.cpp_honest_shared_hit_kind_last(),
                            stub.cpp_honest_shared_intercept_fail_reason_last(),
                        ),
                    )
        out = {
            "obs_raw": obs_raw,
            "metrics": metrics,
            "orbit_episode_done": self._orbit_episode_done_tensor_from_cpp_terminal(terminal_b),
            **cpp_extra,
        }
        _orbit_fill_output_inactive_action_noops(out, num_agents=self._num_agents)
        return out

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        with self._wall_span("orbit_reset"):
            return self._orbit_reset_impl(**kwargs)

    def _orbit_reset_impl(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        if self._cpp_env_obs_full:
            return self._reset_cpp_env_obs_full()
        self._flush_accumulated_orbit_tape()
        episode_seed = self._sample_orbit_episode_seed_on_reset()
        reference_random_state = orbit_reference_upstream_random_derived_dict(
            seed=episode_seed,
            num_agents=self._num_agents,
            comet_speed=float(self._env.configuration.cometSpeed),
        )
        _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_reset_trace = []
        _orbit_wars_sim_detail_to_reset_trace("01\treset")
        _orbit_wars_sim_detail_to_reset_trace(
            "02\tangular_velocity\t"
            + str(
                _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_reset_trace_fmt_d(
                    float(reference_random_state["angular_velocity"])
                )
            )
        )
        _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_sim_subtrace_append = (
            _orbit_wars_sim_detail_to_reset_trace
        )
        assert hasattr(self._env.configuration, "seed"), type(self._env.configuration)
        self._env.configuration.seed = int(episode_seed)
        if getattr(self._env, "info", None) is not None:
            self._env.info.pop("seed", None)
        self._env.reset(self._num_agents)
        assert len(self._env.state) == self._num_agents
        _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_sim_subtrace_append = (
            _orbit_wars_step_trace_append
        )
        assert len(self._env.state) == self._num_agents
        self._orbit_episode_index = int(self._orbit_episode_index) + 1

        self._episode_step = 0
        self._orbit_policy_obs_invariance_check_disabled = False
        self._orbit_tape_value_hold = [0.0] * self._num_agents
        obs_all = _observations_all_plain(self._env)
        attach_zeros_action_taken_index_on_seats(obs_all)
        self._prev_fleet_by_seat = [fleet_total_seat(obs_all, i) for i in range(self._num_agents)]
        self._prev_alive_by_seat = [player_alive_seat(obs_all, i) for i in range(self._num_agents)]
        self._orbit_elimination_game_result_credited = [False] * self._num_agents
        self._prev_planets_by_seat = [planet_count_seat(obs_all, i) for i in range(self._num_agents)]
        self._prev_production_by_seat = [production_sum_seat(obs_all, i) for i in range(self._num_agents)]

        dev = self._device()
        z = torch.tensor(0.0, dtype=torch.float32, device=dev)
        fleet_zeros = [z.clone() for _ in range(self._num_agents)]
        step_zeros = [z.clone() for _ in range(self._num_agents)]
        planet_zeros = [z.clone() for _ in range(self._num_agents)]
        production_zeros = [z.clone() for _ in range(self._num_agents)]
        game_result_zeros = [z.clone() for _ in range(self._num_agents)]
        fleet_total_tensors = [
            torch.tensor(self._prev_fleet_by_seat[i], dtype=torch.float32, device=dev)
            for i in range(self._num_agents)
        ]
        production_total_tensors = [
            torch.tensor(self._prev_production_by_seat[i], dtype=torch.float32, device=dev)
            for i in range(self._num_agents)
        ]
        obs_raw: dict[str, Any] = {
            "episode_step": torch.full(
                (ORBIT_PLAYER_AXIS_SLOTS,),
                0,
                dtype=torch.int64,
                device=dev,
            ),
            "orbit_fleet_total": _stack_policy_slot_scalar_metrics(
                fleet_total_tensors, dtype=torch.float32, device=dev, num_agents=self._num_agents
            ),
            "production_total": _stack_policy_slot_scalar_metrics(
                production_total_tensors,
                dtype=torch.float32,
                device=dev,
                num_agents=self._num_agents,
            ),
        }
        player_obs = batched_player_obs_from_plain_seats(
            seats_plain=obs_all,
            device=dev,
            ship_speed=float(self._env.configuration.shipSpeed),
        )
        assert self._cpp_obs_stub is not None
        cpp_extra_from_stub = self._cpp_obs_stub.reset_from_external_random_state(
            random_state=reference_random_state,
            plain_state=_orbit_plain_minimal_for_cpp_reset_from_reference_random_state(
                reference_random_state=reference_random_state,
            ),
        )
        self._episode_step = self._cpp_clock_step_for_noop_and_dataset()
        obs_raw["episode_step"].fill_(int(self._episode_step))
        if self._cpp_env_obs_validate:
            self._reset_cpp_kaggle_cache_validator(
                obs_all,
                reference_random_state=reference_random_state,
            )
        planet_rows_for_masks = self._cpp_obs_stub.cpp_noop_trajectory_planets_row_tensor(
            int(self._episode_step)
        )
        planet_count_for_masks = int(cpp_extra_from_stub["orbit_planet_mask_CPP"][0].sum().item())
        self._split_cpp_extra_available_masks(
            cpp_extra_from_stub,
            planet_rows=planet_rows_for_masks,
            planet_count=planet_count_for_masks,
        )
        player_obs["available_action_mask"] = cpp_extra_from_stub[
            "available_action_mask_CPP"
        ].clone()
        if self._cpp_env_obs_validate:
            self._cpp_obs_stub.cpp_assert_planets_match_noop_cache(int(self._episode_step))
        player_obs[
            "fleet_arrival_features"
        ] = _orbit_fleet_arrival_features_from_plain_seats(
            seats_plain=obs_all,
            stub=self._cpp_obs_stub,
            device=dev,
        )
        metrics: dict[str, Any] = {
            "orbit_fleet_delta": _stack_policy_slot_scalar_metrics(
                fleet_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
            ),
            "env_step": _stack_policy_slot_scalar_metrics(
                step_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
            ),
            "planets_delta": _stack_policy_slot_scalar_metrics(
                planet_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
            ),
            "production_delta": _stack_policy_slot_scalar_metrics(
                production_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
            ),
            "game_result": _stack_policy_slot_scalar_metrics(
                game_result_zeros, dtype=torch.float32, device=dev, num_agents=self._num_agents
            ),
        }
        self._orbit_cpp_obs_validate_transition_was_reset = True
        self._orbit_cpp_obs_validate_last_py_digest_lines = list(
            _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_reset_trace
        )
        self._maybe_record_orbit_tape_snapshot(
            obs_all,
            episode_step=int(self._episode_step),
            episode_done=bool(self._env.done),
            fleet_arrival_traces=self._cpp_obs_stub.cpp_fleet_hit_traces_from_state(
                horizon=self._orbit_tape_fleet_hit_trace_horizon()
            ),
            action_edges_fn=lambda: self._orbit_tape_action_edges_from_policy_obs(
                obs_all,
                self._orbit_policy_obs_from_cpp_extra(cpp_extra_from_stub),
            ),
            orbit_planet_feature_pack_fn=lambda: self._orbit_tape_feature_pack_from_policy_obs(
                obs_all,
                self._orbit_policy_obs_from_cpp_extra(cpp_extra_from_stub),
                self._cpp_obs_stub.cpp_honest_shared_hit_kind_last(),
                self._cpp_obs_stub.cpp_honest_shared_intercept_fail_reason_last(),
            ),
        )
        if self._cpp_env_obs_validate:
            cpp_extra = cpp_extra_from_stub
            _orbit_wars_reset_trace_report_mismatch(
                _ORBIT_WARS_LOCAL_REFERENCE._orbit_wars_reset_trace,
                self._cpp_obs_stub.last_cpp_reset_trace,
            )
            self._orbit_assert_cpp_terminal_and_fleet_parity_vs_python(obs_all, metrics)
            self._orbit_assert_validate_live_and_noop_cache_rows_vs_python(
                obs_all,
                source="reset",
            )
            cpp_extra["available_action_mask_CPP"] = player_obs[
                "available_action_mask"
            ].clone()
            self._cpp_obs_stub.refill_arrival_features_from_state(
                out=cpp_extra,
            )
            _orbit_fill_output_inactive_action_noops(cpp_extra, num_agents=self._num_agents)
            self._assert_live_cpp_policy_obs_matches_external_plain_cpp(
                live_cpp_extra=cpp_extra,
                obs_all=obs_all,
                source="reset",
            )
            self._orbit_validate_policy_obs_invariance_or_disable(
                policy_obs=self._orbit_policy_obs_from_cpp_extra(cpp_extra),
                source="reset",
            )
        else:
            cpp_extra = self._policy_obs_cpp_extra_from_python(player_obs)
        out = {
            "obs_raw": obs_raw,
            "metrics": metrics,
            "orbit_episode_done": self._orbit_episode_done_tensor(),
            **cpp_extra,
        }
        _orbit_fill_output_inactive_action_noops(out, num_agents=self._num_agents)
        result = validated_dict_io_contract_output(
            self._flags, out, "orbit_wars_env_unpadded_output"
        )
        return result

    def _step_cpp_env_obs_full(
        self,
        *,
        policy_cls_by: list[torch.Tensor],
        executed_cls_by: list[torch.Tensor],
    ) -> dict[str, Any]:
        w = self._wall_span
        with w("cpp_full_preflight"):
            stub = self._cpp_obs_stub
            assert stub is not None
            assert int(stub._cpp_env.episode_step()) == int(self._episode_step), (
                int(stub._cpp_env.episode_step()),
                int(self._episode_step),
            )
            stub.apply_comet_sync_scheduled_for_current_cpp_clock(
                include_static_cache=False,
            )
            self._episode_step += 1
            assert int(self._episode_step) >= 1

        with w("cpp_stub_step"):
            with w("cpp_stub_action_classes"):
                action_classes = (
                    torch.stack(list(executed_cls_by), dim=0).to(dtype=torch.int32).contiguous()
                )
            with w("cpp_stub_step_cpp"):
                cpp_extra = stub.step(action_classes=action_classes)
                cpp_extra["action_taken_index_CPP"] = orbit_place_compact_agent_axis(
                    torch.stack(
                        [
                            tensor_action_taken_index_from_per_planet_classes(classes=cls)
                            for cls in policy_cls_by
                        ],
                        dim=0,
                    ).to(dtype=torch.int32),
                    num_agents=self._num_agents,
                )
            with w("cpp_stub_step_mask_override"):
                with w("cpp_stub_available_mask"):
                    planet_rows_for_masks = stub.cpp_noop_trajectory_planets_row_tensor(
                        int(self._episode_step)
                    )
                    planet_count_for_masks = int(cpp_extra["orbit_planet_mask_CPP"][0].sum().item())
                    self._split_cpp_extra_available_masks(
                        cpp_extra,
                        planet_rows=planet_rows_for_masks,
                        planet_count=planet_count_for_masks,
                    )
                with w("cpp_stub_refill_arrival_features"):
                    stub.refill_arrival_features_from_state(
                        out=cpp_extra,
                    )
            if self._cpp_env_obs_validate:
                with w("cpp_stub_clock_and_noop_asserts"):
                    assert int(self._episode_step) == self._cpp_clock_step_for_noop_and_dataset(), (
                        int(self._episode_step),
                        self._cpp_clock_step_for_noop_and_dataset(),
                    )
                    stub.cpp_assert_planets_match_noop_cache(int(self._episode_step))

        with w("cpp_build_metrics"):
            fleet_delta, planets_delta, production_delta, fleet_total = (
                stub.cpp_step_metric_tensors()
            )
            dev = self._device()
            production_total = torch.tensor(
                [stub.cpp_production_sum_for_owner(i) for i in range(self._num_agents)],
                dtype=torch.float32,
                device=dev,
            )
            obs_raw: dict[str, Any] = {
                "episode_step": torch.full(
                    (ORBIT_PLAYER_AXIS_SLOTS,),
                    int(self._episode_step),
                    dtype=torch.int64,
                    device=dev,
                ),
                "orbit_fleet_total": orbit_place_compact_agent_axis(
                    fleet_total.to(device=dev, dtype=torch.float32),
                    num_agents=self._num_agents,
                ),
                "production_total": orbit_place_compact_agent_axis(
                    production_total,
                    num_agents=self._num_agents,
                ),
            }
            metrics: dict[str, Any] = {
                "orbit_fleet_delta": orbit_place_compact_agent_axis(
                    fleet_delta.to(device=dev, dtype=torch.float32),
                    num_agents=self._num_agents,
                ),
                "env_step": orbit_place_compact_agent_axis(
                    torch.ones((self._num_agents,), dtype=torch.float32, device=dev),
                    num_agents=self._num_agents,
                ),
                "planets_delta": orbit_place_compact_agent_axis(
                    planets_delta.to(device=dev, dtype=torch.float32),
                    num_agents=self._num_agents,
                ),
                "production_delta": orbit_place_compact_agent_axis(
                    production_delta.to(device=dev, dtype=torch.float32),
                    num_agents=self._num_agents,
                ),
            }
        with w("cpp_game_result"):
            terminal_b = stub.cpp_orbit_episode_terminal()
            new_fleet_by_seat = [int(x) for x in fleet_total.tolist()]
            new_alive_by_seat = [
                stub.cpp_player_alive_for_owner(i) for i in range(self._num_agents)
            ]
            game_gr = self._orbit_game_result_from_fleet_transition(
                list(self._prev_alive_by_seat),
                new_alive_by_seat,
                new_fleet_by_seat,
                terminal_b,
            )
            metrics["game_result"] = orbit_place_compact_agent_axis(
                torch.tensor(game_gr, dtype=torch.float32, device=dev),
                num_agents=self._num_agents,
            )
            self._prev_fleet_by_seat = new_fleet_by_seat
            self._prev_alive_by_seat = new_alive_by_seat
        with w("cpp_tape"):
            with w("cpp_tape_enabled_check"):
                tape_enabled = self._orbit_tape_recording_enabled()
            if tape_enabled:
                with w("cpp_tape_plain_obs"):
                    obs_all_tape = self._orbit_plain_obs_all_for_tape_from_cpp_stub(stub)
                with w("cpp_tape_fleet_hit_traces"):
                    fleet_arrival_traces = stub.cpp_fleet_hit_traces_from_state(
                        horizon=self._orbit_tape_fleet_hit_trace_horizon()
                    )
                with w("cpp_tape_record_snapshot"):
                    self._maybe_record_orbit_tape_snapshot(
                        obs_all_tape,
                        episode_step=int(self._episode_step),
                        episode_done=terminal_b,
                        fleet_arrival_traces=fleet_arrival_traces,
                        action_edges_fn=lambda: self._orbit_tape_action_edges_from_policy_obs(
                            obs_all_tape,
                            self._orbit_policy_obs_from_cpp_extra(cpp_extra),
                        ),
                        orbit_planet_feature_pack_fn=lambda: self._orbit_tape_feature_pack_from_policy_obs(
                            obs_all_tape,
                            self._orbit_policy_obs_from_cpp_extra(cpp_extra),
                            stub.cpp_honest_shared_hit_kind_last(),
                            stub.cpp_honest_shared_intercept_fail_reason_last(),
                        ),
                    )
        with w("cpp_full_output_pack"):
            out = {
                "obs_raw": obs_raw,
                "metrics": metrics,
                "orbit_episode_done": self._orbit_episode_done_tensor_from_cpp_terminal(terminal_b),
                **cpp_extra,
            }
            _orbit_fill_output_inactive_action_noops(out, num_agents=self._num_agents)
        return out

    def step(self, actions: dict[str, Any]) -> dict[str, Any]:
        with self._wall_span("orbit_step"):
            return self._orbit_step_impl(actions)

    def _step_patch_supervised_tape_payload(
        self,
        *,
        tsp: Any,
        tst: Any,
        tsv: Any,
    ) -> None:
        assert isinstance(tsp, dict) and isinstance(tst, dict) and isinstance(tsv, dict), (
            "tape supervised payload requires prediction/target/valid dicts",
            type(tsp),
            type(tst),
            type(tsv),
        )
        assert self._orbit_tape_recording_enabled(), (
            f"{ORBIT_STEP_KEY_TAPE_SUPERVISED_PRED} / {ORBIT_STEP_KEY_TAPE_SUPERVISED_TARGET} / "
            f"{ORBIT_STEP_KEY_TAPE_SUPERVISED_VALID} "
            "require record_tape visualization queue"
        )
        assert len(self._orbit_tape_snapshots) >= 1
        obs_snap = self._orbit_tape_snapshots[-1][0]
        assert isinstance(obs_snap, list) and len(obs_snap) == self._num_agents
        head_names = sorted(set(tsp.keys()) | set(tst.keys()) | set(tsv.keys()))
        patched_heads: dict[str, dict[str, Any]] = {}
        for head_name in head_names:
            assert head_name in tsp and head_name in tst and head_name in tsv, (
                "supervised tape head must exist in prediction/target/valid dicts",
                head_name,
                sorted(tsp.keys()),
                sorted(tst.keys()),
                sorted(tsv.keys()),
            )
            pred_t = tsp[head_name]
            tgt_t = tst[head_name]
            valid_t = tsv[head_name]
            assert isinstance(pred_t, (torch.Tensor, list)) and isinstance(tgt_t, torch.Tensor) and isinstance(valid_t, torch.Tensor), (
                head_name,
                type(pred_t),
                type(tgt_t),
                type(valid_t),
            )
            if isinstance(pred_t, torch.Tensor):
                assert tuple(pred_t.shape) == (self._num_agents, ORBIT_MAX_PLANETS), (
                    head_name,
                    tuple(pred_t.shape),
                )
            else:
                assert len(pred_t) == self._num_agents, (
                    head_name,
                    len(pred_t),
                    self._num_agents,
                )
                for row in pred_t:
                    assert isinstance(row, list), (head_name, type(row))
                    assert len(row) == ORBIT_MAX_PLANETS, (
                        head_name,
                        len(row),
                        ORBIT_MAX_PLANETS,
                    )
            assert tuple(tgt_t.shape) == (self._num_agents, ORBIT_MAX_PLANETS), (
                head_name,
                tuple(tgt_t.shape),
            )
            assert tuple(valid_t.shape) == (self._num_agents, ORBIT_MAX_PLANETS), (
                head_name,
                tuple(valid_t.shape),
            )
            if isinstance(pred_t, torch.Tensor):
                assert pred_t.dtype == torch.float32, (head_name, pred_t.dtype)
                assert not pred_t.is_cuda, head_name
            assert tgt_t.dtype == torch.float32 and valid_t.dtype == torch.bool, (
                head_name,
                tgt_t.dtype,
                valid_t.dtype,
            )
            assert (not tgt_t.is_cuda) and (not valid_t.is_cuda), head_name
            hold = self._orbit_tape_supervised_hold.setdefault(
                str(head_name),
                {
                    "prediction": [[0.0] * ORBIT_MAX_PLANETS for _ in range(self._num_agents)],
                    "target": [[0.0] * ORBIT_MAX_PLANETS for _ in range(self._num_agents)],
                    "valid": [[False] * ORBIT_MAX_PLANETS for _ in range(self._num_agents)],
                },
            )
            hold_pred = hold["prediction"]
            hold_tgt = hold["target"]
            hold_valid = hold["valid"]
            assert isinstance(hold_pred, list) and isinstance(hold_tgt, list) and isinstance(hold_valid, list)
            assert len(hold_pred) == self._num_agents
            assert len(hold_tgt) == self._num_agents
            assert len(hold_valid) == self._num_agents
            patched_pred: list[list[Any]] = []
            patched_tgt: list[list[float]] = []
            patched_valid: list[list[bool]] = []
            for i in range(self._num_agents):
                assert len(hold_pred[i]) == ORBIT_MAX_PLANETS
                assert len(hold_tgt[i]) == ORBIT_MAX_PLANETS
                assert len(hold_valid[i]) == ORBIT_MAX_PLANETS
                if player_alive_seat(obs_snap, i):
                    row_pred: list[Any] = []
                    row_tgt: list[float] = []
                    row_valid: list[bool] = []
                    for j in range(ORBIT_MAX_PLANETS):
                        if isinstance(pred_t, torch.Tensor):
                            pv: Any = float(pred_t[i, j].item())
                        else:
                            pv = copy.deepcopy(pred_t[i][j])
                        tv = float(tgt_t[i, j].item())
                        vv = bool(valid_t[i, j].item())
                        hold_pred[i][j] = copy.deepcopy(pv)
                        hold_tgt[i][j] = tv
                        hold_valid[i][j] = vv
                        row_pred.append(pv)
                        row_tgt.append(tv)
                        row_valid.append(vv)
                    patched_pred.append(row_pred)
                    patched_tgt.append(row_tgt)
                    patched_valid.append(row_valid)
                else:
                    patched_pred.append(copy.deepcopy(hold_pred[i]))
                    patched_tgt.append([float(hold_tgt[i][j]) for j in range(ORBIT_MAX_PLANETS)])
                    patched_valid.append([bool(hold_valid[i][j]) for j in range(ORBIT_MAX_PLANETS)])
            patched_heads[str(head_name)] = {
                "prediction": patched_pred,
                "target": patched_tgt,
                "valid": patched_valid,
            }
        self._orbit_tape_snapshots[-1][4] = patched_heads

    def _orbit_step_impl(self, actions: dict[str, Any]) -> dict[str, Any]:
        w = self._wall_span
        with w("step_input_contract"):
            assert isinstance(actions, dict)
            maybe_validate_dict_io_contract_step_input(
                self._flags, actions, "orbit_wars_env_step_actions"
            )
        with w("step_action_payload_fetch"):
            tb = actions.get(ORBIT_STEP_KEY_TAPE_BASELINE_LEARN)
            tsp = actions.get(ORBIT_STEP_KEY_TAPE_SUPERVISED_PRED)
            tst = actions.get(ORBIT_STEP_KEY_TAPE_SUPERVISED_TARGET)
            tsv = actions.get(ORBIT_STEP_KEY_TAPE_SUPERVISED_VALID)
        with w("step_tape_baseline_patch"):
            if tb is not None:
                assert isinstance(tb, torch.Tensor)
                assert tuple(tb.shape) == (self._num_agents,) and tb.dtype == torch.float32
                assert not tb.is_cuda
                assert self._orbit_tape_recording_enabled(), (
                    f"{ORBIT_STEP_KEY_TAPE_BASELINE_LEARN} requires record_tape visualization queue"
                )
                assert len(self._orbit_tape_snapshots) >= 1
                obs_snap = self._orbit_tape_snapshots[-1][0]
                assert isinstance(obs_snap, list) and len(obs_snap) == self._num_agents
                patched: list[float] = []
                for i in range(self._num_agents):
                    if player_alive_seat(obs_snap, i):
                        vi = float(tb[i].item())
                        self._orbit_tape_value_hold[i] = vi
                        patched.append(vi)
                    else:
                        patched.append(float(self._orbit_tape_value_hold[i]))
                self._orbit_tape_snapshots[-1][3] = patched
        with w("step_tape_supervised_patch"):
            if tsp is not None or tst is not None or tsv is not None:
                self._step_patch_supervised_tape_payload(
                    tsp=tsp,
                    tst=tst,
                    tsv=tsv,
                )
        with w("step_action_contract"):
            cls_by = actions[ORBIT_STEP_KEY_ORBIT_PAIRWISE_CLASSES]
            assert isinstance(cls_by, list)
            assert len(cls_by) == self._num_agents
            for c in cls_by:
                assert isinstance(c, torch.Tensor)
                assert tuple(c.shape) == (ORBIT_PLANET_ACTION_SLOTS,)
                assert c.dtype == torch.int64
                assert not c.is_cuda
            policy_cls_by = cls_by
        if self._cpp_env_obs_full:
            return self._step_cpp_env_obs_full(
                policy_cls_by=policy_cls_by,
                executed_cls_by=policy_cls_by,
            )
        with w("python_step_preflight"):
            assert len(self._env.state) == self._num_agents
            assert self._cpp_obs_stub is not None
            assert int(self._cpp_obs_stub._cpp_env.episode_step()) == int(self._episode_step), (
                int(self._cpp_obs_stub._cpp_env.episode_step()),
                int(self._episode_step),
            )
            if self._cpp_env_obs_validate:
                assert self._cpp_kaggle_cache_stub is not None
                assert int(self._cpp_kaggle_cache_stub._cpp_env.episode_step()) == int(
                    self._episode_step
                ), (
                    int(self._cpp_kaggle_cache_stub._cpp_env.episode_step()),
                    int(self._episode_step),
                )
            obs_pre = _observations_all_plain(self._env)
            self._cpp_obs_stub.cpp_assert_planets_match_noop_cache(int(self._episode_step))
            batched_pre = batched_planet_geometry_only_from_plain_seats(
                seats_plain=obs_pre,
                device=torch.device("cpu"),
            )
            slot0_pre = orbit_policy_slot_for_compact_agent(0, self._num_agents)
            planet_rows_pre = batched_pre["planet_rows"][slot0_pre].to(
                dtype=torch.float32,
                device=torch.device("cpu"),
            ).contiguous()
            planet_count_pre = int(batched_pre["planet_count"][slot0_pre].item())
            honest_available_action_mask_by_seat = _orbit_available_action_mask_from_send_all_hit_mask_for_plain_seats(
                send_all_hit_mask=_orbit_honest_send_all_hit_mask_from_cpp_env(
                    self._cpp_obs_stub._cpp_env
                ),
                planet_rows=planet_rows_pre,
                planet_count=planet_count_pre,
                obs_all=obs_pre,
                num_agents=self._num_agents,
                device=torch.device("cpu"),
                wall_profiler=None,
            )
            honest_send_ships = _orbit_honest_send_ships_from_cpp_env(
                self._cpp_obs_stub._cpp_env
            )
            executed_cls_by = policy_cls_by
            honest_angle_source: Any = self._cpp_obs_stub
            if self._cpp_env_obs_validate:
                assert self._cpp_kaggle_cache_stub is not None
                honest_angle_source = _OrbitCppHonestAngleDualValidateSource(
                    self._cpp_obs_stub,
                    self._cpp_kaggle_cache_stub,
                )
        with w("python_step_action_decode"):
            kms = [
                kaggle_moves_for_seat_from_classes_honest_angle(
                    seat_plain=obs_pre[i],
                    classes=executed_cls_by[i],
                    ship_speed=float(self._env.configuration.shipSpeed),
                    honest_available_action_mask=honest_available_action_mask_by_seat[
                        orbit_policy_slot_for_compact_agent(i, self._num_agents)
                    ],
                    honest_send_ships=honest_send_ships,
                    honest_angle_source=honest_angle_source,
                )
                for i in range(self._num_agents)
            ]
            for a in kms:
                assert_orbit_action(a)
        with w("python_step_reference_step"):
            self._cpp_obs_stub.apply_comet_sync_scheduled_for_current_cpp_clock(
                include_static_cache=False,
            )
            if self._cpp_env_obs_validate:
                assert self._cpp_kaggle_cache_stub is not None
                self._cpp_kaggle_cache_stub.apply_comet_sync_scheduled_for_current_cpp_clock(
                    include_static_cache=False,
                )
            self._episode_step += 1
            assert int(self._episode_step) >= 1
            _ORBIT_WARS_STEP_TRACE.clear()
            _orbit_wars_step_trace_append(
                "10\tepisode_step\t" + str(int(self._episode_step))
            )
            self._env.step(list(kms))
            assert len(self._env.state) == self._num_agents
            obs_all = _observations_all_plain(self._env)
            _orbit_wars_step_trace_emit_planets("11", obs_all[0]["planets"])
            attach_action_taken_index_from_per_planet_classes(obs_all, policy_cls_by)
        with w("python_step_metric_precompute"):
            new_by_seat = [fleet_total_seat(obs_all, i) for i in range(self._num_agents)]
            new_alive_by_seat = [player_alive_seat(obs_all, i) for i in range(self._num_agents)]
            new_planets_by_seat = [planet_count_seat(obs_all, i) for i in range(self._num_agents)]
            new_production_by_seat = [production_sum_seat(obs_all, i) for i in range(self._num_agents)]
            deltas = [
                float(new_by_seat[i] - self._prev_fleet_by_seat[i]) for i in range(self._num_agents)
            ]
            planets_deltas = [
                float(new_planets_by_seat[i] - self._prev_planets_by_seat[i])
                for i in range(self._num_agents)
            ]
            production_deltas = [
                float(new_production_by_seat[i] - self._prev_production_by_seat[i])
                for i in range(self._num_agents)
            ]
            game_gr = self._orbit_game_result_from_fleet_transition(
                list(self._prev_alive_by_seat),
                new_alive_by_seat,
                new_by_seat,
                bool(self._env.done),
            )
        with w("python_step_cpp_stub_sync"):
            dev = self._device()
            action_classes = (
                torch.stack(list(executed_cls_by), dim=0).to(dtype=torch.int32).contiguous()
            )
            cpp_extra_from_stub = self._cpp_obs_stub.step(action_classes=action_classes)
            if self._cpp_env_obs_validate:
                assert self._cpp_kaggle_cache_stub is not None
                _ = self._cpp_kaggle_cache_stub.step(
                    action_classes=_orbit_noop_action_classes_for_agents(self._num_agents)
                )
            self._cpp_obs_stub.update_static_cache_comets_from_kaggle_plain(
                plain=obs_all[0],
                num_agents=self._num_agents,
            )
            if self._cpp_env_obs_validate:
                assert self._cpp_kaggle_cache_stub is not None
                self._cpp_kaggle_cache_stub.update_static_cache_comets_from_kaggle_plain(
                    plain=obs_all[0],
                    num_agents=self._num_agents,
                )
        with w("python_step_build_obs_metrics"):
            one = torch.tensor(1.0, dtype=torch.float32, device=dev)
            delta_tensors = [
                torch.tensor(deltas[i], dtype=torch.float32, device=dev) for i in range(self._num_agents)
            ]
            planets_delta_tensors = [
                torch.tensor(planets_deltas[i], dtype=torch.float32, device=dev)
                for i in range(self._num_agents)
            ]
            production_delta_tensors = [
                torch.tensor(production_deltas[i], dtype=torch.float32, device=dev)
                for i in range(self._num_agents)
            ]
            fleet_total_tensors = [
                torch.tensor(new_by_seat[i], dtype=torch.float32, device=dev)
                for i in range(self._num_agents)
            ]
            production_total_tensors = [
                torch.tensor(new_production_by_seat[i], dtype=torch.float32, device=dev)
                for i in range(self._num_agents)
            ]
            step_ones = [one.clone() for _ in range(self._num_agents)]
            obs_raw: dict[str, Any] = {
                "episode_step": torch.full(
                    (ORBIT_PLAYER_AXIS_SLOTS,),
                    int(self._episode_step),
                    dtype=torch.int64,
                    device=dev,
                ),
                "orbit_fleet_total": _stack_policy_slot_scalar_metrics(
                    fleet_total_tensors, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "production_total": _stack_policy_slot_scalar_metrics(
                    production_total_tensors,
                    dtype=torch.float32,
                    device=dev,
                    num_agents=self._num_agents,
                ),
            }
            player_obs = batched_player_obs_from_plain_seats(
                seats_plain=obs_all,
                device=dev,
                ship_speed=float(self._env.configuration.shipSpeed),
            )
            planet_rows_for_masks = self._cpp_obs_stub.cpp_noop_trajectory_planets_row_tensor(
                int(self._episode_step)
            )
            planet_count_for_masks = int(cpp_extra_from_stub["orbit_planet_mask_CPP"][0].sum().item())
            self._split_cpp_extra_available_masks(
                cpp_extra_from_stub,
                planet_rows=planet_rows_for_masks,
                planet_count=planet_count_for_masks,
            )
            player_obs["available_action_mask"] = cpp_extra_from_stub[
                "available_action_mask_CPP"
            ].clone()
            player_obs[
                "fleet_arrival_features"
            ] = _orbit_fleet_arrival_features_from_plain_seats(
                seats_plain=obs_all,
                stub=self._cpp_obs_stub,
                device=dev,
            )
            metrics: dict[str, Any] = {
                "orbit_fleet_delta": _stack_policy_slot_scalar_metrics(
                    delta_tensors, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "env_step": _stack_policy_slot_scalar_metrics(
                    step_ones, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "planets_delta": _stack_policy_slot_scalar_metrics(
                    planets_delta_tensors, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "production_delta": _stack_policy_slot_scalar_metrics(
                    production_delta_tensors, dtype=torch.float32, device=dev, num_agents=self._num_agents
                ),
                "game_result": _stack_policy_slot_scalar_metrics(
                    [
                        torch.tensor(game_gr[i], dtype=torch.float32, device=dev)
                        for i in range(self._num_agents)
                    ],
                    dtype=torch.float32,
                    device=dev,
                    num_agents=self._num_agents,
                ),
            }
            self._prev_fleet_by_seat = new_by_seat
            self._prev_alive_by_seat = new_alive_by_seat
            self._prev_planets_by_seat = new_planets_by_seat
            self._prev_production_by_seat = new_production_by_seat
        with w("python_step_tape_snapshot"):
            self._maybe_record_orbit_tape_snapshot(
                obs_all,
                episode_step=int(self._episode_step),
                episode_done=bool(self._env.done),
                fleet_arrival_traces=self._cpp_obs_stub.cpp_fleet_hit_traces_from_state(
                    horizon=self._orbit_tape_fleet_hit_trace_horizon()
                ),
                action_edges_fn=lambda: self._orbit_tape_action_edges_from_policy_obs(
                    obs_all,
                    self._orbit_policy_obs_from_cpp_extra(cpp_extra_from_stub),
                ),
                orbit_planet_feature_pack_fn=lambda: self._orbit_tape_feature_pack_from_policy_obs(
                    obs_all,
                    self._orbit_policy_obs_from_cpp_extra(cpp_extra_from_stub),
                    self._cpp_obs_stub.cpp_honest_shared_hit_kind_last(),
                    self._cpp_obs_stub.cpp_honest_shared_intercept_fail_reason_last(),
                ),
            )
        with w("python_step_validate_or_policy_obs"):
            self._orbit_cpp_obs_validate_transition_was_reset = False
            self._orbit_cpp_obs_validate_last_py_digest_lines = list(_ORBIT_WARS_STEP_TRACE)
            if self._cpp_env_obs_validate:
                assert self._cpp_obs_stub is not None
                cpp_extra = cpp_extra_from_stub
                _orbit_wars_step_trace_report_mismatch(
                    _ORBIT_WARS_STEP_TRACE,
                    self._cpp_obs_stub.last_cpp_step_trace,
                )
                assert int(self._episode_step) == self._cpp_clock_step_for_noop_and_dataset(), (
                    int(self._episode_step),
                    self._cpp_clock_step_for_noop_and_dataset(),
                )
                self._cpp_obs_stub.cpp_assert_planets_match_noop_cache(int(self._episode_step))
                self._update_cpp_kaggle_cache_validator_if_needed(obs_all)
                self._orbit_assert_cpp_terminal_and_fleet_parity_vs_python(obs_all, metrics)
                cpp_extra["available_action_mask_CPP"] = player_obs[
                    "available_action_mask"
                ].clone()
                self._cpp_obs_stub.refill_arrival_features_from_state(
                    out=cpp_extra,
                )
                _orbit_fill_output_inactive_action_noops(cpp_extra, num_agents=self._num_agents)
                self._assert_live_cpp_policy_obs_matches_external_plain_cpp(
                    live_cpp_extra=cpp_extra,
                    obs_all=obs_all,
                    source="step",
                )
                self._orbit_validate_policy_obs_invariance_or_disable(
                    policy_obs=self._orbit_policy_obs_from_cpp_extra(cpp_extra),
                    source="step",
                )
            else:
                assert self._cpp_obs_stub is not None
                cpp_extra = self._policy_obs_cpp_extra_from_python(player_obs)
        with w("python_step_output_pack"):
            out = {
                "obs_raw": obs_raw,
                "metrics": metrics,
                "orbit_episode_done": self._orbit_episode_done_tensor(),
                **cpp_extra,
            }
            _orbit_fill_output_inactive_action_noops(out, num_agents=self._num_agents)
            return validated_dict_io_contract_output(
                self._flags, out, "orbit_wars_env_unpadded_output"
            )


