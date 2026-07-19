"""Per-episode behavior cloning tensors from Kaggle Orbit Wars replays (C++ obs, RL move-class GT, BC loss mask).

Episodes on disk are **gzip**-wrapped ``torch.save`` streams (low ``compresslevel`` for fast load). Save/load
stream through ``gzip.open`` and the file object to avoid holding a full uncompressed ``bytes`` copy of the
episode.

For ``num_agents == 2``, Kaggle seats are stored in policy slots 0 and 3; slots 1 and 2 stay masked.

``bc_planet_ship_count`` is pre-step ships on each planet slot per policy seat (same slot order as ``planets``
in the paired observation).

``rl_per_planet_move_class`` matches RL per-source policy heads: shape ``[T, ORBIT_PLAYER_AXIS_SLOTS,
ORBIT_MAX_PLANETS]`` with class ``c = dst_slot * ORBIT_MOVE_CLASSES_PER_TARGET + amount_class``.

``replay_obs_step_before_action`` is the zero-based replay frame index before the transition;
``replay_transition_index`` is the ``steps`` row index where that transition's ``action`` is recorded.
Spawn edges use **pre-step** ``planets`` for source/destination ids. Replays may spawn multiple fleets from
one source planet to different targets in one step; RL allows only one. Those cases are logged and the edge
with the largest ship count is kept (ties: smaller ``dst_slot``).
"""
from __future__ import annotations

import gzip
import logging
import math
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# gzip ``compresslevel`` 1..9 (1 = fast encode/decode, weaker ratio).
_BC_DATASET_GZIP_COMPRESSLEVEL = 1
_BC_EPISODE_PT_SERIALIZATION = (
    f"torch_save_gzip_compresslevel_{_BC_DATASET_GZIP_COMPRESSLEVEL}_compact_active_bc_rows"
)
_BC_PT_FILE_GZIP_MAGIC = b"\x1f\x8b"
_BC_POLICY_INPUT_DUMP_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
)
_BC_POLICY_INPUT_DUMP_FORMAT = "orbit_policy_network_inputs_dump_v1"


def _atomic_write_bc_episode_gzip(final_path: Path, payload: dict[str, Any]) -> None:
    """Stream ``torch.save`` into a gzip file, then ``replace`` into place (no full uncompressed ``bytes``)."""
    final_path = final_path.resolve()
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=f".tmp.{os.getpid()}",
        dir=str(parent),
    )
    os.close(fd)
    try:
        with gzip.open(
            tmp_path,
            "wb",
            compresslevel=_BC_DATASET_GZIP_COMPRESSLEVEL,
        ) as zf:
            torch.save(payload, zf)
        with open(tmp_path, "r+b") as syncf:
            os.fsync(syncf.fileno())
    except BaseException:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)
        raise
    os.replace(tmp_path, final_path)


def load_behavior_clone_episode_pt(path: Path) -> dict[str, Any]:
    """Load a BC episode written by ``write_behavior_clone_episode_pt`` (gzip-wrapped ``torch.save`` only)."""
    path = path.resolve()
    assert path.is_file(), path
    with open(path, "rb") as f:
        head = f.read(2)
        assert head == _BC_PT_FILE_GZIP_MAGIC, (path, head)
        f.seek(0)
        with gzip.open(f, "rb") as gf:
            payload = torch.load(gf, map_location="cpu", weights_only=False)
        assert isinstance(payload, dict), type(payload)
        ser = payload.get("serialization")
        assert ser == _BC_EPISODE_PT_SERIALIZATION, (path, ser)
    return payload


from .obs_wrapper import (
    ORBIT_EDGE_FEATURES,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX,
    ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ARRIVAL_FEATURES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
    ORBIT_POLICY_OBS_KEYS,
    _orbit_edge_feature_channel_name,
    _orbit_planet_feature_channel_name,
    orbit_active_policy_slots,
    orbit_move_ship_count,
    orbit_policy_slot_for_compact_agent,
)
from .orbit_kaggle_cpp_cache import OrbitKaggleCppObservationCache
from .orbit_kaggle_replay_fleet_gt import (
    _new_spawn_fleet_rows,
    assert_replay_frame_contract,
    replay_observation_to_plain,
)
from .orbit_kaggle_replay_tape import kaggle_replay_obs_all_from_plain
from .wall_tree_profiler import WallTreeProfiler, profiler_span
from .orbit_wars_env import (
    _ORBIT_PLANET_ROW_LEN,
    attach_zeros_action_taken_index_on_seats,
    fleet_ship_count_for_player,
    player_alive_for_player,
    production_sum_for_player,
)
from .reward_wrapper import (
    _EARLY_GAME_RESULT_STEP_CAP,
    _ESTIMATED_POWER_SMOOTH_STEPS,
    estimated_power_early_stop_triggered,
    estimated_power_from_raw,
)

_LOG = logging.getLogger(__name__)

# Human dump: full named feature decode for rows where seat-0 ``step`` matches this value.
BC_HUMAN_DUMP_FULL_DECODE_REPLAY_OBS_STEP = 10

@dataclass
class BcFleetStats:
    spawned_fleets: int = 0
    resolved_hit_fleets: int = 0
    no_hit_fleets: int = 0
    hit_steps_histogram: dict[int, int] = field(default_factory=dict)
    ship_count_histogram: dict[int, int] = field(default_factory=dict)
    source_remaining_ship_count_histogram: dict[int, int] = field(default_factory=dict)
    source_sent_ship_percent_bucket_histogram: dict[int, int] = field(default_factory=dict)

    def _record_ships(self, *, ships: int, source_ships: int) -> None:
        n = int(ships)
        assert n >= 1, n
        cap = int(source_ships)
        assert cap >= n, (cap, n)
        self.ship_count_histogram[n] = self.ship_count_histogram.get(n, 0) + 1
        remaining = cap - n
        self.source_remaining_ship_count_histogram[remaining] = (
            self.source_remaining_ship_count_histogram.get(remaining, 0) + 1
        )
        percent_bucket = (n * 100 + cap - 1) // cap
        assert 1 <= percent_bucket <= 100, (percent_bucket, n, cap)
        self.source_sent_ship_percent_bucket_histogram[percent_bucket] = (
            self.source_sent_ship_percent_bucket_histogram.get(percent_bucket, 0) + 1
        )

    def record_no_hit(self, *, ships: int, source_ships: int) -> None:
        self.spawned_fleets += 1
        self.no_hit_fleets += 1
        self._record_ships(ships=int(ships), source_ships=int(source_ships))

    def record_hit(self, *, hit_steps: int, ships: int, source_ships: int) -> None:
        hs = int(hit_steps)
        assert hs >= 1, hs
        self.spawned_fleets += 1
        self.resolved_hit_fleets += 1
        self.hit_steps_histogram[hs] = self.hit_steps_histogram.get(hs, 0) + 1
        self._record_ships(ships=int(ships), source_ships=int(source_ships))


@dataclass(frozen=True)
class BcReplayPassContext:
    steps: list[Any]
    cfg: Mapping[str, Any]
    n_agents: int
    winner_seats: frozenset[int] | None
    filter_seats: frozenset[int] | None
    hit_horizon: int
    replay_episode_id: int
    wall_profiler: WallTreeProfiler | None
    profile_summary_every_steps: int
    fleet_stats: BcFleetStats | None


@dataclass(frozen=True)
class BcRowKey:
    transition_index: int
    replay_obs_step_before_action: int
    policy_slot: int


@dataclass(frozen=True)
class BcObsPassRow:
    key: BcRowKey
    obs: dict[str, torch.Tensor]
    ship_count: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True)
class BcGtPassRow:
    key: BcRowKey
    rl_class: torch.Tensor
    rl_loss_source_mask: torch.Tensor
    honest_hit_debug_by_src_dst: dict[tuple[int, int], list[dict[str, int | float]]]


@dataclass(frozen=True)
class BcEarlyStopInfo:
    transition_index: int
    replay_obs_step: int
    heuristic_winner_seats: frozenset[int]


class BcEpisodeEmptyLossMaskError(Exception):
    """Every BC timestep has zero ``bc_loss_player_mask`` after applying filters."""


def _replay_winner_seats(*, ep: Mapping[str, Any], n_agents: int) -> frozenset[int]:
    """Seats with final envelope ``reward == 1`` (Orbit Wars reference ``orbit_wars.py`` terminal block)."""
    na = int(n_agents)
    rewards = ep["rewards"]
    assert isinstance(rewards, (list, tuple)), type(rewards)
    assert len(rewards) == na, (len(rewards), na)
    return frozenset(i for i in range(na) if float(rewards[i]) == 1.0)


def kaggle_replay_team_names(ep: Mapping[str, Any]) -> tuple[str, ...]:
    """``ep['info']['TeamNames']`` in seat order (length ``len(ep['steps'][0])``)."""
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 1, len(steps)
    row0 = steps[0]
    assert isinstance(row0, list)
    na = len(row0)
    info = ep["info"]
    assert isinstance(info, Mapping)
    raw = info["TeamNames"]
    assert isinstance(raw, (list, tuple)), type(raw)
    assert len(raw) == na, (len(raw), na, raw)
    out: list[str] = []
    for x in raw:
        assert isinstance(x, str), type(x)
        out.append(x)
    return tuple(out)


def episode_matches_teams_filter(ep: Mapping[str, Any], teams_filter: Sequence[str]) -> bool:
    """True if ``teams_filter`` is empty, else if some filter string equals some ``TeamNames`` entry (exact)."""
    if len(teams_filter) == 0:
        return True
    for s in teams_filter:
        assert isinstance(s, str), type(s)
        assert len(s) > 0, s
    names = set(kaggle_replay_team_names(ep))
    want = set(teams_filter)
    return bool(names & want)


def _replay_filter_seats(*, ep: Mapping[str, Any], teams_filter: tuple[str, ...]) -> frozenset[int]:
    """In-game seats whose ``TeamNames`` string is in ``teams_filter`` (exact)."""
    assert len(teams_filter) > 0
    want = set(teams_filter)
    names = kaggle_replay_team_names(ep)
    out = [p for p, nm in enumerate(names) if nm in want]
    assert len(out) >= 1, (names, teams_filter)
    return frozenset(out)


def is_kaggle_replay_episode_json_path(path: Path) -> bool:
    """``replay_<episode_id>.json`` or ``replay_<episode_id>.json.gz`` (basename only; no disk read)."""
    name = path.name
    if name.endswith(".json.gz"):
        stem = name[: -len(".json.gz")]
    elif name.endswith(".json"):
        stem = name[: -len(".json")]
    else:
        return False
    if not stem.startswith("replay_"):
        return False
    suffix = stem[len("replay_") :]
    return len(suffix) > 0 and suffix.isdigit()


def iter_episode_json_files(episodes_dir: Path) -> Iterator[Path]:
    assert episodes_dir.is_dir(), episodes_dir
    out: list[Path] = []
    for p in episodes_dir.iterdir():
        if not p.is_file():
            continue
        if not is_kaggle_replay_episode_json_path(p):
            continue
        out.append(p)
    out.sort(key=lambda x: x.name)
    yield from out


def episode_pt_stem(path: Path) -> str:
    if path.name.endswith(".json.gz"):
        return path.name[: -len(".json.gz")]
    if path.suffix == ".json":
        return path.name[: -len(".json")]
    raise AssertionError(path)


def kaggle_replay_episode_id_from_path(path: Path) -> int:
    stem = episode_pt_stem(path.resolve())
    assert stem.startswith("replay_"), (path, stem)
    suffix = stem[len("replay_") :]
    assert suffix.isdigit(), (path, stem)
    return int(suffix)


def bc_loss_player_mask_vector(
    *,
    plain: dict[str, Any],
    n_agents: int,
    winner_seats: frozenset[int] | None = None,
    filter_seats: frozenset[int] | None = None,
) -> torch.Tensor:
    """Per policy slot: 1 only if the compact Kaggle seat is in-game.

    In 2p, compact seat 1 is stored in policy slot 3; slots 1 and 2 stay zero.

    If ``winner_seats`` is set, only those in-game seats may be non-zero.
    If ``filter_seats`` is set, only those in-game seats may be non-zero after ``winner_seats``.
    """
    na = int(n_agents)
    assert na in (2, 4), na
    planets = plain["planets"]
    fleets = plain["fleets"]
    assert isinstance(planets, list) and isinstance(fleets, list)
    out = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32)
    for compact_idx in range(na):
        alive = player_alive_for_player(planets, fleets, compact_idx)

        # Optional episode filters only narrow the in-game seats eligible for BC.
        winner_ok = winner_seats is None or compact_idx in winner_seats
        filter_ok = filter_seats is None or compact_idx in filter_seats
        slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
        out[slot] = 1.0 if alive and winner_ok and filter_ok else 0.0
    return out


def _planet_id_to_slot(*, planets: list[Any]) -> dict[int, int]:
    assert isinstance(planets, list)
    assert len(planets) <= ORBIT_MAX_PLANETS, len(planets)
    out: dict[int, int] = {}
    for slot, row in enumerate(planets):
        assert isinstance(row, (list, tuple)) and len(row) >= 1
        pid = int(row[0])
        assert pid not in out, (pid, slot, len(out))
        out[pid] = int(slot)
    return out


def _launch_fleet_rows_from_spawn_rows_pre_planets(
    *,
    new_spawn_rows: list[list[Any]],
    planets_pre_step: list[Any],
) -> list[list[Any]]:
    planet_by_id: dict[int, list[Any]] = {}
    for row in planets_pre_step:
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        pid = int(row[0])
        assert pid not in planet_by_id, pid
        planet_by_id[pid] = list(row)
    out: list[list[Any]] = []
    for frow in new_spawn_rows:
        assert isinstance(frow, list) and len(frow) == 7
        from_pid = int(frow[5])
        assert from_pid in planet_by_id, (from_pid, sorted(planet_by_id.keys()))
        src = planet_by_id[from_pid]
        angle = float(frow[4])
        radius = float(src[4])
        out.append(
            [
                int(frow[0]),
                int(frow[1]),
                float(src[2]) + math.cos(angle) * (radius + 0.1),
                float(src[3]) + math.sin(angle) * (radius + 0.1),
                angle,
                from_pid,
                float(frow[6]),
            ]
        )
    return out


def _spawn_edges_total_ships_by_dst(
    *,
    new_spawn_rows: list[list[Any]],
    cache: OrbitKaggleCppObservationCache,
    hit_horizon: int,
    planet_id_to_slot_pre_step: Mapping[int, int],
    planets_pre_step: list[Any],
    n_agents: int,
    active_policy_slots: tuple[int, ...],
    fleet_stats: BcFleetStats | None,
) -> dict[tuple[int, int], dict[int, int]]:
    """``(owner, src_slot) -> {dst_slot: total_ships}`` for new spawns with a resolved hit."""
    hz = int(hit_horizon)
    assert hz >= 1
    na = int(n_agents)
    assert na in (2, 4), na
    keep_slots = frozenset(int(slot) for slot in active_policy_slots)
    assert len(keep_slots) == len(active_policy_slots), active_policy_slots
    id_pre = planet_id_to_slot_pre_step
    launch_rows = _launch_fleet_rows_from_spawn_rows_pre_planets(
        new_spawn_rows=new_spawn_rows,
        planets_pre_step=planets_pre_step,
    )
    assert len(launch_rows) == len(new_spawn_rows), (len(launch_rows), len(new_spawn_rows))
    source_ships_by_planet_id: dict[int, int] = {}
    for row in planets_pre_step:
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        pid = int(row[0])
        assert pid not in source_ships_by_planet_id, pid
        source_ships_by_planet_id[pid] = int(round(float(row[5])))
    out: dict[tuple[int, int], dict[int, int]] = {}
    for frow in launch_rows:
        assert isinstance(frow, list) and len(frow) == 7
        owner = int(frow[1])
        assert 0 <= owner < na, (owner, na)
        policy_slot = orbit_policy_slot_for_compact_agent(owner, na)
        if policy_slot not in keep_slots:
            continue
        from_pid = int(frow[5])
        ships = int(round(float(frow[6])))
        if ships <= 0:
            continue
        assert from_pid in source_ships_by_planet_id, (from_pid, sorted(source_ships_by_planet_id.keys()))
        source_ships = int(source_ships_by_planet_id[from_pid])
        assert ships <= source_ships, (from_pid, ships, source_ships)
        traces = cache.fleet_hit_traces_from_plain_fleets(fleets=[frow], horizon=hz)
        if len(traces) == 0:
            if fleet_stats is not None:
                fleet_stats.record_no_hit(ships=ships, source_ships=source_ships)
            continue
        assert len(traces) == 1, traces
        tr = traces[0]
        if fleet_stats is not None:
            fleet_stats.record_hit(
                hit_steps=int(tr["hit_steps"]),
                ships=ships,
                source_ships=source_ships,
            )
        hit_pid = int(tr["hit_planet_id"])
        assert from_pid in id_pre, (from_pid, sorted(id_pre.keys()))
        if hit_pid not in id_pre:
            continue
        si = id_pre[from_pid]
        sj = int(id_pre[hit_pid])
        key = (owner, si)
        bucket = out.setdefault(key, {})
        bucket[sj] = bucket.get(sj, 0) + ships
    return out


def _rl_per_planet_move_class_from_spawn_edges(
    *,
    edges_by_owner_src: dict[tuple[int, int], dict[int, int]],
    ships_bpn: torch.Tensor,
    frame_idx: int,
    replay_episode_id: int,
    n_agents: int,
    active_policy_slots: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    na = int(n_agents)
    assert na in (2, 4), na
    assert len(active_policy_slots) >= 1, active_policy_slots
    row_by_policy_slot = {int(slot): i for i, slot in enumerate(active_policy_slots)}
    assert len(row_by_policy_slot) == len(active_policy_slots), active_policy_slots
    nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)
    pc = int(ORBIT_PER_PLANET_MOVE_CLASSES)
    idx = torch.arange(ORBIT_MAX_PLANETS, dtype=torch.int64)
    out = (idx * nb).unsqueeze(0).expand(len(active_policy_slots), -1).clone()
    loss_source_mask = torch.ones((len(active_policy_slots), ORBIT_MAX_PLANETS), dtype=torch.float32)
    assert ships_bpn.shape == (len(active_policy_slots), ORBIT_MAX_PLANETS), ships_bpn.shape
    for (owner, si), by_dst in edges_by_owner_src.items():
        if not by_dst:
            continue
        if len(by_dst) > 1:
            _LOG.warning(
                "BC_RL_GT_multiple_destinations_one_src replay=%s frame=%s owner=%s src_slot=%s edges=%s",
                int(replay_episode_id),
                int(frame_idx),
                int(owner),
                int(si),
                sorted(by_dst.items()),
            )
        sj, ships = max(by_dst.items(), key=lambda kv: (kv[1], -kv[0]))
        if ships <= 0:
            continue
        policy_slot = orbit_policy_slot_for_compact_agent(owner, na)
        assert policy_slot in row_by_policy_slot, (policy_slot, active_policy_slots)
        row_idx = row_by_policy_slot[policy_slot]
        cap = int(round(float(ships_bpn[row_idx, int(si)].item())))
        assert ships <= cap, (owner, si, sj, ships, cap, by_dst)
        send_all_ships = orbit_move_ship_count(cap, ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX)
        send_half_ships = orbit_move_ship_count(cap, ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX)
        if ships == send_all_ships or ships == send_all_ships - 1:
            amount_class = int(ORBIT_MOVE_CLASS_SEND_ALL_SUBINDEX)
        elif ships == send_half_ships:
            amount_class = int(ORBIT_MOVE_CLASS_SEND_HALF_SUBINDEX)
        else:
            loss_source_mask[row_idx, int(si)] = 0.0
            continue
        c = int(sj) * nb + amount_class
        assert 0 <= c < pc, (c, owner, si, sj, ships)
        out[row_idx, int(si)] = c
    return out, loss_source_mask


def bc_planet_ship_count_bpn(
    *,
    plain: dict[str, Any],
    n_agents: int,
    active_policy_slots: tuple[int, ...],
) -> torch.Tensor:
    """Per active policy seat and planet slot: ships on that body if owner matches seat, else 0."""
    na = int(n_agents)
    assert na in (2, 4), na
    assert len(active_policy_slots) >= 1, active_policy_slots
    row_by_policy_slot = {int(slot): i for i, slot in enumerate(active_policy_slots)}
    assert len(row_by_policy_slot) == len(active_policy_slots), active_policy_slots
    planets = plain["planets"]
    assert isinstance(planets, list)
    out = torch.zeros((len(active_policy_slots), ORBIT_MAX_PLANETS), dtype=torch.float32)
    for slot, row in enumerate(planets):
        if slot >= ORBIT_MAX_PLANETS:
            break
        assert isinstance(row, (list, tuple)) and len(row) == _ORBIT_PLANET_ROW_LEN
        owner = int(row[1])
        ships = float(row[5])
        if 0 <= owner < na:
            policy_slot = orbit_policy_slot_for_compact_agent(owner, na)
            if policy_slot in row_by_policy_slot:
                out[row_by_policy_slot[policy_slot], slot] = ships
    return out


def _assert_rl_gt_send_le_planet_ships(
    *,
    edges_by_owner_src: dict[tuple[int, int], dict[int, int]],
    ships_bpn: torch.Tensor,
    n_agents: int,
    active_policy_slots: tuple[int, ...],
) -> None:
    na = int(n_agents)
    assert na in (2, 4), na
    assert ships_bpn.shape == (len(active_policy_slots), ORBIT_MAX_PLANETS), (
        ships_bpn.shape,
        active_policy_slots,
    )
    row_by_policy_slot = {int(slot): i for i, slot in enumerate(active_policy_slots)}
    assert len(row_by_policy_slot) == len(active_policy_slots), active_policy_slots
    for (owner, si), by_dst in edges_by_owner_src.items():
        if not by_dst:
            continue
        if len(by_dst) > 1:
            sj, ships = max(by_dst.items(), key=lambda kv: (kv[1], -kv[0]))
        else:
            (sj, ships) = next(iter(by_dst.items()))
        policy_slot = orbit_policy_slot_for_compact_agent(owner, na)
        assert policy_slot in row_by_policy_slot, (policy_slot, active_policy_slots)
        cap = int(round(float(ships_bpn[row_by_policy_slot[policy_slot], int(si)].item())))
        assert ships <= cap, (owner, si, sj, ships, cap, by_dst)


def _assert_rl_gt_matches_available_action_mask(
    *,
    available_action_mask: torch.Tensor,
    planet_mask: torch.Tensor,
    player_mask: torch.Tensor,
    rl_cls: torch.Tensor,
    hit_horizon: int,
    honest_hit_debug_by_src_dst: dict[tuple[int, int], list[dict[str, int | float]]],
    replay_episode_id: int,
    policy_slot: int,
    frame_idx: int,
    replay_obs_step_before_action: int,
) -> None:
    assert isinstance(available_action_mask, torch.Tensor)
    assert isinstance(planet_mask, torch.Tensor)
    assert isinstance(player_mask, torch.Tensor)
    assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
    assert tuple(available_action_mask.shape) == (
        1,
        ORBIT_MAX_PLANETS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), available_action_mask.shape
    _ = int(hit_horizon)
    assert torch.all(available_action_mask >= 0), (
        int(available_action_mask.min().item()),
        int(available_action_mask.max().item()),
    )
    assert torch.all(available_action_mask <= 1), int(available_action_mask.max().item())
    assert planet_mask.shape == (1, ORBIT_MAX_PLANETS), planet_mask.shape
    assert player_mask.shape == (1,), player_mask.shape
    assert rl_cls.shape == (1, ORBIT_MAX_PLANETS), rl_cls.shape
    tgt = rl_cls.to(dtype=torch.long)
    assert torch.all((0 <= tgt) & (tgt < ORBIT_PER_PLANET_MOVE_CLASSES)), (
        int(tgt.min().item()),
        int(tgt.max().item()),
        ORBIT_PER_PLANET_MOVE_CLASSES,
    )
    target_ok = (
        torch.gather(available_action_mask, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1) > 0
    )
    active_player = player_mask > 0.5
    nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)
    dst_slot_bpn = tgt // nb
    ship_subindex_bpn = tgt % nb
    src_slot_bpn = torch.arange(ORBIT_MAX_PLANETS, dtype=torch.long).view(1, ORBIT_MAX_PLANETS)
    self_send_non_noop = (dst_slot_bpn == src_slot_bpn) & (ship_subindex_bpn > 0)
    bad = (~target_ok) & active_player.unsqueeze(-1) & (~self_send_non_noop)
    if not bool(torch.any(bad).item()):
        return
    bi = bad.nonzero(as_tuple=False)[0]
    row = int(bi[0].item())
    assert row == 0, row
    src_slot = int(bi[1].item())
    cls = int(tgt[row, src_slot].item())
    dst_slot = cls // nb
    amount_class = cls % nb
    avail_count = int((available_action_mask[row, src_slot] > 0).sum().item())
    dst_base = int(dst_slot) * nb
    dst_avail_amount_classes = torch.nonzero(
        available_action_mask[row, src_slot, dst_base : dst_base + nb] > 0,
        as_tuple=False,
    ).flatten()
    payload: dict[str, Any] = {
        "replay_episode_id": int(replay_episode_id),
        "frame_idx": int(frame_idx),
        "replay_obs_step_before_action": int(replay_obs_step_before_action),
        "policy_slot": policy_slot,
        "src_slot": src_slot,
        "class": cls,
        "dst_slot": int(dst_slot),
        "amount_class": int(amount_class),
        "available_count_for_src": avail_count,
        "available_amount_classes_for_dst": [
            int(x) for x in dst_avail_amount_classes.tolist()
        ],
        "src_planet_mask": float(planet_mask[row, src_slot].item()),
        "dst_planet_mask": float(planet_mask[row, int(dst_slot)].item()),
        "player_mask": float(player_mask[row].item()),
    }
    _ = honest_hit_debug_by_src_dst
    _LOG.warning("BC_RL_GT_unavailable_in_available_action_mask %s", payload)


def _bc_active_policy_slots_for_plain(
    *,
    plain: dict[str, Any],
    n_agents: int,
    winner_seats: frozenset[int] | None,
    filter_seats: frozenset[int] | None,
) -> tuple[int, ...]:
    loss_vec = bc_loss_player_mask_vector(
        plain=plain,
        n_agents=n_agents,
        winner_seats=winner_seats,
        filter_seats=filter_seats,
    )
    assert loss_vec.shape == (ORBIT_PLAYER_AXIS_SLOTS,), loss_vec.shape
    assert torch.all((loss_vec >= 0.0) & (loss_vec <= 1.0)), (loss_vec.min(), loss_vec.max())
    return tuple(int(x) for x in torch.nonzero(loss_vec > 0.5, as_tuple=False).flatten().tolist())


def _bc_replay_first_plain(ctx: BcReplayPassContext) -> dict[str, Any]:
    row0 = ctx.steps[0]
    assert isinstance(row0, list) and len(row0) == ctx.n_agents
    st00 = row0[0]
    assert isinstance(st00, dict)
    obs00 = st00["observation"]
    assert isinstance(obs00, dict)
    return replay_observation_to_plain(obs00, replay_step=0)


def _bc_policy_slot_scalar_tensor(values_by_seat: Sequence[float], *, n_agents: int) -> torch.Tensor:
    na = int(n_agents)
    assert na in (2, 4), na
    assert len(values_by_seat) == na, (len(values_by_seat), na)
    out = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32)
    for compact_idx, value in enumerate(values_by_seat):
        slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
        out[slot] = float(value)
    return out


def _bc_estimated_power_raw_from_plain(
    *,
    plain: dict[str, Any],
    n_agents: int,
) -> dict[str, torch.Tensor]:
    na = int(n_agents)
    assert na in (2, 4), na
    planets = plain["planets"]
    fleets = plain["fleets"]
    assert isinstance(planets, list) and isinstance(fleets, list)
    fleet_total = [
        float(fleet_ship_count_for_player(planets, fleets, compact_idx))
        for compact_idx in range(na)
    ]
    production_total = [
        float(production_sum_for_player(planets, compact_idx)) for compact_idx in range(na)
    ]
    return {
        "orbit_fleet_total": _bc_policy_slot_scalar_tensor(fleet_total, n_agents=na),
        "production_total": _bc_policy_slot_scalar_tensor(production_total, n_agents=na),
    }


def _bc_alive_by_seat(*, plain: dict[str, Any], n_agents: int) -> list[bool]:
    na = int(n_agents)
    assert na in (2, 4), na
    planets = plain["planets"]
    fleets = plain["fleets"]
    assert isinstance(planets, list) and isinstance(fleets, list)
    return [player_alive_for_player(planets, fleets, compact_idx) for compact_idx in range(na)]


def _bc_current_game_result_from_alive_transition(
    *,
    prev_alive_by_seat: list[bool],
    new_alive_by_seat: list[bool],
    already_credited_by_slot: list[bool],
    n_agents: int,
) -> torch.Tensor:
    na = int(n_agents)
    assert na in (2, 4), na
    assert len(prev_alive_by_seat) == na, (len(prev_alive_by_seat), na)
    assert len(new_alive_by_seat) == na, (len(new_alive_by_seat), na)
    assert len(already_credited_by_slot) == ORBIT_PLAYER_AXIS_SLOTS
    out = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32)
    for compact_idx in range(na):
        slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
        if (
            bool(prev_alive_by_seat[compact_idx])
            and not bool(new_alive_by_seat[compact_idx])
            and not bool(already_credited_by_slot[slot])
        ):
            out[slot] = -1.0
    return out


def _bc_replay_frame_episode_done(steps_row: list[Any]) -> bool:
    for st in steps_row:
        assert isinstance(st, dict)
        if st["status"] == "ACTIVE":
            return False
    return True


def _bc_estimated_power_smooth(
    estimated_power_history: list[torch.Tensor],
    estimated_power: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(estimated_power, torch.Tensor)
    assert tuple(estimated_power.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert estimated_power.dtype == torch.float32, estimated_power.dtype
    estimated_power_history.append(estimated_power.clone())
    if len(estimated_power_history) > int(_ESTIMATED_POWER_SMOOTH_STEPS):
        estimated_power_history.pop(0)
    assert 1 <= len(estimated_power_history) <= int(_ESTIMATED_POWER_SMOOTH_STEPS)
    history = torch.stack(estimated_power_history, dim=0)
    assert tuple(history.shape) == (len(estimated_power_history), ORBIT_PLAYER_AXIS_SLOTS)
    return torch.mean(history, dim=0)


def _bc_early_stop_winner_seats_from_estimated_power(
    estimated_power_smooth: torch.Tensor,
    *,
    already_credited_by_slot: list[bool],
    current_game_result: torch.Tensor,
    active_slots: tuple[int, ...],
    n_agents: int,
) -> frozenset[int]:
    na = int(n_agents)
    assert na in (2, 4), na
    assert isinstance(estimated_power_smooth, torch.Tensor)
    assert isinstance(current_game_result, torch.Tensor)
    assert tuple(estimated_power_smooth.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert tuple(current_game_result.shape) == (ORBIT_PLAYER_AXIS_SLOTS,)
    assert estimated_power_smooth.dtype == torch.float32, estimated_power_smooth.dtype
    assert current_game_result.dtype == torch.float32, current_game_result.dtype
    assert len(already_credited_by_slot) == ORBIT_PLAYER_AXIS_SLOTS
    eligible_slots = tuple(
        slot
        for slot in active_slots
        if not bool(already_credited_by_slot[slot])
        and abs(float(current_game_result[slot].item()) + 1.0) >= 1e-5
    )
    assert len(eligible_slots) >= 2, (eligible_slots, already_credited_by_slot, current_game_result)
    eligible_power = estimated_power_smooth[list(eligible_slots)]
    assert bool(torch.all(eligible_power >= 0.0).item()), eligible_power
    top_power = torch.max(eligible_power)
    winner_slots = frozenset(
        slot for slot in eligible_slots if bool((estimated_power_smooth[slot] == top_power).item())
    )
    assert len(winner_slots) >= 1, (eligible_slots, estimated_power_smooth)
    winner_seats: list[int] = []
    for compact_idx in range(na):
        slot = orbit_policy_slot_for_compact_agent(compact_idx, na)
        if slot in winner_slots:
            winner_seats.append(compact_idx)
    assert len(winner_seats) == len(winner_slots), (winner_seats, winner_slots, na)
    return frozenset(winner_seats)


def _bc_replay_early_stop_info(ctx: BcReplayPassContext) -> BcEarlyStopInfo | None:
    episode_cap = OrbitKaggleCppObservationCache.configuration_episode_steps(ctx.cfg)
    assert episode_cap == _EARLY_GAME_RESULT_STEP_CAP, (episode_cap, _EARLY_GAME_RESULT_STEP_CAP)
    active_slots = orbit_active_policy_slots(ctx.n_agents)
    env_step = torch.zeros((ORBIT_PLAYER_AXIS_SLOTS,), dtype=torch.float32)
    env_step[list(active_slots)] = 1.0
    estimated_power_history: list[torch.Tensor] = []
    already_credited_by_slot = [False] * ORBIT_PLAYER_AXIS_SLOTS

    plain_prev = _bc_replay_first_plain(ctx)
    estimated_power_0 = estimated_power_from_raw(
        _bc_estimated_power_raw_from_plain(plain=plain_prev, n_agents=ctx.n_agents),
        episode_step=0,
        episode_cap=episode_cap,
    )
    _bc_estimated_power_smooth(estimated_power_history, estimated_power_0)
    prev_alive_by_seat = _bc_alive_by_seat(plain=plain_prev, n_agents=ctx.n_agents)

    for frame_idx in range(1, len(ctx.steps)):
        row_next = ctx.steps[frame_idx]
        assert isinstance(row_next, list) and len(row_next) == ctx.n_agents
        st_next0 = row_next[0]
        assert isinstance(st_next0, dict)
        obs_next = st_next0["observation"]
        assert isinstance(obs_next, dict)
        plain_next = replay_observation_to_plain(obs_next, replay_step=frame_idx)
        new_alive_by_seat = _bc_alive_by_seat(plain=plain_next, n_agents=ctx.n_agents)
        current_game_result = _bc_current_game_result_from_alive_transition(
            prev_alive_by_seat=prev_alive_by_seat,
            new_alive_by_seat=new_alive_by_seat,
            already_credited_by_slot=already_credited_by_slot,
            n_agents=ctx.n_agents,
        )
        estimated_power = estimated_power_from_raw(
            _bc_estimated_power_raw_from_plain(plain=plain_next, n_agents=ctx.n_agents),
            episode_step=int(plain_next["step"]),
            episode_cap=episode_cap,
        )
        estimated_power_smooth = _bc_estimated_power_smooth(
            estimated_power_history,
            estimated_power,
        )
        done_b = _bc_replay_frame_episode_done(row_next)
        if not done_b:
            if estimated_power_early_stop_triggered(
                estimated_power_smooth,
                already_credited_by_slot=already_credited_by_slot,
                current_game_result=current_game_result,
                active_slots=active_slots,
                episode_step=int(plain_next["step"]),
                env_step=env_step,
            ):
                return BcEarlyStopInfo(
                    transition_index=int(frame_idx),
                    replay_obs_step=int(plain_next["step"]),
                    heuristic_winner_seats=_bc_early_stop_winner_seats_from_estimated_power(
                        estimated_power_smooth,
                        already_credited_by_slot=already_credited_by_slot,
                        current_game_result=current_game_result,
                        active_slots=active_slots,
                        n_agents=ctx.n_agents,
                    ),
                )
        for slot in active_slots:
            if abs(float(current_game_result[slot].item()) + 1.0) < 1e-5:
                already_credited_by_slot[slot] = True
        prev_alive_by_seat = new_alive_by_seat
    return None


def _bc_build_network_input_rows_from_replay(ctx: BcReplayPassContext) -> list[BcObsPassRow]:
    cache = OrbitKaggleCppObservationCache(configuration=ctx.cfg, num_agents=ctx.n_agents)
    cache.reset_from_kaggle_plain(
        plain=_bc_replay_first_plain(ctx),
        num_agents=ctx.n_agents,
    )
    rows: list[BcObsPassRow] = []
    for frame_idx in range(1, len(ctx.steps)):
        with profiler_span(ctx.wall_profiler, "bc_obs_step"):
            with profiler_span(ctx.wall_profiler, "plain_rows"):
                row_prev0 = ctx.steps[frame_idx - 1][0]
                row_next0 = ctx.steps[frame_idx][0]
                assert isinstance(row_prev0, dict) and isinstance(row_next0, dict)
                obs_prev = row_prev0["observation"]
                obs_next = row_next0["observation"]
                assert isinstance(obs_prev, dict) and isinstance(obs_next, dict)
                plain_prev = replay_observation_to_plain(obs_prev, replay_step=frame_idx - 1)
                plain_next = replay_observation_to_plain(obs_next, replay_step=frame_idx)
            with profiler_span(ctx.wall_profiler, "active_policy_slots"):
                active_policy_slots = _bc_active_policy_slots_for_plain(
                    plain=plain_prev,
                    n_agents=ctx.n_agents,
                    winner_seats=ctx.winner_seats,
                    filter_seats=ctx.filter_seats,
                )
            if len(active_policy_slots) > 0:
                with profiler_span(ctx.wall_profiler, "obs_all_and_ship_counts"):
                    obs_all_prev = kaggle_replay_obs_all_from_plain(plain_prev, ctx.n_agents)
                    attach_zeros_action_taken_index_on_seats(obs_all_prev)
                    ships_bpn = bc_planet_ship_count_bpn(
                        plain=plain_prev,
                        n_agents=ctx.n_agents,
                        active_policy_slots=active_policy_slots,
                    )
                with profiler_span(ctx.wall_profiler, "snapshot_policy_obs"):
                    snap = cache.snapshot_policy_obs_from_plain_seats_cpu(
                        plain=plain_prev,
                        seats_plain=obs_all_prev,
                        policy_slots=active_policy_slots,
                        ship_speed=float(ctx.cfg["shipSpeed"]),
                        wall_profiler=ctx.wall_profiler,
                    )
                assert snap["player_mask"].shape == (len(active_policy_slots),), snap["player_mask"].shape
                assert torch.all(snap["player_mask"] > 0.5), (active_policy_slots, snap["player_mask"])
                with profiler_span(ctx.wall_profiler, "append_obs_rows"):
                    for row_idx, policy_slot in enumerate(active_policy_slots):
                        key = BcRowKey(
                            transition_index=int(frame_idx),
                            replay_obs_step_before_action=int(plain_prev["step"]),
                            policy_slot=int(policy_slot),
                        )
                        rows.append(
                            BcObsPassRow(
                                key=key,
                                obs={k: snap[k][row_idx : row_idx + 1] for k in ORBIT_POLICY_OBS_KEYS},
                                ship_count=ships_bpn[row_idx : row_idx + 1],
                                loss_mask=torch.ones((1,), dtype=torch.float32),
                            )
                        )
            with profiler_span(ctx.wall_profiler, "cache_step_noop_update"):
                cache.step_noop_and_update_comets_from_kaggle_plain(
                    plain=plain_next,
                    num_agents=ctx.n_agents,
                )
        if ctx.wall_profiler is not None and frame_idx % ctx.profile_summary_every_steps == 0:
            ctx.wall_profiler.summary_stdout(
                f"bc_obs_pass replay_episode_id={ctx.replay_episode_id} frame_idx={frame_idx}",
                line_prefix="WALL_TREE_BC ",
                file=sys.stderr,
            )
    return rows


def _bc_build_gt_rows_from_replay(ctx: BcReplayPassContext) -> list[BcGtPassRow]:
    cache = OrbitKaggleCppObservationCache(configuration=ctx.cfg, num_agents=ctx.n_agents)
    cache.reset_from_kaggle_plain(
        plain=_bc_replay_first_plain(ctx),
        num_agents=ctx.n_agents,
    )
    rows: list[BcGtPassRow] = []
    for frame_idx in range(1, len(ctx.steps)):
        with profiler_span(ctx.wall_profiler, "bc_gt_step"):
            with profiler_span(ctx.wall_profiler, "plain_rows"):
                row_prev0 = ctx.steps[frame_idx - 1][0]
                row_next0 = ctx.steps[frame_idx][0]
                assert isinstance(row_prev0, dict) and isinstance(row_next0, dict)
                obs_prev = row_prev0["observation"]
                obs_next = row_next0["observation"]
                assert isinstance(obs_prev, dict) and isinstance(obs_next, dict)
                plain_prev = replay_observation_to_plain(obs_prev, replay_step=frame_idx - 1)
                plain_next = replay_observation_to_plain(obs_next, replay_step=frame_idx)
            with profiler_span(ctx.wall_profiler, "active_policy_slots"):
                active_policy_slots = _bc_active_policy_slots_for_plain(
                    plain=plain_prev,
                    n_agents=ctx.n_agents,
                    winner_seats=ctx.winner_seats,
                    filter_seats=ctx.filter_seats,
                )
            if len(active_policy_slots) > 0:
                with profiler_span(ctx.wall_profiler, "spawn_edges"):
                    fleets_prev = obs_prev["fleets"]
                    fleets_next = obs_next["fleets"]
                    assert isinstance(fleets_prev, list) and isinstance(fleets_next, list)
                    planets_prev = plain_prev["planets"]
                    assert isinstance(planets_prev, list)
                    edges = _spawn_edges_total_ships_by_dst(
                        new_spawn_rows=_new_spawn_fleet_rows(fleets_prev, fleets_next),
                        cache=cache,
                        hit_horizon=ctx.hit_horizon,
                        planet_id_to_slot_pre_step=_planet_id_to_slot(planets=planets_prev),
                        planets_pre_step=planets_prev,
                        n_agents=ctx.n_agents,
                        active_policy_slots=active_policy_slots,
                        fleet_stats=ctx.fleet_stats,
                    )
                with profiler_span(ctx.wall_profiler, "ship_counts"):
                    ships_bpn = bc_planet_ship_count_bpn(
                        plain=plain_prev,
                        n_agents=ctx.n_agents,
                        active_policy_slots=active_policy_slots,
                    )
                with profiler_span(ctx.wall_profiler, "rl_move_class"):
                    rl_cls, rl_loss_source_mask = _rl_per_planet_move_class_from_spawn_edges(
                        edges_by_owner_src=edges,
                        ships_bpn=ships_bpn,
                        frame_idx=int(frame_idx),
                        replay_episode_id=int(ctx.replay_episode_id),
                        n_agents=ctx.n_agents,
                        active_policy_slots=active_policy_slots,
                    )
                with profiler_span(ctx.wall_profiler, "assert_and_append_gt_rows"):
                    _assert_rl_gt_send_le_planet_ships(
                        edges_by_owner_src=edges,
                        ships_bpn=ships_bpn,
                        n_agents=ctx.n_agents,
                        active_policy_slots=active_policy_slots,
                    )
                    for row_idx, policy_slot in enumerate(active_policy_slots):
                        key = BcRowKey(
                            transition_index=int(frame_idx),
                            replay_obs_step_before_action=int(plain_prev["step"]),
                            policy_slot=int(policy_slot),
                        )
                        rl_cls_row = rl_cls[row_idx : row_idx + 1]
                        rows.append(
                            BcGtPassRow(
                                key=key,
                                rl_class=rl_cls_row,
                                rl_loss_source_mask=rl_loss_source_mask[row_idx : row_idx + 1],
                                honest_hit_debug_by_src_dst={},
                            )
                        )
            with profiler_span(ctx.wall_profiler, "cache_step_noop_update"):
                cache.step_noop_and_update_comets_from_kaggle_plain(
                    plain=plain_next,
                    num_agents=ctx.n_agents,
                )
        if ctx.wall_profiler is not None and frame_idx % ctx.profile_summary_every_steps == 0:
            ctx.wall_profiler.summary_stdout(
                f"bc_gt_pass replay_episode_id={ctx.replay_episode_id} frame_idx={frame_idx}",
                line_prefix="WALL_TREE_BC ",
                file=sys.stderr,
            )
    return rows


def _bc_assemble_payload_from_passes(
    *,
    ctx: BcReplayPassContext,
    obs_rows_in: list[BcObsPassRow],
    gt_rows_in: list[BcGtPassRow],
) -> dict[str, Any]:
    with profiler_span(ctx.wall_profiler, "bc_assemble_contract"):
        if len(obs_rows_in) == 0:
            raise BcEpisodeEmptyLossMaskError()
        assert len(obs_rows_in) == len(gt_rows_in), (len(obs_rows_in), len(gt_rows_in))
        obs_keys = tuple(row.key for row in obs_rows_in)
        assert len(set(obs_keys)) == len(obs_keys), len(obs_keys)
        gt_by_key = {row.key: row for row in gt_rows_in}
        assert len(gt_by_key) == len(gt_rows_in), len(gt_rows_in)
        assert set(obs_keys) == set(gt_by_key.keys()), (
            sorted(obs_keys, key=lambda k: (k.transition_index, k.policy_slot)),
            sorted(gt_by_key.keys(), key=lambda k: (k.transition_index, k.policy_slot)),
        )
    obs_rows: dict[str, list[torch.Tensor]] = {k: [] for k in ORBIT_POLICY_OBS_KEYS}
    targets: list[torch.Tensor] = []
    source_loss_masks: list[torch.Tensor] = []
    ship_counts: list[torch.Tensor] = []
    loss_masks: list[torch.Tensor] = []
    policy_slots: list[int] = []
    obs_step_before: list[int] = []
    transition_indices: list[int] = []
    with profiler_span(ctx.wall_profiler, "bc_assemble_rows"):
        for obs_row in obs_rows_in:
            gt_row = gt_by_key[obs_row.key]
            _assert_rl_gt_matches_available_action_mask(
                available_action_mask=obs_row.obs["available_action_mask"],
                planet_mask=obs_row.obs["orbit_planet_mask"],
                player_mask=obs_row.obs["player_mask"],
                rl_cls=gt_row.rl_class,
                hit_horizon=ctx.hit_horizon,
                honest_hit_debug_by_src_dst=gt_row.honest_hit_debug_by_src_dst,
                replay_episode_id=int(ctx.replay_episode_id),
                policy_slot=obs_row.key.policy_slot,
                frame_idx=obs_row.key.transition_index,
                replay_obs_step_before_action=obs_row.key.replay_obs_step_before_action,
            )
            for k in ORBIT_POLICY_OBS_KEYS:
                obs_rows[k].append(obs_row.obs[k])
            targets.append(gt_row.rl_class)
            source_loss_masks.append(gt_row.rl_loss_source_mask)
            ship_counts.append(obs_row.ship_count)
            loss_masks.append(obs_row.loss_mask)
            policy_slots.append(obs_row.key.policy_slot)
            obs_step_before.append(obs_row.key.replay_obs_step_before_action)
            transition_indices.append(obs_row.key.transition_index)
    with profiler_span(ctx.wall_profiler, "bc_assemble_stack"):
        loss_stacked = torch.stack(loss_masks, dim=0)
        if int(torch.count_nonzero(loss_stacked).item()) == 0:
            raise BcEpisodeEmptyLossMaskError()
        stacked_obs = {k: torch.stack(obs_rows[k], dim=0) for k in ORBIT_POLICY_OBS_KEYS}
        t = int(len(targets))
    return {
        "num_agents": int(ctx.n_agents),
        "num_bc_timesteps": t,
        "configuration": dict(ctx.cfg),
        "hit_horizon": int(ctx.hit_horizon),
        "rl_per_planet_move_class": torch.stack(targets, dim=0),
        "bc_loss_source_mask": torch.stack(source_loss_masks, dim=0),
        "bc_planet_ship_count": torch.stack(ship_counts, dim=0),
        "bc_loss_player_mask": loss_stacked,
        "bc_policy_slot": torch.tensor(policy_slots, dtype=torch.int64),
        **stacked_obs,
        "replay_obs_step_before_action": torch.tensor(obs_step_before, dtype=torch.int64),
        "replay_transition_index": torch.tensor(transition_indices, dtype=torch.int64),
    }


def _bc_early_stop_real_winner_mismatch_diagnostic(
    *,
    ep: Mapping[str, Any],
    n_agents: int,
    early_stop_info: BcEarlyStopInfo | None,
) -> dict[str, Any] | None:
    if early_stop_info is None:
        return None
    real_winner_seats = _replay_winner_seats(ep=ep, n_agents=n_agents)
    if early_stop_info.heuristic_winner_seats == real_winner_seats:
        return None
    return {
        "transition_index": int(early_stop_info.transition_index),
        "replay_obs_step": int(early_stop_info.replay_obs_step),
        "heuristic_winner_seats": tuple(sorted(early_stop_info.heuristic_winner_seats)),
        "real_winner_seats": tuple(sorted(real_winner_seats)),
    }


def build_behavior_clone_episode_torch_dict(
    ep: Mapping[str, Any],
    *,
    replay_episode_id: int,
    hit_horizon: int,
    bc_loss_player_mask_winner_only: bool = False,
    bc_loss_player_mask_teams_filter_seats_only: bool = False,
    teams_filter: tuple[str, ...] = (),
    wall_profiler: WallTreeProfiler | None = None,
    profile_summary_every_steps: int = 10,
    fleet_stats: BcFleetStats | None = None,
) -> dict[str, Any]:
    assert profile_summary_every_steps >= 1, profile_summary_every_steps
    if bool(bc_loss_player_mask_teams_filter_seats_only):
        assert len(teams_filter) > 0, "bc_loss_player_mask_teams_filter_seats_only requires non-empty teams_filter"
    if len(teams_filter) > 0:
        assert episode_matches_teams_filter(ep, teams_filter), (
            kaggle_replay_team_names(ep),
            teams_filter,
        )
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 2
    cfg = ep["configuration"]
    assert isinstance(cfg, Mapping)
    row0 = steps[0]
    assert isinstance(row0, list)
    n_agents = len(row0)
    winner_seats = _replay_winner_seats(ep=ep, n_agents=n_agents) if bc_loss_player_mask_winner_only else None
    filter_seats = (
        _replay_filter_seats(ep=ep, teams_filter=teams_filter)
        if bc_loss_player_mask_teams_filter_seats_only
        else None
    )
    assert_replay_frame_contract(ep)
    hz = int(hit_horizon)
    assert hz >= 1
    ctx = BcReplayPassContext(
        steps=steps,
        cfg=cfg,
        n_agents=n_agents,
        winner_seats=winner_seats,
        filter_seats=filter_seats,
        hit_horizon=hz,
        replay_episode_id=int(replay_episode_id),
        wall_profiler=wall_profiler,
        profile_summary_every_steps=int(profile_summary_every_steps),
        fleet_stats=fleet_stats,
    )
    early_stop_info = _bc_replay_early_stop_info(ctx)
    mismatch = _bc_early_stop_real_winner_mismatch_diagnostic(
        ep=ep,
        n_agents=n_agents,
        early_stop_info=early_stop_info,
    )
    if early_stop_info is not None and mismatch is None:
        ctx = BcReplayPassContext(
            steps=steps[: int(early_stop_info.transition_index) + 1],
            cfg=cfg,
            n_agents=n_agents,
            winner_seats=winner_seats,
            filter_seats=filter_seats,
            hit_horizon=hz,
            replay_episode_id=int(replay_episode_id),
            wall_profiler=wall_profiler,
            profile_summary_every_steps=int(profile_summary_every_steps),
            fleet_stats=fleet_stats,
        )
    obs_rows = _bc_build_network_input_rows_from_replay(ctx)
    gt_rows = _bc_build_gt_rows_from_replay(ctx)
    with profiler_span(wall_profiler, "bc_assemble_payload"):
        payload = _bc_assemble_payload_from_passes(ctx=ctx, obs_rows_in=obs_rows, gt_rows_in=gt_rows)
    if early_stop_info is not None and mismatch is None:
        payload["bc_early_stop_info"] = {
            "transition_index": int(early_stop_info.transition_index),
            "replay_obs_step": int(early_stop_info.replay_obs_step),
            "heuristic_winner_seats": tuple(sorted(early_stop_info.heuristic_winner_seats)),
        }
    if mismatch is not None:
        payload["bc_early_stop_real_winner_mismatch"] = mismatch
    return payload


def _append_human_dump_full_feature_decode(
    lines: list[str],
    *,
    payload: Mapping[str, Any],
    target_replay_obs_step: int,
) -> None:
    """Append one block: first BC row whose seat-0 pre-action ``step`` equals ``target_replay_obs_step``."""
    t = int(payload["num_bc_timesteps"])
    ros = payload["replay_obs_step_before_action"]
    rti = payload["replay_transition_index"]
    assert isinstance(ros, torch.Tensor) and isinstance(rti, torch.Tensor)
    ti_hit: int | None = None
    tgt = int(target_replay_obs_step)
    for ti in range(t):
        if int(ros[ti].item()) == tgt:
            ti_hit = ti
            break
    lines.append("\n")
    lines.append("#" * 80 + "\n")
    lines.append(
        f"# FULL policy_obs decode (named channels): first bc_timestep with "
        f"replay_obs_step_before_action == {tgt} "
        f"(zero-based replay frame index; orbit tape ``episode_step``).\n"
    )
    lines.append("#" * 80 + "\n")
    if ti_hit is None:
        lines.append(
            f"(no matching row; episode replay_obs_step_before_action range "
            f"{int(ros.min().item())}..{int(ros.max().item())})\n"
        )
        return

    ti = ti_hit
    na = int(payload["num_agents"])
    policy_slots = payload["bc_policy_slot"]
    assert isinstance(policy_slots, torch.Tensor) and policy_slots.shape == (t,), policy_slots.shape
    lines.append(
        f"bc_timestep t={ti}  replay_transition_index={int(rti[ti].item())}  "
        f"replay_obs_step_before_action={int(ros[ti].item())}\n"
        f"num_agents={na} original_policy_slot={int(policy_slots[ti].item())}\n\n"
    )

    pf = payload["orbit_planet_features"][ti]
    af = payload["orbit_planet_arrival_features"][ti]
    pm = payload["orbit_planet_mask"][ti]
    pwm = payload["orbit_planet_pairwise_mask"][ti]
    pxf = payload["orbit_planet_pairwise_features"][ti]
    aam = payload["available_action_mask"][ti]
    ati = payload["action_taken_index"][ti]
    plm = payload["player_mask"][ti]
    lm = payload["bc_loss_player_mask"][ti]

    assert isinstance(pf, torch.Tensor)
    p_rows = int(pf.shape[0])
    assert p_rows == 1, p_rows
    assert tuple(pf.shape) == (p_rows, ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES), pf.shape
    assert isinstance(af, torch.Tensor)
    assert tuple(af.shape) == (
        p_rows,
        ORBIT_MAX_PLANETS,
        ORBIT_PLANET_ARRIVAL_HORIZON,
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLANET_ARRIVAL_FEATURES,
    ), af.shape
    assert isinstance(pm, torch.Tensor) and tuple(pm.shape) == (p_rows, ORBIT_MAX_PLANETS)
    assert isinstance(pwm, torch.Tensor) and tuple(pwm.shape) == (
        p_rows,
        ORBIT_PLANET_PAIRWISE_COUNT,
    ), pwm.shape
    assert isinstance(pxf, torch.Tensor) and tuple(pxf.shape) == (
        p_rows,
        ORBIT_PLANET_PAIRWISE_COUNT,
        ORBIT_EDGE_FEATURES,
    ), pxf.shape
    assert isinstance(aam, torch.Tensor) and tuple(aam.shape) == (
        p_rows,
        ORBIT_MAX_PLANETS,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), aam.shape
    assert isinstance(ati, torch.Tensor) and tuple(ati.shape) == (
        p_rows,
        ORBIT_MAX_PLANETS,
        1,
    ), ati.shape
    assert isinstance(plm, torch.Tensor) and plm.shape == (p_rows,), plm.shape
    assert isinstance(lm, torch.Tensor) and lm.shape == (p_rows,), lm.shape

    hz = int(ORBIT_PLANET_ARRIVAL_HORIZON)
    assert ORBIT_PLANET_ARRIVAL_FEATURES == ORBIT_PLANET_TEMPORAL_FEATURES

    for p in range(p_rows):
        lines.append("=" * 72 + "\n")
        lines.append(
            f"POLICY_ROW p={p} original_policy_slot={int(policy_slots[ti].item())}  "
            f"player_mask={float(plm[p].item()):.6g}  "
            f"bc_loss_player_mask={float(lm[p].item()):.6g}\n"
        )
        lines.append("=" * 72 + "\n")

        lines.append("--- orbit_planet_mask + orbit_planet_features ---\n")
        for i in range(ORBIT_MAX_PLANETS):
            m = float(pm[p, i].item())
            if m == 0.0:
                lines.append(f"  planet_slot={i:2d}  MASK=0  (padded)\n")
                continue
            parts = [
                f"{_orbit_planet_feature_channel_name(fch)}={float(pf[p, i, fch].item()):.6g}"
                for fch in range(ORBIT_PLANET_FEATURES)
            ]
            lines.append(f"  planet_slot={i:2d}  mask={m:.6g}\n")
            lines.append("    " + "\n    ".join(parts) + "\n")

        lines.append("\n--- orbit_planet_arrival_features (planet slot × time × rel_block × feature) ---\n")
        for i in range(ORBIT_MAX_PLANETS):
            if float(pm[p, i].item()) == 0.0:
                continue
            lines.append(f"  planet_slot={i:2d}\n")
            for rel in range(ORBIT_PLAYER_AXIS_SLOTS):
                for fidx in range(ORBIT_PLANET_TEMPORAL_FEATURES):
                    seg = af[p, i, :, rel, fidx].detach().float().tolist()
                    lines.append(
                        f"    rel_block_{rel}_feature_{fidx}_in_1..{hz}_steps: "
                        + " ".join(f"{v:.6g}" for v in seg)
                        + "\n"
                    )

        lines.append("\n--- orbit_planet_pairwise (flat index row-major src,dst; mask>0.5 only) ---\n")
        for e in range(ORBIT_PLANET_PAIRWISE_COUNT):
            if float(pwm[p, e].item()) <= 0.5:
                continue
            src = e // ORBIT_MAX_PLANETS
            dst = e % ORBIT_MAX_PLANETS
            ech = [
                f"{_orbit_edge_feature_channel_name(c)}={float(pxf[p, e, c].item()):.6g}"
                for c in range(ORBIT_EDGE_FEATURES)
            ]
            lines.append(f"  edge_flat={e:4d}  src={src:2d} dst={dst:2d}  " + " ".join(ech) + "\n")

        lines.append("\n--- available_action_mask + action_taken_index (valid source slots only) ---\n")
        for i in range(ORBIT_MAX_PLANETS):
            if float(pm[p, i].item()) == 0.0:
                continue
            taken = int(ati[p, i, 0].item())
            row = aam[p, i]
            assert row.dtype == torch.int8, row.dtype
            avail_idx = torch.nonzero(row > 0, as_tuple=False).reshape(-1).tolist()
            lines.append(
                f"  src_slot={i:2d}  action_taken_index={taken}  num_available={len(avail_idx)}\n"
            )
            if len(avail_idx) <= 64:
                lines.append(f"    available_class_indices={avail_idx!r}\n")
            else:
                lines.append(
                    f"    available_class_indices head={avail_idx[:32]!r} "
                    f"... tail={avail_idx[-16:]!r}\n"
                )
        lines.append("\n")


def _human_dump_tensor_stats(name: str, x: torch.Tensor) -> str:
    assert isinstance(x, torch.Tensor)
    if x.dtype == torch.bool:
        n = int(x.numel())
        nt = int(torch.count_nonzero(x).item())
        return f"{name}: bool shape={tuple(x.shape)} true_count={nt}/{n}\n"
    xf = x.detach().float()
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={float(xf.min().item()):.6g} max={float(xf.max().item()):.6g} "
        f"mean={float(xf.mean().item()):.6g}\n"
    )


def write_behavior_clone_episode_human_dump(
    *,
    payload: Mapping[str, Any],
    dest: Path,
    source_path: Path | None,
) -> None:
    """Text summary for manual cross-check with orbit tape / replay (alignment, masks, RL classes)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    na = int(payload["num_agents"])
    t = int(payload["num_bc_timesteps"])
    hz = int(payload["hit_horizon"])
    lines: list[str] = []
    lines.append("# BC episode human dump\n")
    if source_path is not None:
        lines.append(f"source_replay={source_path.resolve()}\n")
    lines.append(f"hit_horizon={hz}\n")
    lines.append(f"num_agents={na} num_bc_timesteps={t}\n")
    lines.append(f"configuration={payload['configuration']!r}\n")
    lines.append(
        "\n# Alignment: BC row index t uses the same pre-action observation as Kaggle seat 0 "
        "``steps[t]['observation']`` (and orbit tape frame ``episode_step == replay_obs_step_before_action[t]``).\n"
        "# ``replay_transition_index[t]`` is the Kaggle ``steps`` row index where that step's ``action`` applies.\n"
        "# Orbit debug tape has one frame per ``steps`` row (len(steps) frames); BC has len(steps)-1 rows "
        "(no row after the final observation).\n"
        f"# End of file: full named ``policy_obs`` decode for the first row with "
        f"``replay_obs_step_before_action == {BC_HUMAN_DUMP_FULL_DECODE_REPLAY_OBS_STEP}`` "
        "(all policy seats).\n\n"
    )

    rti = payload["replay_transition_index"]
    ros = payload["replay_obs_step_before_action"]
    assert isinstance(rti, torch.Tensor) and isinstance(ros, torch.Tensor)
    assert int(rti.shape[0]) == t and int(ros.shape[0]) == t, (rti.shape, ros.shape, t)
    policy_slots = payload["bc_policy_slot"]
    assert isinstance(policy_slots, torch.Tensor) and policy_slots.shape == (t,), policy_slots.shape

    rl_cls = payload["rl_per_planet_move_class"]
    assert isinstance(rl_cls, torch.Tensor) and rl_cls.shape == (t, 1, ORBIT_MAX_PLANETS)
    ships_bpn = payload["bc_planet_ship_count"]
    assert isinstance(ships_bpn, torch.Tensor) and ships_bpn.shape == (t, 1, ORBIT_MAX_PLANETS)
    loss_m = payload["bc_loss_player_mask"]
    assert isinstance(loss_m, torch.Tensor) and loss_m.shape == (t, 1)
    nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)

    for ti in range(t):
        lines.append(f"## bc_timestep t={ti}\n")
        lines.append(
            f"replay_transition_index={int(rti[ti].item())} "
            f"replay_obs_step_before_action={int(ros[ti].item())} "
            f"original_policy_slot={int(policy_slots[ti].item())}\n"
        )
        lm = [float(loss_m[ti, p].item()) for p in range(1)]
        lines.append(f"bc_loss_player_mask={lm!r}\n")

        nz_ships: list[tuple[int, int, float]] = []
        for p in range(1):
            for i in range(ORBIT_MAX_PLANETS):
                v = float(ships_bpn[ti, p, i].item())
                if v > 0.0:
                    nz_ships.append((p, i, v))
        if nz_ships:
            lines.append(f"bc_planet_ship_count_nonzero count={len(nz_ships)} (seat,slot,ships):\n")
            for p, i, v in nz_ships[:200]:
                lines.append(f"  ({p},{i})={v:.6g}\n")
            if len(nz_ships) > 200:
                lines.append(f"  ... {len(nz_ships) - 200} more\n")
        else:
            lines.append("bc_planet_ship_count_nonzero: (none)\n")

        lines.append("rl_per_planet_move_class (non-noop sends; noop is class i*NB for source slot i):\n")
        for p in range(1):
            for i in range(ORBIT_MAX_PLANETS):
                c = int(rl_cls[ti, p, i].item())
                noop = int(i * nb)
                if c == noop:
                    continue
                j = c // nb
                sn = c % nb
                lines.append(
                    f"  seat={p} src_slot={i} class={c} dst_slot={j} sn={sn} "
                    "\n"
                )

        lines.append("policy_obs_tensor_stats:\n")
        for k in ORBIT_POLICY_OBS_KEYS:
            full = payload[k]
            assert isinstance(full, torch.Tensor), k
            assert int(full.shape[0]) == t, (k, tuple(full.shape), t)
            lines.append("  " + _human_dump_tensor_stats(k, full[ti]).rstrip("\n") + "\n")

        pf = payload["orbit_planet_features"]
        assert isinstance(pf, torch.Tensor) and pf.ndim == 4
        head = pf[ti, 0, 0, : min(8, int(pf.shape[-1]))].detach().float().tolist()
        lines.append(f"  orbit_planet_features[row=0,planet=0,0:8]={head!r}\n")
        lines.append("\n")

    _append_human_dump_full_feature_decode(
        lines,
        payload=payload,
        target_replay_obs_step=BC_HUMAN_DUMP_FULL_DECODE_REPLAY_OBS_STEP,
    )

    dest.write_text("".join(lines), encoding="utf-8")


def write_behavior_clone_policy_input_dump(
    *,
    payload: Mapping[str, Any],
    dest: Path,
) -> None:
    t = int(payload["num_bc_timesteps"])
    assert t >= 1, t
    policy_slots = payload["bc_policy_slot"]
    assert isinstance(policy_slots, torch.Tensor), type(policy_slots)
    assert tuple(policy_slots.shape) == (t,), policy_slots.shape
    samples: list[tuple[torch.Tensor, ...]] = []
    for ti in range(t):
        if int(policy_slots[ti].item()) != 0:
            continue
        sample: list[torch.Tensor] = []
        for key in _BC_POLICY_INPUT_DUMP_KEYS:
            x = payload[key]
            assert isinstance(x, torch.Tensor), (key, type(x))
            assert x.ndim >= 2, (key, tuple(x.shape))
            assert int(x.shape[0]) == t, (key, tuple(x.shape), t)
            assert int(x.shape[1]) == 1, (key, tuple(x.shape))
            sample.append(x[ti : ti + 1].detach().cpu().contiguous())
        samples.append(tuple(sample))
    assert len(samples) >= 1, "BC policy input dump requires at least one policy_slot == 0 row"
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _BC_POLICY_INPUT_DUMP_FORMAT,
            "input_keys": _BC_POLICY_INPUT_DUMP_KEYS,
            "samples_per_step": 1,
            "samples": samples,
        },
        str(dest),
    )


def write_behavior_clone_episode_pt(
    ep: Mapping[str, Any],
    dest: Path,
    *,
    replay_episode_id: int,
    hit_horizon: int,
    bc_loss_player_mask_winner_only: bool = False,
    bc_loss_player_mask_teams_filter_seats_only: bool = False,
    teams_filter: tuple[str, ...] = (),
    human_dump: bool = False,
    human_dump_path: Path | None = None,
    human_dump_source: Path | None = None,
    wall_profiler: WallTreeProfiler | None = None,
    profile_summary_every_steps: int = 10,
    fleet_stats: BcFleetStats | None = None,
) -> dict[str, Any]:
    payload = build_behavior_clone_episode_torch_dict(
        ep,
        replay_episode_id=int(replay_episode_id),
        hit_horizon=hit_horizon,
        bc_loss_player_mask_winner_only=bc_loss_player_mask_winner_only,
        bc_loss_player_mask_teams_filter_seats_only=bc_loss_player_mask_teams_filter_seats_only,
        teams_filter=teams_filter,
        wall_profiler=wall_profiler,
        profile_summary_every_steps=int(profile_summary_every_steps),
        fleet_stats=fleet_stats,
    )
    with profiler_span(wall_profiler, "bc_serialize_payload"):
        payload["serialization"] = _BC_EPISODE_PT_SERIALIZATION
        _atomic_write_bc_episode_gzip(dest, payload)
    if human_dump:
        assert human_dump_path is not None, human_dump_path
        with profiler_span(wall_profiler, "bc_human_dump"):
            write_behavior_clone_episode_human_dump(
                payload=payload,
                dest=human_dump_path,
                source_path=human_dump_source,
            )
    else:
        assert human_dump_path is None, human_dump_path
        assert human_dump_source is None, human_dump_source
    return payload
