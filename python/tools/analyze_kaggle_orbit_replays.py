"""Load Orbit Wars replays serialized like ``kaggle_environments`` ``Environment.toJSON()``."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_ORBIT_WARS_NAME = "orbit_wars"
_AGENT_STATE_KEYS = frozenset({"action", "reward", "info", "observation", "status"})
_ENVELOPE_KEYS = frozenset(
    {
        "id",
        "name",
        "title",
        "description",
        "version",
        "module_version",
        "configuration",
        "specification",
        "steps",
        "rewards",
        "statuses",
        "schema_version",
        "info",
    }
)

_PY_ROOT = Path(__file__).resolve().parents[1]
_IMPALA_ROOT = _PY_ROOT.parent
_DEFAULT_ANALYZE_TAPE_DIR = _IMPALA_ROOT / "outputs" / "analyze"
_DEFAULT_ANALYZE_TAPE_NAME = "replay_vis"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.gym.orbit_kaggle_replay_bc_dataset import (
    BcEpisodeEmptyLossMaskError,
    BcFleetStats,
    episode_matches_teams_filter,
    episode_pt_stem,
    is_kaggle_replay_episode_json_path,
    iter_episode_json_files,
    kaggle_replay_episode_id_from_path,
    kaggle_replay_team_names,
    write_behavior_clone_episode_pt,
    write_behavior_clone_policy_input_dump,
)
from src.gym.orbit_kaggle_replay_fleet_gt import default_hit_horizon, print_replay_fleet_spawn_ground_truth
from src.gym.orbit_kaggle_replay_tape import (
    default_replay_tape_hit_horizon,
    write_kaggle_replay_orbit_tape,
)
from src.gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from src.gym.obs_wrapper import (
    ORBIT_MAX_PLANETS,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    orbit_policy_slot_for_compact_agent,
)
from kaggle_submission.submission import (
    OrbitSubmissionRunner,
    load_submission_model,
)


def load_json_maybe_gzip(path: Path) -> object:
    assert path.is_file(), path
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_first_jsonl_object(path: Path) -> dict[str, object]:
    assert path.is_file(), path
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert isinstance(obj, dict), type(obj)
            return obj
    raise AssertionError("jsonl replay file has no non-empty lines")


def _load_json_replay_raw(path: Path, *, container: str) -> object:
    assert container in ("json", "jsonl"), container
    if container == "jsonl":
        return load_first_jsonl_object(path)
    return load_json_maybe_gzip(path)


def assert_orbit_wars_kaggle_envelope(ep: Mapping[str, object]) -> None:
    assert ep["name"] == _ORBIT_WARS_NAME, ep["name"]
    missing = _ENVELOPE_KEYS.difference(ep.keys())
    assert not missing, f"replay envelope missing keys: {sorted(missing)}"
    steps = ep["steps"]
    assert isinstance(steps, list), type(steps)
    assert len(steps) >= 2, len(steps)
    row0 = steps[0]
    assert isinstance(row0, list), type(row0)
    n_agents = len(row0)
    assert n_agents in (2, 4), n_agents
    for si, row in enumerate(steps):
        assert isinstance(row, list), (si, type(row))
        assert len(row) == n_agents, (si, len(row), n_agents)
        for ai, st in enumerate(row):
            assert isinstance(st, dict), (si, ai, type(st))
            missing_a = _AGENT_STATE_KEYS.difference(st.keys())
            assert not missing_a, (si, ai, sorted(missing_a))
            assert isinstance(st["observation"], dict), (si, ai, type(st["observation"]))
            act = st["action"]
            assert act is None or isinstance(act, list), (si, ai, type(act))


def normalize_kaggle_replay_actions_null_to_empty_inplace(ep: Mapping[str, object]) -> None:
    """Turn ``action: null`` in JSON (``None`` here) into ``[]``, matching reference ``process_moves``."""
    steps = ep["steps"]
    assert isinstance(steps, list)
    for row in steps:
        assert isinstance(row, list)
        for st in row:
            assert isinstance(st, dict)
            if st.get("action") is None:
                st["action"] = []


def load_orbit_wars_kaggle_replay(path: Path, *, container: str) -> dict[str, object]:
    raw = _load_json_replay_raw(path, container=container)
    assert isinstance(raw, dict), (path, type(raw))
    assert_orbit_wars_kaggle_envelope(raw)
    normalize_kaggle_replay_actions_null_to_empty_inplace(raw)
    return raw


def try_load_orbit_wars_kaggle_replay(path: Path, *, container: str) -> dict[str, object] | None:
    """Return episode dict, or ``None`` if the file is not an Orbit Wars Kaggle envelope (skip without raising)."""
    try:
        raw = _load_json_replay_raw(path, container=container)
    except json.JSONDecodeError as e:
        print(
            f"bc_dataset_skip_invalid_json path={path} line={e.lineno} col={e.colno} msg={e.msg!r}",
            file=sys.stderr,
        )
        return None
    if not isinstance(raw, dict):
        print(
            f"bc_dataset_skip_not_json_object path={path} type={type(raw).__name__}",
            file=sys.stderr,
        )
        return None
    if raw.get("name") != _ORBIT_WARS_NAME:
        print(
            f"bc_dataset_skip_wrong_replay_type path={path} name={raw.get('name')!r}",
            file=sys.stderr,
        )
        return None
    assert_orbit_wars_kaggle_envelope(raw)
    normalize_kaggle_replay_actions_null_to_empty_inplace(raw)
    return raw


def _kaggle_replay_sidecar_metadata_path(episode_json_path: Path) -> Path:
    stem = episode_pt_stem(episode_json_path)
    return episode_json_path.parent / f"{stem}_metadata.json"


def public_leaderboard_submission_id_for_team_name(
    metadata: Mapping[str, Any],
    *,
    team_name: str,
    metadata_path: Path,
) -> int | None:
    """Resolve Kaggle ``submissionId`` for ``teamName`` from ``leaderboard.teams`` + ``publicLeaderboard``."""
    lb = metadata["leaderboard"]
    assert isinstance(lb, dict), (metadata_path, type(lb).__name__)
    teams_raw = lb["teams"]
    assert isinstance(teams_raw, list), (metadata_path, type(teams_raw).__name__)
    team_id: int | None = None
    for entry in teams_raw:
        assert isinstance(entry, dict), (metadata_path, type(entry).__name__)
        if entry.get("teamName") != team_name:
            continue
        cur = int(entry["teamId"])
        if team_id is not None:
            assert team_id == cur, (metadata_path, team_name, team_id, cur)
        team_id = cur
    if team_id is None:
        return None
    pub = lb["publicLeaderboard"]
    assert isinstance(pub, list), (metadata_path, type(pub).__name__)
    for row in pub:
        assert isinstance(row, dict), (metadata_path, type(row).__name__)
        if int(row["teamId"]) != team_id:
            continue
        return int(row["submissionId"])
    raise AssertionError(
        f"{metadata_path}: teamId={team_id} teamName={team_name!r} not in leaderboard.publicLeaderboard "
        f"(n_rows={len(pub)})"
    )


def _bc_replay_seat_count(ep: Mapping[str, Any]) -> int:
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 1, len(steps)
    row0 = steps[0]
    assert isinstance(row0, list)
    return len(row0)


def _bc_replay_winner_seats(ep: Mapping[str, Any], n_agents: int) -> frozenset[int]:
    rewards = ep["rewards"]
    assert isinstance(rewards, (list, tuple)), type(rewards)
    assert len(rewards) == n_agents, (len(rewards), n_agents)
    return frozenset(i for i in range(n_agents) if float(rewards[i]) == 1.0)


def _bc_replay_team_filter_seats(ep: Mapping[str, Any], teams_filter: tuple[str, ...]) -> frozenset[int]:
    assert len(teams_filter) > 0
    want = set(teams_filter)
    names = kaggle_replay_team_names(ep)
    out = frozenset(i for i, name in enumerate(names) if name in want)
    assert len(out) >= 1, (names, teams_filter)
    return out


def _bc_replay_matches_loss_seat_filters(
    ep: Mapping[str, Any],
    *,
    winner_only: bool,
    teams_filter_seats_only: bool,
    teams_filter: tuple[str, ...],
) -> bool:
    if not bool(winner_only):
        return True
    n_agents = _bc_replay_seat_count(ep)
    winner_seats = _bc_replay_winner_seats(ep, n_agents)
    if not bool(teams_filter_seats_only):
        return len(winner_seats) > 0
    filter_seats = _bc_replay_team_filter_seats(ep, teams_filter)
    return bool(winner_seats & filter_seats)


def summarize_replay(ep: Mapping[str, object]) -> str:
    steps = ep["steps"]
    assert isinstance(steps, list)
    cfg = ep["configuration"]
    assert isinstance(cfg, Mapping)
    n_agents = len(steps[0])
    assert isinstance(steps[0], list)
    lines = [
        f"name={ep['name']!r} version={ep['version']!r}",
        f"agents={n_agents} frame_count={len(steps)}",
        f"configuration={dict(cfg)}",
        f"final_rewards={ep['rewards']!r} statuses={ep['statuses']!r}",
    ]
    return "\n".join(lines)


def iter_obs_action_pairs_for_seat(
    ep: Mapping[str, object],
    *,
    seat: int,
    start_frame: int,
) -> Iterator[tuple[Mapping[str, object], list[object]]]:
    """Yield ``(steps[i-1][seat].observation, steps[i][seat].action)`` for ``i >= start_frame``."""
    steps = ep["steps"]
    assert isinstance(steps, list)
    assert isinstance(steps[0], list)
    n_agents = len(steps[0])
    assert 0 <= seat < n_agents, (seat, n_agents)
    assert start_frame >= 1, start_frame
    for i in range(start_frame, len(steps)):
        prev = steps[i - 1][seat]
        cur = steps[i][seat]
        assert isinstance(prev, dict) and isinstance(cur, dict)
        obs = prev["observation"]
        act = cur["action"]
        assert isinstance(obs, dict)
        if act is None:
            act = []
        else:
            assert isinstance(act, list)
        yield obs, act


def _default_sample_path() -> Path:
    return _PY_ROOT / "data" / "kaggle_orbit_replays" / "orbit_wars_sample_kaggle_env.json"


def _bc_episode_human_dump_path(pt_path: Path) -> Path:
    return pt_path.with_suffix(".bc_dump.txt")


def _bc_policy_input_dump_path(pt_path: Path) -> Path:
    return pt_path.with_suffix(".bc_policy_inputs.pt")


def _bc_workers_from_env_or_arg(cli_workers: int | None) -> int:
    if cli_workers is not None:
        assert cli_workers >= 1, cli_workers
        return int(cli_workers)
    raw = os.environ.get("N_WORKERS", "").strip()
    if not raw:
        return 1
    v = int(raw)
    assert v >= 1, v
    return v


def _bc_fleet_stats_add(
    dst: BcFleetStats,
    src: BcFleetStats | None,
) -> None:
    if src is None:
        return
    dst.spawned_fleets += int(src.spawned_fleets)
    dst.resolved_hit_fleets += int(src.resolved_hit_fleets)
    dst.no_hit_fleets += int(src.no_hit_fleets)
    for hit_steps, count in src.hit_steps_histogram.items():
        hs = int(hit_steps)
        n = int(count)
        assert hs >= 1, hs
        assert n >= 0, n
        dst.hit_steps_histogram[hs] = dst.hit_steps_histogram.get(hs, 0) + n
    for ships, count in src.ship_count_histogram.items():
        sh = int(ships)
        n = int(count)
        assert sh >= 1, sh
        assert n >= 0, n
        dst.ship_count_histogram[sh] = dst.ship_count_histogram.get(sh, 0) + n
    for remaining, count in src.source_remaining_ship_count_histogram.items():
        rem = int(remaining)
        n = int(count)
        assert rem >= 0, rem
        assert n >= 0, n
        dst.source_remaining_ship_count_histogram[rem] = (
            dst.source_remaining_ship_count_histogram.get(rem, 0) + n
        )
    for percent_bucket, count in src.source_sent_ship_percent_bucket_histogram.items():
        pct_bucket = int(percent_bucket)
        n = int(count)
        assert 0 <= pct_bucket <= 100, pct_bucket
        assert n >= 0, n
        dst.source_sent_ship_percent_bucket_histogram[pct_bucket] = (
            dst.source_sent_ship_percent_bucket_histogram.get(pct_bucket, 0) + n
        )


def _bc_fleet_stats_text(
    *,
    stats: BcFleetStats,
    files_done: int,
    n_written: int,
    n_skipped: int,
    n_write_failed: int,
    final: bool,
) -> str:
    resolved = int(stats.resolved_hit_fleets)
    spawned = int(stats.spawned_fleets)
    lines = [
        "bc_gt_active_fleet_stats",
        f"kind={'final' if bool(final) else 'progress'}",
        f"files_done={int(files_done)}",
        f"written={int(n_written)}",
        f"skipped={int(n_skipped)}",
        f"failed={int(n_write_failed)}",
        f"spawned={spawned}",
        f"resolved_hit={resolved}",
        f"no_hit={int(stats.no_hit_fleets)}",
        "",
        "hit_steps_histogram",
        "hit_steps count percent_of_resolved_hit",
    ]
    for hit_steps in sorted(stats.hit_steps_histogram):
        count = int(stats.hit_steps_histogram[hit_steps])
        pct = 100.0 * float(count) / float(resolved) if resolved > 0 else 0.0
        lines.append(f"{int(hit_steps)} {count} {pct:.6f}")
    if len(stats.hit_steps_histogram) == 0:
        lines.append("none 0 0.000000")
    lines.extend(
        [
            "",
            "ship_count_histogram",
            "ships count percent_of_spawned",
        ]
    )
    for ships in sorted(stats.ship_count_histogram):
        count = int(stats.ship_count_histogram[ships])
        pct = 100.0 * float(count) / float(spawned) if spawned > 0 else 0.0
        lines.append(f"{int(ships)} {count} {pct:.6f}")
    if len(stats.ship_count_histogram) == 0:
        lines.append("none 0 0.000000")
    lines.extend(
        [
            "",
            "source_remaining_ship_count_histogram",
            "remaining_ships count percent_of_spawned",
        ]
    )
    for remaining in sorted(stats.source_remaining_ship_count_histogram):
        count = int(stats.source_remaining_ship_count_histogram[remaining])
        pct = 100.0 * float(count) / float(spawned) if spawned > 0 else 0.0
        lines.append(f"{int(remaining)} {count} {pct:.6f}")
    if len(stats.source_remaining_ship_count_histogram) == 0:
        lines.append("none 0 0.000000")
    lines.extend(
        [
            "",
            "source_sent_ship_percent_bucket_histogram",
            "sent_percent_bucket count percent_of_spawned",
        ]
    )
    for percent_bucket in sorted(stats.source_sent_ship_percent_bucket_histogram):
        count = int(stats.source_sent_ship_percent_bucket_histogram[percent_bucket])
        pct = 100.0 * float(count) / float(spawned) if spawned > 0 else 0.0
        lines.append(f"{int(percent_bucket)} {count} {pct:.6f}")
    if len(stats.source_sent_ship_percent_bucket_histogram) == 0:
        lines.append("none 0 0.000000")
    lines.append("")
    return "\n".join(lines)


def _bc_write_fleet_stats_file(
    *,
    path: Path,
    stats: BcFleetStats,
    files_done: int,
    n_written: int,
    n_skipped: int,
    n_write_failed: int,
    final: bool,
) -> None:
    path.write_text(
        _bc_fleet_stats_text(
            stats=stats,
            files_done=files_done,
            n_written=n_written,
            n_skipped=n_skipped,
            n_write_failed=n_write_failed,
            final=final,
        ),
        encoding="utf-8",
    )


def _bc_compact_seat_by_policy_slot(*, num_agents: int) -> dict[int, int]:
    return {
        orbit_policy_slot_for_compact_agent(seat, num_agents): seat
        for seat in range(num_agents)
    }


def _bc_empty_tape_head_payload(
    *,
    num_agents: int,
) -> dict[str, dict[str, Any]]:
    na = int(num_agents)
    n_planets = int(ORBIT_MAX_PLANETS)
    return {
        "final_policy": {
            "prediction": [[[] for _ in range(n_planets)] for _ in range(na)],
            "target": [[0.0] * n_planets for _ in range(na)],
            "valid": [[False] * n_planets for _ in range(na)],
        },
    }


def _bc_fill_tape_head_row(
    heads: dict[str, dict[str, Any]],
    *,
    seat: int,
    pred: dict[str, Any],
    target: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
) -> None:
    for head_name in ("final_policy",):
        pred_cell = pred[head_name]
        assert isinstance(pred_cell, list), type(pred_cell)
        pred_row = pred_cell
        target_row = target[head_name].to(dtype=torch.float32).tolist()
        valid_row = valid[head_name].to(dtype=torch.bool).tolist()
        assert len(pred_row) == ORBIT_MAX_PLANETS, (head_name, len(pred_row))
        assert len(target_row) == ORBIT_MAX_PLANETS, (head_name, len(target_row))
        assert len(valid_row) == ORBIT_MAX_PLANETS, (head_name, len(valid_row))
        payload = heads[head_name]
        payload_pred = payload["prediction"]
        payload_target = payload["target"]
        payload_valid = payload["valid"]
        assert isinstance(payload_pred, list) and isinstance(payload_target, list)
        assert isinstance(payload_valid, list)
        payload_pred[seat] = pred_row
        payload_target[seat] = [float(x) for x in target_row]
        payload_valid[seat] = [bool(x) for x in valid_row]


def _bc_final_policy_rows_for_submission_result(
    result: Any,
    *,
    rl_class_row: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    seat = int(result.player_index)
    n_agents = int(result.num_agents)
    assert 0 <= seat < n_agents, (seat, n_agents)
    n = int(ORBIT_MAX_PLANETS)
    rl_cls = rl_class_row.to(dtype=torch.int64).cpu()
    assert rl_cls.shape == (n,), rl_cls.shape
    final_logits_by_action = result.model_output["final_policy_logits_LEARN"]
    assert isinstance(final_logits_by_action, dict), type(final_logits_by_action)
    assert tuple(final_logits_by_action.keys()) == ("spawn_fleet",), (
        tuple(final_logits_by_action.keys()),
    )
    final_logits = final_logits_by_action["spawn_fleet"]
    assert isinstance(final_logits, torch.Tensor)
    final_logits = final_logits.detach().cpu()
    assert final_logits.shape == (1, 1, n, ORBIT_PER_PLANET_MOVE_CLASSES), final_logits.shape
    batch_obs = result.batch_obs
    planet_mask = batch_obs["orbit_planet_mask"]
    available_action_mask = batch_obs["available_action_mask"]
    assert isinstance(planet_mask, torch.Tensor)
    assert isinstance(available_action_mask, torch.Tensor)
    assert planet_mask.shape == (1, 1, n), planet_mask.shape
    assert available_action_mask.shape == (
        1,
        1,
        n,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), available_action_mask.shape
    assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
    available = available_action_mask.detach().cpu() > 0
    valid_policy = (planet_mask[0, 0].detach().cpu() > 0.5).to(dtype=torch.bool)
    valid_policy = valid_policy & (available[0, 0].sum(dim=-1) > 1)
    neg_large = torch.finfo(final_logits.dtype).min / 16.0
    probs = torch.softmax(final_logits.masked_fill(~available, neg_large), dim=-1)[0, 0]
    k = min(5, ORBIT_PER_PLANET_MOVE_CLASSES)
    vals, idx = probs.topk(k, dim=-1)
    final_pred: list[list[list[float]]] = []
    for i in range(n):
        row: list[list[float]] = []
        for j in range(k):
            row.append([float(idx[i, j].item()), float(vals[i, j].item())])
        final_pred.append(row)
    pred = {
        "final_policy": final_pred,
    }
    target = {
        "final_policy": rl_cls.to(dtype=torch.float32),
    }
    valid = {
        "final_policy": valid_policy,
    }
    return pred, target, valid


def _bc_submission_infer_tape_heads_by_frame(
    ep: Mapping[str, object],
    *,
    bc_payload: Mapping[str, Any],
    checkpoint_path: Path,
    wall_profiler: WallTreeProfiler | None = None,
) -> list[dict[str, dict[str, Any]] | None]:
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 2
    row0 = steps[0]
    assert isinstance(row0, list)
    n_agents = len(row0)
    seat_by_policy_slot = _bc_compact_seat_by_policy_slot(num_agents=n_agents)
    rti = bc_payload["replay_transition_index"]
    ros = bc_payload["replay_obs_step_before_action"]
    policy_slots = bc_payload["bc_policy_slot"]
    rl_cls = bc_payload["rl_per_planet_move_class"]
    assert isinstance(rti, torch.Tensor)
    assert isinstance(ros, torch.Tensor)
    assert isinstance(policy_slots, torch.Tensor)
    assert isinstance(rl_cls, torch.Tensor)
    t = int(bc_payload["num_bc_timesteps"])
    assert rti.shape == (t,) and ros.shape == (t,) and policy_slots.shape == (t,), (
        rti.shape,
        ros.shape,
        policy_slots.shape,
        t,
    )
    assert rl_cls.shape == (t, 1, ORBIT_MAX_PLANETS), rl_cls.shape
    row_by_frame_seat: dict[tuple[int, int], int] = {}
    with profiler_span(wall_profiler, "tape_infer_index"):
        for ti in range(t):
            transition_index = int(rti[ti].item())
            pre_frame_index = transition_index - 1
            assert pre_frame_index == int(ros[ti].item()), (
                transition_index,
                pre_frame_index,
                int(ros[ti].item()),
            )
            policy_slot = int(policy_slots[ti].item())
            assert policy_slot in seat_by_policy_slot, (policy_slot, seat_by_policy_slot)
            seat = seat_by_policy_slot[policy_slot]
            key = (pre_frame_index, seat)
            assert key not in row_by_frame_seat, key
            row_by_frame_seat[key] = ti
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "CUDA is required for --bc-infer-checkpoint analyzer inference"
    with profiler_span(wall_profiler, "tape_infer_model_load"):
        model = load_submission_model(checkpoint_path, device=device)
    with profiler_span(wall_profiler, "tape_infer_runner_init"):
        runners = [
            OrbitSubmissionRunner(
                model=model,
                device=device,
            )
            for _seat in range(n_agents)
        ]
    out: list[dict[str, dict[str, Any]] | None] = [None for _frame in range(len(steps))]
    cfg = ep["configuration"]
    for frame_index in range(len(steps) - 1):
        with profiler_span(wall_profiler, "tape_infer_frame"):
            row = steps[frame_index]
            assert isinstance(row, list) and len(row) == n_agents
            heads = _bc_empty_tape_head_payload(num_agents=n_agents)
            have_head = False
            for seat in range(n_agents):
                st = row[seat]
                assert isinstance(st, dict)
                obs = st["observation"]
                assert isinstance(obs, dict)
                with profiler_span(wall_profiler, "submission_runner_step"):
                    result = runners[seat].step(
                        obs,
                        cfg,
                        include_policy_logits_pre_action_mask=True,
                    )
                key = (frame_index, seat)
                if key not in row_by_frame_seat:
                    continue
                with profiler_span(wall_profiler, "tape_infer_head_rows"):
                    ti = row_by_frame_seat[key]
                    pred, target, valid = _bc_final_policy_rows_for_submission_result(
                        result,
                        rl_class_row=rl_cls[ti, 0],
                    )
                    _bc_fill_tape_head_row(
                        heads,
                        seat=seat,
                        pred=pred,
                        target=target,
                        valid=valid,
                    )
                    have_head = True
            if have_head:
                out[frame_index] = heads
        if wall_profiler is not None and (frame_index + 1) % 10 == 0:
            wall_profiler.summary_stdout(
                f"bc_tape_infer step_count={frame_index + 1}",
                line_prefix="WALL_TREE_ANALYZE ",
                file=sys.stderr,
            )
    return out


def _bc_episode_for_tape_from_payload(
    ep: Mapping[str, Any],
    *,
    bc_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    early_stop_info = bc_payload.get("bc_early_stop_info")
    if early_stop_info is None:
        return ep
    assert isinstance(early_stop_info, Mapping), type(early_stop_info)
    steps = ep["steps"]
    assert isinstance(steps, list) and len(steps) >= 2
    transition_index = int(early_stop_info["transition_index"])
    cutoff = transition_index + 1
    assert 2 <= cutoff <= len(steps), (transition_index, cutoff, len(steps))
    ep_for_tape = dict(ep)
    ep_for_tape["steps"] = steps[:cutoff]
    return ep_for_tape


@dataclass(frozen=True)
class BcDatasetFileJob:
    fpath: Path
    outdir: Path
    hz_bc: int
    bc_teams_filter: tuple[str, ...]
    bc_submission_pairs: tuple[tuple[str, int], ...]
    only_2p: bool
    bc_loss_winner_only: bool
    bc_loss_teams_filter_seats_only: bool
    dump_data: bool
    tape: bool
    tape_out: Path | None
    hit_horizon: int | None
    tape_name: str
    profile_tape: bool
    bc_infer_checkpoint: Path | None
    profile: bool
    bc_fleet_stats: bool
    bc_dump_inputs: bool


@dataclass(frozen=True)
class BcDatasetFileResult:
    outcome: Literal["written", "skipped", "failed"]
    stderr_lines: tuple[str, ...]
    stdout_lines: tuple[str, ...]
    errors_txt_block: str | None
    fleet_stats: BcFleetStats | None = None


def _bc_dataset_process_one_file_inner(job: BcDatasetFileJob) -> BcDatasetFileResult:
    """Export one replay to ``.pt`` (and optional tape / dump). Safe for worker processes."""
    fpath = job.fpath
    stderr_lines: list[str] = []
    stdout_lines: list[str] = []
    wall_profiler = WallTreeProfiler() if bool(job.profile) else None
    stem = episode_pt_stem(fpath)
    dest_pt = job.outdir / f"{stem}.pt"

    if dest_pt.is_file():
        stderr_lines.append(f"bc_dataset_skip_existing_pt path={fpath} out_pt={dest_pt}")
        return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)

    with profiler_span(wall_profiler, "load_replay"):
        ep = try_load_orbit_wars_kaggle_replay(fpath, container="json")
    if ep is None:
        return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
    with profiler_span(wall_profiler, "filters"):
        if bool(job.only_2p):
            row0 = ep["steps"]
            assert isinstance(row0, list) and len(row0) >= 1
            r0 = row0[0]
            assert isinstance(r0, list)
            na = len(r0)
            if na != 2:
                stderr_lines.append(f"bc_dataset_skip_not_2p path={fpath} num_agents={na}")
                return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
        if len(job.bc_teams_filter) > 0 and not episode_matches_teams_filter(ep, job.bc_teams_filter):
            stderr_lines.append(
                f"bc_dataset_skip_teams_filter path={fpath} "
                f"team_names={kaggle_replay_team_names(ep)!r} filter={list(job.bc_teams_filter)!r}"
            )
            return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
        if not _bc_replay_matches_loss_seat_filters(
            ep,
            winner_only=bool(job.bc_loss_winner_only),
            teams_filter_seats_only=bool(job.bc_loss_teams_filter_seats_only),
            teams_filter=job.bc_teams_filter,
        ):
            stderr_lines.append(
                f"bc_dataset_skip_loss_seat_filters path={fpath} "
                f"team_names={kaggle_replay_team_names(ep)!r} rewards={ep['rewards']!r} "
                f"winner_only={bool(job.bc_loss_winner_only)} "
                f"teams_filter_seats_only={bool(job.bc_loss_teams_filter_seats_only)} "
                f"filter={list(job.bc_teams_filter)!r}"
            )
            return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
        if len(job.bc_submission_pairs) > 0:
            meta_path = _kaggle_replay_sidecar_metadata_path(fpath)
            assert meta_path.is_file(), (
                f"--bc-submission-pair / --bc-submission-id requires sidecar metadata next to the replay JSON "
                f"(expected {meta_path} for replay {fpath})"
            )
            meta_raw = load_json_maybe_gzip(meta_path)
            assert isinstance(meta_raw, dict), (fpath, meta_path, type(meta_raw).__name__)
            got_triples: list[tuple[str, int, int | None]] = []
            for meta_team_name, want_sid in job.bc_submission_pairs:
                got_sid = public_leaderboard_submission_id_for_team_name(
                    meta_raw,
                    team_name=meta_team_name,
                    metadata_path=meta_path,
                )
                got_triples.append((meta_team_name, want_sid, got_sid))
            missing_team_names = tuple(t for t, _w, g in got_triples if g is None)
            if len(missing_team_names) > 0:
                stderr_lines.append(
                    f"bc_dataset_skip_submission_pair_team_missing path={fpath} "
                    f"metadata_path={meta_path} missing_team_names={missing_team_names!r} "
                    f"want_team_submission_pairs={[(t, w) for t, w, _ in got_triples]!r} "
                    f"metadata_team_submission_ids={[(t, g) for t, _, g in got_triples]!r}"
                )
                return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
            matched = any(want_sid == got_sid for (_tn, want_sid, got_sid) in got_triples)
            if not matched:
                stderr_lines.append(
                    f"bc_dataset_skip_submission_id path={fpath} "
                    f"want_team_submission_pairs={[(t, w) for t, w, _ in got_triples]!r} "
                    f"metadata_team_submission_ids={[(t, g) for t, _, g in got_triples]!r}"
                )
                return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
    dump_path = _bc_episode_human_dump_path(dest_pt) if bool(job.dump_data) else None
    input_dump_path = _bc_policy_input_dump_path(dest_pt) if bool(job.bc_dump_inputs) else None
    bc_payload: dict[str, Any]
    fleet_stats = BcFleetStats() if bool(job.bc_fleet_stats) else None
    try:
        bc_payload = write_behavior_clone_episode_pt(
            ep,
            dest_pt,
            replay_episode_id=kaggle_replay_episode_id_from_path(fpath),
            hit_horizon=job.hz_bc,
            bc_loss_player_mask_winner_only=bool(job.bc_loss_winner_only),
            bc_loss_player_mask_teams_filter_seats_only=bool(job.bc_loss_teams_filter_seats_only),
            teams_filter=job.bc_teams_filter,
            human_dump=bool(job.dump_data),
            human_dump_path=dump_path,
            human_dump_source=fpath if bool(job.dump_data) else None,
            wall_profiler=wall_profiler,
            profile_summary_every_steps=10,
            fleet_stats=fleet_stats,
        )
    except BcEpisodeEmptyLossMaskError:
        stderr_lines.append(f"bc_dataset_skip_empty_bc_loss_player_mask path={fpath}")
        return BcDatasetFileResult("skipped", tuple(stderr_lines), tuple(stdout_lines), None)
    except Exception:
        tb = traceback.format_exc()
        block = (
            f"==== {datetime.now().isoformat(timespec='seconds')} "
            f"path={fpath} out_pt={dest_pt} ====\n"
            f"{tb}"
            + ("" if tb.endswith("\n") else "\n")
        )
        return BcDatasetFileResult("failed", tuple(stderr_lines), tuple(stdout_lines), block)

    if input_dump_path is not None:
        with profiler_span(wall_profiler, "bc_policy_input_dump"):
            write_behavior_clone_policy_input_dump(
                payload=bc_payload,
                dest=input_dump_path,
            )

    if bool(job.tape):
        ep_for_tape = _bc_episode_for_tape_from_payload(ep, bc_payload=bc_payload)
        tape_root = (
            job.tape_out if job.tape_out is not None else _DEFAULT_ANALYZE_TAPE_DIR
        ).resolve()
        hz_tape = (
            int(job.hit_horizon)
            if job.hit_horizon is not None
            else default_replay_tape_hit_horizon(ep_for_tape["configuration"])
        )
        tape_name = str(job.tape_name)
        if tape_name == _DEFAULT_ANALYZE_TAPE_NAME:
            tape_name = stem
        player_supervised_heads_by_frame = (
            None
            if job.bc_infer_checkpoint is None
            else _bc_submission_infer_tape_heads_by_frame(
                ep_for_tape,
                bc_payload=bc_payload,
                checkpoint_path=job.bc_infer_checkpoint,
                wall_profiler=wall_profiler,
            )
        )
        tape_path = write_kaggle_replay_orbit_tape(
            ep_for_tape,
            tape_root=tape_root,
            tape_name=tape_name,
            hit_horizon=hz_tape,
            player_supervised_heads_by_frame=player_supervised_heads_by_frame,
            profile=bool(job.profile_tape) or bool(job.profile),
            wall_profiler=wall_profiler if bool(job.profile) else None,
            profile_summary_every_steps=1 if bool(job.profile_tape) else 10,
        )
        stdout_lines.append(f"bc_tape_frames_path={tape_path} stem={stem}")
    if dump_path is not None:
        stdout_lines.append(f"bc_human_dump_path={dump_path} stem={stem}")
    if input_dump_path is not None:
        stdout_lines.append(f"bc_policy_input_dump_path={input_dump_path} stem={stem}")
    if wall_profiler is not None:
        buf = io.StringIO()
        wall_profiler.summary_stdout(
            f"bc_dataset_file path={fpath}",
            line_prefix="WALL_TREE_ANALYZE ",
            file=buf,
        )
        stderr_lines.extend(buf.getvalue().splitlines())
    errors_txt_block = None
    early_stop_mismatch = bc_payload.get("bc_early_stop_real_winner_mismatch")
    if early_stop_mismatch is not None:
        assert isinstance(early_stop_mismatch, dict), type(early_stop_mismatch)
        errors_txt_block = (
            f"==== {datetime.now().isoformat(timespec='seconds')} "
            f"BC_EARLY_STOP_REAL_WINNER_MISMATCH path={fpath} out_pt={dest_pt} ====\n"
            f"replay_episode_id={kaggle_replay_episode_id_from_path(fpath)} "
            f"transition_index={int(early_stop_mismatch['transition_index'])} "
            f"replay_obs_step={int(early_stop_mismatch['replay_obs_step'])} "
            f"heuristic_winner_seats={tuple(early_stop_mismatch['heuristic_winner_seats'])!r} "
            f"real_winner_seats={tuple(early_stop_mismatch['real_winner_seats'])!r}\n"
        )
    return BcDatasetFileResult(
        "written",
        tuple(stderr_lines),
        tuple(stdout_lines),
        errors_txt_block,
        fleet_stats,
    )


def _bc_dataset_process_one_file(job: BcDatasetFileJob) -> BcDatasetFileResult:
    try:
        return _bc_dataset_process_one_file_inner(job)
    except Exception:
        fpath = job.fpath
        out_pt_hint = job.outdir / f"{fpath.name}.pt"
        tb = traceback.format_exc()
        block = (
            f"==== {datetime.now().isoformat(timespec='seconds')} "
            f"path={fpath} out_pt_hint={out_pt_hint} ====\n"
            f"{tb}"
            + ("" if tb.endswith("\n") else "\n")
        )
        stderr_lines = (f"bc_dataset_file_failed path={fpath} out_pt_hint={out_pt_hint}",)
        return BcDatasetFileResult("failed", stderr_lines, (), block)


def _bc_dataset_worker_loop(job_queue: Any, result_queue: Any) -> None:
    while True:
        job = job_queue.get()
        if job is None:
            result_queue.put(None)
            return
        assert isinstance(job, BcDatasetFileJob), type(job)
        result_queue.put(_bc_dataset_process_one_file(job))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Inspect Kaggle-format Orbit Wars replay JSON.")
    p.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Path to one episode JSON (or .json.gz). If unset, uses the packaged sample.",
    )
    p.add_argument(
        "--container",
        choices=("json", "jsonl"),
        default="json",
        help="json: single JSON object; jsonl: first non-empty line is one episode object.",
    )
    p.add_argument(
        "--seat",
        type=int,
        default=0,
        help="Agent index for --print-first-pair.",
    )
    p.add_argument(
        "--fleet-spawn-gt",
        action="store_true",
        help="Print per-frame Kaggle moves and spawn_gt (from,target,ships) via C++ fleet hit traces.",
    )
    p.add_argument(
        "--hit-horizon",
        type=int,
        default=None,
        help="Horizon for C++ fleet hit simulation: "
        "--fleet-spawn-gt default ORBIT_PLANET_ARRIVAL_HORIZON; "
        "--tape / --tape-out default configuration episodeSteps.",
    )
    p.add_argument(
        "--tape",
        action="store_true",
        help=f"Write orbit debug tape under {_DEFAULT_ANALYZE_TAPE_DIR}/<{_DEFAULT_ANALYZE_TAPE_NAME}> unless --tape-out/--tape-name override. "
        f"With --bc-dataset, runs per episode; if --tape-name is the default ({_DEFAULT_ANALYZE_TAPE_NAME!r}), the tape folder name is the episode stem.",
    )
    p.add_argument(
        "--tape-out",
        type=Path,
        default=None,
        help="Tape root directory (creates <tape-out>/<tape-name>/frames.jsonl). With --tape only, default is repo outputs/analyze/.",
    )
    p.add_argument(
        "--tape-name",
        default=_DEFAULT_ANALYZE_TAPE_NAME,
        help=f"Subfolder under --tape-out (default: {_DEFAULT_ANALYZE_TAPE_NAME}).",
    )
    p.add_argument(
        "--print-first-pair",
        action="store_true",
        help="Print keys of first observation and first action for --seat.",
    )
    p.add_argument(
        "--profile-tape",
        action="store_true",
        help="When writing tape: WallTreeProfiler tree per frame to stderr (prefix WALL_TREE_TAPE).",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable analyzer WallTreeProfiler around BC dataset work; emits cumulative summaries every 10 replay steps.",
    )
    p.add_argument(
        "--bc-dataset",
        action="store_true",
        help="Write one .pt per episode under --bc-output-dir from *.json / *.json.gz in --bc-episodes-dir, "
        "or from a single path via --bc-replay. "
        "Optional: --dump-data (text alongside .pt), --tape (orbit frames.jsonl per episode).",
    )
    p.add_argument(
        "--bc-episodes-dir",
        type=Path,
        default=None,
        help="Directory of Kaggle Orbit Wars episode JSON files (used with --bc-dataset; mutually exclusive with --bc-replay).",
    )
    p.add_argument(
        "--bc-replay",
        type=Path,
        default=None,
        metavar="PATH",
        help="With --bc-dataset: process only this episode .json or .json.gz (no directory scan). Mutually exclusive with --bc-episodes-dir.",
    )
    p.add_argument(
        "--bc-output-dir",
        type=Path,
        default=None,
        help="Output directory for behavior cloning .pt files (used with --bc-dataset).",
    )
    p.add_argument(
        "--bc-hit-horizon",
        type=int,
        default=None,
        help="Hit-trace horizon for rl_per_planet_move_class in --bc-dataset; default same as --fleet-spawn-gt (ORBIT_PLANET_ARRIVAL_HORIZON).",
    )
    p.add_argument(
        "--bc-max-episodes",
        type=int,
        default=None,
        help="With --bc-dataset: stop after this many successfully written .pt files (skipped files do not count); default: no limit.",
    )
    p.add_argument(
        "--dump-data",
        action="store_true",
        help="With --bc-dataset: write <stem>.bc_dump.txt next to each .pt (human-readable BC tensors, masks, RL classes, alignment hints).",
    )
    p.add_argument(
        "--bc-dump-inputs",
        action="store_true",
        help="With --bc-dataset: write <stem>.bc_policy_inputs.pt next to each .pt for policy input dump comparisons.",
    )
    p.add_argument(
        "--bc-teams-filter",
        action="append",
        default=None,
        metavar="NAME",
        help="With --bc-dataset: only write .pt when replay info.TeamNames contains at least one of these "
        "strings (exact match; repeat flag for multiple names).",
    )
    p.add_argument(
        "--bc-submission-pair",
        action="append",
        nargs=2,
        default=None,
        metavar=("TEAM_NAME", "SUBMISSION_ID"),
        help="With --bc-dataset: require sidecar <episode_stem>_metadata.json; for each TEAM_NAME resolve "
        "leaderboard.publicLeaderboard submissionId; keep the replay if any pair matches its SUBMISSION_ID (int). "
        "Repeat the flag for multiple pairs (OR). Mutually exclusive with --bc-submission-id.",
    )
    p.add_argument(
        "--bc-submission-id",
        type=int,
        default=None,
        metavar="ID",
        help="With --bc-dataset: same as one --bc-submission-pair with the single --bc-teams-filter NAME "
        "(requires exactly one --bc-teams-filter). Mutually exclusive with --bc-submission-pair.",
    )
    p.add_argument(
        "--bc-loss-winner-only",
        action="store_true",
        help="With --bc-dataset: bc_loss_player_mask is non-zero only on seats with final envelope reward +1.",
    )
    p.add_argument(
        "--bc-loss-teams-filter-seats-only",
        action="store_true",
        help="With --bc-dataset: bc_loss_player_mask is non-zero only on seats whose TeamNames string is listed "
        "in --bc-teams-filter (requires --bc-teams-filter; stacks with --bc-loss-winner-only). "
        "Episodes with all-zero mask after filters are skipped.",
    )
    p.add_argument(
        "--only-2p",
        action="store_true",
        help="With --bc-dataset: skip episodes that are not exactly two agents (len(steps[0]) != 2).",
    )
    p.add_argument(
        "--bc-workers",
        type=int,
        default=None,
        help="With --bc-dataset: number of worker processes. After listing episode files they are shuffled once, "
        "then worker processes pull one file at a time from a shared queue and return one result per replay "
        "through a result queue, so the main process can aggregate progress immediately. "
        "Default: int(os.environ['N_WORKERS']) when set and non-empty, otherwise 1. "
        "When --bc-max-episodes is set, runs sequentially in the main process "
        "(parallel disabled).",
    )
    p.add_argument(
        "--bc-fleet-hit-steps-log-every",
        type=int,
        default=0,
        metavar="N",
        help="With --bc-dataset: in the main process, aggregate BC GT active-seat spawned-fleet hit_steps "
        "ship-count, source-remaining-ship-count, and source-sent-ship-percent-bucket histograms and "
        "overwrite bc_gt_active_fleet_stats.txt every N processed replay files; "
        "0 disables.",
    )
    p.add_argument(
        "--bc-infer-checkpoint",
        type=Path,
        default=None,
        help="With --bc-dataset --tape: load this submission/BC checkpoint, run the shared submission inference path on each replay pre-action step, and overlay GT/PREDICTION for the flat spawn_fleet policy. Without BC team/submission filters, this runs over all active replay seats.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if bool(args.dump_data) and not bool(args.bc_dataset):
        p.error("--dump-data requires --bc-dataset")
    if bool(args.bc_dump_inputs) and not bool(args.bc_dataset):
        p.error("--bc-dump-inputs requires --bc-dataset")
    if bool(args.only_2p) and not bool(args.bc_dataset):
        p.error("--only-2p requires --bc-dataset")
    if args.bc_infer_checkpoint is not None and not bool(args.bc_dataset):
        p.error("--bc-infer-checkpoint requires --bc-dataset")
    if args.bc_infer_checkpoint is not None and not bool(args.tape):
        p.error("--bc-infer-checkpoint requires --tape")
    if args.bc_submission_id is not None and not bool(args.bc_dataset):
        p.error("--bc-submission-id requires --bc-dataset")
    if args.bc_submission_pair is not None and not bool(args.bc_dataset):
        p.error("--bc-submission-pair requires --bc-dataset")
    if (
        args.bc_submission_id is not None
        and args.bc_submission_pair is not None
    ):
        p.error("--bc-submission-id is mutually exclusive with --bc-submission-pair")
    if args.bc_replay is not None and not bool(args.bc_dataset):
        p.error("--bc-replay requires --bc-dataset")

    if bool(args.bc_dataset):
        if args.bc_output_dir is None:
            p.error("--bc-dataset requires --bc-output-dir")
        if (args.bc_episodes_dir is None) == (args.bc_replay is None):
            p.error("--bc-dataset requires exactly one of --bc-episodes-dir or --bc-replay")
        bc_has_team_filter = args.bc_teams_filter is not None and len(args.bc_teams_filter) > 0
        bc_infer_all_replay_seats = args.bc_infer_checkpoint is not None and not bc_has_team_filter
        if bool(args.bc_loss_teams_filter_seats_only) and (
            args.bc_teams_filter is None or len(args.bc_teams_filter) == 0
        ) and not bc_infer_all_replay_seats:
            p.error("--bc-loss-teams-filter-seats-only requires at least one --bc-teams-filter NAME")
        if args.bc_submission_id is not None:
            if args.bc_teams_filter is None or len(args.bc_teams_filter) != 1:
                p.error("--bc-submission-id requires exactly one --bc-teams-filter NAME")
        if args.bc_workers is not None and int(args.bc_workers) < 1:
            p.error("--bc-workers must be >= 1")
        if int(args.bc_fleet_hit_steps_log_every) < 0:
            p.error("--bc-fleet-hit-steps-log-every must be >= 0")
        bc_infer_checkpoint = (
            args.bc_infer_checkpoint.expanduser().resolve()
            if args.bc_infer_checkpoint is not None
            else None
        )
        if bc_infer_checkpoint is not None:
            assert bc_infer_checkpoint.is_file(), bc_infer_checkpoint

        outdir = args.bc_output_dir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        if args.bc_replay is not None:
            one = args.bc_replay.expanduser().resolve()
            assert one.is_file(), one
            assert is_kaggle_replay_episode_json_path(one), (
                f"--bc-replay must be replay_<episode_id>.json or .json.gz, got {one.name!r}"
            )
            files = [one]
        else:
            epdir = args.bc_episodes_dir.expanduser().resolve()
            assert epdir.is_dir(), epdir
            files = list(iter_episode_json_files(epdir))
        random.shuffle(files)
        hz_bc = (
            int(args.bc_hit_horizon)
            if args.bc_hit_horizon is not None
            else default_hit_horizon()
        )
        bc_teams_filter = tuple(args.bc_teams_filter) if args.bc_teams_filter is not None else ()
        bc_loss_teams_filter_seats_only = (
            bool(args.bc_loss_teams_filter_seats_only)
            and not bool(bc_infer_all_replay_seats)
        )
        pair_rows: list[tuple[str, int]] = []
        if args.bc_submission_pair is not None:
            for row in args.bc_submission_pair:
                assert len(row) == 2, row
                team_nm, sid_raw = row[0], row[1]
                pair_rows.append((str(team_nm), int(sid_raw)))
        if args.bc_submission_id is not None:
            assert len(pair_rows) == 0, "mutually exclusive submission filters"
            pair_rows.append((str(args.bc_teams_filter[0]), int(args.bc_submission_id)))
        bc_submission_pairs = tuple(pair_rows)
        mx = int(args.bc_max_episodes) if args.bc_max_episodes is not None else None
        assert mx is None or mx >= 1, mx
        fleet_stats_write_every = int(args.bc_fleet_hit_steps_log_every)
        assert fleet_stats_write_every >= 0, fleet_stats_write_every
        n_workers_req = _bc_workers_from_env_or_arg(args.bc_workers)
        use_pool = mx is None and n_workers_req > 1 and len(files) > 1
        n_workers = n_workers_req if use_pool else 1
        n_written = 0
        n_skipped = 0
        n_write_failed = 0
        n_processed = 0
        fleet_stats = BcFleetStats()
        fleet_stats_path = outdir / "bc_gt_active_fleet_stats.txt"
        errors_path = outdir / "errors.txt"
        tape_out_resolved = (
            args.tape_out.expanduser().resolve() if args.tape_out is not None else None
        )
        hit_horizon_arg = int(args.hit_horizon) if args.hit_horizon is not None else None

        def _make_bc_job(fpath: Path) -> BcDatasetFileJob:
            return BcDatasetFileJob(
                fpath=fpath,
                outdir=outdir,
                hz_bc=hz_bc,
                bc_teams_filter=bc_teams_filter,
                bc_submission_pairs=bc_submission_pairs,
                only_2p=bool(args.only_2p),
                bc_loss_winner_only=bool(args.bc_loss_winner_only),
                bc_loss_teams_filter_seats_only=bc_loss_teams_filter_seats_only,
                dump_data=bool(args.dump_data),
                tape=bool(args.tape),
                tape_out=tape_out_resolved,
                hit_horizon=hit_horizon_arg,
                tape_name=str(args.tape_name),
                profile_tape=bool(args.profile_tape),
                bc_infer_checkpoint=bc_infer_checkpoint,
                profile=bool(args.profile),
                bc_fleet_stats=fleet_stats_write_every > 0,
                bc_dump_inputs=bool(args.bc_dump_inputs),
            )

        def _apply_bc_result(res: BcDatasetFileResult) -> None:
            nonlocal n_written, n_skipped, n_write_failed, n_processed
            for line in res.stderr_lines:
                print(line, file=sys.stderr)
            for line in res.stdout_lines:
                print(line)
            if res.errors_txt_block is not None:
                with open(errors_path, "a", encoding="utf-8") as ef:
                    ef.write(res.errors_txt_block)
            n_processed += 1
            if res.outcome == "written":
                n_written += 1
                _bc_fleet_stats_add(fleet_stats, res.fleet_stats)
            elif res.outcome == "skipped":
                n_skipped += 1
            else:
                assert res.outcome == "failed", res.outcome
                n_write_failed += 1
            if fleet_stats_write_every > 0 and n_processed % fleet_stats_write_every == 0:
                _bc_write_fleet_stats_file(
                    path=fleet_stats_path,
                    stats=fleet_stats,
                    files_done=n_processed,
                    n_written=n_written,
                    n_skipped=n_skipped,
                    n_write_failed=n_write_failed,
                    final=False,
                )

        if fleet_stats_write_every > 0 and fleet_stats_path.is_file():
            fleet_stats_path.unlink()

        if use_pool:
            if errors_path.is_file():
                errors_path.unlink()
            ctx = mp.get_context()
            job_queue = ctx.Queue()
            result_queue = ctx.Queue()
            workers = [
                ctx.Process(target=_bc_dataset_worker_loop, args=(job_queue, result_queue))
                for _ in range(n_workers)
            ]
            for worker in workers:
                worker.start()
            for fpath in files:
                job_queue.put(_make_bc_job(fpath))
            for _ in range(n_workers):
                job_queue.put(None)
            n_workers_done = 0
            while n_workers_done < n_workers:
                res = result_queue.get()
                if res is None:
                    n_workers_done += 1
                    continue
                assert isinstance(res, BcDatasetFileResult), type(res)
                _apply_bc_result(res)
            for worker in workers:
                worker.join()
                assert worker.exitcode == 0, (worker.pid, worker.exitcode)
            assert n_processed == len(files), (n_processed, len(files))
        else:
            for fpath in files:
                if mx is not None and n_written >= mx:
                    break
                res = _bc_dataset_process_one_file(_make_bc_job(fpath))
                _apply_bc_result(res)

        if fleet_stats_write_every > 0:
            _bc_write_fleet_stats_file(
                path=fleet_stats_path,
                stats=fleet_stats,
                files_done=n_processed,
                n_written=n_written,
                n_skipped=n_skipped,
                n_write_failed=n_write_failed,
                final=True,
            )
            print(f"bc_gt_active_fleet_stats_path={fleet_stats_path}", file=sys.stderr, flush=True)

        extra = (
            f" bc_max_episodes={int(args.bc_max_episodes)}"
            if args.bc_max_episodes is not None
            else ""
        )
        print(
            f"bc_dataset_episodes_written={n_written} bc_dataset_files_skipped={n_skipped} "
            f"bc_dataset_write_failed={n_write_failed} bc_errors_log={errors_path} "
            f"bc_output_dir={outdir}{extra} bc_workers={n_workers}"
        )
        return

    wall_profiler = WallTreeProfiler() if bool(args.profile) else None
    with profiler_span(wall_profiler, "load_replay"):
        if args.replay is not None:
            replay_path = args.replay.resolve()
            print(f"replay_file={replay_path}")
            ep = load_orbit_wars_kaggle_replay(replay_path, container=args.container)
        else:
            sample = _default_sample_path()
            print(f"replay_file={sample} (default sample)")
            ep = load_orbit_wars_kaggle_replay(sample, container=args.container)

    print(summarize_replay(ep))

    if bool(args.tape) or args.tape_out is not None:
        tape_root = (
            args.tape_out if args.tape_out is not None else _DEFAULT_ANALYZE_TAPE_DIR
        ).resolve()
        hz_tape = (
            int(args.hit_horizon)
            if args.hit_horizon is not None
            else default_replay_tape_hit_horizon(ep["configuration"])
        )
        t_tape0 = time.perf_counter()
        out_path = write_kaggle_replay_orbit_tape(
            ep,
            tape_root=tape_root,
            tape_name=str(args.tape_name),
            hit_horizon=hz_tape,
            profile=bool(args.profile_tape) or bool(args.profile),
            wall_profiler=wall_profiler,
            profile_summary_every_steps=1 if bool(args.profile_tape) else 10,
        )
        if bool(args.profile_tape):
            print(
                f"WALL_TREE_TAPE analyze_script tape_wall_sec={time.perf_counter() - t_tape0:.3f} "
                f"tape_frames_path={out_path}",
                file=sys.stderr,
                flush=True,
            )
        print(f"tape_frames_path={out_path}")

    if args.fleet_spawn_gt:
        hz = int(args.hit_horizon) if args.hit_horizon is not None else default_hit_horizon()
        with profiler_span(wall_profiler, "fleet_spawn_gt"):
            print_replay_fleet_spawn_ground_truth(ep, horizon=hz, out=sys.stdout)
        if wall_profiler is not None:
            wall_profiler.summary_stdout(
                "analyze_replay",
                line_prefix="WALL_TREE_ANALYZE ",
                file=sys.stderr,
            )
        return

    if args.print_first_pair:
        with profiler_span(wall_profiler, "print_first_pair"):
            obs, act = next(iter_obs_action_pairs_for_seat(ep, seat=args.seat, start_frame=1))
        print("first_pair_observation_keys:", sorted(obs.keys()))
        print("first_pair_action:", act)

    if wall_profiler is not None:
        wall_profiler.summary_stdout(
            "analyze_replay",
            line_prefix="WALL_TREE_ANALYZE ",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
