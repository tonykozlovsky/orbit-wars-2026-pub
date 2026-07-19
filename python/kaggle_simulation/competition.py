from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import signal
import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_match import MatchResult, run_match


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VERSIONS_DIR = _REPO_ROOT / "versions_competition"
_STATE_SCHEMA_VERSION = 2
_INITIAL_RATING = 0.0


@dataclass(frozen=True)
class ScheduledGame:
    game_id: int
    seed: int
    version_ids: tuple[str, ...]
    archives: tuple[Path, ...]


@dataclass(frozen=True)
class CompletedGame:
    scheduled: ScheduledGame
    result: MatchResult
    wall_seconds: float


@dataclass(frozen=True)
class GameWorker:
    process: mp.Process


def _archive_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _archive_submission_supports_local_simulation(path: Path) -> bool:
    with tarfile.open(path, "r:gz") as tf:
        member = tf.extractfile("./kaggle_submission/submission.py")
        assert member is not None, path
        text = member.read().decode("utf-8")
    return "LOCAL_SIMULATION" in text and "torch.autocast" in text


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _load_state(path: Path, versions_dir: Path) -> dict[str, Any]:
    if path.is_file():
        data = json.loads(path.read_text())
        assert data["schema_version"] == _STATE_SCHEMA_VERSION, data["schema_version"]
        assert data["versions_dir"] == str(versions_dir), (data["versions_dir"], versions_dir)
        return data
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "versions_dir": str(versions_dir),
        "created_at_unix": time.time(),
        "next_game_id": 1,
        "versions": {},
        "games": [],
    }


def _scan_archives(versions_dir: Path) -> dict[str, dict[str, Any]]:
    assert versions_dir.is_dir(), versions_dir
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(versions_dir.glob("*.tar.gz")):
        stat = path.stat()
        vid = path.name
        out[vid] = {
            "path": str(path.resolve()),
            "sha256": _archive_sha256(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "supports_local_simulation_cuda": _archive_submission_supports_local_simulation(path),
        }
    return out


def _sync_versions(state: dict[str, Any], archives: Mapping[str, Mapping[str, Any]]) -> None:
    versions = state["versions"]
    assert isinstance(versions, dict), type(versions)
    for vid, meta in archives.items():
        if vid not in versions:
            versions[vid] = {
                "path": meta["path"],
                "sha256": meta["sha256"],
                "size_bytes": meta["size_bytes"],
                "mtime_ns": meta["mtime_ns"],
                "supports_local_simulation_cuda": meta["supports_local_simulation_cuda"],
                "active": True,
                "deleted": False,
                "rating": _INITIAL_RATING,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
            }
        else:
            v = versions[vid]
            assert v["sha256"] == meta["sha256"], (vid, v["sha256"], meta["sha256"])
            v["path"] = meta["path"]
            v["size_bytes"] = meta["size_bytes"]
            v["mtime_ns"] = meta["mtime_ns"]
            v["supports_local_simulation_cuda"] = meta["supports_local_simulation_cuda"]
            v["active"] = True
            v["deleted"] = False
    for vid, v in versions.items():
        if vid not in archives:
            v["active"] = False
            v["deleted"] = True


def _assert_local_simulation_archives_support_cuda(state: Mapping[str, Any]) -> None:
    if os.environ.get("LOCAL_SIMULATION", "").strip() != "1":
        return
    unsupported = sorted(
        vid
        for vid, v in state["versions"].items()
        if bool(v["active"]) and not bool(v["supports_local_simulation_cuda"])
    )
    assert not unsupported, (
        "LOCAL_SIMULATION=1 requires archives rebuilt with the LOCAL_SIMULATION CUDA submission",
        unsupported,
    )


def _select_group(state: Mapping[str, Any], *, num_players: int) -> tuple[str, ...]:
    active = sorted(
        vid for vid, v in state["versions"].items() if bool(v["active"])
    )
    assert num_players in (2, 4), num_players
    assert len(active) >= 2, active
    group = [random.choice(active) for _ in range(num_players)]
    first_distinct_index = random.randrange(num_players)
    second_distinct_index = random.randrange(num_players - 1)
    if second_distinct_index >= first_distinct_index:
        second_distinct_index += 1
    distinct_choices = [
        version_id
        for version_id in active
        if version_id != group[first_distinct_index]
    ]
    assert distinct_choices, active
    group[second_distinct_index] = random.choice(distinct_choices)
    assert len(set(group)) >= 2, group
    return tuple(group)


def _game_num_players(state: Mapping[str, Any], *, forced_num_players: int | None) -> int:
    active_count = sum(1 for v in state["versions"].values() if bool(v["active"]))
    assert active_count >= 2, active_count
    if forced_num_players is not None:
        assert forced_num_players in (2, 4), forced_num_players
        return forced_num_players
    if random.random() < 0.5:
        return 2
    return 4


def _schedule_game(
    state: dict[str, Any],
    *,
    forced_num_players: int | None,
) -> ScheduledGame:
    num_players = _game_num_players(state, forced_num_players=forced_num_players)
    group = _select_group(state, num_players=num_players)
    game_id = int(state["next_game_id"])
    seed = random.randrange(0, 2**31)
    state["next_game_id"] = game_id + 1
    version_ids_list = list(group)
    random.shuffle(version_ids_list)
    version_ids = tuple(version_ids_list)
    archives = tuple(Path(state["versions"][vid]["path"]) for vid in version_ids)
    assert len(archives) == len(version_ids)
    return ScheduledGame(
        game_id=game_id,
        seed=seed,
        version_ids=tuple(version_ids),
        archives=tuple(archives),
    )


def _rating_deltas_from_rewards(rewards: Sequence[Any]) -> list[float]:
    n = len(rewards)
    assert n in (2, 4), rewards
    winner_indices = [i for i, r in enumerate(rewards) if float(r) == 1.0]
    assert all(float(r) in (-1.0, 1.0) for r in rewards), rewards
    if len(winner_indices) == 1:
        return [float(n - 1) if i == winner_indices[0] else -1.0 for i in range(n)]
    return [0.0] * n


def _outcome_from_rating_delta(delta: float) -> str:
    if delta > 0.0:
        return "win"
    if delta < 0.0:
        return "loss"
    return "tie"


def _record_completed_game(state: dict[str, Any], completed: CompletedGame) -> None:
    ids = completed.scheduled.version_ids
    rewards = completed.result.rewards
    assert len(ids) == len(rewards), (ids, rewards)
    versions = state["versions"]
    rating_deltas = _rating_deltas_from_rewards(rewards)
    outcomes = [_outcome_from_rating_delta(delta) for delta in rating_deltas]
    for vid, delta, outcome in zip(ids, rating_deltas, outcomes, strict=True):
        v = versions[vid]
        v["rating"] = float(v["rating"]) + float(delta)
        v["games"] = int(v["games"]) + 1
        if outcome == "win":
            v["wins"] = int(v["wins"]) + 1
        elif outcome == "loss":
            v["losses"] = int(v["losses"]) + 1
        else:
            assert outcome == "tie", outcome
            v["ties"] = int(v["ties"]) + 1
    state["games"].append(
        {
            "game_id": completed.scheduled.game_id,
            "seed": completed.scheduled.seed,
            "num_players": len(ids),
            "versions": list(ids),
            "rewards": list(rewards),
            "rating_deltas": rating_deltas,
            "statuses": list(completed.result.statuses),
            "outcomes": outcomes,
            "result_seed": completed.result.seed,
            "wall_seconds": completed.wall_seconds,
            "replay_path": None,
        }
    )


def _leaderboard_rows(state: Mapping[str, Any], *, include_deleted: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vid, v in state["versions"].items():
        if not include_deleted and not bool(v["active"]):
            continue
        games = int(v["games"])
        rows.append(
            {
                "version": vid,
                "active": bool(v["active"]),
                "status": "active" if bool(v["active"]) else "deleted",
                "rating": float(v["rating"]),
                "games": games,
                "wins": int(v["wins"]),
                "ties": int(v["ties"]),
                "losses": int(v["losses"]),
                "sha256": str(v["sha256"])[:12],
            }
        )
    rows.sort(key=lambda r: (-float(r["rating"]), str(r["version"])))
    return rows


def _leaderboard_text(state: Mapping[str, Any], *, include_deleted: bool) -> str:
    rows = _leaderboard_rows(state, include_deleted=include_deleted)
    header = (
        f"{'rank':>4s} "
        f"{'rating':>8s} "
        f"{'games':>5s} "
        f"{'wins':>4s} "
        f"{'ties':>4s} "
        f"{'losses':>6s} "
        f"{'status':>7s} "
        f"{'version':<28s} "
        f"{'sha256':<12s}"
    )
    lines = [
        (
            f"games={len(state['games'])} "
            f"next_game_id={state['next_game_id']} "
            f"local_simulation={os.environ.get('LOCAL_SIMULATION', '').strip() == '1'}"
        ),
        header,
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank:4d} "
            f"{row['rating']:8.1f} "
            f"{row['games']:5d} "
            f"{row['wins']:4d} "
            f"{row['ties']:4d} "
            f"{row['losses']:6d} "
            f"{row['status']:>7s} "
            f"{row['version']:<28s} "
            f"{row['sha256']:<12s}"
        )
    return "\n".join(lines) + "\n"


def _write_leaderboards(path: Path, state: Mapping[str, Any]) -> None:
    _atomic_write_text(path, _leaderboard_text(state, include_deleted=False))
    all_path = path.with_name(path.stem + "_all" + path.suffix)
    _atomic_write_text(all_path, _leaderboard_text(state, include_deleted=True))


def _log_in_file_enabled() -> bool:
    raw = os.environ.get("LOG_IN_FILE", "").strip()
    assert raw in ("", "1"), raw
    return raw == "1"


def _version_log_path(versions_dir: Path, version_id: str) -> Path:
    assert version_id.endswith(".tar.gz"), version_id
    return versions_dir / f"{version_id.removesuffix('.tar.gz')}.txt"


def _completed_player_log_text(completed: CompletedGame, player_index: int) -> str:
    version_ids = list(completed.scheduled.version_ids)
    worker_logs = list(completed.result.worker_logs)
    assert len(version_ids) == len(worker_logs), (version_ids, worker_logs)
    assert 0 <= player_index < len(version_ids), (player_index, version_ids)
    return worker_logs[player_index]


def _write_completed_version_logs(
    versions_dir: Path,
    completed: CompletedGame,
) -> None:
    if not _log_in_file_enabled():
        return
    logs_by_version: dict[str, list[str]] = {}
    for player_index, version_id in enumerate(completed.scheduled.version_ids):
        logs_by_version.setdefault(version_id, []).append(
            _completed_player_log_text(completed, player_index),
        )
    for version_id, logs in logs_by_version.items():
        _atomic_write_text(
            _version_log_path(versions_dir, version_id),
            "\n".join(logs),
        )


def _run_scheduled_game(
    scheduled: ScheduledGame,
    *,
    episode_steps: int,
    ship_speed: float,
    comet_speed: float,
    debug: bool,
) -> CompletedGame:
    t0 = time.perf_counter()
    result = run_match(
        archives=scheduled.archives,
        configuration={
            "episodeSteps": int(episode_steps),
            "shipSpeed": float(ship_speed),
            "cometSpeed": float(comet_speed),
            "seed": int(scheduled.seed),
        },
        replay_path=None,
        keep_tmp=False,
        tmp_dir=None,
        debug=bool(debug),
    )
    return CompletedGame(
        scheduled=scheduled,
        result=result,
        wall_seconds=time.perf_counter() - t0,
    )


def _game_worker_process_main(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    episode_steps: int,
    ship_speed: float,
    comet_speed: float,
    debug: bool,
) -> None:
    os.setsid()
    while True:
        message = task_queue.get()
        assert isinstance(message, tuple) and len(message) == 2, message
        kind, payload = message
        assert kind in ("run", "stop"), kind
        if kind == "stop":
            return
        assert isinstance(payload, ScheduledGame), type(payload)
        completed = _run_scheduled_game(
            payload,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            debug=debug,
        )
        result_queue.put(("completed", completed))


def _terminate_process_group(proc: mp.Process) -> None:
    pid = proc.pid
    if pid is None:
        return
    if proc.is_alive():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.join(timeout=5.0)
    if proc.is_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.join(timeout=5.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def _terminate_game_workers(workers: Mapping[int, GameWorker]) -> None:
    for worker in workers.values():
        _terminate_process_group(worker.process)


def _stop_game_workers(workers: dict[int, GameWorker], task_queue: mp.Queue) -> None:
    for _worker in workers.values():
        task_queue.put(("stop", None))
    for pid, worker in workers.items():
        worker.process.join(timeout=5.0)
        assert worker.process.exitcode == 0, (pid, worker.process.exitcode)
    workers.clear()


def _shutdown_signal_handler(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable competition between Orbit Wars submission archives.",
    )
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--versions-dir", type=Path, default=_DEFAULT_VERSIONS_DIR)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--leaderboard", type=Path, default=None)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--comet-speed", type=float, default=4.0)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    signal.signal(signal.SIGTERM, _shutdown_signal_handler)
    args = _parse_args(os.sys.argv[1:] if argv is None else argv)
    assert int(args.parallel) >= 1, args.parallel
    versions_dir = args.versions_dir.expanduser().resolve()
    state_path = (
        args.state.expanduser().resolve()
        if args.state is not None
        else versions_dir / "competition_state.json"
    )
    leaderboard_path = (
        args.leaderboard.expanduser().resolve()
        if args.leaderboard is not None
        else versions_dir / "leaderboard.txt"
    )
    state = _load_state(state_path, versions_dir)
    _sync_versions(state, _scan_archives(versions_dir))
    _assert_local_simulation_archives_support_cuda(state)
    _atomic_write_json(state_path, state)
    _write_leaderboards(leaderboard_path, state)

    max_games = int(args.max_games)
    target_total = None if max_games == 0 else len(state["games"]) + max_games
    in_flight: dict[int, ScheduledGame] = {}
    workers: dict[int, GameWorker] = {}
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for worker_index in range(int(args.parallel)):
        process = context.Process(
            target=_game_worker_process_main,
            args=(
                task_queue,
                result_queue,
                int(args.episode_steps),
                float(args.ship_speed),
                float(args.comet_speed),
                bool(args.debug),
            ),
            name=f"competition_game_worker_{worker_index}",
        )
        process.start()
        assert process.pid is not None
        workers[int(process.pid)] = GameWorker(process=process)
    try:
        while True:
            while len(in_flight) < int(args.parallel):
                if target_total is not None and len(state["games"]) + len(in_flight) >= target_total:
                    break
                _sync_versions(state, _scan_archives(versions_dir))
                _assert_local_simulation_archives_support_cuda(state)
                _atomic_write_json(state_path, state)
                _write_leaderboards(leaderboard_path, state)
                scheduled = _schedule_game(
                    state,
                    forced_num_players=args.num_players,
                )
                in_flight[int(scheduled.game_id)] = scheduled
                task_queue.put(("run", scheduled))
                _atomic_write_json(state_path, state)
            if not in_flight:
                break
            try:
                kind, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                dead_workers = [
                    pid for pid, worker in workers.items() if not worker.process.is_alive()
                ]
                assert not dead_workers, dead_workers
                continue
            except KeyboardInterrupt:
                _atomic_write_json(state_path, state)
                _write_leaderboards(leaderboard_path, state)
                print(
                    f"Interrupt received: killing {len(workers)} game worker process group(s).",
                    flush=True,
                )
                _terminate_game_workers(workers)
                in_flight.clear()
                workers.clear()
                break
            assert kind == "completed", kind
            completed = payload
            assert isinstance(completed, CompletedGame), type(completed)
            game_id = int(completed.scheduled.game_id)
            assert game_id in in_flight, (completed.scheduled, sorted(in_flight))
            assert in_flight.pop(game_id) == completed.scheduled, completed.scheduled
            _record_completed_game(state, completed)
            _write_completed_version_logs(versions_dir, completed)
            _atomic_write_json(state_path, state)
            _write_leaderboards(leaderboard_path, state)
            print(
                json.dumps(
                    {
                        "game_id": completed.scheduled.game_id,
                        "num_players": len(completed.scheduled.version_ids),
                        "versions": list(completed.scheduled.version_ids),
                        "rewards": completed.result.rewards,
                        "statuses": completed.result.statuses,
                        "leaderboard": str(leaderboard_path),
                        "state": str(state_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if target_total is not None and len(state["games"]) >= target_total and not in_flight:
                break
        _stop_game_workers(workers, task_queue)
    except KeyboardInterrupt:
        print(
            f"Interrupt received: killing {len(workers)} game worker process group(s).",
            flush=True,
        )
        _terminate_game_workers(workers)
        in_flight.clear()
        workers.clear()
    finally:
        _terminate_game_workers(workers)
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()


if __name__ == "__main__":
    main()
