from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

# OpenSpiel optional envs log a long INFO block on ``import kaggle_environments``; not needed for orbit_wars.
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").disabled = True


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VERSIONS_DIR = _REPO_ROOT / "versions"
_REFERENCE_ENV_DIR = (
    _REPO_ROOT
    / "python"
    / "cpp"
    / "orbit_wars"
    / "reference_kaggle_upstream_github_no_edit"
)
_REFERENCE_ENV_PATH = _REFERENCE_ENV_DIR / "orbit_wars.py"
_OBSERVATION_KEYS = (
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
_CONFIGURATION_KEYS = (
    "episodeSteps",
    "actTimeout",
    "agentTimeout",
    "shipSpeed",
    "cometSpeed",
    "seed",
)
_WORKER_READY = "ready"
_WORKER_ACT = "act"
_WORKER_CLOSE = "close"
_WORKER_RESULT = "result"
_RUN_TIMEOUT_SECONDS = 2400


@dataclass(frozen=True)
class MatchResult:
    rewards: list[Any]
    statuses: list[str]
    seed: Any
    replay_path: Path | None
    worker_logs: list[str]


def _object_get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        assert key in obj, key
        return obj[key]
    assert hasattr(obj, key), (type(obj), key)
    return getattr(obj, key)


def _plain_observation(observation: Any) -> dict[str, Any]:
    return {k: _object_get(observation, k) for k in _OBSERVATION_KEYS}


def _plain_configuration(configuration: Any) -> dict[str, Any]:
    assert configuration is not None
    return {k: _object_get(configuration, k) for k in _CONFIGURATION_KEYS}


def _load_reference_env_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "orbit_wars_reference_kaggle_upstream_github_no_edit_for_simulation",
        _REFERENCE_ENV_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "specification")
    assert hasattr(module, "interpreter")
    assert hasattr(module, "renderer")
    assert hasattr(module, "html_renderer")
    return module


def make_reference_orbit_wars_env(
    *,
    configuration: Mapping[str, Any],
    debug: bool,
) -> Any:
    make = importlib.import_module("kaggle_environments").make
    module = _load_reference_env_module()
    specification = copy.deepcopy(module.specification)
    specification["configuration"]["runTimeout"] = _RUN_TIMEOUT_SECONDS
    local_env = {
        "specification": specification,
        "interpreter": module.interpreter,
        "renderer": module.renderer,
        "html_renderer": module.html_renderer,
    }
    return make(local_env, configuration=dict(configuration), debug=debug)


def _resolve_archive_path(path_arg: str, versions_dir: Path) -> Path:
    p = Path(path_arg).expanduser()
    if p.is_absolute() or p.exists():
        resolved = p.resolve()
    else:
        resolved = (versions_dir / p).resolve()
    assert resolved.is_file(), resolved
    assert resolved.name.endswith(".tar.gz"), resolved
    return resolved


def _extract_submission_archive(archive_path: Path, extract_root: Path, seat: int) -> Path:
    target = extract_root / f"seat_{seat}"
    target.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(target, filter="data")
    submission_path = target / "kaggle_submission" / "submission.py"
    assert submission_path.is_file(), submission_path
    return target


def _capture_worker_logs_enabled() -> bool:
    silent_raw = os.environ.get("SILENT", "").strip()
    assert silent_raw in ("", "1"), silent_raw
    log_in_file_raw = os.environ.get("LOG_IN_FILE", "").strip()
    assert log_in_file_raw in ("", "1"), log_in_file_raw
    return silent_raw == "1" or log_in_file_raw == "1"


def _worker_main(submission_root: str, conn: Connection, log_path: str) -> None:
    if _capture_worker_logs_enabled():
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
    root = Path(submission_root).resolve()
    assert (root / "kaggle_submission" / "submission.py").is_file(), root
    os.chdir(root)
    sys.path.insert(0, str(root))
    module = importlib.import_module("kaggle_submission.submission")
    agent = getattr(module, "agent")
    assert callable(agent), type(agent)
    conn.send((_WORKER_READY, None))
    while True:
        message = conn.recv()
        assert isinstance(message, tuple) and len(message) == 2, message
        kind, payload = message
        assert kind in (_WORKER_ACT, _WORKER_CLOSE), kind
        if kind == _WORKER_CLOSE:
            conn.close()
            return
        assert isinstance(payload, tuple) and len(payload) == 2, type(payload)
        observation, configuration = payload
        action = agent(observation, configuration)
        assert isinstance(action, list), type(action)
        conn.send((_WORKER_RESULT, action))


class SubmissionWorkerAgent:
    def __init__(
        self,
        *,
        seat: int,
        archive_path: Path,
        submission_root: Path,
        context: mp.context.BaseContext,
    ) -> None:
        self.seat = int(seat)
        self.archive_path = archive_path
        self._log_path = submission_root / "submission_worker.log"
        parent_conn, child_conn = context.Pipe()
        self._conn = parent_conn
        self._process = context.Process(
            target=_worker_main,
            args=(str(submission_root), child_conn, str(self._log_path)),
            name=f"orbit_kaggle_submission_seat_{seat}",
        )
        self._process.start()
        child_conn.close()
        kind, payload = self._conn.recv()
        assert kind == _WORKER_READY, (kind, payload)

    def __call__(self, observation: Any, configuration: Any) -> list[Any]:
        plain_observation = _plain_observation(observation)
        plain_configuration = _plain_configuration(configuration)
        self._conn.send((_WORKER_ACT, (plain_observation, plain_configuration)))
        kind, payload = self._conn.recv()
        assert kind == _WORKER_RESULT, (kind, payload)
        assert isinstance(payload, list), type(payload)
        return payload

    def close(self) -> None:
        if self._process.is_alive():
            self._conn.send((_WORKER_CLOSE, None))
            self._process.join()
        if self._process.exitcode != 0 and self._log_path.is_file():
            log_text = self._log_path.read_text()
            if log_text:
                print(
                    f"\n=== submission worker log: seat={self.seat} "
                    f"archive={self.archive_path} exitcode={self._process.exitcode} ===",
                    file=sys.stderr,
                )
                print(log_text.rstrip(), file=sys.stderr)
                print("=== end submission worker log ===\n", file=sys.stderr)
        assert self._process.exitcode == 0, (
            self.seat,
            self.archive_path,
            self._process.exitcode,
        )
        self._conn.close()

    def log_text(self) -> str:
        assert self._log_path.is_file(), self._log_path
        return self._log_path.read_text()


def _state_rewards(env: Any) -> list[Any]:
    return [_object_get(s, "reward") for s in env.state]


def _state_statuses(env: Any) -> list[str]:
    return [str(_object_get(s, "status")) for s in env.state]


def _write_replay(env: Any, output_path: Path) -> None:
    replay = env.toJSON()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")


def run_match(
    *,
    archives: Sequence[Path],
    configuration: Mapping[str, Any],
    replay_path: Path | None,
    keep_tmp: bool,
    tmp_dir: Path | None,
    debug: bool,
) -> MatchResult:
    assert len(archives) in (2, 4), len(archives)
    context = mp.get_context("spawn")
    temp_kwargs: dict[str, Any] = {"prefix": "orbit_kaggle_sim_"}
    if tmp_dir is not None:
        temp_kwargs["dir"] = str(tmp_dir)
    with ExitStack() as stack:
        if keep_tmp:
            tmp_name = tempfile.mkdtemp(**temp_kwargs)
        else:
            tmp_name = stack.enter_context(tempfile.TemporaryDirectory(**temp_kwargs))
        extract_root = Path(tmp_name).resolve()
        worker_stack = stack.enter_context(ExitStack())
        workers: list[SubmissionWorkerAgent] = []
        for seat, archive_path in enumerate(archives):
            submission_root = _extract_submission_archive(archive_path, extract_root, seat)
            worker = SubmissionWorkerAgent(
                seat=seat,
                archive_path=archive_path,
                submission_root=submission_root,
                context=context,
            )
            workers.append(worker)
            worker_stack.callback(worker.close)
        env = make_reference_orbit_wars_env(configuration=configuration, debug=debug)
        env.run(list(workers))
        if _capture_worker_logs_enabled():
            for worker in workers:
                worker.close()
            worker_stack.pop_all()
            worker_logs = [worker.log_text() for worker in workers]
        else:
            worker_logs = [""] * len(workers)
        if replay_path is not None:
            _write_replay(env, replay_path)
        return MatchResult(
            rewards=_state_rewards(env),
            statuses=_state_statuses(env),
            seed=env.info.get("seed"),
            replay_path=replay_path,
            worker_logs=worker_logs,
        )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Kaggle Orbit Wars match between packaged submission archives.",
    )
    parser.add_argument(
        "archives",
        nargs="+",
        help="Two or four .tar.gz archives, either paths or names under --versions-dir.",
    )
    parser.add_argument(
        "--versions-dir",
        type=Path,
        default=_DEFAULT_VERSIONS_DIR,
        help="Directory used to resolve non-path archive names.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--comet-speed", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional replay JSON path.")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    assert len(args.archives) in (2, 4), args.archives
    versions_dir = args.versions_dir.expanduser().resolve()
    archives = [_resolve_archive_path(a, versions_dir) for a in args.archives]
    configuration = {
        "episodeSteps": int(args.episode_steps),
        "shipSpeed": float(args.ship_speed),
        "cometSpeed": float(args.comet_speed),
        "seed": args.seed,
    }
    result = run_match(
        archives=archives,
        configuration=configuration,
        replay_path=args.output.resolve() if args.output is not None else None,
        keep_tmp=bool(args.keep_tmp),
        tmp_dir=args.tmp_dir.expanduser().resolve() if args.tmp_dir is not None else None,
        debug=bool(args.debug),
    )
    print(
        json.dumps(
            {
                "archives": [str(p) for p in archives],
                "rewards": result.rewards,
                "statuses": result.statuses,
                "seed": result.seed,
                "replay_path": str(result.replay_path) if result.replay_path is not None else None,
                "worker_logs": result.worker_logs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
