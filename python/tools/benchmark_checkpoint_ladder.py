from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from src.torchbeast.core.common import parse_impala_run_checkpoint_step

_DEFAULT_BENCHMARK_CONFIG = _PYTHON_ROOT / "src" / "configs" / "impala_x1_rl_benchmark.py"


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    step = parse_impala_run_checkpoint_step(path.name)
    assert step is not None, f"unsupported checkpoint filename: {path.name}"
    return int(step), path.name


def _load_state(
    path: Path,
    checkpoint_dir: Path,
    benchmark_game_size: str,
    benchmark_sample_a: bool,
    benchmark_sample_b: bool,
) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text())
        assert isinstance(state, dict), type(state)
        assert state["checkpoint_dir"] == str(checkpoint_dir), state["checkpoint_dir"]
        assert state["benchmark_game_size"] == str(benchmark_game_size), state["benchmark_game_size"]
        assert bool(state["benchmark_sample_a"]) == bool(benchmark_sample_a), state["benchmark_sample_a"]
        assert bool(state["benchmark_sample_b"]) == bool(benchmark_sample_b), state["benchmark_sample_b"]
        assert isinstance(state["matches"], list), type(state["matches"])
        state.pop("games_per_match", None)
        return state
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "benchmark_game_size": str(benchmark_game_size),
        "benchmark_sample_a": bool(benchmark_sample_a),
        "benchmark_sample_b": bool(benchmark_sample_b),
        "baseline": None,
        "matches": [],
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(path)


def _match_by_pair(state: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for match in state["matches"]:
        assert isinstance(match, dict), type(match)
        key = (str(match["baseline"]), str(match["challenger"]))
        assert key not in out, key
        out[key] = match
    return out


def _checkpoint_index_by_name(checkpoints: list[Path]) -> dict[str, int]:
    out = {checkpoint.name: i for i, checkpoint in enumerate(checkpoints)}
    assert len(out) == len(checkpoints), [checkpoint.name for checkpoint in checkpoints]
    return out


def _initialize_next_checkpoint_index(
    state: dict[str, Any],
    checkpoint_index_by_name: dict[str, int],
) -> None:
    if "next_checkpoint_index" in state:
        next_checkpoint_index = int(state["next_checkpoint_index"])
        assert 1 <= next_checkpoint_index <= len(checkpoint_index_by_name), next_checkpoint_index
        state["next_checkpoint_index"] = next_checkpoint_index
        return

    max_challenger_index = 0
    for match in state["matches"]:
        assert isinstance(match, dict), type(match)
        challenger_name = str(match["challenger"])
        assert challenger_name in checkpoint_index_by_name, challenger_name
        max_challenger_index = max(max_challenger_index, checkpoint_index_by_name[challenger_name])

    state["next_checkpoint_index"] = max_challenger_index + 1


def _parse_benchmark_result(stdout: str) -> dict[str, Any]:
    prefix = "BENCHMARK_RESULT "
    result_lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    assert len(result_lines) == 1, result_lines
    payload = result_lines[0][len(prefix) :]
    result = json.loads(payload)
    assert isinstance(result, dict), type(result)
    return result


def _print_benchmark_line(line: str) -> None:
    if (
        line.startswith("BENCHMARK_PROGRESS ")
        or line.startswith("BENCHMARK_SPS ")
        or line.startswith("BENCHMARK_RESULT ")
        or line.startswith("BENCHMARK_WALL_TREE ")
    ):
        print(line, flush=True)


def _run_benchmark(
    *,
    run_monobeast: Path,
    config: Path,
    baseline: Path,
    challenger: Path,
    games: int,
    benchmark_game_size: str,
    benchmark_sample_a: bool,
    benchmark_sample_b: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(run_monobeast),
        "--config",
        str(config),
        "--benchmark",
        "--benchmark_checkpoint_a",
        str(baseline),
        "--benchmark_checkpoint_b",
        str(challenger),
        "--benchmark_games",
        str(int(games)),
        "--benchmark_game_size",
        str(benchmark_game_size),
    ]
    if not bool(benchmark_sample_a):
        cmd.append("--sample-false-a")
    if not bool(benchmark_sample_b):
        cmd.append("--sample-false-b")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    stdout_lines: list[str] = []
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        stdout_lines.append(stripped)
        _print_benchmark_line(stripped)
    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            [
                str(x) for x in cmd
            ],
        )
    return _parse_benchmark_result("\n".join(stdout_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable IMPALA checkpoint ladder benchmark")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_BENCHMARK_CONFIG),
        help="Benchmark config passed to run_monobeast",
    )
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoint_*.pt files")
    parser.add_argument("--games", type=int, default=1000, help="Decisive non-draw games per match")
    parser.add_argument(
        "--benchmark_game_size",
        choices=("mixed", "2p", "4p"),
        default="4p",
        help="Benchmark game-size mode passed to run_monobeast",
    )
    parser.add_argument(
        "--sample-false-a",
        action="store_true",
        help="Run benchmark model A with is_sample=False",
    )
    parser.add_argument(
        "--sample-false-b",
        action="store_true",
        help="Run benchmark model B with is_sample=False",
    )
    parser.add_argument("--state_file", help="JSON state file; defaults inside checkpoint_dir")
    parser.add_argument("--run_monobeast", help="Path to run_monobeast.py")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    assert checkpoint_dir.is_dir(), checkpoint_dir
    config = Path(args.config).expanduser().resolve()
    assert config.is_file(), config
    run_monobeast = (
        Path(args.run_monobeast).expanduser().resolve()
        if args.run_monobeast is not None
        else _PYTHON_ROOT / "run_monobeast.py"
    )
    assert run_monobeast.is_file(), run_monobeast
    games = int(args.games)
    assert games > 0, games
    benchmark_game_size = str(args.benchmark_game_size)
    benchmark_sample_a = not bool(args.sample_false_a)
    benchmark_sample_b = not bool(args.sample_false_b)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=_checkpoint_sort_key, reverse=True)
    assert len(checkpoints) >= 2, checkpoint_dir
    checkpoint_by_name = {p.name: p for p in checkpoints}
    checkpoint_index_by_name = _checkpoint_index_by_name(checkpoints)

    state_file = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file is not None
        else checkpoint_dir / "benchmark_ladder_state.json"
    )
    state = _load_state(
        state_file,
        checkpoint_dir,
        benchmark_game_size,
        benchmark_sample_a,
        benchmark_sample_b,
    )
    if state["baseline"] is None:
        state["baseline"] = checkpoints[0].name
        state["next_checkpoint_index"] = 1
        _write_state(state_file, state)
    _initialize_next_checkpoint_index(state, checkpoint_index_by_name)

    matches = _match_by_pair(state)
    baseline_name = str(state["baseline"])
    assert baseline_name in checkpoint_by_name, baseline_name

    for checkpoint_index in range(int(state["next_checkpoint_index"]), len(checkpoints)):
        challenger = checkpoints[checkpoint_index]
        challenger_name = challenger.name
        if challenger_name == baseline_name:
            state["next_checkpoint_index"] = checkpoint_index + 1
            _write_state(state_file, state)
            continue
        key = (baseline_name, challenger_name)
        if key in matches:
            match = matches[key]
            print(
                "LADDER_SKIP "
                f"baseline={baseline_name} challenger={challenger_name} "
                f"b_winrate={float(match['b_winrate']):.4f} "
                f"games={int(match['games'])}",
                flush=True,
            )
        else:
            print(
                "LADDER_MATCH "
                f"baseline={baseline_name} challenger={challenger_name} "
                f"target_games={games} "
                f"benchmark_game_size={benchmark_game_size} "
                f"sample_a={benchmark_sample_a} sample_b={benchmark_sample_b}",
                flush=True,
            )
            result = _run_benchmark(
                run_monobeast=run_monobeast,
                config=config,
                baseline=checkpoint_by_name[baseline_name],
                challenger=challenger,
                games=games,
                benchmark_game_size=benchmark_game_size,
                benchmark_sample_a=benchmark_sample_a,
                benchmark_sample_b=benchmark_sample_b,
            )
            match = {
                "baseline": baseline_name,
                "challenger": challenger_name,
                **result,
            }
            state["matches"].append(match)
            matches[key] = match
            _write_state(state_file, state)

        b_winrate = float(match["b_winrate"])
        print(
            "LADDER_RESULT "
            f"baseline={baseline_name} challenger={challenger_name} "
            f"b_winrate={b_winrate:.4f} "
            f"games={int(match['games'])} "
            f"a_wins={int(match['a_wins'])} b_wins={int(match['b_wins'])} "
            f"draws_excluded={int(match['draws_excluded'])}",
            flush=True,
        )
        if b_winrate > 0.5:
            baseline_name = challenger_name
            state["baseline"] = baseline_name
            print(f"LADDER_BASELINE_UPDATE baseline={baseline_name}", flush=True)
        state["next_checkpoint_index"] = checkpoint_index + 1
        _write_state(state_file, state)

    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
