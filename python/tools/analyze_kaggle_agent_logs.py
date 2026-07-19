#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_STEP_STATUS_RE = re.compile(
    r"\bSUBMISSION_STEP_STATUS\b.*"
    r"\bstep=(\d+)\b.*"
    r"\bnum_agents=(\d+)\b.*"
    r"\bremaining_overage_time=([0-9.]+)\b.*"
    r"\boverage_budget_exhausted=(\d+)\b"
)
_FAILURE_PHRASE_KEYWORDS = (
    "core dumped",
    "segmentation fault",
    "exit with code",
    "exited with code",
    "non-zero",
    "non zero",
    "timed out",
    "out of memory",
    "cuda error",
    "cublas_status_",
    "cudnn_status_",
    "illegal memory access",
    "device-side assert",
)
_FAILURE_TOKEN_KEYWORDS = (
    "traceback",
    "assert",
    "assertion",
    "assertionerror",
    "exception",
    "error",
    "failed",
    "fail",
    "failure",
    "failing",
    "fatal",
    "critical",
    "panic",
    "abort",
    "aborted",
    "segfault",
    "sigabrt",
    "sigbus",
    "sigfpe",
    "sigill",
    "sigint",
    "sigkill",
    "sigsegv",
    "sigterm",
    "exceeded",
    "killed",
    "timeout",
    "deadline",
    "oom",
    "memoryerror",
    "runtimeerror",
    "valueerror",
    "typeerror",
    "keyerror",
    "indexerror",
    "importerror",
    "modulenotfounderror",
    "filenotfounderror",
    "eoferror",
    "unicodedecodeerror",
    "jsondecodeerror",
    "nan",
    "+nan",
    "-nan",
    "+inf",
    "-inf",
    "infinity",
    "invalid",
    "corrupt",
    "corrupted",
    "missing",
    "unavailable",
    "denied",
    "forbidden",
    "unauthorized",
)
_TOKEN_TRANSLATION = str.maketrans(
    {
        "\t": " ",
        "\n": " ",
        "\r": " ",
        "'": " ",
        '"': " ",
        "`": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        ",": " ",
        ";": " ",
        ":": " ",
        "=": " ",
        "|": " ",
        "/": " ",
        "\\": " ",
    }
)


@dataclass(frozen=True)
class StepLogRecord:
    step: int
    num_agents: int
    remaining_overage_time: float
    overage_budget_exhausted: int
    duration: float


def _impala_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_agent_log_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("episode_*_agent_*_logs.json"))
    assert paths, f"no episode_*_agent_*_logs.json under {input_dir}"
    return paths


def load_agent_log(path: Path) -> list[list[dict[str, object]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list), (path, type(raw))
    out: list[list[dict[str, object]]] = []
    for row_index, row in enumerate(raw):
        assert isinstance(row, list), (path, row_index, type(row))
        parsed_row: list[dict[str, object]] = []
        for entry_index, entry in enumerate(row):
            assert isinstance(entry, dict), (path, row_index, entry_index, type(entry))
            parsed_row.append(entry)
        out.append(parsed_row)
    return out


def step_records_for_loaded_log(
    path: Path,
    log_rows: list[list[dict[str, object]]],
) -> list[StepLogRecord]:
    records: list[StepLogRecord] = []
    for row_index, row in enumerate(log_rows):
        for entry_index, entry in enumerate(row):
            stderr = entry["stderr"]
            assert isinstance(stderr, str), (path, row_index, entry_index, type(stderr))
            duration_raw = entry["duration"]
            assert type(duration_raw) is int or type(duration_raw) is float, (
                path,
                row_index,
                entry_index,
                type(duration_raw),
            )
            duration = float(duration_raw)
            for line in stderr.splitlines():
                match = _STEP_STATUS_RE.search(line)
                if match is None:
                    continue
                step = int(match.group(1))
                num_agents = int(match.group(2))
                remaining_overage_time = float(match.group(3))
                overage_budget_exhausted = int(match.group(4))
                assert num_agents in (2, 4), (path, num_agents, line)
                assert remaining_overage_time >= 0.0, (path, remaining_overage_time, line)
                assert overage_budget_exhausted in (0, 1), (path, overage_budget_exhausted, line)
                records.append(
                    StepLogRecord(
                        step=step,
                        num_agents=num_agents,
                        remaining_overage_time=remaining_overage_time,
                        overage_budget_exhausted=overage_budget_exhausted,
                        duration=duration,
                    )
                )
    assert records, path
    return records


def game_num_agents_for_records(path: Path, records: list[StepLogRecord]) -> int:
    values = {record.num_agents for record in records}
    assert len(values) == 1, (path, sorted(values))
    return next(iter(values))


def percentile(sorted_values: list[float], q: float) -> float:
    assert sorted_values, sorted_values
    assert 0.0 <= q <= 1.0, q
    index = int(round(q * float(len(sorted_values) - 1)))
    return sorted_values[index]


def duration_summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    sorted_values = sorted(values)
    total = sum(sorted_values)
    mean = total / float(len(sorted_values))
    return (
        f"n={len(sorted_values)} "
        f"mean={mean:.6f} "
        f"min={sorted_values[0]:.6f} "
        f"p50={percentile(sorted_values, 0.50):.6f} "
        f"p90={percentile(sorted_values, 0.90):.6f} "
        f"p95={percentile(sorted_values, 0.95):.6f} "
        f"p99={percentile(sorted_values, 0.99):.6f} "
        f"max={sorted_values[-1]:.6f} "
        f"sum={total:.6f}"
    )


def add_overage_sliced_durations(
    path: Path,
    records: list[StepLogRecord],
    durations_by_slice: dict[tuple[str, str], list[float]],
) -> None:
    num_agents = game_num_agents_for_records(path, records)
    game_type = f"{num_agents}p"
    measured_records = [record for record in records if record.step > 1]
    assert measured_records, path
    durations_by_slice[(game_type, "all_excluding_step_1")].extend(
        record.duration for record in measured_records
    )

    exhausted_values = {record.overage_budget_exhausted for record in measured_records}
    assert exhausted_values.issubset({0, 1}), (path, exhausted_values)
    if exhausted_values == {0}:
        durations_by_slice[(game_type, "overage_not_triggered")].extend(
            record.duration for record in measured_records
        )
        return

    seen_exhausted = False
    for record in measured_records:
        if record.overage_budget_exhausted == 1:
            seen_exhausted = True
            durations_by_slice[(game_type, "overage_after_trigger")].append(record.duration)
            continue
        assert not seen_exhausted, (path, record)
        durations_by_slice[(game_type, "overage_before_trigger")].append(record.duration)


def final_remaining_overage_time(path: Path, records: list[StepLogRecord]) -> float:
    final_record = max(records, key=lambda record: record.step)
    same_step_records = [record for record in records if record.step == final_record.step]
    values = {record.remaining_overage_time for record in same_step_records}
    assert len(values) == 1, (path, final_record.step, sorted(values))
    return final_record.remaining_overage_time


def failure_keywords_in_line(line: str) -> list[str]:
    lowered = line.casefold()
    normalized = f" {lowered.translate(_TOKEN_TRANSLATION)} "
    matches: list[str] = []
    for phrase in _FAILURE_PHRASE_KEYWORDS:
        if phrase in lowered:
            matches.append(phrase)
    for token in _FAILURE_TOKEN_KEYWORDS:
        if f" {token} " in normalized:
            matches.append(token)
    return matches


def print_failure_keyword_matches(
    path: Path,
    log_rows: list[list[dict[str, object]]],
) -> None:
    for row_index, row in enumerate(log_rows):
        for entry_index, entry in enumerate(row):
            for stream_name in ("stdout", "stderr"):
                text = entry[stream_name]
                assert isinstance(text, str), (path, row_index, entry_index, stream_name, type(text))
                for line_index, line in enumerate(text.splitlines(), start=1):
                    for keyword in failure_keywords_in_line(line):
                        print(
                            "keyword_match "
                            f"path={path} "
                            f"row={row_index} "
                            f"entry={entry_index} "
                            f"stream={stream_name} "
                            f"line={line_index} "
                            f"keyword={keyword!r} "
                            f"text={line!r}"
                        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze downloaded Kaggle Orbit Wars agent logs and print aggregate checks.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_impala_root() / "outputs" / "kaggle_agent_logs",
        help="Directory containing episode_*_agent_*_logs.json files.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print one line per log file after the aggregate summary.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    assert input_dir.is_dir(), input_dir

    paths = iter_agent_log_paths(input_dir)
    counts: Counter[str] = Counter()
    per_file: list[tuple[Path, str]] = []
    durations_by_slice: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    final_remaining_by_game_type: defaultdict[str, list[float]] = defaultdict(list)
    loaded_logs: list[tuple[Path, list[list[dict[str, object]]]]] = []
    for path in paths:
        log_rows = load_agent_log(path)
        loaded_logs.append((path, log_rows))
        records = step_records_for_loaded_log(path, log_rows)
        num_agents = game_num_agents_for_records(path, records)
        game_type = f"{num_agents}p"
        counts[game_type] += 1
        per_file.append((path, game_type))
        add_overage_sliced_durations(path, records, durations_by_slice)
        final_remaining_by_game_type[game_type].append(final_remaining_overage_time(path, records))

    print(f"input_dir={input_dir}")
    print(f"log_files={len(paths)}")
    print(f"game_type_2p={counts['2p']}")
    print(f"game_type_4p={counts['4p']}")
    print("duration_seconds:")
    for game_type in ("2p", "4p"):
        print(f"  {game_type}:")
        for slice_name in (
            "all_excluding_step_1",
            "overage_not_triggered",
            "overage_before_trigger",
            "overage_after_trigger",
        ):
            values = durations_by_slice.get((game_type, slice_name), [])
            print(f"    {slice_name}: {duration_summary(values)}")
    print("final_remaining_overage_time:")
    for game_type in ("2p", "4p"):
        values = final_remaining_by_game_type.get(game_type, [])
        print(f"  {game_type}: {duration_summary(values)}")
    print("failure_keyword_matches:")
    for path, log_rows in loaded_logs:
        print_failure_keyword_matches(path, log_rows)

    if args.show_files:
        print("files:")
        for path, game_type in per_file:
            print(f"  {game_type} {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
