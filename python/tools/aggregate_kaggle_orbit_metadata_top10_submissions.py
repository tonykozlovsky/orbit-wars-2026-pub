"""Scan Kaggle Orbit sidecar ``*_metadata.json`` files and aggregate top-N public leaderboard submissions.

Per submission, also reports mean public leaderboard place (1-based row index) across metadata files
where that submission appeared in the top-N slice.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _impala_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _team_id_to_team_name(leaderboard: Mapping[str, Any]) -> dict[int, str]:
    teams_raw = leaderboard["teams"]
    assert isinstance(teams_raw, list), type(teams_raw)
    out: dict[int, str] = {}
    for entry in teams_raw:
        assert isinstance(entry, dict), type(entry)
        tid = int(entry["teamId"])
        name = str(entry["teamName"])
        assert tid not in out or out[tid] == name, (tid, out.get(tid), name)
        out[tid] = name
    return out


def _captured_epoch_seconds(meta: Mapping[str, Any]) -> float:
    return float(meta["captured_at_epoch_seconds"])


def _top_public_leaderboard_rows(
    meta: Mapping[str, Any],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    lb = meta["leaderboard"]
    assert isinstance(lb, dict), type(lb)
    pub = lb["publicLeaderboard"]
    assert isinstance(pub, list), type(pub)
    out: list[dict[str, Any]] = []
    for i in range(min(top_n, len(pub))):
        row = pub[i]
        assert isinstance(row, dict), (i, type(row))
        out.append(row)
    return out


def _print_progress(processed_files: int, total_files: int, last_progress_pct: int) -> int:
    pct = (100 * processed_files) // total_files
    if pct > last_progress_pct:
        print(
            f"metadata_progress {processed_files}/{total_files} ({pct}%)",
            file=sys.stderr,
            flush=True,
        )
        return pct
    return last_progress_pct


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Aggregate top-N publicLeaderboard rows across all *_metadata.json in a directory: "
        "per team, list each submission_id that ever appeared in top-N with how many metadata files "
        "contained it there, mean public LB place in those files, plus first/last capture timestamps.",
    )
    p.add_argument(
        "--metadata-dir",
        type=Path,
        default=_impala_root() / "datasets" / "replays",
        help="Directory containing sidecar files named <stem>_metadata.json (default: repo datasets/replays).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        metavar="N",
        help="How many publicLeaderboard rows to read from the top (default: 10).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write report to this path (UTF-8). Default: stdout.",
    )
    args = p.parse_args(argv)

    root = args.metadata_dir.expanduser().resolve()
    assert root.is_dir(), root
    top_n = int(args.top_n)
    assert top_n >= 1, top_n

    paths = sorted(root.glob("*_metadata.json"))
    assert len(paths) > 0, f"no *_metadata.json under {root}"
    total_files = len(paths)

    n_files = 0
    n_skipped_files = 0
    hits_per_key: dict[tuple[str, int], int] = defaultdict(int)
    rank_sum_per_key: dict[tuple[str, int], int] = defaultdict(int)
    first_epoch: dict[tuple[str, int], float] = {}
    last_epoch: dict[tuple[str, int], float] = {}
    last_progress_pct = -1

    for processed_files, path in enumerate(paths, start=1):
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            n_skipped_files += 1
            print(
                f"metadata_skip {path}: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            last_progress_pct = _print_progress(processed_files, total_files, last_progress_pct)
            continue
        assert isinstance(raw, dict), (path, type(raw))
        epoch = _captured_epoch_seconds(raw)
        team_by_id = _team_id_to_team_name(raw["leaderboard"])
        rows = _top_public_leaderboard_rows(raw, top_n=top_n)
        key_to_rank: dict[tuple[str, int], int] = {}
        for i, row in enumerate(rows):
            tid = int(row["teamId"])
            sid = int(row["submissionId"])
            assert tid in team_by_id, (path, tid, sorted(team_by_id.keys())[:20])
            tname = team_by_id[tid]
            key = (tname, sid)
            assert key not in key_to_rank, (path, key, row)
            key_to_rank[key] = i + 1
        for key, rank in key_to_rank.items():
            hits_per_key[key] += 1
            rank_sum_per_key[key] += rank
            fe = first_epoch.get(key)
            if fe is None or epoch < fe:
                first_epoch[key] = epoch
            le = last_epoch.get(key)
            if le is None or epoch > le:
                last_epoch[key] = epoch
        n_files += 1
        last_progress_pct = _print_progress(processed_files, total_files, last_progress_pct)

    by_team: dict[str, list[tuple[int, int, float, float, float]]] = defaultdict(list)
    for (tname, sid), nh in hits_per_key.items():
        rs = rank_sum_per_key[(tname, sid)]
        assert nh >= 1, (tname, sid, nh)
        assert rs >= nh, (tname, sid, nh, rs)
        mean_rank = float(rs) / float(nh)
        by_team[tname].append(
            (sid, nh, first_epoch[(tname, sid)], last_epoch[(tname, sid)], mean_rank)
        )

    for tname in by_team:
        by_team[tname].sort(key=lambda r: (r[2], r[0]))

    lines: list[str] = []
    lines.append(
        f"# metadata_dir={root}\n"
        f"# metadata_files_read={n_files}\n"
        f"# metadata_files_skipped={n_skipped_files}\n"
        f"# top_n={top_n}\n"
        f"# Per team: each submission_id that appeared in the top-{top_n} publicLeaderboard rows of a file.\n"
        f"# metadata_files_in_top_n = how many *_metadata.json files listed that submission in their top-{top_n}.\n"
        f"# mean_public_lb_place = mean 1-based row index in publicLeaderboard across those files (1=top).\n"
        f"# Submissions are ordered by first_captured_epoch then submission_id.\n"
    )
    for tname in sorted(by_team.keys()):
        lines.append(f"\n{tname}\n")
        for sid, nh, fe, le, mean_place in by_team[tname]:
            lines.append(
                f"  submission_id={sid}  metadata_files_in_top_{top_n}={nh}  "
                f"mean_public_lb_place={mean_place:.6g}  "
                f"first_captured_epoch={fe}  last_captured_epoch={le}\n"
            )

    text = "".join(lines)
    if args.out is not None:
        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
