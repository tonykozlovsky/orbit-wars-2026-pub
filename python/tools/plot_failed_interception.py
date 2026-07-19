from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


_IMPALA_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FAILURES_PATH = _IMPALA_ROOT / "outputs" / "failed_interceptions.txt"
_BOARD_SIZE = 100.0
_CENTER = 50.0
_SUN_RADIUS = 10.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=_DEFAULT_FAILURES_PATH)
    parser.add_argument("--index", type=int, default=-1)
    return parser.parse_args()


def _load_failed_interception(path: Path, index: int) -> dict[str, Any]:
    lines = [line for line in path.expanduser().read_text().splitlines() if line.strip()]
    assert len(lines) > 0, path
    row = json.loads(lines[index])
    assert isinstance(row, dict), type(row)
    return row


def _segment_color(branch: int) -> str:
    assert branch in (0, 1, 2), branch
    return ("tab:orange", "tab:cyan", "tab:green")[branch]


def _plot_failed_interception(row: dict[str, Any]) -> None:
    src = int(row["src"])
    dst = int(row["dst"])
    assert src != dst, (src, dst)
    src_planet = row["src_planet"]
    dst_planet = row["dst_planet"]
    target_path = row["target_path"]
    segments = row["segments"]
    solver_results = row["solver_results"]
    assert isinstance(src_planet, dict), type(src_planet)
    assert isinstance(dst_planet, dict), type(dst_planet)
    assert isinstance(target_path, list), type(target_path)
    assert isinstance(segments, list), type(segments)
    assert isinstance(solver_results, dict), type(solver_results)
    assert not bool(solver_results["fair_fast"]["valid"])
    assert bool(solver_results["fair_slow"]["valid"])
    assert int(src_planet["slot"]) == src, (src_planet, src)
    assert int(dst_planet["slot"]) == dst, (dst_planet, dst)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, _BOARD_SIZE)
    ax.set_ylim(0.0, _BOARD_SIZE)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    sun = Circle((_CENTER, _CENTER), _SUN_RADIUS, facecolor="gold", edgecolor="black", alpha=0.35)
    ax.add_patch(sun)
    ax.text(_CENTER, _CENTER, "sun", ha="center", va="center", fontsize=9)

    if target_path:
        xs = [float(p["x"]) for p in target_path]
        ys = [float(p["y"]) for p in target_path]
        ax.plot(xs, ys, color="tab:red", linewidth=2.5, marker=".", label="target path")

    for segment in segments:
        assert isinstance(segment, dict), type(segment)
        branch = int(segment["branch"])
        color = _segment_color(branch)
        ax.plot(
            [float(segment["start_x"]), float(segment["end_x"])],
            [float(segment["start_y"]), float(segment["end_y"])],
            color=color,
            linewidth=1.2 if branch != 2 else 2.2,
            alpha=0.35 if branch == 0 else 0.75,
        )
        ax.scatter(
            [float(segment["target_center_x"])],
            [float(segment["target_center_y"])],
            s=10,
            color=color,
            alpha=0.75,
        )

    fair_slow = solver_results["fair_slow"]
    ax.plot(
        [float(src_planet["x"]), float(fair_slow["aim_x"])],
        [float(src_planet["y"]), float(fair_slow["aim_y"])],
        color="black",
        linewidth=3.0,
        alpha=0.85,
        label="fair_slow solution",
    )
    ax.scatter(
        [float(fair_slow["aim_x"])],
        [float(fair_slow["aim_y"])],
        s=55,
        color="black",
        marker="x",
    )

    for planet, color, role in (
        (src_planet, "tab:green", "src"),
        (dst_planet, "tab:red", "dst"),
    ):
        x = float(planet["x"])
        y = float(planet["y"])
        radius = float(planet["radius"])
        assert radius > 0.0, (role, radius)
        circle = Circle((x, y), radius, facecolor=color, edgecolor="black", alpha=0.35)
        ax.add_patch(circle)
        label = f'{role} {int(planet["slot"])}:{int(planet["id"])}'
        ax.text(x, y, label, ha="center", va="center", fontsize=7)

    title = (
        f'failed interception step={int(row["episode_step"])} '
        f'src={src} dst={dst} ships={int(row["ship_count"])} '
        f'aim={int(row["aim_index"])} reason={int(row["fail_reason"])} '
        f'source={row["source"]}\n'
        f'point={bool(solver_results["point"]["valid"])} '
        f'bisect={bool(solver_results["bisect"]["valid"])} '
        f'hybrid={bool(solver_results["hybrid"]["valid"])} '
        f'fair_fast={bool(solver_results["fair_fast"]["valid"])} '
        f'fair_slow={bool(solver_results["fair_slow"]["valid"])}'
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = _parse_args()
    row = _load_failed_interception(args.path, int(args.index))
    _plot_failed_interception(row)


if __name__ == "__main__":
    main()
