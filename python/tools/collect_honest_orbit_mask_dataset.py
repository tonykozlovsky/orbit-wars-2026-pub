from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import os
import random
import secrets
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from src.gym.obs_wrapper import (
    ORBIT_HIT_CLASSES_PER_TARGET,
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_ROW_LEN,
)
from src.gym.orbit_reference_upstream_random_state import (
    orbit_reference_upstream_random_derived_dict,
)
from src.gym.orbit_cpp_plain_sync import (
    orbit_comet_path_by_planet_id,
)
from src.gym.orbit_wars_cpp_ext import orbit_wars_cpp

_ORBIT_KAGGLE_DEFAULT_SHIP_SPEED = 6.0
_ORBIT_KAGGLE_DEFAULT_COMET_SPEED = 4.0
_ORBIT_KAGGLE_DEFAULT_EPISODE_STEPS = 500
_COLLECT_HONEST_MASK_DATASET_NUM_AGENTS = 4
_ORBIT_DATASET_GZIP_COMPRESSLEVEL = 5
_PLANET_ROW_X = 2
_PLANET_ROW_Y = 3
_PLANET_ROW_RADIUS = 4
_PLANET_ROW_ID = 0
_HIT_KIND_NAMES = (
    "none",
    "target",
    "static",
    "dynamic",
    "sun",
    "out_of_board",
    "timeout",
    "end_of_game",
    "interception_failed",
    "verified_timeout",
)
_WORKER_ORBIT_INSTANCE_ID_STRIDE = 1_000_000


def _planet_position_hash_sha256(planet_rows: torch.Tensor) -> str:
    assert isinstance(planet_rows, torch.Tensor)
    assert tuple(planet_rows.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PLANET_ROW_LEN,
    ), planet_rows.shape
    assert not planet_rows.is_cuda
    position_columns = torch.tensor(
        [_PLANET_ROW_ID, _PLANET_ROW_X, _PLANET_ROW_Y],
        dtype=torch.int64,
    )
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


def _planet_rows_tensor_from_plain(planets: list[object]) -> tuple[torch.Tensor, int]:
    assert isinstance(planets, list)
    n = len(planets)
    assert n <= ORBIT_PLANET_ACTION_SLOTS, (n, ORBIT_PLANET_ACTION_SLOTS)
    rows = torch.zeros((ORBIT_PLANET_ACTION_SLOTS, ORBIT_PLANET_ROW_LEN), dtype=torch.float64)
    for i, row_obj in enumerate(planets):
        assert isinstance(row_obj, (list, tuple)) and len(row_obj) == ORBIT_PLANET_ROW_LEN
        for j in range(ORBIT_PLANET_ROW_LEN):
            rows[i, j] = float(row_obj[j])
    return rows.contiguous(), n

def _planet_rows_tensor(
    rows_py: list, *, slots: int, row_len: int
) -> tuple[torch.Tensor, int]:
    n = len(rows_py)
    assert n <= slots, (n, slots)
    t = torch.zeros((slots, row_len), dtype=torch.float32)
    for i in range(n):
        row = rows_py[i]
        assert len(row) == row_len, (len(row), row_len)
        for j in range(row_len):
            t[i, j] = float(row[j])
    return t, n


def _honest_mask_step_tensor_for_dataset(mask: torch.Tensor) -> torch.Tensor:
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.int8, mask.dtype
    assert not mask.is_cuda
    assert tuple(mask.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), mask.shape
    return mask.contiguous().clone()


def _honest_shared_mask(
    *,
    cache: object,
    step_idx: int,
    _buf: torch.Tensor,
) -> torch.Tensor:
    cache.honest_shared_action_mask_all_geometry(int(step_idx), _buf)
    assert isinstance(_buf, torch.Tensor)
    assert tuple(_buf.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), _buf.shape
    assert not _buf.is_cuda
    return _buf


def _honest_shared_hit_kind_last(
    *,
    env: object,
) -> torch.Tensor:
    t = env.honest_shared_hit_kind_last()
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), t.shape
    assert not t.is_cuda
    return t


def _plot_timeout_debug_sample(sample: dict[str, object]) -> None:
    step_index = int(sample["step_index"])
    src_slot = int(sample["src_slot"])
    dst_slot = int(sample["dst_slot"])
    ship_count = int(sample["ship_count"])
    hit_frame = int(sample["hit_frame"])
    fleet_xy = sample["fleet_xy_by_frame"]
    planet_rows = sample["planet_rows_from_step"]
    intercept_trace = sample["intercept_trace_cpp"]
    assert isinstance(fleet_xy, torch.Tensor)
    assert isinstance(planet_rows, torch.Tensor)
    assert isinstance(intercept_trace, torch.Tensor)
    assert fleet_xy.ndim == 2 and fleet_xy.shape[1] == 3, fleet_xy.shape
    assert planet_rows.ndim == 3 and planet_rows.shape[2] == ORBIT_PLANET_ROW_LEN, planet_rows.shape

    board_min = 0.0
    board_max = 100.0
    sun_x = 50.0
    sun_y = 50.0
    sun_r = 10.0

    n_frames = int(fleet_xy.shape[0])
    if n_frames <= 0:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.14)
    slider_ax = fig.add_axes((0.14, 0.04, 0.72, 0.04))
    slider = Slider(
        ax=slider_ax,
        label="frame",
        valmin=0,
        valmax=max(0, n_frames - 1),
        valinit=0,
        valstep=1,
    )
    frame_idx = [0]

    def _draw(i: int) -> None:
        i = max(0, min(i, n_frames - 1))
        frame_abs = int(fleet_xy[i, 0].item())
        frame_rel = frame_abs - step_index
        rows = None
        if 0 <= frame_rel < int(planet_rows.shape[0]):
            rows = planet_rows[frame_rel]
        ax.clear()
        ax.set_xlim(board_min, board_max)
        ax.set_ylim(board_min, board_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"timeout debug frame={frame_abs} src={src_slot} dst={dst_slot} "
            f"ship_count={ship_count} hit_frame={hit_frame}\n"
            "keys: left/right, home/end"
        )
        ax.add_patch(plt.Circle((sun_x, sun_y), sun_r, fill=False, color="orange", linewidth=1.5))
        if rows is not None:
            for slot in range(int(rows.shape[0])):
                p = rows[slot]
                pid = int(p[0].item())
                if pid < 0:
                    continue
                px = float(p[_PLANET_ROW_X].item())
                py = float(p[_PLANET_ROW_Y].item())
                pr = float(p[_PLANET_ROW_RADIUS].item())
                color = (
                    "tab:green"
                    if slot == src_slot
                    else ("tab:red" if slot == dst_slot else "tab:blue")
                )
                ax.add_patch(plt.Circle((px, py), pr, fill=False, color=color, linewidth=1.0))
        if int(intercept_trace.numel()) > 0:
            sx0 = float(planet_rows[0, src_slot, _PLANET_ROW_X].item())
            sy0 = float(planet_rows[0, src_slot, _PLANET_ROW_Y].item())
            for it in range(int(intercept_trace.shape[0])):
                valid = float(intercept_trace[it, 3].item()) > 0.5
                if not valid:
                    continue
                tx = float(intercept_trace[it, 0].item())
                ty = float(intercept_trace[it, 1].item())
                turns = float(intercept_trace[it, 2].item())
                a = math.degrees(math.atan2(ty - sy0, tx - sx0))
                ax.scatter([tx], [ty], color="purple", s=22, zorder=5)
                ax.plot([sx0, tx], [sy0, ty], color="purple", linewidth=0.8, alpha=0.35, linestyle="--")
                ax.text(tx + 0.4, ty + 0.4, f"i{it}:{a:.1f}° t={turns:.2f}", color="purple", fontsize=7)
        path = fleet_xy[: i + 1, 1:3]
        ax.plot(path[:, 0].cpu().numpy(), path[:, 1].cpu().numpy(), color="black", linewidth=1.5)
        fx = float(fleet_xy[i, 1].item())
        fy = float(fleet_xy[i, 2].item())
        ax.scatter([fx], [fy], color="magenta", s=30)
        fig.canvas.draw_idle()

    def _on_slider(val: float) -> None:
        idx = int(val)
        frame_idx[0] = idx
        _draw(idx)

    def _on_key(event: object) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        idx = frame_idx[0]
        if key in ("right", "d"):
            idx += 1
        elif key in ("left", "a"):
            idx -= 1
        elif key == "home":
            idx = 0
        elif key == "end":
            idx = n_frames - 1
        else:
            return
        idx = max(0, min(idx, n_frames - 1))
        frame_idx[0] = idx
        slider.set_val(idx)

    slider.on_changed(_on_slider)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _draw(0)
    plt.show()


def _limit_torch_cpu_threads() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _orbit_wars_episode_seed(*, instance_id: int, episode_index: int) -> int:
    a = int(instance_id)
    b = int(episode_index)
    assert 0 <= a, a
    assert 0 <= b < (1 << 32), b
    seed_u64 = (a << 32) | b
    return int(random.Random(seed_u64).randrange(1 << 31))


def _first_missing_sequential_episode_id(out_dir: Path) -> int:
    assert out_dir.is_dir(), out_dir
    existing_ids: set[int] = set()
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix == ".pt" and path.stem.isdecimal():
            existing_ids.add(int(path.stem))
    episode_id = 0
    while episode_id in existing_ids:
        episode_id += 1
    return episode_id


def _atomic_write_payload_gzip(final_path: Path, payload: dict[str, object]) -> None:
    final_path = final_path.resolve()
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=f".tmp.{os.getpid()}",
        dir=str(parent),
    )
    try:
        with os.fdopen(fd, "wb") as wf:
            with gzip.GzipFile(
                fileobj=wf,
                mode="wb",
                compresslevel=_ORBIT_DATASET_GZIP_COMPRESSLEVEL,
            ) as gf:
                torch.save(payload, gf)
            wf.flush()
            os.fsync(wf.fileno())
    except BaseException:
        os.unlink(tmp_path)
        raise
    os.replace(tmp_path, final_path)


def _collect_one_episode(
    *,
    cache: object,
    na: int,
    orbit_instance_id: int,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
    plot_timeout_debug: bool,
    profile: bool,
) -> tuple[
    dict[str, object],
    int,
    torch.Tensor,
    dict[str, object] | None,
    dict[str, tuple[float, int]],
]:
    ep_i = int(episode_index)
    assert 0 <= ep_i < (1 << 32), ep_i
    episode_seed = int(episode_seed)
    assert 0 <= episode_seed < (1 << 31), episode_seed
    random_state = orbit_reference_upstream_random_derived_dict(
        seed=episode_seed,
        num_agents=na,
        comet_speed=_ORBIT_KAGGLE_DEFAULT_COMET_SPEED,
    )
    planet_rows, planet_count = _planet_rows_tensor_from_plain(random_state["planets"])
    cache.reset(float(random_state["angular_velocity"]), planet_rows, int(planet_count))
    comet_sync_updates = random_state["comet_sync_updates"]
    assert isinstance(comet_sync_updates, list)
    next_comet_u = 0

    def drain_comets_for_clock(clock: int) -> None:
        nonlocal next_comet_u
        while (
            next_comet_u < len(comet_sync_updates)
            and int(comet_sync_updates[next_comet_u]["episode_step"]) == clock
        ):
            upd = comet_sync_updates[next_comet_u]
            c_ids = upd["comet_planet_ids"]
            groups = upd["comets_groups"]
            assert isinstance(c_ids, list), (clock, c_ids)
            assert isinstance(groups, list), (clock, groups)
            assert len(c_ids) in (0, 4), (clock, c_ids)
            if len(c_ids) == 0:
                assert len(groups) == 0, (clock, groups)
                next_comet_u += 1
                continue
            assert len(groups) == 1, (clock, groups)
            planets = upd["planets"]
            assert isinstance(planets, list)
            path_by_pid = orbit_comet_path_by_planet_id(groups, planets)
            ids = sorted(int(pid) for pid in c_ids)
            assert set(path_by_pid.keys()) == set(ids), (clock, sorted(path_by_pid.keys()), ids)
            path_index = int(path_by_pid[ids[0]][0])
            assert all(int(path_by_pid[pid][0]) == path_index for pid in ids), clock
            cache.update_comet_in_noop_cache(
                int(clock),
                path_index,
                int(clock) - path_index,
                ids,
                [list(path_by_pid[pid][1]) for pid in ids],
            )
            next_comet_u += 1

    assert int(cache.noop_trajectory_length()) >= 1
    noop_planet_rows = cache.noop_trajectory_planets_tensor()
    assert isinstance(noop_planet_rows, torch.Tensor)
    assert noop_planet_rows.ndim == 3, noop_planet_rows.shape
    assert tuple(noop_planet_rows.shape[1:]) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PLANET_ROW_LEN,
    ), noop_planet_rows.shape
    honest_mask_buf = torch.zeros(
        (ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_HIT_CLASSES),
        dtype=torch.int8,
        device=torch.device("cpu"),
    )
    steps_out: list[dict[str, object]] = []
    hit_kind_counts = torch.zeros((len(_HIT_KIND_NAMES),), dtype=torch.int64)
    debug_timeout_sample: dict[str, object] | None = None
    debug_timeout_plotted = False
    profile_rows: dict[str, tuple[float, int]] = {}
    cache.set_wall_profile_enabled(bool(profile))

    drain_comets_for_clock(0)
    noop_planet_rows = cache.noop_trajectory_planets_tensor()
    max_episode_steps = int(_ORBIT_KAGGLE_DEFAULT_EPISODE_STEPS)
    clock = 0

    while True:
        cpp_k = int(clock)
        assert 0 <= cpp_k < int(noop_planet_rows.shape[0]), (cpp_k, tuple(noop_planet_rows.shape))
        masks = _honest_shared_mask(
            cache=cache,
            step_idx=cpp_k,
            _buf=honest_mask_buf,
        )
        if profile:
            for row in cache.wall_profile_rows():
                assert isinstance(row, tuple), type(row)
                assert len(row) == 3, row
                name, sum_ms, n_calls = row
                key = str(name)
                prev_sum_ms, prev_n_calls = profile_rows.get(key, (0.0, 0))
                profile_rows[key] = (
                    prev_sum_ms + float(sum_ms),
                    prev_n_calls + int(n_calls),
                )
        hit_kind = _honest_shared_hit_kind_last(env=cache).to(torch.int64)
        for kind_idx in range(len(_HIT_KIND_NAMES)):
            hit_kind_counts[kind_idx] += int((hit_kind == kind_idx).sum().item())
        if (
            plot_timeout_debug
            and debug_timeout_sample is not None
            and not debug_timeout_plotted
        ):
            _plot_timeout_debug_sample(debug_timeout_sample)
            debug_timeout_plotted = True
        steps_out.append(
            {
                "honest_shared_action_mask": _honest_mask_step_tensor_for_dataset(masks),
                "planet_position_hash_sha256": _planet_position_hash_sha256(
                    noop_planet_rows[cpp_k]
                ),
            }
        )
        if cpp_k + 1 >= max_episode_steps:
            break
        if max_steps > 0 and len(steps_out) >= max_steps:
            break
        drain_comets_for_clock(cpp_k)
        noop_planet_rows = cache.noop_trajectory_planets_tensor()
        clock = cpp_k + 1

    n_steps = len(steps_out)

    payload: dict[str, object] = {
        "honest_shared_action_mask_encoding": "int8_tensor_v1",
        "planet_position_hash_encoding": (
            "sha256_orbit_planet_position_hash_sha256_v1_float32_id_x_y_44_slots"
        ),
        "serialization": (
            f"torch_save_gzip_compresslevel_{_ORBIT_DATASET_GZIP_COMPRESSLEVEL}_wb"
        ),
        "honest_shared_note": (
            "CppEnvStaticCacheV2.honest_shared_action_mask_all_geometry over all src/dst buckets; "
            "Matches orbit_wars_env honest-mask phase: drain comet_sync_updates for clock T "
            "only after recording step T (hash noop row T before comets(T)); "
            "then the Python replay clock advances to the next static-cache frame."
        ),
        "num_agents": na,
        "orbit_instance_id": int(orbit_instance_id),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "ship_speed": _ORBIT_KAGGLE_DEFAULT_SHIP_SPEED,
        "episode_steps_cap": _ORBIT_KAGGLE_DEFAULT_EPISODE_STEPS,
        "comet_speed": _ORBIT_KAGGLE_DEFAULT_COMET_SPEED,
        "steps": steps_out,
    }
    if debug_timeout_sample is not None:
        payload["debug_timeout_sample"] = debug_timeout_sample

    cache.set_wall_profile_enabled(False)
    return payload, n_steps, hit_kind_counts, debug_timeout_sample, profile_rows


def _worker_main(
    worker_id: int,
    out_dir_str: str,
    episodes_completed,
    counter_lock,
    print_lock,
    orbit_instance_id_base: int,
    max_steps: int,
    plot_timeout_debug: bool,
    max_episodes: int,
    sequential: bool,
    fixed_episode_seed: int,
    next_sequential_episode_id,
    profile: bool,
) -> None:
    _limit_torch_cpu_threads()
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    na = int(_COLLECT_HONEST_MASK_DATASET_NUM_AGENTS)
    assert na in (2, 4), na
    assert max_steps >= 0, max_steps

    orbit_instance_id = int(orbit_instance_id_base) + int(worker_id) * int(
        _WORKER_ORBIT_INSTANCE_ID_STRIDE
    )
    assert orbit_instance_id >= 0, orbit_instance_id

    cache = orbit_wars_cpp.CppEnvStaticCacheV2(
        na,
        orbit_instance_id,
        _ORBIT_KAGGLE_DEFAULT_SHIP_SPEED,
        _ORBIT_KAGGLE_DEFAULT_EPISODE_STEPS,
        _ORBIT_KAGGLE_DEFAULT_COMET_SPEED,
    )

    while True:
        with counter_lock:
            if max_episodes >= 0 and int(episodes_completed.value) >= max_episodes:
                break
            if fixed_episode_seed >= 0:
                episode_index = int(fixed_episode_seed)
            elif sequential:
                episode_index = int(next_sequential_episode_id.value)
                next_sequential_episode_id.value += 1

        if fixed_episode_seed >= 0:
            episode_seed = int(fixed_episode_seed)
        elif sequential:
            episode_seed = int(episode_index)
        else:
            episode_index = secrets.randbelow(1 << 32)
            episode_seed = _orbit_wars_episode_seed(
                instance_id=int(orbit_instance_id),
                episode_index=int(episode_index),
            )
        out_path = out_dir / f"{episode_seed}.pt"
        if out_path.is_file() and fixed_episode_seed < 0:
            continue

        with print_lock:
            print(
                f"[w{worker_id}] seed={episode_seed} episode_index={episode_index} start",
                flush=True,
            )

        payload, n_steps, hit_kind_counts, debug_timeout_sample, profile_rows = (
            _collect_one_episode(
                cache=cache,
                na=na,
                orbit_instance_id=int(orbit_instance_id),
                episode_index=int(episode_index),
                episode_seed=int(episode_seed),
                max_steps=int(max_steps),
                plot_timeout_debug=bool(plot_timeout_debug),
                profile=bool(profile),
            )
        )

        _atomic_write_payload_gzip(out_path, payload)
        with counter_lock:
            episodes_completed.value += 1

        counts_str = " ".join(
            f"{name}={int(hit_kind_counts[i].item())}"
            for i, name in enumerate(_HIT_KIND_NAMES)
        )
        with print_lock:
            print(
                f"[w{worker_id}] seed={episode_seed} steps={n_steps} saved",
                flush=True,
            )
            print(
                f"[w{worker_id}] seed={episode_seed} hit_kind_counts {counts_str}",
                flush=True,
            )
            if profile:
                for name, (sum_ms, n_calls) in sorted(
                    profile_rows.items(),
                    key=lambda item: item[1][0],
                    reverse=True,
                ):
                    avg_ms = float(sum_ms) / float(n_calls)
                    print(
                        f"[w{worker_id}] seed={episode_seed} profile "
                        f"{name}|sum_ms={sum_ms:.3f}|n={n_calls}|avg_ms={avg_ms:.3f}",
                        flush=True,
                    )
            if debug_timeout_sample is not None:
                print(
                    f"[w{worker_id}] seed={episode_seed} timeout_debug "
                    f"step={int(debug_timeout_sample['step_index'])} "
                    f"src={int(debug_timeout_sample['src_slot'])} "
                    f"dst={int(debug_timeout_sample['dst_slot'])} "
                    f"sn={int(debug_timeout_sample['ship_subindex'])} "
                    f"ship_count={int(debug_timeout_sample['ship_count'])} "
                    f"hit_frame={int(debug_timeout_sample['hit_frame'])}",
                    flush=True,
                )

        if fixed_episode_seed >= 0:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=-1,
        help="Stop after this many episodes in total across workers (-1 = run forever).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--orbit-instance-id",
        type=int,
        default=0,
        help=(
            "Base Orbit instance id for env RNG lineage; worker w uses "
            f"id + w * {_WORKER_ORBIT_INSTANCE_ID_STRIDE} so parallel workers do not "
            "reuse the same (instance_id, episode_index) space."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-reset-traces", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--plot-timeout-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Generate dataset episodes with episode_seed/id 0, 1, 2, ...; "
            "start from the first missing <id>.pt in --out-dir and skip existing ids."
        ),
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=-1,
        help="Generate exactly this episode_seed/id once, save <seed>.pt, then exit (-1 = disabled).",
    )
    args = parser.parse_args()

    num_workers = int(args.num_workers)
    assert num_workers >= 1, num_workers

    max_episodes = int(args.max_episodes)
    assert max_episodes >= -1, max_episodes
    fixed_episode_seed = int(args.episode_seed)
    assert -1 <= fixed_episode_seed < (1 << 31), fixed_episode_seed

    if num_workers > 1 and bool(args.plot_timeout_debug):
        raise SystemExit(
            "--plot-timeout-debug is incompatible with --num-workers > 1 "
            "(use --num-workers 1)."
        )
    if fixed_episode_seed >= 0 and num_workers != 1:
        raise SystemExit("--episode-seed requires --num-workers 1.")
    if fixed_episode_seed >= 0 and bool(args.sequential):
        raise SystemExit("--episode-seed is incompatible with --sequential.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_sequential_episode_id = _first_missing_sequential_episode_id(out_dir)

    ctx = get_context("spawn")
    episodes_completed = ctx.Value("q", 0)
    next_sequential_episode_id = ctx.Value("q", first_sequential_episode_id)
    counter_lock = ctx.Lock()
    print_lock = ctx.Lock()

    worker_args = (
        str(out_dir.resolve()),
        episodes_completed,
        counter_lock,
        print_lock,
        int(args.orbit_instance_id),
        int(args.max_steps),
        bool(args.plot_timeout_debug),
        max_episodes,
        bool(args.sequential),
        fixed_episode_seed,
        next_sequential_episode_id,
        bool(args.profile),
    )

    if num_workers == 1:
        _worker_main(0, *worker_args)
        return

    procs = [
        ctx.Process(
            target=_worker_main,
            args=(wid,) + worker_args,
            name=f"orbit_mask_collect_{wid}",
        )
        for wid in range(num_workers)
    ]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=30.0)


if __name__ == "__main__":
    main()
