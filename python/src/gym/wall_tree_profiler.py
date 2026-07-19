import logging
import os
import sys
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Iterator, TextIO

import torch

logger = logging.getLogger(__name__)


class WallTreeProfiler:
    """Wall-time per path ``a/b/c``; each exit records cumulative and latest duration."""

    __slots__ = ("_stack", "_sum_ms", "_n", "_last_ms")

    def __init__(self) -> None:
        self._stack: list[str] = []
        self._sum_ms: dict[str, float] = {}
        self._n: dict[str, int] = {}
        self._last_ms: dict[str, float] = {}

    def clear(self) -> None:
        assert len(self._stack) == 0
        self._sum_ms.clear()
        self._n.clear()
        self._last_ms.clear()

    @contextmanager
    def __call__(self, name: str):
        self._stack.append(name)
        path = "/".join(self._stack)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._sum_ms[path] = self._sum_ms.get(path, 0.0) + dt_ms
            self._n[path] = self._n.get(path, 0) + 1
            self._last_ms[path] = dt_ms
            popped = self._stack.pop()
            assert popped == name

    span = __call__

    def root_children_accounted_ms(self) -> float:
        return sum(ms for path, ms in self._sum_ms.items() if "/" not in path)

    def add_subtree_rows(self, rows: list[tuple[str, float, int, float]]) -> None:
        assert len(self._stack) >= 1
        parent = "/".join(self._stack)
        for rel_path, sum_ms, n, last_ms in rows:
            assert isinstance(rel_path, str) and rel_path != "" and not rel_path.startswith("/")
            assert isinstance(sum_ms, float), type(sum_ms)
            assert isinstance(n, int) and n >= 1, n
            assert isinstance(last_ms, float), type(last_ms)
            path = f"{parent}/{rel_path}"
            self._sum_ms[path] = self._sum_ms.get(path, 0.0) + sum_ms
            self._n[path] = self._n.get(path, 0) + n
            self._last_ms[path] = last_ms

    @staticmethod
    def _tree_line_start_col(ancestor_anchor_cols: list[int]) -> int:
        if not ancestor_anchor_cols:
            return 0
        return int(ancestor_anchor_cols[-1]) + 2

    @staticmethod
    def _tree_line_prefix(ancestor_anchor_cols: list[int]) -> str:
        line_start_col = WallTreeProfiler._tree_line_start_col(ancestor_anchor_cols)
        if line_start_col == 0:
            return ""
        buf = [" "] * line_start_col
        for col in ancestor_anchor_cols:
            buf[col] = "|"
        return "".join(buf)

    @staticmethod
    def _format_tree_row(
        segment: str,
        ancestor_anchor_cols: list[int],
        *,
        sum_ms: float,
        n: int,
        avg_ms: float,
        last_ms: float,
    ) -> str:
        prefix = WallTreeProfiler._tree_line_prefix(ancestor_anchor_cols)
        line_start_col = WallTreeProfiler._tree_line_start_col(ancestor_anchor_cols)
        assert line_start_col == len(prefix), (line_start_col, len(prefix), prefix)
        return (
            f"{prefix}{segment}|sum_ms={sum_ms:.3f}|n={n}|avg_ms={avg_ms:.3f}|last_ms={last_ms:.3f}"
        )

    def tree_lines(self) -> list[str]:
        assert len(self._stack) == 0
        paths = frozenset(self._sum_ms.keys())

        def avg_ms(path: str) -> float:
            total = float(self._sum_ms[path])
            n = int(self._n[path])
            return total / float(n)

        def direct_children_raw(parent: str) -> list[str]:
            if parent == "":
                return [q for q in paths if "/" not in q]
            plen = len(parent)
            layer: list[str] = []
            prefix = parent + "/"
            for q in paths:
                if not q.startswith(prefix):
                    continue
                rest = q[plen + 1 :]
                if "/" in rest:
                    continue
                layer.append(q)
            return layer

        lines: list[str] = []

        def walk(parent: str, ancestor_anchor_cols: list[int]) -> None:
            raw = direct_children_raw(parent)
            if not raw:
                return

            ch_tot = sum(float(self._sum_ms[c]) for c in raw)
            ch_last = sum(float(self._last_ms[c]) for c in raw)
            if parent != "":
                p_n = int(self._n[parent])
                other_sum = float(self._sum_ms[parent]) - ch_tot
                other_avg = other_sum / float(p_n) if p_n else 0.0
                other_last = float(self._last_ms[parent]) - ch_last
            else:
                p_n = 0
                other_sum = 0.0
                other_avg = 0.0
                other_last = 0.0

            rows: list[tuple[tuple[float, str], str, str | None]] = []
            for c in raw:
                rows.append(((-avg_ms(c), c), "node", c))
            if parent != "":
                rows.append(((-other_avg, "other"), "other", None))

            rows.sort(key=lambda x: x[0])

            for _key, kind, cpath in rows:
                if kind == "node":
                    assert cpath is not None
                    segment = cpath.split("/")[-1]
                    total = float(self._sum_ms[cpath])
                    n = int(self._n[cpath])
                    a = total / float(n)
                    last = float(self._last_ms[cpath])
                    lines.append(
                        self._format_tree_row(
                            segment,
                            ancestor_anchor_cols,
                            sum_ms=total,
                            n=n,
                            avg_ms=a,
                            last_ms=last,
                        )
                    )
                    line_start_col = self._tree_line_start_col(ancestor_anchor_cols)
                    child_anchor_cols = ancestor_anchor_cols + [line_start_col]
                    walk(cpath, child_anchor_cols)
                else:
                    lines.append(
                        self._format_tree_row(
                            "other",
                            ancestor_anchor_cols,
                            sum_ms=other_sum,
                            n=p_n,
                            avg_ms=other_avg,
                            last_ms=other_last,
                        )
                    )

        walk("", [])
        return lines

    def summary_stdout(
        self,
        title: str,
        *,
        iteration_wall_ms: float | None = None,
        line_prefix: str = "BENCHMARK_WALL_TREE ",
        file: TextIO | None = None,
    ) -> None:
        assert len(self._stack) == 0
        accounted = self.root_children_accounted_ms()
        out = sys.stdout if file is None else file
        print(f"{line_prefix}{title} total_profiled_ms={accounted:.3f}", file=out, flush=True)
        if iteration_wall_ms is not None:
            print(f"{line_prefix}{title} last_rollout_wall_ms={iteration_wall_ms:.3f}", file=out, flush=True)
        for line in self.tree_lines():
            print(f"{line_prefix}{line}", file=out, flush=True)

    def summary(self, title: str, *, wall_ms: float | None = None) -> None:
        assert len(self._stack) == 0
        accounted = self.root_children_accounted_ms()
        logger.info("%s total_profiled_ms=%.3f", title, accounted)
        if wall_ms is not None:
            gap_ms = float(wall_ms) - accounted
            logger.info("%s wall_ms=%.3f gap_ms=%.3f", title, wall_ms, gap_ms)
        for line in self.tree_lines():
            logger.info("%s", line)


def profiler_span(profiler: WallTreeProfiler | None, name: str):
    if profiler is None:
        return nullcontext()
    return profiler(name)


def model_profile_enabled() -> bool:
    if torch.compiler.is_compiling():
        return False
    return os.environ.get("MODEL_PROFILE", "").strip() == "1"


def wall_tree_cuda_model_block(
    profiler: WallTreeProfiler | None,
    device: torch.device,
    name: str,
) -> AbstractContextManager[None]:
    """Model wall-tree spans; CUDA syncs at span edges when profiling CUDA kernels."""
    if not model_profile_enabled():
        return nullcontext()
    assert device.type in ("cpu", "cuda"), device
    if device.type == "cpu":
        return profiler_span(profiler, name)
    return _cuda_synchronized_model_block(profiler, device, name)


@contextmanager
def _cuda_synchronized_model_block(
    profiler: WallTreeProfiler | None,
    device: torch.device,
    name: str,
) -> Iterator[None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with profiler_span(profiler, name):
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
