from __future__ import annotations

from typing import Any

from .dict_io_contract import maybe_validate_dict_io_contract
from .wall_tree_profiler import WallTreeProfiler, profiler_span

# Dot-separated source path → dot-separated dest path. Only these paths are kept.
# ``obs_LEARN_INFER``: substring matches both ``INFER`` and ``LEARN`` for ``get_buffers_with_tag``.
# ``*_STAT`` / ``*_LEARN_STAT`` names match buffer tagging (``get_buffers_with_tag``).
REMAP_SCHEMA: list[tuple[str, str]] = [
    ("obs", "obs_LEARN_INFER"),
    ("obs.action_taken_index", "action_taken_index_LEARN_STAT"),
    ("metrics", "metrics_STAT"),
    ("reward", "reward_LEARN_STAT"),
    ("done", "done_LEARN_STAT"),
    ("desync_done", "desync_done"),
    ("original_player_mask_STAT", "original_player_mask_STAT"),
    ("info", "info_STAT"),
]


def get_by_path(d: dict[str, Any], path: str) -> Any:
    """Read a leaf from a nested dict using dot-separated ``path`` (e.g. ``obs_LEARN_INFER.orbit_planet_mask``)."""
    parts = [p for p in path.split(".") if p]
    assert len(parts) > 0, "path must be non-empty"
    cur: Any = d
    for p in parts:
        assert isinstance(cur, dict), f"not a dict at segment {p!r} while resolving {path!r}"
        assert p in cur, f"missing key {p!r} while resolving {path!r}"
        cur = cur[p]
    return cur


def set_by_path(out: dict[str, Any], path: str, value: Any) -> None:
    """Write ``value`` at dot-separated ``path``, creating ``dict`` nodes as needed."""
    parts = [p for p in path.split(".") if p]
    assert len(parts) > 0, "path must be non-empty"
    cur = out
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            assert nxt is None, f"cannot nest under non-dict at {p!r} for {path!r}"
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


class RemapAndFilterWrapper:
    """
    On each ``reset`` / ``step``, builds a **new** dict: only ``REMAP_SCHEMA`` sources are copied,
    written under destination paths. Unlisted keys are dropped.
    """

    def __init__(self, env: Any, flags: Any, wall_profiler: WallTreeProfiler | None = None) -> None:
        self.env = env
        self.flags = flags
        self._wall_prof = wall_profiler
        self._schema = list(REMAP_SCHEMA)

    def _wall_span(self, name: str):
        return profiler_span(self._wall_prof, name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        return self._remap(self.env.reset(**kwargs))

    def step(self, actions: Any) -> dict[str, Any]:
        with self._wall_span("wrap_remap_inner"):
            inner_out = self.env.step(actions)
        with self._wall_span("wrap_remap_output"):
            return self._remap(inner_out)

    def _remap(self, out: dict[str, Any]) -> dict[str, Any]:
        assert isinstance(out, dict)
        result: dict[str, Any] = {}
        for src_path, dst_path in self._schema:
            set_by_path(result, dst_path, get_by_path(out, src_path))
        maybe_validate_dict_io_contract(self.flags, result, "remap_and_filter_wrapper_output")
        return result
