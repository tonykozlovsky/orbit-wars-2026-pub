from __future__ import annotations

from typing import Any

import torch

from .dict_io_contract import validated_dict_io_contract_output
from .obs_wrapper import ORBIT_PLAYER_AXIS_SLOTS
from .wall_tree_profiler import WallTreeProfiler, profiler_span


def _metrics_ref_vector(metrics: dict[str, Any]) -> torch.Tensor:
    v0 = next(iter(metrics.values()))
    assert isinstance(v0, torch.Tensor) and v0.ndim == 1
    assert int(v0.shape[0]) == ORBIT_PLAYER_AXIS_SLOTS
    return v0


class LoggingWrapper:
    """Per-seat cumulative sums of ``metrics`` in ``info`` as ``LOGGING_STAT_{name}_cumsum``.

    Every ``metrics`` value is a length-``ORBIT_PLAYER_AXIS_SLOTS`` 1-D tensor. ``info`` cumsums use the
    same shape. Sums update on ``step`` only; ``reset`` clears sums and does not append.
    """

    def __init__(self, env: Any, flags: Any, wall_profiler: WallTreeProfiler | None = None) -> None:
        self.env = env
        self.flags = flags
        self._wall_prof = wall_profiler
        self._metric_sums: dict[str, list[float]] = {}

    def _wall_span(self, name: str):
        return profiler_span(self._wall_prof, name)

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        self._metric_sums.clear()
        out = self.env.reset(**kwargs)
        merged = self._out_with_current_sums(out)
        return validated_dict_io_contract_output(self.flags, merged, "logging_wrapper_output")

    def step(self, actions: Any) -> dict[str, Any]:
        with self._wall_span("wrap_logging_inner"):
            out = self.env.step(actions)
        with self._wall_span("wrap_logging_merge"):
            self._append_metrics(out["metrics"])
            merged = self._out_with_current_sums(out)
            return validated_dict_io_contract_output(self.flags, merged, "logging_wrapper_output")

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        n = ORBIT_PLAYER_AXIS_SLOTS
        for k, v in metrics.items():
            assert isinstance(v, torch.Tensor) and tuple(v.shape) == (n,)
            base = self._metric_sums.get(k) or [0.0] * n
            assert len(base) == n
            self._metric_sums[k] = [base[i] + float(v[i].item()) for i in range(n)]

    def _out_with_current_sums(self, out: dict[str, Any]) -> dict[str, Any]:
        metrics = out["metrics"]
        ref = _metrics_ref_vector(metrics)
        n = ORBIT_PLAYER_AXIS_SLOTS
        info: dict[str, Any] = {}
        for k in metrics.keys():
            v = metrics[k]
            assert isinstance(v, torch.Tensor) and tuple(v.shape) == (n,)
            sums = self._metric_sums.get(k) or [0.0] * n
            assert len(sums) == n
            info[f"LOGGING_STAT_{k}_cumsum"] = torch.stack(
                [
                    torch.as_tensor(float(sums[i]), dtype=ref.dtype, device=ref.device)
                    for i in range(n)
                ],
                dim=0,
            )
        return {**out, "info": info}
