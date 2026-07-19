from __future__ import annotations

from typing import Any

import torch

from .dict_io_contract import (
    dict_io_contract_validation_enabled,
    validated_dict_io_contract_output,
)
from .obs_wrapper import (
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
)
from .wall_tree_profiler import WallTreeProfiler, profiler_span


def _validate_player_axis_nested(tree: Any, *, pad_to: int) -> Any:
    if isinstance(tree, dict):
        return {k: _validate_player_axis_nested(v, pad_to=pad_to) for k, v in tree.items()}
    assert isinstance(tree, torch.Tensor), (type(tree),)
    assert tree.ndim >= 1, (tree.ndim, tuple(tree.shape))
    n0 = int(tree.shape[0])
    assert n0 == pad_to, (n0, pad_to, tuple(tree.shape))
    return tree


def _noop_action_taken_index(device: torch.device) -> torch.Tensor:
    slots = torch.arange(
        ORBIT_PLANET_ACTION_SLOTS,
        device=device,
        dtype=torch.int32,
    ).view(ORBIT_PLANET_ACTION_SLOTS, 1)
    return slots * int(ORBIT_MOVE_CLASSES_PER_TARGET)


def _validate_padded_slots_have_noops(out: dict[str, Any], *, num_agents: int) -> None:
    na = int(num_agents)
    active_slots = frozenset(orbit_active_policy_slots(na))
    inactive_slots = tuple(i for i in range(ORBIT_PLAYER_AXIS_SLOTS) if i not in active_slots)
    if len(inactive_slots) == 0:
        return
    mask_leaves = (
        out["available_action_mask_CPP"],
    )
    taken_leaves = (
        out["action_taken_index_CPP"],
    )
    for available_action_mask in mask_leaves:
        assert isinstance(available_action_mask, torch.Tensor)
        assert tuple(available_action_mask.shape) == (
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PER_PLANET_MOVE_CLASSES,
        ), available_action_mask.shape
        assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
        expected = torch.zeros_like(available_action_mask[list(inactive_slots)])
        rows = torch.arange(
            ORBIT_PLANET_ACTION_SLOTS,
            device=available_action_mask.device,
            dtype=torch.int64,
        )
        noop_idx = rows * int(ORBIT_MOVE_CLASSES_PER_TARGET)
        expected[:, rows, noop_idx] = 1
        assert torch.equal(available_action_mask[list(inactive_slots)], expected), (
            inactive_slots,
            available_action_mask[list(inactive_slots)],
        )
    for action_taken_index in taken_leaves:
        assert isinstance(action_taken_index, torch.Tensor)
        assert tuple(action_taken_index.shape) == (
            ORBIT_PLAYER_AXIS_SLOTS,
            ORBIT_PLANET_ACTION_SLOTS,
            1,
        ), action_taken_index.shape
        assert action_taken_index.dtype == torch.int32, action_taken_index.dtype
        expected = _noop_action_taken_index(action_taken_index.device).expand(
            len(inactive_slots),
            ORBIT_PLANET_ACTION_SLOTS,
            1,
        )
        assert torch.equal(action_taken_index[list(inactive_slots)], expected), (
            inactive_slots,
            action_taken_index[list(inactive_slots)],
        )


class OrbitPaddingWrapper:
    """Validates that OrbitWarsEnv already emits policy-slot tensors."""

    def __init__(self, env: Any, flags: Any, wall_profiler: WallTreeProfiler | None = None) -> None:
        self.env = env
        self.flags = flags
        self._wall_prof = wall_profiler

    def _wall_span(self, name: str):
        return profiler_span(self._wall_prof, name)

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        out = self.env.reset(**kwargs)
        return self._padded_and_validated(out)

    def step(self, actions: Any) -> dict[str, Any]:
        with self._wall_span("wrap_padding_inner"):
            out = self.env.step(actions)
        with self._wall_span("wrap_padding_pad_validate"):
            return self._padded_and_validated(out)

    def _padded_and_validated(self, out: dict[str, Any]) -> dict[str, Any]:
        if not dict_io_contract_validation_enabled():
            return out
        na = int(self.env.num_agents)
        pad_to = int(ORBIT_PLAYER_AXIS_SLOTS)
        assert 1 <= na <= pad_to, (na, pad_to)
        merged = _validate_player_axis_nested(out, pad_to=pad_to)
        assert isinstance(merged, dict)
        _validate_padded_slots_have_noops(merged, num_agents=na)
        return validated_dict_io_contract_output(self.flags, merged, "orbit_wars_env_output")
