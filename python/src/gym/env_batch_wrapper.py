from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .dict_io_contract import maybe_validate_dict_io_contract
from .obs_wrapper import (
    ORBIT_MAX_PLANETS,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
)
from .orbit_wars_env import (
    ORBIT_STEP_KEY_ORBIT_PAIRWISE_CLASSES,
    ORBIT_STEP_KEY_TAPE_BASELINE_LEARN,
    ORBIT_STEP_KEY_TAPE_SUPERVISED_PRED,
    ORBIT_STEP_KEY_TAPE_SUPERVISED_TARGET,
    ORBIT_STEP_KEY_TAPE_SUPERVISED_VALID,
)
from .wall_tree_profiler import WallTreeProfiler

_ACTION_TAKEN_STAT_KEY = "action_taken_index_LEARN_STAT"
_OBS_KEY = "obs_LEARN_INFER"
_OBS_ACTION_TAKEN_KEY = "action_taken_index"


def _stack_tree_along_env_dim(outs: list[Any]) -> Any:
    assert len(outs) >= 1
    first = outs[0]
    if len(outs) == 1:
        if isinstance(first, dict):
            return {k: _stack_tree_along_env_dim([v]) for k, v in first.items()}
        if isinstance(first, torch.Tensor):
            return first.unsqueeze(0)
        raise TypeError(
            f"EnvBatchWrapper expects dict or tensor leaves, got {type(first)!r}"
        )
    if isinstance(first, dict):
        return {k: _stack_tree_along_env_dim([o[k] for o in outs]) for k in first}
    if isinstance(first, torch.Tensor):
        return torch.stack([o for o in outs], dim=0)
    raise TypeError(
        f"EnvBatchWrapper expects dict or tensor leaves, got {type(first)!r}"
    )


def _stack_remapped_env_outputs(outs: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(outs) >= 1
    assert _OBS_KEY in outs[0], outs[0].keys()
    obs0 = outs[0][_OBS_KEY]
    assert isinstance(obs0, dict)
    assert _OBS_ACTION_TAKEN_KEY in obs0, obs0.keys()
    assert _ACTION_TAKEN_STAT_KEY in outs[0], outs[0].keys()
    out: dict[str, Any] = {}
    for key in outs[0]:
        if key == _ACTION_TAKEN_STAT_KEY:
            continue
        out[key] = _stack_tree_along_env_dim([o[key] for o in outs])
    obs = out[_OBS_KEY]
    assert isinstance(obs, dict)
    action_taken = obs[_OBS_ACTION_TAKEN_KEY]
    assert isinstance(action_taken, torch.Tensor)
    out[_ACTION_TAKEN_STAT_KEY] = action_taken
    return out


class EnvBatchWrapper:
    """Runs ``reset`` / ``step`` over ``n_actor_envs`` inner envs; tensor leaves are ``torch.stack`` on dim 0.

    ``step(actions)`` takes ``Tensor[E, ORBIT_PLAYER_AXIS_SLOTS, ORBIT_PLANET_ACTION_SLOTS, 1]`` (policy
    pairwise class indices). Row ``ei`` is split into per-seat CPU ``int64`` vectors for ``envs[ei].step``.
    """

    def __init__(
        self,
        envs_by_num_agents: dict[int, list[Any]],
        *,
        orbit_num_agents: int,
        flags: Any | None = None,
        wall_profiler: WallTreeProfiler | None = None,
    ) -> None:
        na = int(orbit_num_agents)
        assert na in (2, 4), na
        assert set(envs_by_num_agents.keys()) == {2, 4}, envs_by_num_agents.keys()
        assert len(envs_by_num_agents[2]) >= 1
        assert len(envs_by_num_agents[2]) == len(envs_by_num_agents[4]), (
            len(envs_by_num_agents[2]),
            len(envs_by_num_agents[4]),
        )
        self._envs_by_num_agents = envs_by_num_agents
        self._active_num_agents = [na] * len(envs_by_num_agents[na])
        self.envs = self._active_envs()
        self._flags = flags
        self._wall_prof = wall_profiler

    def _wall_span(self, name: str):
        p = self._wall_prof
        if p is None:
            return nullcontext()
        return p(name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.envs[0], name)

    def _active_envs(self) -> list[Any]:
        return [
            self._envs_by_num_agents[int(na)][i]
            for i, na in enumerate(self._active_num_agents)
        ]

    def _set_active_num_agents(self, orbit_num_agents_by_env: torch.Tensor) -> None:
        assert isinstance(orbit_num_agents_by_env, torch.Tensor)
        v = orbit_num_agents_by_env.reshape(-1)
        assert tuple(v.shape) == (len(self._active_num_agents),), (
            tuple(v.shape),
            len(self._active_num_agents),
        )
        for i in range(len(self._active_num_agents)):
            na = int(v[i].item())
            assert na in (2, 4), na
            self._active_num_agents[i] = na
        self.envs = self._active_envs()

    def reset(
        self,
        *,
        orbit_num_agents_by_env: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if orbit_num_agents_by_env is not None:
            self._set_active_num_agents(orbit_num_agents_by_env)
        outs = [e.reset(**kwargs) for e in self.envs]
        assert all(isinstance(o, dict) for o in outs)
        stacked = _stack_remapped_env_outputs(outs)
        maybe_validate_dict_io_contract(self._flags, stacked, "env_batch_wrapper_output")
        return stacked

    def step(
        self,
        actions: torch.Tensor,
        *,
        tape_baseline_learn: torch.Tensor | None = None,
        tape_supervised_pred: dict[str, Any] | None = None,
        tape_supervised_target: dict[str, torch.Tensor] | None = None,
        tape_supervised_valid: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        w = self._wall_span
        with w("env_step_asserts"):
            assert isinstance(actions, torch.Tensor)
            assert actions.ndim == 4
            e = int(actions.shape[0])
            p_axis = int(actions.shape[1])
            assert e == len(self.envs)
            assert int(actions.shape[2]) == int(ORBIT_PLANET_ACTION_SLOTS)
            assert int(actions.shape[3]) == 1
            assert p_axis == ORBIT_PLAYER_AXIS_SLOTS, (p_axis, ORBIT_PLAYER_AXIS_SLOTS)
        outs: list[dict[str, Any]] = []
        for ei in range(e):
            with w("env_step_row_prepare"):
                na = int(self._active_num_agents[ei])
                assert na in (2, 4), na
                active_slots = orbit_active_policy_slots(na)
                seats_classes: list[torch.Tensor] = []
                for slot in active_slots:
                    cls_cpu = actions[ei, slot, :, 0].detach().cpu().to(dtype=torch.int64).contiguous()
                    assert tuple(cls_cpu.shape) == (ORBIT_PLANET_ACTION_SLOTS,)
                    seats_classes.append(cls_cpu)
                step_in: dict[str, Any] = {
                    ORBIT_STEP_KEY_ORBIT_PAIRWISE_CLASSES: seats_classes,
                }
                if tape_baseline_learn is not None:
                    row = (
                        tape_baseline_learn[ei, list(active_slots)]
                        .detach()
                        .cpu()
                        .to(dtype=torch.float32)
                        .contiguous()
                    )
                    assert tuple(row.shape) == (na,)
                    step_in[ORBIT_STEP_KEY_TAPE_BASELINE_LEARN] = row
                if (
                    tape_supervised_pred is not None
                    or tape_supervised_target is not None
                    or tape_supervised_valid is not None
                ):
                    assert (
                        isinstance(tape_supervised_pred, dict)
                        and isinstance(tape_supervised_target, dict)
                        and isinstance(tape_supervised_valid, dict)
                    ), (
                        "tape supervised payload requires prediction/target/valid dicts",
                        type(tape_supervised_pred),
                        type(tape_supervised_target),
                        type(tape_supervised_valid),
                    )
                    pred_row_by_head: dict[str, Any] = {}
                    tgt_row_by_head: dict[str, torch.Tensor] = {}
                    valid_row_by_head: dict[str, torch.Tensor] = {}
                    head_names = sorted(
                        set(tape_supervised_pred.keys())
                        | set(tape_supervised_target.keys())
                        | set(tape_supervised_valid.keys())
                    )
                    for head_name in head_names:
                        assert (
                            head_name in tape_supervised_pred
                            and head_name in tape_supervised_target
                            and head_name in tape_supervised_valid
                        ), (
                            "supervised tape head must exist in prediction/target/valid dicts",
                            head_name,
                            sorted(tape_supervised_pred.keys()),
                            sorted(tape_supervised_target.keys()),
                            sorted(tape_supervised_valid.keys()),
                        )
                        pred_full = tape_supervised_pred[head_name]
                        tgt_full = tape_supervised_target[head_name]
                        valid_full = tape_supervised_valid[head_name]
                        assert (
                            isinstance(tgt_full, torch.Tensor)
                            and isinstance(valid_full, torch.Tensor)
                        ), (
                            head_name,
                            type(pred_full),
                            type(tgt_full),
                            type(valid_full),
                        )
                        if isinstance(pred_full, torch.Tensor):
                            pred_row = (
                                pred_full[ei, list(active_slots), :]
                                .detach()
                                .cpu()
                                .to(dtype=torch.float32)
                                .contiguous()
                            )
                        else:
                            assert isinstance(pred_full, list), (
                                head_name,
                                type(pred_full),
                            )
                            assert len(pred_full) == e, (head_name, len(pred_full), e)
                            pred_env = pred_full[ei]
                            assert isinstance(pred_env, list), (head_name, ei, type(pred_env))
                            assert len(pred_env) == p_axis, (
                                head_name,
                                len(pred_env),
                                p_axis,
                            )
                            pred_row = [pred_env[slot] for slot in active_slots]
                        tgt_row = (
                            tgt_full[ei, list(active_slots), :]
                            .detach()
                            .cpu()
                            .to(dtype=torch.float32)
                            .contiguous()
                        )
                        valid_row = (
                            valid_full[ei, list(active_slots), :]
                            .detach()
                            .cpu()
                            .to(dtype=torch.bool)
                            .contiguous()
                        )
                        if isinstance(pred_row, torch.Tensor):
                            assert tuple(pred_row.shape) == (na, ORBIT_MAX_PLANETS), (
                                head_name,
                                tuple(pred_row.shape),
                                na,
                                ORBIT_MAX_PLANETS,
                            )
                        else:
                            assert len(pred_row) == na, (head_name, len(pred_row), na)
                            for row in pred_row:
                                assert isinstance(row, list), (head_name, type(row))
                                assert len(row) == ORBIT_MAX_PLANETS, (
                                    head_name,
                                    len(row),
                                    ORBIT_MAX_PLANETS,
                                )
                        assert tuple(tgt_row.shape) == (na, ORBIT_MAX_PLANETS), (
                            head_name,
                            tuple(tgt_row.shape),
                            na,
                            ORBIT_MAX_PLANETS,
                        )
                        assert tuple(valid_row.shape) == (na, ORBIT_MAX_PLANETS), (
                            head_name,
                            tuple(valid_row.shape),
                            na,
                            ORBIT_MAX_PLANETS,
                        )
                        pred_row_by_head[str(head_name)] = pred_row
                        tgt_row_by_head[str(head_name)] = tgt_row
                        valid_row_by_head[str(head_name)] = valid_row
                    step_in[ORBIT_STEP_KEY_TAPE_SUPERVISED_PRED] = pred_row_by_head
                    step_in[ORBIT_STEP_KEY_TAPE_SUPERVISED_TARGET] = tgt_row_by_head
                    step_in[ORBIT_STEP_KEY_TAPE_SUPERVISED_VALID] = valid_row_by_head
            with w("env_step_inner_forward"):
                outs.append(self.envs[ei].step(step_in))
        with w("env_step_stack_batch"):
            assert all(isinstance(o, dict) for o in outs)
            stacked = _stack_remapped_env_outputs(outs)
        with w("env_step_dict_contract"):
            maybe_validate_dict_io_contract(self._flags, stacked, "env_batch_wrapper_output")
        return stacked

    def reset_where_done(
        self,
        env_after_step: dict[str, Any],
        done: torch.Tensor,
        *,
        orbit_num_agents_by_env: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        assert isinstance(env_after_step, dict)
        e = len(self._active_num_agents)
        assert isinstance(done, torch.Tensor) and done.dtype == torch.bool
        d = done.reshape(-1)
        assert tuple(d.shape) == (e,), (tuple(done.shape), e)
        if not bool(d.any().item()):
            return env_after_step
        selected_num_agents = None
        if orbit_num_agents_by_env is not None:
            selected_num_agents = orbit_num_agents_by_env.reshape(-1)
            assert tuple(selected_num_agents.shape) == (e,), (
                tuple(selected_num_agents.shape),
                e,
            )

        def row(t: Any, i: int) -> Any:
            if isinstance(t, dict):
                return {k: row(v, i) for k, v in t.items()}
            assert isinstance(t, torch.Tensor)
            return t[i].clone()

        outs = []
        for i in range(e):
            if bool(d[i].item()):
                if selected_num_agents is not None:
                    na = int(selected_num_agents[i].item())
                    assert na in (2, 4), na
                    self._active_num_agents[i] = na
                env_i = self._envs_by_num_agents[int(self._active_num_agents[i])][i]
                outs.append(env_i.reset())
            else:
                outs.append(row(env_after_step, i))
        self.envs = self._active_envs()
        stacked = _stack_remapped_env_outputs(outs)
        maybe_validate_dict_io_contract(self._flags, stacked, "env_batch_wrapper_output")
        return stacked
