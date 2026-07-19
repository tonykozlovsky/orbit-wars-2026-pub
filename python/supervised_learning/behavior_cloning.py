"""Behavior cloning for the single spawn_fleet action space.

Episode ``.pt`` tensors from ``orbit_kaggle_replay_bc_dataset.write_behavior_clone_episode_pt`` are
gzip-wrapped ``torch.save`` payloads (see ``load_behavior_clone_episode_pt``). Tensors use ``rl_per_planet_move_class`` (shape ``[T, seats, planets]``, values
``0 .. ORBIT_PER_PLANET_MOVE_CLASSES-1``).

BC trains the model directly against the flat ``spawn_fleet`` policy class with masked CE.
Episode sampling emits one sample per active ``bc_loss_player_mask & player_mask`` policy slot.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ["TORCH_MATMUL_PRECISION"] = "high"

import torch

torch.set_float32_matmul_precision("high")

import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from src.configs.base import ImpalaTrainingConfig
from src.configs.impala_x1_rl import build_training_config
from src.torchbeast.core.common import build_optimizer_from_config, compile_impala_model_for_rl
from src.torchbeast.core.losses import (
    compute_baseline_loss,
    compute_teacher_kl_from_log_probs,
    reduce,
)
from src.torchbeast.core.stats import RollingAverage
from src.configs.impala_orbit_model_hyperparams import (
    ORBIT_IMPALA_OBS_FEATURE_LAYOUT,
)
from src.gym.obs_wrapper import (
    ORBIT_EDGE_BASE_FEATURES,
    ORBIT_EDGE_FEATURES,
    ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_BASE_FEATURES,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
    ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_ENEMY_AXIS_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
)
from src.models.orbit_obs_feature_layout import planet_edge_physical_to_logical_from_layout
from src.models.orbit_obs_feature_input_contract import (
    ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES,
    orbit_obs_arrival_temporal_feature_importance_inputs,
    orbit_obs_edge_feature_importance_inputs,
    orbit_obs_edge_player_feature_importance_inputs,
    orbit_obs_planet_feature_importance_inputs,
    orbit_obs_planet_player_feature_importance_inputs,
)
from src.gym.orbit_kaggle_replay_bc_dataset import load_behavior_clone_episode_pt
from src.gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from src.models.models import ImpalaOrbitModel, impala_orbit_model_init_kwargs_from_flags

N_VALIDATION_EPISODES = 100
N_EPISODES = 32

VAL_EVERY_BATCHES = 1000
BC_PROFILE_EVERY_BATCHES = 100


_VALIDATION_EPISODES_FILENAME = "validation_episodes.txt"

_BC_TRAIN_WB_TAG = "BC_TRAIN"
_BC_VAL_WB_TAG = "BC_VAL"
_BC_TRAIN_METRIC_SMOOTH_WINDOW = 400
assert _BC_TRAIN_METRIC_SMOOTH_WINDOW >= 1, _BC_TRAIN_METRIC_SMOOTH_WINDOW

BC_SN_BALANCE_EPS = 1e-6
assert 0.0 < BC_SN_BALANCE_EPS < 0.5, BC_SN_BALANCE_EPS

_BC_GRAD_NORM_EMA_ALPHA = 1.0 / 1000.0
assert 0.0 < _BC_GRAD_NORM_EMA_ALPHA <= 1.0, _BC_GRAD_NORM_EMA_ALPHA


_BC_FINAL_TOPK_LEVELS: tuple[int, ...] = (1, 2, 5)
assert min(_BC_FINAL_TOPK_LEVELS) == 1, _BC_FINAL_TOPK_LEVELS
assert max(_BC_FINAL_TOPK_LEVELS) <= int(ORBIT_PER_PLANET_MOVE_CLASSES), _BC_FINAL_TOPK_LEVELS

_BC_LR_WARMUP_TRAIN_SAMPLES_DEFAULT = 200_000
_BC_MAX_TRAIN_SAMPLES_DEFAULT = 10_000_000
assert _BC_LR_WARMUP_TRAIN_SAMPLES_DEFAULT >= 1, _BC_LR_WARMUP_TRAIN_SAMPLES_DEFAULT
assert _BC_MAX_TRAIN_SAMPLES_DEFAULT > _BC_LR_WARMUP_TRAIN_SAMPLES_DEFAULT, (
    _BC_MAX_TRAIN_SAMPLES_DEFAULT,
    _BC_LR_WARMUP_TRAIN_SAMPLES_DEFAULT,
)


def _set_bc_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    lr_f = float(lr)
    assert math.isfinite(lr_f) and lr_f >= 0.0, lr_f
    for pg in opt.param_groups:
        pg["lr"] = lr_f


def _bc_drop_incompatible_optimizer_state(opt: torch.optim.Optimizer) -> int:
    n_dropped = 0
    for group in opt.param_groups:
        params = group["params"]
        assert isinstance(params, list), type(params)
        for param in params:
            assert isinstance(param, torch.nn.Parameter), type(param)
            state = opt.state.get(param)
            if state is None or len(state) == 0:
                continue
            assert isinstance(state, dict), type(state)
            drop = False
            for state_key, state_value in state.items():
                if not isinstance(state_value, torch.Tensor):
                    continue
                if state_key == "step":
                    assert state_value.ndim == 0, (state_key, tuple(state_value.shape))
                    continue
                if (
                    tuple(state_value.shape) != tuple(param.shape)
                    or state_value.dtype != param.dtype
                    or state_value.device != param.device
                    or state_value.layout != param.layout
                ):
                    drop = True
                    break
            if drop:
                opt.state[param] = {}
                n_dropped += 1
    return n_dropped


def _bc_lr_for_trailing_train_samples(
    train_samples_after_step: int,
    *,
    lr_max: float,
    warmup_samples: int,
    max_train_samples: int,
) -> float:
    assert warmup_samples >= 1, warmup_samples
    assert max_train_samples > warmup_samples, (max_train_samples, warmup_samples)
    s = int(train_samples_after_step)
    if s <= 0:
        return 0.0
    w = int(warmup_samples)
    m = int(max_train_samples)
    lm = float(lr_max)
    assert math.isfinite(lm) and lm > 0.0, lm
    if s < w:
        return lm * (float(s) / float(w))
    if s >= m:
        return 0.0
    return lm * (float(m - s) / float(m - w))


@dataclass(frozen=True)
class BcPolicyBatchTargets:
    batch_obs: dict[str, torch.Tensor]
    loss_mask: torch.Tensor
    move_tgt: torch.Tensor
    target_action_invalid_counts: torch.Tensor


@dataclass(frozen=True)
class BcPolicyBatchMetrics:
    combined_loss: torch.Tensor
    final_policy_loss: torch.Tensor
    final_policy_topk: dict[int, torch.Tensor]
    final_policy_entropy: torch.Tensor
    batch_size: int


def _wandb_float_dict_finite_only(payload: dict[str, float]) -> dict[str, float]:
    """Drop non-finite floats so W&B does not receive NaN/inf (e.g. precision when tp+fp==0 over an epoch)."""
    out: dict[str, float] = {}
    for k, v in payload.items():
        fv = float(v)
        if math.isfinite(fv):
            out[k] = fv
    return out


_OBS_LEARN_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
    "player_mask",
)

_OBS_EMBEDDING_FEATURE_KEY_BY_BASE: dict[str, str] = {
    "orbit_planet_features": "orbit_planet_embedding_features",
    "orbit_planet_arrival_features": "orbit_planet_arrival_embedding_features",
    "orbit_planet_pairwise_features": "orbit_planet_pairwise_embedding_features",
}
_OBS_EMBEDDING_FEATURE_KEYS: tuple[str, ...] = tuple(
    _OBS_EMBEDDING_FEATURE_KEY_BY_BASE.values()
)

_BC_SELF_PLAYER_BLOCK_INDEX = 0
assert int(ORBIT_ENEMY_AXIS_SLOTS) == int(ORBIT_PLAYER_AXIS_SLOTS) - 1, (
    ORBIT_ENEMY_AXIS_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
)

_BC_FEATURE_IMPORTANCE_SHUFFLE_SEED = 0
_BC_FEATURE_IMPORTANCE_VAL_EPISODES_DEFAULT = 10
assert _BC_FEATURE_IMPORTANCE_VAL_EPISODES_DEFAULT >= 1, (
    _BC_FEATURE_IMPORTANCE_VAL_EPISODES_DEFAULT,
)


def _bc_plain_config(obj: object) -> object:
    if isinstance(obj, SimpleNamespace):
        return {k: _bc_plain_config(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _bc_plain_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bc_plain_config(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bc_plain_config(v) for v in obj)
    return obj


def _bc_current_model_config() -> dict[str, Any]:
    flags = build_training_config()
    model_config = _bc_plain_config(flags.model)
    assert isinstance(model_config, dict), type(model_config)
    return model_config


def _bc_checkpoint_model_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    assert "model_config" in ckpt, sorted(ckpt.keys())
    model_config = ckpt["model_config"]
    assert isinstance(model_config, dict), type(model_config)
    return model_config


def bc_model_init_kwargs(model_config: dict[str, Any]) -> dict[str, object]:
    flags = build_training_config()
    flags = SimpleNamespace(**vars(flags))
    flags.model = model_config
    kw = impala_orbit_model_init_kwargs_from_flags(flags)
    kw["entropy_floor_target"] = (0.0,)
    kw["entropy_floor_max_temperature"] = 1.0
    kw["entropy_floor_num_iters"] = 0
    kw["include_rl_policy_value_heads"] = True
    kw["include_rl_value_head"] = False
    return kw


def bc_feature_importance_model_init_kwargs(
    model_config: dict[str, Any],
    *,
    include_rl_value_head: bool,
) -> dict[str, object]:
    kw = bc_model_init_kwargs(model_config)
    kw["include_rl_value_head"] = bool(include_rl_value_head)
    return kw


def _strip_torch_compile_orig_mod_prefix(sd: dict[Any, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    keys = tuple(str(k) for k in sd.keys())
    prefixed = tuple(k.startswith(prefix) for k in keys)
    assert all(prefixed) or not any(prefixed), keys[:8]
    if not any(prefixed):
        return {str(k): v for k, v in sd.items()}
    return {str(k)[len(prefix) :]: v for k, v in sd.items()}


_BC_ALLOWED_CHECKPOINT_ONLY_STATE_DICT_PREFIXES = (
    "_global_value_head.",
    "_global_value_head_production_delta.",
)


def _load_bc_resume_model_state_dict(model: torch.nn.Module, sd: dict[str, Any]) -> None:
    model_sd = model.state_dict()
    checkpoint_only = sorted(str(k) for k in sd if str(k) not in model_sd)
    unexpected_checkpoint_only = sorted(
        k
        for k in checkpoint_only
        if not k.startswith(_BC_ALLOWED_CHECKPOINT_ONLY_STATE_DICT_PREFIXES)
    )
    assert not unexpected_checkpoint_only, unexpected_checkpoint_only
    load_sd = {str(k): v for k, v in sd.items() if str(k) in model_sd}
    missing = sorted(str(k) for k in model_sd if str(k) not in load_sd)
    assert not missing, missing
    for k, v in load_sd.items():
        assert isinstance(v, torch.Tensor), (k, type(v))
        assert tuple(v.shape) == tuple(model_sd[k].shape), (
            k,
            tuple(v.shape),
            tuple(model_sd[k].shape),
        )
    model.load_state_dict(load_sd, strict=True)


def _list_episode_pt_files(data_dir: Path) -> list[Path]:
    assert data_dir.is_dir(), data_dir
    out = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix == ".pt")
    assert len(out) >= 1, (data_dir, "expected at least one .pt episode")
    return out


def _train_val_episode_paths(
    data_dir: Path,
    *,
    regen_val: bool,
) -> tuple[list[Path], list[Path]]:
    """Train vs validation split from ``validation_episodes.txt`` under ``data_dir``.

    The file lists one episode ``*.pt`` basename per line. If missing or fewer than
    ``N_VALIDATION_EPISODES`` valid lines, a new file would be written by sampling
    ``N_VALIDATION_EPISODES`` episodes uniformly without replacement; that write
    requires ``regen_val`` (CLI ``--regen-val``) so the validation holdout is not
    silently replaced.
    """
    all_paths = _list_episode_pt_files(data_dir)
    n_val = int(N_VALIDATION_EPISODES)
    assert n_val >= 1, n_val
    assert len(all_paths) > n_val, (
        data_dir,
        len(all_paths),
        n_val,
        "need more .pt episodes than N_VALIDATION_EPISODES to hold out a val set",
    )
    by_key: dict[str, Path] = {p.name: p for p in all_paths}
    assert len(by_key) == len(all_paths), "duplicate episode basenames under data_dir"
    val_path = data_dir / _VALIDATION_EPISODES_FILENAME
    uniq_names: list[str] = []
    seen: set[str] = set()
    if val_path.is_file():
        text = val_path.read_text(encoding="utf-8")
        for raw in text.splitlines():
            name = raw.strip()
            if not name:
                continue
            assert "/" not in name and "\\" not in name, (val_path, raw)
            assert name.endswith(".pt"), (val_path, raw)
            if name not in by_key:
                continue
            if name not in seen:
                seen.add(name)
                uniq_names.append(name)
    need_write = (not val_path.is_file()) or (len(uniq_names) < n_val)
    if need_write and not regen_val:
        raise SystemExit(
            f"refusing to write {val_path}: need an existing file with at least "
            f"{n_val} valid episode basenames, or pass --regen-val to (re)sample and write it"
        )
    if need_write:
        val_sel = random.sample(all_paths, n_val)
        val_path.write_text(
            "\n".join(sorted(p.name for p in val_sel)) + "\n",
            encoding="utf-8",
        )
        val_paths = sorted(val_sel)
    else:
        val_paths = [by_key[name] for name in sorted(uniq_names)[:n_val]]
    val_set = set(val_paths)
    assert len(val_set) == n_val, (len(val_set), n_val, val_path)
    train_paths = [p for p in all_paths if p not in val_set]
    assert len(train_paths) >= 1, (data_dir, len(all_paths), n_val)
    return train_paths, val_paths


def _assert_episode_contract(ep: dict[str, object]) -> None:
    assert int(ep["num_bc_timesteps"]) >= 1
    na = int(ep["num_agents"])
    assert na in (2, 4), na
    n = ORBIT_MAX_PLANETS
    pax = 1
    t = int(ep["num_bc_timesteps"])
    assert int(ep["hit_horizon"]) >= 1, int(ep["hit_horizon"])
    bc_policy_slot = ep["bc_policy_slot"]
    assert isinstance(bc_policy_slot, torch.Tensor), type(bc_policy_slot)
    assert bc_policy_slot.shape == (t,), bc_policy_slot.shape
    active_slots = orbit_active_policy_slots(na)
    assert torch.all((0 <= bc_policy_slot) & (bc_policy_slot < ORBIT_PLAYER_AXIS_SLOTS)), (
        bc_policy_slot.min(),
        bc_policy_slot.max(),
    )
    assert all(int(slot) in active_slots for slot in bc_policy_slot.tolist()), (active_slots, bc_policy_slot)
    rl_gt = ep["rl_per_planet_move_class"]
    assert isinstance(rl_gt, torch.Tensor), type(rl_gt)
    assert rl_gt.shape == (t, pax, n), rl_gt.shape
    assert rl_gt.dtype in (torch.int64, torch.int32), rl_gt.dtype
    assert torch.all((0 <= rl_gt) & (rl_gt < ORBIT_PER_PLANET_MOVE_CLASSES)), (
        rl_gt.min(),
        rl_gt.max(),
        ORBIT_PER_PLANET_MOVE_CLASSES,
    )
    bc_lp = ep["bc_loss_player_mask"]
    assert isinstance(bc_lp, torch.Tensor), type(bc_lp)
    assert bc_lp.shape == (t, pax), bc_lp.shape
    assert torch.is_floating_point(bc_lp), bc_lp.dtype
    assert torch.all((bc_lp >= 0.0) & (bc_lp <= 1.0)), (bc_lp.min(), bc_lp.max())
    assert torch.all(bc_lp == 1.0), bc_lp
    bc_lsrc = ep["bc_loss_source_mask"]
    assert isinstance(bc_lsrc, torch.Tensor), type(bc_lsrc)
    assert bc_lsrc.shape == (t, pax, n), bc_lsrc.shape
    assert torch.is_floating_point(bc_lsrc), bc_lsrc.dtype
    assert torch.all((bc_lsrc == 0.0) | (bc_lsrc == 1.0)), (
        bc_lsrc.min(),
        bc_lsrc.max(),
    )
    for k in _OBS_LEARN_KEYS:
        assert k in ep, (k, sorted(ep.keys()))
        ten = ep[k]
        assert isinstance(ten, torch.Tensor), (k, type(ten))
        assert ten.shape[0] == t, (k, tuple(ten.shape), t)
        assert ten.shape[1] == pax, (k, tuple(ten.shape), pax)
    action_mask = ep["available_action_mask"]
    assert isinstance(action_mask, torch.Tensor), type(action_mask)
    assert action_mask.shape == (t, pax, n, ORBIT_PER_PLANET_MOVE_CLASSES), (
        action_mask.shape,
        (t, pax, n, ORBIT_PER_PLANET_MOVE_CLASSES),
    )
    assert action_mask.dtype == torch.int8, action_mask.dtype
    assert torch.all(action_mask >= 0), (action_mask.min(), action_mask.max())
    assert torch.all(action_mask <= 1), action_mask.max()
    pm = ep["player_mask"]
    assert isinstance(pm, torch.Tensor), type(pm)
    assert pm.shape == (t, pax), pm.shape
    assert torch.all(pm > 0.5), pm


def _bc_step_dict_from_episode(
    ep: dict[str, object],
    step: int,
    policy_slot: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    slot = int(policy_slot)
    assert slot == 0, slot
    assert slot in _bc_training_policy_slots_for_step(ep, step), (step, slot)
    for k in _OBS_LEARN_KEYS:
        ten = ep[k]
        assert isinstance(ten, torch.Tensor)
        out[k] = ten[step, slot : slot + 1].detach()
    out["rl_per_planet_move_class"] = (
        ep["rl_per_planet_move_class"][step, slot : slot + 1].detach().to(dtype=torch.int64)
    )
    out["bc_loss_source_mask"] = (
        ep["bc_loss_source_mask"][step, slot : slot + 1].detach().to(dtype=torch.float32)
    )
    return out


def _bc_training_policy_slots_for_step(ep: dict[str, object], step: int) -> tuple[int, ...]:
    st = int(step)
    t = int(ep["num_bc_timesteps"])
    assert 0 <= st < t, (st, t)
    player_mask = ep["player_mask"]
    bc_loss_player_mask = ep["bc_loss_player_mask"]
    assert isinstance(player_mask, torch.Tensor), type(player_mask)
    assert isinstance(bc_loss_player_mask, torch.Tensor), type(bc_loss_player_mask)
    pax = 1
    assert player_mask.shape == (t, pax), player_mask.shape
    assert bc_loss_player_mask.shape == (t, pax), bc_loss_player_mask.shape
    assert torch.is_floating_point(player_mask), player_mask.dtype
    assert torch.is_floating_point(bc_loss_player_mask), bc_loss_player_mask.dtype
    pm = player_mask[st].detach().to(dtype=torch.float32)
    bc = bc_loss_player_mask[st].detach().to(dtype=torch.float32)
    assert torch.all((pm >= 0.0) & (pm <= 1.0)), (pm.min(), pm.max())
    assert torch.all((bc >= 0.0) & (bc <= 1.0)), (bc.min(), bc.max())
    assert torch.all(pm > 0.5), pm
    assert torch.all(bc == 1.0), bc
    return (0,)


def _bc_training_items(ep: dict[str, object]) -> tuple[tuple[int, int], ...]:
    t = int(ep["num_bc_timesteps"])
    assert t >= 1, t
    keep: list[tuple[int, int]] = []
    for step in range(t):
        for policy_slot in _bc_training_policy_slots_for_step(ep, step):
            keep.append((step, policy_slot))
    return tuple(keep)


@dataclass(frozen=True)
class BcValidationEpisodeLoadTask:
    index: int
    path: Path
    count: int


def _bc_load_validation_episode_samples(
    task: BcValidationEpisodeLoadTask,
) -> tuple[int, tuple[dict[str, torch.Tensor], ...]]:
    idx = int(task.index)
    count = int(task.count)
    assert 0 <= idx < count, (idx, count)
    print(
        f"BC val worker pid={os.getpid()}: loading {task.path.name} ({idx + 1}/{count})",
        flush=True,
    )
    ep = BcEpisodePoolIterable._load_episode_pt(task.path)
    samples = tuple(
        _bc_step_dict_from_episode(ep, step, policy_slot)
        for step, policy_slot in _bc_training_items(ep)
    )
    return idx, samples


def _bc_precompute_validation_batches_parallel(
    val_paths: list[Path],
    *,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    bs = int(batch_size)
    assert bs >= 1, bs
    n_paths = len(val_paths)
    assert n_paths >= 1, n_paths
    nw = int(workers)
    assert 1 <= nw <= n_paths, (nw, n_paths)
    tasks = tuple(
        BcValidationEpisodeLoadTask(index=i, path=path, count=n_paths)
        for i, path in enumerate(sorted(val_paths, key=lambda p: p.name))
    )
    batches: list[dict[str, torch.Tensor]] = []
    buf: list[dict[str, torch.Tensor]] = []
    pending: dict[int, tuple[dict[str, torch.Tensor], ...]] = {}
    next_index = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nw) as pool:
        for idx, samples in pool.imap_unordered(
            _bc_load_validation_episode_samples,
            tasks,
            chunksize=1,
        ):
            pending[int(idx)] = samples
            while next_index in pending:
                episode_samples = pending.pop(next_index)
                for sample in episode_samples:
                    buf.append(sample)
                    if len(buf) == bs:
                        batches.append(_collate_bc_batch(buf))
                        buf.clear()
                next_index += 1
    assert next_index == n_paths, (next_index, n_paths, sorted(pending.keys()))
    return tuple(batches)


class BcEpisodePoolIterable(IterableDataset[dict[str, torch.Tensor]]):
    """Samples timesteps from a resident episode pool; RAM holds at most ``pool_episodes`` episodes.

    ``deterministic=True`` is only valid with ``infinite=False`` (validation): episodes are visited in
    sorted basename order, timesteps in ascending step index; only batches of exactly ``batch_size`` are
    yielded (tail shorter than ``batch_size`` is dropped). ``pool_episodes`` is ignored in that mode.
    """

    def __init__(
        self,
        episode_paths: list[Path],
        *,
        pool_episodes: int,
        batch_size: int,
        infinite: bool,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self._episode_paths = list(episode_paths)
        assert len(self._episode_paths) >= 1, self._episode_paths
        by_name = {p.name: p for p in self._episode_paths}
        assert len(by_name) == len(self._episode_paths), "duplicate episode basenames"
        pe = int(pool_episodes)
        assert pe >= 1, pe
        self._pool_n = min(pe, len(self._episode_paths))
        self._bs = int(batch_size)
        assert self._bs >= 1, self._bs
        self._infinite = bool(infinite)
        self._deterministic = bool(deterministic)
        assert not (self._deterministic and self._infinite), (
            "BcEpisodePoolIterable: deterministic=True requires infinite=False",
        )

    @staticmethod
    def _load_episode_pt(path: Path) -> dict[str, object]:
        assert path.is_file(), path
        ep = load_behavior_clone_episode_pt(path)
        _assert_episode_contract(ep)
        return ep

    def _iter_deterministic_finite(self) -> Iterator[dict[str, torch.Tensor]]:
        paths = sorted(self._episode_paths, key=lambda p: p.name)
        buf: list[dict[str, torch.Tensor]] = []
        n_paths = len(paths)
        for i, path in enumerate(paths, start=1):
            print(f"BC val: loading {path.name} ({i}/{n_paths})", flush=True)
            ep = self._load_episode_pt(path)
            for step, policy_slot in _bc_training_items(ep):
                buf.append(_bc_step_dict_from_episode(ep, step, policy_slot))
                if len(buf) == self._bs:
                    yield _collate_bc_batch(buf)
                    buf.clear()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is None:
            episode_paths = self._episode_paths
            pool_n = self._pool_n
        else:
            episode_paths = self._episode_paths[worker_info.id :: worker_info.num_workers]
            assert len(episode_paths) >= 1, (
                worker_info.id,
                worker_info.num_workers,
                len(self._episode_paths),
            )
            pool_n = self._pool_n // worker_info.num_workers
            if worker_info.id < self._pool_n % worker_info.num_workers:
                pool_n += 1
            assert pool_n >= 1, (worker_info.id, worker_info.num_workers, self._pool_n)
        if self._deterministic:
            assert worker_info is None, "deterministic validation iterator is single-process"
            yield from self._iter_deterministic_finite()
            return

        rng = random.Random()
        pool: dict[str, tuple[dict[str, object], set[tuple[int, int]]]] = {}
        empty_episode_names: set[str] = set()
        pending = list(episode_paths)
        rng.shuffle(pending)

        def load_into_pool(p: Path) -> None:
            assert p.name not in pool, p.name
            assert p.name not in empty_episode_names, p.name
            ep = self._load_episode_pt(p)
            training_items = _bc_training_items(ep)
            if len(training_items) == 0:
                empty_episode_names.add(p.name)
                return
            pool[p.name] = (ep, set(training_items))

        def fill_pool() -> None:
            while len(pool) < pool_n:
                if self._infinite:
                    candidates = [
                        p
                        for p in episode_paths
                        if p.name not in pool and p.name not in empty_episode_names
                    ]
                    if len(candidates) == 0:
                        assert len(pool) >= 1, "BC pool: no episodes with active bc_loss_player_mask & player_mask"
                        break
                    load_into_pool(rng.choice(candidates))
                else:
                    if len(pending) == 0:
                        break
                    load_into_pool(pending.pop())

        fill_pool()

        def one_batch(n_take: int) -> dict[str, torch.Tensor]:
            assert n_take >= 1, n_take
            samples: list[dict[str, torch.Tensor]] = []
            for _ in range(n_take):
                names_active = [n for n, (_, r) in pool.items() if len(r) > 0]
                assert len(names_active) >= 1, "BC pool: no remaining steps"
                weights = [len(pool[n][1]) for n in names_active]
                name = rng.choices(names_active, weights=weights, k=1)[0]
                ep, rem_items = pool[name]
                step, policy_slot = rng.choice(tuple(rem_items))
                rem_items.remove((step, policy_slot))
                samples.append(_bc_step_dict_from_episode(ep, step, policy_slot))
                if len(rem_items) == 0:
                    del pool[name]
                    del ep
                    fill_pool()
            assert len(samples) == n_take
            return _collate_bc_batch(samples)

        if self._infinite:
            while True:
                yield one_batch(self._bs)
        else:
            while len(pool) > 0:
                total = sum(len(rem) for _, rem in pool.values())
                if total < self._bs:
                    break
                yield one_batch(self._bs)


def _collate_bc_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    assert len(samples) >= 1
    keys = list(samples[0].keys())
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out


def _batch_obs_to_device(batch_cpu: dict[str, torch.Tensor], *, device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in _OBS_LEARN_KEYS:
        if k == "available_action_mask":
            assert batch_cpu[k].dtype == torch.int8, batch_cpu[k].dtype
            dtype = torch.int8
        else:
            dtype = torch.float32
        out[k] = batch_cpu[k].to(device=device, dtype=dtype, non_blocking=True)
    embedding_key_present = tuple(k in batch_cpu for k in _OBS_EMBEDDING_FEATURE_KEYS)
    if any(embedding_key_present):
        assert all(embedding_key_present), (
            "embedding feature inputs must be provided as a complete planet/arrival/edge set",
            _OBS_EMBEDDING_FEATURE_KEYS,
            sorted(batch_cpu.keys()),
        )
        for k in _OBS_EMBEDDING_FEATURE_KEYS:
            out[k] = batch_cpu[k].to(device=device, dtype=torch.float32, non_blocking=True)
    return out


def compute_bc_rl_move_loss_mask(
    batch_obs: dict[str, torch.Tensor],
    rl_tgt: torch.Tensor,
    bc_loss_source_mask: torch.Tensor,
    *,
    n_planets: int,
    n_policy_seats: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[batch, policy_seat, source_planet]``: include this RL move slot in BC loss and metrics.

    The iterable dataset has already sliced each sample to one BC-active policy slot.
    """
    assert rl_tgt.ndim == 3, rl_tgt.shape
    b, p, n = rl_tgt.shape
    assert n == int(n_planets), (n, n_planets)
    assert p == int(n_policy_seats), (p, n_policy_seats)
    assert batch_obs["orbit_planet_features"].shape[:3] == (b, p, n), (
        batch_obs["orbit_planet_features"].shape,
        (b, p, n),
    )
    assert bc_loss_source_mask.shape == (b, p, n), (
        bc_loss_source_mask.shape,
        (b, p, n),
    )
    assert torch.is_floating_point(bc_loss_source_mask), bc_loss_source_mask.dtype
    source_supervised = bc_loss_source_mask > 0.5
    planet_mask = batch_obs["orbit_planet_mask"]
    player_mask = batch_obs["player_mask"]
    available_action_mask = batch_obs["available_action_mask"]
    assert planet_mask.shape == (b, p, n), (planet_mask.shape, (b, p, n))
    assert player_mask.shape == (b, p), (player_mask.shape, (b, p))
    assert available_action_mask.ndim == 4, available_action_mask.shape
    assert available_action_mask.shape[:3] == (b, p, n), (
        available_action_mask.shape,
        (b, p, n),
    )
    assert available_action_mask.shape[-1] == int(ORBIT_PER_PLANET_MOVE_CLASSES), (
        available_action_mask.shape,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    )
    assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
    available = available_action_mask > 0
    assert torch.all((0 <= rl_tgt) & (rl_tgt < available_action_mask.shape[-1])), (
        rl_tgt.min(),
        rl_tgt.max(),
        available_action_mask.shape[-1],
    )

    nb = int(available_action_mask.shape[-1]) // n
    assert nb * n == int(available_action_mask.shape[-1]), (nb, n, available_action_mask.shape[-1])
    assert nb == int(ORBIT_MOVE_CLASSES_PER_TARGET), (nb, ORBIT_MOVE_CLASSES_PER_TARGET)
    available_by_dst_sn = available.reshape(b, p, n, n, nb)
    assert available_by_dst_sn.shape == (b, p, n, n, nb), available_by_dst_sn.shape
    policy_seat_has_planets = (planet_mask > 0.5).any(dim=-1)
    assert policy_seat_has_planets.shape == (b, p), policy_seat_has_planets.shape
    src_slot = torch.arange(n, device=rl_tgt.device, dtype=torch.long).view(1, 1, n)
    src_idx = torch.arange(n, device=rl_tgt.device, dtype=torch.long)
    self_noop_available = available_by_dst_sn[:, :, src_idx, src_idx, 0]
    assert self_noop_available.shape == (b, p, n), self_noop_available.shape
    expected_self_noop_available = policy_seat_has_planets.unsqueeze(-1).expand(b, p, n)
    assert torch.all(self_noop_available == expected_self_noop_available), (
        "occupied policy seats must expose source-self noop; empty policy slots must stay zero",
    )
    sn0_available = available_by_dst_sn[..., 0]
    expected_sn0 = torch.eye(n, device=rl_tgt.device, dtype=torch.bool).view(1, 1, n, n).expand(
        b,
        p,
        n,
        n,
    ) & policy_seat_has_planets.view(b, p, 1, 1)
    assert torch.all(sn0_available == expected_sn0), (
        "ship bucket 0 is reserved for source-self noop on occupied policy seats",
    )
    dst_slot = rl_tgt // nb
    ship_subindex = rl_tgt % nb
    self_send_non_noop = (dst_slot == src_slot) & (ship_subindex > 0)

    target_idx = rl_tgt.unsqueeze(-1)
    target_available = (
        torch.gather(available, dim=-1, index=target_idx).squeeze(-1)
    )
    target_action_invalid = ~target_available
    target_action_invalid_for_log = target_action_invalid & (~self_send_non_noop)

    # Seat gate: dataset filters (winner/team/non-empty seat) are folded into player_mask at batch sampling.
    player_active = player_mask > 0.5
    seat_active_for_loss = player_active

    # Source gate: only real source planets with at least one non-noop action should train a policy slot.
    source_planet_valid = planet_mask > 0.5
    has_choice = available.sum(dim=-1) > 1
    seat_active_bpn = seat_active_for_loss.unsqueeze(-1)

    # Target gate: replay target must be an action the RL policy can legally choose in this state.
    target_valid = target_available

    loss_mask = seat_active_bpn & source_planet_valid & has_choice & target_valid & source_supervised
    target_action_invalid_count = (
        target_action_invalid_for_log & seat_active_bpn & source_supervised
    ).sum()
    target_action_invalid_active_choice_count = (
        target_action_invalid_for_log
        & seat_active_bpn
        & source_planet_valid
        & has_choice
        & source_supervised
    ).sum()
    target_action_invalid_src_invalid_planet_count = (
        target_action_invalid_for_log
        & seat_active_bpn
        & (~source_planet_valid)
        & source_supervised
    ).sum()
    target_action_invalid_no_choice_count = (
        target_action_invalid_for_log
        & seat_active_bpn
        & source_planet_valid
        & (~has_choice)
        & source_supervised
    ).sum()
    target_self_send_non_noop_active_choice_count = (
        self_send_non_noop & seat_active_bpn & source_planet_valid & has_choice & source_supervised
    ).sum()
    invalid_counts = torch.stack(
        [
            target_action_invalid_count,
            target_action_invalid_active_choice_count,
            target_action_invalid_src_invalid_planet_count,
            target_action_invalid_no_choice_count,
            target_self_send_non_noop_active_choice_count,
        ]
    )
    return loss_mask, invalid_counts


def bc_valid_loss_mask_class_balance_vectors(
    target: torch.Tensor,
    loss_mask_bpn: torch.Tensor,
    *,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nc = int(n_classes)
    assert nc >= 2, nc
    assert target.shape == loss_mask_bpn.shape, (target.shape, loss_mask_bpn.shape)
    assert loss_mask_bpn.dtype == torch.bool, loss_mask_bpn.dtype
    valid_target = target[loss_mask_bpn]
    n_valid = int(valid_target.numel())
    if n_valid > 0:
        assert torch.all((0 <= valid_target) & (valid_target < nc)), (
            valid_target.min(),
            valid_target.max(),
            nc,
        )
        numerator = torch.bincount(valid_target.reshape(-1), minlength=nc).to(dtype=torch.float32)
    else:
        numerator = torch.zeros(nc, dtype=torch.float32, device=target.device)
    assert numerator.shape == (nc,), (numerator.shape, nc)
    denominator = torch.full((nc,), float(n_valid), dtype=torch.float32, device=target.device)
    return numerator, denominator


def assert_masked_targets_available(
    target: torch.Tensor,
    loss_mask_bpn: torch.Tensor,
    class_mask: torch.Tensor,
    *,
    n_classes: int,
) -> None:
    nc = int(n_classes)
    assert target.shape == loss_mask_bpn.shape, (target.shape, loss_mask_bpn.shape)
    assert class_mask.shape == (*target.shape, nc), (class_mask.shape, target.shape, nc)
    assert loss_mask_bpn.dtype == torch.bool, loss_mask_bpn.dtype
    assert class_mask.dtype == torch.bool, class_mask.dtype
    assert torch.all((0 <= target) & (target < nc)), (target.min(), target.max(), nc)
    target_available = torch.gather(class_mask, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    assert target_available.shape == target.shape, (target_available.shape, target.shape)
    assert torch.all(target_available[loss_mask_bpn]), (
        "supervised targets under loss mask must be available in the matching action mask",
    )


def masked_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask_bpn: torch.Tensor,
    class_mask: torch.Tensor,
    *,
    n_classes: int,
) -> torch.Tensor:
    nc = int(n_classes)
    assert logits.shape == (*target.shape, nc), (logits.shape, target.shape, nc)
    assert loss_mask_bpn.shape == target.shape, (loss_mask_bpn.shape, target.shape)
    assert loss_mask_bpn.dtype == torch.bool, loss_mask_bpn.dtype
    assert class_mask.shape == logits.shape, (class_mask.shape, logits.shape)
    assert class_mask.dtype == torch.bool, class_mask.dtype
    assert torch.all((0 <= target) & (target < nc)), (target.min(), target.max(), nc)
    assert_masked_targets_available(
        target,
        loss_mask_bpn,
        class_mask,
        n_classes=nc,
    )
    neg_large = torch.finfo(logits.dtype).min / 16.0
    logits_for_ce = logits.masked_fill(~class_mask, neg_large)
    ce_flat = F.cross_entropy(
        logits_for_ce.reshape(-1, nc),
        target.reshape(-1),
        reduction="none",
    )
    ce = ce_flat.view_as(target)
    w = loss_mask_bpn.to(dtype=ce.dtype)
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def masked_policy_topk_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask_bpn: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    n_classes: int,
    k: int,
) -> torch.Tensor:
    nc = int(n_classes)
    kk = int(k)
    assert 1 <= kk <= nc, (kk, nc)
    assert logits.shape == (*target.shape, nc), (logits.shape, target.shape, nc)
    assert loss_mask_bpn.shape == target.shape, (loss_mask_bpn.shape, target.shape)
    assert action_mask.shape == logits.shape, (action_mask.shape, logits.shape)
    assert action_mask.dtype == torch.bool, action_mask.dtype
    assert_masked_targets_available(
        target,
        loss_mask_bpn,
        action_mask,
        n_classes=nc,
    )
    neg_large = torch.finfo(logits.dtype).min / 16.0
    logits_for_topk = logits.masked_fill(~action_mask, neg_large)
    _, topk_idx = torch.topk(logits_for_topk, kk, dim=-1)
    hits = (topk_idx == target.unsqueeze(-1)).any(dim=-1) & loss_mask_bpn
    return torch.stack((hits.sum(dtype=torch.int64), loss_mask_bpn.sum(dtype=torch.int64)))


def masked_policy_entropy_sum_count(
    logits: torch.Tensor,
    loss_mask_bpn: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    n_classes: int,
) -> torch.Tensor:
    nc = int(n_classes)
    assert logits.shape == (*loss_mask_bpn.shape, nc), (logits.shape, loss_mask_bpn.shape, nc)
    assert loss_mask_bpn.dtype == torch.bool, loss_mask_bpn.dtype
    assert action_mask.shape == logits.shape, (action_mask.shape, logits.shape)
    assert action_mask.dtype == torch.bool, action_mask.dtype
    neg_large = torch.finfo(logits.dtype).min / 16.0
    logits_for_entropy = logits.masked_fill(~action_mask, neg_large)
    log_probs = F.log_softmax(logits_for_entropy, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.masked_fill(~action_mask, 0.0)).sum(dim=-1)
    assert entropy.shape == loss_mask_bpn.shape, (entropy.shape, loss_mask_bpn.shape)
    w = loss_mask_bpn.to(dtype=entropy.dtype)
    return torch.stack(((entropy * w).sum(), w.sum()))


def confusion_tp_tn_fp_fn_to_metrics(counts: torch.Tensor) -> dict[str, float]:
    """Scalar accuracy / precision / recall from stacked ``[tp, tn, fp, fn]``."""
    assert counts.shape == (4,) and counts.dtype == torch.int64, (counts.shape, counts.dtype)
    tp = int(counts[0].item())
    tn = int(counts[1].item())
    fp = int(counts[2].item())
    fn = int(counts[3].item())
    out: dict[str, float] = {}
    tot = tp + tn + fp + fn
    if tot > 0:
        out["accuracy"] = float(tp + tn) / float(tot)
    if tp + fp > 0:
        out["precision"] = float(tp) / float(tp + fp)
    if tp + fn > 0:
        out["recall"] = float(tp) / float(tp + fn)
    return out


def _bc_batch_targets(
    batch_cpu: dict[str, torch.Tensor],
    *,
    device: torch.device,
    n_planets: int,
    n_grid_ship_buckets: int,
) -> BcPolicyBatchTargets:
    n = int(n_planets)
    nb = int(n_grid_ship_buckets)
    batch_obs = _batch_obs_to_device(batch_cpu, device=device)
    rl_tgt = batch_cpu["rl_per_planet_move_class"].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    bc_loss_source_mask = batch_cpu["bc_loss_source_mask"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    loss_mask, target_action_invalid_counts = compute_bc_rl_move_loss_mask(
        batch_obs,
        rl_tgt,
        bc_loss_source_mask,
        n_planets=n,
        n_policy_seats=int(rl_tgt.shape[1]),
    )
    assert torch.all((0 <= rl_tgt) & (rl_tgt < n * nb)), (rl_tgt.min(), rl_tgt.max(), n, nb)
    return BcPolicyBatchTargets(
        batch_obs=batch_obs,
        loss_mask=loss_mask,
        move_tgt=rl_tgt,
        target_action_invalid_counts=target_action_invalid_counts,
    )


def _record_bc_targets_stream(targets: BcPolicyBatchTargets, stream: torch.cuda.Stream) -> None:
    for tensor in targets.batch_obs.values():
        tensor.record_stream(stream)
    targets.loss_mask.record_stream(stream)
    targets.move_tgt.record_stream(stream)
    targets.target_action_invalid_counts.record_stream(stream)


class CudaBcTargetPrefetcher:
    def __init__(
        self,
        source: DataLoader[dict[str, torch.Tensor]],
        *,
        device: torch.device,
        n_planets: int,
        n_grid_ship_buckets: int,
    ) -> None:
        self._source_iter = iter(source)
        self._device = device
        self._n_planets = int(n_planets)
        self._n_grid_ship_buckets = int(n_grid_ship_buckets)
        self._stream = torch.cuda.Stream(device=device)
        self._next_targets: BcPolicyBatchTargets | None = None
        self._preload()

    def __iter__(self) -> "CudaBcTargetPrefetcher":
        return self

    def _preload(self) -> None:
        batch_cpu = next(self._source_iter)
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            self._next_targets = _bc_batch_targets(
                batch_cpu,
                device=self._device,
                n_planets=self._n_planets,
                n_grid_ship_buckets=self._n_grid_ship_buckets,
            )

    def __next__(self) -> BcPolicyBatchTargets:
        torch.cuda.current_stream(self._device).wait_stream(self._stream)
        targets = self._next_targets
        assert targets is not None
        _record_bc_targets_stream(targets, torch.cuda.current_stream(self._device))
        self._preload()
        return targets


def _final_policy_logits_from_model_output(
    out: dict[str, object],
    *,
    n_planets: int,
    n_policy_seats: int,
) -> tuple[torch.Tensor, int, int]:
    n = int(n_planets)
    p = int(n_policy_seats)
    final_logits_by_action = out["final_policy_logits_LEARN"]
    assert isinstance(final_logits_by_action, dict)
    assert tuple(final_logits_by_action.keys()) == ("spawn_fleet",), (
        tuple(final_logits_by_action.keys()),
    )
    final_policy_logits = final_logits_by_action["spawn_fleet"]
    assert isinstance(final_policy_logits, torch.Tensor)
    bsz = int(final_policy_logits.shape[0])
    assert final_policy_logits.shape[:3] == (bsz, p, n), final_policy_logits.shape
    action_width = int(final_policy_logits.shape[3])
    assert action_width % n == 0, (final_policy_logits.shape, n)
    model_actions_per_target = action_width // n
    assert 1 <= model_actions_per_target <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        model_actions_per_target,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    return final_policy_logits, bsz, model_actions_per_target


def _bc_env_action_mask_for_model(
    action_mask: torch.Tensor,
    *,
    model_actions_per_target: int,
) -> torch.Tensor:
    assert action_mask.ndim >= 2, action_mask.shape
    n = int(action_mask.shape[-2])
    env_width = int(action_mask.shape[-1])
    assert env_width == n * int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        action_mask.shape,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    model_nb = int(model_actions_per_target)
    assert 1 <= model_nb <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        model_nb,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    action_by_dst = action_mask.reshape(
        *action_mask.shape[:-1],
        n,
        int(ORBIT_MOVE_CLASSES_PER_TARGET),
    )
    return action_by_dst[..., :model_nb].reshape(
        *action_mask.shape[:-1],
        n * model_nb,
    ) > 0


def _bc_env_action_target_to_model(
    target: torch.Tensor,
    *,
    model_actions_per_target: int,
) -> torch.Tensor:
    model_nb = int(model_actions_per_target)
    assert 1 <= model_nb <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        model_nb,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    dst = target // int(ORBIT_MOVE_CLASSES_PER_TARGET)
    subindex = target % int(ORBIT_MOVE_CLASSES_PER_TARGET)
    assert torch.all(subindex < model_nb), (
        "BC target subaction must exist in the model policy head",
        int(subindex.max().item()),
        model_nb,
    )
    return dst * model_nb + subindex


def _bc_floating_outputs_to_float32(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            return value.to(dtype=torch.float32)
        return value
    if isinstance(value, dict):
        return {
            key: _bc_floating_outputs_to_float32(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_bc_floating_outputs_to_float32(item) for item in value)
    if isinstance(value, list):
        return [_bc_floating_outputs_to_float32(item) for item in value]
    return value


def _bc_policy_loss_metrics_from_output(
    out: dict[str, object],
    targets: BcPolicyBatchTargets,
    *,
    n_planets: int,
    n_grid_ship_buckets: int,
) -> BcPolicyBatchMetrics:
    n = int(n_planets)
    nb = int(n_grid_ship_buckets)
    p = int(targets.loss_mask.shape[1])
    final_policy_logits, bsz, model_actions_per_target = _final_policy_logits_from_model_output(
        out,
        n_planets=n,
        n_policy_seats=p,
    )
    action_mask = targets.batch_obs["available_action_mask"]
    assert isinstance(action_mask, torch.Tensor)
    assert action_mask.shape == (bsz, p, n, n * nb), action_mask.shape
    assert action_mask.dtype == torch.int8, action_mask.dtype
    action_mask_bool = _bc_env_action_mask_for_model(
        action_mask,
        model_actions_per_target=model_actions_per_target,
    )
    move_tgt = _bc_env_action_target_to_model(
        targets.move_tgt,
        model_actions_per_target=model_actions_per_target,
    )
    n_model_classes = n * int(model_actions_per_target)
    final_policy_loss = masked_ce_loss(
        final_policy_logits,
        move_tgt,
        targets.loss_mask,
        action_mask_bool,
        n_classes=n_model_classes,
    )
    combined_loss = final_policy_loss
    with torch.no_grad():
        final_policy_logits_ng = final_policy_logits.detach()
        final_policy_topk: dict[int, torch.Tensor] = {}
        for k in _BC_FINAL_TOPK_LEVELS:
            ki = int(k)
            final_policy_topk[ki] = masked_policy_topk_counts(
                final_policy_logits_ng,
                move_tgt,
                targets.loss_mask,
                action_mask_bool,
                n_classes=n_model_classes,
                k=ki,
            )
        final_policy_entropy = masked_policy_entropy_sum_count(
            final_policy_logits_ng,
            targets.loss_mask,
            action_mask_bool,
            n_classes=n_model_classes,
        )
    return BcPolicyBatchMetrics(
        combined_loss=combined_loss,
        final_policy_loss=final_policy_loss,
        final_policy_topk=final_policy_topk,
        final_policy_entropy=final_policy_entropy,
        batch_size=bsz,
    )


def _bc_policy_forward_loss_metrics(
    model: torch.nn.Module,
    targets: BcPolicyBatchTargets,
    *,
    n_planets: int,
    n_grid_ship_buckets: int,
) -> BcPolicyBatchMetrics:
    model.set_is_sample(False)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            {"obs_LEARN_INFER": targets.batch_obs},
            output_full_policy_log_probs=False,
            include_policy_logits_pre_action_mask=True,
            include_final_policy_logits=True,
        )
    out = _bc_floating_outputs_to_float32(out)
    return _bc_policy_loss_metrics_from_output(
        out,
        targets,
        n_planets=n_planets,
        n_grid_ship_buckets=n_grid_ship_buckets,
    )


def _count_mean(counts: torch.Tensor) -> float:
    assert counts.shape == (2,), (counts.shape, counts)
    n = float(counts[1].item())
    return float(counts[0].item()) / n if n > 0.0 else float("nan")


def _final_topk_means_from_counts(
    final_policy_topk: dict[int, torch.Tensor],
) -> dict[int, float]:
    final_policy_mean: dict[int, float] = {}
    for k in _BC_FINAL_TOPK_LEVELS:
        ki = int(k)
        final_policy_mean[ki] = _count_mean(final_policy_topk[ki])
    return final_policy_mean


def _final_policy_log_message(
    *,
    final_policy_topk_mean: dict[int, float],
    final_policy_entropy: float,
) -> str:
    return "".join(
        f" final_policy_top{ki}={final_policy_topk_mean[ki]:.6f}"
        for ki in (int(kk) for kk in _BC_FINAL_TOPK_LEVELS)
    ) + f" final_policy_entropy={final_policy_entropy:.6f}"


@dataclass(frozen=True)
class BcPermutationFeatureSpec:
    name: str
    obs_key: str
    channel_indices: tuple[tuple[int | slice, ...], ...]
    feature_input: str


def _bc_perm_feature_append_feature_inputs(
    specs: list[BcPermutationFeatureSpec],
    *,
    name: str,
    obs_key: str,
    channel_indices: tuple[tuple[int | slice, ...], ...],
    feature_inputs: tuple[str, ...],
) -> None:
    assert obs_key in _OBS_EMBEDDING_FEATURE_KEY_BY_BASE, obs_key
    assert len(feature_inputs) >= 1, (name, feature_inputs)
    if "continuous" in feature_inputs:
        specs.append(
            BcPermutationFeatureSpec(
                f"continuous.{name}",
                obs_key,
                channel_indices,
                "continuous",
            )
        )
    if "embedding" in feature_inputs:
        specs.append(
            BcPermutationFeatureSpec(
                f"embedding.{name}",
                obs_key,
                channel_indices,
                "embedding",
            )
        )


def _bc_perm_feature_append_enemy_grouped(
    specs: list[BcPermutationFeatureSpec],
    enemy_pending: dict[str, list[tuple[int | slice, ...]]],
    *,
    obs_key: str,
    feature_inputs_for_logical: Callable[[str], tuple[str, ...]],
) -> None:
    for name in sorted(enemy_pending.keys()):
        idxs = tuple(enemy_pending[name])
        assert len(idxs) == int(ORBIT_ENEMY_AXIS_SLOTS), (name, len(idxs), ORBIT_ENEMY_AXIS_SLOTS)
        logical = name.split(".", maxsplit=1)[1].split("@", maxsplit=1)[0]
        _bc_perm_feature_append_feature_inputs(
            specs,
            name=name,
            obs_key=obs_key,
            channel_indices=idxs,
            feature_inputs=feature_inputs_for_logical(logical),
        )
    enemy_pending.clear()


def _bc_perm_feature_catalog(*, per_horizon_arrival: bool) -> tuple[BcPermutationFeatureSpec, ...]:
    planet_phys, planet_logical, edge_phys, edge_logical = (
        planet_edge_physical_to_logical_from_layout(ORBIT_IMPALA_OBS_FEATURE_LAYOUT)
    )
    specs: list[BcPermutationFeatureSpec] = []
    planet_enemy_pending: dict[str, list[tuple[int | slice, ...]]] = {}
    for c in range(int(ORBIT_PLANET_FEATURES)):
        logical = planet_logical[int(planet_phys[c].item())]
        if c < int(ORBIT_PLANET_BASE_FEATURES):
            _bc_perm_feature_append_feature_inputs(
                specs,
                name=f"planet.{logical}",
                obs_key="orbit_planet_features",
                channel_indices=((int(c),),),
                feature_inputs=orbit_obs_planet_feature_importance_inputs(logical),
            )
            continue
        off = c - int(ORBIT_PLANET_PLAYER_FEATURE_OFFSET)
        block = off // int(ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER)
        if block == _BC_SELF_PLAYER_BLOCK_INDEX:
            _bc_perm_feature_append_feature_inputs(
                specs,
                name=f"planet.{logical}@self",
                obs_key="orbit_planet_features",
                channel_indices=((int(c),),),
                feature_inputs=orbit_obs_planet_player_feature_importance_inputs(logical),
            )
            continue
        key = f"planet.{logical}@enemy"
        planet_enemy_pending.setdefault(key, []).append((int(c),))
    _bc_perm_feature_append_enemy_grouped(
        specs,
        planet_enemy_pending,
        obs_key="orbit_planet_features",
        feature_inputs_for_logical=orbit_obs_planet_player_feature_importance_inputs,
    )

    edge_enemy_pending: dict[str, list[tuple[int | slice, ...]]] = {}
    for c in range(int(ORBIT_EDGE_FEATURES)):
        logical = edge_logical[int(edge_phys[c].item())]
        if c < int(ORBIT_EDGE_BASE_FEATURES):
            _bc_perm_feature_append_feature_inputs(
                specs,
                name=f"edge.{logical}",
                obs_key="orbit_planet_pairwise_features",
                channel_indices=((int(c),),),
                feature_inputs=orbit_obs_edge_feature_importance_inputs(logical),
            )
            continue
        off = c - int(ORBIT_EDGE_PLAYER_FEATURE_OFFSET)
        block = off // int(ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER)
        if block == _BC_SELF_PLAYER_BLOCK_INDEX:
            _bc_perm_feature_append_feature_inputs(
                specs,
                name=f"edge.{logical}@self",
                obs_key="orbit_planet_pairwise_features",
                channel_indices=((int(c),),),
                feature_inputs=orbit_obs_edge_player_feature_importance_inputs(logical),
            )
            continue
        key = f"edge.{logical}@enemy"
        edge_enemy_pending.setdefault(key, []).append((int(c),))
    _bc_perm_feature_append_enemy_grouped(
        specs,
        edge_enemy_pending,
        obs_key="orbit_planet_pairwise_features",
        feature_inputs_for_logical=orbit_obs_edge_player_feature_importance_inputs,
    )

    if per_horizon_arrival:
        for h in range(int(ORBIT_PLANET_ARRIVAL_HORIZON)):
            arrival_enemy_pending: dict[str, list[tuple[int | slice, ...]]] = {}
            for s in range(int(ORBIT_PLAYER_AXIS_SLOTS)):
                for f, fname in enumerate(ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES):
                    ch = (int(h), int(s), int(f))
                    if s == _BC_SELF_PLAYER_BLOCK_INDEX:
                        _bc_perm_feature_append_feature_inputs(
                            specs,
                            name=f"arrival.{fname}@horizon{h:03d}@self",
                            obs_key="orbit_planet_arrival_features",
                            channel_indices=(ch,),
                            feature_inputs=orbit_obs_arrival_temporal_feature_importance_inputs(
                                fname
                            ),
                        )
                        continue
                    key = f"arrival.{fname}@horizon{h:03d}@enemy"
                    arrival_enemy_pending.setdefault(key, []).append(ch)
            _bc_perm_feature_append_enemy_grouped(
                specs,
                arrival_enemy_pending,
                obs_key="orbit_planet_arrival_features",
                feature_inputs_for_logical=orbit_obs_arrival_temporal_feature_importance_inputs,
            )
    else:
        arrival_enemy_pending: dict[str, list[tuple[int | slice, ...]]] = {}
        for s in range(int(ORBIT_PLAYER_AXIS_SLOTS)):
            for f, fname in enumerate(ORBIT_ARRIVAL_TEMPORAL_FEATURE_NAMES):
                ch = (slice(None), int(s), int(f))
                if s == _BC_SELF_PLAYER_BLOCK_INDEX:
                    _bc_perm_feature_append_feature_inputs(
                        specs,
                        name=f"arrival.{fname}@self",
                        obs_key="orbit_planet_arrival_features",
                        channel_indices=(ch,),
                        feature_inputs=orbit_obs_arrival_temporal_feature_importance_inputs(fname),
                    )
                    continue
                key = f"arrival.{fname}@enemy"
                arrival_enemy_pending.setdefault(key, []).append(ch)
        _bc_perm_feature_append_enemy_grouped(
            specs,
            arrival_enemy_pending,
            obs_key="orbit_planet_arrival_features",
            feature_inputs_for_logical=orbit_obs_arrival_temporal_feature_importance_inputs,
        )
    specs.append(
        BcPermutationFeatureSpec("mask.planet", "orbit_planet_mask", (), "shared")
    )
    specs.append(BcPermutationFeatureSpec("mask.enemy", "orbit_enemy_mask", (), "shared"))
    specs.append(
        BcPermutationFeatureSpec("mask.pairwise", "orbit_planet_pairwise_mask", (), "shared")
    )
    names = tuple(s.name for s in specs)
    assert len(names) == len(set(names)), "duplicate permutation feature names"
    return tuple(specs)


def _bc_perm_feature_views(
    batch: dict[str, torch.Tensor], spec: BcPermutationFeatureSpec
) -> tuple[torch.Tensor, ...]:
    obs_key = spec.obs_key
    if spec.feature_input == "embedding":
        obs_key = _OBS_EMBEDDING_FEATURE_KEY_BY_BASE[spec.obs_key]
    else:
        assert spec.feature_input in ("continuous", "shared"), spec.feature_input
    assert obs_key in batch, (obs_key, sorted(batch.keys()))
    t = batch[obs_key]
    assert isinstance(t, torch.Tensor), (obs_key, type(t))
    if len(spec.channel_indices) == 0:
        return (t,)
    return tuple(t[(Ellipsis, *channel_index)] for channel_index in spec.channel_indices)


def _bc_perm_player_block_valid_mask(
    batch: dict[str, torch.Tensor],
    player_block: int,
) -> torch.Tensor:
    assert 0 <= int(player_block) < int(ORBIT_PLAYER_AXIS_SLOTS), player_block
    if int(player_block) == _BC_SELF_PLAYER_BLOCK_INDEX:
        mask = batch["orbit_planet_mask"]
        assert isinstance(mask, torch.Tensor), type(mask)
        return torch.ones(mask.shape[:-1], dtype=torch.bool, device=mask.device)
    enemy_mask = batch["orbit_enemy_mask"]
    assert isinstance(enemy_mask, torch.Tensor), type(enemy_mask)
    enemy_idx = int(player_block) - 1
    assert 0 <= enemy_idx < int(ORBIT_ENEMY_AXIS_SLOTS), (player_block, enemy_idx)
    return enemy_mask[(Ellipsis, enemy_idx)] > 0.5


def _bc_perm_expand_prefix_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert mask.ndim <= target.ndim, (mask.shape, target.shape)
    for _ in range(target.ndim - mask.ndim):
        mask = mask.unsqueeze(-1)
    return mask.expand_as(target)


def _bc_perm_feature_valid_views(
    batch: dict[str, torch.Tensor],
    spec: BcPermutationFeatureSpec,
    views: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if len(spec.channel_indices) == 0:
        return tuple(torch.ones_like(view, dtype=torch.bool) for view in views)

    masks: list[torch.Tensor] = []
    for channel_index, view in zip(spec.channel_indices, views, strict=True):
        if spec.obs_key == "orbit_planet_features":
            assert len(channel_index) == 1, (spec.name, channel_index)
            planet_mask = batch["orbit_planet_mask"]
            assert isinstance(planet_mask, torch.Tensor), type(planet_mask)
            valid = planet_mask > 0.5
            channel = int(channel_index[0])
            if channel >= int(ORBIT_PLANET_PLAYER_FEATURE_OFFSET):
                off = channel - int(ORBIT_PLANET_PLAYER_FEATURE_OFFSET)
                player_block = off // int(ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER)
                block_valid = _bc_perm_player_block_valid_mask(batch, player_block)
                valid = valid & _bc_perm_expand_prefix_mask(block_valid, valid)
        elif spec.obs_key == "orbit_planet_pairwise_features":
            assert len(channel_index) == 1, (spec.name, channel_index)
            pair_mask = batch["orbit_planet_pairwise_mask"]
            assert isinstance(pair_mask, torch.Tensor), type(pair_mask)
            valid = pair_mask > 0.5
            channel = int(channel_index[0])
            if channel >= int(ORBIT_EDGE_PLAYER_FEATURE_OFFSET):
                off = channel - int(ORBIT_EDGE_PLAYER_FEATURE_OFFSET)
                player_block = off // int(ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER)
                block_valid = _bc_perm_player_block_valid_mask(batch, player_block)
                valid = valid & _bc_perm_expand_prefix_mask(block_valid, valid)
        elif spec.obs_key == "orbit_planet_arrival_features":
            assert len(channel_index) == 3, (spec.name, channel_index)
            planet_mask = batch["orbit_planet_mask"]
            assert isinstance(planet_mask, torch.Tensor), type(planet_mask)
            horizon_index, player_block_raw, _feature_index = channel_index
            if isinstance(horizon_index, slice):
                valid = (planet_mask > 0.5).unsqueeze(-1).expand_as(view)
            else:
                assert isinstance(horizon_index, int), (spec.name, channel_index)
                valid = planet_mask > 0.5
            player_block = int(player_block_raw)
            block_valid = _bc_perm_player_block_valid_mask(batch, player_block)
            valid = valid & _bc_perm_expand_prefix_mask(block_valid, valid)
        else:
            valid = torch.ones_like(view, dtype=torch.bool)
        assert valid.shape == view.shape, (spec.name, valid.shape, view.shape)
        masks.append(valid)
    return tuple(masks)


def _bc_share_val_batches_(
    val_batches_cpu: tuple[dict[str, torch.Tensor], ...],
) -> None:
    for batch in val_batches_cpu:
        for key, tensor in batch.items():
            assert isinstance(tensor, torch.Tensor), (key, type(tensor))
            assert tensor.device.type == "cpu", (key, tensor.device)
            tensor.share_memory_()


def _bc_perm_shuffle_feature_in_batch_(
    batch_cpu: dict[str, torch.Tensor],
    spec: BcPermutationFeatureSpec,
    rng: torch.Generator,
) -> None:
    views = _bc_perm_feature_views(batch_cpu, spec)
    valid_views = _bc_perm_feature_valid_views(batch_cpu, spec, views)
    flat_saved = torch.cat(
        [view[valid] for view, valid in zip(views, valid_views, strict=True)],
        dim=0,
    )
    perm = torch.randperm(int(flat_saved.numel()), generator=rng)
    flat_shuffled = flat_saved[perm]
    off = 0
    for view, valid in zip(views, valid_views, strict=True):
        n = int(valid.sum().item())
        view[valid] = flat_shuffled[off : off + n]
        off += n
    assert off == int(flat_saved.numel()), (off, flat_saved.numel())


def _bc_copy_batch_with_perm_shuffled_feature(
    batch_cpu: dict[str, torch.Tensor],
    spec: BcPermutationFeatureSpec,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    out = dict(batch_cpu)
    split_feature_inputs = spec.feature_input in ("continuous", "embedding")
    if split_feature_inputs:
        for base_key, embedding_key in _OBS_EMBEDDING_FEATURE_KEY_BY_BASE.items():
            assert embedding_key not in out, (embedding_key, sorted(out.keys()))
            base = batch_cpu[base_key]
            assert isinstance(base, torch.Tensor), (base_key, type(base))
            out[embedding_key] = base.clone()
        if spec.feature_input == "continuous":
            base = batch_cpu[spec.obs_key]
            assert isinstance(base, torch.Tensor), (spec.obs_key, type(base))
            out[spec.obs_key] = base.clone()
        else:
            assert spec.feature_input == "embedding", spec.feature_input
    else:
        assert spec.feature_input == "shared", spec.feature_input
        tensor = batch_cpu[spec.obs_key]
        assert isinstance(tensor, torch.Tensor), (spec.obs_key, type(tensor))
        out[spec.obs_key] = tensor.clone()
    _bc_perm_shuffle_feature_in_batch_(out, spec, rng)
    return out


def _bc_perm_shuffled_batches(
    val_batches_cpu: tuple[dict[str, torch.Tensor], ...],
    spec: BcPermutationFeatureSpec,
    rng: torch.Generator,
) -> Iterator[dict[str, torch.Tensor]]:
    for batch_cpu in val_batches_cpu:
        yield _bc_copy_batch_with_perm_shuffled_feature(batch_cpu, spec, rng)


def _bc_validation_metrics_flat(
    model: torch.nn.Module,
    val_batches_cpu: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    n_planets: int,
    n_grid_ship_buckets: int,
) -> dict[str, float]:
    n = int(n_planets)
    nb = int(n_grid_ship_buckets)
    v_total_combined_loss = 0.0
    v_total_final_policy_loss = 0.0
    v_final_policy_topk = {
        int(k): torch.zeros(2, dtype=torch.int64) for k in _BC_FINAL_TOPK_LEVELS
    }
    v_final_policy_entropy = torch.zeros(2, dtype=torch.float64)
    v_target_action_invalid_counts = torch.zeros(5, dtype=torch.int64)
    v_n_batches = 0
    with torch.no_grad():
        for batch_cpu in val_batches_cpu:
            targets = _bc_batch_targets(
                batch_cpu,
                device=device,
                n_planets=n,
                n_grid_ship_buckets=nb,
            )
            v_target_action_invalid_counts += (
                targets.target_action_invalid_counts.detach().cpu()
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model.set_is_sample(False)
                out = model(
                    {"obs_LEARN_INFER": targets.batch_obs},
                    output_full_policy_log_probs=False,
                    include_policy_logits_pre_action_mask=True,
                    include_final_policy_logits=True,
                )
            out = _bc_floating_outputs_to_float32(out)
            metrics = _bc_policy_loss_metrics_from_output(
                out,
                targets,
                n_planets=n,
                n_grid_ship_buckets=nb,
            )
            v_total_combined_loss += float(metrics.combined_loss.detach().item())
            v_total_final_policy_loss += float(metrics.final_policy_loss.detach().item())
            for ki in metrics.final_policy_topk:
                v_final_policy_topk[ki] += metrics.final_policy_topk[ki].detach().cpu()
            v_final_policy_entropy += (
                metrics.final_policy_entropy.detach().cpu().to(dtype=torch.float64)
            )
            v_n_batches += 1
    assert v_n_batches >= 1, (
        "validation produced no full batches",
        v_n_batches,
    )
    v_n_b = float(v_n_batches)
    v_final_policy_topk_mean = _final_topk_means_from_counts(
        v_final_policy_topk,
    )
    out_metrics: dict[str, float] = {
        "loss_combined": v_total_combined_loss / v_n_b,
        "loss_final_policy": v_total_final_policy_loss / v_n_b,
        "final_policy_entropy": _count_mean(
            v_final_policy_entropy
        ),
    }
    for ki in _BC_FINAL_TOPK_LEVELS:
        out_metrics[f"final_policy_top{int(ki)}"] = float(
            v_final_policy_topk_mean[int(ki)]
        )
    assert v_target_action_invalid_counts.shape == (5,), v_target_action_invalid_counts.shape
    out_metrics["target_action_invalid_count"] = int(v_target_action_invalid_counts[0].item())
    out_metrics["target_action_invalid_active_choice_count"] = int(
        v_target_action_invalid_counts[1].item()
    )
    out_metrics["target_action_invalid_src_invalid_planet_count"] = int(
        v_target_action_invalid_counts[2].item()
    )
    out_metrics["target_action_invalid_no_choice_count"] = int(
        v_target_action_invalid_counts[3].item()
    )
    out_metrics["target_self_send_non_noop_active_choice_count"] = int(
        v_target_action_invalid_counts[4].item()
    )
    return out_metrics


def _bc_self_rl_importance_forward(
    model: torch.nn.Module,
    batch_obs: dict[str, torch.Tensor],
) -> dict[str, object]:
    model.set_is_sample(False)
    return model(
        {"obs_LEARN_INFER": batch_obs},
        output_full_policy_log_probs=True,
        include_policy_logits_pre_action_mask=False,
        include_final_policy_logits=False,
        include_value_head=True,
    )


def _bc_self_rl_baseline_tensor(out: dict[str, object]) -> torch.Tensor:
    baseline_by_head = out["baseline_LEARN"]
    assert isinstance(baseline_by_head, dict), type(baseline_by_head)
    assert "baseline" in baseline_by_head, sorted(baseline_by_head.keys())
    baseline = baseline_by_head["baseline"]
    assert isinstance(baseline, torch.Tensor), type(baseline)
    return baseline


def _bc_self_rl_importance_policy_class_mask(
    batch_obs: dict[str, torch.Tensor],
) -> torch.Tensor:
    planet_mask = batch_obs["orbit_planet_mask"]
    player_mask = batch_obs["player_mask"]
    available_action_mask = batch_obs["available_action_mask"]
    assert planet_mask.ndim == 3, planet_mask.shape
    assert player_mask.ndim == 2, player_mask.shape
    assert available_action_mask.ndim == 4, available_action_mask.shape
    b, p, n = planet_mask.shape
    assert player_mask.shape == (b, p), (player_mask.shape, (b, p))
    assert available_action_mask.shape[:3] == (b, p, n), (
        available_action_mask.shape,
        (b, p, n),
    )
    assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
    available = available_action_mask > 0
    source_planet_valid = planet_mask > 0.5
    has_choice = available.sum(dim=-1) > 1
    player_active = player_mask > 0.5
    mask = player_active.unsqueeze(-1) & source_planet_valid & has_choice
    assert mask.shape == (b, p, n), mask.shape
    return mask


def _bc_self_rl_importance_loss_metrics_from_output(
    out: dict[str, object],
    *,
    target_baseline: torch.Tensor,
    target_policy_log_probs: torch.Tensor,
    target_batch_obs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    baseline = _bc_self_rl_baseline_tensor(out)
    assert baseline.shape == target_baseline.shape, (
        baseline.shape,
        target_baseline.shape,
    )
    value_mask = target_batch_obs["player_mask"] > 0.5
    assert value_mask.shape == baseline.shape, (value_mask.shape, baseline.shape)
    value_loss = compute_baseline_loss(
        baseline.unsqueeze(0),
        target_baseline.unsqueeze(0),
        reduction="mean",
        mask=value_mask.unsqueeze(0),
    )

    policy_log_probs_by_key = out["policy_log_probs_LEARN"]
    assert isinstance(policy_log_probs_by_key, dict)
    assert tuple(policy_log_probs_by_key.keys()) == ("spawn_fleet",), (
        policy_log_probs_by_key.keys(),
    )
    policy_log_probs = policy_log_probs_by_key["spawn_fleet"]
    assert isinstance(policy_log_probs, torch.Tensor)
    assert policy_log_probs.shape == target_policy_log_probs.shape, (
        policy_log_probs.shape,
        target_policy_log_probs.shape,
    )
    available_masks = out["available_action_mask_LEARN"]
    assert isinstance(available_masks, dict), type(available_masks)
    assert tuple(available_masks.keys()) == ("spawn_fleet",), available_masks.keys()
    available = available_masks["spawn_fleet"]
    assert isinstance(available, torch.Tensor), type(available)
    assert available.shape == policy_log_probs.shape, (
        available.shape,
        policy_log_probs.shape,
    )
    assert available.dtype == torch.bool, available.dtype
    policy_kl = compute_teacher_kl_from_log_probs(
        policy_log_probs,
        target_policy_log_probs,
        available_action_mask=available,
        teacher_available_action_mask=available,
        zero_missing_policy_actions=False,
    )
    assert policy_kl.shape == value_mask.shape, (policy_kl.shape, value_mask.shape)
    policy_loss = reduce(
        policy_kl.unsqueeze(0),
        reduction="mean",
        mask=value_mask.unsqueeze(0),
    )

    target_policy_class = target_policy_log_probs.argmax(dim=-1)
    assert target_policy_class.shape == policy_log_probs.shape[:-1], (
        target_policy_class.shape,
        policy_log_probs.shape,
    )
    selected_target_available = torch.gather(
        available,
        dim=-1,
        index=target_policy_class.unsqueeze(-1),
    ).squeeze(-1)
    assert torch.all(selected_target_available), "clean policy class targets must be available"
    selected_log_probs = torch.gather(
        policy_log_probs.to(dtype=torch.float32),
        dim=-1,
        index=target_policy_class.unsqueeze(-1),
    ).squeeze(-1)
    assert selected_log_probs.shape == target_policy_class.shape, (
        selected_log_probs.shape,
        target_policy_class.shape,
    )
    policy_class_mask = _bc_self_rl_importance_policy_class_mask(target_batch_obs)
    policy_ce_loss = reduce(
        -selected_log_probs.unsqueeze(0),
        reduction="mean",
        mask=policy_class_mask.unsqueeze(0),
    )
    policy_class_pred = policy_log_probs.argmax(dim=-1)
    assert policy_class_pred.shape == target_policy_class.shape, (
        policy_class_pred.shape,
        target_policy_class.shape,
    )
    policy_class_correct = (policy_class_pred == target_policy_class).to(dtype=torch.float32)
    policy_class_accuracy = reduce(
        policy_class_correct.unsqueeze(0),
        reduction="mean",
        mask=policy_class_mask.unsqueeze(0),
    )
    combined_loss = value_loss + policy_loss + policy_ce_loss
    return {
        "loss_self_rl_combined": combined_loss,
        "loss_self_rl_value_smooth": value_loss,
        "loss_self_rl_policy_kl": policy_loss,
        "loss_self_rl_policy_class_ce": policy_ce_loss,
        "accuracy_self_rl_policy_class": policy_class_accuracy,
    }


def _bc_self_rl_importance_metrics_flat(
    model: torch.nn.Module,
    val_batches_cpu: tuple[dict[str, torch.Tensor], ...],
    *,
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {
        "loss_self_rl_combined": 0.0,
        "loss_self_rl_value_smooth": 0.0,
        "loss_self_rl_policy_kl": 0.0,
        "loss_self_rl_policy_class_ce": 0.0,
        "accuracy_self_rl_policy_class": 0.0,
    }
    n_batches = 0
    with torch.no_grad():
        for batch_cpu in val_batches_cpu:
            batch_obs = _batch_obs_to_device(batch_cpu, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = _bc_self_rl_importance_forward(model, batch_obs)
            out = _bc_floating_outputs_to_float32(out)
            baseline = _bc_self_rl_baseline_tensor(out)
            policy_log_probs_by_key = out["policy_log_probs_LEARN"]
            assert isinstance(policy_log_probs_by_key, dict)
            target_policy_log_probs = policy_log_probs_by_key["spawn_fleet"]
            assert isinstance(target_policy_log_probs, torch.Tensor)
            metrics = _bc_self_rl_importance_loss_metrics_from_output(
                out,
                target_baseline=baseline.detach(),
                target_policy_log_probs=target_policy_log_probs.detach(),
                target_batch_obs=batch_obs,
            )
            for key, value in metrics.items():
                totals[key] += float(value.detach().item())
            n_batches += 1
    assert n_batches >= 1, ("validation produced no full batches", n_batches)
    return {key: val / float(n_batches) for key, val in totals.items()}


def _bc_self_rl_importance_permutation_metrics_flat(
    model: torch.nn.Module,
    val_batches_cpu: tuple[dict[str, torch.Tensor], ...],
    spec: BcPermutationFeatureSpec,
    rng: torch.Generator,
    *,
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {
        "loss_self_rl_combined": 0.0,
        "loss_self_rl_value_smooth": 0.0,
        "loss_self_rl_policy_kl": 0.0,
        "loss_self_rl_policy_class_ce": 0.0,
        "accuracy_self_rl_policy_class": 0.0,
    }
    n_batches = 0
    with torch.no_grad():
        for batch_cpu in val_batches_cpu:
            target_batch_obs = _batch_obs_to_device(batch_cpu, device=device)
            shuffled_batch_cpu = _bc_copy_batch_with_perm_shuffled_feature(
                batch_cpu,
                spec,
                rng,
            )
            shuffled_batch_obs = _batch_obs_to_device(shuffled_batch_cpu, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                target_out = _bc_self_rl_importance_forward(model, target_batch_obs)
                shuffled_out = _bc_self_rl_importance_forward(model, shuffled_batch_obs)
            target_out = _bc_floating_outputs_to_float32(target_out)
            shuffled_out = _bc_floating_outputs_to_float32(shuffled_out)
            target_baseline = _bc_self_rl_baseline_tensor(target_out)
            target_policy_log_probs_by_key = target_out["policy_log_probs_LEARN"]
            assert isinstance(target_policy_log_probs_by_key, dict)
            target_policy_log_probs = target_policy_log_probs_by_key["spawn_fleet"]
            assert isinstance(target_policy_log_probs, torch.Tensor)
            metrics = _bc_self_rl_importance_loss_metrics_from_output(
                shuffled_out,
                target_baseline=target_baseline.detach(),
                target_policy_log_probs=target_policy_log_probs.detach(),
                target_batch_obs=target_batch_obs,
            )
            for key, value in metrics.items():
                totals[key] += float(value.detach().item())
            n_batches += 1
    assert n_batches >= 1, ("validation produced no full batches", n_batches)
    return {key: val / float(n_batches) for key, val in totals.items()}


def _bc_write_feature_importance_metric_file(
    out_dir: Path,
    metric_name: str,
    *,
    baseline: float,
    shuffled_by_feature: dict[str, float],
) -> None:
    rows = [
        (feat_name, float(shuf_val), float(shuf_val) - float(baseline))
        for feat_name, shuf_val in shuffled_by_feature.items()
    ]
    rows.sort(key=lambda row: row[2], reverse=True)
    lines = [
        f"# baseline={baseline:.8f}",
        "# feature shuffled delta",
    ]
    lines.extend(f"{feat} {shuf:.8f} {delta:.8f}" for feat, shuf, delta in rows)
    out_path = out_dir / f"{metric_name}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


@dataclass(frozen=True)
class BcFeatureImportanceWorkerTask:
    feature_index: int
    feature_count: int
    spec: BcPermutationFeatureSpec


_BC_FI_WORKER_VAL_BATCHES: tuple[dict[str, torch.Tensor], ...] | None = None
_BC_FI_WORKER_MODEL: torch.nn.Module | None = None
_BC_FI_WORKER_DEVICE: torch.device | None = None
_BC_FI_WORKER_N_PLANETS: int | None = None
_BC_FI_WORKER_N_GRID_SHIP_BUCKETS: int | None = None
_BC_FI_WORKER_SELF_RL_TARGETS: bool | None = None


def _bc_load_feature_importance_model(
    *,
    resume_checkpoint: Path | None,
    rl_checkpoint: Path | None,
    device: torch.device,
    include_rl_value_head: bool,
) -> torch.nn.Module:
    assert resume_checkpoint is not None or rl_checkpoint is not None, (
        "feature importance requires --resume-checkpoint or --rl-checkpoint"
    )
    include_value = bool(include_rl_value_head)
    if resume_checkpoint is not None:
        assert resume_checkpoint.is_file(), resume_checkpoint
        resume_ckpt = torch.load(str(resume_checkpoint), map_location="cpu", weights_only=False)
        assert isinstance(resume_ckpt, dict), type(resume_ckpt)
        model = ImpalaOrbitModel(
            **bc_feature_importance_model_init_kwargs(
                _bc_checkpoint_model_config(resume_ckpt),
                include_rl_value_head=include_value,
            )
        ).to(device)
        sd = _strip_torch_compile_orig_mod_prefix(resume_ckpt["model_state_dict"])
        _load_bc_resume_model_state_dict(model, sd)
    else:
        assert rl_checkpoint is not None
        assert rl_checkpoint.is_file(), rl_checkpoint
        ckpt = torch.load(str(rl_checkpoint), map_location="cpu", weights_only=False)
        assert isinstance(ckpt, dict) and "model_state_dict" in ckpt
        model = ImpalaOrbitModel(
            **bc_feature_importance_model_init_kwargs(
                _bc_checkpoint_model_config(ckpt),
                include_rl_value_head=include_value,
            )
        ).to(device)
        sd = _strip_torch_compile_orig_mod_prefix(ckpt["model_state_dict"])
        _load_bc_resume_model_state_dict(model, sd)
    #model = torch.compile(model, fullgraph=True, dynamic=False)
    model.eval()
    return model


def _bc_feature_importance_shuffle_seed(feature_index: int) -> int:
    idx = int(feature_index)
    assert idx >= 0, idx
    return int(_BC_FEATURE_IMPORTANCE_SHUFFLE_SEED) + idx * 1_000_003


def _bc_feature_importance_worker_init(
    val_batches_cpu: tuple[dict[str, torch.Tensor], ...],
    resume_checkpoint: Path | None,
    rl_checkpoint: Path | None,
    gpu_id: int,
    n_planets: int,
    n_grid_ship_buckets: int,
    self_rl_targets: bool,
) -> None:
    global _BC_FI_WORKER_VAL_BATCHES
    global _BC_FI_WORKER_MODEL
    global _BC_FI_WORKER_DEVICE
    global _BC_FI_WORKER_N_PLANETS
    global _BC_FI_WORKER_N_GRID_SHIP_BUCKETS
    global _BC_FI_WORKER_SELF_RL_TARGETS

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(f"cuda:{int(gpu_id)}")
    _BC_FI_WORKER_VAL_BATCHES = val_batches_cpu
    _BC_FI_WORKER_DEVICE = device
    _BC_FI_WORKER_N_PLANETS = int(n_planets)
    _BC_FI_WORKER_N_GRID_SHIP_BUCKETS = int(n_grid_ship_buckets)
    _BC_FI_WORKER_SELF_RL_TARGETS = bool(self_rl_targets)
    _BC_FI_WORKER_MODEL = _bc_load_feature_importance_model(
        resume_checkpoint=resume_checkpoint,
        rl_checkpoint=rl_checkpoint,
        device=device,
        include_rl_value_head=bool(self_rl_targets),
    )


def _bc_feature_importance_worker_run(
    task: BcFeatureImportanceWorkerTask,
) -> tuple[str, dict[str, float]]:
    assert _BC_FI_WORKER_VAL_BATCHES is not None
    assert _BC_FI_WORKER_MODEL is not None
    assert _BC_FI_WORKER_DEVICE is not None
    assert _BC_FI_WORKER_N_PLANETS is not None
    assert _BC_FI_WORKER_N_GRID_SHIP_BUCKETS is not None
    assert _BC_FI_WORKER_SELF_RL_TARGETS is not None
    feat_i = int(task.feature_index)
    feat_count = int(task.feature_count)
    assert 0 <= feat_i < feat_count, (feat_i, feat_count)
    print(
        f"BC feature importance worker pid={os.getpid()}: "
        f"feature {feat_i + 1}/{feat_count} {task.spec.name}",
        flush=True,
    )
    rng = torch.Generator(device="cpu")
    rng.manual_seed(_bc_feature_importance_shuffle_seed(feat_i))
    if _BC_FI_WORKER_SELF_RL_TARGETS:
        metrics = _bc_self_rl_importance_permutation_metrics_flat(
            _BC_FI_WORKER_MODEL,
            _BC_FI_WORKER_VAL_BATCHES,
            task.spec,
            rng,
            device=_BC_FI_WORKER_DEVICE,
        )
    else:
        metrics = _bc_validation_metrics_flat(
            _BC_FI_WORKER_MODEL,
            _bc_perm_shuffled_batches(_BC_FI_WORKER_VAL_BATCHES, task.spec, rng),
            device=_BC_FI_WORKER_DEVICE,
            n_planets=_BC_FI_WORKER_N_PLANETS,
            n_grid_ship_buckets=_BC_FI_WORKER_N_GRID_SHIP_BUCKETS,
        )
    return task.spec.name, metrics


def _bc_run_feature_importance(
    *,
    args: argparse.Namespace,
    val_batches_cpu: tuple[dict[str, torch.Tensor], ...],
    device: torch.device,
) -> None:
    assert args.resume_checkpoint is not None or args.rl_checkpoint is not None, (
        "feature importance requires --resume-checkpoint or --rl-checkpoint"
    )
    n = int(ORBIT_MAX_PLANETS)
    nb = int(ORBIT_MOVE_CLASSES_PER_TARGET)
    self_rl_targets = bool(args.feature_importance_self_rl_targets)
    feature_importance_workers = int(args.feature_importance_workers)
    assert feature_importance_workers >= 1, feature_importance_workers
    print("BC feature importance: moving validation batches to shared memory", flush=True)
    _bc_share_val_batches_(val_batches_cpu)
    model = _bc_load_feature_importance_model(
        resume_checkpoint=args.resume_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        device=device,
        include_rl_value_head=self_rl_targets,
    )
    out_dir = _PY_ROOT.parent / "outputs" / "importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_dir / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"BC feature importance: out_dir={out_dir}", flush=True)
    print("BC feature importance: baseline validation pass", flush=True)
    if self_rl_targets:
        baseline_metrics = _bc_self_rl_importance_metrics_flat(
            model,
            val_batches_cpu,
            device=device,
        )
    else:
        baseline_metrics = _bc_validation_metrics_flat(
            model,
            val_batches_cpu,
            device=device,
            n_planets=n,
            n_grid_ship_buckets=nb,
        )
    del model
    torch.cuda.empty_cache()
    per_horizon_arrival = bool(args.feature_importance_per_horizon)
    feature_specs = _bc_perm_feature_catalog(per_horizon_arrival=per_horizon_arrival)
    print(
        f"BC feature importance: {len(feature_specs)} features "
        f"(per_horizon_arrival={per_horizon_arrival}), "
        f"self_rl_targets={self_rl_targets}, "
        f"{len(baseline_metrics)} metrics, "
        f"workers={feature_importance_workers}",
        flush=True,
    )
    shuffled_metrics_by_feature: dict[str, dict[str, float]] = {
        spec.name: {} for spec in feature_specs
    }
    tasks = tuple(
        BcFeatureImportanceWorkerTask(
            feature_index=feat_i,
            feature_count=len(feature_specs),
            spec=spec,
        )
        for feat_i, spec in enumerate(feature_specs)
    )
    ctx = mp.get_context("spawn")
    completed = 0
    with ctx.Pool(
        processes=feature_importance_workers,
        initializer=_bc_feature_importance_worker_init,
        initargs=(
            val_batches_cpu,
            args.resume_checkpoint,
            args.rl_checkpoint,
            int(args.gpu_id),
            n,
            nb,
            self_rl_targets,
        ),
    ) as pool:
        for feat_name, shuffled_metrics in pool.imap_unordered(
            _bc_feature_importance_worker_run,
            tasks,
            chunksize=1,
        ):
            completed += 1
            print(
                f"BC feature importance: completed {completed}/{len(feature_specs)} {feat_name}",
                flush=True,
            )
            for metric_name, metric_val in shuffled_metrics.items():
                shuffled_metrics_by_feature[feat_name][metric_name] = float(metric_val)
    assert set(shuffled_metrics_by_feature) == {spec.name for spec in feature_specs}
    for feat_name, metrics in shuffled_metrics_by_feature.items():
        assert set(metrics) == set(baseline_metrics), (feat_name, sorted(metrics))
    for metric_name, baseline_val in baseline_metrics.items():
        if metric_name.startswith("target_action_invalid"):
            continue
        shuffled_by_feature = {
            feat_name: shuffled_metrics_by_feature[feat_name][metric_name]
            for feat_name in shuffled_metrics_by_feature
        }
        _bc_write_feature_importance_metric_file(
            out_dir,
            metric_name,
            baseline=float(baseline_val),
            shuffled_by_feature=shuffled_by_feature,
        )


def _bc_policy_metric_wandb_payload(
    wb_tag: str,
    *,
    combined_loss: float,
    final_policy_loss: float,
    final_policy_topk_mean: dict[int, float],
    final_policy_entropy: float,
) -> dict[str, float]:
    payload: dict[str, float] = {
        f"Loss.combined_{wb_tag}": float(combined_loss),
        f"Loss.final_policy_{wb_tag}": float(final_policy_loss),
        f"Everything.final_policy_entropy_{wb_tag}": float(
            final_policy_entropy
        ),
    }
    for k in _BC_FINAL_TOPK_LEVELS:
        ki = int(k)
        if math.isfinite(final_policy_topk_mean[ki]):
            payload[f"Everything.final_policy_top{ki}_{wb_tag}"] = (
                final_policy_topk_mean[ki]
            )
    return _wandb_float_dict_finite_only(payload)


def _bc_train_metric_smoothed_update(
    smoothed: dict[str, RollingAverage],
    stats: dict[str, float],
) -> None:
    for key, value in stats.items():
        if key not in smoothed:
            smoothed[key] = RollingAverage(window_size=_BC_TRAIN_METRIC_SMOOTH_WINDOW)
        smoothed[key].add(value)


def _bc_train_smoothed_wandb_payload(
    smoothed: dict[str, RollingAverage],
    batch_stats: dict[str, float],
) -> dict[str, float]:
    """Rolling mean over the last _BC_TRAIN_METRIC_SMOOTH_WINDOW train batches; keys unchanged."""
    out = {
        key: float(smoothed[key].average())
        for key in batch_stats
        if (key in smoothed) and smoothed[key].is_full()
    }
    return _wandb_float_dict_finite_only(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=_PY_ROOT.parent / "datasets" / "bc_v1",
    )
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Validation batch size; defaults to --batch-size.",
    )
    ap.add_argument(
        "--val-loader-workers",
        type=int,
        default=8,
        help="Worker processes that load validation episodes during CPU precompute.",
    )
    ap.add_argument(
        "--val-every-batches",
        type=int,
        default=VAL_EVERY_BATCHES,
        help="Run one fixed validation pass every this many train batches.",
    )
    ap.add_argument(
        "--no-val",
        action="store_true",
        help="Disable validation split loading, validation precompute, and validation passes.",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--warmup-training-samples",
        type=int,
        default=_BC_LR_WARMUP_TRAIN_SAMPLES_DEFAULT,
        help="Linear LR warmup: 0 -> --lr over this many training samples.",
    )
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=_BC_MAX_TRAIN_SAMPLES_DEFAULT,
        help=(
            "Training sample budget for the LR schedule: after warmup, LR decays linearly to 0 "
            "by this total train sample count (inclusive); training stops once reached."
        ),
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index (model + train step).",
    )
    ap.add_argument(
        "--rl-checkpoint",
        type=Path,
        default=None,
        help="Optional IMPALA checkpoint; weights load strictly except explicit RL value-head checkpoint-only keys.",
    )
    ap.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "BC ``last.pt`` / ``samples_*.pt`` from this script: restore weights (handles torch.compile "
            "``_orig_mod.`` keys), train batch/sample counters, class-balance accumulators, and optimizer "
            "when present. New outputs go under a fresh ``outputs/supervised/<timestamp>/`` directory "
            "(same as a cold start). Incompatible with --rl-checkpoint."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional extra path for the latest validation checkpoint copy; "
            "periodic ``samples_*.pt`` checkpoints and last.pt go under outputs/supervised/<timestamp>/."
        ),
    )
    ap.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project (default: same as RL ``ImpalaTrainingConfig.wandb_project``).",
    )
    ap.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: bc_rl_policy_<timestamp>).",
    )
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    ap.add_argument(
        "--pool-episodes",
        type=int,
        default=N_EPISODES,
        help="Train and val: how many episode .pt tensors are resident in RAM at once (pool).",
    )
    ap.add_argument(
        "--train-loader-workers",
        type=int,
        default=32,
        help="Worker processes that prepare train CPU batches ahead of the GPU step.",
    )
    ap.add_argument(
        "--regen-val",
        action="store_true",
        help=(
            "Allow creating or overwriting validation_episodes.txt when it is missing "
            "or has fewer than N_VALIDATION_EPISODES valid lines (random holdout)."
        ),
    )
    ap.add_argument(
        "--feature-importance",
        action="store_true",
        help=(
            "Skip training: run validation baseline, then permutation importance for each "
            "observation feature channel; write outputs/importance/<timestamp>/<metric>.txt."
        ),
    )
    ap.add_argument(
        "--feature-importance-per-horizon",
        action="store_true",
        help=(
            "With --feature-importance: treat each arrival horizon step as its own feature "
            "(~5200 arrival channels). Default: shuffle each arrival temporal channel across "
            "all horizon steps together (~52 arrival features)."
        ),
    )
    ap.add_argument(
        "--feature-importance-self-rl-targets",
        action="store_true",
        help=(
            "With --feature-importance: use the model's clean value/policy outputs as targets, "
            "then score shuffled features with smooth value loss and policy KL."
        ),
    )
    ap.add_argument(
        "--feature-importance-workers",
        type=int,
        default=2,
        help="Worker processes for --feature-importance permutation passes.",
    )
    ap.add_argument(
        "--feature-importance-val-episodes",
        type=int,
        default=_BC_FEATURE_IMPORTANCE_VAL_EPISODES_DEFAULT,
        help=(
            "With --feature-importance: how many validation holdout episodes to load "
            "(first N by sorted basename; default "
            f"{_BC_FEATURE_IMPORTANCE_VAL_EPISODES_DEFAULT})."
        ),
    )
    args = ap.parse_args()
    assert not (
        args.rl_checkpoint is not None and args.resume_checkpoint is not None
    ), "use either --rl-checkpoint or --resume-checkpoint, not both"
    use_validation = not bool(args.no_val)
    feature_importance = bool(args.feature_importance)
    if not use_validation:
        assert not feature_importance, "--feature-importance requires validation; remove --no-val"
    if feature_importance:
        assert args.resume_checkpoint is not None or args.rl_checkpoint is not None, (
            "--feature-importance requires --resume-checkpoint or --rl-checkpoint"
        )
    if bool(args.feature_importance_per_horizon):
        assert feature_importance, "--feature-importance-per-horizon requires --feature-importance"
    if bool(args.feature_importance_self_rl_targets):
        assert feature_importance, "--feature-importance-self-rl-targets requires --feature-importance"
    assert int(args.feature_importance_workers) >= 1, args.feature_importance_workers
    feature_importance_val_episodes = int(args.feature_importance_val_episodes)
    assert feature_importance_val_episodes >= 1, feature_importance_val_episodes

    warmup_training_samples = int(args.warmup_training_samples)
    max_train_samples = int(args.max_train_samples)
    assert warmup_training_samples >= 1, warmup_training_samples
    assert max_train_samples > warmup_training_samples, (
        max_train_samples,
        warmup_training_samples,
    )

    _bc_optim_cfg = ImpalaTrainingConfig().optimizer_config
    _bc_optimizer_kwargs = dict(_bc_optim_cfg.optimizer_kwargs)
    _bc_optimizer_kwargs["lr"] = float(args.lr)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    assert torch.cuda.is_available(), "CUDA is required."
    gid = int(args.gpu_id)
    assert 0 <= gid < torch.cuda.device_count(), (gid, torch.cuda.device_count())
    device = torch.device(f"cuda:{gid}")
    print(
        "BC: lr schedule "
        f"lr_max={float(args.lr)} "
        f"warmup_training_samples={warmup_training_samples} "
        f"max_train_samples={max_train_samples}",
        flush=True,
    )
    if use_validation:
        print(f"BC: resolving train/val episode paths under {args.data_dir}", flush=True)
        train_paths, val_paths = _train_val_episode_paths(args.data_dir, regen_val=args.regen_val)
    else:
        print(f"BC: resolving train episode paths under {args.data_dir} (--no-val)", flush=True)
        train_paths = _list_episode_pt_files(args.data_dir)
        val_paths = []
    n_train_ep = len(train_paths)
    n_val_ep = len(val_paths)
    bs = int(args.batch_size)
    val_bs = bs if args.val_batch_size is None else int(args.val_batch_size)
    assert val_bs >= 1, val_bs
    val_loader_workers = int(args.val_loader_workers)
    assert val_loader_workers >= 1, val_loader_workers
    train_loader_workers = int(args.train_loader_workers)
    assert train_loader_workers >= 1, train_loader_workers
    if use_validation:
        val_paths_for_precompute = val_paths
    if feature_importance:
        assert feature_importance_val_episodes <= n_val_ep, (
            feature_importance_val_episodes,
            n_val_ep,
        )
        val_paths_for_precompute = sorted(val_paths, key=lambda p: p.name)[
            :feature_importance_val_episodes
        ]
        assert len(val_paths_for_precompute) == feature_importance_val_episodes, (
            len(val_paths_for_precompute),
            feature_importance_val_episodes,
        )
    if use_validation:
        n_val_ep_for_precompute = len(val_paths_for_precompute)
        assert val_loader_workers <= n_val_ep_for_precompute, (
            val_loader_workers,
            n_val_ep_for_precompute,
        )
    assert train_loader_workers <= n_train_ep, (train_loader_workers, n_train_ep)
    assert train_loader_workers <= int(args.pool_episodes), (
        train_loader_workers,
        int(args.pool_episodes),
    )
    if use_validation:
        print(
            f"BC: train_episodes={n_train_ep} val_episodes={n_val_ep} batch_size={bs} "
            f"val_batch_size={val_bs} pool_episodes={int(args.pool_episodes)} "
            f"train_loader_workers={train_loader_workers} val_loader_workers={val_loader_workers}",
            flush=True,
        )
    else:
        print(
            f"BC: train_episodes={n_train_ep} val_episodes=0 batch_size={bs} "
            f"pool_episodes={int(args.pool_episodes)} train_loader_workers={train_loader_workers}",
            flush=True,
        )
    if feature_importance:
        print(
            f"BC feature importance: val_episodes={feature_importance_val_episodes}/{n_val_ep} "
            f"(sorted holdout prefix)",
            flush=True,
        )
    if use_validation:
        print(
            "BC: precomputing full validation on CPU (every val episode; large gzip .pt can still take a bit)",
            flush=True,
        )
        t_val_pre = time.monotonic()
        val_batches_cpu = _bc_precompute_validation_batches_parallel(
            val_paths_for_precompute,
            batch_size=val_bs,
            workers=val_loader_workers,
        )
        print(
            f"BC: validation precompute done batches={len(val_batches_cpu)} "
            f"elapsed_s={time.monotonic() - t_val_pre:.2f}",
            flush=True,
        )
        assert len(val_batches_cpu) >= 1, (
            "BC validation yielded no full batches",
            len(val_batches_cpu),
            val_bs,
        )

    if feature_importance:
        _bc_run_feature_importance(
            args=args,
            val_batches_cpu=val_batches_cpu,
            device=device,
        )
        return

    train_iter_ds = BcEpisodePoolIterable(
        train_paths,
        pool_episodes=int(args.pool_episodes),
        batch_size=bs,
        infinite=True,
    )
    loader = DataLoader(
        train_iter_ds,
        batch_size=None,
        num_workers=train_loader_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    if args.resume_checkpoint is not None:
        assert args.resume_checkpoint.is_file(), args.resume_checkpoint
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _PY_ROOT.parent / "outputs" / "supervised" / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}", flush=True)

    wandb_project = (
        args.wandb_project
        if args.wandb_project is not None
        else ImpalaTrainingConfig().wandb_project
    )
    use_wandb = not bool(args.no_wandb)
    if use_wandb:
        run_name = args.wandb_run_name or f"bc_rl_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            tags=["bc", "rl_policy_head"],
            config={
                "data_dir": str(args.data_dir),
                "num_train_episodes": int(n_train_ep),
                "num_val_episodes": int(n_val_ep),
                "pool_episodes": int(args.pool_episodes),
                "train_loader_workers": int(args.train_loader_workers),
                "use_validation": bool(use_validation),
                "n_validation_episodes": int(N_VALIDATION_EPISODES) if use_validation else 0,
                "validation_episodes_file": (
                    str(args.data_dir / _VALIDATION_EPISODES_FILENAME)
                    if use_validation
                    else None
                ),
                "val_every_batches": int(args.val_every_batches),
                "batch_size": int(args.batch_size),
                "val_batch_size": int(val_bs),
                "lr": float(args.lr),
                "warmup_training_samples": int(warmup_training_samples),
                "max_train_samples": int(max_train_samples),
                "gpu_id": int(args.gpu_id),
                "rl_checkpoint": str(args.rl_checkpoint) if args.rl_checkpoint else None,
                "resume_checkpoint": str(args.resume_checkpoint)
                if args.resume_checkpoint
                else None,
                "run_dir": str(run_dir),
                "out": str(args.out) if args.out is not None else str(run_dir / "last.pt"),
                "per_planet_move_classes": int(ORBIT_PER_PLANET_MOVE_CLASSES),
                "optimizer_name": _bc_optim_cfg.optimizer_name,
                "optimizer_kwargs": {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in _bc_optimizer_kwargs.items()
                },
            },
        )
    try:
        resume_ckpt: dict[str, Any] | None = None
        if args.resume_checkpoint is not None:
            resume_ckpt = torch.load(
                str(args.resume_checkpoint), map_location="cpu", weights_only=False
            )
            assert isinstance(resume_ckpt, dict), type(resume_ckpt)
            assert "model_state_dict" in resume_ckpt, sorted(resume_ckpt.keys())

        rl_ckpt: dict[str, Any] | None = None
        if resume_ckpt is not None:
            checkpoint_model_config = _bc_checkpoint_model_config(resume_ckpt)
        elif args.rl_checkpoint is not None:
            assert args.rl_checkpoint.is_file(), args.rl_checkpoint
            rl_ckpt = torch.load(str(args.rl_checkpoint), map_location="cpu", weights_only=False)
            assert isinstance(rl_ckpt, dict) and "model_state_dict" in rl_ckpt
            checkpoint_model_config = _bc_checkpoint_model_config(rl_ckpt)
        else:
            checkpoint_model_config = _bc_current_model_config()

        model = ImpalaOrbitModel(**bc_model_init_kwargs(checkpoint_model_config)).to(device)
        if resume_ckpt is not None:
            sd = _strip_torch_compile_orig_mod_prefix(resume_ckpt["model_state_dict"])
            assert isinstance(sd, dict), type(sd)
            _load_bc_resume_model_state_dict(model, sd)
        elif args.rl_checkpoint is not None:
            assert rl_ckpt is not None
            sd = _strip_torch_compile_orig_mod_prefix(rl_ckpt["model_state_dict"])
            assert isinstance(sd, dict)
            _load_bc_resume_model_state_dict(model, sd)

        n_param = sum(int(p.numel()) for p in model.parameters())
        n_trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        print(
            f"model_parameters={n_param} trainable_parameters={n_trainable}",
            flush=True,
        )

        model = compile_impala_model_for_rl(model)

        model.train()

        if use_validation:
            val_model = ImpalaOrbitModel(**bc_model_init_kwargs(checkpoint_model_config)).to(device)
            val_model = compile_impala_model_for_rl(val_model)
            val_model.load_state_dict(model.state_dict(), strict=True)
            val_model.eval()

        val_every_batches = int(args.val_every_batches)
        assert val_every_batches >= 1, val_every_batches
        profile_every_batches = int(BC_PROFILE_EVERY_BATCHES)
        assert profile_every_batches >= 1, profile_every_batches

        opt = build_optimizer_from_config(
            {
                "optimizer_name": _bc_optim_cfg.optimizer_name,
                "optimizer_kwargs": _bc_optimizer_kwargs,
            },
            model,
        )[0]
        n = ORBIT_MAX_PLANETS
        pc = int(ORBIT_PER_PLANET_MOVE_CLASSES)
        nb_bc = int(ORBIT_MOVE_CLASSES_PER_TARGET)
        assert pc == n * nb_bc, (pc, n, nb_bc)
        train_batch_idx = 0
        train_samples_seen = 0
        if resume_ckpt is not None:
            for k in (
                "train_batch",
                "train_samples",
            ):
                assert k in resume_ckpt, (k, sorted(resume_ckpt.keys()))
            train_batch_idx = int(resume_ckpt["train_batch"])
            train_samples_seen = int(resume_ckpt["train_samples"])
            if "optimizer_state_dict" in resume_ckpt:
                osd = resume_ckpt["optimizer_state_dict"]
                assert isinstance(osd, dict), type(osd)
                opt.load_state_dict(osd)
                n_dropped_opt_states = _bc_drop_incompatible_optimizer_state(opt)
                if n_dropped_opt_states > 0:
                    print(
                        "BC resume: dropped incompatible optimizer state for "
                        f"{n_dropped_opt_states} parameter(s).",
                        flush=True,
                    )
            else:
                print(
                    "BC resume: checkpoint has no optimizer_state_dict; optimizer reinitialized.",
                    flush=True,
                )
            print(
                f"BC resume: train_batch_idx={train_batch_idx} train_samples_seen={train_samples_seen}",
                flush=True,
            )
        wall_prof = WallTreeProfiler()
        grad_norm_ema = 0.0
        train_metric_smoothed: dict[str, RollingAverage] = {}

        def write_checkpoint(checkpoint_after_train_samples: int) -> None:
            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "model_config": checkpoint_model_config,
                "optimizer_state_dict": opt.state_dict(),
                "train_batch": int(train_batch_idx),
                "train_samples": int(checkpoint_after_train_samples),
            }
            ckpt_batch = run_dir / f"samples_{int(checkpoint_after_train_samples):012d}.pt"
            torch.save(ckpt_payload, ckpt_batch)
            print(f"wrote {ckpt_batch}")
            last_pt = run_dir / "last.pt"
            torch.save(ckpt_payload, last_pt)
            print(f"wrote {last_pt}")
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt_payload, args.out)
                print(f"wrote {args.out}")

        def run_validation(validation_after_train_samples: int) -> None:
            assert use_validation
            val_model.load_state_dict(model.state_dict(), strict=True)
            val_model.eval()
            val_metrics = _bc_validation_metrics_flat(
                val_model,
                val_batches_cpu,
                device=device,
                n_planets=n,
                n_grid_ship_buckets=nb_bc,
            )
            v_mean_combined_loss = val_metrics["loss_combined"]
            v_mean_final_policy_loss = val_metrics["loss_final_policy"]
            v_final_policy_topk_mean = {
                int(ki): val_metrics[f"final_policy_top{int(ki)}"]
                for ki in _BC_FINAL_TOPK_LEVELS
            }
            v_final_msg = _final_policy_log_message(
                final_policy_topk_mean=v_final_policy_topk_mean,
                final_policy_entropy=val_metrics["final_policy_entropy"],
            )
            print(
                f"train_samples {validation_after_train_samples} val loss_combined={v_mean_combined_loss:.6f} "
                f"loss_final_policy={v_mean_final_policy_loss:.6f} "
                f"{v_final_msg} "
                f"target_action_invalid_count={int(val_metrics['target_action_invalid_count'])} "
                f"target_action_invalid_active_choice_count={int(val_metrics['target_action_invalid_active_choice_count'])} "
                f"target_action_invalid_src_invalid_planet_count={int(val_metrics['target_action_invalid_src_invalid_planet_count'])} "
                f"target_action_invalid_no_choice_count={int(val_metrics['target_action_invalid_no_choice_count'])} "
                f"target_self_send_non_noop_active_choice_count={int(val_metrics['target_self_send_non_noop_active_choice_count'])}"
            )
            if use_wandb:
                clean_v = _bc_policy_metric_wandb_payload(
                    _BC_VAL_WB_TAG,
                    combined_loss=v_mean_combined_loss,
                    final_policy_loss=v_mean_final_policy_loss,
                    final_policy_topk_mean=v_final_policy_topk_mean,
                    final_policy_entropy=val_metrics["final_policy_entropy"],
                )
                if clean_v:
                    wandb.log(clean_v, step=int(validation_after_train_samples))

            write_checkpoint(validation_after_train_samples)

        train_prefetcher = CudaBcTargetPrefetcher(
            loader,
            device=device,
            n_planets=n,
            n_grid_ship_buckets=nb_bc,
        )
        for targets in train_prefetcher:
            with profiler_span(wall_prof, "batch"):
                bc_step_lr = _bc_lr_for_trailing_train_samples(
                    train_samples_seen + bs,
                    lr_max=float(args.lr),
                    warmup_samples=warmup_training_samples,
                    max_train_samples=max_train_samples,
                )
                _set_bc_optimizer_lr(opt, bc_step_lr)
                with profiler_span(wall_prof, "zero_grad"):
                    opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with profiler_span(wall_prof, "forward"):
                        model.set_is_sample(False)
                        out = model(
                            {"obs_LEARN_INFER": targets.batch_obs},
                            output_full_policy_log_probs=False,
                            include_policy_logits_pre_action_mask=True,
                            include_final_policy_logits=True,
                            wall_profiler=wall_prof,
                        )
                out = _bc_floating_outputs_to_float32(out)
                with profiler_span(wall_prof, "loss"):
                    metrics = _bc_policy_loss_metrics_from_output(
                        out,
                        targets,
                        n_planets=n,
                        n_grid_ship_buckets=nb_bc,
                    )
                with profiler_span(wall_prof, "backward"):
                    metrics.combined_loss.backward()
                with profiler_span(wall_prof, "clip_grad_norm"):
                    with torch.no_grad():
                        clip_grad_value = min(grad_norm_ema * 1.5, 10000.0)
                        if clip_grad_value == 0.0:
                            clip_grad_value = 10.0
                        clip_grad_value = 10000
                        grad_total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_value
                        )
                        grad_total_norm = float(grad_total_norm_tensor.item())
                        if math.isfinite(grad_total_norm):
                            a = _BC_GRAD_NORM_EMA_ALPHA
                            grad_norm_ema = (1.0 - a) * grad_norm_ema + a * grad_total_norm
                with profiler_span(wall_prof, "optimizer_step"):
                    opt.step()

            train_batch_idx += 1
            assert int(metrics.batch_size) == bs, (metrics.batch_size, bs)
            train_samples_seen += bs
            target_action_invalid_counts_py = [
                int(x) for x in targets.target_action_invalid_counts.detach().cpu().tolist()
            ]
            train_final_policy_topk_mean = _final_topk_means_from_counts(
                {
                    ki: topk.detach().cpu()
                    for ki, topk in metrics.final_policy_topk.items()
                },
            )
            train_final_msg = _final_policy_log_message(
                final_policy_topk_mean=train_final_policy_topk_mean,
                final_policy_entropy=_count_mean(
                    metrics.final_policy_entropy.detach().cpu()
                ),
            )
            print(
                f"batch {train_batch_idx} loss_combined={metrics.combined_loss.detach().item():.6f} "
                f"loss_final_policy={metrics.final_policy_loss.detach().item():.6f} "
                f"{train_final_msg} "
                f"target_action_invalid_count={target_action_invalid_counts_py[0]} "
                f"target_action_invalid_active_choice_count={target_action_invalid_counts_py[1]} "
                f"target_action_invalid_src_invalid_planet_count={target_action_invalid_counts_py[2]} "
                f"target_action_invalid_no_choice_count={target_action_invalid_counts_py[3]} "
                f"target_self_send_non_noop_active_choice_count={target_action_invalid_counts_py[4]}"
            )
            if use_wandb:
                wb_s = _BC_TRAIN_WB_TAG
                train_batch_stats = _bc_policy_metric_wandb_payload(
                    wb_s,
                    combined_loss=float(metrics.combined_loss.detach().item()),
                    final_policy_loss=float(metrics.final_policy_loss.detach().item()),
                    final_policy_topk_mean=train_final_policy_topk_mean,
                    final_policy_entropy=_count_mean(
                        metrics.final_policy_entropy.detach().cpu()
                    ),
                )
                if math.isfinite(grad_total_norm):
                    train_batch_stats[f"Everything.grad_total_norm_{wb_s}"] = grad_total_norm
                if math.isfinite(clip_grad_value):
                    train_batch_stats[f"Everything.grad_clip_threshold_{wb_s}"] = (
                        clip_grad_value
                    )
                if math.isfinite(grad_norm_ema):
                    train_batch_stats[f"Everything.grad_norm_ema_{wb_s}"] = grad_norm_ema
                train_batch_stats[f"Everything.bc_lr_{wb_s}"] = float(bc_step_lr)
                _bc_train_metric_smoothed_update(train_metric_smoothed, train_batch_stats)
                clean = _bc_train_smoothed_wandb_payload(
                    train_metric_smoothed,
                    train_batch_stats,
                )
                if clean:
                    wandb.log(
                        clean,
                        step=int(train_samples_seen),
                        commit=(train_batch_idx % val_every_batches) != 0,
                    )

            if train_batch_idx % profile_every_batches == 0:
                wall_prof.summary_stdout(
                    f"bc_train_batches_{train_batch_idx - profile_every_batches + 1}_{train_batch_idx}",
                    line_prefix="BC_WALL_TREE ",
                )
                wall_prof = WallTreeProfiler()

            if train_batch_idx % val_every_batches == 0:
                if use_validation:
                    run_validation(train_samples_seen)
                else:
                    write_checkpoint(train_samples_seen)

            if train_samples_seen >= max_train_samples:
                if train_batch_idx % val_every_batches != 0:
                    if use_validation:
                        run_validation(train_samples_seen)
                    else:
                        write_checkpoint(train_samples_seen)
                break
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
