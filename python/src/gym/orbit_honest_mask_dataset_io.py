from __future__ import annotations

import gzip
from pathlib import Path
import zlib

import torch

from .obs_wrapper import (
    ORBIT_PER_PLANET_HIT_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_ARRIVAL_HORIZON,
)
from .wall_tree_profiler import WallTreeProfiler, profiler_span


class OrbitHonestMaskEpisodeArchiveError(Exception):
    pass


def _assert_honest_mask_step_tensor(mask: torch.Tensor, *, path: Path) -> torch.Tensor:
    assert isinstance(mask, torch.Tensor), (path, type(mask))
    assert mask.dtype == torch.int8, (path, mask.dtype)
    assert not mask.is_cuda, path
    assert tuple(mask.shape) == (
        ORBIT_PLANET_ACTION_SLOTS,
        ORBIT_PER_PLANET_HIT_CLASSES,
    ), (path, tuple(mask.shape))
    return mask.contiguous()


def load_orbit_honest_mask_episode_from_pt_path(
    path: Path,
    *,
    wall_profiler: WallTreeProfiler | None = None,
) -> dict[str, object]:
    path = path.resolve()
    try:
        with gzip.open(path, "rb") as f:
            payload = torch.load(f, map_location="cpu", weights_only=False)
    except (gzip.BadGzipFile, EOFError, RuntimeError, zlib.error) as exc:
        raise OrbitHonestMaskEpisodeArchiveError(path) from exc
    assert isinstance(payload, dict), type(payload)
    assert payload.get("honest_shared_action_mask_encoding") == "int8_tensor_v1", path
    assert "orbit_instance_id" in payload, path
    assert "episode_index" in payload, path
    assert "episode_seed" in payload, path
    assert "steps" in payload, path
    orbit_instance_id = int(payload["orbit_instance_id"])
    episode_index = int(payload["episode_index"])
    episode_seed = int(payload["episode_seed"])
    steps = payload["steps"]
    assert isinstance(steps, list), type(steps)
    step_masks: list[torch.Tensor] = []
    planet_position_hashes: list[str] = []
    for step in steps:
        assert isinstance(step, dict), type(step)
        assert "honest_shared_action_mask" in step, path
        assert "planet_position_hash_sha256" in step, path
        mask = _assert_honest_mask_step_tensor(step["honest_shared_action_mask"], path=path)
        planet_position_hash = step["planet_position_hash_sha256"]
        assert isinstance(planet_position_hash, str), type(planet_position_hash)
        step_masks.append(mask)
        planet_position_hashes.append(planet_position_hash)
    assert len(step_masks) > 0, f"dataset episode has no steps: {path}"
    assert len(planet_position_hashes) == len(step_masks), path
    with profiler_span(wall_profiler, "honest_mask_dataset_stack"):
        episode_tensor = torch.stack(step_masks, dim=0).contiguous()
    assert episode_tensor.dtype == torch.int8, (path, episode_tensor.dtype)
    with profiler_span(wall_profiler, "honest_mask_dataset_horizon_clip"):
        episode_tensor.masked_fill_(episode_tensor > int(ORBIT_PLANET_ARRIVAL_HORIZON), 0)
    return {
        "orbit_instance_id": orbit_instance_id,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "honest_shared_action_mask": episode_tensor,
        "planet_position_hash_sha256": tuple(planet_position_hashes),
    }
