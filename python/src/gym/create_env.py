from __future__ import annotations

import copy
import multiprocessing as mp
from types import SimpleNamespace
from typing import Any

import torch

from .env_batch_wrapper import EnvBatchWrapper
from .logging_wrapper import LoggingWrapper
from .obs_wrapper import ObsWrapper
from .orbit_wars_env import OrbitWarsEnv
from .wall_tree_profiler import WallTreeProfiler
from .padding_wrapper import OrbitPaddingWrapper
from .remap_and_filter_wrapper import RemapAndFilterWrapper
from .reward_wrapper import RewardWrapper
def _namespace_tree_to_plain(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_tree_to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_namespace_tree_to_plain(x) for x in obj]
    return obj


def _create_inner_env(
    flags: Any,
    *,
    device: str | torch.device,
    visualize: bool,
    visualize_sim_env: bool,
    visualization_queue: mp.Queue | None,
    record_tape: bool,
    cpp_env_obs_full: bool,
    cpp_env_obs_validate: bool,
    orbit_instance_id: int,
    wall_profiler: WallTreeProfiler | None,
) -> Any:
    num_agents = int(flags.orbit_num_agents)
    assert isinstance(flags.orbit_configuration, SimpleNamespace)
    configuration = _namespace_tree_to_plain(flags.orbit_configuration)
    env = OrbitWarsEnv(
        num_agents=num_agents,
        configuration=configuration,
        orbit_instance_id=orbit_instance_id,
        debug=False,
        visualize=visualize,
        visualize_sim_env=visualize_sim_env,
        visualization_queue=visualization_queue,
        record_tape=record_tape,
        cpp_env_obs_full=cpp_env_obs_full,
        cpp_env_obs_validate=cpp_env_obs_validate,
        flags=flags,
        wall_profiler=wall_profiler,
    )
    env = OrbitPaddingWrapper(env, flags, wall_profiler=wall_profiler)
    env = RewardWrapper(env, flags, wall_profiler=wall_profiler)
    env = ObsWrapper(env, flags, wall_profiler=wall_profiler)
    env = LoggingWrapper(env, flags, wall_profiler=wall_profiler)
    env = RemapAndFilterWrapper(env, flags, wall_profiler=wall_profiler)
    _ = device
    return env


def _flags_with_orbit_num_agents(flags: Any, num_agents: int) -> Any:
    out = copy.copy(flags)
    out.orbit_num_agents = int(num_agents)
    return out


def create_env(
    flags: Any,
    *,
    device: str | torch.device = "cpu",
    visualize: bool = False,
    visualize_sim_env: bool = False,
    visualization_queue: mp.Queue | None = None,
    record_tape: bool = False,
    cpp_env_obs_full: bool = False,
    cpp_env_obs_validate: bool = False,
    wall_profiler: WallTreeProfiler | None = None,
) -> Any:
    n = int(flags.n_actor_envs)
    assert n >= 1
    envs_by_num_agents: dict[int, list[Any]] = {2: [], 4: []}
    for num_agents in (2, 4):
        inner_flags = _flags_with_orbit_num_agents(flags, num_agents)
        envs_by_num_agents[int(num_agents)] = [
            _create_inner_env(
                inner_flags,
                device=device,
                visualize=visualize,
                visualize_sim_env=visualize_sim_env,
                visualization_queue=visualization_queue,
                record_tape=record_tape,
                cpp_env_obs_full=cpp_env_obs_full,
                cpp_env_obs_validate=cpp_env_obs_validate,
                orbit_instance_id=ei,
                wall_profiler=wall_profiler,
            )
            for ei in range(n)
        ]
    env = EnvBatchWrapper(
        envs_by_num_agents,
        orbit_num_agents=int(flags.orbit_num_agents),
        flags=flags,
        wall_profiler=wall_profiler,
    )

    return env # dont change that line
