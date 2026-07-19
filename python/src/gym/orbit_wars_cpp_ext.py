from __future__ import annotations

import fcntl
import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load

_LOG = logging.getLogger(__name__)
_PYTHON_PKG_ROOT = Path(__file__).resolve().parents[2]
_ORBIT_WARS_CPP_DIR = _PYTHON_PKG_ROOT / "cpp" / "orbit_wars"
# Must match python/cpp/orbit_wars/setup.py CppExtension sources (split TU layout).
_JIT_SOURCES = [
    _ORBIT_WARS_CPP_DIR / "bindings.cpp",
    _ORBIT_WARS_CPP_DIR / "io.cpp",
    _ORBIT_WARS_CPP_DIR / "library.cpp",
    _ORBIT_WARS_CPP_DIR / "simulation.cpp",
    _ORBIT_WARS_CPP_DIR / "masks.cpp",
    _ORBIT_WARS_CPP_DIR / "kaggle_integration.cpp",
    _ORBIT_WARS_CPP_DIR / "honest_shared_intercept.cpp",
    _ORBIT_WARS_CPP_DIR / "honest_shared_features.cpp",
    _ORBIT_WARS_CPP_DIR / "cpp_env_v2" / "cpp_env_static_cache_v2.cpp",
    _ORBIT_WARS_CPP_DIR / "cpp_env_v2" / "cpp_env_static_cache_v2_mask.cpp",
    _ORBIT_WARS_CPP_DIR / "cpp_env_v2" / "cpp_env_static_cache_v2_features.cpp",
    _ORBIT_WARS_CPP_DIR / "cpp_env_v2" / "cpp_env_live_v2.cpp",
]

_EXT_NAME = "orbit_wars_cpp"
_THIS_DIR = Path(__file__).resolve().parent
_VERBOSE_BUILD = os.environ.get("ORBIT_WARS_CPP_VERBOSE", "0") == "1"
_FORCE_REBUILD = os.environ.get("ORBIT_WARS_CPP_FORCE_REBUILD", "0") == "1"

os.environ["CCACHE_DISABLE"] = "1"

_prebuilt_exts = [".so", ".pyd", ".dylib"]
_prebuilt_paths = []
for _suffix in _prebuilt_exts:
    _prebuilt_paths.extend(sorted(_THIS_DIR.glob(f"{_EXT_NAME}*{_suffix}")))

if _prebuilt_paths:
    # Kaggle submission path: load prebuilt native module bundled in the archive.
    _prebuilt = _prebuilt_paths[0]
    _LOG.info("orbit_wars_cpp loading prebuilt native module: %s", _prebuilt)
    _spec = importlib.util.spec_from_file_location(_EXT_NAME, str(_prebuilt))
    assert _spec is not None and _spec.loader is not None, _prebuilt
    orbit_wars_cpp = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(orbit_wars_cpp)
    except ImportError as e:
        meta_path = _THIS_DIR / "orbit_wars_cpp_build_meta.txt"
        meta = meta_path.read_text() if meta_path.is_file() else "<missing build metadata>"
        raise ImportError(
            f"{e}\n"
            f"orbit_wars_cpp runtime python={sys.version}\n"
            f"orbit_wars_cpp runtime torch={torch.__version__} cuda={torch.version.cuda}\n"
            f"orbit_wars_cpp build metadata:\n{meta}"
        ) from e
else:
    # Dev/training path: JIT-build from C++ source in workspace.
    for _src in _JIT_SOURCES:
        assert _src.is_file(), _src
    _get_build_directory = cpp_extension._get_build_directory
    _BUILD_DIR = _get_build_directory(_EXT_NAME, verbose=_VERBOSE_BUILD)
    Path(_BUILD_DIR).mkdir(parents=True, exist_ok=True)
    _LOCK_PATH = Path(_BUILD_DIR).parent / ".orbit_wars_cpp_jit_load.lock"
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG.info(
        "orbit_wars_cpp building JIT C++ extension (sources_dir=%s n_sources=%d build_dir=%s pid=%s verbose=%s force_rebuild=%s)",
        _ORBIT_WARS_CPP_DIR,
        len(_JIT_SOURCES),
        _BUILD_DIR,
        os.getpid(),
        _VERBOSE_BUILD,
        _FORCE_REBUILD,
    )
    _lock_fp = open(_LOCK_PATH, "a+", buffering=1)
    fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX)
    try:
        orbit_wars_cpp = load(
            name=_EXT_NAME,
            sources=[str(p) for p in _JIT_SOURCES],
            extra_cflags=[
                "-std=c++17",
                "-O3",
                "-g",
                "-fno-omit-frame-pointer",
                "-DNDEBUG",
                f"-I{str(_ORBIT_WARS_CPP_DIR)}",
            ],
            with_cuda=False,
            verbose=_VERBOSE_BUILD,
            build_directory=_BUILD_DIR,
        )
    finally:
        fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_UN)

assert hasattr(orbit_wars_cpp, "orbit_wars_format_double_for_reset_trace")
assert hasattr(orbit_wars_cpp, "orbit_wars_format_double_two_decimals_for_reset_trace")
assert hasattr(orbit_wars_cpp, "orbit_wars_policy_obs_edge_distance")
assert hasattr(orbit_wars_cpp, "orbit_wars_fill_inactive_policy_action_noops")
assert hasattr(orbit_wars_cpp, "orbit_wars_fleet_speed")
assert hasattr(orbit_wars_cpp, "CppEnvStaticCacheV2")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "reset")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "noop_trajectory_length")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "noop_trajectory_planets_tensor")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "noop_trajectory_planets_row_tensor")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "update_comet_in_noop_cache")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_action_mask_limited")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_action_mask_all_geometry")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_action_mask_full_cache_warmup_one")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_action_mask_full_cache_prune_before")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_action_mask_full_cache_warmup_stats")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "send_all_from_external")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "fill_policy_obs_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "fleet_arrivals_from_rows")
assert hasattr(
    orbit_wars_cpp.CppEnvStaticCacheV2,
    "fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows",
)
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "fleet_arrival_features_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "fleet_arrivals_resolution_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "fleet_hit_traces_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_hit_kind_last")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_hit_slot_last")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_hit_steps_last")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_intercept_fail_reason_last")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_send_ships_last")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_dir_last")
assert hasattr(orbit_wars_cpp.CppEnvStaticCacheV2, "honest_shared_angle_last_for_action_class")
assert hasattr(orbit_wars_cpp, "CppEnvLiveV2")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "reset")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "update_comets_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "step")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "reset_trace_get")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "step_trace_get")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "set_wall_profile_enabled")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "wall_profile_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "orbit_episode_terminal")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_ship_total_int_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "player_alive_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "planet_count_int_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "production_sum_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "game_result_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_delta_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "planets_delta_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "production_delta_for_owner")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "step_metric_tensors")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "tape_kaggle_planets_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "tape_kaggle_fleets_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "angular_velocity")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "ship_speed")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "episode_step")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "kaggle_observation_step")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_send_all_hit_mask")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_hit_kind_last")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_hit_slot_last")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_hit_steps_last")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_intercept_fail_reason_last")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_send_ships_last")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_dir_last")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_angle_last_for_action_class")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_angle")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_angle_or_nan")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "honest_shared_intercept_trace")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrivals_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrivals_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrival_features_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrival_features_from_rows")
assert hasattr(
    orbit_wars_cpp.CppEnvLiveV2,
    "fleet_arrival_features_and_fill_future_resolution_planet_features_from_state",
)
assert hasattr(
    orbit_wars_cpp.CppEnvLiveV2,
    "fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows",
)
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fill_policy_obs_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fill_future_resolution_planet_features_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fill_future_resolution_planet_features_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_takeover_cost_features_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrivals_resolution")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrivals_resolution_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_arrivals_resolution_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_hit_traces_from_state")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "fleet_hit_traces_from_rows")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "comet_mask_inputs_py")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "noop_trajectory_length")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "noop_trajectory_planets_tensor")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "noop_trajectory_planets_row_tensor")
assert hasattr(orbit_wars_cpp.CppEnvLiveV2, "assert_planets_match_noop_cache")
_LOG.info(
    "orbit_wars_cpp native module loaded (policy obs fill binding visible); pid=%s",
    os.getpid(),
)
