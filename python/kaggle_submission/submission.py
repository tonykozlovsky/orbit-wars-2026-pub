# pyright: reportMissingImports=false
# ./kaggle_submission/package_kaggle_submission.sh orbit_wars_submit.tar.gz /path/to/2p/checkpoint.pt /path/to/4p/checkpoint.pt
"""
Kaggle ``orbit_wars`` submission entry: ``agent(observation, configuration)`` → list of moves.

Observation / action schema (official env):
  https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/orbit_wars/orbit_wars.json

Local smoke test::

  from kaggle_environments import make
  from kaggle_submission.submission import agent

  env = make("orbit_wars", debug=True)
  env.run([agent, agent])

Weights: set env ``ORBIT_IMPALA_CHECKPOINT_2P`` and ``ORBIT_IMPALA_CHECKPOINT_4P`` to
``.pt`` files from training containing ``model_state_dict`` (see
``batch_and_learn._build_checkpoint_dict_at_steps``), or place the same files next to this
module as ``checkpoint_weights_2p.pt`` and ``checkpoint_weights_4p.pt``.
"""
from __future__ import annotations

import copy
import faulthandler
import importlib.util
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_LOCAL_SIMULATION_RAW = os.environ.get("LOCAL_SIMULATION", "").strip()
assert _LOCAL_SIMULATION_RAW in ("", "1"), _LOCAL_SIMULATION_RAW
_LOCAL_SIMULATION = _LOCAL_SIMULATION_RAW == "1"
_IMPALA_PROJECT_ROOT_RAW = os.environ.get(
    "IMPALA_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[2]),
).strip()
assert _IMPALA_PROJECT_ROOT_RAW, "IMPALA_PROJECT_ROOT must be non-empty"
_IMPALA_PROJECT_ROOT = Path(_IMPALA_PROJECT_ROOT_RAW).expanduser().resolve()

_SUBMISSION_OMP_THREADS = 1
_SUBMISSION_CPU_THREADS = 1
_SUBMISSION_INTEROP_THREADS = 1

if not _LOCAL_SIMULATION:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(_SUBMISSION_OMP_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_SUBMISSION_OMP_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_SUBMISSION_OMP_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(_SUBMISSION_OMP_THREADS)

import torch

torch.set_num_threads(_SUBMISSION_CPU_THREADS)

if torch.get_num_interop_threads() != _SUBMISSION_INTEROP_THREADS:
    torch.set_num_interop_threads(_SUBMISSION_INTEROP_THREADS)

if _LOCAL_SIMULATION:
    assert torch.cuda.is_available(), "LOCAL_SIMULATION=1 requires CUDA"

_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from src.gym.orbit_kaggle_cpp_cache import (
    OrbitKaggleCppObservationCache,
    planet_rows_tensor_from_plain_planets_float64,
)
from src.gym.orbit_wars_env import (
    attach_zeros_action_taken_index_on_seats,
    assert_orbit_action,
    kaggle_moves_for_seat_from_classes_honest_angle,
    orbit_observation_to_plain,
)
from src.gym.obs_wrapper import (
    ORBIT_EDGE_FEATURES,
    ORBIT_MAX_PLANETS,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ARRIVAL_HORIZON,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PAIRWISE_COUNT,
    ORBIT_PLANET_TEMPORAL_FEATURES,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_policy_slot_for_compact_agent,
)
from src.models.models import ImpalaOrbitModel, impala_orbit_model_init_kwargs_from_flags
from src.gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from kaggle_submission.submission_runtime_config import (
    SUBMISSION_IS_SAMPLE,
    SUBMISSION_SHUFFLE_IDENTITY_IDS,
)

_SUBMISSION_DIR = Path(__file__).resolve().parent
_AOTI_RUNNER_CPP_EXT_NAME = "aoti_policy_runner_cpp"
_POLICY_INPUT_DUMP_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
)
_POLICY_INPUT_DUMP_FORMAT = "orbit_policy_network_inputs_dump_v1"
_SUBMISSION_DUMP_INPUTS = False
_SUBMISSION_DUMP_INPUTS_PATH = _IMPALA_PROJECT_ROOT / "replays/replay_submission_inputs.pt"
_SUBMISSION_OVERAGE_DUMMY_USED_FRACTION = 0.90

_SUBMISSION_FAIL_ENABLED_RAW = os.environ.get("ORBIT_SUBMISSION_FAIL_ENABLED", "0").strip()
assert _SUBMISSION_FAIL_ENABLED_RAW in ("0", "1"), _SUBMISSION_FAIL_ENABLED_RAW
_SUBMISSION_FAIL_ENABLED = _SUBMISSION_FAIL_ENABLED_RAW == "1"


_SUBMISSION_FORCE_NOOP_RATE_RAW = os.environ.get("FORCE_NOOP_RATE", "0").strip()
assert _SUBMISSION_FORCE_NOOP_RATE_RAW, "FORCE_NOOP_RATE must be non-empty"
_SUBMISSION_FORCE_NOOP_RATE = float(_SUBMISSION_FORCE_NOOP_RATE_RAW)
assert 0.0 <= _SUBMISSION_FORCE_NOOP_RATE <= 1.0, _SUBMISSION_FORCE_NOOP_RATE

_SUBMISSION_FORCE_NOOP_GAME_RATE_RAW = os.environ.get("FORCE_NOOP_GAME_RATE", "0").strip()
assert _SUBMISSION_FORCE_NOOP_GAME_RATE_RAW, "FORCE_NOOP_GAME_RATE must be non-empty"
_SUBMISSION_FORCE_NOOP_GAME_RATE = float(_SUBMISSION_FORCE_NOOP_GAME_RATE_RAW)
assert 0.0 <= _SUBMISSION_FORCE_NOOP_GAME_RATE <= 1.0, _SUBMISSION_FORCE_NOOP_GAME_RATE


_SUBMISSION_PERF_TEST_FAIL_STEPS_BY_SEAT_INDEX = (100, 100, 100, 100)
_SUBMISSION_FAIL_BEFORE_MODEL_LOAD = False
_SUBMISSION_FAIL_BEFORE_FORWARD = False
_SUBMISSION_REQUIRED_NUM_AGENTS = 2
_SUBMISSION_AOTI_LOAD_WATCHDOG_SECONDS = 30.0
_SUBMISSION_HONEST_FULL_CACHE_WARMUP_ENABLED = False
_AOTI_RUNNER_CPP_MODULE: Any | None = None
_SUBMISSION_INPUT_DUMP_SAMPLES: list[tuple[torch.Tensor, ...]] = []


def _submission_force_noop_game_enabled() -> bool:
    if _SUBMISSION_FORCE_NOOP_RATE <= 0.0:
        return False
    if _SUBMISSION_FORCE_NOOP_GAME_RATE <= 0.0:
        return False
    draw = float(secrets.randbelow(1_000_000_000)) / 1_000_000_000.0
    force_noop_game_enabled = draw < _SUBMISSION_FORCE_NOOP_GAME_RATE
    print(
        "SUBMISSION_FORCE_NOOP_GAME_STATUS "
        f"game_rate={_SUBMISSION_FORCE_NOOP_GAME_RATE:.6f} "
        f"draw={draw:.6f} "
        f"force_noop_game_enabled={int(force_noop_game_enabled)}",
        file=sys.stderr,
        flush=True,
    )
    return force_noop_game_enabled


def _submission_perf_test_fail_step_for_seat_index(seat_index: int) -> int:
    assert 0 <= seat_index < len(_SUBMISSION_PERF_TEST_FAIL_STEPS_BY_SEAT_INDEX), seat_index
    return _SUBMISSION_PERF_TEST_FAIL_STEPS_BY_SEAT_INDEX[seat_index]


def _submission_force_noop(result: OrbitSubmissionInferenceResult) -> bool:
    if not result.force_noop_game_enabled:
        return False
    if _SUBMISSION_FORCE_NOOP_RATE <= 0.0:
        return False
    draw = float(secrets.randbelow(1_000_000_000)) / 1_000_000_000.0
    force_noop = draw < _SUBMISSION_FORCE_NOOP_RATE
    print(
        "SUBMISSION_FORCE_NOOP_STATUS "
        f"step={int(result.episode_step_index)} "
        f"player={int(result.player_index)} "
        f"num_agents={int(result.num_agents)} "
        f"game_enabled={int(result.force_noop_game_enabled)} "
        f"rate={_SUBMISSION_FORCE_NOOP_RATE:.6f} "
        f"draw={draw:.6f} "
        f"force_noop={int(force_noop)} "
        f"original_moves={len(result.moves)}",
        file=sys.stderr,
        flush=True,
    )
    return force_noop


def _assert_submission_perf_test_not_failed(my_pid: int, completed_step_count: int) -> None:
    fail_step = _submission_perf_test_fail_step_for_seat_index(my_pid)
    assert completed_step_count < fail_step, (my_pid, fail_step, completed_step_count)


def _log_submission_cpu_info() -> None:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    first_processor_block = cpuinfo.split("\n\n", 1)[0]
    fields: dict[str, str] = {}
    for line in first_processor_block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    model_name = fields["model name"]
    flags = fields["flags"]
    print(
        "SUBMISSION_CPU_INFO "
        f"model_name={model_name!r} "
        f"os_cpu_count={os.cpu_count()} "
        f"torch_num_threads={torch.get_num_threads()} "
        f"flags={flags}",
        file=sys.stderr,
        flush=True,
    )


def _start_aoti_load_watchdog(path: Path) -> subprocess.Popen[bytes]:
    parent_pid = os.getpid()
    code = (
        "import os\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "parent_pid = int(sys.argv[1])\n"
        "seconds = float(sys.argv[2])\n"
        "path = sys.argv[3]\n"
        "time.sleep(seconds)\n"
        "os.write(2, f'SUBMISSION_AOTI_LOAD_WATCHDOG_TIMEOUT seconds={seconds} path={path!r}\\n'.encode())\n"
        "os.kill(parent_pid, signal.SIGABRT)\n"
        "time.sleep(2.0)\n"
        "os.write(2, f'SUBMISSION_AOTI_LOAD_WATCHDOG_SIGKILL parent_pid={parent_pid}\\n'.encode())\n"
        "os.kill(parent_pid, signal.SIGKILL)\n"
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            code,
            str(parent_pid),
            str(_SUBMISSION_AOTI_LOAD_WATCHDOG_SECONDS),
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )


def _stop_aoti_load_watchdog(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    proc.wait()


def _assert_submission_policy_tensor_contract(batch_obs: dict[str, torch.Tensor]) -> None:
    expected: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = (
        (
            "orbit_planet_features",
            (1, 1, ORBIT_MAX_PLANETS, ORBIT_PLANET_FEATURES),
            torch.float32,
        ),
        (
            "orbit_planet_arrival_features",
            (
                1,
                1,
                ORBIT_MAX_PLANETS,
                ORBIT_PLANET_ARRIVAL_HORIZON,
                ORBIT_PLAYER_AXIS_SLOTS,
                ORBIT_PLANET_TEMPORAL_FEATURES,
            ),
            torch.float32,
        ),
        ("orbit_enemy_mask", (1, 1, ORBIT_PLAYER_AXIS_SLOTS - 1), torch.float32),
        ("orbit_planet_mask", (1, 1, ORBIT_MAX_PLANETS), torch.float32),
        ("orbit_planet_pairwise_mask", (1, 1, ORBIT_PLANET_PAIRWISE_COUNT), torch.float32),
        (
            "orbit_planet_pairwise_features",
            (1, 1, ORBIT_PLANET_PAIRWISE_COUNT, ORBIT_EDGE_FEATURES),
            torch.float32,
        ),
        (
            "available_action_mask",
            (1, 1, ORBIT_PLANET_ACTION_SLOTS, ORBIT_PER_PLANET_MOVE_CLASSES),
            torch.int8,
        ),
    )
    for key, shape, dtype in expected:
        t = batch_obs[key]
        assert isinstance(t, torch.Tensor), (key, type(t))
        assert tuple(t.shape) == shape, (key, tuple(t.shape), shape)
        assert t.dtype == dtype, (key, t.dtype, dtype)
        assert t.is_contiguous(), (key, tuple(t.shape), tuple(t.stride()))


def _assert_aoti_policy_input_contract(batch_obs: dict[str, torch.Tensor]) -> None:
    _assert_submission_policy_tensor_contract(batch_obs)


def _submission_policy_input_dump_sample(
    batch_obs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    for key in _POLICY_INPUT_DUMP_KEYS:
        assert key in batch_obs, key
        t = batch_obs[key]
        assert isinstance(t, torch.Tensor), (key, type(t))
        assert t.ndim >= 2, (key, tuple(t.shape))
        assert int(t.shape[0]) == 1, (key, tuple(t.shape))
        assert int(t.shape[1]) == 1, (key, tuple(t.shape))
    return tuple(
        batch_obs[key].detach().cpu().contiguous()
        for key in _POLICY_INPUT_DUMP_KEYS
    )


def _maybe_dump_submission_policy_inputs(
    *,
    batch_obs: dict[str, torch.Tensor],
    player_index: int,
) -> None:
    if not _SUBMISSION_DUMP_INPUTS:
        return
    if int(player_index) != 0:
        return
    _SUBMISSION_INPUT_DUMP_SAMPLES.append(_submission_policy_input_dump_sample(batch_obs))
    _SUBMISSION_DUMP_INPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _POLICY_INPUT_DUMP_FORMAT,
            "input_keys": _POLICY_INPUT_DUMP_KEYS,
            "samples_per_step": 1,
            "samples": _SUBMISSION_INPUT_DUMP_SAMPLES,
        },
        str(_SUBMISSION_DUMP_INPUTS_PATH),
    )


def _submission_num_agents_from_reset_plain(plain_obs: dict[str, Any]) -> int:
    owners: set[int] = set()
    for key in ("planets", "fleets"):
        rows = plain_obs[key]
        assert isinstance(rows, list), (key, type(rows))
        for row in rows:
            assert isinstance(row, (list, tuple)) and len(row) >= 2, (key, row)
            owner = int(row[1])
            if owner >= 0:
                owners.add(owner)
    assert owners in ({0, 1}, {0, 1, 2, 3}), owners
    return len(owners)


class _RunState:
    __slots__ = (
        "episode_step_index",
        "cached_cpp",
        "num_agents",
        "initial_remaining_overage_time",
        "dummy_actions_until_episode_end",
        "force_noop_game_enabled",
    )

    def __init__(self) -> None:
        self.episode_step_index = 0
        self.cached_cpp: OrbitKaggleCppObservationCache | None = None
        self.num_agents = 0
        self.initial_remaining_overage_time = -1.0
        self.dummy_actions_until_episode_end = False
        self.force_noop_game_enabled = _submission_force_noop_game_enabled()


@dataclass(frozen=True)
class _LoadedSubmissionModel:
    model: Any
    artifact_kind: str


_MODELS_BY_NUM_AGENTS: dict[int, _LoadedSubmissionModel] = {}
_OVERAGE_PRIMARY_MODEL: _LoadedSubmissionModel | None = None
_RUNNER: "OrbitSubmissionRunner | None" = None


def _configure_submission_model_runtime(model: ImpalaOrbitModel) -> None:
    zero_heads = torch.zeros(1, dtype=torch.float32)
    model.set_entropy_floor_targets(zero_heads)
    model.set_is_sample(SUBMISSION_IS_SAMPLE)
    model.set_compile_friendly_sample(True)
    model.set_shuffle_identity_ids(SUBMISSION_SHUFFLE_IDENTITY_IDS)


def _submission_device() -> torch.device:
    if _LOCAL_SIMULATION:
        return torch.device("cuda:1")
    return torch.device("cpu")


def _submission_model_init_kwargs(model_config: Mapping[str, Any]) -> dict[str, Any]:
    policy_model_config = copy.deepcopy(model_config)
    assert isinstance(policy_model_config, dict), type(policy_model_config)
    orbit_impala_config = policy_model_config["orbit_impala"]
    assert isinstance(orbit_impala_config, dict), type(orbit_impala_config)
    orbit_impala_config["use_value_opponent_model_embedding"] = False
    flags = SimpleNamespace(
        model=policy_model_config,
        target_min_entropy={"spawn_fleet": 0.0},
        entropy_floor_max_temperature=1,
        entropy_floor_num_iters=0,
    )
    kw = impala_orbit_model_init_kwargs_from_flags(flags)
    kw["entropy_floor_target"] = (0.0,)
    kw["entropy_floor_max_temperature"] = 1
    kw["entropy_floor_num_iters"] = 0
    kw["include_rl_policy_value_heads"] = True
    kw["include_rl_value_head"] = False
    return kw


_TRAINING_ONLY_STATE_DICT_PREFIXES = (
    "_global_value_head.",
    "_global_value_head_production_delta.",
    "_value_opponent_model_encoder.",
    "_value_opponent_identity_model_fusion.",
)


def _strip_torch_compile_orig_mod_prefix(sd: dict[Any, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    keys = tuple(str(k) for k in sd.keys())
    prefixed = tuple(k.startswith(prefix) for k in keys)
    assert all(prefixed) or not any(prefixed), keys[:8]
    if not any(prefixed):
        return {str(k): v for k, v in sd.items()}
    return {str(k)[len(prefix) :]: v for k, v in sd.items()}


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict), type(ckpt)
    assert "model_state_dict" in ckpt, (
        f"Expected IMPALA checkpoint with model_state_dict; keys={sorted(ckpt.keys())}"
    )
    assert "model_config" in ckpt, (
        f"Expected IMPALA checkpoint with model_config; keys={sorted(ckpt.keys())}"
    )
    assert isinstance(ckpt["model_config"], Mapping), type(ckpt["model_config"])
    return ckpt


def _load_model_weights(model: ImpalaOrbitModel, ckpt: dict[str, Any]) -> None:
    raw_sd = ckpt["model_state_dict"]
    assert isinstance(raw_sd, dict), type(raw_sd)
    sd = _strip_torch_compile_orig_mod_prefix(raw_sd)
    model_sd = model.state_dict()
    checkpoint_only = sorted(k for k in sd if k not in model_sd)
    unexpected_checkpoint_only = sorted(
        k
        for k in checkpoint_only
        if not k.startswith(_TRAINING_ONLY_STATE_DICT_PREFIXES)
    )
    assert not unexpected_checkpoint_only, unexpected_checkpoint_only
    load_sd = {k: v for k, v in sd.items() if k in model_sd}
    missing = sorted(k for k in model_sd if k not in load_sd)
    assert not missing, missing
    for k, v in load_sd.items():
        assert isinstance(v, torch.Tensor), (k, type(v))
        assert v.shape == model_sd[k].shape, (k, tuple(v.shape), tuple(model_sd[k].shape))
    model.load_state_dict(load_sd, strict=True)


def load_submission_model(path: Path, *, device: torch.device | None = None) -> ImpalaOrbitModel:
    assert path.is_file(), path
    dev = torch.device("cpu") if device is None else device
    ckpt = _load_checkpoint_payload(path)
    m = ImpalaOrbitModel(**_submission_model_init_kwargs(ckpt["model_config"]))
    _load_model_weights(m, ckpt)
    m.to(device=dev)
    m.eval()
    _configure_submission_model_runtime(m)
    return m


def _load_packaged_eager_model_artifact(
    path: Path,
    *,
    device: torch.device,
) -> _LoadedSubmissionModel:
    assert device.type == "cpu", "Packaged optimized eager model artifacts are CPU-only"
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(payload, dict), type(payload)
    assert payload["artifact_kind"] == "impala_eager_module", payload.keys()
    assert isinstance(payload["is_sample"], bool), type(payload["is_sample"])
    assert payload["is_sample"] == SUBMISSION_IS_SAMPLE, (
        payload["is_sample"],
        SUBMISSION_IS_SAMPLE,
    )
    assert isinstance(payload["shuffle_identity_ids"], bool), type(payload["shuffle_identity_ids"])
    assert payload["shuffle_identity_ids"] == SUBMISSION_SHUFFLE_IDENTITY_IDS, (
        payload["shuffle_identity_ids"],
        SUBMISSION_SHUFFLE_IDENTITY_IDS,
    )
    model = payload["model"]
    assert isinstance(model, torch.nn.Module), type(model)
    model.eval()
    assert isinstance(model, ImpalaOrbitModel), type(model)
    _configure_submission_model_runtime(model)
    return _LoadedSubmissionModel(model=model, artifact_kind="impala_eager")


def _load_torchscript_policy_artifact(
    path: Path,
    *,
    device: torch.device,
) -> _LoadedSubmissionModel:
    assert device.type == "cpu", "TorchScript submission policy artifacts are CPU-only"
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return _LoadedSubmissionModel(model=model, artifact_kind="policy_logits_torchscript")


def _load_aoti_runner_cpp_module() -> Any:
    global _AOTI_RUNNER_CPP_MODULE
    if _AOTI_RUNNER_CPP_MODULE is None:
        ext_paths = tuple(
            sorted(Path(__file__).resolve().parent.glob(f"{_AOTI_RUNNER_CPP_EXT_NAME}*.so"))
        )
        assert len(ext_paths) == 1, ext_paths
        spec = importlib.util.spec_from_file_location(_AOTI_RUNNER_CPP_EXT_NAME, str(ext_paths[0]))
        assert spec is not None and spec.loader is not None, ext_paths[0]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "AOTIPolicyRunnerCpu"), ext_paths[0]
        _AOTI_RUNNER_CPP_MODULE = module
    return _AOTI_RUNNER_CPP_MODULE


class _AOTIPolicyRunnerCallable:
    def __init__(self, path: Path) -> None:
        module = _load_aoti_runner_cpp_module()
        self._runner = module.AOTIPolicyRunnerCpu(str(path))

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        outputs = self._runner.run(list(inputs))
        assert isinstance(outputs, list), type(outputs)
        assert len(outputs) == 1, len(outputs)
        output = outputs[0]
        assert isinstance(output, torch.Tensor), type(output)
        return output


def _load_aoti_policy_artifact(
    path: Path,
    *,
    device: torch.device,
) -> _LoadedSubmissionModel:
    assert device.type == "cpu", "AOTInductor submission policy artifacts are CPU-only"
    print(
        f"SUBMISSION_AOTI_LOAD_START path={str(path)!r}",
        file=sys.stderr,
        flush=True,
    )
    faulthandler.enable(file=2, all_threads=True)
    watchdog_proc = _start_aoti_load_watchdog(path)
    model = _AOTIPolicyRunnerCallable(path)
    _stop_aoti_load_watchdog(watchdog_proc)
    print(
        f"SUBMISSION_AOTI_LOAD_DONE path={str(path)!r}",
        file=sys.stderr,
        flush=True,
    )
    return _LoadedSubmissionModel(model=model, artifact_kind="policy_logits_aoti")


def _submission_model_artifact_paths(num_agents: int) -> tuple[Path, Path, Path, Path]:
    assert num_agents in (2, 4), num_agents
    suffix = f"_{num_agents}p"
    return (
        _SUBMISSION_DIR / f"checkpoint_weights{suffix}.pt",
        _SUBMISSION_DIR / f"checkpoint_model_eager{suffix}.pt",
        _SUBMISSION_DIR / f"checkpoint_policy_logits_torchscript{suffix}.pt",
        _SUBMISSION_DIR / f"checkpoint_policy_logits_aoti{suffix}.so",
    )


def _submission_shared_model_artifact_paths() -> tuple[Path, Path, Path, Path]:
    return (
        _SUBMISSION_DIR / "checkpoint_weights.pt",
        _SUBMISSION_DIR / "checkpoint_model_eager.pt",
        _SUBMISSION_DIR / "checkpoint_policy_logits_torchscript.pt",
        _SUBMISSION_DIR / "checkpoint_policy_logits_aoti.so",
    )


def _submission_overage_primary_model_artifact_paths() -> tuple[Path, Path, Path, Path]:
    return (
        _SUBMISSION_DIR / "checkpoint_weights_overage_primary.pt",
        _SUBMISSION_DIR / "checkpoint_model_eager_overage_primary.pt",
        _SUBMISSION_DIR / "checkpoint_policy_logits_torchscript_overage_primary.pt",
        _SUBMISSION_DIR / "checkpoint_policy_logits_aoti_overage_primary.so",
    )


def _submission_checkpoint_env_name(num_agents: int) -> str:
    assert num_agents in (2, 4), num_agents
    return f"ORBIT_IMPALA_CHECKPOINT_{num_agents}P"


def _submission_checkpoint_env_paths(num_agents: int) -> tuple[Path, ...]:
    assert num_agents in (2, 4), num_agents
    shared = os.environ.get("ORBIT_IMPALA_CHECKPOINT", "").strip()
    overage_primary = os.environ.get("ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY", "").strip()
    specific_by_num_agents = {
        n: os.environ.get(_submission_checkpoint_env_name(n), "").strip()
        for n in (2, 4)
    }
    assert not (shared and (any(specific_by_num_agents.values()) or overage_primary)), (
        "ORBIT_IMPALA_CHECKPOINT",
        shared,
        specific_by_num_agents,
        overage_primary,
    )
    assert all(specific_by_num_agents.values()) or not any(specific_by_num_agents.values()), (
        specific_by_num_agents
    )
    assert not overage_primary or all(specific_by_num_agents.values()), (
        overage_primary,
        specific_by_num_agents,
    )
    specific = specific_by_num_agents[num_agents]
    if specific:
        return (Path(specific),)
    if shared:
        return (Path(shared),)
    return ()


def _load_submission_artifact_from_paths(
    *,
    paths: tuple[Path, Path, Path, Path],
    device: torch.device,
) -> _LoadedSubmissionModel:
    default_ckpt, eager_artifact, torchscript_artifact, aoti_artifact = paths
    packaged_artifacts = tuple(
        p
        for p in (
            default_ckpt,
            eager_artifact,
            torchscript_artifact,
            aoti_artifact,
        )
        if p.is_file()
    )
    assert len(packaged_artifacts) == 1, (
        "Expected exactly one packaged model artifact",
        packaged_artifacts,
    )
    path = packaged_artifacts[0]
    if path == default_ckpt:
        return _LoadedSubmissionModel(
            model=load_submission_model(path, device=device),
            artifact_kind="impala_eager",
        )
    if path == eager_artifact:
        return _load_packaged_eager_model_artifact(
            path,
            device=device,
        )
    if path == torchscript_artifact:
        return _load_torchscript_policy_artifact(
            path,
            device=device,
        )
    assert path == aoti_artifact, path
    return _load_aoti_policy_artifact(path, device=device)


def _submission_packaged_artifacts(paths: tuple[Path, Path, Path, Path]) -> tuple[Path, ...]:
    return tuple(p for p in paths if p.is_file())


def _submission_has_overage_primary_model() -> bool:
    env_path = os.environ.get("ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY", "").strip()
    packaged_artifacts = _submission_packaged_artifacts(
        _submission_overage_primary_model_artifact_paths()
    )
    assert not (env_path and packaged_artifacts), (
        "ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY",
        env_path,
        packaged_artifacts,
    )
    assert len(packaged_artifacts) in (0, 1), packaged_artifacts
    return bool(env_path) or bool(packaged_artifacts)


def _submission_uses_shared_model(num_agents: int) -> bool:
    assert num_agents in (2, 4), num_agents
    shared_env = os.environ.get("ORBIT_IMPALA_CHECKPOINT", "").strip()
    specific_env = os.environ.get(_submission_checkpoint_env_name(num_agents), "").strip()
    assert not (shared_env and specific_env), (shared_env, specific_env)
    if shared_env:
        return True
    if specific_env:
        return False
    shared_artifacts = _submission_packaged_artifacts(_submission_shared_model_artifact_paths())
    specific_artifacts = _submission_packaged_artifacts(_submission_model_artifact_paths(num_agents))
    assert not (shared_artifacts and specific_artifacts), (
        num_agents,
        shared_artifacts,
        specific_artifacts,
    )
    return bool(shared_artifacts)


def _get_overage_primary_model() -> _LoadedSubmissionModel:
    global _OVERAGE_PRIMARY_MODEL
    if _OVERAGE_PRIMARY_MODEL is None:
        env_path = os.environ.get("ORBIT_IMPALA_CHECKPOINT_OVERAGE_PRIMARY", "").strip()
        dev = _submission_device()
        if env_path:
            path = Path(env_path)
            assert path.is_file(), path
            _OVERAGE_PRIMARY_MODEL = _LoadedSubmissionModel(
                model=load_submission_model(path, device=dev),
                artifact_kind="impala_eager",
            )
        else:
            _OVERAGE_PRIMARY_MODEL = _load_submission_artifact_from_paths(
                paths=_submission_overage_primary_model_artifact_paths(),
                device=dev,
            )
    return _OVERAGE_PRIMARY_MODEL


def _get_model(num_agents: int) -> _LoadedSubmissionModel:
    assert num_agents in (2, 4), num_agents
    if num_agents not in _MODELS_BY_NUM_AGENTS:
        env_paths = _submission_checkpoint_env_paths(num_agents)
        dev = _submission_device()
        if env_paths:
            assert len(env_paths) == 1, env_paths
            path = env_paths[0]
            assert path.is_file(), path
            _MODELS_BY_NUM_AGENTS[num_agents] = _LoadedSubmissionModel(
                model=load_submission_model(path, device=dev),
                artifact_kind="impala_eager",
            )
            return _MODELS_BY_NUM_AGENTS[num_agents]

        specific_paths = _submission_model_artifact_paths(num_agents)
        shared_paths = _submission_shared_model_artifact_paths()
        specific_artifacts = _submission_packaged_artifacts(specific_paths)
        shared_artifacts = _submission_packaged_artifacts(shared_paths)
        assert not (specific_artifacts and shared_artifacts), (
            num_agents,
            specific_artifacts,
            shared_artifacts,
        )
        if specific_artifacts:
            _MODELS_BY_NUM_AGENTS[num_agents] = _load_submission_artifact_from_paths(
                paths=specific_paths,
                device=dev,
            )
        else:
            _MODELS_BY_NUM_AGENTS[num_agents] = _load_submission_artifact_from_paths(
                paths=shared_paths,
                device=dev,
            )
    return _MODELS_BY_NUM_AGENTS[num_agents]


def _submission_configuration_field(configuration: Any, name: str) -> Any:
    if isinstance(configuration, Mapping):
        return configuration[name]
    return getattr(configuration, name)


@dataclass(frozen=True)
class OrbitSubmissionInferenceResult:
    moves: list[Any]
    classes: torch.Tensor
    batch_obs: dict[str, torch.Tensor]
    model_output: dict[str, Any]
    player_index: int
    num_agents: int
    episode_step_index: int
    force_noop_game_enabled: bool


class OrbitSubmissionRunner:
    def __init__(
        self,
        *,
        model: Any,
        model_artifact_kind: str = "impala_eager",
        overage_fallback_model: _LoadedSubmissionModel | None = None,
        submission_model_count: int = 1,
        active_model_label: str = "shared",
        overage_fallback_model_label: str = "",
        device: torch.device | None = None,
        emit_wall_profile_summary: bool = True,
    ) -> None:
        self._model = model
        assert model_artifact_kind in ("impala_eager", "policy_logits_torchscript", "policy_logits_aoti"), (
            model_artifact_kind,
        )
        self._model_artifact_kind = model_artifact_kind
        self._overage_fallback_model = overage_fallback_model
        assert int(submission_model_count) in (1, 2, 3), submission_model_count
        self._submission_model_count = int(submission_model_count)
        assert active_model_label in ("shared", "2p", "4p", "overage_primary"), active_model_label
        self._active_model_label = active_model_label
        if overage_fallback_model is None:
            assert overage_fallback_model_label == "", overage_fallback_model_label
        else:
            assert overage_fallback_model_label in ("2p", "4p"), overage_fallback_model_label
        self._overage_fallback_model_label = overage_fallback_model_label
        self._using_overage_fallback_model = False
        self._device = _submission_device() if device is None else device
        if self._model_artifact_kind in ("policy_logits_torchscript", "policy_logits_aoti"):
            assert self._device.type == "cpu", self._device
        self._cuda_bfloat16_inference = bool(_LOCAL_SIMULATION) and self._device.type == "cuda"
        self._state = _RunState()
        self._wall_profiler = WallTreeProfiler()
        self._emit_wall_profile_summary = bool(emit_wall_profile_summary)

    @property
    def dummy_actions_until_episode_end(self) -> bool:
        return self._state.dummy_actions_until_episode_end

    def _overage_budget_status(self, plain_obs: Mapping[str, Any]) -> tuple[float, float, float, bool]:
        remaining = float(plain_obs["remainingOverageTime"])
        assert remaining >= 0.0, remaining
        if self._state.initial_remaining_overage_time < 0.0:
            self._state.initial_remaining_overage_time = remaining
        initial = float(self._state.initial_remaining_overage_time)
        trigger_remaining = self._state.initial_remaining_overage_time * (
            1.0 - _SUBMISSION_OVERAGE_DUMMY_USED_FRACTION
        )
        return remaining, initial, trigger_remaining, remaining <= trigger_remaining

    def _switch_to_overage_fallback_model(self) -> None:
        fallback_model = self._overage_fallback_model
        assert fallback_model is not None
        self._model = fallback_model.model
        self._model_artifact_kind = fallback_model.artifact_kind
        if self._model_artifact_kind in ("policy_logits_torchscript", "policy_logits_aoti"):
            assert self._device.type == "cpu", self._device
        self._active_model_label = self._overage_fallback_model_label
        self._using_overage_fallback_model = True

    def _log_submission_step_status(
        self,
        *,
        current_step: int,
        player_index: int,
        num_agents: int,
        remaining_overage_time: float,
        initial_overage_time: float,
        trigger_remaining_overage_time: float,
        overage_budget_exhausted: bool,
    ) -> None:
        print(
            "SUBMISSION_STEP_STATUS "
            f"step={int(current_step)} "
            f"player={int(player_index)} "
            f"num_agents={int(num_agents)} "
            f"submission_models={self._submission_model_count} "
            f"active_model={self._active_model_label} "
            f"remaining_overage_time={remaining_overage_time:.6f} "
            f"initial_overage_time={initial_overage_time:.6f} "
            f"trigger_remaining_overage_time={trigger_remaining_overage_time:.6f} "
            f"overage_budget_exhausted={int(overage_budget_exhausted)}",
            file=sys.stderr,
            flush=True,
        )

    def step(
        self,
        observation: Any,
        configuration: Any,
        *,
        include_policy_logits_pre_action_mask: bool = False,
    ) -> OrbitSubmissionInferenceResult:
        assert configuration is not None
        step_prof = self._wall_profiler
        step_wall_t0 = time.perf_counter()
        with profiler_span(step_prof, "submission_step"):
            with profiler_span(step_prof, "observation_plain"):
                plain_obs = orbit_observation_to_plain(observation)
            current_step = int(self._state.episode_step_index)
            assert current_step >= 0, current_step
            if current_step == 0:
                _log_submission_cpu_info()
            if self._state.cached_cpp is None:
                self._state.num_agents = _submission_num_agents_from_reset_plain(plain_obs)
            num_agents = int(self._state.num_agents)
            if _SUBMISSION_FAIL_ENABLED:
                assert num_agents >= _SUBMISSION_REQUIRED_NUM_AGENTS, num_agents
            my_pid = int(plain_obs.get("player", 0))
            assert 0 <= my_pid < ORBIT_PLAYER_AXIS_SLOTS, my_pid
            assert my_pid < num_agents, (my_pid, num_agents)
            completed_step_count = current_step + 1
            (
                remaining_overage_time,
                initial_overage_time,
                trigger_remaining_overage_time,
                overage_budget_exhausted,
            ) = self._overage_budget_status(plain_obs)
            if overage_budget_exhausted:
                if self._overage_fallback_model is None:
                    self._state.dummy_actions_until_episode_end = True
                    self._log_submission_step_status(
                        current_step=current_step,
                        player_index=my_pid,
                        num_agents=num_agents,
                        remaining_overage_time=remaining_overage_time,
                        initial_overage_time=initial_overage_time,
                        trigger_remaining_overage_time=trigger_remaining_overage_time,
                        overage_budget_exhausted=overage_budget_exhausted,
                    )
                    if _SUBMISSION_FAIL_ENABLED:
                        _assert_submission_perf_test_not_failed(my_pid, completed_step_count)
                    self._state.episode_step_index = current_step + 1
                    return OrbitSubmissionInferenceResult(
                        moves=[],
                        classes=torch.empty((0,), dtype=torch.int64),
                        batch_obs={},
                        model_output={"overage_dummy_actions": True},
                        player_index=my_pid,
                        num_agents=num_agents,
                        episode_step_index=current_step,
                        force_noop_game_enabled=self._state.force_noop_game_enabled,
                    )
                if not self._using_overage_fallback_model:
                    self._switch_to_overage_fallback_model()
            self._log_submission_step_status(
                current_step=current_step,
                player_index=my_pid,
                num_agents=num_agents,
                remaining_overage_time=remaining_overage_time,
                initial_overage_time=initial_overage_time,
                trigger_remaining_overage_time=trigger_remaining_overage_time,
                overage_budget_exhausted=overage_budget_exhausted,
            )
            with profiler_span(step_prof, "state_and_seats"):
                plain_obs = dict(plain_obs)
                plain_obs["step"] = current_step
                my_policy_slot = orbit_policy_slot_for_compact_agent(my_pid, num_agents)

                seats_plain: list[dict[str, Any]] = []
                for seat in range(num_agents):
                    pl = dict(plain_obs)
                    pl["player"] = int(seat)
                    seats_plain.append(pl)
                attach_zeros_action_taken_index_on_seats(seats_plain)

                model_dev = self._device
                ship_speed = float(_submission_configuration_field(configuration, "shipSpeed"))
            cached_cpp = self._state.cached_cpp
            with profiler_span(step_prof, "cache_clock"):
                if cached_cpp is None:
                    cached_cpp = OrbitKaggleCppObservationCache(
                        configuration=configuration,
                        num_agents=num_agents,
                    )
                    self._state.cached_cpp = cached_cpp
                    cached_cpp.reset_from_kaggle_plain(
                        plain=plain_obs,
                        num_agents=num_agents,
                    )
                else:
                    cached_cpp.step_noop_and_update_comets_from_kaggle_plain(
                        plain=plain_obs,
                        num_agents=num_agents,
                    )

            with profiler_span(step_prof, "active_policy_slots"):
                active_policy_slots = (my_policy_slot,)
                assert active_policy_slots == (my_policy_slot,), (
                    active_policy_slots,
                    my_pid,
                    my_policy_slot,
                )
            with profiler_span(step_prof, "snapshot_policy_obs"):
                snap = cached_cpp.snapshot_policy_obs_from_plain_seats_cpu(
                    plain=plain_obs,
                    seats_plain=seats_plain,
                    policy_slots=active_policy_slots,
                    ship_speed=ship_speed,
                    wall_profiler=step_prof,
                )

            with profiler_span(step_prof, "batch_obs_to_device"):
                batch_obs = {k: v.unsqueeze(0).to(device=model_dev) for k, v in snap.items()}
                _assert_submission_policy_tensor_contract(batch_obs)
                _maybe_dump_submission_policy_inputs(
                    batch_obs=batch_obs,
                    player_index=my_pid,
                )
            if _SUBMISSION_FAIL_ENABLED and _SUBMISSION_FAIL_BEFORE_FORWARD:
                _assert_submission_perf_test_not_failed(my_pid, completed_step_count)
            with profiler_span(step_prof, "model_forward"):
                with torch.no_grad():
                    if self._model_artifact_kind == "policy_logits_torchscript":
                        assert not include_policy_logits_pre_action_mask
                        actions_grid = self._model(
                            batch_obs["orbit_planet_features"],
                            batch_obs["orbit_planet_arrival_features"],
                            batch_obs["orbit_enemy_mask"],
                            batch_obs["orbit_planet_mask"],
                            batch_obs["orbit_planet_pairwise_mask"],
                            batch_obs["orbit_planet_pairwise_features"],
                            batch_obs["available_action_mask"],
                        )
                        assert isinstance(actions_grid, torch.Tensor), type(actions_grid)
                        out = {"actions_LEARN": {"spawn_fleet": actions_grid}}
                    elif self._model_artifact_kind == "policy_logits_aoti":
                        assert not include_policy_logits_pre_action_mask
                        _assert_aoti_policy_input_contract(batch_obs)
                        aoti_inputs = (
                            batch_obs["orbit_planet_features"],
                            batch_obs["orbit_planet_arrival_features"],
                            batch_obs["orbit_enemy_mask"],
                            batch_obs["orbit_planet_mask"],
                            batch_obs["orbit_planet_pairwise_mask"],
                            batch_obs["orbit_planet_pairwise_features"],
                            batch_obs["available_action_mask"],
                        )
                        actions_grid = self._model(
                            *aoti_inputs,
                        )
                        assert isinstance(actions_grid, torch.Tensor), type(actions_grid)
                        out = {"actions_LEARN": {"spawn_fleet": actions_grid}}
                    elif self._cuda_bfloat16_inference:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = self._model(
                                {"obs_LEARN_INFER": batch_obs},
                                output_full_policy_log_probs=False,
                                include_final_policy_logits=False,
                                include_policy_logits_pre_action_mask=bool(
                                    include_policy_logits_pre_action_mask
                                ),
                                include_value_head=False,
                                wall_profiler=step_prof,
                            )
                    else:
                        out = self._model(
                            {"obs_LEARN_INFER": batch_obs},
                            output_full_policy_log_probs=False,
                            include_final_policy_logits=False,
                            include_policy_logits_pre_action_mask=bool(
                                include_policy_logits_pre_action_mask
                            ),
                            include_value_head=False,
                            wall_profiler=step_prof,
                        )

            with profiler_span(step_prof, "action_select"):
                actions_learn = out["actions_LEARN"]
                assert isinstance(actions_learn, dict), type(actions_learn)
                actions_grid = actions_learn["spawn_fleet"]
                assert isinstance(actions_grid, torch.Tensor), type(actions_grid)
                assert actions_grid.shape == (
                    1,
                    1,
                    ORBIT_PLANET_ACTION_SLOTS,
                    1,
                ), actions_grid.shape
                action_mask = batch_obs["available_action_mask"]
                assert isinstance(action_mask, torch.Tensor), type(action_mask)
                assert action_mask.shape == (
                    1,
                    1,
                    ORBIT_PLANET_ACTION_SLOTS,
                    ORBIT_PER_PLANET_MOVE_CLASSES,
                ), action_mask.shape
                assert action_mask.dtype == torch.int8, action_mask.dtype
                actions = actions_grid[:, 0, :, 0].to(dtype=torch.int64)
                assert actions.shape == (1, ORBIT_PLANET_ACTION_SLOTS), actions.shape
                selected_available = action_mask.to(dtype=torch.bool)[:, 0].gather(
                    -1,
                    actions.unsqueeze(-1),
                ).squeeze(-1)
                assert selected_available.shape == actions.shape, selected_available.shape
                assert torch.all(selected_available[0]), selected_available[0]
                classes = actions[0].detach().cpu()
                assert tuple(classes.shape) == (ORBIT_PLANET_ACTION_SLOTS,), classes.shape

            with profiler_span(step_prof, "moves_decode"):
                my_plain = seats_plain[my_pid]
                honest_send_ships = cached_cpp.honest_shared_send_ships_last()
                moves = kaggle_moves_for_seat_from_classes_honest_angle(
                    seat_plain=my_plain,
                    classes=classes,
                    ship_speed=ship_speed,
                    honest_available_action_mask=snap["available_action_mask"][0],
                    honest_send_ships=honest_send_ships,
                    honest_angle_source=cached_cpp,
                )
                assert_orbit_action(moves)

            if _SUBMISSION_HONEST_FULL_CACHE_WARMUP_ENABLED:
                with profiler_span(step_prof, "honest_full_cache_warmup"):
                    with profiler_span(step_prof, "warmup_deadline_setup"):
                        act_timeout_seconds = float(_submission_configuration_field(configuration, "actTimeout"))
                        assert act_timeout_seconds > 0.0, act_timeout_seconds
                        warmup_deadline = step_wall_t0 + act_timeout_seconds
                    with profiler_span(step_prof, "warmup_prune_before_current"):
                        warmup_pruned = cached_cpp.honest_shared_action_mask_full_cache_prune_before_current_future()
                    with profiler_span(step_prof, "warmup_plain_planets_to_tensor_once"):
                        warmup_planet_rows, warmup_planet_count = planet_rows_tensor_from_plain_planets_float64(
                            plain_obs["planets"],
                        )
                    warmup_ship_buckets = 0
                    warmup_finished = False
                    while True:
                        with profiler_span(step_prof, "warmup_deadline_check"):
                            warmup_has_time = time.perf_counter() < warmup_deadline
                        if not warmup_has_time:
                            break
                        warmed, finished = cached_cpp.honest_shared_action_mask_full_cache_warmup_one_from_rows(
                            planet_rows=warmup_planet_rows,
                            planet_count=warmup_planet_count,
                            wall_profiler=step_prof,
                        )
                        with profiler_span(step_prof, "warmup_loop_accounting"):
                            warmup_ship_buckets += warmed
                            if finished:
                                warmup_finished = True
                                break
                    with profiler_span(step_prof, "warmup_stats"):
                        (
                            warmup_complete_before_step,
                            warmup_cursor_done_src_count,
                            warmup_cursor_src_total,
                            warmup_entry_count,
                            warmup_extra_sn,
                            warmup_last_lookahead_steps,
                        ) = cached_cpp.honest_shared_action_mask_full_cache_warmup_stats()
                    with profiler_span(step_prof, "warmup_stderr_log"):
                        print(
                            "SUBMISSION_HONEST_FULL_CACHE_WARMUP "
                            f"step={current_step} "
                            f"pruned={warmup_pruned} "
                            f"ship_buckets={warmup_ship_buckets} "
                            f"finished={int(warmup_finished)} "
                            f"complete_before_step={warmup_complete_before_step} "
                            f"cursor_done_src_count={warmup_cursor_done_src_count} "
                            f"cursor_src_total={warmup_cursor_src_total} "
                            f"entry_count={warmup_entry_count} "
                            f"extra_sn={warmup_extra_sn} "
                            f"last_lookahead_steps={warmup_last_lookahead_steps}",
                            file=sys.stderr,
                            flush=True,
                        )

            with profiler_span(step_prof, "state_advance"):
                self._state.episode_step_index = current_step + 1
        if self._emit_wall_profile_summary:
            step_prof.summary_stdout(
                f"orbit_submission step={current_step} player={my_pid} num_agents={num_agents}",
                iteration_wall_ms=(time.perf_counter() - step_wall_t0) * 1000.0,
                line_prefix="WALL_TREE_SUBMISSION ",
                file=sys.stderr,
            )
        completed_step_count = current_step + 1
        if _SUBMISSION_FAIL_ENABLED and not _SUBMISSION_FAIL_BEFORE_FORWARD:
            _assert_submission_perf_test_not_failed(my_pid, completed_step_count)
        return OrbitSubmissionInferenceResult(
            moves=moves,
            classes=classes,
            batch_obs=batch_obs,
            model_output=out,
            player_index=my_pid,
            num_agents=num_agents,
            episode_step_index=current_step,
            force_noop_game_enabled=self._state.force_noop_game_enabled,
        )


def agent(observation: Any, configuration: Any = None) -> list[Any]:
    global _RUNNER
    if _RUNNER is None:
        if _SUBMISSION_FAIL_ENABLED and _SUBMISSION_FAIL_BEFORE_MODEL_LOAD:
            assert not _SUBMISSION_FAIL_BEFORE_MODEL_LOAD, "before_model_load"
        plain_obs = orbit_observation_to_plain(observation)
        num_agents = _submission_num_agents_from_reset_plain(plain_obs)
        if _submission_has_overage_primary_model():
            loaded_model = _get_overage_primary_model()
            overage_fallback_model = _get_model(num_agents)
            submission_model_count = 3
            active_model_label = "overage_primary"
            overage_fallback_model_label = f"{num_agents}p"
        else:
            loaded_model = _get_model(num_agents)
            overage_fallback_model = None
            if _submission_uses_shared_model(num_agents):
                submission_model_count = 1
                active_model_label = "shared"
            else:
                submission_model_count = 2
                active_model_label = f"{num_agents}p"
            overage_fallback_model_label = ""
        _RUNNER = OrbitSubmissionRunner(
            model=loaded_model.model,
            model_artifact_kind=loaded_model.artifact_kind,
            overage_fallback_model=overage_fallback_model,
            submission_model_count=submission_model_count,
            active_model_label=active_model_label,
            overage_fallback_model_label=overage_fallback_model_label,
        )
    result = _RUNNER.step(observation, configuration)
    if _submission_force_noop(result):
        return []
    return result.moves
