import os
import multiprocessing as pymp
import shutil
import uuid

os.environ['OMP_PROC_BIND'] = 'FALSE'  # avoids forked workers binding to one core via OpenMP
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

os.environ['TORCH_MATMUL_PRECISION'] = 'high'

_CACHE_ID_ENV = "TORCHINDUCTOR_CACHE_ID"
if _CACHE_ID_ENV not in os.environ:
    os.environ[_CACHE_ID_ENV] = f"monobeast_{uuid.uuid4().hex}"

import argparse
import logging
import runpy
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.utils.cpp_extension as cpp_extension
from torch import multiprocessing as mp
from src.torchbeast.core.common import (
    checkpoint_config_from_checkpoint_file,
    configure_process_cpu_thread_limits,
)

# Apply compile/runtime knobs at module import so spawn children inherit behavior.
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.recompile_limit = 64


def _install_cuda_import_guard_in_entrypoint() -> None:
    """
    Install CUDA guard at the earliest entrypoint, before importing
    train/config modules that may touch CUDA during import.
    """
    if os.getenv("MONOBEAST_CRASH_ON_CUDA_INIT", "0") != "1":
        return
    if pymp.current_process().name != "MainProcess":
        return
    if getattr(torch.cuda, "_monobeast_entrypoint_cuda_guard_installed", False):
        return

    def _crash(api_name: str):
        stack = "".join(traceback.format_stack())
        raise AssertionError(
            f"{api_name} was called before allowed CUDA init point (entrypoint guard).\n"
            "Call stack:\n"
            f"{stack}"
        )

    setattr(torch.cuda, "_monobeast_original_cuda_lazy_init_entrypoint", torch.cuda._lazy_init)  # noqa: SLF001
    setattr(torch.cuda, "_monobeast_original_cuda_init_entrypoint", torch.cuda.init)
    setattr(torch.cuda, "_monobeast_original_cuda_is_available_entrypoint", torch.cuda.is_available)
    setattr(torch.cuda, "_monobeast_original_cuda_device_count_entrypoint", torch.cuda.device_count)

    torch.cuda._lazy_init = lambda *args, **kwargs: _crash("torch.cuda._lazy_init")  # type: ignore[assignment]  # noqa: SLF001,E731
    torch.cuda.init = lambda *args, **kwargs: _crash("torch.cuda.init")  # type: ignore[assignment]  # noqa: E731
    torch.cuda.is_available = lambda *args, **kwargs: _crash("torch.cuda.is_available")  # type: ignore[assignment]  # noqa: E731
    torch.cuda.device_count = lambda *args, **kwargs: _crash("torch.cuda.device_count")  # type: ignore[assignment]  # noqa: E731
    setattr(torch.cuda, "_monobeast_entrypoint_cuda_guard_installed", True)


def _configure_torch_runtime() -> None:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir:
        print("TORCHINDUCTOR_CACHE_DIR:", cache_dir)

    torch.set_float32_matmul_precision('high')
    configure_process_cpu_thread_limits()
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.recompile_limit = 64
    torch.backends.cudnn.benchmark = True


logging.basicConfig(
    format='{asctime}.{msecs:03.0f} | {name} | {process} | {levelname}: {message}',
    style='{',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
# set this to INFO or DEBUG to see more about torch compilation
logging.getLogger('torch').setLevel(logging.WARN)
logging.getLogger('filelock').setLevel(logging.WARN)


# Impala repo root = parent of ``python/`` (same as ``configs.base.IMPALA_PROJECT_ROOT``).
_IMPALA_RUN_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"
_ENV_RUN_ARTIFACT_ROOT = "IMPALA_RUN_ARTIFACT_ROOT"
_NUM_PLAYERS_ENV = "NUM_PLAYERS"
_ORBIT_WARS_CPP_EXT_NAME = "orbit_wars_cpp"
_ORBIT_WARS_CPP_FORCE_REBUILD = "ORBIT_WARS_CPP_FORCE_REBUILD"


def _format_run_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _num_players_run_suffix() -> str:
    num_players = os.environ[_NUM_PLAYERS_ENV].strip()
    assert num_players in {"2", "4"}, f"{_NUM_PLAYERS_ENV} must be 2 or 4"
    return f"{num_players}P"


def _run_artifact_kind_subdir(cfg: SimpleNamespace) -> str:
    if int(cfg.num_actors) == 0 and int(cfg.num_rl_vis_actors) > 0:
        return "vis"
    return "train"


def _apply_impala_run_artifact_root(cfg: SimpleNamespace, run_root: Path) -> None:
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "runs").mkdir(parents=True, exist_ok=True)
    (run_root / "tapes").mkdir(parents=True, exist_ok=True)

    cfg.output_dir = str(run_root)
    cfg.torch_io_config.main_torch_io.local.dirpath = str(run_root / "runs")


def _load_training_config(config_py: str):
    path = Path(config_py).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    g = runpy.run_path(str(path), run_name="__monobeast_config__")
    if "build_training_config" not in g:
        raise RuntimeError(
            f"{path} must define build_training_config() returning the training configuration object"
        )
    cfg = g["build_training_config"]()
    assert hasattr(cfg, "sharing_strategy"), "training config must expose sharing_strategy"
    return cfg


def _prepare_orbit_wars_cpp_force_rebuild_once() -> None:
    if pymp.current_process().name != "MainProcess":
        return
    if os.environ.get(_ORBIT_WARS_CPP_FORCE_REBUILD, "").strip() != "1":
        return
    build_dir = Path(
        cpp_extension._get_build_directory(  # noqa: SLF001
            _ORBIT_WARS_CPP_EXT_NAME,
            verbose=os.environ.get("ORBIT_WARS_CPP_VERBOSE", "0") == "1",
        )
    )
    logging.info("ORBIT_WARS_CPP_FORCE_REBUILD: removing build dir once: %s", build_dir)
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ[_ORBIT_WARS_CPP_FORCE_REBUILD] = "0"


def main() -> None:
    _install_cuda_import_guard_in_entrypoint()
    _configure_torch_runtime()
    _prepare_orbit_wars_cpp_force_rebuild_once()

    parser = argparse.ArgumentParser(description="Run MonoBeast training")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a .py file that defines build_training_config()",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark actors instead of training",
    )
    parser.add_argument(
        "--benchmark_checkpoint_a",
        help="Baseline checkpoint path for benchmark model A",
    )
    parser.add_argument(
        "--benchmark_checkpoint_b",
        help="Challenger checkpoint path for benchmark model B",
    )
    parser.add_argument(
        "--benchmark_games",
        type=int,
        default=1000,
        help="Number of decisive non-draw benchmark games to collect",
    )
    parser.add_argument(
        "--benchmark_game_size",
        choices=("mixed", "2p", "4p"),
        default="4p",
        help="Benchmark game-size mode; mixed uses orbit_actor_target_4p_sample_ratio",
    )
    parser.add_argument(
        "--benchmark_position_seed",
        type=int,
        default=None,
        help="Optional RNG seed for per-episode seat assignment (model A vs B); actors offset the seed",
    )
    parser.add_argument(
        "--sample-false-a",
        action="store_true",
        help="Run benchmark model A with is_sample=False",
    )
    parser.add_argument(
        "--sample-false-b",
        action="store_true",
        help="Run benchmark model B with is_sample=False",
    )
    args = parser.parse_args()

    cfg = _load_training_config(args.config)
    assert isinstance(cfg, SimpleNamespace), "training config must be a SimpleNamespace from build_training_config()"

    mp.set_sharing_strategy(cfg.sharing_strategy)

    logging.info(os.path.dirname(__file__))
    logging.info(os.getcwd())

    if bool(args.benchmark):
        from src.torchbeast.benchmark import benchmark

        assert args.benchmark_checkpoint_a is not None, "--benchmark_checkpoint_a is required with --benchmark"
        assert args.benchmark_checkpoint_b is not None, "--benchmark_checkpoint_b is required with --benchmark"
        assert int(args.benchmark_games) > 0, "--benchmark_games must be > 0"
        checkpoint_a = checkpoint_config_from_checkpoint_file(args.benchmark_checkpoint_a)
        checkpoint_b = checkpoint_config_from_checkpoint_file(args.benchmark_checkpoint_b)
        cfg.benchmark = True
        cfg.benchmark_checkpoint_a = checkpoint_a
        cfg.benchmark_checkpoint_b = checkpoint_b
        cfg.benchmark_games = int(args.benchmark_games)
        cfg.benchmark_game_size = str(args.benchmark_game_size)
        cfg.benchmark_sample_a = not bool(args.sample_false_a)
        cfg.benchmark_sample_b = not bool(args.sample_false_b)
        cfg.resume_checkpoint = checkpoint_a
        cfg.enable_desync = False
        cfg.enable_wandb = False
        cfg.enable_tensorboard = False
        cfg.num_rl_vis_actors = 0
        benchmark(cfg)
    else:
        from src.torchbeast.monobeast import train

        kind = _run_artifact_kind_subdir(cfg)
        run_root = (
            _IMPALA_RUN_OUTPUT_ROOT
            / kind
            / f"{_format_run_timestamp()}_{_num_players_run_suffix()}"
        )
        os.environ[_ENV_RUN_ARTIFACT_ROOT] = str(run_root.resolve())
        _apply_impala_run_artifact_root(cfg, run_root)
        logging.info("Run artifact directory: %s", run_root.resolve())

        train(cfg)


if __name__ == "__main__":
    main()
