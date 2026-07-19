import json
import logging
import os
import time
import traceback
import copy
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from ..models.models import create_impala_model
from .monobeast_utils import (
    _compute_buffer_spec_via_spawn,
    _make_process,
    _mp_context_from_flags,
    _raise_if_uninitialized_module_tensors,
    _start_processes_in_threadpool,
    shutdown_all_processes,
)
from .core.bench_act import bench_act_func
from .core.common import (
    ProcessWithOptions,
    queue_get_or_timeout,
    queue_put_or_stop,
    checkpoint_model_config_from_checkpoint,
    compile_impala_model_for_rl,
    load_model_from_checkpoint,
    set_stop_event_with_reason,
)
from .core.create_buffers import allocate_cpu_buffers_from_spec
from .core.infer_bench import benchmark_inference_process
from .core.inference_worker import inference_batcher_process
from .core.visualize_actor import note_process_start_for_rl_debug_tape


@dataclass(frozen=True)
class _BenchmarkRuntime:
    ipc: dict[str, object]
    all_processes: list[ProcessWithOptions]


def _benchmark_inference_worker_devices(flags) -> list[int]:
    cuda0_workers = int(flags.num_benchmark_inference_workers_cuda0)
    cuda1_workers = int(flags.num_benchmark_inference_workers_cuda1)
    assert cuda0_workers >= 0, cuda0_workers
    assert cuda1_workers >= 0, cuda1_workers
    return [0] * cuda0_workers + [1] * cuda1_workers


def _num_benchmark_inference_workers_total(flags) -> int:
    return len(_benchmark_inference_worker_devices(flags))


def _flags_with_model_config(flags, model_config: dict):
    out = copy.copy(flags)
    out.model = model_config
    return out


def _flags_with_checkpoint_model_config(flags, checkpoint):
    model_config = checkpoint_model_config_from_checkpoint(checkpoint)
    return _flags_with_model_config(flags, model_config)


def _build_actor_model_from_checkpoint_for_device(
    *,
    flags,
    checkpoint,
    device_id: int,
    share_memory: bool,
    validate_initialized: bool,
) -> torch.nn.Module:
    device_str = f"cuda:{int(device_id)}"
    model = create_impala_model(_flags_with_checkpoint_model_config(flags, checkpoint))
    load_model_from_checkpoint(model, checkpoint)
    num_params = sum(int(p.numel()) for p in model.parameters())
    logging.info("Benchmark model parameter count: %d", num_params)
    model = model.to(device_str)
    model.eval()
    model = compile_impala_model_for_rl(model, dynamic=True)
    module_name = f"benchmark_actor_model[{device_str}]"
    if validate_initialized:
        _raise_if_uninitialized_module_tensors(model, module_name=module_name)
    if share_memory:
        model.share_memory()
    return model


def _create_benchmark_ipc(
    *,
    ctx,
    flags,
    infer_request_buffers_benchmark,
    infer_result_buffers_benchmark,
    stop_event,
):
    train_inference_workers_total = _num_benchmark_inference_workers_total(flags)
    assert train_inference_workers_total > 0, (
        "benchmark requires num_benchmark_inference_workers_cuda0 "
        "or num_benchmark_inference_workers_cuda1 > 0"
    )
    assert int(flags.num_actors) > 0, "benchmark requires num_actors > 0"

    n_infer_clients = int(flags.num_actors)
    infer_req_queue = ctx.Queue(maxsize=flags.num_inference_buffers_train)
    infer_batch_queue = ctx.Queue(maxsize=flags.num_inference_buffers_train)
    infer_res_queues = [
        ctx.Queue(maxsize=flags.num_inference_buffers_train) for _ in range(n_infer_clients)
    ]
    infer_req_free_queue = ctx.Queue(maxsize=flags.num_inference_buffers_train)
    for i in range(flags.num_inference_buffers_train):
        queue_put_or_stop(infer_req_free_queue, i, stop_event, timeout_sec=0.1)

    always_set_event = ctx.Event()
    always_set_event.set()
    benchmark_result_queue = ctx.Queue(maxsize=1000)
    benchmark_env_steps_total = ctx.Value("Q", 0)

    return dict(
        stop_event=stop_event,
        infer_req_queue_benchmark=infer_req_queue,
        infer_batch_queue_benchmark=infer_batch_queue,
        infer_res_queues_benchmark=infer_res_queues,
        infer_req_free_queue_benchmark=infer_req_free_queue,
        infer_request_buffers_benchmark=infer_request_buffers_benchmark,
        infer_result_buffers_benchmark=infer_result_buffers_benchmark,
        benchmark_result_queue=benchmark_result_queue,
        always_set_event=always_set_event,
        benchmark_env_steps_total=benchmark_env_steps_total,
    )


def _start_benchmark_cpu_processes_via_spawn(flags, ipc, *, ctx) -> list[ProcessWithOptions]:
    def make_proc(target, args: tuple, name: str) -> mp.Process:
        return _make_process(
            ctx=ctx,
            target=target,
            args=args,
            name=name,
            kind="cpu",
            torch_compile=False,
        )

    cpu_procs: list[ProcessWithOptions] = []
    cpu_procs.append(
        ProcessWithOptions(
            make_proc(
                inference_batcher_process,
                (
                    flags,
                    ipc["infer_req_queue_benchmark"],
                    ipc["infer_batch_queue_benchmark"],
                    "INFER_BATCHER_BENCH",
                    ipc["stop_event"],
                ),
                "INFER_BATCHER_BENCH",
            ),
            disable_gpu=True,
        )
    )

    for i in range(int(flags.num_actors)):
        cpu_procs.append(
            ProcessWithOptions(
                make_proc(
                    bench_act_func,
                    (
                        flags,
                        i,
                        i,
                        ipc["infer_req_queue_benchmark"],
                        ipc["infer_res_queues_benchmark"][i],
                        ipc["infer_req_free_queue_benchmark"],
                        ipc["infer_request_buffers_benchmark"],
                        ipc["infer_result_buffers_benchmark"],
                        ipc["benchmark_result_queue"],
                        ipc["benchmark_env_steps_total"],
                        f"BENCH_ACT_{i}",
                        ipc["always_set_event"],
                        ipc["stop_event"],
                    ),
                    f"BENCH_ACT_{i}",
                ),
                disable_gpu=True,
            )
        )

    _start_processes_in_threadpool(cpu_procs)
    return cpu_procs


def _start_benchmark_gpu_processes_via_spawn(flags, ipc, *, ctx) -> list[ProcessWithOptions]:
    procs: list[ProcessWithOptions] = []
    worker_devices = _benchmark_inference_worker_devices(flags)
    assert len(worker_devices) > 0, worker_devices
    benchmark_models_by_device: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    for actor_device in sorted(set(worker_devices)):
        benchmark_model_a = _build_actor_model_from_checkpoint_for_device(
            flags=flags,
            checkpoint=flags.benchmark_checkpoint_a,
            device_id=actor_device,
            share_memory=True,
            validate_initialized=True,
        )
        benchmark_model_b = _build_actor_model_from_checkpoint_for_device(
            flags=flags,
            checkpoint=flags.benchmark_checkpoint_b,
            device_id=actor_device,
            share_memory=True,
            validate_initialized=True,
        )
        benchmark_models_by_device[int(actor_device)] = (
            benchmark_model_a,
            benchmark_model_b,
        )

    for wi, actor_device in enumerate(worker_devices):
        benchmark_model_a, benchmark_model_b = benchmark_models_by_device[int(actor_device)]
        name = f"INFER_BENCH_CUDA{int(actor_device)}_{wi}"
        procs.append(
            ProcessWithOptions(
                _make_process(
                    ctx=ctx,
                    target=benchmark_inference_process,
                    args=(
                        wi,
                        flags,
                        ipc["infer_batch_queue_benchmark"],
                        ipc["infer_res_queues_benchmark"],
                        ipc["infer_request_buffers_benchmark"],
                        ipc["infer_result_buffers_benchmark"],
                        actor_device,
                        benchmark_model_a,
                        benchmark_model_b,
                        name,
                        ipc["stop_event"],
                    ),
                    name=name,
                    kind="gpu",
                    torch_compile=True,
                ),
                use_torch_compile=True,
            )
        )

    _start_processes_in_threadpool(procs)
    return procs


def _start_benchmark_runtime(flags, *, ctx, stop_event) -> _BenchmarkRuntime:
    assert hasattr(flags, "benchmark_checkpoint_a")
    assert hasattr(flags, "benchmark_checkpoint_b")
    assert int(flags.benchmark_games) > 0, flags.benchmark_games
    assert hasattr(flags, "benchmark_game_size")
    assert str(flags.benchmark_game_size) in ("mixed", "2p", "4p"), flags.benchmark_game_size
    assert hasattr(flags, "benchmark_sample_a")
    assert hasattr(flags, "benchmark_sample_b")

    logging.info(
        "Benchmark startup: checkpoint_a=%s checkpoint_b=%s game_size=%s sample_a=%s sample_b=%s",
        flags.benchmark_checkpoint_a,
        flags.benchmark_checkpoint_b,
        flags.benchmark_game_size,
        bool(flags.benchmark_sample_a),
        bool(flags.benchmark_sample_b),
    )
    assert not torch.cuda.is_initialized(), "CUDA must NOT be initialized in main during benchmark CPU-buffer allocation"
    spec_flags = _flags_with_checkpoint_model_config(flags, flags.benchmark_checkpoint_a)
    spec_flags.resume_checkpoint = flags.benchmark_checkpoint_a
    spec = _compute_buffer_spec_via_spawn(spec_flags, log_prefix="Benchmark startup")
    (
        _learn_buffers,
        _stats_buffers,
        infer_request_buffers_benchmark,
        infer_result_buffers_benchmark,
    ) = allocate_cpu_buffers_from_spec(flags=spec_flags, spec=spec)
    assert not torch.cuda.is_initialized(), "CUDA must NOT be initialized in main before benchmark CPU process start"

    ipc = _create_benchmark_ipc(
        ctx=ctx,
        flags=spec_flags,
        infer_request_buffers_benchmark=infer_request_buffers_benchmark,
        infer_result_buffers_benchmark=infer_result_buffers_benchmark,
        stop_event=stop_event,
    )
    cpu_procs = _start_benchmark_cpu_processes_via_spawn(spec_flags, ipc, ctx=ctx)
    logging.info(
        "Benchmark startup: CPU processes started via %s (count=%d)",
        flags.multiprocessing_start_method,
        len(cpu_procs),
    )
    assert not torch.cuda.is_initialized(), "CUDA must NOT be initialized in main after benchmark CPU process start"
    gpu_procs = _start_benchmark_gpu_processes_via_spawn(spec_flags, ipc, ctx=ctx)
    logging.info(
        "Benchmark startup: GPU processes started via %s (count=%d)",
        flags.multiprocessing_start_method,
        len(gpu_procs),
    )

    return _BenchmarkRuntime(
        ipc=ipc,
        all_processes=[*cpu_procs, *gpu_procs],
    )


def benchmark(flags) -> dict[str, object]:
    note_process_start_for_rl_debug_tape()
    logging.info("Starting benchmark")

    ctx = _mp_context_from_flags(flags)
    logging.info(
        "Benchmark multiprocessing_start_method=%s",
        flags.multiprocessing_start_method,
    )
    stop_event = ctx.Event()
    runtime = _start_benchmark_runtime(flags, ctx=ctx, stop_event=stop_event)
    ipc = runtime.ipc
    all_processes = runtime.all_processes

    games = 0
    a_wins = 0
    b_wins = 0
    draws = 0
    progress_interval = 1
    finished_normally = False
    result: dict[str, object] | None = None
    sps_log_interval_sec = 60.0
    target_games = int(flags.benchmark_games)
    last_sps_mono = time.monotonic()
    step_counter = ipc["benchmark_env_steps_total"]
    with step_counter.get_lock():
        last_sps_env_steps = int(step_counter.value)

    def _maybe_print_benchmark_sps() -> None:
        nonlocal last_sps_mono, last_sps_env_steps
        now = time.monotonic()
        if now - last_sps_mono < sps_log_interval_sec:
            return
        with step_counter.get_lock():
            cur_steps = int(step_counter.value)
        dt = now - last_sps_mono
        d_steps = cur_steps - last_sps_env_steps
        sps = (float(d_steps) / dt) if dt > 0.0 else 0.0
        print(
            "BENCHMARK_SPS "
            f"window_sec={dt:.3f} "
            f"env_steps_delta={d_steps} "
            f"sps={sps:.2f} "
            f"games={games}/{target_games} "
            f"a_wins={a_wins} b_wins={b_wins} draws_excluded={draws}",
            flush=True,
        )
        last_sps_mono = now
        last_sps_env_steps = cur_steps

    def _ensure_processes_alive() -> None:
        for proc_with_opts in all_processes:
            proc = proc_with_opts.process
            if proc.exitcode not in (None, 0):
                set_stop_event_with_reason(
                    stop_event,
                    process_name="benchmark",
                    reason="benchmark process exited",
                )
                raise RuntimeError(
                    f"Process {proc.name} exited with code {proc.exitcode}"
                )
            if stop_event.is_set():
                return
            if not proc.is_alive():
                set_stop_event_with_reason(
                    stop_event,
                    process_name="benchmark",
                    reason="benchmark process not alive",
                )
                raise RuntimeError(f"Process {proc.name} is not running")

    try:
        while games < target_games and not stop_event.is_set():
            item, timed_out = queue_get_or_timeout(
                ipc["benchmark_result_queue"],
                stop_event,
                timeout_sec=1.0,
            )
            if timed_out:
                _ensure_processes_alive()
                _maybe_print_benchmark_sps()
                continue
            assert isinstance(item, dict), type(item)
            assert "winner" in item, item
            winner = int(item["winner"])
            if winner == -1:
                draws += 1
            elif winner == 0:
                a_win_events = int(item["a_win_events"])
                b_win_events = int(item["b_win_events"])
                assert a_win_events > 0, item
                assert b_win_events == 0, item
                a_wins += a_win_events
                games += 1
            elif winner == 1:
                a_win_events = int(item["a_win_events"])
                b_win_events = int(item["b_win_events"])
                assert a_win_events == 0, item
                assert b_win_events > 0, item
                b_wins += b_win_events
                games += 1
            else:
                raise RuntimeError(f"Unexpected benchmark winner value: {winner}")
            if games > 0 and (games % progress_interval == 0 or games == target_games):
                win_events = int(a_wins) + int(b_wins)
                assert win_events > 0, (a_wins, b_wins)
                b_winrate_now = float(b_wins) / float(win_events)
                with step_counter.get_lock():
                    env_steps_total = int(step_counter.value)
                print(
                    "BENCHMARK_PROGRESS "
                    f"games={games}/{target_games} "
                    f"a_wins={a_wins} b_wins={b_wins} "
                    f"b_winrate={b_winrate_now:.4f} "
                    f"draws_excluded={draws} "
                    f"env_steps_total={env_steps_total}",
                    flush=True,
                )
            _maybe_print_benchmark_sps()
            _ensure_processes_alive()

        assert games == target_games, (games, target_games)
        win_events = int(a_wins) + int(b_wins)
        assert win_events > 0, (a_wins, b_wins)
        b_winrate = float(b_wins) / float(win_events)
        result = {
            "games": int(games),
            "a_wins": int(a_wins),
            "b_wins": int(b_wins),
            "pairwise_win_events": int(win_events),
            "draws_excluded": int(draws),
            "b_winrate": float(b_winrate),
            "seat_assignment": "reshuffled_each_episode",
            "scoring": "pairwise_opponent_seats",
            "benchmark_game_size": str(flags.benchmark_game_size),
            "benchmark_sample_a": bool(flags.benchmark_sample_a),
            "benchmark_sample_b": bool(flags.benchmark_sample_b),
        }
        logging.info("Benchmark finished: %s", result)
        finished_normally = True

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received in benchmark, shutting down")
    except Exception:
        logging.info(traceback.format_exc())

    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name="benchmark",
            reason="benchmark finally",
        )
        logging.info("Waiting for benchmark processes to finish")
        shutdown_all_processes(all_processes, timeout_sec=1200)

    logging.info("Exit?")
    if finished_normally:
        assert result is not None
        logging.info("Exit 0")
        payload = json.dumps(result, sort_keys=True)
        print(f"BENCHMARK_RESULT {payload}", flush=True)
        os._exit(0)
    logging.info("Exit -1")
    os._exit(-1)

