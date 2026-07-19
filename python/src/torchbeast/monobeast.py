import copy
import gc
import logging
import torch.multiprocessing as mp
import os
import threading
import time
import timeit
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import setproctitle
import torch
import yaml

from .core.act import act_rollout_func
from .core.inference_worker import inference_batcher_process, inference_process
from ..models.models import create_impala_model
from .monobeast_utils import (
    _compute_buffer_spec_via_spawn,
    _make_process,
    _mp_context_from_flags,
    _mp_context_from_start_method,
    _raise_if_uninitialized_module_tensors,
    _start_processes_in_threadpool,
    shutdown_all_processes,
)
from .core.visualize_actor import (
    note_process_start_for_rl_debug_tape,
    process_start_rl_debug_tape_name,
    run_rl_visualization_tape_consumer,
    visualize_process_with_scene_queue,
)
from .core.batch_and_learn import batch_and_learn
from .core.batch_prepare import batch_prepare_process_func
from .core.create_buffers import (
    allocate_cpu_buffers_from_spec,
    allocate_gpu_buffers_from_spec,
)
from .core.stats import stats_func
from .core.common import (
    compile_impala_model_for_rl,
    checkpoint_model_config_from_checkpoint_or_fallback,
    compute_lr_lambda,
    configure_process_cpu_thread_limits,
    ENTROPY_HEAD_KEYS,
    entropy_ipc_tuple_for_resume,
    initial_entropy_ipc_from_flags,
    initial_new_controller_temperature_threshold_from_flags,
    load_model_from_resume_sources,
    new_controller_temperature_threshold_tuple_for_resume,
)
from .core.common import get_checkpoint_file
from .core.common import _checkpoint_reader_from_cfg
from .core.common import (
    ProcessWithOptions,
    StopRequested,
    raise_if_stop_requested,
    queue_put_or_stop,
    set_stop_event_with_reason,
    stop_event_wait_or_raise,
)


@dataclass(frozen=True)
class _TrainingRuntime:
    ipc: dict[str, object]
    all_processes: list[ProcessWithOptions]
    popart_manager: object | None
    learner_gpu_buffers: object | None


def _yaml_plain_value(value):
    if isinstance(value, SimpleNamespace):
        return {key: _yaml_plain_value(item) for key, item in vars(value).items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _yaml_plain_value(asdict(value))
    if isinstance(value, dict):
        return {key: _yaml_plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_yaml_plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_yaml_artifact(path: Path, payload) -> None:
    path.write_text(
        yaml.safe_dump(
            _yaml_plain_value(payload),
            sort_keys=False,
            allow_unicode=False,
        )
    )


def _save_run_config_artifacts(flags) -> None:
    assert hasattr(flags, "output_dir"), "training flags must include output_dir"
    output_dir = Path(flags.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml_artifact(output_dir / "model_config.yaml", flags.model)
    _write_yaml_artifact(output_dir / "launch_flags.yaml", flags)


def _has_training_actors(flags) -> bool:
    return int(flags.num_actors) > 0


def _is_visualization_only_mode(flags) -> bool:
    return (not _has_training_actors(flags)) and int(flags.num_rl_vis_actors) > 0


def _train_inference_worker_devices(flags) -> list[int]:
    cuda0_workers = int(flags.num_inference_workers_train_cuda0)
    cuda1_workers = int(flags.num_inference_workers_train_cuda1)
    assert cuda0_workers >= 0, cuda0_workers
    assert cuda1_workers >= 0, cuda1_workers
    return [0] * cuda0_workers + [1] * cuda1_workers


def _num_train_inference_workers_total(flags) -> int:
    return len(_train_inference_worker_devices(flags))


def _flags_for_train_actor(flags, actor_index: int):
    _ = actor_index
    assert int(flags.orbit_num_agents) == 4, int(flags.orbit_num_agents)
    assert int(flags.agents_max_cnt) == 4, int(flags.agents_max_cnt)
    actor_flags = copy.copy(flags)
    return actor_flags


def _num_train_actor_processes(flags) -> int:
    n_act = int(flags.num_actors)
    n_app = int(flags.n_actors_per_process)
    assert n_act % n_app == 0, (n_act, n_app)
    return n_act // n_app


def _train_actor_group_rollout_main(
    flags,
    ipc: dict[str, object],
    group_index: int,
    group_global_indices: tuple[int, ...],
) -> None:
    try:
        n_app = int(flags.n_actors_per_process)
        assert len(group_global_indices) == n_app, (len(group_global_indices), n_app)
        setproctitle.setproctitle(f"RL_TRAIN_GRP_{group_index}")
        configure_process_cpu_thread_limits()
        threads: list[threading.Thread] = []
        for g in group_global_indices:
            actor_flags = _flags_for_train_actor(flags, g)
            name = f"RL_TRAIN_{g}"
            actor_args = (
                actor_flags,
                g,
                g,
                ipc["free_queue"],
                ipc["full_queue"],
                ipc["full_queue_lock"],
                ipc["learn_buffers"],
                ipc["stats_free_queue"],
                ipc["stats_full_queue"],
                ipc["stats_buffers"],
                ipc["infer_req_queue_train"],
                ipc["infer_res_queues_train"][g],
                ipc["infer_req_free_queue_train"],
                ipc["infer_request_buffers_train"],
                ipc["infer_result_buffers_train"],
                name,
                ipc["always_set_event"],
                ipc["stop_event"],
                ipc["visualization_queue_rl"],
                ipc["actor_desync_done_env_count"],
                False,
            )
            threads.append(
                threading.Thread(target=act_rollout_func, args=actor_args, name=name, daemon=False)
            )
        for t in threads:
            t.start()
        for t in threads:
            t.join()


    except KeyboardInterrupt:
        pass
    except StopRequested:
        logging.info("%s received StopRequested, exiting", f"RL_TRAIN_GRP_{group_index}")
    except Exception:
        logging.exception("%s failed in _train_actor_group_rollout_main", f"RL_TRAIN_GRP_{group_index}")
    finally:
        set_stop_event_with_reason(
            ipc["stop_event"],
            process_name=f"RL_TRAIN_GRP_{group_index}",
            reason="_train_actor_group_rollout_main finally",
        )
        os._exit(0)



def _build_actor_model_for_device(
    *,
    flags,
    device_id: int,
    share_memory: bool,
    validate_initialized: bool,
) -> torch.nn.Module:
    device_str = f"cuda:{int(device_id)}"
    model = create_impala_model(flags)
    # Load checkpoint weights before moving the visualization model onto its CUDA device.
    load_model_from_resume_sources(
        model,
        resume_checkpoint=flags.resume_checkpoint,
        load_as_much_as_possible=flags.load_as_much_as_possible,
    )
    num_params = sum(int(p.numel()) for p in model.parameters())
    logging.info("Model parameter count: %d", num_params)
    model = model.to(device_str)
    model.eval()
    model = compile_impala_model_for_rl(model, dynamic=True)
    module_name = f"actor_model[{device_str}]"
    if validate_initialized:
        _raise_if_uninitialized_module_tensors(model, module_name=module_name)
    if share_memory:
        model.share_memory()
    return model


def _create_actor_weight_buffers(flags) -> dict[str, torch.Tensor]:
    model = create_impala_model(flags)
    load_model_from_resume_sources(
        model,
        resume_checkpoint=flags.resume_checkpoint,
        load_as_much_as_possible=flags.load_as_much_as_possible,
    )
    model = compile_impala_model_for_rl(model, dynamic=True)
    state_dict = model.state_dict()
    buffers: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        assert isinstance(value, torch.Tensor), (
            f"model state_dict values must be tensors, got {type(value)} at {key}"
        )
        buffers[key] = torch.zeros(
            tuple(value.shape),
            dtype=value.dtype,
            device="cpu",
        ).share_memory_()
    return buffers


def _apply_resume_checkpoint_model_config(flags) -> None:
    if flags.resume_checkpoint is None:
        return
    if not flags.use_model_config_from_checkpoint:
        return
    flags.model = checkpoint_model_config_from_checkpoint_or_fallback(
        flags.resume_checkpoint,
        flags.checkpoint_model_config_fallback_checkpoint,
        flags.model,
    )


def _create_cpu_ipc(
    *,
    ctx,
    flags,
    learn_buffers,
    stats_buffers,
    infer_request_buffers_train,
    infer_result_buffers_train,
    checkpoint_steps: int,
    stop_event,
    resume_entropy_ipc: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None,
    resume_new_controller_temperature_threshold: tuple[float, ...] | None = None,
):
    train_inference_workers_total = _num_train_inference_workers_total(flags)
    assert int(flags.num_actors) > 0, "_create_cpu_ipc requires num_actors > 0"

    free_queue = ctx.Queue(maxsize=int(flags.num_buffers))
    full_queue = ctx.Queue(maxsize=int(flags.num_buffers))
    full_queue_lock = ctx.Lock()
    batch_queues = [ctx.Queue(maxsize=flags.prepare_batches)]
    learner_free_batch_queues = [ctx.Queue(maxsize=flags.prepare_batches)]
    for q in learner_free_batch_queues:
        for i in range(flags.prepare_batches):
            queue_put_or_stop(q, i, stop_event, timeout_sec=0.1)

    for i in range(int(flags.num_buffers)):
        queue_put_or_stop(free_queue, i, stop_event, timeout_sec=0.1)

    stats_queue_learner_train = ctx.Queue(maxsize=100)

    stats_free_queue = ctx.Queue(maxsize=flags.num_stats_buffers)
    stats_full_queue = ctx.Queue(maxsize=flags.num_stats_buffers)

    for local_idx in range(int(flags.num_stats_buffers)):
        queue_put_or_stop(stats_free_queue, int(local_idx), stop_event, timeout_sec=0.1)

    n_infer_clients_train = int(flags.num_actors)

    infer_req_queue_train = ctx.Queue(maxsize=flags.num_inference_buffers_train)
    infer_batch_queue_train = ctx.Queue(maxsize=flags.num_inference_buffers_train)
    infer_res_queues_train = [
        ctx.Queue(maxsize=flags.num_inference_buffers_train) for _ in range(n_infer_clients_train)
    ]
    infer_req_free_queue_train = ctx.Queue(maxsize=flags.num_inference_buffers_train)
    for i in range(flags.num_inference_buffers_train):
        queue_put_or_stop(infer_req_free_queue_train, i, stop_event, timeout_sec=0.1)
    actor_weight_buffers_train = _create_actor_weight_buffers(flags)
    actor_weight_update_queues_train = [
        ctx.Queue(maxsize=1)
        for _ in range(train_inference_workers_total)
    ]
    actor_weight_ack_queue_train = ctx.Queue(maxsize=max(1, train_inference_workers_total))

    checkpoint_queue = ctx.Queue(maxsize=0)

    always_set_event = ctx.Event()
    always_set_event.set()

    latest_checkpoint_steps = ctx.Value("q", int(checkpoint_steps))

    shared_lr_lambda = ctx.Value(
        "d",
        compute_lr_lambda(
            step=int(checkpoint_steps),
            total_steps=float(flags.total_steps),
            lr_warmup_steps=int(flags.lr_warmup_steps),
            initial_lr_lambda=float(flags.initial_lr_lambda),
            final_lr_lambda=float(flags.final_lr_lambda),
        ),
    )
    if resume_entropy_ipc is not None:
        init_te, init_sf, init_ema = resume_entropy_ipc
    else:
        init_te, init_sf, init_ema = initial_entropy_ipc_from_flags(
            flags, int(checkpoint_steps)
        )
    if resume_new_controller_temperature_threshold is not None:
        init_controller_threshold = resume_new_controller_temperature_threshold
    else:
        init_controller_threshold = initial_new_controller_temperature_threshold_from_flags(flags)
    assert len(init_te) == len(ENTROPY_HEAD_KEYS)
    assert len(init_sf) == len(ENTROPY_HEAD_KEYS)
    assert len(init_ema) == len(ENTROPY_HEAD_KEYS)
    assert len(init_controller_threshold) == len(ENTROPY_HEAD_KEYS)
    shared_target_entropy = ctx.Array("d", [float(v) for v in init_te])
    shared_shortfall_entropy = ctx.Array("d", [float(v) for v in init_sf])
    shared_mean_entropy_ema = ctx.Array("d", [float(v) for v in init_ema])
    shared_new_controller_temperature_threshold = ctx.Array(
        "d",
        [float(v) for v in init_controller_threshold],
    )
    shared_steps = ctx.Value("d", float(checkpoint_steps))

    visualization_queue_rl = None
    if int(flags.num_actors) > 0 and bool(flags.enable_rl_actor_visualization):
        visualization_queue_rl = ctx.Queue(maxsize=100)

    na_ipc = int(flags.num_actors)
    actor_desync_done_env_count = ctx.Array("i", na_ipc)

    return dict(
        free_queue=free_queue,
        full_queue=full_queue,
        full_queue_lock=full_queue_lock,
        batch_queues=batch_queues,
        learner_free_batch_queues=learner_free_batch_queues,
        stats_free_queue=stats_free_queue,
        stats_full_queue=stats_full_queue,
        stats_buffers=stats_buffers,
        stats_queue_learner_train=stats_queue_learner_train,
        infer_req_queue_train=infer_req_queue_train,
        infer_batch_queue_train=infer_batch_queue_train,
        infer_res_queues_train=infer_res_queues_train,
        infer_req_free_queue_train=infer_req_free_queue_train,
        actor_weight_buffers_train=actor_weight_buffers_train,
        actor_weight_update_queues_train=actor_weight_update_queues_train,
        actor_weight_ack_queue_train=actor_weight_ack_queue_train,
        checkpoint_queue=checkpoint_queue,
        infer_request_buffers_train=infer_request_buffers_train,
        infer_result_buffers_train=infer_result_buffers_train,
        always_set_event=always_set_event,
        stop_event=stop_event,
        latest_checkpoint_steps=latest_checkpoint_steps,
        learn_buffers=learn_buffers,
        shared_lr_lambda=shared_lr_lambda,
        shared_target_entropy=shared_target_entropy,
        shared_shortfall_entropy=shared_shortfall_entropy,
        shared_mean_entropy_ema=shared_mean_entropy_ema,
        shared_new_controller_temperature_threshold=shared_new_controller_temperature_threshold,
        shared_steps=shared_steps,
        start_steps=int(checkpoint_steps),
        visualization_queue_rl=visualization_queue_rl,
        actor_desync_done_env_count=actor_desync_done_env_count,
    )


def _start_cpu_processes_via_spawn(flags, ipc, *, ctx) -> list[ProcessWithOptions]:
    train_inference_workers_total = _num_train_inference_workers_total(flags)

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

    # STATS (blocks until stats_full_queue has data; fine for startup)
    cpu_procs.append(
        ProcessWithOptions(
            make_proc(
                stats_func,
                (
                    flags,
                    ipc["stats_free_queue"],
                    ipc["stats_full_queue"],
                    ipc["stats_buffers"],
                    ipc["stats_queue_learner_train"],
                    ipc["shared_target_entropy"],
                    ipc["shared_shortfall_entropy"],
                    ipc["shared_mean_entropy_ema"],
                    ipc["shared_new_controller_temperature_threshold"],
                    ipc["shared_steps"],
                    ipc["shared_lr_lambda"],
                    "STATS_TRAIN",
                    ipc["stop_event"],
                ),
                "STATS_TRAIN",
            ),
            disable_gpu=True,
        )
    )

    if train_inference_workers_total > 0:
        # Inference request batcher: groups (actor_id, req_idx) into fixed-size batches and sends IDs only.
        cpu_procs.append(
            ProcessWithOptions(
                make_proc(
                    inference_batcher_process,
                    (
                        flags,
                        ipc["infer_req_queue_train"],
                        ipc["infer_batch_queue_train"],
                        "INFER_BATCHER_TRAIN",
                        ipc["stop_event"],
                    ),
                    "INFER_BATCHER_TRAIN",
                ),
                disable_gpu=True,
            )
        )

    n_infer_clients_train = int(flags.num_actors)
    n_app = int(flags.n_actors_per_process)
    n_groups = _num_train_actor_processes(flags)
    assert n_groups * n_app == n_infer_clients_train, (n_groups, n_app, n_infer_clients_train)

    # RL actors: ``n_groups`` processes, each runs ``n_app`` threads with global indices 0..num_actors-1.
    if int(flags.num_actors) > 0:
        for group_index in range(n_groups):
            base = int(group_index * n_app)
            group_global_indices = tuple(range(base, base + n_app))
            cpu_procs.append(
                ProcessWithOptions(
                    make_proc(
                        _train_actor_group_rollout_main,
                        (
                            flags,
                            ipc,
                            group_index,
                            group_global_indices,
                        ),
                        f"RL_TRAIN_GRP_{group_index}",
                    ),
                    disable_gpu=True,
                )
            )

    if ipc["visualization_queue_rl"] is not None:
        run_tape_name = process_start_rl_debug_tape_name()
        cpu_procs.append(
            ProcessWithOptions(
                make_proc(
                    run_rl_visualization_tape_consumer,
                    (
                        ipc["visualization_queue_rl"],
                        ipc["stop_event"],
                        run_tape_name,
                    ),
                    "RL_VIS_TAPE",
                ),
                disable_gpu=True,
            )
        )

    _start_processes_in_threadpool(cpu_procs)

    return cpu_procs


def _start_gpu_processes_via_spawn(
    *,
    ctx,
    flags,
    ipc,
    learner_gpu_buffers,
    spec,
    popart_shared_dict,
    popart_lock,
) -> list[ProcessWithOptions]:
    """
    Start GPU processes (inference workers, batch_prepare, learners) using spawn.
    CPU actors should already be running and will begin to make progress once inference + batch_prepare are up.
    """
    procs: list[ProcessWithOptions] = []
    train_inference_workers_total = _num_train_inference_workers_total(flags)

    worker_devices = _train_inference_worker_devices(flags)

    learner_device_shift = int(flags.learner_cuda_device)

    # Inference workers
    def start_inference_workers(
        split: str,
        num_workers: int,
        infer_batch_queue,
        infer_res_queues,
        infer_request_buffers,
        infer_result_buffers,
    ) -> None:
        assert int(num_workers) == len(worker_devices), (num_workers, worker_devices)
        actor_weight_update_queues = ipc["actor_weight_update_queues_train"]
        assert len(actor_weight_update_queues) == len(worker_devices), (
            len(actor_weight_update_queues),
            len(worker_devices),
        )
        for wi, inference_cuda_device in enumerate(worker_devices):
            inference_cuda_device = int(inference_cuda_device)
            name = f"INFER_{split}_CUDA{inference_cuda_device}_{wi}"
            procs.append(
                ProcessWithOptions(
                    _make_process(
                        ctx=ctx,
                        target=inference_process,
                        args=(
                            wi,
                            flags,
                            infer_batch_queue,
                            infer_res_queues,
                            infer_request_buffers,
                            infer_result_buffers,
                            ipc["full_queue"],
                            inference_cuda_device,
                            ipc["actor_weight_buffers_train"],
                            actor_weight_update_queues[wi],
                            ipc["actor_weight_ack_queue_train"],
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

    if flags.inference_batch_size > 0 and train_inference_workers_total > 0:
        start_inference_workers(
            "TRAIN",
            train_inference_workers_total,
            ipc["infer_batch_queue_train"],
            ipc["infer_res_queues_train"],
            ipc["infer_request_buffers_train"],
            ipc["infer_result_buffers_train"],
        )

    # Batch prepare (GPU)
    def create_batch_prepare_worker(i: int) -> ProcessWithOptions:
        rank = i // flags.n_batch_prepare_processes
        device = learner_device_shift + rank
        name = f"BATCH_PREP_{i}"
        return ProcessWithOptions(
            _make_process(
                ctx=ctx,
                target=batch_prepare_process_func,
                args=(
                    flags,
                    ipc["full_queue"],
                    ipc["full_queue_lock"],
                    ipc["batch_queues"][rank],
                    device,
                    ipc["learn_buffers"],
                    ipc["learner_free_batch_queues"][rank],
                    learner_gpu_buffers[rank],
                    ipc["free_queue"],
                    name,
                    ipc["stop_event"],
                ),
                name=name,
                kind="gpu",
                torch_compile=False,
            )
        )

    if int(flags.num_actors) > 0 and len(learner_gpu_buffers[0]) > 0:
        for i in range(flags.n_batch_prepare_processes):
            procs.append(create_batch_prepare_worker(i))

    # Learners
    if int(flags.num_actors) > 0:
        name = "LEARN_0"
        procs.append(
            ProcessWithOptions(
                _make_process(
                    ctx=ctx,
                    target=batch_and_learn,
                    args=(
                        flags,
                        ipc["shared_steps"],
                        ipc["batch_queues"],
                        ipc["shared_lr_lambda"],
                        ipc["shared_target_entropy"],
                        ipc["shared_shortfall_entropy"],
                        ipc["shared_mean_entropy_ema"],
                        ipc["shared_new_controller_temperature_threshold"],
                        ipc["stats_queue_learner_train"],
                        ipc["learner_free_batch_queues"],
                        learner_gpu_buffers[0],
                        ipc["actor_weight_buffers_train"],
                        ipc["actor_weight_update_queues_train"],
                        ipc["actor_weight_ack_queue_train"],
                        popart_shared_dict,
                        popart_lock,
                        ipc["checkpoint_queue"],
                        name,
                        ipc["stop_event"],
                        ipc["latest_checkpoint_steps"],
                        int(flags.learner_cuda_device),
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


def _start_training_runtime(
    flags,
    *,
    ipc_ctx,
    cpu_ctx,
    gpu_ctx,
    stop_event,
    popart_shared_dict,
    popart_lock,
    popart_manager,
) -> _TrainingRuntime:
    _apply_resume_checkpoint_model_config(flags)
    _save_run_config_artifacts(flags)

    if _is_visualization_only_mode(flags):
        logging.info(
            "Startup: visualize-only mode enabled (no training actors, num_rl_vis_actors=%d)",
            int(flags.num_rl_vis_actors),
        )
        actor_device = int(flags.inference_cuda_device)
        assert int(flags.learner_cuda_device) == actor_device, (
            flags.learner_cuda_device,
            actor_device,
        )
        actor_model = _build_actor_model_for_device(
            flags=flags,
            device_id=actor_device,
            share_memory=True,
            validate_initialized=True,
        )

        ipc = {
            "stop_event": stop_event,
        }
        visualization_queue_rl = ipc_ctx.Queue(maxsize=100)
        ipc["visualization_queue_rl"] = visualization_queue_rl
        run_tape_name = process_start_rl_debug_tape_name()
        all_processes: list[ProcessWithOptions] = [
            ProcessWithOptions(
                _make_process(
                    ctx=gpu_ctx,
                    target=run_rl_visualization_tape_consumer,
                    args=(visualization_queue_rl, stop_event, run_tape_name),
                    name="RL_VIS_TAPE",
                    kind="cpu",
                    torch_compile=False,
                ),
                disable_gpu=True,
            )
        ]
        for i in range(flags.num_rl_vis_actors):
            all_processes.append(
                ProcessWithOptions(
                    _make_process(
                        ctx=gpu_ctx,
                        target=visualize_process_with_scene_queue,
                        args=(
                            flags,
                            i,
                            actor_model,
                            actor_device,
                            f"VISUALIZE_RL_{i}",
                            visualization_queue_rl,
                            stop_event,
                        ),
                        name=f"VISUALIZE_RL_{i}",
                        kind="gpu",
                        torch_compile=False,
                    )
                )
            )
        _start_processes_in_threadpool(all_processes)

        return _TrainingRuntime(
            ipc=ipc,
            all_processes=all_processes,
            popart_manager=popart_manager,
            learner_gpu_buffers=None,
        )

    logging.info(
        "Startup: before buffer spec, torch.cuda.is_initialized()=%s",
        str(torch.cuda.is_initialized()),
    )
    assert not torch.cuda.is_initialized(), "CUDA must NOT be initialized in main during CPU-buffer allocation stage"

    spec = _compute_buffer_spec_via_spawn(flags, log_prefix="Startup")
    assert int(flags.num_actors) > 0, "num_actors must be > 0 for training"
    (
        learn_buffers,
        stats_buffers,
        infer_request_buffers_train,
        infer_result_buffers_train,
    ) = allocate_cpu_buffers_from_spec(flags=flags, spec=spec)
    logging.info(
        "Startup: allocated CPU buffers (learn=%d, stats=%d, infer_train=%d)",
        len(learn_buffers),
        len(stats_buffers),
        len(infer_request_buffers_train),
    )
    logging.info(
        "Startup: after buffer spec, torch.cuda.is_initialized()=%s",
        str(torch.cuda.is_initialized()),
    )
    assert not torch.cuda.is_initialized(), "CUDA must NOT be initialized in main during CPU-buffer allocation stage"

    initial_steps = 0
    resume_entropy_ipc = None
    resume_new_controller_temperature_threshold = None
    if flags.resume_checkpoint and (not flags.start_from_scratch):
        reader = _checkpoint_reader_from_cfg(flags.resume_checkpoint)
        ckpt_name = get_checkpoint_file(reader, flags.resume_checkpoint.name)
        ckpt_file = reader.read_torch(
            ckpt_name,
            map_location="cpu",
            weights_only=False,
        )
        assert "steps" in ckpt_file, "Resume checkpoint missing steps"
        initial_steps = int(ckpt_file["steps"])
        has_all_entropy = "shared_target_entropy_by_head" in ckpt_file
        resume_entropy_ipc = entropy_ipc_tuple_for_resume(
            ckpt_file,
            flags,
            initial_steps,
        )
        resume_new_controller_temperature_threshold = (
            new_controller_temperature_threshold_tuple_for_resume(
                ckpt_file,
                flags,
            )
        )
        if not has_all_entropy:
            logging.info(
                "Resume checkpoint missing split-head entropy fields; initialized from config at steps=%d",
                initial_steps,
            )

    ipc = _create_cpu_ipc(
        ctx=ipc_ctx,
        flags=flags,
        learn_buffers=learn_buffers,
        stats_buffers=stats_buffers,
        infer_request_buffers_train=infer_request_buffers_train,
        infer_result_buffers_train=infer_result_buffers_train,
        checkpoint_steps=initial_steps,
        stop_event=stop_event,
        resume_entropy_ipc=resume_entropy_ipc,
        resume_new_controller_temperature_threshold=resume_new_controller_temperature_threshold,
    )
    cpu_procs = _start_cpu_processes_via_spawn(flags, ipc, ctx=cpu_ctx)
    logging.info(
        "Startup: CPU processes started via %s (count=%d)",
        flags.multiprocessing_start_method,
        len(cpu_procs),
    )

    assert not torch.cuda.is_initialized(), "CUDA must NOT be initialized in main after CPU process start"
    logging.info("Startup: allocating GPU learner buffers in main (this will initialize CUDA)")
    learner_gpu_buffers = allocate_gpu_buffers_from_spec(
        flags=flags,
        spec=spec,
        learner_cuda_device=int(flags.learner_cuda_device),
    )
    logging.info(
        "Startup: allocated GPU learner buffers (prepare_batches=%d)",
        int(flags.prepare_batches),
    )
    logging.info(
        "Startup: after GPU alloc, torch.cuda.is_initialized()=%s",
        str(torch.cuda.is_initialized()),
    )
    assert torch.cuda.is_initialized(), "Expected CUDA to be initialized in main after GPU buffer allocation"

    logging.info(
        "Startup: starting GPU processes via %s",
        "spawn",
    )
    if flags.enable_popart:
        assert popart_shared_dict is not None, "enable_popart requires popart_shared_dict"
        assert popart_lock is not None, "enable_popart requires popart_lock"
        assert popart_manager is not None, "enable_popart requires popart_manager"
    else:
        assert popart_shared_dict is None, "popart_shared_dict must be None when enable_popart is false"
        assert popart_lock is None, "popart_lock must be None when enable_popart is false"
        assert popart_manager is None, "popart_manager must be None when enable_popart is false"
    gpu_procs = _start_gpu_processes_via_spawn(
        ctx=gpu_ctx,
        flags=flags,
        ipc=ipc,
        learner_gpu_buffers=learner_gpu_buffers,
        spec=spec,
        popart_shared_dict=popart_shared_dict,
        popart_lock=popart_lock,
    )
    logging.info(
        "Startup: GPU processes started via %s (count=%d)",
        "spawn",
        len(gpu_procs),
    )

    del spec

    all_processes = [*cpu_procs, *gpu_procs]
    logging.info("All processes started (cpu=%d, gpu=%d)", len(cpu_procs), len(gpu_procs))

    return _TrainingRuntime(
        ipc=ipc,
        all_processes=all_processes,
        popart_manager=popart_manager,
        learner_gpu_buffers=learner_gpu_buffers,
    )


def train(flags, main_io_writer=None) -> None:
    note_process_start_for_rl_debug_tape()
    logging.info("Starting training")

    assert not (
        int(flags.num_actors) > 0 and int(flags.num_rl_vis_actors) > 0
    ), "training rollouts (num_actors>0) cannot run together with num_rl_vis_actors>0"

    cpu_ctx = _mp_context_from_flags(flags)
    gpu_ctx = _mp_context_from_start_method("spawn")
    ipc_ctx = gpu_ctx
    logging.info(
        "Training multiprocessing_start_method=%s",
        flags.multiprocessing_start_method,
    )
    logging.info("Training gpu_multiprocessing_start_method=spawn")
    stop_event = ipc_ctx.Event()
    all_processes: list[ProcessWithOptions] = []
    finished_normally = False

    try:
        setproctitle.setproctitle('MAIN')

        popart_manager = None
        popart_shared_dict = None
        popart_lock = None
        if flags.enable_popart:
            # IMPORTANT: keep the Manager alive for the whole training run; otherwise PopArt proxies
            # will start failing with BrokenPipeError once the manager is GC'd/shutdown.
            popart_manager = ipc_ctx.Manager()
            popart_shared_dict = popart_manager.dict()
            popart_lock = popart_manager.RLock()

        runtime = _start_training_runtime(
            flags=flags,
            ipc_ctx=ipc_ctx,
            cpu_ctx=cpu_ctx,
            gpu_ctx=gpu_ctx,
            stop_event=stop_event,
            popart_shared_dict=popart_shared_dict,
            popart_lock=popart_lock,
            popart_manager=popart_manager,
        )
        
        ipc = runtime.ipc
        all_processes = runtime.all_processes
        timer = timeit.default_timer
        if _is_visualization_only_mode(flags):
            logging.info("Running in visualize-only mode: training loops are disabled")
            while not stop_event.is_set():
                for proc_with_opts in all_processes:
                    proc = proc_with_opts.process
                    if proc.exitcode not in (None, 0):
                        set_stop_event_with_reason(
                            stop_event,
                            process_name="monobeast",
                            reason="visualization process exited",
                        )
                        raise RuntimeError(
                            f"Process {proc.name} exited with code {proc.exitcode}"
                        )
                    if not proc.is_alive():
                        set_stop_event_with_reason(
                            stop_event,
                            process_name="monobeast",
                            reason="visualization process not alive",
                        )
                        raise RuntimeError(f"Process {proc.name} is not running")
                stop_event_wait_or_raise(stop_event, timeout_sec=1.0)

        sps_sleep_sec = 30.0

        finalization_poll_sec = 10
        finalization_timeout_sec = 3600.0

        def _log_processes_snapshot(prefix: str) -> None:
            rows = []
            for proc_with_opts in all_processes:
                proc = proc_with_opts.process
                rows.append(
                    f"{proc.name}(pid={proc.pid},alive={proc.is_alive()},exitcode={proc.exitcode})"
                )
            logging.info("%s: %s", prefix, ", ".join(rows))

        def _ensure_processes_alive() -> None:
            for proc_with_opts in all_processes:
                if proc_with_opts.process.exitcode not in (None, 0):
                    set_stop_event_with_reason(
                        stop_event,
                        process_name="monobeast",
                        reason="worker process exited",
                    )
                    raise RuntimeError(
                        f"Process {proc_with_opts.process.name} exited with code {proc_with_opts.process.exitcode}"
                    )
                if stop_event.is_set():
                    return
                if not proc_with_opts.process.is_alive():
                    set_stop_event_with_reason(
                        stop_event,
                        process_name="monobeast",
                        reason="worker process not alive",
                    )
                    raise RuntimeError(f"Process {proc_with_opts.process.name} is not running")

        def _wait_for_final_checkpoint(target_steps: int) -> int:
            deadline = time.time() + finalization_timeout_sec
            while int(ipc["latest_checkpoint_steps"].value) < int(target_steps):
                _ensure_processes_alive()
                if time.time() >= deadline:
                    set_stop_event_with_reason(
                        stop_event,
                        process_name="monobeast",
                        reason="final checkpoint timeout",
                    )
                    raise RuntimeError("Timed out waiting for final checkpoint write")
                logging.info(
                    f"Waiting for final checkpoint write: {int(ipc['latest_checkpoint_steps'].value)} < {int(target_steps)}"
                )
                stop_event_wait_or_raise(stop_event, timeout_sec=finalization_poll_sec)
            return int(ipc["latest_checkpoint_steps"].value)

        while int(ipc["shared_steps"].value) < flags.total_steps and not stop_event.is_set():
            start_step = int(ipc["shared_steps"].value)
            start_time = timer()
            stop_event_wait_or_raise(stop_event, timeout_sec=sps_sleep_sec)

            t = flags.unroll_length
            b = flags.batch_size

            cur_step = int(ipc["shared_steps"].value)
            sps = (cur_step - start_step) / (timer() - start_time)
            bps = (cur_step - start_step) / (t * b) / (timer() - start_time)
            logging.info(f"Steps {cur_step:d} @ {sps:.1f} SPS / {bps:.1f} BPS.")

            bqszs = str(ipc["batch_queues"][0].qsize())
            na = int(flags.num_actors)
            eb = int(flags.n_actor_envs)
            arr = ipc["actor_desync_done_env_count"]
            n_desynced = sum(int(arr[i]) for i in range(na))
            m_envs = na * eb
            logging.info(
                "  queues: stats_q_learner_train=%d, batch_q=%s, full_queue=%d, "
                "free_queue=%d, stats_free_queue=%d, stats_full_queue=%d, envs_desynced=%d/%d",
                ipc["stats_queue_learner_train"].qsize(),
                bqszs,
                ipc["full_queue"].qsize(),
                ipc["free_queue"].qsize(),
                ipc["stats_free_queue"].qsize(),
                ipc["stats_full_queue"].qsize(),
                n_desynced,
                m_envs,
            )

            _ensure_processes_alive()
        
        
        logging.info("Learning finished after %d steps.", int(ipc["shared_steps"].value))
        _wait_for_final_checkpoint(int(flags.total_steps))

        raise_if_stop_requested(stop_event)

        finished_normally = True

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received in monobeast, shutting down")
    except StopRequested:
        if all_processes:
            _log_processes_snapshot("Stop requested snapshot")
        logging.info("Stop requested in monobeast, shutting down")
    except Exception as e:
        logging.info(traceback.format_exc())

    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name="monobeast",
            reason="train finally",
        )
        logging.info("Waiting for processes to finish")
        shutdown_all_processes(all_processes, timeout_sec=1200)

        logging.info("Exit?")
        if finished_normally:
            logging.info("Exit 0")
            os._exit(0)
        else:
            logging.info("Exit -1")
            os._exit(-1)


