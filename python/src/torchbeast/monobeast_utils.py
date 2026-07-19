import concurrent.futures
import logging
import multiprocessing as pymp
import os
import time
from typing import Literal

import psutil
import torch
import torch.multiprocessing as mp
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from .core.common import ProcessWithOptions
from .core.create_buffers import compute_buffer_spec_process

ProcessKind = Literal["cpu", "gpu"]
StartMethod = Literal["fork", "spawn"]


def _mp_context_from_start_method(start_method: StartMethod):
    assert start_method in ("fork", "spawn"), start_method
    return mp.get_context(start_method)


def _mp_context_from_flags(flags):
    return _mp_context_from_start_method(flags.multiprocessing_start_method)


def _raise_if_uninitialized_module_tensors(
    module: torch.nn.Module, *, module_name: str
) -> None:
    uninitialized_parameters = [
        name
        for name, parameter in module.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]
    uninitialized_buffers = [
        name
        for name, buffer in module.named_buffers()
        if isinstance(buffer, UninitializedBuffer)
    ]
    if not uninitialized_parameters and not uninitialized_buffers:
        return

    logging.error(
        "%s has uninitialized tensors before share_memory: "
        "uninitialized_parameters=%s, uninitialized_buffers=%s",
        module_name,
        uninitialized_parameters,
        uninitialized_buffers,
    )
    raise RuntimeError(
        f"{module_name} has uninitialized tensors before share_memory: "
        f"uninitialized_parameters={uninitialized_parameters}, "
        f"uninitialized_buffers={uninitialized_buffers}"
    )


def _run_with_env(target, target_args: tuple, env_overrides: dict[str, str]) -> None:
    for k, v in env_overrides.items():
        os.environ[k] = v
    target(*target_args)


def _make_process(
    *,
    ctx,
    target,
    args: tuple,
    name: str,
    kind: ProcessKind,
    torch_compile: bool = False,
) -> mp.Process:
    env_overrides: dict[str, str] = {}
    if kind == "cpu":
        env_overrides["CUDA_VISIBLE_DEVICES"] = ""

    return ctx.Process(
        target=_run_with_env,
        args=(target, args, env_overrides),
        name=name,
    )


def _start_processes_in_threadpool(procs: list[ProcessWithOptions]) -> None:
    if len(procs) == 0:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(proc_with_opts.process.start) for proc_with_opts in procs]
        for future in futures:
            future.result()


def _compute_buffer_spec_via_spawn(flags, *, log_prefix: str):
    spec_ctx = pymp.get_context("spawn")
    parent_conn, child_conn = spec_ctx.Pipe(duplex=True)

    p = spec_ctx.Process(
        target=compute_buffer_spec_process,
        args=(
            flags,
            [int(flags.inference_cuda_device)],
            child_conn,
        ),
        name="BUFFER_SPEC",
    )
    p.start()
    spec = parent_conn.recv()
    parent_conn.send("ok")
    p.join()
    assert p.exitcode == 0
    logging.info("%s: buffer spec computed in dedicated process", log_prefix)
    return spec


def kill_all_child_processes():
    current_process = psutil.Process(os.getpid())
    for child in current_process.children(recursive=True):
        try:
            child.kill()
            logging.info(f'Force killed child process {child.pid}')
        except psutil.NoSuchProcess:
            continue


def shutdown_all_processes(all_processes: list[ProcessWithOptions], *, timeout_sec: float) -> None:
    logging.info("Shutting down processes, waiting for them to finish")
    deadline = time.time() + float(timeout_sec)
    for proc_with_opts in all_processes:
        proc = proc_with_opts.process
        if proc.pid is None:
            continue
        remaining = max(0.0, deadline - time.time())
        logging.info(f"Joining process {proc.pid} for {remaining} seconds")
        proc.join(timeout=remaining)
    still_running = []
    for proc_with_opts in all_processes:
        proc = proc_with_opts.process
        if proc.pid is None:
            continue
        if proc.is_alive():
            still_running.append(
                f"{proc.name} pid={proc.pid} exitcode={proc.exitcode}"
            )
    if still_running:
        logging.info(
            "Processes still running before kill: %s",
            ", ".join(still_running),
        )
    logging.info("Killing remaining child processes")
    kill_all_child_processes()
