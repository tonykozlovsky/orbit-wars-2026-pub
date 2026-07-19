from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any
import logging
import sys
import time

import torch

from ...gym.create_env import create_env
from ...gym.dict_io_contract import maybe_validate_dict_io_contract
from ...gym.obs_wrapper import ORBIT_PLAYER_AXIS_SLOTS
from .buffer_utils import (
    buffers_apply,
    get_buffers_with_tag,
    stack_buffers,
    stack_learn_timestep_template_along_time,
    tree_to_device,
)
from .common import assert_spawn_fleet_actions_available, load_model_from_resume_sources
from ...models.models import create_impala_model


def _alloc_shared_cpu_like_meta(t: torch.Tensor) -> torch.Tensor:
    assert isinstance(t, torch.Tensor)
    assert t.device.type == "meta"
    return torch.zeros(t.shape, dtype=t.dtype, device="cpu").share_memory_()


def _alloc_cuda_like_meta(t: torch.Tensor, *, device: int) -> torch.Tensor:
    assert isinstance(t, torch.Tensor)
    assert t.device.type == "meta"
    return torch.zeros(t.shape, dtype=t.dtype, device=f"cuda:{device}")


def _allocate_tree_from_meta(
    tree: Any,
    *,
    path: str,
    alloc_tensor,
) -> Any:
    if isinstance(tree, dict):
        return {
            key: _allocate_tree_from_meta(
                value,
                path=_path_join(path, str(key)),
                alloc_tensor=alloc_tensor,
            )
            for key, value in tree.items()
        }

    if isinstance(tree, torch.Tensor):
        return alloc_tensor(tree)

    return tree


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _path_join(prefix: str, suffix: str) -> str:
    if prefix == "":
        return suffix
    return f"{prefix}.{suffix}"


def _learner_gpu_single_lane_template(
    single_buffer: dict[str, Any],
    *,
    envs_per_actor: int,
    players: int,
) -> dict[str, Any]:
    def select_first_player_axis(t: torch.Tensor) -> torch.Tensor:
        assert isinstance(t, torch.Tensor)
        assert t.ndim >= 3, tuple(t.shape)
        assert int(t.shape[1]) == int(envs_per_actor), (
            tuple(t.shape),
            envs_per_actor,
        )
        assert int(t.shape[2]) == int(players), (
            tuple(t.shape),
            players,
        )
        return t[:, 0:1, 0:1, ...].contiguous()

    return buffers_apply(single_buffer, select_first_player_axis)


def _accumulate_tensor_bytes_by_key(
    tree: Any,
    *,
    path: str,
    multiplier: int,
    logical_bytes_by_key: dict[str, int],
    physical_bytes_by_key: dict[str, int],
    tensor_meta_by_key: dict[str, dict[str, Any]],
) -> None:
    if isinstance(tree, torch.Tensor):
        key = path
        nb = multiplier * _tensor_nbytes(tree)
        logical_bytes_by_key[key] = logical_bytes_by_key.get(key, 0) + nb
        physical_bytes_by_key[key] = physical_bytes_by_key.get(key, 0) + nb
        dtype_name = str(tree.dtype).replace("torch.", "")
        shape = tuple(int(v) for v in tree.shape)
        if key not in tensor_meta_by_key:
            tensor_meta_by_key[key] = {
                "dtypes": {dtype_name},
                "shape": shape,
                "mixed_shapes": False,
            }
        else:
            meta = tensor_meta_by_key[key]
            meta["dtypes"].add(dtype_name)
            if tuple(meta["shape"]) != shape:
                meta["mixed_shapes"] = True
        return
    if isinstance(tree, dict):
        for key, value in tree.items():
            _accumulate_tensor_bytes_by_key(
                value,
                path=_path_join(path, str(key)),
                multiplier=multiplier,
                logical_bytes_by_key=logical_bytes_by_key,
                physical_bytes_by_key=physical_bytes_by_key,
                tensor_meta_by_key=tensor_meta_by_key,
            )
        return


def _human_bytes(num_bytes: int) -> str:
    assert num_bytes >= 0
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kib = num_bytes / 1024.0
    if kib < 1024.0:
        return f"{kib:.2f} KiB"
    mib = kib / 1024.0
    if mib < 1024.0:
        return f"{mib:.2f} MiB"
    gib = mib / 1024.0
    return f"{gib:.2f} GiB"


def _format_tensor_meta(meta: dict[str, Any]) -> tuple[str, str]:
    dtypes = sorted(list(meta["dtypes"]))
    if len(dtypes) == 1:
        dtype_str = dtypes[0]
    else:
        dtype_str = "mixed[" + ",".join(dtypes) + "]"
    if bool(meta["mixed_shapes"]):
        shape_str = "mixed"
    else:
        shape_str = str(tuple(meta["shape"]))
    return dtype_str, shape_str


def _log_top_tensor_keys_by_bytes(
    *,
    label: str,
    entries: list[tuple[str, Any, int]],
    top_k: int = 50,
) -> None:
    logical_bytes_by_key: dict[str, int] = {}
    physical_bytes_by_key: dict[str, int] = {}
    tensor_meta_by_key: dict[str, dict[str, Any]] = {}
    for path, tree, multiplier in entries:
        if multiplier <= 0:
            continue
        _accumulate_tensor_bytes_by_key(
            tree,
            path=path,
            multiplier=int(multiplier),
            logical_bytes_by_key=logical_bytes_by_key,
            physical_bytes_by_key=physical_bytes_by_key,
            tensor_meta_by_key=tensor_meta_by_key,
        )
    if len(physical_bytes_by_key) == 0:
        logging.info("BUFFER_BYTES_TOP50[%s]: no tensor keys", label)
        return
    sorted_items = sorted(physical_bytes_by_key.items(), key=lambda kv: kv[1], reverse=True)
    total_physical_bytes = sum(physical_bytes_by_key.values())
    total_logical_bytes = sum(logical_bytes_by_key.values())
    logging.info(
        "BUFFER_BYTES_TOP50[%s]: unique_keys=%d total_physical=%d (%s) total_logical=%d (%s)",
        label,
        len(sorted_items),
        total_physical_bytes,
        _human_bytes(total_physical_bytes),
        total_logical_bytes,
        _human_bytes(total_logical_bytes),
    )
    for rank, (key, physical_bytes) in enumerate(sorted_items[: int(top_k)], start=1):
        assert key in tensor_meta_by_key, f"Missing tensor meta for key '{key}'"
        logical_bytes = logical_bytes_by_key.get(key, 0)
        dtype_str, shape_str = _format_tensor_meta(tensor_meta_by_key[key])
        logging.info(
            (
                "BUFFER_BYTES_TOP50[%s] #%02d %s "
                "physical_bytes=%d (%s) logical_bytes=%d (%s) dtype=%s shape=%s"
            ),
            label,
            rank,
            key,
            physical_bytes,
            _human_bytes(physical_bytes),
            logical_bytes,
            _human_bytes(logical_bytes),
            dtype_str,
            shape_str,
        )


def allocate_cpu_buffers_from_spec(
    *,
    flags: SimpleNamespace,
    spec: BufferSpec,
) -> tuple[list, list, list, list]:
    """
    Allocate all CPU shared-memory buffers required to start CPU processes.

    Returns:
      learn_buffers: list[buffer_tree] (LEARN-tag buffers)
      stats_buffers: list[buffer_tree] (STAT-tag buffers)
      infer_request_buffers_train: list[buffer_tree]
      infer_result_buffers_train: list[buffer_tree]
    """
    stats_buffers = [
        _allocate_tree_from_meta(
            spec.single_cpu_buffers,
            path="",
            alloc_tensor=_alloc_shared_cpu_like_meta,
        )
        for _ in range(flags.num_stats_buffers)
    ]

    learn_buffers: list = []
    if int(flags.num_actors) > 0:
        for _ in range(flags.num_buffers):
            buffer_tree = _allocate_tree_from_meta(
                spec.single_gpu_buffers,
                path="",
                alloc_tensor=_alloc_shared_cpu_like_meta,
            )
            learn_buffers.append(buffer_tree)

    infer_request_buffers_train = [
        _allocate_tree_from_meta(
            spec.infer_request_template,
            path="",
            alloc_tensor=_alloc_shared_cpu_like_meta,
        )
        for _ in range(flags.num_inference_buffers_train)
    ]
    infer_result_buffers_train = [
        _allocate_tree_from_meta(
            spec.infer_result_template,
            path="",
            alloc_tensor=_alloc_shared_cpu_like_meta,
        )
        for _ in range(flags.num_inference_buffers_train)
    ]
    cpu_entries: list[tuple[str, Any, int]] = []
    if len(learn_buffers) > 0:
        cpu_entries.append(("learn", learn_buffers[0], len(learn_buffers)))
    if len(stats_buffers) > 0:
        cpu_entries.append(("stats", stats_buffers[0], len(stats_buffers)))
    if len(infer_request_buffers_train) > 0:
        cpu_entries.append(
            (
                "infer_request.train",
                infer_request_buffers_train[0],
                len(infer_request_buffers_train),
            )
        )
    if len(infer_result_buffers_train) > 0:
        cpu_entries.append(
            (
                "infer_result.train",
                infer_result_buffers_train[0],
                len(infer_result_buffers_train),
            )
        )
    _log_top_tensor_keys_by_bytes(
        label="cpu_shared",
        entries=cpu_entries,
        top_k=50,
    )

    return (
        learn_buffers,
        stats_buffers,
        infer_request_buffers_train,
        infer_result_buffers_train,
    )


def allocate_gpu_buffers_from_spec(
    *,
    flags: SimpleNamespace,
    spec: BufferSpec,
    learner_cuda_device: int,
) -> list[list]:
    """
    Allocate GPU resident learner batch buffers.

    Returns:
      learner_gpu_buffers: length-1 outer list (single GPU); inner list is prepare_batches GPU trees.
    """
    learner_gpu_buffers: list[list] = []
    per_device: list = []
    if int(flags.num_actors) > 0:
        dev = int(learner_cuda_device)
        per_device = [
            _allocate_tree_from_meta(
                spec.gpu_buffers,
                path="",
                alloc_tensor=lambda t, dev=dev: _alloc_cuda_like_meta(t, device=dev),
            )
            for _ in range(flags.prepare_batches)
        ]
    learner_gpu_buffers.append(per_device)

    gpu_entries: list[tuple[str, Any, int]] = []
    for device_idx, per_device_buffers in enumerate(learner_gpu_buffers):
        if len(per_device_buffers) == 0:
            continue
        gpu_entries.append(
            (
                f"device{device_idx}",
                per_device_buffers[0],
                len(per_device_buffers),
            )
        )
    _log_top_tensor_keys_by_bytes(
        label="gpu_learner",
        entries=gpu_entries,
        top_k=50,
    )

    return learner_gpu_buffers


def _to_meta_tensors(structuring_tree):
    if isinstance(structuring_tree, dict):
        return {k: _to_meta_tensors(v) for k, v in structuring_tree.items()}
    if isinstance(structuring_tree, torch.Tensor):
        return torch.empty(structuring_tree.shape, dtype=structuring_tree.dtype, device="meta")
    return structuring_tree


def _expand_done_learn_stat_per_player(payload: dict[str, Any]) -> None:
    reward_learn_stat = payload["reward_LEARN_STAT"]
    done = payload["done_LEARN_STAT"]
    assert isinstance(reward_learn_stat, dict)
    assert "baseline" in reward_learn_stat, sorted(reward_learn_stat.keys())
    reward = reward_learn_stat["baseline"]
    assert isinstance(reward, torch.Tensor)
    assert isinstance(done, torch.Tensor) and done.dtype == torch.bool
    assert reward.ndim == 2
    assert done.shape == reward.shape[:1], (done.shape, reward.shape)
    payload["done_LEARN_STAT"] = done.unsqueeze(-1).expand_as(reward)


def _empty_critic_mc_stat(n_actor_envs: int) -> dict[str, torch.Tensor]:
    shape = (int(n_actor_envs), ORBIT_PLAYER_AXIS_SLOTS)
    return {
        "valid": torch.zeros(shape, dtype=torch.bool),
        "count": torch.zeros(shape, dtype=torch.float32),
        "sqerr_sum": torch.zeros(shape, dtype=torch.float32),
        "abserr_sum": torch.zeros(shape, dtype=torch.float32),
        "err_sum": torch.zeros(shape, dtype=torch.float32),
        "return_sum": torch.zeros(shape, dtype=torch.float32),
        "return_sq_sum": torch.zeros(shape, dtype=torch.float32),
        "value_sum": torch.zeros(shape, dtype=torch.float32),
        "value_sq_sum": torch.zeros(shape, dtype=torch.float32),
        "value_return_sum": torch.zeros(shape, dtype=torch.float32),
        "first_error": torch.zeros(shape, dtype=torch.float32),
    }


@dataclasses.dataclass(frozen=True)
class BufferSpec:
    """
    Light-weight templates (with meta tensors) that preserve nested dict/tensor structure
    so the main process can allocate real CPU/GPU buffers later without touching GPU during spec computation.
    """

    sample_cpu: Any
    single_buffer: Any
    stacked_buffers: Any
    single_cpu_buffers: Any
    single_gpu_buffers: Any
    gpu_buffers: Any
    infer_request_template: Any
    infer_result_template: Any


@torch.no_grad()
def compute_buffer_spec(
    *,
    flags: SimpleNamespace,
    devices: list[int],
) -> BufferSpec:
    assert len(devices) == 1, f"single-GPU training expects one device id, got {devices!r}"
    flags_for_buffers = flags

    _envs_val = bool(flags_for_buffers.enable_envs_validation)
    env_rl = create_env(
        flags_for_buffers,
        device="cpu",
        visualize=False,
        cpp_env_obs_full=not _envs_val,
        cpp_env_obs_validate=_envs_val,
    )
    env_output_rl = env_rl.reset()
    infer_req = get_buffers_with_tag(env_output_rl, device="cpu", tag="INFER")
    assert infer_req is not None

    ref = infer_req["obs_LEARN_INFER"]["orbit_planet_features"]
    assert isinstance(ref, torch.Tensor) and ref.ndim == 4, tuple(ref.shape)
    e_infer = int(ref.shape[0])
    assert e_infer == int(flags_for_buffers.n_actor_envs), (e_infer, flags_for_buffers.n_actor_envs)
    assert int(ORBIT_PLAYER_AXIS_SLOTS) == int(flags_for_buffers.agents_max_cnt), (
        ORBIT_PLAYER_AXIS_SLOTS,
        flags_for_buffers.agents_max_cnt,
    )
    infer_req["benchmark_model_by_seat"] = torch.zeros(
        e_infer,
        ORBIT_PLAYER_AXIS_SLOTS,
        dtype=torch.int64,
    )
    infer_req["frozen_model_by_player_axis_LEARN"] = torch.zeros(
        e_infer,
        ORBIT_PLAYER_AXIS_SLOTS,
        ORBIT_PLAYER_AXIS_SLOTS,
        dtype=torch.int64,
    )

    actor_models: list[torch.nn.Module] = []
    for device in devices:
        device_str = f"cuda:{device}"
        actor_model = create_impala_model(flags)
        load_model_from_resume_sources(
            actor_model,
            resume_checkpoint=flags.resume_checkpoint,
            load_as_much_as_possible=flags.load_as_much_as_possible,
        )
        actor_model = actor_model.to(device_str)
        actor_model.eval()
        actor_models.append(actor_model)

    is_sample = bool(flags.enable_sampling)
    agent_output: dict | None = None
    for i, actor_model in enumerate(actor_models):
        actor_model.set_is_sample(is_sample)
        infer_req_device = tree_to_device(infer_req, f"cuda:{devices[i]}")
        out = actor_model(
            infer_req_device,
            output_full_policy_log_probs=False,
            include_value_head=True,
        )
        assert isinstance(out, dict), "policy must return a dict"
        assert_spawn_fleet_actions_available(out, infer_req_device)
        agent_output = out

    assert agent_output is not None
    assert "actions_LEARN" in agent_output and "behavior_log_prob_sum_LEARN" in agent_output
    assert "baseline_LEARN" in agent_output

    assert isinstance(env_output_rl["obs_LEARN_INFER"], dict)
    assert "available_action_mask" in env_output_rl["obs_LEARN_INFER"]
    assert "player_mask" in env_output_rl["obs_LEARN_INFER"]

    agent_output_for_learn = dict(agent_output)
    del agent_output_for_learn["baseline_LEARN"]
    sample_main_payload_for_learn = get_buffers_with_tag(
        {**env_output_rl, **agent_output_for_learn},
        device="cpu",
        tag="LEARN",
    )
    assert sample_main_payload_for_learn is not None
    _expand_done_learn_stat_per_player(sample_main_payload_for_learn)
    ref_player_mask = sample_main_payload_for_learn["obs_LEARN_INFER"]["player_mask"]
    assert isinstance(ref_player_mask, torch.Tensor)
    sample_main_payload_for_learn["game_num_players_LEARN"] = torch.zeros(
        tuple(ref_player_mask.shape),
        dtype=torch.int64,
    )
    sample_main_payload_for_learn["frozen_model_by_player_axis_LEARN"] = torch.zeros(
        tuple(ref_player_mask.shape) + (ORBIT_PLAYER_AXIS_SLOTS,),
        dtype=torch.int64,
    )

    sample_main_payload_for_stats = get_buffers_with_tag(
        env_output_rl, device="cpu", tag="STAT"
    )
    assert sample_main_payload_for_stats is not None
    assert "action_taken_index_LEARN_STAT" in sample_main_payload_for_stats
    sample_main_payload_for_stats["frozen_model_by_seat_STAT"] = torch.zeros(
        int(flags_for_buffers.n_actor_envs),
        ORBIT_PLAYER_AXIS_SLOTS,
        dtype=torch.int64,
    )
    sample_main_payload_for_stats["critic_mc_STAT"] = _empty_critic_mc_stat(
        int(flags_for_buffers.n_actor_envs)
    )

    sample_cpu = tree_to_device(sample_main_payload_for_learn, "cpu")
    single_buffer = stack_learn_timestep_template_along_time(
        sample_cpu,
        time_steps=int(flags.unroll_length) + 1,
    )

    learner_single_buffer = _learner_gpu_single_lane_template(
        single_buffer,
        envs_per_actor=int(flags.n_actor_envs),
        players=int(flags.agents_max_cnt),
    )
    stacked_buffers = stack_buffers([learner_single_buffer] * int(flags.batch_size), dim=1)

    sample_main_stats_cpu = tree_to_device(sample_main_payload_for_stats, "cpu")
    single_buffer_for_stats = stack_learn_timestep_template_along_time(
        sample_main_stats_cpu,
        time_steps=int(flags.unroll_length) + 1,
    )
    single_cpu_buffers = get_buffers_with_tag(single_buffer_for_stats, device="cpu", tag="STAT")

    single_gpu_buffers = get_buffers_with_tag(single_buffer, device="cpu", tag="LEARN")
    gpu_buffers = get_buffers_with_tag(stacked_buffers, device="cpu", tag="LEARN")
    assert gpu_buffers is not None

    infer_request_template = infer_req
    infer_result_template = get_buffers_with_tag(
        agent_output, device="cpu", tag="LEARN"
    )
    assert infer_result_template is not None

    maybe_validate_dict_io_contract(
        flags_for_buffers, infer_request_template, "infer_request_buffer_template"
    )
    maybe_validate_dict_io_contract(
        flags_for_buffers, infer_result_template, "infer_result_buffer_template"
    )
    assert single_gpu_buffers is not None
    maybe_validate_dict_io_contract(
        flags_for_buffers, single_gpu_buffers, "actor_learn_buffer_template"
    )
    assert gpu_buffers is not None
    maybe_validate_dict_io_contract(
        flags_for_buffers, gpu_buffers, "learner_gpu_buffer_template"
    )

    return BufferSpec(
        sample_cpu=_to_meta_tensors(sample_cpu),
        single_buffer=_to_meta_tensors(single_buffer),
        stacked_buffers=_to_meta_tensors(stacked_buffers),
        single_cpu_buffers=_to_meta_tensors(single_cpu_buffers),
        single_gpu_buffers=_to_meta_tensors(single_gpu_buffers),
        gpu_buffers=_to_meta_tensors(gpu_buffers),
        infer_request_template=_to_meta_tensors(infer_request_template),
        infer_result_template=_to_meta_tensors(infer_result_template),
    )


def compute_buffer_spec_process(
    flags: SimpleNamespace, devices: list[int], conn
) -> None:
    """
    Compute BufferSpec in a spawned process.

    IMPORTANT: keep this process alive until the parent acknowledges receipt.
    Torch may transfer tensor storages via a per-process resource_sharer socket; if the
    child exits too early, the parent can fail unpickling with FileNotFoundError.
    """
    spec = compute_buffer_spec(flags=flags, devices=devices)
    conn.send(spec)
    _ = conn.recv()  # wait for parent ack
    conn.close()


