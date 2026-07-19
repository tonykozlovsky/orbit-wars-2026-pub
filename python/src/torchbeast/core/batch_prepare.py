import logging
import traceback
import setproctitle
import torch
import os
from ...gym.dict_io_contract import maybe_validate_dict_io_contract
from .buffer_utils import buffers_apply
from .common import (
    configure_process_cpu_thread_limits,
    queue_get_or_stop,
    queue_put_or_stop,
    raise_if_stop_requested,
    set_stop_event_with_reason,
    StopRequested,
)


def _pin_tensor_for_batch_prepare(x: torch.Tensor) -> torch.Tensor:
    return x.pin_memory()


def _assert_rollout_player_mask_contract(
    part: dict,
    *,
    rollout_time_steps: int,
    envs_per_actor: int,
    players: int,
) -> torch.Tensor:
    assert "obs_LEARN_INFER" in part
    obs = part["obs_LEARN_INFER"]
    assert isinstance(obs, dict)
    assert "player_mask" in obs
    player_mask = obs["player_mask"]
    assert isinstance(player_mask, torch.Tensor)
    assert tuple(int(x) for x in player_mask.shape) == (
        int(rollout_time_steps),
        int(envs_per_actor),
        int(players),
    ), player_mask.shape
    return player_mask


def _valid_player_rows_from_part(
    *,
    buffer_idx: int,
    part: dict,
    buffers,
    rollout_time_steps: int,
    envs_per_actor: int,
    players: int,
) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    assert buffers[buffer_idx] is part
    player_mask = _assert_rollout_player_mask_contract(
        part,
        rollout_time_steps=rollout_time_steps,
        envs_per_actor=envs_per_actor,
        players=players,
    )
    transition_player_mask = player_mask[:-1]
    assert int(transition_player_mask.shape[0]) == int(rollout_time_steps) - 1, (
        tuple(transition_player_mask.shape),
        rollout_time_steps,
    )
    valid_players = (transition_player_mask > 0.5).any(dim=0)
    assert tuple(int(x) for x in valid_players.shape) == (
        int(envs_per_actor),
        int(players),
    ), valid_players.shape
    for env_idx in range(int(envs_per_actor)):
        for player_idx in range(int(players)):
            if bool(valid_players[env_idx, player_idx].item()):
                rows.append((int(buffer_idx), int(env_idx), int(player_idx)))
    assert len(rows) > 0, (buffer_idx, tuple(player_mask.shape))
    return rows


def _copy_valid_player_lane_to_gpu(
    dst,
    src,
    *,
    src_env_idx: int,
    src_player_idx: int,
    dst_batch_idx: int,
    envs_per_actor: int,
    batch_size: int,
    players: int,
    key_path: str = "",
) -> None:
    if isinstance(src, dict):
        assert isinstance(dst, dict), (
            f"Expected dict destination for dict source at key_path='{key_path}', "
            f"got {type(dst)}"
        )
        assert set(src.keys()) == set(dst.keys()), (
            f"batch_prepare copy keys differ at key_path='{key_path}': "
            f"src={set(src.keys())} dst={set(dst.keys())}"
        )
        for key, val in src.items():
            child_key_path = key if not key_path else f"{key_path}.{key}"
            _copy_valid_player_lane_to_gpu(
                dst[key],
                val,
                src_env_idx=src_env_idx,
                src_player_idx=src_player_idx,
                dst_batch_idx=dst_batch_idx,
                envs_per_actor=envs_per_actor,
                batch_size=batch_size,
                players=players,
                key_path=child_key_path,
            )
        return
    assert isinstance(src, torch.Tensor), (
        f"Expected tensor source for key_path='{key_path}', got {type(src)}"
    )
    assert isinstance(dst, torch.Tensor), (
        f"Expected tensor destination for key_path='{key_path}', got {type(dst)}"
    )
    assert src.ndim >= 3 and dst.ndim >= 3, (
        key_path,
        tuple(src.shape),
        tuple(dst.shape),
    )
    assert int(src.shape[1]) == int(envs_per_actor), (
        key_path,
        tuple(src.shape),
        envs_per_actor,
    )
    assert int(src.shape[2]) == int(players), (
        key_path,
        tuple(src.shape),
        players,
    )
    assert int(dst.shape[1]) == int(batch_size), (
        key_path,
        tuple(dst.shape),
        batch_size,
    )
    assert int(dst.shape[2]) == 1, (key_path, tuple(dst.shape))
    src_slice = src[:, int(src_env_idx), int(src_player_idx), ...]
    dst_slice = dst[:, int(dst_batch_idx), 0, ...]
    assert tuple(dst_slice.shape) == tuple(src_slice.shape), (
        "player lane copy shape mismatch "
        f"key_path='{key_path}' dst={tuple(dst_slice.shape)} src={tuple(src_slice.shape)}"
    )
    dst_slice.copy_(src_slice, non_blocking=True)


def batch_prepare_process_func(
    flags,
    full_queue,
    full_queue_lock,
    batch_queue,
    device,
    buffers,
    learner_free_batch_queue,
    learner_gpu_buffers,
    # free_queues,
    free_queue,
    name: str,
    stop_event,
):
    try:
        setproctitle.setproctitle(name)
        configure_process_cpu_thread_limits()
        torch.cuda.set_device(device)
        torch.set_default_device(f'cuda:{device}')

        # Create a dedicated CUDA stream for this process to enable parallel copying
        # across multiple batch_prepare processes
        copy_stream = torch.cuda.Stream(device=device)

        logging.info(f'BATCH PREPARE device: {device} name: {name}')
        pending_rows: list[tuple[int, int, int]] = []
        pending_buffer_remaining_rows: dict[int, int] = {}
        while True:

            envs_per_actor = int(flags.n_actor_envs)
            assert envs_per_actor > 0
            batch_size = int(flags.batch_size)
            players = int(flags.agents_max_cnt)
            assert players == 4, players
            rollout_time_steps = int(flags.rollout_time_steps)
            assert rollout_time_steps > 0
            selected_rows: list[tuple[int, int, int]] = []
            selected_buffer_ids: set[int] = set()
            release_buffer_ids: set[int] = set()

            while len(selected_rows) < batch_size:
                raise_if_stop_requested(stop_event)
                if not pending_rows:
                    item = queue_get_or_stop(full_queue, stop_event, timeout_sec=0.1)
                    assert isinstance(item, tuple) and len(item) == 2
                    _timestamp, buffer_idx = item
                    buffer_idx = int(buffer_idx)
                    rows = _valid_player_rows_from_part(
                        buffer_idx=buffer_idx,
                        part=buffers[buffer_idx],
                        buffers=buffers,
                        rollout_time_steps=rollout_time_steps,
                        envs_per_actor=envs_per_actor,
                        players=players,
                    )
                    assert buffer_idx not in pending_buffer_remaining_rows, (
                        buffer_idx,
                        pending_buffer_remaining_rows,
                    )
                    pending_buffer_remaining_rows[buffer_idx] = len(rows)
                    pending_rows.extend(rows)

                row = pending_rows.pop(0)
                buffer_idx = int(row[0])
                assert buffer_idx in pending_buffer_remaining_rows, (
                    buffer_idx,
                    pending_buffer_remaining_rows,
                )
                pending_buffer_remaining_rows[buffer_idx] -= 1
                assert pending_buffer_remaining_rows[buffer_idx] >= 0, (
                    buffer_idx,
                    pending_buffer_remaining_rows[buffer_idx],
                )
                if pending_buffer_remaining_rows[buffer_idx] == 0:
                    del pending_buffer_remaining_rows[buffer_idx]
                    release_buffer_ids.add(buffer_idx)
                selected_buffer_ids.add(buffer_idx)
                selected_rows.append(row)

            assert len(selected_rows) == batch_size, len(selected_rows)

            pinned_by_buffer_idx = {}
            for buffer_idx in sorted(selected_buffer_ids):
                pinned_by_buffer_idx[buffer_idx] = buffers_apply(
                    buffers[buffer_idx], _pin_tensor_for_batch_prepare
                )

            learner_buffer_idx = queue_get_or_stop(
                learner_free_batch_queue, stop_event, timeout_sec=0.1
            )

            with torch.cuda.stream(copy_stream):
                dst = learner_gpu_buffers[learner_buffer_idx]
                for dst_batch_idx, (buffer_idx, env_idx, player_idx) in enumerate(selected_rows):
                    _copy_valid_player_lane_to_gpu(
                        dst,
                        pinned_by_buffer_idx[int(buffer_idx)],
                        src_env_idx=int(env_idx),
                        src_player_idx=int(player_idx),
                        dst_batch_idx=int(dst_batch_idx),
                        envs_per_actor=envs_per_actor,
                        batch_size=batch_size,
                        players=players,
                    )
            # Synchronize the copy stream to ensure all copies are complete
            # before the learner uses the data
            copy_stream.synchronize()

            maybe_validate_dict_io_contract(
                flags,
                learner_gpu_buffers[learner_buffer_idx],
                "learner_gpu_buffer_template",
            )

            for buffer_idx in sorted(release_buffer_ids):
                assert buffer_idx in selected_buffer_ids, (
                    buffer_idx,
                    selected_buffer_ids,
                )
                queue_put_or_stop(
                    free_queue, buffer_idx, stop_event, timeout_sec=0.1
                )

            queue_put_or_stop(
                batch_queue, learner_buffer_idx, stop_event, timeout_sec=0.1
            )
        
    except KeyboardInterrupt:
        pass
    except StopRequested:
        pass
    except Exception as e:
        logging.info(traceback.format_exc())
    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name=name,
            reason="batch_prepare_process_func finally",
        )
        os._exit(0)

