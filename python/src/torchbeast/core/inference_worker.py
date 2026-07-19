import logging
import traceback
import os
import time
import queue
import setproctitle
import torch
from types import SimpleNamespace
from typing import Any

from ...gym.dict_io_contract import maybe_validate_dict_io_contract
from ...gym.obs_wrapper import (
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PLANET_ACTION_SLOTS,
    orbit_assert_available_action_mask_contract,
)
from ...models.models import create_impala_model
from .buffer_utils import (
    stack_buffers,
    tree_to_device,
)
from .common import (
    RollingImmediateWaitLogger,
    StopRequested,
    assert_spawn_fleet_actions_available,
    checkpoint_model_config_from_checkpoint_or_fallback,
    compile_impala_model_for_rl,
    configure_process_cpu_thread_limits,
    load_model_from_checkpoint,
    load_model_from_resume_sources,
    orbit_model_by_player_axis_from_seats,
    queue_get_or_stop,
    queue_get_or_timeout,
    queue_put_or_stop,
    set_stop_event_with_reason,
    stop_event_wait_or_raise,
    timed_queue_get_or_stop,
)

_CUDA0_INFERENCE_FULL_QUEUE_START_THRESHOLD = 32


def inference_batcher_process(
    flags: SimpleNamespace,
    infer_req_queue,
    infer_batch_queue,
    name: str,
    stop_event,
):
    try:
        setproctitle.setproctitle(name)
        configure_process_cpu_thread_limits()

        inference_batch_size = int(flags.inference_batch_size)
        assert inference_batch_size > 0
        timeout_s = 1.0

        while True:
            actor_ids = []
            req_indices = []
            model_by_seat_parts = []
            while len(actor_ids) < inference_batch_size:
                item, timed_out = queue_get_or_timeout(
                    infer_req_queue, stop_event, timeout_sec=timeout_s
                )
                if timed_out:
                    break
                assert isinstance(item, tuple), f"Expected tuple request, got {type(item)}"
                assert len(item) == 3, (
                    f"Expected request tuple of len 3, got len={len(item)}"
                )
                actor_id = item[0]
                req_idx = item[1]
                model_by_seat = item[2]
                assert isinstance(model_by_seat, torch.Tensor)
                assert model_by_seat.ndim == 2, tuple(model_by_seat.shape)
                actor_ids.append(actor_id)
                req_indices.append(req_idx)
                model_by_seat_parts.append(model_by_seat)

            if actor_ids:
                #logging.info(f"Putting batch of {len(actor_ids)} requests")
                queue_put_or_stop(
                    infer_batch_queue,
                    (actor_ids, req_indices, model_by_seat_parts),
                    stop_event,
                    timeout_sec=0.1,
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
            reason="inference_batcher_process finally",
        )
        os._exit(0)


def _compact_obs_active_players_cpu(
    obs: Any,
    row_idx: torch.Tensor,
    player_idx: torch.Tensor,
    *,
    full_rows: int,
    players: int,
    key_path: str = "",
) -> Any:
    if isinstance(obs, dict):
        return {
            key: _compact_obs_active_players_cpu(
                val,
                row_idx,
                player_idx,
                full_rows=full_rows,
                players=players,
                key_path=key if not key_path else f"{key_path}.{key}",
            )
            for key, val in obs.items()
        }
    assert isinstance(obs, torch.Tensor), (
        f"Expected tensor obs leaf at key_path='{key_path}', got {type(obs)}"
    )
    assert obs.ndim >= 2, (key_path, obs.ndim, tuple(obs.shape))
    assert int(obs.shape[0]) == int(full_rows), (key_path, tuple(obs.shape), full_rows)
    assert int(obs.shape[1]) == int(players), (key_path, tuple(obs.shape), players)
    compact = obs[row_idx, player_idx].unsqueeze(1).contiguous()
    assert int(compact.shape[0]) == int(row_idx.shape[0]), (
        key_path,
        tuple(compact.shape),
        tuple(row_idx.shape),
    )
    assert int(compact.shape[1]) == 1, (key_path, tuple(compact.shape))
    return compact


def _compact_active_player_batch_cpu(
    batch: dict,
    *,
    players: int,
) -> tuple[dict, torch.Tensor]:
    assert "obs_LEARN_INFER" in batch
    obs = batch["obs_LEARN_INFER"]
    assert isinstance(obs, dict)
    assert "player_mask" in obs
    player_mask = obs["player_mask"]
    assert isinstance(player_mask, torch.Tensor)
    assert player_mask.ndim == 2, tuple(player_mask.shape)
    assert int(player_mask.shape[1]) == int(players), (tuple(player_mask.shape), players)
    full_rows = int(player_mask.shape[0])
    assert full_rows > 0
    active_positions = (player_mask > 0.5).nonzero(as_tuple=False)
    assert active_positions.ndim == 2 and int(active_positions.shape[1]) == 2, (
        tuple(active_positions.shape),
    )
    assert int(active_positions.shape[0]) > 0, "inference batch must contain active players"
    row_idx = active_positions[:, 0]
    player_idx = active_positions[:, 1]
    compact_obs = _compact_obs_active_players_cpu(
        obs,
        row_idx,
        player_idx,
        full_rows=full_rows,
        players=players,
    )
    out = {"obs_LEARN_INFER": compact_obs}
    if "frozen_model_by_player_axis_LEARN" in batch:
        frozen_axis = batch["frozen_model_by_player_axis_LEARN"]
        compact_frozen_axis = _compact_obs_active_players_cpu(
            frozen_axis,
            row_idx,
            player_idx,
            full_rows=full_rows,
            players=players,
            key_path="frozen_model_by_player_axis_LEARN",
        )
        out["frozen_model_by_player_axis_LEARN"] = compact_frozen_axis
    return out, active_positions


def _flags_with_model_config(flags: SimpleNamespace, model_config: dict) -> SimpleNamespace:
    out = SimpleNamespace(**vars(flags))
    out.model = model_config
    return out


def _build_frozen_opponent_models_for_worker(
    flags: SimpleNamespace,
    device_str: str,
) -> list[torch.nn.Module]:
    frozen_cfg = flags.frozen_opponent
    if float(frozen_cfg.probability) == 0.0:
        return []
    assert len(frozen_cfg.checkpoints) > 0
    models: list[torch.nn.Module] = []
    for frozen_checkpoint in frozen_cfg.checkpoints:
        checkpoint = frozen_checkpoint.checkpoint
        model_config = checkpoint_model_config_from_checkpoint_or_fallback(
            checkpoint,
            flags.checkpoint_model_config_fallback_checkpoint,
            flags.model,
        )
        model = create_impala_model(_flags_with_model_config(flags, model_config))
        load_model_from_checkpoint(model, checkpoint)
        model = model.to(device_str)
        model.eval()
        model = compile_impala_model_for_rl(model, dynamic=True)
        model.eval()
        models.append(model)
    return models


def _fill_infer_result_defaults_cpu(dst: dict) -> None:
    assert "behavior_log_prob_sum_LEARN" in dst
    assert "actions_LEARN" in dst
    behavior = dst["behavior_log_prob_sum_LEARN"]
    actions = dst["actions_LEARN"]
    assert isinstance(behavior, dict) and "spawn_fleet" in behavior
    assert isinstance(actions, dict) and "spawn_fleet" in actions
    baseline = dst["baseline_LEARN"]
    assert isinstance(baseline, dict)
    assert set(baseline.keys()) == {"baseline", "production_delta"}, sorted(baseline.keys())
    baseline_value = baseline["baseline"]
    production_delta_value = baseline["production_delta"]
    assert isinstance(baseline_value, torch.Tensor)
    assert isinstance(production_delta_value, torch.Tensor)
    baseline_value.zero_()
    production_delta_value.zero_()
    behavior_spawn = behavior["spawn_fleet"]
    actions_spawn = actions["spawn_fleet"]
    assert isinstance(behavior_spawn, torch.Tensor)
    assert isinstance(actions_spawn, torch.Tensor)
    assert actions_spawn.ndim == 4, tuple(actions_spawn.shape)
    assert int(actions_spawn.shape[2]) == int(ORBIT_PLANET_ACTION_SLOTS), (
        tuple(actions_spawn.shape),
        ORBIT_PLANET_ACTION_SLOTS,
    )
    assert int(actions_spawn.shape[3]) == 1, tuple(actions_spawn.shape)
    assert tuple(behavior_spawn.shape) == tuple(actions_spawn.shape[:2]), (
        tuple(behavior_spawn.shape),
        tuple(actions_spawn.shape),
    )
    behavior_spawn.zero_()
    rows = torch.arange(
        ORBIT_PLANET_ACTION_SLOTS,
        device=actions_spawn.device,
        dtype=actions_spawn.dtype,
    ).view(1, 1, ORBIT_PLANET_ACTION_SLOTS, 1)
    actions_spawn.copy_(rows * int(ORBIT_MOVE_CLASSES_PER_TARGET), non_blocking=False)


def _copy_compact_result_slot_to_dst_cpu(
    dst: Any,
    src: Any,
    *,
    compact_idx: int,
    dst_env_idx: int,
    dst_player_idx: int,
    key_path: str = "",
) -> None:
    if isinstance(dst, dict):
        assert isinstance(src, dict), (
            f"Expected dict source for dict destination at key_path='{key_path}', got {type(src)}"
        )
        for key in dst:
            assert key in src, f"Missing source key '{key}' at key_path='{key_path}'"
            _copy_compact_result_slot_to_dst_cpu(
                dst[key],
                src[key],
                compact_idx=compact_idx,
                dst_env_idx=dst_env_idx,
                dst_player_idx=dst_player_idx,
                key_path=key if not key_path else f"{key_path}.{key}",
            )
        return
    assert isinstance(dst, torch.Tensor), (
        f"Expected tensor destination at key_path='{key_path}', got {type(dst)}"
    )
    assert isinstance(src, torch.Tensor), (
        f"Expected tensor source at key_path='{key_path}', got {type(src)}"
    )
    assert src.ndim == dst.ndim, (key_path, tuple(src.shape), tuple(dst.shape))
    assert int(src.shape[1]) == 1, (key_path, tuple(src.shape))
    src_slot = src[int(compact_idx), 0].to("cpu", non_blocking=False)
    dst_slot = dst[int(dst_env_idx), int(dst_player_idx)]
    assert tuple(dst_slot.shape) == tuple(src_slot.shape), (
        "compact inference scatter shape mismatch "
        f"key_path='{key_path}' dst={tuple(dst_slot.shape)} src={tuple(src_slot.shape)}"
    )
    dst_slot.copy_(src_slot, non_blocking=False)


def _index_batch_axis(tree: Any, row_idx: torch.Tensor) -> Any:
    if isinstance(tree, dict):
        return {key: _index_batch_axis(val, row_idx) for key, val in tree.items()}
    assert isinstance(tree, torch.Tensor), type(tree)
    assert row_idx.dtype == torch.long, row_idx.dtype
    assert tree.ndim >= 1, tuple(tree.shape)
    return tree.index_select(0, row_idx)


def _scatter_compact_result_slots_to_infer_buffers_cpu(
    infer_result_buffers,
    req_indices: list[int],
    compact_res: dict,
    active_positions: torch.Tensor,
    *,
    e_roll: int,
) -> None:
    assert isinstance(compact_res, dict)
    assert active_positions.ndim == 2 and int(active_positions.shape[1]) == 2, (
        tuple(active_positions.shape),
    )
    for compact_idx in range(int(active_positions.shape[0])):
        global_row = int(active_positions[compact_idx, 0].item())
        dst_player_idx = int(active_positions[compact_idx, 1].item())
        req_pos = global_row // int(e_roll)
        dst_env_idx = global_row - req_pos * int(e_roll)
        assert 0 <= req_pos < len(req_indices), (global_row, e_roll, len(req_indices))
        _copy_compact_result_slot_to_dst_cpu(
            infer_result_buffers[req_indices[req_pos]],
            compact_res,
            compact_idx=compact_idx,
            dst_env_idx=dst_env_idx,
            dst_player_idx=dst_player_idx,
        )



def inference_process(
    worker_id: int,
    flags: SimpleNamespace,
    infer_batch_queue,
    infer_res_queues,
    infer_request_buffers,
    infer_result_buffers,
    full_queue,
    device: int,
    actor_weight_buffers,
    actor_weight_update_queue,
    actor_weight_ack_queue,
    name: str,
    stop_event,
):
    try:
        setproctitle.setproctitle(name)
        configure_process_cpu_thread_limits()

        torch.cuda.set_device(device)
        device_str = f"cuda:{device}"

        actor_model = create_impala_model(flags)
        load_model_from_resume_sources(
            actor_model,
            resume_checkpoint=flags.resume_checkpoint,
            load_as_much_as_possible=flags.load_as_much_as_possible,
        )
        actor_model = actor_model.to(device_str)
        actor_model = compile_impala_model_for_rl(actor_model, dynamic=True)
        actor_model.eval()
        frozen_opponent_models = _build_frozen_opponent_models_for_worker(
            flags,
            device_str,
        )
        for frozen_model in frozen_opponent_models:
            frozen_model.eval()

        get_batch_wait_logger = RollingImmediateWaitLogger(
            metric_name="inference_worker.get_batch",
            log_interval_sec=60.0,
        )

        def _load_actor_weights_from_shared_buffer() -> None:
            assert isinstance(actor_weight_buffers, dict), type(actor_weight_buffers)
            actor_model.load_state_dict(actor_weight_buffers, strict=True)
            actor_model.eval()

        def _ack_actor_weights_update() -> None:
            actor_weight_ack_queue.put_nowait(int(worker_id))

        def _load_actor_weights_after_update_signal() -> None:
            try:
                signal = actor_weight_update_queue.get_nowait()
            except queue.Empty:
                return
            assert int(signal) == 1, signal
            _load_actor_weights_from_shared_buffer()
            _ack_actor_weights_update()

        initial_signal = queue_get_or_stop(
            actor_weight_update_queue,
            stop_event,
            timeout_sec=0.1,
        )
        assert int(initial_signal) == 1, initial_signal
        _load_actor_weights_from_shared_buffer()
        _ack_actor_weights_update()

        while True:
            if int(device) == 0:
                while (
                    int(full_queue.qsize())
                    >= _CUDA0_INFERENCE_FULL_QUEUE_START_THRESHOLD
                ):
                    stop_event_wait_or_raise(stop_event, timeout_sec=0.01)
            request_item = timed_queue_get_or_stop(
                infer_batch_queue,
                stop_event,
                timeout_sec=0.1,
                wait_logger=get_batch_wait_logger,
                stream_key=f"train:worker={int(worker_id)}",
                extra_context=f"device={int(device)}",
            )
            assert isinstance(request_item, tuple), (
                f"Expected infer_batch_queue tuple, got {type(request_item)}"
            )
            assert len(request_item) == 3, (
                f"Expected infer_batch_queue tuple of len 3, got len={len(request_item)}"
            )
            actor_ids, req_indices, model_by_seat_parts = request_item
            _load_actor_weights_after_update_signal()
            parts = [infer_request_buffers[req_idx] for req_idx in req_indices]

            # Stacked infer requests: dict tree from ``get_buffers_with_tag(..., tag="INFER")`` per actor
            # (e.g. ``obs_LEARN_INFER``).
            batch = stack_buffers(parts, dim=0)
            assert isinstance(batch, dict) and "obs_LEARN_INFER" in batch
            fe = batch["obs_LEARN_INFER"]["orbit_planet_features"]
            assert isinstance(fe, torch.Tensor) and fe.ndim == 4
            n_infer = len(req_indices)
            assert int(fe.shape[0]) % n_infer == 0, (tuple(fe.shape), n_infer)
            e_roll = int(fe.shape[0]) // n_infer
            model_by_seat_full = torch.cat(model_by_seat_parts, dim=0)
            assert isinstance(model_by_seat_full, torch.Tensor)
            assert model_by_seat_full.dtype == torch.int64, model_by_seat_full.dtype
            assert tuple(model_by_seat_full.shape) == (
                int(fe.shape[0]),
                int(flags.agents_max_cnt),
            ), (tuple(model_by_seat_full.shape), tuple(fe.shape), int(flags.agents_max_cnt))
            assert int(model_by_seat_full.min().item()) >= 0
            max_model_id = int(model_by_seat_full.max().item())
            assert max_model_id <= len(frozen_opponent_models), (
                max_model_id,
                len(frozen_opponent_models),
            )
            enemy_mask = batch["obs_LEARN_INFER"]["orbit_enemy_mask"]
            assert isinstance(enemy_mask, torch.Tensor)
            assert tuple(enemy_mask.shape) == (
                int(fe.shape[0]),
                int(flags.agents_max_cnt),
                3,
            ), (tuple(enemy_mask.shape), tuple(fe.shape), int(flags.agents_max_cnt))
            has_4p_enemy_axis = (enemy_mask[:, :, 1:] > 0.5).any(dim=(1, 2))
            game_num_players = torch.where(
                has_4p_enemy_axis,
                torch.full_like(has_4p_enemy_axis, 4, dtype=torch.int64),
                torch.full_like(has_4p_enemy_axis, 2, dtype=torch.int64),
            )
            assert game_num_players.shape == (int(fe.shape[0]),), game_num_players.shape
            batch["frozen_model_by_player_axis_LEARN"] = orbit_model_by_player_axis_from_seats(
                model_by_seat_full,
                game_num_players,
            )

            compact_batch, active_positions = _compact_active_player_batch_cpu(
                batch,
                players=int(flags.agents_max_cnt),
            )
            active_model_ids = model_by_seat_full[
                active_positions[:, 0],
                active_positions[:, 1],
            ].to(dtype=torch.long)

            use_bf16 = bool(flags.inference_use_bf16)
            # Policy inference uses the same sampling regime as training unless disabled.
            is_sample_for_inference = bool(flags.enable_sampling)

            def _run_model_once(model, infer_batch, *, include_value_head: bool):
                orbit_assert_available_action_mask_contract(
                    infer_batch["obs_LEARN_INFER"]["available_action_mask"],
                    label="inference_worker",
                )
                model.set_is_sample(is_sample_for_inference)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        out = model(
                            infer_batch,
                            output_full_policy_log_probs=False,
                            include_value_head=bool(include_value_head),
                        )
                assert isinstance(out, dict), "policy must return a dict"
                assert "actions_LEARN" in out and "behavior_log_prob_sum_LEARN" in out, (
                    "policy output must contain 'actions_LEARN' and 'behavior_log_prob_sum_LEARN'"
                )
                assert_spawn_fleet_actions_available(out, infer_batch)
                if bool(include_value_head):
                    assert "baseline_LEARN" in out, "policy output must contain 'baseline_LEARN'"
                else:
                    ref = out["behavior_log_prob_sum_LEARN"]["spawn_fleet"]
                    assert isinstance(ref, torch.Tensor)
                    assert ref.ndim == 2 and int(ref.shape[1]) == 1, tuple(ref.shape)
                    out["baseline_LEARN"] = {
                        "baseline": torch.zeros(
                            tuple(ref.shape),
                            dtype=torch.float32,
                            device=ref.device,
                        ),
                        "production_delta": torch.zeros(
                            tuple(ref.shape),
                            dtype=torch.float32,
                            device=ref.device,
                        ),
                    }
                return out

            def _model_for_id(model_id: int):
                if int(model_id) == 0:
                    return actor_model
                return frozen_opponent_models[int(model_id) - 1]

            for req_idx in req_indices:
                _fill_infer_result_defaults_cpu(infer_result_buffers[req_idx])

            for model_id_tensor in torch.unique(active_model_ids, sorted=True):
                model_id = int(model_id_tensor.item())
                group_cpu = (active_model_ids == model_id).nonzero(as_tuple=False).reshape(-1)
                assert int(group_cpu.shape[0]) > 0
                group_batch_cpu = _index_batch_axis(compact_batch, group_cpu)
                group_batch = tree_to_device(group_batch_cpu, device_str)
                res = _run_model_once(
                    _model_for_id(model_id),
                    group_batch,
                    include_value_head=model_id == 0,
                )
                ref_ls = res["behavior_log_prob_sum_LEARN"]["spawn_fleet"]
                assert tuple(ref_ls.shape) == (int(group_cpu.shape[0]), 1), (
                    tuple(ref_ls.shape),
                    tuple(group_cpu.shape),
                )
                _scatter_compact_result_slots_to_infer_buffers_cpu(
                    infer_result_buffers,
                    req_indices,
                    res,
                    active_positions.index_select(0, group_cpu),
                    e_roll=e_roll,
                )
            torch.cuda.synchronize(device)

            for i in range(len(req_indices)):
                idx = req_indices[i]
                maybe_validate_dict_io_contract(
                    flags, infer_result_buffers[idx], "infer_result_buffer_template"
                )
                queue_put_or_stop(
                    infer_res_queues[actor_ids[i]], idx, stop_event, timeout_sec=0.1
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
            reason="inference_process finally",
        )
        os._exit(0)

