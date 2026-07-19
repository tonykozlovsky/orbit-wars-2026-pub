import logging
import os
import setproctitle
import traceback
from types import SimpleNamespace

import torch

from ...gym.dict_io_contract import maybe_validate_dict_io_contract
from ...gym.obs_wrapper import orbit_assert_available_action_mask_contract
from .buffer_utils import stack_buffers, tree_to_device
from .common import (
    RollingImmediateWaitLogger,
    StopRequested,
    orbit_model_by_player_axis_from_seats,
    queue_put_or_stop,
    set_stop_event_with_reason,
    timed_queue_get_or_stop,
)
from .inference_worker import (
    _compact_active_player_batch_cpu,
    _fill_infer_result_defaults_cpu,
    _index_batch_axis,
    _scatter_compact_result_slots_to_infer_buffers_cpu,
)


def benchmark_inference_process(
    worker_id: int,
    flags: SimpleNamespace,
    infer_batch_queue,
    infer_res_queues,
    infer_request_buffers,
    infer_result_buffers,
    device: int,
    benchmark_model_a,
    benchmark_model_b,
    name: str,
    stop_event,
) -> None:
    try:
        _benchmark_inference_loop(
            worker_id,
            flags,
            infer_batch_queue,
            infer_res_queues,
            infer_request_buffers,
            infer_result_buffers,
            device,
            benchmark_model_a,
            benchmark_model_b,
            name,
            stop_event,
        )
    except KeyboardInterrupt:
        pass
    except StopRequested:
        pass
    except Exception:
        logging.info(traceback.format_exc())
    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name=name,
            reason="benchmark_inference_process finally",
        )
        os._exit(0)


def _benchmark_inference_loop(
    worker_id: int,
    flags: SimpleNamespace,
    infer_batch_queue,
    infer_res_queues,
    infer_request_buffers,
    infer_result_buffers,
    device: int,
    benchmark_model_a,
    benchmark_model_b,
    name: str,
    stop_event,
) -> None:
    setproctitle.setproctitle(name)
    torch.cuda.set_device(device)
    device_str = f"cuda:{device}"

    benchmark_model_a.eval()
    benchmark_model_b.eval()

    logging.info("%s benchmark inference: compact active-player model-id forward", name)

    get_batch_wait_logger = RollingImmediateWaitLogger(
        metric_name="benchmark_inference_worker.get_batch",
        log_interval_sec=60.0,
    )

    while True:
        request_item = timed_queue_get_or_stop(
            infer_batch_queue,
            stop_event,
            timeout_sec=0.1,
            wait_logger=get_batch_wait_logger,
            stream_key=f"benchmark:worker={int(worker_id)}",
            extra_context=f"device={int(device)}",
        )
        assert isinstance(request_item, tuple), (
            f"Expected infer_batch_queue tuple, got {type(request_item)}"
        )
        assert len(request_item) == 3, (
            f"Expected infer_batch_queue tuple of len 3, got len={len(request_item)}"
        )
        actor_ids, req_indices, model_by_seat_parts = request_item
        parts = [infer_request_buffers[req_idx] for req_idx in req_indices]

        batch = stack_buffers(parts, dim=0)
        assert isinstance(batch, dict) and "obs_LEARN_INFER" in batch
        obs = batch["obs_LEARN_INFER"]
        assert isinstance(obs, dict)
        ref = obs["orbit_planet_features"]
        assert isinstance(ref, torch.Tensor) and ref.ndim == 4
        n_infer = len(req_indices)
        assert int(ref.shape[0]) % n_infer == 0, (tuple(ref.shape), n_infer)
        e_roll = int(ref.shape[0]) // n_infer

        model_by_seat_full = torch.cat(model_by_seat_parts, dim=0)
        assert isinstance(model_by_seat_full, torch.Tensor)
        assert model_by_seat_full.dtype == torch.int64, model_by_seat_full.dtype
        assert tuple(model_by_seat_full.shape) == (
            int(ref.shape[0]),
            int(flags.agents_max_cnt),
        ), (
            tuple(model_by_seat_full.shape),
            tuple(ref.shape),
            int(flags.agents_max_cnt),
        )
        assert int(model_by_seat_full.min().item()) >= 0
        assert int(model_by_seat_full.max().item()) <= 1
        enemy_mask = batch["obs_LEARN_INFER"]["orbit_enemy_mask"]
        assert isinstance(enemy_mask, torch.Tensor)
        assert tuple(enemy_mask.shape) == (
            int(ref.shape[0]),
            int(flags.agents_max_cnt),
            3,
        ), (tuple(enemy_mask.shape), tuple(ref.shape), int(flags.agents_max_cnt))
        has_4p_enemy_axis = (enemy_mask[:, :, 1:] > 0.5).any(dim=(1, 2))
        game_num_players = torch.where(
            has_4p_enemy_axis,
            torch.full_like(has_4p_enemy_axis, 4, dtype=torch.int64),
            torch.full_like(has_4p_enemy_axis, 2, dtype=torch.int64),
        )
        assert game_num_players.shape == (int(ref.shape[0]),), game_num_players.shape
        batch["frozen_model_by_player_axis_LEARN"] = orbit_model_by_player_axis_from_seats(
            model_by_seat_full,
            game_num_players,
        )

        assert hasattr(flags, "benchmark_sample_a")
        assert hasattr(flags, "benchmark_sample_b")
        benchmark_sample_by_model_id = (
            bool(flags.benchmark_sample_a),
            bool(flags.benchmark_sample_b),
        )
        use_bf16 = bool(flags.inference_use_bf16)

        def _run_model_once(
            model,
            infer_batch,
            *,
            is_sample: bool,
            include_value_head: bool,
        ):
            orbit_assert_available_action_mask_contract(
                infer_batch["obs_LEARN_INFER"]["available_action_mask"],
                label="benchmark_inference",
            )
            model.set_is_sample(bool(is_sample))
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

        compact_batch, active_positions = _compact_active_player_batch_cpu(
            batch,
            players=int(flags.agents_max_cnt),
        )
        active_model_ids = model_by_seat_full[
            active_positions[:, 0],
            active_positions[:, 1],
        ].to(dtype=torch.long)

        benchmark_models = (benchmark_model_a, benchmark_model_b)
        for req_idx in req_indices:
            _fill_infer_result_defaults_cpu(infer_result_buffers[req_idx])

        for model_id_tensor in torch.unique(active_model_ids, sorted=True):
            model_id = int(model_id_tensor.item())
            assert 0 <= model_id < len(benchmark_models), model_id
            group_cpu = (active_model_ids == model_id).nonzero(as_tuple=False).reshape(-1)
            assert int(group_cpu.shape[0]) > 0
            group_batch_cpu = _index_batch_axis(compact_batch, group_cpu)
            group_batch = tree_to_device(group_batch_cpu, device_str)
            res = _run_model_once(
                benchmark_models[model_id],
                group_batch,
                is_sample=benchmark_sample_by_model_id[model_id],
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
