import json
import logging
import multiprocessing as mp
import os
import queue
from datetime import datetime
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path
import setproctitle
import torch

from ...gym.create_env import create_env
from ...gym.wall_tree_profiler import WallTreeProfiler, profiler_span
from ...gym.obs_wrapper import (
    ORBIT_MAX_PLANETS,
    ORBIT_MOVE_CLASSES_PER_TARGET,
    ORBIT_PER_PLANET_MOVE_CLASSES,
    ORBIT_PLANET_ACTION_SLOTS,
    ORBIT_PLAYER_AXIS_SLOTS,
    orbit_active_policy_slots,
    orbit_assert_cpp_py_policy_obs_trace_matches,
)
from ...gym.orbit_wars_env import (
    _orbit_policy_obs_canonical_planet_perm_for_slot,
    _orbit_policy_obs_canonicalize_slot_tensor,
)
from ...gym.debug_viewer import DebugViewer, append_frames_to_tape
from .buffer_utils import copy_buffers, get_buffers_with_tag, tree_to_device
from .common import (
    StopRequested,
    orbit_model_by_player_axis_from_seats,
    raise_if_stop_requested,
    set_stop_event_with_reason,
)
_IMPALA_RUN_ARTIFACT_ROOT_ENV = "IMPALA_RUN_ARTIFACT_ROOT"
_AOTI_POLICY_INPUT_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_enemy_mask",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_planet_pairwise_features",
    "available_action_mask",
)
_POLICY_INPUT_DUMP_FORMAT = "orbit_policy_network_inputs_dump_v1"


def _rl_debug_tape_root() -> Path:
    return Path(os.environ[_IMPALA_RUN_ARTIFACT_ROOT_ENV]) / "tapes"

_RL_DEBUG_TAPE_NAME_AT_PROCESS_START: str | None = None


def _policy_tape_rows(
    infer_obs_ep: dict[str, torch.Tensor],
    agent_out: dict,
    ac_orbit: torch.Tensor,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    assert "final_policy_logits_LEARN" in agent_out
    final_logits_by_action = agent_out["final_policy_logits_LEARN"]
    assert isinstance(final_logits_by_action, dict), type(final_logits_by_action)
    assert tuple(final_logits_by_action.keys()) == ("spawn_fleet",), (
        tuple(final_logits_by_action.keys()),
    )
    final_logits = final_logits_by_action["spawn_fleet"].detach().cpu()
    planet_mask = infer_obs_ep["orbit_planet_mask"].detach().cpu()
    available_action_mask = infer_obs_ep["available_action_mask"].detach().cpu()
    assert isinstance(planet_mask, torch.Tensor)
    assert isinstance(available_action_mask, torch.Tensor)
    e, p, n, one = ac_orbit.shape
    assert one == 1
    assert tuple(final_logits.shape[:3]) == (e, p, n), final_logits.shape
    model_action_width = int(final_logits.shape[3])
    assert model_action_width % n == 0, (final_logits.shape, n)
    model_actions_per_target = model_action_width // n
    assert 1 <= model_actions_per_target <= int(ORBIT_MOVE_CLASSES_PER_TARGET), (
        model_actions_per_target,
        ORBIT_MOVE_CLASSES_PER_TARGET,
    )
    assert tuple(planet_mask.shape) == (e, p, n), planet_mask.shape
    assert tuple(available_action_mask.shape) == (
        e,
        p,
        n,
        ORBIT_PER_PLANET_MOVE_CLASSES,
    ), available_action_mask.shape
    assert available_action_mask.dtype == torch.int8, available_action_mask.dtype
    available_action_mask = available_action_mask.reshape(
        e,
        p,
        n,
        n,
        int(ORBIT_MOVE_CLASSES_PER_TARGET),
    )[..., :model_actions_per_target].reshape(e, p, n, model_action_width)
    cls = ac_orbit[..., 0].to(dtype=torch.int64).cpu()
    assert tuple(cls.shape) == (e, p, n), cls.shape
    available = available_action_mask > 0
    valid_policy = (planet_mask > 0.5) & (available.sum(dim=-1) > 1)
    neg_large = torch.finfo(final_logits.dtype).min / 16.0
    probs = torch.softmax(final_logits.masked_fill(~available, neg_large), dim=-1)
    k = min(5, model_action_width)
    top_vals, top_idx = probs.topk(k, dim=-1)
    top_dst = top_idx // int(model_actions_per_target)
    top_subindex = top_idx % int(model_actions_per_target)
    top_idx_env = top_dst * int(ORBIT_MOVE_CLASSES_PER_TARGET) + top_subindex
    final_pred: list[list[list[list[list[float]]]]] = []
    for ei in range(e):
        env_rows: list[list[list[list[float]]]] = []
        for pi in range(p):
            seat_rows: list[list[list[float]]] = []
            for ni in range(n):
                row: list[list[float]] = []
                for ki in range(k):
                    row.append(
                        [
                            float(top_idx_env[ei, pi, ni, ki].item()),
                            float(top_vals[ei, pi, ni, ki].item()),
                        ]
                    )
                seat_rows.append(row)
            env_rows.append(seat_rows)
        final_pred.append(env_rows)

    pred: dict[str, object] = {
        "final_policy": final_pred,
    }
    target = {
        "final_policy": cls.to(dtype=torch.float32),
    }
    valid = {
        "final_policy": valid_policy,
    }
    return pred, target, valid


_ORBIT_POLICY_OUTPUT_LOGITS_RTOL = 1e-5
_ORBIT_POLICY_OUTPUT_LOGITS_ATOL = 1e-2


def _orbit_policy_output_mismatch_detail(
    *,
    key: str,
    ref_slot: int,
    other_slot: int,
    ref: torch.Tensor,
    other: torch.Tensor,
) -> str:
    assert isinstance(key, str) and len(key) > 0, key
    assert isinstance(ref, torch.Tensor)
    assert isinstance(other, torch.Tensor)
    assert tuple(ref.shape) == tuple(other.shape), (key, tuple(ref.shape), tuple(other.shape))
    if ref.dtype == torch.bool or other.dtype == torch.bool:
        diff = ref != other
        score = diff.to(dtype=torch.float32)
    elif torch.is_floating_point(ref) or torch.is_floating_point(other):
        ref_f = ref.to(dtype=torch.float32)
        other_f = other.to(dtype=torch.float32)
        finite_pair = torch.isfinite(ref_f) & torch.isfinite(other_f)
        same_nonfinite = (~finite_pair) & (ref_f == other_f)
        raw_score = (ref_f - other_f).abs()
        score = torch.where(
            finite_pair,
            raw_score,
            torch.where(same_nonfinite, torch.zeros_like(raw_score), torch.full_like(raw_score, float("inf"))),
        )
        diff = score > 0
    else:
        diff = ref != other
        score = diff.to(dtype=torch.float32)
    assert bool(diff.any().item()), (key, ref_slot, other_slot)
    flat = int(torch.argmax(score.reshape(-1)).item())
    idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat), ref.shape))
    ref_v = ref[idx].item()
    other_v = other[idx].item()
    return (
        f"key={key}; ref_slot={int(ref_slot)}; other_slot={int(other_slot)}; "
        f"shape={tuple(ref.shape)}; idx={idx}; ref={ref_v}; other={other_v}; "
        f"max_abs_diff={float(score.reshape(-1)[flat].item())}"
    )


def _orbit_canonicalize_policy_prediction_slot_tensor(
    *,
    key: str,
    slot_tensor: torch.Tensor,
    planet_perm: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(key, str) and len(key) > 0, key
    assert isinstance(slot_tensor, torch.Tensor)
    assert isinstance(planet_perm, torch.Tensor)
    assert tuple(planet_perm.shape) == (ORBIT_MAX_PLANETS,), planet_perm.shape
    if key == "final_policy_logits":
        assert slot_tensor.ndim == 2, (key, tuple(slot_tensor.shape))
        assert int(slot_tensor.shape[0]) == int(ORBIT_PLANET_ACTION_SLOTS), (
            key,
            tuple(slot_tensor.shape),
        )
        action_width = int(slot_tensor.shape[1])
        assert action_width % int(ORBIT_PLANET_ACTION_SLOTS) == 0, (
            key,
            tuple(slot_tensor.shape),
        )
        actions_per_target = action_width // int(ORBIT_PLANET_ACTION_SLOTS)
        by_dst_action = slot_tensor.reshape(
            ORBIT_PLANET_ACTION_SLOTS,
            ORBIT_PLANET_ACTION_SLOTS,
            actions_per_target,
        )
        by_dst_action = by_dst_action.index_select(0, planet_perm).index_select(1, planet_perm)
        return by_dst_action.reshape(ORBIT_PLANET_ACTION_SLOTS, action_width)
    raise AssertionError(("unexpected orbit policy prediction key", key))


def _orbit_policy_action_mismatch_detail(
    *,
    ref_slot: int,
    other_slot: int,
    ref_action: torch.Tensor,
    other_action: torch.Tensor,
    ref_logits: torch.Tensor,
    other_logits: torch.Tensor,
) -> str:
    detail = _orbit_policy_output_mismatch_detail(
        key="actions_LEARN.spawn_fleet",
        ref_slot=ref_slot,
        other_slot=other_slot,
        ref=ref_action,
        other=other_action,
    )
    diff = ref_action != other_action
    flat = int(torch.argmax(diff.to(dtype=torch.float32).reshape(-1)).item())
    planet_slot, trailing_dim = (
        int(v) for v in torch.unravel_index(torch.tensor(flat), ref_action.shape)
    )
    assert trailing_dim == 0, (planet_slot, trailing_dim)
    ref_class = int(ref_action[planet_slot, 0].item())
    other_class = int(other_action[planet_slot, 0].item())
    ref_row = ref_logits[planet_slot].to(dtype=torch.float32)
    other_row = other_logits[planet_slot].to(dtype=torch.float32)
    ref_top_vals, ref_top_idx = torch.topk(ref_row, k=2, dim=0)
    other_top_vals, other_top_idx = torch.topk(other_row, k=2, dim=0)
    return (
        detail
        + f"; ref_row_ref_action_logit={float(ref_row[ref_class].item())}"
        + f"; ref_row_other_action_logit={float(ref_row[other_class].item())}"
        + f"; ref_row_top_class={int(ref_top_idx[0].item())}"
        + f"; ref_row_top2_margin={float((ref_top_vals[0] - ref_top_vals[1]).item())}"
        + f"; other_row_ref_action_logit={float(other_row[ref_class].item())}"
        + f"; other_row_other_action_logit={float(other_row[other_class].item())}"
        + f"; other_row_top_class={int(other_top_idx[0].item())}"
        + f"; other_row_top2_margin={float((other_top_vals[0] - other_top_vals[1]).item())}"
    )


def _orbit_assert_policy_outputs_invariant_across_active_players(
    *,
    infer_obs_ep: dict[str, torch.Tensor],
    agent_out: dict,
    ac_orbit: torch.Tensor,
    num_agents: int,
) -> str | None:
    na = int(num_agents)
    active_slots = orbit_active_policy_slots(na)
    assert len(active_slots) >= 2, active_slots
    assert isinstance(infer_obs_ep, dict)
    assert isinstance(agent_out, dict)
    assert isinstance(ac_orbit, torch.Tensor)
    e, p, n, one = ac_orbit.shape
    assert one == 1
    assert p == ORBIT_PLAYER_AXIS_SLOTS, (p, ORBIT_PLAYER_AXIS_SLOTS)
    assert n == ORBIT_PLANET_ACTION_SLOTS, (n, ORBIT_PLANET_ACTION_SLOTS)
    assert "final_policy_logits_LEARN" in agent_out
    final_logits_by_action = agent_out["final_policy_logits_LEARN"]
    assert isinstance(final_logits_by_action, dict), type(final_logits_by_action)
    assert tuple(final_logits_by_action.keys()) == ("spawn_fleet",), (
        tuple(final_logits_by_action.keys()),
    )
    final_logits = final_logits_by_action["spawn_fleet"].detach().cpu()
    assert isinstance(final_logits, torch.Tensor)
    for ei in range(e):
        policy_obs = {
            "orbit_planet_features": infer_obs_ep["orbit_planet_features"][ei],
            "orbit_planet_mask": infer_obs_ep["orbit_planet_mask"][ei],
        }
        planet_perm_by_slot = {
            int(slot): _orbit_policy_obs_canonical_planet_perm_for_slot(
                policy_obs=policy_obs,
                slot=int(slot),
            )
            for slot in active_slots
        }
        ref_slot = int(active_slots[0])
        ref_action = _orbit_policy_obs_canonicalize_slot_tensor(
            key="action_taken_index",
            slot_tensor=ac_orbit[ei, ref_slot].detach().cpu(),
            planet_perm=planet_perm_by_slot[ref_slot].cpu(),
        )
        ref_final_logits = _orbit_canonicalize_policy_prediction_slot_tensor(
            key="final_policy_logits",
            slot_tensor=final_logits[ei, ref_slot],
            planet_perm=planet_perm_by_slot[ref_slot].cpu(),
        )
        for slot in active_slots[1:]:
            other_slot = int(slot)
            other_action = _orbit_policy_obs_canonicalize_slot_tensor(
                key="action_taken_index",
                slot_tensor=ac_orbit[ei, other_slot].detach().cpu(),
                planet_perm=planet_perm_by_slot[other_slot].cpu(),
            )
            if not torch.equal(ref_action, other_action):
                other_final_logits = _orbit_canonicalize_policy_prediction_slot_tensor(
                    key="final_policy_logits",
                    slot_tensor=final_logits[ei, other_slot],
                    planet_perm=planet_perm_by_slot[other_slot].cpu(),
                )
                detail = _orbit_policy_action_mismatch_detail(
                    ref_slot=ref_slot,
                    other_slot=other_slot,
                    ref_action=ref_action,
                    other_action=other_action,
                    ref_logits=ref_final_logits,
                    other_logits=other_final_logits,
                )
                return (
                    "orbit policy output invariance mismatch before env.step\n"
                    f"env_index={int(ei)}\n"
                    f"num_agents={na}\n"
                    f"active_slots={active_slots}\n"
                    "planet_order=canonical_sort_by_player_relative_planet_features\n"
                    + detail
                )
    return None


def note_process_start_for_rl_debug_tape() -> None:
    """Call once at training process entry (e.g. ``monobeast.train``); fixes tape folder name for the run."""
    global _RL_DEBUG_TAPE_NAME_AT_PROCESS_START
    assert _RL_DEBUG_TAPE_NAME_AT_PROCESS_START is None, (
        "note_process_start_for_rl_debug_tape must be called at most once"
    )
    _RL_DEBUG_TAPE_NAME_AT_PROCESS_START = (
        f"rl_vis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )


def process_start_rl_debug_tape_name() -> str:
    assert _RL_DEBUG_TAPE_NAME_AT_PROCESS_START is not None, (
        "note_process_start_for_rl_debug_tape must run before RL debug tape is used"
    )
    return _RL_DEBUG_TAPE_NAME_AT_PROCESS_START


def _append_aoti_example_inputs_from_policy_obs(
    *,
    samples: list[tuple[torch.Tensor, ...]],
    infer_obs_ep: dict[str, torch.Tensor],
    num_agents: int,
) -> None:
    assert int(num_agents) == 4, num_agents
    assert isinstance(infer_obs_ep, dict), type(infer_obs_ep)
    active_slots = orbit_active_policy_slots(int(num_agents))
    assert tuple(active_slots) == (0, 1, 2, 3), active_slots
    for key in _AOTI_POLICY_INPUT_KEYS:
        assert key in infer_obs_ep, key
        t = infer_obs_ep[key]
        assert isinstance(t, torch.Tensor), (key, type(t))
        assert t.ndim >= 2, (key, tuple(t.shape))
        assert int(t.shape[0]) == 1, (key, tuple(t.shape))
        assert int(t.shape[1]) == ORBIT_PLAYER_AXIS_SLOTS, (key, tuple(t.shape))
    for slot in active_slots:
        slot_i = int(slot)
        samples.append(
            tuple(
                infer_obs_ep[key][:, slot_i : slot_i + 1].detach().cpu().contiguous()
                for key in _AOTI_POLICY_INPUT_KEYS
            )
        )


def _save_aoti_example_inputs(
    *,
    samples: list[tuple[torch.Tensor, ...]],
    path: Path,
) -> None:
    assert len(samples) > 0, "AOTI example input capture completed without samples"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "orbit_policy_logits_aoti_example_inputs_v1",
            "input_keys": _AOTI_POLICY_INPUT_KEYS,
            "samples_per_step": 4,
            "samples": samples,
        },
        str(path),
    )
    logging.info("Saved %d AOTI example input samples -> %s", len(samples), path)


def _append_first_player_policy_input_dump(
    *,
    samples: list[tuple[torch.Tensor, ...]],
    infer_obs_ep: dict[str, torch.Tensor],
) -> None:
    for key in _AOTI_POLICY_INPUT_KEYS:
        assert key in infer_obs_ep, key
        t = infer_obs_ep[key]
        assert isinstance(t, torch.Tensor), (key, type(t))
        assert t.ndim >= 2, (key, tuple(t.shape))
        assert int(t.shape[0]) == 1, (key, tuple(t.shape))
        assert int(t.shape[1]) == ORBIT_PLAYER_AXIS_SLOTS, (key, tuple(t.shape))
    samples.append(
        tuple(
            infer_obs_ep[key][:, 0:1].detach().cpu().contiguous()
            for key in _AOTI_POLICY_INPUT_KEYS
        )
    )


def _save_policy_input_dump(
    *,
    samples: list[tuple[torch.Tensor, ...]],
    path: Path,
) -> None:
    assert len(samples) > 0, "policy input dump completed without samples"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _POLICY_INPUT_DUMP_FORMAT,
            "input_keys": _AOTI_POLICY_INPUT_KEYS,
            "samples_per_step": 1,
            "samples": samples,
        },
        str(path),
    )
    logging.info("Saved %d policy input dump samples -> %s", len(samples), path)


def visualize_process_with_scene_queue(
    flags,
    actor_index,
    actor_model,
    actor_device,
    name: str,
    visualization_queue: mp.Queue,
    stop_event,
) -> None:
    try:
        setproctitle.setproctitle(name)

        device = torch.device(f"cuda:{actor_device}")

        model = actor_model
        model.eval()

        flags_for_vis = deepcopy(flags)
        save_aoti_example_inputs = bool(flags_for_vis.rl_vis_save_aoti_example_inputs)
        dump_inputs = bool(flags_for_vis.rl_vis_dump_inputs)
        rl_vis_model_is_sample = bool(flags_for_vis.rl_vis_model_is_sample)
        rl_vis_model_shuffle_identity_ids = bool(flags_for_vis.rl_vis_model_shuffle_identity_ids)
        use_bf16 = bool(flags_for_vis.inference_use_bf16)
        fixed_episode_seed = int(flags_for_vis.rl_vis_episode_seed)
        assert fixed_episode_seed >= -1, fixed_episode_seed
        if dump_inputs:
            assert int(actor_index) == 0, actor_index
            assert int(flags_for_vis.num_rl_vis_actors) == 1, flags_for_vis.num_rl_vis_actors
            assert len(str(flags_for_vis.rl_vis_dump_inputs_path)) > 0
            flags_for_vis.vis_n_actor_envs = 1
        if save_aoti_example_inputs:
            assert int(actor_index) == 0, actor_index
            assert int(flags_for_vis.num_rl_vis_actors) == 1, flags_for_vis.num_rl_vis_actors
            assert len(str(flags_for_vis.rl_vis_aoti_example_inputs_path)) > 0
            flags_for_vis.orbit_num_agents = 4
            flags_for_vis.vis_n_actor_envs = 1
        if fixed_episode_seed >= 0:
            assert int(actor_index) == 0, actor_index
            assert int(flags_for_vis.num_rl_vis_actors) == 1, flags_for_vis.num_rl_vis_actors
            flags_for_vis.vis_n_actor_envs = 1
            flags_for_vis.orbit_configuration.seed = int(fixed_episode_seed)
        vbs = int(flags_for_vis.vis_n_actor_envs)
        assert vbs >= 1
        flags_for_vis.n_actor_envs = vbs
        # Training-only random episode cut (RewardWrapper); keep viz episodes until real terminal.
        flags_for_vis.enable_desync = False
        _envs_val = bool(flags_for_vis.enable_envs_validation)
        # Alternating 2p/4p uses one ``orbit_num_agents`` per batch (see EnvBatchWrapper).
        vis_alternate_2p_4p = (
            vbs == 1 and not save_aoti_example_inputs and fixed_episode_seed < 0
        )
        record_policy_prediction_tape = int(flags_for_vis.num_actors) == 0
        vis_completed_episodes = 0
        aoti_example_input_samples: list[tuple[torch.Tensor, ...]] = []
        input_dump_samples: list[tuple[torch.Tensor, ...]] = []
        policy_output_invariance_check_disabled = False
        wall_prof = (
            WallTreeProfiler() if bool(flags_for_vis.enable_actor_wall_tree_profiler) else None
        )
        vis_wall_prof_episode_ix = 0
        episode_wall_t0: float | None = None

        def _vis_env_from_flags():
            if vis_alternate_2p_4p:
                na = 4 if vis_completed_episodes % 2 != 0 else 2
                flags_for_vis.orbit_num_agents = na
            return create_env(
                flags_for_vis,
                device="cpu",
                visualize=True,
                visualize_sim_env=True,
                visualization_queue=visualization_queue,
                record_tape=True,
                cpp_env_obs_full=not _envs_val,
                cpp_env_obs_validate=_envs_val,
                wall_profiler=wall_prof,
            )

        env = _vis_env_from_flags()
        env_output = env.reset()
        if wall_prof is not None:
            episode_wall_t0 = time.perf_counter()
        while True:
            raise_if_stop_requested(stop_event)
            batch_cpu = get_buffers_with_tag(env_output, device=None, tag="INFER")
            assert batch_cpu is not None
            if _envs_val:
                orbit_assert_cpp_py_policy_obs_trace_matches(env)
            infer_obs_ep = copy_buffers(batch_cpu["obs_LEARN_INFER"])
            fe0 = infer_obs_ep["orbit_planet_features"]
            assert isinstance(fe0, torch.Tensor) and fe0.ndim == 4
            e_vis = int(fe0.shape[0])
            p_vis = int(fe0.shape[1])
            assert p_vis == ORBIT_PLAYER_AXIS_SLOTS, (p_vis, ORBIT_PLAYER_AXIS_SLOTS)
            game_num_players = torch.full(
                (e_vis,),
                int(flags_for_vis.orbit_num_agents),
                dtype=torch.int64,
            )
            model_by_seat = torch.zeros(
                e_vis,
                ORBIT_PLAYER_AXIS_SLOTS,
                dtype=torch.int64,
            )
            batch_cpu["frozen_model_by_player_axis_LEARN"] = (
                orbit_model_by_player_axis_from_seats(model_by_seat, game_num_players)
            )
            if save_aoti_example_inputs:
                assert e_vis == 1, e_vis
                assert int(flags_for_vis.orbit_num_agents) == 4, flags_for_vis.orbit_num_agents
                _append_aoti_example_inputs_from_policy_obs(
                    samples=aoti_example_input_samples,
                    infer_obs_ep=infer_obs_ep,
                    num_agents=int(flags_for_vis.orbit_num_agents),
                )
            if dump_inputs:
                assert e_vis == 1, e_vis
                assert int(flags_for_vis.orbit_num_agents) == 4, flags_for_vis.orbit_num_agents
                _append_first_player_policy_input_dump(
                    samples=input_dump_samples,
                    infer_obs_ep=infer_obs_ep,
                )

            with profiler_span(wall_prof, "policy"):
                model.set_is_sample(rl_vis_model_is_sample)
                model.set_shuffle_identity_ids(rl_vis_model_shuffle_identity_ids)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        agent_out = model(
                            tree_to_device(batch_cpu, device),
                            output_full_policy_log_probs=False,
                            include_policy_logits_pre_action_mask=(
                                record_policy_prediction_tape or _envs_val
                            ),
                            include_final_policy_logits=_envs_val,
                            include_value_head=True,
                        )
            assert isinstance(agent_out, dict), "policy must return a dict"
            assert "actions_LEARN" in agent_out, "policy output must contain 'actions_LEARN'"
            assert "baseline_LEARN" in agent_out, "policy output must contain 'baseline_LEARN'"
            baseline_learn = agent_out["baseline_LEARN"]
            assert isinstance(baseline_learn, dict)
            assert "baseline" in baseline_learn, sorted(baseline_learn.keys())
            bl = baseline_learn["baseline"]
            assert isinstance(bl, torch.Tensor)
            bl_cpu = bl.detach().cpu()
            ac_orbit = agent_out["actions_LEARN"]["spawn_fleet"].detach().cpu()
            assert tuple(ac_orbit.shape) == (
                e_vis,
                p_vis,
                ORBIT_PLANET_ACTION_SLOTS,
                1,
            )
            tape_pred_rows: dict[str, object] | None = None
            tape_tgt_rows: dict[str, torch.Tensor] | None = None
            tape_valid_rows: dict[str, torch.Tensor] | None = None
            if record_policy_prediction_tape:
                tape_pred_rows, tape_tgt_rows, tape_valid_rows = _policy_tape_rows(
                    infer_obs_ep, agent_out, ac_orbit
                )
            if _envs_val and not policy_output_invariance_check_disabled:
                policy_output_invariance_mismatch = (
                    _orbit_assert_policy_outputs_invariant_across_active_players(
                        infer_obs_ep=infer_obs_ep,
                        agent_out=agent_out,
                        ac_orbit=ac_orbit,
                        num_agents=int(flags_for_vis.orbit_num_agents),
                    )
                )
                if policy_output_invariance_mismatch is not None:
                    policy_output_invariance_check_disabled = True
                    logging.info(
                        "Disabling orbit policy output invariance check for the rest of this episode:\n%s",
                        policy_output_invariance_mismatch,
                    )
            env_output = env.step(
                ac_orbit,
                tape_baseline_learn=bl_cpu,
                tape_supervised_pred=tape_pred_rows,
                tape_supervised_target=tape_tgt_rows,
                tape_supervised_valid=tape_valid_rows,
            )
            d = env_output["done_LEARN_STAT"] 
            assert isinstance(d, torch.Tensor) and d.dtype == torch.bool
            env_output = env.reset_where_done(env_output, d)
            done_any = bool(d.any().item())
            if done_any:
                policy_output_invariance_check_disabled = False
            if wall_prof is not None and done_any:
                assert episode_wall_t0 is not None
                wall_ms = (time.perf_counter() - episode_wall_t0) * 1000.0
                wall_prof.summary(
                    f"rl_vis_actor_episode_{vis_wall_prof_episode_ix}",
                    wall_ms=wall_ms,
                )
                wall_prof.clear()
                vis_wall_prof_episode_ix += 1
            if save_aoti_example_inputs and done_any:
                _save_aoti_example_inputs(
                    samples=aoti_example_input_samples,
                    path=Path(str(flags_for_vis.rl_vis_aoti_example_inputs_path)),
                )
                set_stop_event_with_reason(
                    stop_event,
                    process_name=name,
                    reason="saved rl_vis AOTI example inputs",
                )
                return
            if dump_inputs and done_any:
                _save_policy_input_dump(
                    samples=input_dump_samples,
                    path=Path(str(flags_for_vis.rl_vis_dump_inputs_path)),
                )
                set_stop_event_with_reason(
                    stop_event,
                    process_name=name,
                    reason="saved rl_vis policy input dump",
                )
                return
            if done_any and vis_alternate_2p_4p:
                vis_completed_episodes += int(d.to(dtype=torch.int64).sum().item())
                env = _vis_env_from_flags()
                env_output = env.reset()
            if wall_prof is not None and done_any:
                episode_wall_t0 = time.perf_counter()
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
            reason="visualize_process_with_scene_queue finally",
        )
        os._exit(0)


def run_rl_visualization_tape_consumer(
    visualization_queue_rl,
    stop_event,
    run_tape_name: str,
) -> None:
    assert isinstance(run_tape_name, str), type(run_tape_name)
    assert run_tape_name.startswith("rl_vis_"), run_tape_name
    tape_root = _rl_debug_tape_root()
    run_tape = DebugViewer(tape_root, run_tape_name)
    logging.info("RL debug tape for this run: %s", run_tape.frames_path)
    try:
        while True:
            time.sleep(0.001)
            try:
                drawings_rl = visualization_queue_rl.get_nowait()
            except queue.Empty:
                raise_if_stop_requested(stop_event)
                drawings_rl = None
            if drawings_rl is not None:
                if isinstance(drawings_rl, str):
                    tape = json.loads(drawings_rl)
                else:
                    tape = drawings_rl
                frames = tape["frames"]
                append_frames_to_tape(tape_root, run_tape_name, frames)
                logging.info(
                    "Appended RL debug tape episode (%d frames) -> %s",
                    len(frames),
                    run_tape.frames_path,
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
            process_name="run_rl_visualization_tape_consumer",
            reason="run_rl_visualization_tape_consumer finally",
        )
