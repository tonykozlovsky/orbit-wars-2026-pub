from collections import deque
import math
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace
import logging
import setproctitle
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from .common import (
    ENTROPY_HEAD_KEYS,
    compute_lr_lambda,
    compute_target_entropy_combined,
    configure_process_cpu_thread_limits,
    ema_alpha_compound_per_step,
    entropy_head_tuple_from_config,
    get_checkpoint_file,
    raise_if_stop_requested,
    queue_get_or_timeout,
    queue_put_or_stop,
    set_stop_event_with_reason,
    StopRequested,
)


_REWARD_HEAD_KEYS = ("baseline", "production_delta")
_REWARD_HEAD_LOSS_COST_KEYS = (
    "baseline_cost",
    "baseline_terminal_loss_cost",
    "upgo_cost",
    "upgo_original_cost",
    "vtrace_cost",
)


def _reward_head_config_mapping(
    values: dict[str, object] | SimpleNamespace,
    name: str,
) -> dict[str, object]:
    if isinstance(values, SimpleNamespace):
        mapping = vars(values)
    else:
        assert isinstance(values, dict), (name, type(values))
        mapping = values
    assert tuple(mapping.keys()) == _REWARD_HEAD_KEYS, (
        name,
        tuple(mapping.keys()),
        _REWARD_HEAD_KEYS,
    )
    return mapping


def _reward_head_float_config(flags, name: str, reward_head: str) -> float:
    values = getattr(flags, name)
    mapping = _reward_head_config_mapping(values, name)
    return float(mapping[reward_head])


def _reward_head_logging_enabled(flags, reward_head: str) -> bool:
    assert reward_head in _REWARD_HEAD_KEYS, reward_head
    return any(
        _reward_head_float_config(flags, cost_key, reward_head) != 0.0
        for cost_key in _REWARD_HEAD_LOSS_COST_KEYS
    )


class RollingAverage:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)
        self.total = 0.0

    def add(self, value):
        if len(self.window) == self.window.maxlen:
            self.total -= self.window[0]
        self.window.append(value)
        self.total += value

    def average(self):
        return self.total / len(self.window) if self.window else 0.0

    def count(self):
        return len(self.window)

    def is_full(self):
        return len(self.window) == self.window.maxlen


NEW_CONTROLLER_TARGET_MIN = 0.0001
NEW_CONTROLLER_THRESHOLD_MAX = 1000.0


def _log_scale_unit_interval_update(value: float, signed_delta: float) -> float:
    v = max(NEW_CONTROLLER_TARGET_MIN, min(float(value), 1.0))
    delta = float(signed_delta)
    return max(NEW_CONTROLLER_TARGET_MIN, min(1.0, v * math.exp(delta)))


def _log_scale_threshold_decrease(value: float, delta: float) -> float:
    v = max(1.0, min(float(value), NEW_CONTROLLER_THRESHOLD_MAX))
    d = float(delta)
    assert v >= 1.0, value
    assert d > 0.0, delta
    return max(1.0, v * math.exp(-d))


def _log_scale_threshold_increase(value: float, delta: float) -> float:
    v = max(1.0, min(float(value), NEW_CONTROLLER_THRESHOLD_MAX))
    d = float(delta)
    assert v >= 1.0, value
    assert d > 0.0, delta
    return min(NEW_CONTROLLER_THRESHOLD_MAX, v * math.exp(d))


lock = threading.Lock()

LOSS_GROUPS = ('rl_train',)
losses = {k: None for k in LOSS_GROUPS}
clip_grad = {k: None for k in LOSS_GROUPS}


def _wandb_config_value(value):
    if isinstance(value, SimpleNamespace):
        return {
            k: _wandb_config_value(v)
            for k, v in vars(value).items()
        }
    if isinstance(value, dict):
        return {
            k: _wandb_config_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_wandb_config_value(v) for v in value]
    if isinstance(value, tuple):
        return [_wandb_config_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def fill_losses(stats_queue_learner, stop_event):
    global losses
    global clip_grad
    try:
        while True:
            learner_item, timed_out = queue_get_or_timeout(
                stats_queue_learner, stop_event, timeout_sec=0.1
            )
            if timed_out:
                continue
            learner_batch, loss_group = learner_item
            with lock:
                assert loss_group in losses, f'Unknown loss group={loss_group}'
                if 'losses' in learner_batch:
                    losses[loss_group] = learner_batch['losses']
                # Explicitly reset clip_grad payload when it is not sent by learner
                # (e.g. enable_clip_grad=False) to avoid stale logging.
                clip_grad[loss_group] = learner_batch.get('clip_grad', None)
    except KeyboardInterrupt:
        pass
    except StopRequested:
        pass
    except Exception as e:
        logging.info(traceback.format_exc())
    finally:
        set_stop_event_with_reason(
            stop_event,
            process_name="fill_losses",
            reason="fill_losses finally",
        )
        os._exit(0)



def fix_keys(stats):
    result = {}

    def recursive(s, prefix):
        if isinstance(s, dict):
            for key, value in s.items():
                if isinstance(value, dict):
                    recursive(value, prefix + key + '.')
                else:
                    result[prefix + key] = value

    recursive(stats, '')

    return result


def smoothed_update(smoothed, stats):
    for key, value in stats.items():
        if key not in smoothed:
            window_size = 100 if 'Winrate.' in key else 100
            smoothed[key] = RollingAverage(window_size=window_size)

        smoothed[key].add(value)


def smoothed_get_dict(smoothed, stats):
    # Only emit smoothed metrics once we have a full window of samples.
    # IMPORTANT: Only emit _SMO_ for keys that are present in this batch's stats.
    # Otherwise we can publish a stale constant _SMO_ value even when no new samples
    # arrived for that metric (e.g. some batches omit a metric or a metric is missing/NaN
    # in the current batch), while _RAW_ disappears. That looks like a "stuck" metric.
    result = {
        key + '_SMO_': smoothed[key].average()
        for key in stats
        if (key in smoothed) and (smoothed[key].is_full() or ('Winrate.' in key))
    }
    for key, value in stats.items():
        result[key + '_RAW_'] = value

    return result


def _critic_mc_logging_result(batch: dict, t_steps: int, bt: str) -> dict[str, float]:
    critic_mc = batch["critic_mc_STAT"]
    assert isinstance(critic_mc, dict)
    valid = critic_mc["valid"][:t_steps]
    assert isinstance(valid, torch.Tensor) and valid.dtype == torch.bool
    assert valid.ndim == 3, tuple(valid.shape)
    count = critic_mc["count"][:t_steps].to(dtype=torch.float64)
    assert isinstance(count, torch.Tensor)
    assert tuple(count.shape) == tuple(valid.shape), (
        tuple(count.shape),
        tuple(valid.shape),
    )

    count_sum = float(count.sum().item())
    if count_sum <= 0.0:
        return {}

    def summed(key: str) -> float:
        value = critic_mc[key][:t_steps].to(dtype=torch.float64)
        assert isinstance(value, torch.Tensor)
        assert tuple(value.shape) == tuple(valid.shape), (
            key,
            tuple(value.shape),
            tuple(valid.shape),
        )
        assert torch.isfinite(value).all().item(), key
        return float(value.sum().item())

    sqerr_sum = summed("sqerr_sum")
    abserr_sum = summed("abserr_sum")
    err_sum = summed("err_sum")
    return_sum = summed("return_sum")
    return_sq_sum = summed("return_sq_sum")
    value_sum = summed("value_sum")
    value_sq_sum = summed("value_sq_sum")
    value_return_sum = summed("value_return_sum")

    return_mean = return_sum / count_sum
    value_mean = value_sum / count_sum
    err_mean = err_sum / count_sum
    var_return = return_sq_sum / count_sum - return_mean * return_mean
    var_value = value_sq_sum / count_sum - value_mean * value_mean
    var_error = sqerr_sum / count_sum - err_mean * err_mean

    out = {
        f"critic_mc_mse{bt}": sqerr_sum / count_sum,
        f"critic_mc_mae{bt}": abserr_sum / count_sum,
        f"critic_mc_bias{bt}": err_mean,
        f"critic_mc_transition_count{bt}": count_sum,
        f"critic_mc_episode_seat_count{bt}": float(valid.sum().item()),
    }
    first_error = critic_mc["first_error"][:t_steps]
    assert isinstance(first_error, torch.Tensor)
    assert tuple(first_error.shape) == tuple(valid.shape), (
        tuple(first_error.shape),
        tuple(valid.shape),
    )
    if valid.any().item():
        out[f"critic_mc_first_error{bt}"] = float(
            first_error[valid].to(dtype=torch.float64).mean().item()
        )
    min_ev_target_var = 1e-6
    if var_return > min_ev_target_var:
        out[f"critic_mc_ev{bt}"] = 1.0 - var_error / var_return
    if var_return > min_ev_target_var and var_value > min_ev_target_var:
        cov = value_return_sum / count_sum - value_mean * return_mean
        out[f"critic_mc_corr{bt}"] = cov / math.sqrt(var_value * var_return)

    return out


@torch.no_grad()
def stats_func(
    flags,
    stats_free_queue,
    stats_full_queue,
    stats_buffers,
    stats_queue_learner,
    shared_target_entropy,
    shared_shortfall_entropy,
    shared_mean_entropy_ema,
    shared_new_controller_temperature_threshold,
    shared_steps,
    shared_lr_lambda,
    name,
    stop_event,
):
    enable_wandb = bool(flags.enable_wandb)
    enable_tensorboard = bool(flags.enable_tensorboard)
    assert not (enable_wandb and enable_tensorboard), (
        "enable_wandb and enable_tensorboard are mutually exclusive"
    )
    if enable_tensorboard:
        output_dir = Path(flags.output_dir)
        assert output_dir.is_dir(), flags.output_dir
        tensorboard_log_dir = output_dir
        logging.info("TensorBoard log directory: %s", tensorboard_log_dir)
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    all_threads = []
    try:
        setproctitle.setproctitle(name)
        configure_process_cpu_thread_limits()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        global losses
        global clip_grad

        smoothed = dict()
        # Only log metrics after they've changed at least once (i.e. not constant).
        # Keeps logs clean when a metric is permanently constant (e.g. always zero).
        changed_metric_keys = set()
        last_metric_values = {}

        if enable_wandb:
            wandb_project = flags.wandb_project
            wandb.init(
                project=wandb_project,
                name=name,
                reinit=True,
                config=_wandb_config_value(flags),
            )

        logging.info('stats_func started')

        n_batches = 0
        last_logged_time_by_group = {k: 0.0 for k in LOSS_GROUPS}

        lt = threading.Thread(target=fill_losses, args=(stats_queue_learner, stop_event), daemon=True)
        lt.start()
        all_threads.append(lt)

        last_shortfall_update_steps: int | None = None
        assert len(shared_new_controller_temperature_threshold) == len(ENTROPY_HEAD_KEYS)
        for threshold in shared_new_controller_temperature_threshold:
            assert threshold >= 1.0, threshold

        while True:
            raise_if_stop_requested(stop_event)
            batch_idx, timed_out = queue_get_or_timeout(
                stats_full_queue, stop_event, timeout_sec=0.001
            )
            if timed_out:
                continue

            batch = stats_buffers[batch_idx]

            assert isinstance(batch, dict), "Expected dict batch for stats rollout"

            steps = shared_steps.value

            regular_entropy_loss = bool(flags.classic_entropy) or bool(flags.ce_entropy)
            if regular_entropy_loss:
                target_mean_entropy_by_head = {
                    head: 0.0
                    for head in ENTROPY_HEAD_KEYS
                }
            else:
                max_te = entropy_head_tuple_from_config(
                    flags.target_mean_entropy_max,
                    label="target_mean_entropy_max",
                )
                if str(flags.target_mean_entropy_mode) == "ema_tracking":
                    mult = entropy_head_tuple_from_config(
                        flags.mean_entropy_ema_multiplier,
                        label="mean_entropy_ema_multiplier",
                    )
                    target_mean_entropy_by_head = {}
                    for i, head in enumerate(ENTROPY_HEAD_KEYS):
                        target_mean_entropy_by_head[head] = (
                            min(
                                float(shared_mean_entropy_ema[i]) * float(mult[i]),
                                float(max_te[i]),
                            )
                        )
                        shared_target_entropy[i] = float(target_mean_entropy_by_head[head])
                else:
                    final_targets = entropy_head_tuple_from_config(
                        flags.target_mean_entropy,
                        label="target_mean_entropy",
                    )
                    target_mean_entropy_by_head = {}
                    for i, head in enumerate(ENTROPY_HEAD_KEYS):
                        scheduled_target_mean_entropy = compute_target_entropy_combined(
                            step=int(steps),
                            total_steps=float(flags.total_steps),
                            warmup_steps=int(flags.target_entropy_warmup_steps),
                            final_target=float(final_targets[i]),
                            enable_decay=bool(flags.enable_entropy_decay),
                        )
                        target_mean_entropy_by_head[head] = min(
                            float(scheduled_target_mean_entropy),
                            float(max_te[i]),
                        )
                        shared_target_entropy[i] = float(target_mean_entropy_by_head[head])
            shared_lr_lambda.value = compute_lr_lambda(
                step=int(steps),
                total_steps=float(flags.total_steps),
                lr_warmup_steps=int(flags.lr_warmup_steps),
                initial_lr_lambda=float(flags.initial_lr_lambda),
                final_lr_lambda=float(flags.final_lr_lambda),
            )

            if (not regular_entropy_loss) and (not bool(getattr(flags, "use_new_controller", False))):
                _mn_sf = entropy_head_tuple_from_config(
                    flags.shortfall_entropy_min,
                    label="shortfall_entropy_min",
                )
                _mx_sf = entropy_head_tuple_from_config(
                    flags.shortfall_entropy_max,
                    label="shortfall_entropy_max",
                )
                for i in range(len(ENTROPY_HEAD_KEYS)):
                    head = ENTROPY_HEAD_KEYS[i]
                    upper_sf = min(float(_mx_sf[i]), float(target_mean_entropy_by_head[head]))
                    _sf_v = float(shared_shortfall_entropy[i])
                    shared_shortfall_entropy[i] = max(float(_mn_sf[i]), min(_sf_v, upper_sf))

            n_batches += 1

            # Rollout buffer time dim is unroll_length + 1; the last index is bootstrap-only obs
            # (dummy r/d). Stats use only the first unroll_length timesteps (real env transitions).
            T_stat = int(flags.unroll_length)
            reward_learn_stat = batch["reward_LEARN_STAT"]
            assert isinstance(reward_learn_stat, dict)
            assert tuple(reward_learn_stat.keys()) == _REWARD_HEAD_KEYS, (
                tuple(reward_learn_stat.keys()),
                _REWARD_HEAD_KEYS,
            )
            for reward_head, rv_full in reward_learn_stat.items():
                rv = rv_full[:T_stat]
                assert isinstance(rv, torch.Tensor)
                assert torch.isfinite(rv).all().item(), (
                    f"reward_LEARN_STAT.{reward_head}: NaN/inf detected"
                )
                if rv.numel() > 0:
                    max_seen = torch.abs(rv).max().item()
                    assert max_seen <= 1e9, (
                        f"reward_LEARN_STAT.{reward_head}: abs too large max={max_seen} "
                        f"shape={tuple(rv.shape)} dtype={rv.dtype}"
                    )

            bt = '_RL_TRAIN'
            algo_label = 'RL'

            phase_label = 'TRAIN'
            loss_group = 'rl_train'

            now = time.time()
            should_log_this_group = (now - last_logged_time_by_group[loss_group]) >= 1.0

            logging_result: dict[str, float] = {}
            done_stat = batch["done_LEARN_STAT"][:T_stat]
            assert isinstance(done_stat, torch.Tensor) and done_stat.dtype == torch.bool
            assert done_stat.ndim == 2
            mask = done_stat.unsqueeze(-1)
            plus_games = mask.detach().sum().item()
            if plus_games > 0:

                def get_logging_result_fast(
                    batch,
                    mask,
                    bt,
                ):
                    if "info_STAT" not in batch:
                        return {}, {}
                    assert mask.ndim == 3 and mask.dtype == torch.bool
                    t_steps = int(mask.shape[0])
                    ebs = int(mask.shape[1])
                    assert int(mask.shape[2]) == 1
                    m = mask.squeeze(-1)
                    assert tuple(m.shape) == (t_steps, ebs)
                    original_player_mask = batch["original_player_mask_STAT"][:t_steps] > 0.5
                    assert isinstance(original_player_mask, torch.Tensor)
                    assert original_player_mask.ndim == 3
                    assert int(original_player_mask.shape[0]) == t_steps
                    assert int(original_player_mask.shape[1]) == ebs

                    info = batch["info_STAT"]
                    logging_key_prefix = "LOGGING_STAT_"
                    plen = len(logging_key_prefix)
                    outcome_key = "LOGGING_STAT_game_result_cumsum"
                    assert outcome_key in info, sorted(info.keys())
                    outcome_val = info[outcome_key]
                    assert isinstance(outcome_val, torch.Tensor)
                    assert int(outcome_val.shape[0]) == t_steps + 1, (
                        "info_STAT LOGGING leaf time dim must be unroll_length + 1 "
                        f"(bootstrap row); key={outcome_key!r} got {tuple(outcome_val.shape)}"
                    )
                    outcome_val = outcome_val[:t_steps]
                    assert outcome_val.ndim == 3
                    assert int(outcome_val.shape[0]) == t_steps and int(outcome_val.shape[1]) == ebs
                    frozen_model_by_seat = batch["frozen_model_by_seat_STAT"][:t_steps]
                    assert isinstance(frozen_model_by_seat, torch.Tensor)
                    assert frozen_model_by_seat.dtype == torch.int64
                    assert frozen_model_by_seat.ndim == 3
                    assert int(frozen_model_by_seat.shape[0]) == t_steps
                    assert int(frozen_model_by_seat.shape[1]) == ebs
                    terminal_outcome = outcome_val[m]
                    terminal_model_by_seat = frozen_model_by_seat[m].to(
                        device=terminal_outcome.device,
                        dtype=torch.int64,
                    )
                    terminal_original_player_mask = original_player_mask[m].to(
                        device=terminal_outcome.device,
                        dtype=torch.bool,
                    )
                    assert terminal_model_by_seat.shape == terminal_original_player_mask.shape
                    terminal_player_mask_base = terminal_original_player_mask & (
                        terminal_model_by_seat == 0
                    )
                    active_counts = terminal_original_player_mask.sum(dim=1)
                    assert bool(torch.all((active_counts == 2) | (active_counts == 4)).item()), (
                        outcome_key,
                        active_counts,
                    )
                    assert bool(
                        torch.isfinite(
                            terminal_outcome[terminal_original_player_mask]
                        ).all().item()
                    ), terminal_outcome
                    original_winner_slots = (terminal_outcome > 0.5) & terminal_original_player_mask
                    winner_slots = original_winner_slots & terminal_player_mask_base
                    loser_slots = (terminal_outcome < -0.5) & terminal_player_mask_base
                    winner_counts = original_winner_slots.sum(dim=1)
                    valid_non_tie_games = winner_counts == 1
                    out = {}
                    winrate_pair_scores: dict[str, float] = {}

                    def accumulate_self_vs_opponent(
                        self_score: torch.Tensor,
                        opp_idx: int,
                        opponent_model_id: int,
                        player_label: str,
                    ) -> None:
                        assert opponent_model_id >= 0
                        opp_score = outcome_row[opp_idx]
                        assert torch.isfinite(opp_score).item()
                        metric_key = f"self_vs_{opponent_model_id}_{player_label}"
                        if bool((self_score > opp_score).item()):
                            winrate_pair_scores[metric_key] = (
                                winrate_pair_scores.get(metric_key, 0.0) + 1.0
                            )
                        elif bool((self_score < opp_score).item()):
                            winrate_pair_scores[metric_key] = (
                                winrate_pair_scores.get(metric_key, 0.0) - 1.0
                            )

                    for game_i in range(int(terminal_outcome.shape[0])):
                        active_count = int(active_counts[game_i].item())
                        assert active_count in (2, 4), active_count
                        player_label = "2P" if active_count == 2 else "4P"
                        active_mask = terminal_original_player_mask[game_i]
                        model_row = terminal_model_by_seat[game_i]
                        outcome_row = terminal_outcome[game_i]
                        self_indices = (
                            active_mask & (model_row == 0)
                        ).nonzero(as_tuple=False).reshape(-1)
                        active_indices = active_mask.nonzero(as_tuple=False).reshape(-1)
                        for self_idx_t in self_indices:
                            self_idx = int(self_idx_t.item())
                            self_score = outcome_row[self_idx]
                            assert torch.isfinite(self_score).item()
                            for opp_idx_t in active_indices:
                                opp_idx = int(opp_idx_t.item())
                                if opp_idx == self_idx:
                                    continue
                                accumulate_self_vs_opponent(
                                    self_score,
                                    opp_idx,
                                    int(model_row[opp_idx].item()),
                                    player_label,
                                )
                    for key, val in info.items():
                        if not key.startswith(logging_key_prefix):
                            continue
                        assert isinstance(val, torch.Tensor)
                        assert int(val.shape[0]) == t_steps + 1, (
                            "info_STAT LOGGING leaf time dim must be unroll_length + 1 "
                            f"(bootstrap row); key={key!r} got {tuple(val.shape)}"
                        )
                        val = val[:t_steps]
                        assert val.ndim == 3
                        assert int(val.shape[0]) == t_steps and int(val.shape[1]) == ebs
                        terminal = val[m]
                        terminal_player_mask = original_player_mask[m].to(
                            device=terminal.device,
                            dtype=torch.bool,
                        ) & (terminal_model_by_seat.to(device=terminal.device) == 0)
                        if terminal.numel() == 0:
                            continue
                        assert torch.equal(
                            terminal_player_mask.cpu(),
                            terminal_player_mask_base.cpu(),
                        ), key
                        finite = torch.isfinite(terminal) & terminal_player_mask
                        for num_players, player_label in ((2, "2P"), (4, "4P")):
                            player_game_mask = active_counts == int(num_players)
                            non_tie_player_game_mask = player_game_mask & valid_non_tie_games
                            win_finite = (
                                finite
                                & winner_slots.to(device=terminal.device)
                                & non_tie_player_game_mask.unsqueeze(1)
                            )
                            lose_finite = (
                                finite
                                & loser_slots.to(device=terminal.device)
                                & non_tie_player_game_mask.unsqueeze(1)
                            )
                            if win_finite.any().item():
                                metric_key = f"{key[plen:]}_{player_label}_WIN{bt}"
                                out[metric_key] = float(terminal[win_finite].mean().item())
                            if lose_finite.any().item():
                                metric_key = f"{key[plen:]}_{player_label}_LOSE{bt}"
                                out[metric_key] = float(terminal[lose_finite].mean().item())

                    return out, winrate_pair_scores

                logging_result, winrate_result = get_logging_result_fast(batch, mask, bt)
            else:
                winrate_result = {}

            original_player_mask = batch["original_player_mask_STAT"][:T_stat] > 0.5
            assert isinstance(original_player_mask, torch.Tensor)
            frozen_model_by_seat = batch["frozen_model_by_seat_STAT"][:T_stat]
            assert isinstance(frozen_model_by_seat, torch.Tensor)
            assert frozen_model_by_seat.dtype == torch.int64

            everything: dict[str, float] = {
                f'done_true_count{bt}': float(plus_games),
            }
            everything.update(logging_result)
            stats = {
                'Everything': everything,
                'Loss': {},
                'Features': {},
                'Model': {},
                'Entropy': {},
                'Action': {},
                'Diag': {},
                'Winrate': winrate_result,
            }
            stats['Diag'].update(_critic_mc_logging_result(batch, T_stat, bt))

            for reward_head, rewards_full in reward_learn_stat.items():
                rewards = rewards_full[:T_stat]
                assert isinstance(rewards, torch.Tensor)
                assert original_player_mask.shape == rewards.shape
                assert frozen_model_by_seat.shape == rewards.shape
                reward_mask = original_player_mask.to(device=rewards.device) & (
                    frozen_model_by_seat.to(device=rewards.device) == 0
                )
                masked_rewards = rewards[reward_mask]
                if masked_rewards.numel() > 0:
                    reward_metric_suffix = "" if reward_head == "baseline" else f"_{reward_head}"
                    stats['Everything'][f'batch_reward_sum{reward_metric_suffix}{bt}'] = (
                        masked_rewards.sum().item()
                    )
                    stats['Everything'][f'batch_reward_max{reward_metric_suffix}{bt}'] = (
                        masked_rewards.max().item()
                    )
                    stats['Everything'][f'batch_reward_mean{reward_metric_suffix}{bt}'] = (
                        masked_rewards.mean().item()
                    )
            baseline_rewards = reward_learn_stat["baseline"][:T_stat]
            assert isinstance(baseline_rewards, torch.Tensor)
            baseline_reward_mask = original_player_mask.to(device=baseline_rewards.device) & (
                frozen_model_by_seat.to(device=baseline_rewards.device) == 0
            )
            if baseline_rewards[baseline_reward_mask].numel() > 0:
                if not regular_entropy_loss:
                    for head in ENTROPY_HEAD_KEYS:
                        stats['Entropy'][f'target_mean_entropy_{head}{bt}'] = (
                            target_mean_entropy_by_head[head]
                        )
            #if flags.teacher:

            with lock:
                # Always incorporate the most recent learner losses into stats so SMO updates
                # are driven by per-batch cadence. Rate limiting should only affect publishing
                # to W&B (below), not the rolling windows.
                if losses[loss_group] is not None:
                    loss_payload = losses[loss_group]
                    clip_payload = clip_grad[loss_group]
                    if should_log_this_group:
                        losses[loss_group] = None
                        clip_grad[loss_group] = None

                    lr_key = 'learning_rate'
                    if lr_key in loss_payload:
                        stats['Everything'][f'learning_rate_{algo_label}_{phase_label}'] = loss_payload[lr_key]
                    learner_4p_ratio_key = 'learner_4p_sample_ratio'
                    assert learner_4p_ratio_key in loss_payload, (
                        f"learner stats payload must include {learner_4p_ratio_key!r}"
                    )
                    stats['Everything'][f'{learner_4p_ratio_key}_{algo_label}_{phase_label}'] = loss_payload[
                        learner_4p_ratio_key
                    ]

                    if clip_payload is not None:
                        if 'total_norm' in clip_payload:
                            stats['Everything'][f'total_norm_{algo_label}_{phase_label}'] = clip_payload['total_norm']
                        if 'clip_grad_value' in clip_payload:
                            stats['Everything'][f'clip_grad_value_{algo_label}_{phase_label}'] = clip_payload['clip_grad_value']

                    loss_keys = [
                        'vtrace_pg_loss', 'upgo_pg_loss', 'upgo_original_pg_loss',
                        'baseline_loss', 'baseline_loss_terminal',
                        'baseline_explained_variance',
                        'teacher_kl_loss',
                        'teacher_kl_loss_orig',
                        'teacher_kl_loss_orig_2p',
                        'teacher_kl_loss_orig_4p',
                        'teacher_baseline_loss',
                        'teacher_baseline_loss_orig',
                        'teacher_baseline_explained_variance',
                        'entropy_loss_spawn_fleet',
                        'policy_logits_l2_loss',
                        'policy_centered_logits_l2_loss',
                        'policy_logits_l2_raw',
                        'policy_centered_logits_l2_raw',
                        'temperature_compensation_kl_loss',
                        'temperature_compensation_kl_loss_raw',
                        'temperature_compensation_kl_loss_spawn_fleet',
                        'entropy_target_loss',
                        'total_loss',
                        'learner_kl_div',
                        'learner_replay_debug_n_cells_compared',
                        'learner_replay_debug_n_missing_version',
                        'learner_replay_debug_max_abs_diff_speed',
                        'learner_replay_debug_max_abs_diff_yaw',
                        'learn_ver_min',
                        'learn_ver_max',
                    ]

                    for k in loss_keys:
                        if k in loss_payload and loss_payload[k] is not None:
                            value = loss_payload[k]
                            #if k == 'entropy_loss':
                            #    value = -value
                            stats['Loss'][f'{k}_{algo_label}_{phase_label}'] = value
                    for k, v in loss_payload.items():
                        if v is None:
                            continue
                        if k.startswith('obs_ema_'):
                            stats['Features'][f'{k}_{algo_label}_{phase_label}'] = v
                        if k.startswith('target_min_entropy_'):
                            stats['Entropy'][f'{k}_{algo_label}_{phase_label}'] = v
                        if k.startswith('entropy_floor_'):
                            stats['Entropy'][f'{k}_{algo_label}_{phase_label}'] = v
                        if k.startswith('Entropy.class_freq_'):
                            entropy_key = k[len('Entropy.'):]
                            stats['Entropy'][f'{entropy_key}_{algo_label}_{phase_label}'] = v
                        if k.startswith('Action.class_freq_'):
                            action_key = k[len('Action.'):]
                            stats['Action'][f'{action_key}_{algo_label}_{phase_label}'] = v
                        if k.startswith('sampled_') and k.endswith('_class_freq'):
                            stats['Entropy'][f'{k}_{algo_label}_{phase_label}'] = v
                        if (
                            k.startswith('policy_logits_l2_')
                            or k.startswith('policy_centered_logits_l2_')
                        ):
                            stats['Loss'][f'{k}_{algo_label}_{phase_label}'] = v
                        if k == 'value_total_norm' or k.startswith('value_clip_grad_'):
                            stats['Everything'][f'{k}_{algo_label}_{phase_label}'] = v
                        if (
                            k.startswith('vtrace_pg_loss_')
                            or k.startswith('upgo_pg_loss_')
                            or k.startswith('upgo_original_pg_loss_')
                            or k.startswith('baseline_loss_')
                            or k.startswith('baseline_explained_variance_')
                        ):
                            stats['Loss'][f'{k}_{algo_label}_{phase_label}'] = v
                        if k.startswith('model_param_') or k.startswith('model_grad_'):
                            stats['Model'][f'{k}_{algo_label}_{phase_label}'] = v
                        if k.startswith('Diagnostic_'):
                            stats['Diag'][f'{k}_{algo_label}_{phase_label}'] = v

                    actual_mean_entropy_by_head: dict[str, float] = {}
                    entropy_sample_count_by_head: dict[str, float] = {}
                    for head in ENTROPY_HEAD_KEYS:
                        key = f'mean_entropy_{head}'
                        assert key in loss_payload, (
                            f"learner stats payload must include {key!r}"
                        )
                        assert loss_payload[key] is not None, (
                            f"learner stats payload has null {key!r}"
                        )
                        actual_mean_entropy_by_head[head] = float(loss_payload[key])
                        count_key = f'entropy_sample_count_{head}'
                        assert count_key in loss_payload, (
                            f"learner stats payload must include {count_key!r}"
                        )
                        assert loss_payload[count_key] is not None, (
                            f"learner stats payload has null {count_key!r}"
                        )
                        entropy_sample_count_by_head[head] = float(loss_payload[count_key])
                    cur_steps = int(shared_steps.value)
                    if last_shortfall_update_steps is None:
                        step_span = 1
                    else:
                        step_span = cur_steps - last_shortfall_update_steps
                        assert step_span >= 0, (
                            f"shared_steps must not decrease: cur_steps={cur_steps} "
                            f"last_shortfall_update_steps={last_shortfall_update_steps}"
                        )

                    if regular_entropy_loss:
                        last_shortfall_update_steps = cur_steps
                        if step_span > 0:
                            alpha_eff = ema_alpha_compound_per_step(
                                step_span,
                                float(flags.mean_entropy_ema_alpha_per_step),
                            )
                            for i, head in enumerate(ENTROPY_HEAD_KEYS):
                                if entropy_sample_count_by_head[head] <= 0.0:
                                    continue
                                ema_old = float(shared_mean_entropy_ema[i])
                                shared_mean_entropy_ema[i] = (
                                    alpha_eff * actual_mean_entropy_by_head[head]
                                    + (1.0 - alpha_eff) * ema_old
                                )
                    elif bool(getattr(flags, "use_new_controller", False)):
                        if step_span == 0:
                            for i, head in enumerate(ENTROPY_HEAD_KEYS):
                                stats['Entropy'][
                                    f'new_controller_target_min_entropy_{head}_{algo_label}_{phase_label}'
                                ] = float(shared_shortfall_entropy[i])
                                stats['Entropy'][
                                    f'new_controller_temperature_threshold_{head}_{algo_label}_{phase_label}'
                                ] = float(shared_new_controller_temperature_threshold[i])
                        else:
                            last_shortfall_update_steps = cur_steps

                            alpha_eff = ema_alpha_compound_per_step(
                                step_span,
                                float(flags.mean_entropy_ema_alpha_per_step),
                            )
                            target_up_log_delta = (
                                float(flags.new_controller_target_up_log_delta_per_step)
                                * float(step_span)
                            )
                            target_down_log_delta = (
                                float(flags.new_controller_target_down_log_delta_per_step)
                                * float(step_span)
                            )
                            threshold_up_log_delta = (
                                float(flags.new_controller_threshold_up_log_delta_per_step)
                                * float(step_span)
                            )
                            threshold_down_log_delta = (
                                float(flags.new_controller_threshold_down_log_delta_per_step)
                                * float(step_span)
                            )
                            assert target_up_log_delta > 0.0, target_up_log_delta
                            assert target_down_log_delta > 0.0, target_down_log_delta
                            assert threshold_up_log_delta > 0.0, threshold_up_log_delta
                            assert threshold_down_log_delta > 0.0, threshold_down_log_delta
                            target_cap = entropy_head_tuple_from_config(
                                flags.new_controller_target_min_entropy,
                                label="new_controller_target_min_entropy",
                            )
                            target_min = entropy_head_tuple_from_config(
                                flags.new_controller_target_min_entropy_min_value,
                                label="new_controller_target_min_entropy_min_value",
                            )
                            for i, head in enumerate(ENTROPY_HEAD_KEYS):
                                if entropy_sample_count_by_head[head] > 0.0:
                                    actual_mean_entropy = actual_mean_entropy_by_head[head]
                                    ema_old = float(shared_mean_entropy_ema[i])
                                    shared_mean_entropy_ema[i] = (
                                        alpha_eff * actual_mean_entropy
                                        + (1.0 - alpha_eff) * ema_old
                                    )
                                temperature_key = f'entropy_floor_{head}_temperature_mean'
                                if (
                                    temperature_key not in loss_payload
                                    or loss_payload[temperature_key] is None
                                ):
                                    continue
                                temperature_signal = float(loss_payload[temperature_key])
                                threshold = float(shared_new_controller_temperature_threshold[i])
                                assert threshold >= 1.0, threshold
                                cur_sf = float(shared_shortfall_entropy[i])
                                if temperature_signal > threshold:
                                    cur_sf = _log_scale_unit_interval_update(
                                        cur_sf,
                                        -target_down_log_delta,
                                    )
                                    threshold = _log_scale_threshold_increase(
                                        threshold,
                                        threshold_up_log_delta,
                                    )
                                else:
                                    cur_sf = _log_scale_unit_interval_update(
                                        cur_sf,
                                        target_up_log_delta,
                                    )
                                    threshold = _log_scale_threshold_decrease(
                                        threshold,
                                        threshold_down_log_delta,
                                    )
                                cur_sf = max(
                                    float(target_min[i]),
                                    min(cur_sf, float(target_cap[i])),
                                )
                                shared_new_controller_temperature_threshold[i] = threshold
                                shared_shortfall_entropy[i] = cur_sf
                                stats['Entropy'][
                                    f'new_controller_target_min_entropy_{head}_{algo_label}_{phase_label}'
                                ] = cur_sf
                                stats['Entropy'][
                                    f'new_controller_temperature_threshold_{head}_{algo_label}_{phase_label}'
                                ] = threshold
                    elif step_span == 0:
                        for i, head in enumerate(ENTROPY_HEAD_KEYS):
                            stats['Entropy'][
                                f'target_min_entropy_{head}_{algo_label}_{phase_label}'
                            ] = float(shared_shortfall_entropy[i])
                    else:
                        last_shortfall_update_steps = cur_steps

                        alpha_eff = ema_alpha_compound_per_step(
                            step_span,
                            float(flags.mean_entropy_ema_alpha_per_step),
                        )
                        mn = entropy_head_tuple_from_config(
                            flags.shortfall_entropy_min,
                            label="shortfall_entropy_min",
                        )
                        mx = entropy_head_tuple_from_config(
                            flags.shortfall_entropy_max,
                            label="shortfall_entropy_max",
                        )
                        mult = entropy_head_tuple_from_config(
                            flags.mean_entropy_ema_multiplier,
                            label="mean_entropy_ema_multiplier",
                        )
                        max_te = entropy_head_tuple_from_config(
                            flags.target_mean_entropy_max,
                            label="target_mean_entropy_max",
                        )
                        increase_delta = entropy_head_tuple_from_config(
                            flags.shortfall_entropy_increase_delta_per_step,
                            label="shortfall_entropy_increase_delta_per_step",
                        )
                        decrease_delta = entropy_head_tuple_from_config(
                            flags.shortfall_entropy_decrease_delta_per_step,
                            label="shortfall_entropy_decrease_delta_per_step",
                        )
                        for i, head in enumerate(ENTROPY_HEAD_KEYS):
                            if entropy_sample_count_by_head[head] <= 0.0:
                                stats['Entropy'][
                                    f'target_min_entropy_{head}_{algo_label}_{phase_label}'
                                ] = float(shared_shortfall_entropy[i])
                                continue
                            actual_mean_entropy = actual_mean_entropy_by_head[head]
                            ema_old = float(shared_mean_entropy_ema[i])
                            ema_new = (
                                alpha_eff * actual_mean_entropy
                                + (1.0 - alpha_eff) * ema_old
                            )
                            shared_mean_entropy_ema[i] = ema_new
                            if str(flags.target_mean_entropy_mode) == "ema_tracking":
                                tm = min(ema_new * float(mult[i]), float(max_te[i]))
                                shared_target_entropy[i] = float(tm)
                            else:
                                tm = float(target_mean_entropy_by_head[head])
                            cur_sf = float(shared_shortfall_entropy[i])

                            if actual_mean_entropy < tm:
                                adjust = (
                                    float(increase_delta[i])
                                    * float(step_span)
                                    * float(ema_new)
                                )
                                cur_sf = min(float(mx[i]), cur_sf + adjust)
                            else:
                                adjust = (
                                    float(decrease_delta[i])
                                    * float(step_span)
                                    * float(ema_new)
                                )
                                cur_sf = max(float(mn[i]), cur_sf - adjust)
                            upper_sf = min(float(mx[i]), float(tm))
                            cur_sf = max(float(mn[i]), min(cur_sf, upper_sf))
                            shared_shortfall_entropy[i] = cur_sf
                            stats['Entropy'][
                                f'target_min_entropy_{head}_{algo_label}_{phase_label}'
                            ] = cur_sf

                    assert 'mean_entropy' in loss_payload, (
                        "learner stats payload must include aggregate 'mean_entropy'"
                    )
                    stats['Entropy'][f'mean_entropy_{algo_label}_{phase_label}'] = (
                        loss_payload['mean_entropy']
                    )
                    for head in ENTROPY_HEAD_KEYS:
                        stats['Entropy'][
                            f'mean_entropy_{head}_{algo_label}_{phase_label}'
                        ] = actual_mean_entropy_by_head[head]
                        stats['Entropy'][
                            f'entropy_sample_count_{head}_{algo_label}_{phase_label}'
                        ] = entropy_sample_count_by_head[head]

                    for reward_head in _REWARD_HEAD_KEYS:
                        for k in (
                            f'popart_mean_{reward_head}',
                            f'popart_std_{reward_head}',
                            f'reward_ema_mean_{reward_head}',
                            f'reward_ema_std_{reward_head}',
                        ):
                            if k in loss_payload and loss_payload[k] is not None:
                                stats['Everything'][f'{k}_{algo_label}_{phase_label}'] = (
                                    loss_payload[k]
                                )

                    adv_metrics = (
                        'vtrace_pg_adv_mean',
                        'vtrace_pg_adv_std',
                        'vtrace_pg_adv_abs_mean',
                        'vtrace_pg_adv_abs_max',
                        'vtrace_rho_mean',
                        'vtrace_rho_std',
                        'vtrace_rho_abs_max',
                        'vtrace_log_rho_mean',
                        'vtrace_log_rho_std',
                        'vtrace_rho_clip_frac',
                        'upgo_adv_mean',
                        'upgo_adv_std',
                        'upgo_adv_abs_mean',
                        'upgo_adv_abs_max',
                        'upgo_original_adv_mean',
                        'upgo_original_adv_std',
                        'upgo_original_adv_abs_mean',
                        'upgo_original_adv_abs_max',
                        'td_error_mean',
                        'td_error_std',
                        'td_error_abs_mean',
                        'td_error_abs_max',
                    )
                    for m in adv_metrics:
                        if m in loss_payload and loss_payload[m] is not None:
                            stats['Everything'][f'{m}_{algo_label}_{phase_label}'] = loss_payload[m]

            if regular_entropy_loss or str(flags.target_mean_entropy_mode) == "ema_tracking":
                for i, head in enumerate(ENTROPY_HEAD_KEYS):
                    stats['Entropy'][
                        f'mean_entropy_ema_{head}_{algo_label}_{phase_label}'
                    ] = float(shared_mean_entropy_ema[i])

            stats = fix_keys(stats)
            if not _reward_head_logging_enabled(flags, "production_delta"):
                stats = {
                    k: v for k, v in stats.items()
                    if "production_delta" not in k
                }

            # Update "changed" set using the *base* (pre-smoothing) keys.
            for k, v in stats.items():
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    assert v.numel() == 1, f"Non-scalar metric for key={k} shape={tuple(v.shape)}"
                    v = v.item()
                if k in last_metric_values:
                    if True or v != last_metric_values[k]:
                        changed_metric_keys.add(k)
                last_metric_values[k] = v

            smoothed_update(smoothed, stats)
            stats = smoothed_get_dict(smoothed, stats)

            # Filter out metrics that have never changed.
            def _base_metric_key(metric_key: str) -> str:
                if metric_key.endswith('_SMO_') or metric_key.endswith('_RAW_'):
                    return metric_key[:-5]
                return metric_key

            stats = {
                k: v for k, v in stats.items()
                if _base_metric_key(k) in changed_metric_keys
            }

            step = int(shared_steps.value)
            publish_logs = should_log_this_group or (plus_games > 0)
            if publish_logs:
                if should_log_this_group:
                    last_logged_time_by_group[loss_group] = now
                stats_float = {}
                for k, v in stats.items():
                    if isinstance(v, torch.Tensor):
                        assert v.numel() == 1, (
                            f"Non-scalar metric for log key={k} shape={tuple(v.shape)}"
                        )
                        stats_float[k] = float(v.item())
                    else:
                        stats_float[k] = float(v)
                if enable_wandb:
                    wandb.log(stats_float, step=step)
                if enable_tensorboard:
                    for k, v in stats_float.items():
                        tensorboard_writer.add_scalar(k, v, step)

            queue_put_or_stop(
                stats_free_queue, batch_idx, stop_event, timeout_sec=0.1
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
            reason="stats_func finally",
        )
        if enable_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logging.info(traceback.format_exc())
        if enable_tensorboard:
            tensorboard_writer.close()
        os._exit(0)
