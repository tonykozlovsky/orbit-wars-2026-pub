import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from ..torchbeast.core.common import (
    CheckpointConfig,
    ENTROPY_HEAD_KEYS,
    checkpoint_config_from_checkpoint_file,
    load_yaml,
)
from .impala_orbit_model_hyperparams import default_training_model_dict

_IMPALA_PROJECT_ROOT_RAW = os.environ.get(
    "IMPALA_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[3]),
).strip()
assert _IMPALA_PROJECT_ROOT_RAW, "IMPALA_PROJECT_ROOT must be non-empty"
IMPALA_PROJECT_ROOT = Path(_IMPALA_PROJECT_ROOT_RAW).expanduser().resolve()
assert (IMPALA_PROJECT_ROOT / "python" / "src").is_dir(), (
    f"expected Impala layout python/src under {IMPALA_PROJECT_ROOT}"
)
# Same directory as ``run_monobeast`` timestamped runs (``<run_id>/runs``, ``tapes``).
IMPALA_OUTPUT_ARTIFACT_ROOT = IMPALA_PROJECT_ROOT / "outputs"


def coerce_resume_checkpoint_value(value: Any) -> CheckpointConfig | None:
    """Normalize ``resume_checkpoint`` from YAML/overrides: str path, dict, or :class:`CheckpointConfig`."""
    if value is None:
        return None
    if isinstance(value, CheckpointConfig):
        return value
    if isinstance(value, dict):
        return CheckpointConfig(
            torch_io=value["torch_io"],
            name=value.get("name", "latest"),
        )
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        raw = Path(s)
        if raw.is_absolute():
            p = raw.expanduser().resolve()
        else:
            norm = s.replace("\\", "/").lstrip("./")
            if norm.startswith("outputs/") or norm == "outputs":
                p = (IMPALA_PROJECT_ROOT / s).resolve()
            else:
                p = (IMPALA_OUTPUT_ARTIFACT_ROOT / s).resolve()
        return checkpoint_config_from_checkpoint_file(p)
    raise TypeError(
        f"resume_checkpoint must be None, str, dict, or CheckpointConfig, got {type(value)}"
    )

# ``Field(json_schema_extra=…)``: marks fields recomputed in ``ImpalaTrainingConfig._derive_and_validate``
# when the merged YAML left them unset (``None``). Used to strip baked floats from ``default_training_config_dict`` dumps.
_JSON_EXTRA_DERIVE_AFTER_MERGE = "x_derive_after_merge"


def _default_entropy_head_values(value: float) -> dict[str, float]:
    return {
        head: float(value)
        for head in ENTROPY_HEAD_KEYS
    }


def _default_reward_head_values(value: float) -> dict[str, float]:
    return {
        "baseline": float(value),
        "production_delta": float(value),
    }


def _default_reward_head_bool_values(value: bool) -> dict[str, bool]:
    return {
        "baseline": bool(value),
        "production_delta": bool(value),
    }


def _default_model_stub() -> dict[str, Any]:
    return default_training_model_dict()


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class OptimizerConfig(_StrictConfig):
    # optimizer_name: str = "Adam"
    # optimizer_kwargs: dict[str, Any] = Field(
    #     default_factory=lambda: {
    #         "lr": 1e-4,
    #     }
    # )
    optimizer_name: str = "AdamW"
    optimizer_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "lr": 1e-6,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0, #1e-4,
            "fused": True,
        }
    )
    # optimizer_name: str = "RMSprop"
    # optimizer_kwargs: dict[str, Any] = Field(
    #     default_factory=lambda: {
    #         "lr": 1e-6,
    #         "alpha": 0.99,
    #         "eps": 1e-8,
    #         "weight_decay": 0.001,
    #         "momentum": 0.0,
    #     }
    # )


class LoggingConfig(_StrictConfig):
    """Training log cadence (metrics go to Weights & Biases via ``wandb_project`` on the root config)."""

    frequency: int = 1000
    model_stats_frequency: int = 0


def teacher_kl_cost_at_step(
    kl_cost_initial: float | dict[str, float],
    kl_cost_decay_steps: int,
    step: int,
) -> float | dict[str, float]:
    if int(kl_cost_decay_steps) <= 0:
        if isinstance(kl_cost_initial, dict):
            return {k: float(v) for k, v in kl_cost_initial.items()}
        return float(kl_cost_initial)
    bounded_step = min(max(int(step), 0), int(kl_cost_decay_steps))
    remaining = 1.0 - (float(bounded_step) / float(kl_cost_decay_steps))
    if isinstance(kl_cost_initial, dict):
        return {
            k: max(0.0, float(v) * remaining)
            for k, v in kl_cost_initial.items()
        }
    return max(0.0, float(kl_cost_initial) * remaining)


class TeacherCheckpointConfig(_StrictConfig):
    num_players: Literal[2, 4] | None
    checkpoint: CheckpointConfig

    @pydantic.field_validator("checkpoint", mode="before")
    @classmethod
    def _checkpoint_coerce(cls, value: Any) -> CheckpointConfig:
        coerced = coerce_resume_checkpoint_value(value)
        if coerced is None:
            raise ValueError("teacher.checkpoints checkpoint must be set")
        return coerced


class TeacherConfig(_StrictConfig):
    checkpoint: CheckpointConfig | None = None
    checkpoints: list[TeacherCheckpointConfig] = Field(default_factory=list)
    model: str = "model.yaml"
    kl_cost_initial: float | dict[str, float] = 1.0
    kl_cost_decay_steps: int = 0
    baseline_cost: float = 0.0
    moving_steps: int = Field(default=0, ge=0)
    zero_missing_policy_actions: bool = False

    @pydantic.field_validator("checkpoint", mode="before")
    @classmethod
    def _shared_checkpoint_coerce(cls, value: Any) -> CheckpointConfig | None:
        return coerce_resume_checkpoint_value(value)

    @pydantic.field_validator("checkpoints", mode="before")
    @classmethod
    def _checkpoint_list_coerce(cls, value: Any) -> list[TeacherCheckpointConfig]:
        if value is None:
            return []
        assert isinstance(value, list), (
            f"teacher.checkpoints must be a list, got {type(value)}"
        )
        out: list[TeacherCheckpointConfig] = []
        for item in value:
            if isinstance(item, tuple):
                assert len(item) == 2, item
                num_players, checkpoint = item
                assert int(num_players) in (2, 4), num_players
                num_players = int(num_players)
            else:
                num_players = None
                checkpoint = item
            out.append(
                TeacherCheckpointConfig(
                    num_players=num_players,
                    checkpoint=checkpoint,
                )
            )
        return out

    @pydantic.model_validator(mode="after")
    def _validate_checkpoints(self) -> "TeacherConfig":
        if self.checkpoint is not None:
            if len(self.checkpoints) != 0:
                raise ValueError("teacher.checkpoint cannot be combined with teacher.checkpoints")
            object.__setattr__(
                self,
                "checkpoints",
                [TeacherCheckpointConfig(num_players=None, checkpoint=self.checkpoint)],
            )
        if len(self.checkpoints) == 0:
            raise ValueError("teacher.checkpoints must be non-empty")
        if float(self.baseline_cost) < 0.0:
            raise ValueError("teacher.baseline_cost must be >= 0")
        shared_checkpoints = [
            checkpoint
            for checkpoint in self.checkpoints
            if checkpoint.num_players is None
        ]
        if len(shared_checkpoints) > 0 and len(self.checkpoints) != 1:
            raise ValueError("teacher shared checkpoint cannot be combined with per-size checkpoints")
        if len(self.checkpoints) > 1 and int(self.moving_steps) != 0:
            raise ValueError("teacher.moving_steps must be 0 for multi-teacher distillation")
        num_players = [
            int(checkpoint.num_players)
            for checkpoint in self.checkpoints
            if checkpoint.num_players is not None
        ]
        if len(set(num_players)) != len(num_players):
            raise ValueError("teacher.checkpoints must not repeat num_players")
        return self

    def kl_cost_at_step(self, step: int) -> float | dict[str, float]:
        return teacher_kl_cost_at_step(
            self.kl_cost_initial,
            int(self.kl_cost_decay_steps),
            int(step),
        )


class FrozenOpponentCheckpointConfig(_StrictConfig):
    num_players: Literal[2, 4] | None
    checkpoint: CheckpointConfig

    @pydantic.field_validator("checkpoint", mode="before")
    @classmethod
    def _checkpoint_coerce(cls, value: Any) -> CheckpointConfig:
        coerced = coerce_resume_checkpoint_value(value)
        assert coerced is not None, "frozen_opponent.checkpoints checkpoint must be set"
        return coerced


class FrozenOpponentConfig(_StrictConfig):
    probability: float = Field(default=0.0, ge=0.0, le=1.0)
    all_frozen: bool = False
    no_selfplay: bool = False
    learn_on_frozen: bool = False
    checkpoint: CheckpointConfig | None = None
    checkpoints: list[FrozenOpponentCheckpointConfig] = Field(default_factory=list)
    seed: int = 0

    @pydantic.field_validator("checkpoint", mode="before")
    @classmethod
    def _shared_checkpoint_coerce(cls, value: Any) -> CheckpointConfig | None:
        return coerce_resume_checkpoint_value(value)

    @pydantic.field_validator("checkpoints", mode="before")
    @classmethod
    def _checkpoint_list_coerce(cls, value: Any) -> list[FrozenOpponentCheckpointConfig]:
        if value is None:
            return []
        assert isinstance(value, list), (
            f"frozen_opponent.checkpoints must be a list, got {type(value)}"
        )
        out: list[FrozenOpponentCheckpointConfig] = []
        for item in value:
            if isinstance(item, tuple):
                assert len(item) == 2, item
                num_players, checkpoint = item
                assert int(num_players) in (2, 4), num_players
                num_players = int(num_players)
            else:
                num_players = None
                checkpoint = item
            out.append(
                FrozenOpponentCheckpointConfig(
                    num_players=num_players,
                    checkpoint=checkpoint,
                )
            )
        return out

    @pydantic.model_validator(mode="after")
    def _validate_probability_requires_checkpoints(self) -> "FrozenOpponentConfig":
        if self.checkpoint is not None:
            object.__setattr__(
                self,
                "checkpoints",
                [
                    FrozenOpponentCheckpointConfig(
                        num_players=None,
                        checkpoint=self.checkpoint,
                    ),
                    *self.checkpoints,
                ],
            )
        if float(self.probability) > 0.0 and len(self.checkpoints) == 0:
            raise ValueError(
                "frozen_opponent.checkpoints must be non-empty when probability > 0"
            )
        if bool(self.no_selfplay) and float(self.probability) != 1.0:
            raise ValueError("frozen_opponent.no_selfplay requires probability == 1.0")
        if bool(self.no_selfplay) and not bool(self.learn_on_frozen):
            raise ValueError("frozen_opponent.no_selfplay requires learn_on_frozen")
        return self


class ImpalaTrainingConfig(_StrictConfig):
    # Env / action-space (concrete simulator is wired in gym).
    agents_max_cnt: int = 4

    # Runtime / directories
    experiment_name: str | None = None

    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    # Checkpoint I/O: local dir only (see batch_and_learn._main_torch_io_root).
    torch_io_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "main_torch_io": {"local": {"dirpath": "artifacts/runs"}},
        }
    )

    wandb_project: str = "orbit_wars"
    enable_wandb: bool = False
    enable_tensorboard: bool = False

    # Training Model Config
    model: dict[str, Any] = Field(default_factory=_default_model_stub)
    checkpoint_model_config_fallback_checkpoint: CheckpointConfig | None = None #"fallback/checkpoint_0.pt"

    # Core training
    total_steps: float = 3e7
    warmup_steps: int = 0
    lr_warmup_steps: int = 0
    target_entropy_warmup_steps: int = 0
    unroll_length: int = 19
    #: Rollout / learner tensor time extent (= ``unroll_length + 1``); set in ``_derive_and_validate``.
    rollout_time_steps: int = 0
    #: Flat policy row count for one actor buffer (= ``n_actor_envs * agents_max_cnt``).
    orbit_actor_policy_flat: int = 0
    discounting: dict[str, float] = Field(default_factory=lambda: _default_reward_head_values(0.999))
    value_target_bound_min: dict[str, float] = Field(
        default_factory=lambda: _default_reward_head_values(-10.0)
    )
    value_target_bound_max: dict[str, float] = Field(
        default_factory=lambda: _default_reward_head_values(10.0)
    )

    enable_popart: bool = False
    
    #: If True: PopArt still normalizes baseline loss, but mean/std are not updated (fixed from checkpoint).
    lock_popart: bool = False
    reduction: Literal["mean", "sum"] = "mean"

    enable_clip_grad: bool = True

    learner_forward_bf16: bool = True
    learner_loss_bf16: bool = True
    learner_backward_bf16: bool = True
    inference_use_bf16: bool = True

    checkpoint_freq: float = 2  # minutes

    # IMPALA / actors (local processes or threads only)
    num_actors: int = 0
    #: Training rollouts: one CPU process hosts this many ``act_rollout_func`` threads; infer queues stay
    #: ``num_actors`` wide and each thread uses global indices ``0 .. num_actors-1``.
    n_actors_per_process: int = Field(default=1, ge=1)

    num_rl_vis_actors: int = 0
    vis_n_actor_envs: int = 1
    rl_vis_dump_inputs: bool = False
    rl_vis_dump_inputs_path: str = str(IMPALA_PROJECT_ROOT / "replays/replay_rl_vis_inputs.pt")
    rl_vis_save_aoti_example_inputs: bool = False
    rl_vis_aoti_example_inputs_path: str = str(IMPALA_PROJECT_ROOT / "aoti_example_inputs.pt")
    rl_vis_episode_seed: int = -1
    rl_vis_model_is_sample: bool = True
    rl_vis_model_shuffle_identity_ids: bool = True
    #: CUDA device index for learner, batch_prepare, and learner GPU buffers.
    learner_cuda_device: int = Field(default=0, ge=0)
    #: CUDA device index for single-device inference paths and RL vis policy forward.
    inference_cuda_device: int = Field(default=1, ge=0)
    #: If False: training rollout actors do not allocate RL vis queue, clone frames, or enqueue tape JSON.
    enable_rl_actor_visualization: bool = False

    #: If True: learner may write a one-shot V-trace text dump and call ``sys.exit(0)`` when
    #: ``IMPALA_TERMINAL_VTRACE_DUMP_FILE`` is non-empty and the batch contains a terminal step (debug only).
    #: If False: no dump path is read and training is unaffected.
    enable_terminal_vtrace_dump: bool = False

    batch_size: int = 1
    n_actor_envs: int = 1

    #: If True: each sub-env may emit ``desync_done`` on a random step (uniform in ``1 .. 4 * unroll_length``)
    #: after each reset, forcing a reset without training on rollouts that contain any desync (see ``act.py``).
    enable_desync: bool = False

    #: If True: Orbit reward wrapper may terminate an episode early when smoothed estimated power is decisive.
    enable_orbit_estimated_power_early_stop: bool = False

    #: Orbit Wars (Kaggle ``orbit_wars``): 2 or 4 players.
    orbit_num_agents: int = 4
    orbit_actor_target_4p_sample_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Passed to Kaggle ``make(..., configuration=…)`` (e.g. ``episodeSteps``).
    orbit_configuration: dict[str, Any] = Field(default_factory=dict)
    
    #: If True: training actors, RL visualization actor, benchmark actor (``bench_act``), and env stack emit
    #: cumulative ``WallTreeProfiler`` tree lines. ``create_env`` forwards the profiler into wrappers / ``OrbitWarsEnv``.
    #: If False: no actor/env wall-tree accumulation or stdout timing breakdown (default).
    enable_actor_wall_tree_profiler: bool = False
    #: If True: learner emits cumulative ``WallTreeProfiler`` tree lines for batch/forward/backward/update sections.
    #: If False: no learner wall-tree accumulation or stdout timing breakdown (default).
    enable_learner_wall_tree_profiler: bool = False

    #: If True: ``OrbitWarsEnv`` runs Python Kaggle stepping plus C++ stub parity (three live sim surfaces) and all
    #: cross-env validation asserts. If False: C++-only rollout path (``cpp_env_obs_full``); reset from dataset-derived
    #: reference random state, no Kaggle env stepping, validations off. Applies to training actors, buffer spec env, RL
    #: visualization, and benchmarks — not gated on ``enable_rl_actor_visualization``.
    enable_envs_validation: bool = False
    #: Orbit env dict IO (paths + tensor shapes); loaded from ``dict_io_contract_orbit.yaml`` in
    #: :func:`default_training_config_dict` unless overridden.
    dict_io_contracts: dict[str, Any] = Field(default_factory=dict)

    # Buffers / workers
    inference_batch_size: int = 16

    num_buffers: int = 128

    num_stats_buffers: int = 100

    n_batch_prepare_processes: int = 1
    prepare_batches: int = 2
    disable_batch_prepare_drop_on_full_queue: bool = False
    num_inference_workers_train_cuda0: int = Field(default=0, ge=0)
    num_inference_workers_train_cuda1: int = Field(default=1, ge=0)
    num_benchmark_inference_workers_cuda0: int = Field(default=0, ge=0)
    num_benchmark_inference_workers_cuda1: int = Field(default=0, ge=0)

    enable_sampling: bool = True

    num_inference_buffers_train: int = 128

    #: Push learner weights into the shared actor every N learner iterations (1 = every iteration).
    learner_actor_sync_every_iterations: int = Field(default=10, ge=1)

    # Sharing
    sharing_strategy: Literal["file_descriptor", "file_system"] = "file_descriptor"
    multiprocessing_start_method: Literal["fork", "spawn"] = "spawn"

    # Optimizer & scheduler
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    enable_separate_value_model: bool = False
    value_optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    enable_lr_scheduler: bool = True

    # Loss weights / RL knobs
    baseline_cost: dict[str, float] = Field(default_factory=lambda: _default_reward_head_values(10.0))
    #: Weight for terminal-segment baseline loss (MC return on prefix up to first ``done`` only).
    baseline_terminal_loss_cost: dict[str, float] = Field(
        default_factory=lambda: _default_reward_head_values(0.0)
    )
    #: If True: baseline fits squared error ``(V - target)^2``; if False: Huber ``smooth_l1``.
    baseline_loss_use_mse: dict[str, bool] = Field(
        default_factory=lambda: _default_reward_head_bool_values(False)
    )
    baseline_smooth_l1_beta: dict[str, float] = Field(
        default_factory=lambda: _default_reward_head_values(1.0)
    )
    upgo_cost: dict[str, float] = Field(default_factory=lambda: _default_reward_head_values(1.0))
    upgo_original_cost: dict[str, float] = Field(
        default_factory=lambda: _default_reward_head_values(0.0)
    )
    vtrace_cost: dict[str, float] = Field(default_factory=lambda: _default_reward_head_values(1.0))
    classic_entropy: bool = False
    ce_entropy: bool = False
    entropy_cost: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(1.0)
    )
    policy_logits_l2_cost: float = 0.0
    policy_centered_logits_l2_cost: float = 0.0
    temperature_compensation_kl_cost: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.0)
    )
    logit_limit: float = 0.0

    #: V-trace / off-policy UPGO trace decay.
    lmb: dict[str, float] = Field(default_factory=lambda: _default_reward_head_values(0.8))
    # Entropy schedule: if enable_entropy_decay, ramp 0 → target_mean_entropy over target_entropy_warmup_steps, then
    # linear decay to 0 by total_steps. If enable_entropy_decay is False, uses final target from step 0
    # (target_entropy_warmup_steps is ignored). Applies to all policy heads in ``POLICY_ACTION_KEYS``.
    target_mean_entropy: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.0)
    )
    target_mean_entropy_max: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(1.0)
    )
    enable_entropy_decay: bool = True
    #: ``scheduled``: ``compute_target_entropy_combined`` (warmup/decay). ``ema_tracking``: EMA of learner
    #: ``mean_entropy`` — mean **normalized** entropy ``H / log k`` over planets with ≥2 legal actions (same scale as
    #: shortfall ``H_norm``; typically in ``[0, 1]``). Effective policy target is ``ema * mean_entropy_ema_multiplier``.
    #: EMA blend: ``1 - (1 - mean_entropy_ema_alpha_per_step) ** steps_since_last_learner_stats``.
    target_mean_entropy_mode: Literal["scheduled", "ema_tracking"] = "ema_tracking"
    #: Per-env-step coefficient for EMA of mean normalized entropy; compound over learner step span.
    mean_entropy_ema_alpha_per_step: float = 2e-6
    #: ``shared_target_entropy`` reference becomes ``ema * multiplier`` (``ema_tracking`` only).
    mean_entropy_ema_multiplier: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.9)
    )

    target_min_entropy: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.5)
    )
    use_new_controller: bool = False
    new_controller_target_min_entropy: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.5)
    )
    new_controller_target_min_entropy_min_value: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.0)
    )
    new_controller_temperature_threshold: float = 1000.0
    new_controller_target_up_log_delta_per_step: float = 1e-6
    new_controller_target_down_log_delta_per_step: float = 1e-6
    new_controller_threshold_up_log_delta_per_step: float = 1e-6
    new_controller_threshold_down_log_delta_per_step: float = 1e-6
    entropy_floor_max_temperature: float = 64.0
    entropy_floor_num_iters: int = 10
    shortfall_entropy_increase_delta_per_step: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(2e-5)
    )
    shortfall_entropy_decrease_delta_per_step: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(2e-5)
    )
    shortfall_entropy_min: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(0.0)
    )
    shortfall_entropy_max: dict[str, float] = Field(
        default_factory=lambda: _default_entropy_head_values(1.0)
    )

    # Derived (can be None in YAML -> computed)
    initial_lr_lambda: float = 1.0
    final_lr_lambda: float = 0
    lr_lambda_change_per_step: float | None = Field(
        default=None,
        json_schema_extra={_JSON_EXTRA_DERIVE_AFTER_MERGE: True},
    )

    # Optional EMA normalization for reward heads.
    enable_reward_ema_norm: bool = False
    #: If True: reward EMA still normalizes rewards, but EMA stats are not updated (fixed from checkpoint).
    lock_reward_ema: bool = False
    reward_ema_alpha: float = 2e-5
    #: EMA base per env step for PopArt running mean/m²; effective batch alpha is ``1 - (1 - popart_alpha_per_step) ** steps_in_batch``.
    popart_alpha_per_step: float = 2e-5

    reward_ema_eps: float = 1e-6

    # Teacher / KL
    teacher: TeacherConfig | None = None
    frozen_opponent: FrozenOpponentConfig = Field(default_factory=FrozenOpponentConfig)

    # Resume
    resume_checkpoint: CheckpointConfig | None = None
    use_model_config_from_checkpoint: bool = False
    load_as_much_as_possible: bool = False
    start_from_scratch: bool = True

    # ------------------------
    # Global validation & derived defaults
    # ------------------------
    @pydantic.field_validator("resume_checkpoint", mode="before")
    @classmethod
    def _resume_checkpoint_coerce(cls, value: Any) -> CheckpointConfig | None:
        return coerce_resume_checkpoint_value(value)

    @pydantic.field_validator("checkpoint_model_config_fallback_checkpoint", mode="before")
    @classmethod
    def _model_config_fallback_checkpoint_coerce(cls, value: Any) -> CheckpointConfig | None:
        return coerce_resume_checkpoint_value(value)

    @pydantic.field_validator(
        "entropy_cost",
        "target_mean_entropy",
        "target_mean_entropy_max",
        "mean_entropy_ema_multiplier",
        "target_min_entropy",
        "new_controller_target_min_entropy",
        "new_controller_target_min_entropy_min_value",
        "shortfall_entropy_increase_delta_per_step",
        "shortfall_entropy_decrease_delta_per_step",
        "shortfall_entropy_min",
        "shortfall_entropy_max",
        "temperature_compensation_kl_cost",
    )
    @classmethod
    def _entropy_head_values_validate(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value.keys()) != set(ENTROPY_HEAD_KEYS):
            raise ValueError(
                f"entropy head values must contain exactly {ENTROPY_HEAD_KEYS}, got {tuple(sorted(value.keys()))}"
            )
        return {
            head: float(value[head])
            for head in ENTROPY_HEAD_KEYS
        }

    @pydantic.model_validator(mode="after")
    def _derive_and_validate(self) -> "ImpalaTrainingConfig":
        if bool(self.enable_wandb) and bool(self.enable_tensorboard):
            raise ValueError("enable_wandb and enable_tensorboard are mutually exclusive")

        expected = int(self.n_actor_envs)
        if self.batch_size % expected != 0:
            raise ValueError(
                f"batch_size must be divisible by n_actor_envs: {self.batch_size} % ({expected}) != 0"
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        _p_slots = int(self.agents_max_cnt)
        object.__setattr__(self, "rollout_time_steps", int(self.unroll_length) + 1)
        object.__setattr__(
            self, "orbit_actor_policy_flat", int(self.n_actor_envs) * _p_slots
        )

        if int(self.vis_n_actor_envs) < 1:
            raise ValueError("vis_n_actor_envs must be >= 1")
        _rl_vis_episode_seed = int(self.rl_vis_episode_seed)
        if _rl_vis_episode_seed < -1:
            raise ValueError("rl_vis_episode_seed must be >= -1")
        if bool(self.rl_vis_dump_inputs):
            if str(self.rl_vis_dump_inputs_path) == "":
                raise ValueError("rl_vis_dump_inputs_path must be non-empty when rl_vis_dump_inputs is true")
            object.__setattr__(self, "vis_n_actor_envs", 1)
        if bool(self.rl_vis_save_aoti_example_inputs):
            if str(self.rl_vis_aoti_example_inputs_path) == "":
                raise ValueError(
                    "rl_vis_aoti_example_inputs_path must be non-empty when "
                    "rl_vis_save_aoti_example_inputs is true"
                )
            object.__setattr__(self, "vis_n_actor_envs", 1)
            object.__setattr__(self, "orbit_num_agents", 4)
        if _rl_vis_episode_seed >= 0:
            if bool(self.rl_vis_save_aoti_example_inputs):
                raise ValueError("rl_vis_episode_seed is incompatible with rl_vis_save_aoti_example_inputs")
            object.__setattr__(self, "vis_n_actor_envs", 1)

        _nvis = int(self.num_rl_vis_actors)
        if _nvis > 0:
            if _nvis > 1:
                raise ValueError("num_rl_vis_actors must be <= 1")
            object.__setattr__(self, "num_actors", 0)
            vis_cuda = int(self.inference_cuda_device)
            object.__setattr__(self, "learner_cuda_device", vis_cuda)
            object.__setattr__(self, "inference_cuda_device", vis_cuda)

        _na = int(self.num_actors)
        _napp = int(self.n_actors_per_process)
        if _na > 0 and _na % _napp != 0:
            raise ValueError(
                f"num_actors ({_na}) must be divisible by n_actors_per_process ({_napp})"
            )

        _ona = int(self.orbit_num_agents)
        if _ona not in (2, 4):
            raise ValueError("orbit_num_agents must be 2 or 4")
        sample_4p_ratio = float(self.orbit_actor_target_4p_sample_ratio)
        if self.teacher is not None:
            teacher_has_shared_checkpoint = any(
                checkpoint.num_players is None
                for checkpoint in self.teacher.checkpoints
            )
            teacher_num_players = {
                int(checkpoint.num_players)
                for checkpoint in self.teacher.checkpoints
                if checkpoint.num_players is not None
            }
            if (
                sample_4p_ratio < 1.0
                and not teacher_has_shared_checkpoint
                and 2 not in teacher_num_players
            ):
                raise ValueError(
                    "teacher.checkpoints must include a 2p checkpoint "
                    "when orbit_actor_target_4p_sample_ratio < 1"
                )
            if (
                sample_4p_ratio > 0.0
                and not teacher_has_shared_checkpoint
                and 4 not in teacher_num_players
            ):
                raise ValueError(
                    "teacher.checkpoints must include a 4p checkpoint "
                    "when orbit_actor_target_4p_sample_ratio > 0"
                )
        if float(self.frozen_opponent.probability) > 0.0:
            frozen_has_shared_checkpoint = any(
                checkpoint.num_players is None
                for checkpoint in self.frozen_opponent.checkpoints
            )
            frozen_num_players = {
                int(checkpoint.num_players)
                for checkpoint in self.frozen_opponent.checkpoints
                if checkpoint.num_players is not None
            }
            if (
                sample_4p_ratio < 1.0
                and not frozen_has_shared_checkpoint
                and 2 not in frozen_num_players
            ):
                raise ValueError(
                    "frozen_opponent.checkpoints must include at least one 2p checkpoint "
                    "when orbit_actor_target_4p_sample_ratio < 1"
                )
            if (
                sample_4p_ratio > 0.0
                and not frozen_has_shared_checkpoint
                and 4 not in frozen_num_players
            ):
                raise ValueError(
                    "frozen_opponent.checkpoints must include at least one 4p checkpoint "
                    "when orbit_actor_target_4p_sample_ratio > 0"
                )

        if float(self.total_steps) <= 0.0:
            raise ValueError("total_steps must be > 0")
        if float(self.policy_logits_l2_cost) < 0.0:
            raise ValueError("policy_logits_l2_cost must be >= 0")
        if float(self.policy_centered_logits_l2_cost) < 0.0:
            raise ValueError("policy_centered_logits_l2_cost must be >= 0")
        for head in ENTROPY_HEAD_KEYS:
            if float(self.temperature_compensation_kl_cost[head]) < 0.0:
                raise ValueError(
                    f"temperature_compensation_kl_cost[{head!r}] must be >= 0"
                )
        if float(self.logit_limit) < 0.0:
            raise ValueError("logit_limit must be >= 0")
        if int(self.lr_warmup_steps) < 0:
            raise ValueError("lr_warmup_steps must be >= 0")
        if float(self.lr_warmup_steps) > float(self.total_steps):
            raise ValueError("lr_warmup_steps must be <= total_steps")
        if bool(self.classic_entropy) and bool(self.ce_entropy):
            raise ValueError("classic_entropy and ce_entropy are mutually exclusive")
        if not (bool(self.classic_entropy) or bool(self.ce_entropy)):
            if int(self.target_entropy_warmup_steps) < 0:
                raise ValueError("target_entropy_warmup_steps must be >= 0")
            if float(self.target_entropy_warmup_steps) > float(self.total_steps):
                raise ValueError("target_entropy_warmup_steps must be <= total_steps")

            for head in ENTROPY_HEAD_KEYS:
                _target = float(self.target_mean_entropy[head])
                if not (0.0 <= _target <= 1.0):
                    raise ValueError(
                        f"target_mean_entropy[{head!r}] must be in [0, 1]"
                    )
                _target_max = float(self.target_mean_entropy_max[head])
                if not (0.0 <= _target_max <= 1.0):
                    raise ValueError(
                        f"target_mean_entropy_max[{head!r}] must be in [0, 1]"
                    )
                _sf_lo = float(self.shortfall_entropy_min[head])
                _sf_hi = float(self.shortfall_entropy_max[head])
                if _sf_lo > _sf_hi:
                    raise ValueError(
                        f"shortfall_entropy_min[{head!r}] must be <= shortfall_entropy_max[{head!r}]"
                    )
                _sf0 = float(self.target_min_entropy[head])
                if not (_sf_lo <= _sf0 <= _sf_hi):
                    raise ValueError(
                        f"target_min_entropy[{head!r}] must lie in "
                        f"[shortfall_entropy_min[{head!r}], shortfall_entropy_max[{head!r}]]"
                    )
                _new_controller_target = float(self.new_controller_target_min_entropy[head])
                if not (0.0 <= _new_controller_target <= 1.0):
                    raise ValueError(
                        f"new_controller_target_min_entropy[{head!r}] must be in [0, 1]"
                    )
                _new_controller_target_min = float(
                    self.new_controller_target_min_entropy_min_value[head]
                )
                if not (0.0 <= _new_controller_target_min <= _new_controller_target):
                    raise ValueError(
                        f"new_controller_target_min_entropy_min_value[{head!r}] must lie in "
                        f"[0, new_controller_target_min_entropy[{head!r}]]"
                    )
                if float(self.shortfall_entropy_increase_delta_per_step[head]) <= 0.0:
                    raise ValueError(
                        f"shortfall_entropy_increase_delta_per_step[{head!r}] must be > 0"
                    )
                if float(self.shortfall_entropy_decrease_delta_per_step[head]) <= 0.0:
                    raise ValueError(
                        f"shortfall_entropy_decrease_delta_per_step[{head!r}] must be > 0"
                    )
            if float(self.entropy_floor_max_temperature) < 1.0:
                raise ValueError("entropy_floor_max_temperature must be >= 1")
            if int(self.entropy_floor_num_iters) < 0:
                raise ValueError("entropy_floor_num_iters must be >= 0")
            if float(self.new_controller_temperature_threshold) <= 1.0:
                raise ValueError("new_controller_temperature_threshold must be > 1")
            if float(self.new_controller_target_up_log_delta_per_step) <= 0.0:
                raise ValueError("new_controller_target_up_log_delta_per_step must be > 0")
            if float(self.new_controller_target_down_log_delta_per_step) <= 0.0:
                raise ValueError("new_controller_target_down_log_delta_per_step must be > 0")
            if float(self.new_controller_threshold_up_log_delta_per_step) <= 0.0:
                raise ValueError("new_controller_threshold_up_log_delta_per_step must be > 0")
            if float(self.new_controller_threshold_down_log_delta_per_step) <= 0.0:
                raise ValueError("new_controller_threshold_down_log_delta_per_step must be > 0")

            _mode = str(self.target_mean_entropy_mode)
            if _mode not in ("scheduled", "ema_tracking"):
                raise ValueError(
                    "target_mean_entropy_mode must be 'scheduled' or 'ema_tracking'"
                )
            _ema_a = float(self.mean_entropy_ema_alpha_per_step)
            if not (0.0 < _ema_a <= 1.0):
                raise ValueError("mean_entropy_ema_alpha_per_step must be in (0, 1]")
            if _mode == "ema_tracking":
                for head in ENTROPY_HEAD_KEYS:
                    if float(self.mean_entropy_ema_multiplier[head]) <= 0.0:
                        raise ValueError(
                            f"mean_entropy_ema_multiplier[{head!r}] must be > 0"
                        )

        # Validate inference configuration to prevent deadlock (train actors only).
        train_inference_actors = self.num_actors

        if (
            train_inference_actors > 0
            and self.inference_batch_size > train_inference_actors
            and (
                self.num_inference_workers_train_cuda0
                + self.num_inference_workers_train_cuda1
            ) > 0
        ):
            raise ValueError(
                f"CONFIGURATION ERROR (train): inference_batch_size ({self.inference_batch_size}) is greater than "
                f"the number of train actors that use inference ({train_inference_actors}). This will cause a DEADLOCK "
                f"because the inference worker waits for exactly {self.inference_batch_size} requests before "
                f"processing, but only {train_inference_actors} actors can make concurrent inference requests.\n"
                f"Train inference actors: "
                f"selfplay={self.num_actors}\n"
                f"Solutions:\n"
                f"  1. Reduce inference_batch_size to <= {train_inference_actors}\n"
                f"  2. Increase train selfplay actors to >= {self.inference_batch_size}\n"
                f"  3. Set num_inference_workers_train_cuda0 and num_inference_workers_train_cuda1 to 0 if you don't need batched inference"
            )

        # Derived schedules (if not provided)
        if self.lr_lambda_change_per_step is None:
            decay_steps = self.total_steps - self.lr_warmup_steps
            if decay_steps <= 0:
                self.lr_lambda_change_per_step = 0.0
            else:
                self.lr_lambda_change_per_step = (self.final_lr_lambda - self.initial_lr_lambda) / decay_steps

        return self

    # ------------------------
    # Helpers
    # ------------------------
    @classmethod
    def null_derive_after_merge_fields_in_dict(cls, d: dict[str, Any]) -> None:
        """Set ``x_derive_after_merge`` fields to ``None`` so ``model_validate(merge)`` re-runs derivation."""
        for name, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra
            if isinstance(extra, dict) and extra.get(_JSON_EXTRA_DERIVE_AFTER_MERGE):
                d[name] = None

    def with_experiment_subdirs(self) -> "ImpalaTrainingConfig":
        """Append experiment_name to the checkpoint directory."""
        if self.experiment_name is None:
            user = os.getenv("GITHUB_USER", os.getenv("USER", "unknown_user"))
            prefix = "full"
            ts = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.experiment_name = f"{user}/{prefix}/{ts}"

        main = self.torch_io_config["main_torch_io"]
        local = main.get("local")
        assert local is not None and isinstance(local, dict), (
            "torch_io_config.main_torch_io must use local storage with a dict 'local' block"
        )
        local["dirpath"] = os.path.join(local["dirpath"], self.experiment_name)

        return self

    def to_namespace(self) -> SimpleNamespace:
        """If train() still expects a flags namespace."""
        as_dict = self.model_dump()
        return SimpleNamespace(**as_dict)


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``; override wins on leaves."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def default_training_config_dict() -> dict[str, Any]:
    """Full dict of training keys from ``ImpalaTrainingConfig`` defaults."""
    out = ImpalaTrainingConfig().model_dump(mode="python")
    ImpalaTrainingConfig.null_derive_after_merge_fields_in_dict(out)
    _orbit_io_path = Path(__file__).resolve().parent / "dict_io_contract_orbit.yaml"
    out["dict_io_contracts"] = load_yaml(str(_orbit_io_path))
    assert isinstance(out["dict_io_contracts"], dict)
    return out


