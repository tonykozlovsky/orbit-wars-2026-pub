"""
x1 experiment: overrides on top of :func:`configs.base.default_training_config_dict`.

Paths under ``python/artifacts/`` (created at runtime as needed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# ``runpy.run_path`` loads this file as a top-level script (no package); relative imports fail.
_IMPALA_PY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_IMPALA_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPALA_PY_ROOT))

from src.configs.base import (
    IMPALA_PROJECT_ROOT,
    ImpalaTrainingConfig,
    deep_merge_dicts,
    default_training_config_dict,
)
from src.gym.obs_wrapper import ORBIT_PLANET_ARRIVAL_HORIZON

_CONF_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _CONF_DIR.parent
_ARTIFACT_ROOT = _PYTHON_ROOT / "artifacts"

_LOCAL_MAIN_RUNS = _ARTIFACT_ROOT / "runs"


def _torch_io_local(dirpath: Path) -> dict:
    return {"local": {"dirpath": str(dirpath)}}


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


# Only fields that differ from ``default_training_config_dict()`` (see base.py).
IMPALA_X1_RL_OVERRIDES: dict = {
    "enable_wandb": False,
    "enable_tensorboard": True,
    "unroll_length": 20,

    "enable_desync": True,
    "enable_orbit_estimated_power_early_stop": False,

    "model": {
        "orbit_impala": {
            "num_policy_actions": 5,
            "arrival_temporal_horizon": ORBIT_PLANET_ARRIVAL_HORIZON,
            "policy_head_fp32": True,
            "value_head_fp32": True,
            "use_value_opponent_model_embedding": True,
            "zeroed_obs_feature_inputs": """
continuous.edge.edge_min_neutralize_ships	131.375000	129	141	145	124	128
embedding.planet.planet_player_flip_time_by_player@enemy	133.000000	142	138	138	146	125
embedding.planet.planet_player_last_decisive_battle_step@enemy	134.500000	147	144	141	148	124
embedding.edge.edge_min_takeover_bucket	134.875000	128	125	123	135	142
embedding.edge.edge_min_neutralize_ships	135.000000	133	123	121	127	144
continuous.arrival.neutralization_cost@self	137.875000	154	153	159	153	121
embedding.planet.planet_production	137.875000	138	146	143	144	133
embedding.edge.edge_min_time_stable_takeover_ships	138.375000	123	143	144	133	141
embedding.edge.edge_min_neutralize_bucket	139.125000	136	128	127	126	149
embedding.planet.planet_player_flip_time_by_player@self	141.125000	146	150	158	151	131
continuous.arrival.hold_cost@enemy	141.250000	151	142	139	150	137
embedding.planet.planet_player_stable_flip_time_by_player@enemy	143.750000	143	149	157	149	138
embedding.edge.edge_min_time_neutralize_bucket	144.625000	145	135	134	143	150
embedding.edge.edge_min_time_neutralize_bucket_hit_steps	146.625000	139	145	142	139	152
embedding.edge.edge_min_neutralize_bucket_hit_steps	147.250000	150	140	137	147	151
embedding.edge.edge_min_time_neutralize_ships	150.250000	148	161	161	140	148
embedding.planet.planet_comet_time_before_despawn	151.250000	152	148	146	152	153
continuous.arrival.neutralization_cost@enemy	152.750000	153	152	147	154	154
continuous.edge.edge_snipe_score_at_min_takeover_time	154.250000	155	154	150	155	155
continuous.planet.planet_player_owner_churn@enemy	155.250000	156	155	151	156	156
continuous.planet.planet_player_owner_churn@self	156.250000	157	156	152	157	157
embedding.arrival.stable_takeover_cost@self	157.250000	158	157	153	158	158
embedding.arrival.takeover_cost@self	158.250000	159	158	154	159	159
embedding.planet.planet_player_owner_churn@enemy	159.250000	160	159	155	160	160
embedding.planet.planet_player_owner_churn@self	160.250000	161	160	156	161	161
continuous.edge.edge_min_takeover_bucket
continuous.edge.edge_min_time_takeover_bucket
embedding.edge.edge_min_time_takeover_bucket
continuous.edge.edge_min_stable_takeover_bucket
embedding.edge.edge_min_stable_takeover_bucket
continuous.edge.edge_min_time_stable_takeover_bucket
embedding.edge.edge_min_time_stable_takeover_bucket
continuous.edge.edge_min_time_stable_takeover_ships
continuous.edge.edge_min_time_stable_takeover_bucket_available
continuous.edge.edge_min_time_stable_takeover_bucket_hit_steps
embedding.edge.edge_min_time_stable_takeover_bucket_hit_steps
continuous.edge.edge_min_neutralize_bucket
continuous.edge.edge_min_neutralize_bucket_available
continuous.edge.edge_min_neutralize_bucket_hit_steps
continuous.edge.edge_min_time_neutralize_bucket
continuous.edge.edge_min_time_neutralize_ships
continuous.edge.edge_min_time_neutralize_bucket_available
continuous.edge.edge_min_time_neutralize_bucket_hit_steps


"""

# embedding.edge.edge_min_time_takeover_ships	 NOW ITS half send_ships
# embedding.edge.edge_min_time_takeover_bucket_hit_steps NOW ITS half hit_steps

# continuous.edge.edge_min_time_takeover_ships +- NOW ITS half send_ships
# continuous.edge.edge_min_time_takeover_bucket_hit_steps + NOW ITS half hit_steps

# continuous.edge.edge_min_time_takeover_bucket_available + NOW ITS half available

# embedding.edge.edge_min_takeover_bucket_hit_steps	129.875000	117	124	124	130	136
# embedding.edge.edge_min_takeover_ships	136.875000	141	129	128	137	140
# embedding.edge.edge_min_stable_takeover_bucket_hit_steps	137.375000	126	126	125	134	147
# embedding.edge.edge_min_stable_takeover_ships	145.125000	134	151	160	136	145


# continuous.edge.edge_min_takeover_ships
# continuous.edge.edge_min_takeover_bucket_available
# continuous.edge.edge_min_takeover_bucket_hit_steps

# continuous.edge.edge_min_stable_takeover_ships
# continuous.edge.edge_min_stable_takeover_bucket_available
# continuous.edge.edge_min_stable_takeover_bucket_hit_steps

# continuous.edge.edge_source_stable_hold_margin_after_min_takeover
# continuous.edge.edge_source_stable_hold_margin_after_min_stable_takeover
# continuous.edge.edge_min_stable_takeover_bucket_roi
# continuous.edge.edge_overkill_with_min_stable_bucket
# continuous.edge.edge_stable_capture_vs_current_owner_value

,
        },
    },

    "orbit_actor_target_4p_sample_ratio": 0.35,


    "resume_checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-22_14-33-16_2P/runs/checkpoint_114667526.pt"),
    "use_model_config_from_checkpoint": False,
    "load_as_much_as_possible": False,

    "start_from_scratch": False,



    "num_actors": 20,
    "n_actors_per_process": 1,
    "multiprocessing_start_method": "fork",
    
    "batch_size": 4,
    "n_actor_envs": 1,

    "inference_batch_size": 4,
     
    "learner_forward_bf16": True,
    "learner_loss_bf16": False,
    "learner_backward_bf16": False,

    "inference_use_bf16": True,
    
    "orbit_num_agents": 4,
    "rl_vis_episode_seed": -1,
    "rl_vis_model_is_sample": False,
    "rl_vis_model_shuffle_identity_ids": False,

    "num_rl_vis_actors": 0,
    "vis_n_actor_envs": 1,

    "rl_vis_dump_inputs": False,
    "rl_vis_dump_inputs_path": str(IMPALA_PROJECT_ROOT / "replays/replay_rl_vis_inputs_40_send_all.pt"),

    "rl_vis_save_aoti_example_inputs": False,
    "rl_vis_aoti_example_inputs_path": str(IMPALA_PROJECT_ROOT / "aoti_example_inputs_40_send_all_half.pt"),

    "enable_envs_validation": False,
    "enable_actor_wall_tree_profiler": False,
    "enable_learner_wall_tree_profiler": True,

    "optimizer_config": {
        "optimizer_kwargs": {
            "lr": 2e-6,

            "weight_decay": 1e-4,
        },
    },
    "enable_separate_value_model": False,
    "value_optimizer_config": {
        "optimizer_kwargs": {
            "lr": 2e-6,

            "weight_decay": 1e-4,
        },
    },

    "total_steps": 300_000_000,
    "warmup_steps": 0,

    "mean_entropy_ema_alpha_per_step": 1e-30, #4e-6,

    "target_mean_entropy": {
        "spawn_fleet": 0,
    },
    "target_mean_entropy_max": {
        "spawn_fleet": 0,
    },
    "mean_entropy_ema_multiplier": {
        "spawn_fleet": 0.8,
    },
    "target_min_entropy": {
        "spawn_fleet": 0.0,
    },


    "use_new_controller": True,
    "new_controller_target_min_entropy": {
        "spawn_fleet": 0,
    },
    "new_controller_target_min_entropy_min_value": {
        "spawn_fleet": 0,
    },


    "new_controller_temperature_threshold": 1000.0,

    "new_controller_target_up_log_delta_per_step": 2e-6,
    "new_controller_target_down_log_delta_per_step": 4e-6,

    "new_controller_threshold_up_log_delta_per_step": 2e-6,
    "new_controller_threshold_down_log_delta_per_step": 4e-6,


    "entropy_floor_max_temperature": 1,
    "entropy_floor_num_iters": 0,

    "shortfall_entropy_increase_delta_per_step": {
        "spawn_fleet": 1e-3,
    },
    "shortfall_entropy_decrease_delta_per_step": {
        "spawn_fleet": 1e-6,
    },

    "enable_entropy_decay": False,

    "baseline_cost": {
        "baseline": 5.,
        "production_delta": 0,
    },
    "baseline_terminal_loss_cost": {
        "baseline": 0.,
        "production_delta": 0.,
    },
    "baseline_loss_use_mse": {
        "baseline": False,
        "production_delta": False,
    },
    "baseline_smooth_l1_beta": {
        "baseline": 1,
        "production_delta": 1,
    },
    "upgo_cost": {
        "baseline": 0, #1,
        "production_delta": 0.00,
    },
    "upgo_original_cost": {
        "baseline": 0, #1.,
        "production_delta": 0.,
    },
    "vtrace_cost": {
        "baseline": 1,
        "production_delta": 0,
    },

    "discounting": {
        "baseline": 1.,
        "production_delta": 0.99,
    },

    "lmb": {
        "baseline": 1,
        "production_delta": 1,
    },

    "value_target_bound_min": {
        "baseline": -1.1,
        "production_delta": -1000000,
    },
    "value_target_bound_max": {
        "baseline": 1.1,
        "production_delta": 1000000,
    },

    "classic_entropy": False,
    "ce_entropy": False,
    "entropy_cost": {
        #"spawn_fleet": 0.001,
        "spawn_fleet": 0.00008,
    },

    "policy_logits_l2_cost": 0,
    "policy_centered_logits_l2_cost": 0,


    "temperature_compensation_kl_cost": {
        "spawn_fleet": 0,
    },


    "logit_limit": 10000.0,

    "final_lr_lambda": 1.0,

    "lr_warmup_steps": 0,
    "target_entropy_warmup_steps": 0,


    "enable_reward_ema_norm": False,
    "enable_popart": False,

    "reward_ema_alpha": 1e-5,
    "popart_alpha_per_step": 1e-5,

    "lock_popart": True,
    "lock_reward_ema": True,





    # Optional fixed teachers for distillation, keyed by game size.
    "teacher": {
        #"checkpoint": "outputs/train/2026-05-28_17-06-51/runs/checkpoint_15589840.pt",
        #"checkpoint": "outputs/train/2026-05-30_01-14-35/runs/checkpoint_26922000.pt",
        #"checkpoint": "outputs/supervised/2026-05-31_17-37-37/samples_000006750000.pt",
        #"checkpoint": "outputs/train/2026-06-02_03-20-22/runs/checkpoint_13867500.pt",

        #"checkpoint": "outputs/train/2026-06-02_05-19-10/runs/checkpoint_37062420.pt",
        
        #"checkpoint": "outputs/train/2026-06-02_19-02-39/runs/checkpoint_40274160.pt",
        
        #"checkpoint": "outputs/train/2026-06-02_19-02-39/runs/checkpoint_40274160.pt",

        #"checkpoint":  "outputs/train/2026-06-03_05-47-14_benchmark/runs/checkpoint_52454370.pt",

        #"checkpoint":  "outputs/train/2026-06-04_21-24-03/runs/checkpoint_55259370.pt",

        #"checkpoint":  "outputs/supervised/2026-06-06_19-50-28/samples_000013050000.pt",
        #outputs/train/2026-06-07_05-58-40/runs/checkpoint_1261950.pt
        
        #"checkpoint":  "outputs/supervised/2026-06-08_06-24-49/samples_000008900000.pt",
        #"checkpoint": "outputs/train/2026-06-09_03-09-34/runs/checkpoint_6594450.pt",
        #outputs/train/2026-06-09_07-39-48/runs/checkpoint_20515950.pt
        
        # "checkpoint":    "outputs/train/2026-06-09_17-45-14/runs/checkpoint_22257846.pt",
        # "checkpoint":    "outputs/train/2026-06-09_20-16-44/runs/checkpoint_23002746.pt",
         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-10_03-28-36/runs/checkpoint_23246262.pt"),
         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-10_21-05-56/runs/checkpoint_36029994.pt"),
         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_02-36-25/runs/checkpoint_44859126.pt"), #before distilation (3 actions)

         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_07-25-12/runs/checkpoint_45358482.pt"), # distilled to 5 actions
        
         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_08-54-37/runs/checkpoint_53295918.pt"),  low entropy

         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_17-10-01_2P/runs/checkpoint_55502382.pt"), # hi entropy

         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-12_07-35-41_2P/runs/checkpoint_65331318.pt"), # kek

         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-12_17-34-08_2P/runs/checkpoint_67285842.pt"), # kek

         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-12_18-28-07_2P/runs/checkpoint_69465630.pt"), # kek
         
         #"checkpoint": str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-12_21-08-18_2P/runs/checkpoint_71075238.pt"), # kek

         "checkpoint" : str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-16_01-45-38_2P/runs/checkpoint_7618884.pt"), # big
         
        # "checkpoints": [
        #     (2, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_08-54-37_2P_TOP/runs/checkpoint_53333670.pt")),
        #     (4, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_08-54-37/runs/checkpoint_53295918.pt")),
        # ],

        "kl_cost_initial": 0.2,
        "kl_cost_decay_steps": 0,
        "moving_steps": 500000,

        "baseline_cost": 2.5,

        "zero_missing_policy_actions": True,
    },

    "frozen_opponent": {
        "probability": 0.5,
        "all_frozen": False,
        "no_selfplay": False,
        "learn_on_frozen": False,
        "checkpoints": [
            #"outputs/supervised/2026-05-31_17-37-37/samples_000006750000.pt",
            #"outputs/train/2026-06-01_23-09-04/runs/checkpoint_5409330.pt",
            #"outputs/train/2026-06-02_19-02-39/runs/checkpoint_40274160.pt",
            #"outputs/train/2026-06-03_05-47-14_benchmark/runs/checkpoint_52454370.pt",
            #"outputs/supervised/2026-06-06_19-50-28/samples_000013050000.pt",
            #"outputs/train/2026-06-07_08-29-15/runs/checkpoint_2065200.pt",
            #"outputs/train/2026-06-07_08-56-46/runs/checkpoint_12544950.pt",
            #"outputs/supervised/2026-06-08_06-24-49/samples_000008900000.pt",
            #"outputs/train/2026-06-09_01-09-42/runs/checkpoint_3036000.pt",
            #"outputs/train/2026-06-09_17-06-39/runs/checkpoint_21267558.pt",
            #"outputs/train/2026-06-09_17-45-14/runs/checkpoint_22257846.pt",
            #"outputs/train/2026-06-09_20-16-44/runs/checkpoint_23002746.pt",
            (2, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-10_03-28-36/runs/checkpoint_23246262.pt")), # > 0.85        1
            (2, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-10_07-43-44/runs/checkpoint_29085966.pt")), # > 0.55        2 
            (2, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_07-25-12/runs/checkpoint_45358482.pt")), # > 0.2         3
            (2, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_08-54-37_2P_TOP/runs/checkpoint_53333670.pt")), # = 0.0  4
            (4, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-10_15-27-31/runs/checkpoint_20946510.pt")), # = 1           5 
            (4, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-10_21-16-56/runs/checkpoint_31094154.pt")), # = 1           6
            (4, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-11_16-55-25/runs/checkpoint_59453394.pt")), # = 1           7
            (4, str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-13_23-49-40/runs/checkpoint_80988414.pt")), # = 0.5         8 
            str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-16_01-45-38_2P/runs/checkpoint_7618884.pt"), # = 0              9
            str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-17_00-10-22_4P/runs/checkpoint_44595824.pt"), # = 0              10
            str(IMPALA_PROJECT_ROOT / "outputs/train/2026-06-19_00-07-56_4P/runs/checkpoint_83110768.pt"), # = 0              11
        ],
    },



    "num_buffers": 40,
    "num_stats_buffers": 40,
    "num_inference_buffers_train": 40,

    "num_inference_workers_train_cuda0": 0,
    "num_inference_workers_train_cuda1": 8,

    "n_batch_prepare_processes": 1,
    "prepare_batches": 4,

    "logging_config": {
        "frequency": 1000,
        "model_stats_frequency": 100,
    },
    "torch_io_config": {
        "main_torch_io": _torch_io_local(_LOCAL_MAIN_RUNS),
    },
    
    "learner_cuda_device": 0,
    "inference_cuda_device": 1,

}


def build_training_config() -> SimpleNamespace:
    merged = deep_merge_dicts(default_training_config_dict(), IMPALA_X1_RL_OVERRIDES)
    validated = ImpalaTrainingConfig.model_validate(merged)
    out = _ns(validated.model_dump(mode="python"))
    out.resume_checkpoint = validated.resume_checkpoint
    return out
