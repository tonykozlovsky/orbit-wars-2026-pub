"""IMPALA ``ImpalaOrbitModel`` hyperparameters (merged into training config ``model``)."""

from __future__ import annotations

import copy
from typing import Any

from src.gym.obs_wrapper import ORBIT_PLANET_ARRIVAL_HORIZON

ORBIT_IMPALA_HIDDEN_DIM: int = 128
ORBIT_IMPALA_USE_TOKEN_BATCH_NORM: bool = False
ORBIT_IMPALA_ENCODER_NUM_LAYERS: int = 3

ORBIT_IMPALA_ARRIVALS_ENCODER_NUM_LAYERS: int = 3

ORBIT_IMPALA_ARRIVALS_FUSION_NUM_LAYERS: int = 2
ORBIT_IMPALA_DECODER_NUM_LAYERS: int = 2
ORBIT_IMPALA_TRANSFORMER_NUM_LAYERS: int = 16
ORBIT_IMPALA_TRANSFORMER_NUM_HEADS: int = 4
ORBIT_IMPALA_EDGE_ENDPOINT_UPDATE: str = "mha_speedup"
ORBIT_IMPALA_MLP_ACTIVATION: str = "leaky_relu"
ORBIT_IMPALA_TRANSFORMER_ACTIVATION: str = "gelu"
ORBIT_IMPALA_RESIDUAL_DROPOUT: float = 0.0
ORBIT_IMPALA_FFN_MULTIPLIER: int = 4
ORBIT_IMPALA_FFN_DROPOUT: float = 0.0
ORBIT_IMPALA_USE_OBS_DISCRETE_EMBEDDINGS: bool = True

ORBIT_IMPALA_USE_IDENTITY_EMBEDDING: bool = True
ORBIT_IMPALA_USE_EDGE_RELATION_FEATURES: bool = False
ORBIT_IMPALA_USE_ARRIVAL_ATTENTION_FUSION: bool = True
ORBIT_IMPALA_ARRIVAL_ATTENTION_NUM_QUERIES: int = 4
ORBIT_IMPALA_USE_ARRIVAL_ATTENTION_FUSION_GATE: bool = False
ORBIT_IMPALA_ALLOW_SELF_EDGES: bool = True

ORBIT_IMPALA_ENTROPY_FLOOR_TARGET: tuple[float] = (0.0,)
ORBIT_IMPALA_ENTROPY_FLOOR_MAX_TEMPERATURE: float = 1
ORBIT_IMPALA_ENTROPY_FLOOR_NUM_ITERS: int = 0

ORBIT_IMPALA_OBS_FEATURE_LAYOUT: dict[str, Any] = {
    "planet_base_feature_names": (
        "planet_x",
        "planet_y",
        "planet_neutral_ships",
        "planet_episode_step",
        "planet_is_static",
        "planet_is_dynamic",
        "planet_is_comet",
        "planet_comet_time_before_despawn",
        "planet_radius",
        "planet_production",
        "planet_orbit_radius",
        "planet_angular_velocity",
        "planet_sun_angle",
    ),
    "planet_player_block_feature_names": (
        "planet_player_ships_if_owned",
        "planet_player_total_fleet_frac",
        "planet_player_production_if_owned",
        "planet_player_owner_survival_margin",
        "planet_player_flip_time_by_player",
        "planet_player_stable_flip_time_by_player",
        "planet_player_owner_churn",
        "planet_player_last_decisive_battle_step",
        "planet_player_post_horizon_owner_margin",
    ),
    "edge_base_feature_names": (
        "edge_distance",
        "edge_src_neutral",
        "edge_dst_neutral",
        "edge_min_takeover_bucket",
        "edge_min_takeover_ships",
        "edge_min_takeover_bucket_available",
        "edge_min_takeover_bucket_hit_steps",
        "edge_min_time_takeover_bucket",
        "edge_min_time_takeover_ships",
        "edge_min_time_takeover_bucket_available",
        "edge_min_time_takeover_bucket_hit_steps",
        "edge_min_stable_takeover_bucket",
        "edge_min_stable_takeover_ships",
        "edge_min_stable_takeover_bucket_available",
        "edge_min_stable_takeover_bucket_hit_steps",
        "edge_min_time_stable_takeover_bucket",
        "edge_min_time_stable_takeover_ships",
        "edge_min_time_stable_takeover_bucket_available",
        "edge_min_time_stable_takeover_bucket_hit_steps",
        "edge_min_neutralize_bucket",
        "edge_min_neutralize_ships",
        "edge_min_neutralize_bucket_available",
        "edge_min_neutralize_bucket_hit_steps",
        "edge_min_time_neutralize_bucket",
        "edge_min_time_neutralize_ships",
        "edge_min_time_neutralize_bucket_available",
        "edge_min_time_neutralize_bucket_hit_steps",
        "edge_takeover_margin_with_max_send",
        "edge_stable_margin_with_max_send",
        "edge_neutralize_margin_with_max_send",
        "edge_time_to_hit_with_max_send",
        "edge_is_available_with_max_send",
        "edge_dst_motion_angle_to_src_dst",
        "edge_velocity_dx",
        "edge_velocity_dy",
        "edge_closing_speed",
        "edge_min_stable_takeover_bucket_roi",
        "edge_max_send_stable_roi",
        "edge_source_stable_hold_margin_after_min_takeover",
        "edge_source_stable_hold_margin_after_min_stable_takeover",
        "edge_capture_deadline_slack",
        "edge_arrival_tactical_pressure",
        "edge_snipe_score_at_min_takeover_time",
        "edge_overkill_with_min_stable_bucket",
        "edge_stable_capture_vs_current_owner_value",
        "edge_dst_final_owner_is_src_owner_without_action",
        "edge_attack_redundancy_score",
    ),
    "edge_player_block_feature_names": (
        "edge_player_src_owned",
        "edge_player_dst_owned",
    ),
}

from .impala_orbit_obs_feature_normalization import ORBIT_IMPALA_OBS_FEATURE_NORMALIZATION


IMPALA_ORBIT_MODEL_CONFIG: dict[str, Any] = {
    "value_heads": {},
    "orbit_impala": {
        "hidden_dim": ORBIT_IMPALA_HIDDEN_DIM,
        "use_token_batch_norm": ORBIT_IMPALA_USE_TOKEN_BATCH_NORM,
        "obs_feature_normalization": copy.deepcopy(ORBIT_IMPALA_OBS_FEATURE_NORMALIZATION),
        "encoder_num_layers": ORBIT_IMPALA_ENCODER_NUM_LAYERS,
        "arrivals_encoder_num_layers": ORBIT_IMPALA_ARRIVALS_ENCODER_NUM_LAYERS,
        "arrivals_fusion_num_layers": ORBIT_IMPALA_ARRIVALS_FUSION_NUM_LAYERS,
        "arrival_temporal_horizon": ORBIT_PLANET_ARRIVAL_HORIZON,
        "decoder_num_layers": ORBIT_IMPALA_DECODER_NUM_LAYERS,
        "transformer_num_layers": ORBIT_IMPALA_TRANSFORMER_NUM_LAYERS,
        "transformer_num_heads": ORBIT_IMPALA_TRANSFORMER_NUM_HEADS,
        "edge_endpoint_update": ORBIT_IMPALA_EDGE_ENDPOINT_UPDATE,
        "mlp_activation": ORBIT_IMPALA_MLP_ACTIVATION,
        "transformer_activation": ORBIT_IMPALA_TRANSFORMER_ACTIVATION,
        "num_policy_actions": 2,
        "residual_dropout": ORBIT_IMPALA_RESIDUAL_DROPOUT,
        "ffn_multiplier": ORBIT_IMPALA_FFN_MULTIPLIER,
        "ffn_dropout": ORBIT_IMPALA_FFN_DROPOUT,
        "use_obs_discrete_embeddings": ORBIT_IMPALA_USE_OBS_DISCRETE_EMBEDDINGS,
        "use_identity_embedding": ORBIT_IMPALA_USE_IDENTITY_EMBEDDING,
        "use_edge_relation_features": ORBIT_IMPALA_USE_EDGE_RELATION_FEATURES,
        "use_arrival_attention_fusion": ORBIT_IMPALA_USE_ARRIVAL_ATTENTION_FUSION,
        "arrival_attention_num_queries": ORBIT_IMPALA_ARRIVAL_ATTENTION_NUM_QUERIES,
        "use_arrival_attention_fusion_gate": ORBIT_IMPALA_USE_ARRIVAL_ATTENTION_FUSION_GATE,
        "allow_self_edges": ORBIT_IMPALA_ALLOW_SELF_EDGES,
        "zeroed_obs_feature_inputs": (),
    },
}


def default_training_model_dict() -> dict[str, Any]:
    return copy.deepcopy(IMPALA_ORBIT_MODEL_CONFIG)


def default_impala_orbit_model_init_kwargs() -> dict[str, Any]:
    return {
        "hidden_dim": ORBIT_IMPALA_HIDDEN_DIM,
        "use_token_batch_norm": ORBIT_IMPALA_USE_TOKEN_BATCH_NORM,
        "obs_feature_normalization": copy.deepcopy(ORBIT_IMPALA_OBS_FEATURE_NORMALIZATION),
        "encoder_num_layers": ORBIT_IMPALA_ENCODER_NUM_LAYERS,
        "arrivals_encoder_num_layers": ORBIT_IMPALA_ARRIVALS_ENCODER_NUM_LAYERS,
        "arrivals_fusion_num_layers": ORBIT_IMPALA_ARRIVALS_FUSION_NUM_LAYERS,
        "arrival_temporal_horizon": ORBIT_PLANET_ARRIVAL_HORIZON,
        "decoder_num_layers": ORBIT_IMPALA_DECODER_NUM_LAYERS,
        "transformer_num_layers": ORBIT_IMPALA_TRANSFORMER_NUM_LAYERS,
        "transformer_num_heads": ORBIT_IMPALA_TRANSFORMER_NUM_HEADS,
        "edge_endpoint_update": ORBIT_IMPALA_EDGE_ENDPOINT_UPDATE,
        "mlp_activation": ORBIT_IMPALA_MLP_ACTIVATION,
        "transformer_activation": ORBIT_IMPALA_TRANSFORMER_ACTIVATION,
        "num_policy_actions": 2,
        "residual_dropout": ORBIT_IMPALA_RESIDUAL_DROPOUT,
        "ffn_multiplier": ORBIT_IMPALA_FFN_MULTIPLIER,
        "ffn_dropout": ORBIT_IMPALA_FFN_DROPOUT,
        "use_obs_discrete_embeddings": ORBIT_IMPALA_USE_OBS_DISCRETE_EMBEDDINGS,
        "use_identity_embedding": ORBIT_IMPALA_USE_IDENTITY_EMBEDDING,
        "use_edge_relation_features": ORBIT_IMPALA_USE_EDGE_RELATION_FEATURES,
        "use_arrival_attention_fusion": ORBIT_IMPALA_USE_ARRIVAL_ATTENTION_FUSION,
        "arrival_attention_num_queries": ORBIT_IMPALA_ARRIVAL_ATTENTION_NUM_QUERIES,
        "use_arrival_attention_fusion_gate": ORBIT_IMPALA_USE_ARRIVAL_ATTENTION_FUSION_GATE,
        "allow_self_edges": ORBIT_IMPALA_ALLOW_SELF_EDGES,
        "zeroed_obs_feature_inputs": (),
        "entropy_floor_target": ORBIT_IMPALA_ENTROPY_FLOOR_TARGET,
        "entropy_floor_max_temperature": ORBIT_IMPALA_ENTROPY_FLOOR_MAX_TEMPERATURE,
        "entropy_floor_num_iters": ORBIT_IMPALA_ENTROPY_FLOOR_NUM_ITERS,
    }
