"""IMPALA Orbit observation feature normalization parameters."""
from __future__ import annotations

import copy
from typing import Any

from .impala_orbit_model_hyperparams import ORBIT_IMPALA_OBS_FEATURE_LAYOUT


def _obs_feature_norm(
    *,
    mean: float,
    std: float,
    clip_down: float,
    clip_up: float,
    spike_values: tuple[float, ...],
    enabled: bool = True,
    norm_enabled: bool = True,
    clip_enabled: bool = True,
) -> dict[str, Any]:
    assert std > 0.0, std
    assert clip_down <= clip_up, (clip_down, clip_up)
    return {
        "mean": float(mean),
        "std": float(std),
        "clip_down": float(clip_down),
        "clip_up": float(clip_up),
        "spike_values": tuple(float(v) for v in spike_values),
        "enabled": bool(enabled),
        "norm_enabled": bool(norm_enabled),
        "clip_enabled": bool(clip_enabled),
    }


def _identity_obs_feature_norm() -> dict[str, Any]:
    return _obs_feature_norm(
        mean=0.0,
        std=1.0,
        clip_down=-1_000_000_000.0,
        clip_up=1_000_000_000.0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    )


def _obs_feature_norm_entries(prefix: str, names: tuple[str, ...]) -> dict[str, Any]:
    return {f"continuous.{prefix}.{name}": _identity_obs_feature_norm() for name in names}


ORBIT_IMPALA_OBS_FEATURE_NORMALIZATION_OVERRIDES: dict[str, Any] = {
    'continuous.edge.edge_distance': _obs_feature_norm(
        mean=59.29584885,
        std=26.14145088,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_ships_if_owned': _obs_feature_norm(
    #     mean=116.4384689,
    #     std=241.3293915,
    #     clip_down=1,
    #     clip_up=1632,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),
    'continuous.planet.planet_player_ships_if_owned': _obs_feature_norm(
        mean=41.98107529,
        std=80.45072174,
        clip_down=1,
        clip_up=1123,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.planet.planet_orbit_radius': _obs_feature_norm(
        mean=43.82223511,
        std=10.84406853,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.planet.planet_neutral_ships': _obs_feature_norm(
        mean=34.36028671,
        std=21.10679626,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_owner_survival_margin': _obs_feature_norm(
    #     mean=83.75833893,
    #     std=162.1826019,
    #     clip_down=-412,
    #     clip_up=866,
    #     spike_values=(0, 1000),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.planet.planet_player_owner_survival_margin': _obs_feature_norm(
        mean=38.73189163,
        std=107.448204,
        clip_down=-734,
        clip_up=1225,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.arrival.arrival_ships': _obs_feature_norm(
    #     mean=149.5485382,
    #     std=334.4903564,
    #     clip_down=5,
    #     clip_up=2486,
    #     spike_values=(0, 1, 2, 3, 4),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.arrival.arrival_ships': _obs_feature_norm(
        mean=53.87214661,
        std=100.2301331,
        clip_down=1,
        clip_up=1087,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.planet.planet_radius': _obs_feature_norm(
        mean=1.771131635,
        std=0.6226575971,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_flip_time_by_player': _obs_feature_norm(
    #     mean=3.471387863,
    #     std=2.839404583,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(17, 0),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.planet.planet_player_flip_time_by_player': _obs_feature_norm(
        mean=3.664484501,
        std=3.152090311,
        clip_down=0,
        clip_up=0,
        spike_values=(41, 0),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_post_horizon_owner_margin': _obs_feature_norm(
    #     mean=-33.89117813,
    #     std=199.2225342,
    #     clip_down=-973,
    #     clip_up=960,
    #     spike_values=(-1000, 1000),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.planet.planet_player_post_horizon_owner_margin': _obs_feature_norm(
        mean=-29.31481361,
        std=172.9951019,
        clip_down=-1225,
        clip_up=1085,
        spike_values=(),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.planet.planet_x': _obs_feature_norm(
        mean=50.00000381,
        std=31.92163849,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.planet.planet_is_comet': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),


    'continuous.planet.planet_player_production_if_owned': _obs_feature_norm(
        mean=2.928603649,
        std=1.43857491,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_stable_flip_time_by_player': _obs_feature_norm(
    #     mean=3.586918354,
    #     std=2.857773542,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(17, 0),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.planet.planet_player_stable_flip_time_by_player': _obs_feature_norm(
        mean=3.841017485,
        std=3.191686153,
        clip_down=0,
        clip_up=0,
        spike_values=(41, 0),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.planet.planet_y': _obs_feature_norm(
        mean=49.99999619,
        std=31.92163849,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_total_fleet_frac': _obs_feature_norm(
    #     mean=3.094261169,
    #     std=3.985989571,
    #     clip_down=0.01300000027,
    #     clip_up=21.13699913,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.planet.planet_player_total_fleet_frac': _obs_feature_norm(
        mean=0.9304391146,
        std=0.8516796231,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.arrival.battle_tie_valid': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    'continuous.planet.planet_production': _obs_feature_norm(
        mean=2.592775822,
        std=1.467259884,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.planet.planet_is_static': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_time_to_hit_with_max_send': _obs_feature_norm(
    #     mean=9.38849926,
    #     std=4.086244583,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_time_to_hit_with_max_send': _obs_feature_norm(
        mean=18.26072693,
        std=9.87480545,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.edge.edge_player_src_owned': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),


    'continuous.edge.edge_is_available_with_max_send': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    'continuous.edge.edge_velocity_dy': _obs_feature_norm(
        mean=-2.727297901e-11,
        std=1.423472524,
        clip_down=-5.437144279,
        clip_up=5.437144279,
        spike_values=(0,),
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.edge.edge_player_dst_owned': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_max_send_stable_roi': _obs_feature_norm(
    #     mean=0.5751566887,
    #     std=1.150947452,
    #     clip_down=0.0008918617386,
    #     clip_up=9.333333015,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_max_send_stable_roi': _obs_feature_norm(
        mean=4.533049583,
        std=8.688809395,
        clip_down=0.02755101956,
        clip_up=120,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.arrival.resolution_ships': _obs_feature_norm(
    #     mean=190.2492218,
    #     std=331.8403625,
    #     clip_down=1,
    #     clip_up=2323,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.arrival.resolution_ships': _obs_feature_norm(
        mean=119.0479736,
        std=125.6720886,
        clip_down=2,
        clip_up=1376,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.edge.edge_dst_motion_angle_to_src_dst': _obs_feature_norm(
        mean=1.57083118,
        std=0.8401789069,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        norm_enabled=True,
        clip_enabled=False,
    ),


    'continuous.planet.planet_is_dynamic': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),



    'continuous.edge.edge_min_stable_takeover_bucket_available': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),


    # 'continuous.arrival.battle_tie_distance': _obs_feature_norm(
    #     mean=57.35634613,
    #     std=118.2638931,
    #     clip_down=1,
    #     clip_up=783,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.arrival.battle_tie_distance': _obs_feature_norm(
        mean=47.11297607,
        std=85.83888245,
        clip_down=1,
        clip_up=1016,
        spike_values=(10000,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),


    # 'continuous.edge.edge_min_stable_takeover_bucket_hit_steps': _obs_feature_norm(
    #     mean=10.46306324,
    #     std=3.580165625,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 16),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_stable_takeover_bucket_hit_steps': _obs_feature_norm(
        mean=22.60308266,
        std=10.75621796,
        clip_down=0,
        clip_up=0,
        spike_values=(0, 40),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),



    'continuous.edge.edge_min_time_takeover_bucket_available': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    'continuous.edge.edge_closing_speed': _obs_feature_norm(
        mean=0.03015619703,
        std=1.310275793,
        clip_down=-5.181089401,
        clip_up=5.280976295,
        spike_values=(-0,),
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.planet.planet_episode_step': _obs_feature_norm(
        mean=242.1731262,
        std=143.1326599,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_neutralize_margin_with_max_send': _obs_feature_norm( # BROKEN REGEN DATASET
    #     mean=-594.7349854,
    #     std=496.2426758,
    #     clip_down=-999,
    #     clip_up=1000,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_neutralize_margin_with_max_send': _obs_feature_norm(
        mean=41.72373199,
        std=75.42422485,
        clip_down=1,
        clip_up=975,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),


    # 'continuous.edge.edge_overkill_with_min_stable_bucket': _obs_feature_norm(
    #     mean=29.05380058,
    #     std=58.257267,
    #     clip_down=0,
    #     clip_up=369,
    #     spike_values=(-1000,),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_overkill_with_min_stable_bucket': _obs_feature_norm(
        mean=4.718305111,
        std=9.030800819,
        clip_down=0,
        clip_up=168,
        spike_values=(-10000,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.edge.edge_velocity_dx': _obs_feature_norm(
        mean=1.09397158e-10,
        std=1.453283548,
        clip_down=0,
        clip_up=0,
        spike_values=(-0,),
        norm_enabled=True,
        clip_enabled=False,
    ),

    

    'continuous.edge.edge_min_time_neutralize_bucket_available': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_takeover_bucket_hit_steps': _obs_feature_norm(
    #     mean=10.45420074,
    #     std=3.579662561,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 16),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_takeover_bucket_hit_steps': _obs_feature_norm(
        mean=22.57386017,
        std=10.76664734,
        clip_down=0,
        clip_up=0,
        spike_values=(0, 40),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_time_takeover_bucket_hit_steps': _obs_feature_norm(
    #     mean=9.222241402,
    #     std=4.132406235,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_takeover_bucket_hit_steps': _obs_feature_norm(
        mean=18.26072693,
        std=9.87480545,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.arrival.arrival_leverage': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(1,),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_time_stable_takeover_bucket_hit_steps': _obs_feature_norm(
    #     mean=9.229992867,
    #     std=4.131333828,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_stable_takeover_bucket_hit_steps': _obs_feature_norm(
        mean=16.03710938,
        std=9.348034859,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.edge.edge_dst_neutral': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    'continuous.edge.edge_min_neutralize_bucket_available': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_last_decisive_battle_step': _obs_feature_norm(
    #     mean=3.476229191,
    #     std=2.75744462,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.planet.planet_player_last_decisive_battle_step': _obs_feature_norm(
        mean=3.705586195,
        std=3.027259111,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),
    

    # 'continuous.planet.planet_player_post_horizon_owner_margin': _obs_feature_norm(
    #     mean=-33.89117813,
    #     std=199.2225342,
    #     clip_down=-973,
    #     clip_up=960,
    #     spike_values=(-1000, 1000),
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.planet.planet_player_post_horizon_owner_margin': _obs_feature_norm(
        mean=-29.31481361,
        std=172.9951019,
        clip_down=-1225,
        clip_up=1085,
        spike_values=(),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.planet.planet_player_owner_churn': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        enabled=False,
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.planet.planet_player_owner_survival_margin': _obs_feature_norm(
    #     mean=86.59828186,
    #     std=189.4117126,
    #     clip_down=-1000,
    #     clip_up=985,
    #     spike_values=(0, 1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.planet.planet_player_owner_survival_margin': _obs_feature_norm(
        mean=38.73189163,
        std=107.448204,
        clip_down=-734,
        clip_up=1225,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),
    

    # 'continuous.edge.edge_stable_margin_with_max_send': _obs_feature_norm(
    #     mean=103.7325516,
    #     std=245.6249084,
    #     clip_down=-1000,
    #     clip_up=990,
    #     spike_values=(0, 1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_stable_margin_with_max_send': _obs_feature_norm(
        mean=-7.381067753,
        std=127.3797836,
        clip_down=-1061,
        clip_up=1122,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),


    'continuous.edge.edge_min_time_stable_takeover_bucket_available': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_stable_takeover_bucket_roi': _obs_feature_norm(
    #     mean=5.99651289,
    #     std=10.49528885,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_stable_takeover_bucket_roi': _obs_feature_norm(
        mean=25.9167881,
        std=37.9801445,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_takeover_bucket': _obs_feature_norm(
    #     mean=23.11586952,
    #     std=20.17166901,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 1),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_takeover_bucket': _obs_feature_norm(
        mean=13.52582359,
        std=11.6024456,
        clip_down=0,
        clip_up=0,
        spike_values=(0, 1),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),


    # 'continuous.edge.edge_min_stable_takeover_bucket': _obs_feature_norm(
    #     mean=23.17058182,
    #     std=20.19833183,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 1),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_stable_takeover_bucket': _obs_feature_norm(
        mean=13.54441357,
        std=11.62891579,
        clip_down=0,
        clip_up=0,
        spike_values=(0, 1),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_time_stable_takeover_bucket': _obs_feature_norm(
    #     mean=39.03744125,
    #     std=27.99494553,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_stable_takeover_bucket': _obs_feature_norm(
        mean=21.74017143,
        std=16.80196953,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),


    # 'continuous.edge.edge_stable_capture_vs_current_owner_value': _obs_feature_norm(
    #     mean=64.59157562,
    #     std=62.67448044,
    #     clip_down=1,
    #     clip_up=425,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_stable_capture_vs_current_owner_value': _obs_feature_norm(
        mean=112.3997269,
        std=98.05007172,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_takeover_ships': _obs_feature_norm(
    #     mean=58.78822327,
    #     std=86.57944489,
    #     clip_down=1,
    #     clip_up=540,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_min_takeover_ships': _obs_feature_norm(
        mean=22.91984177,
        std=31.25196457,
        clip_down=1,
        clip_up=312,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_takeover_margin_with_max_send': _obs_feature_norm(
    #     mean=104.0722427,
    #     std=245.480545,
    #     clip_down=-1000,
    #     clip_up=990,
    #     spike_values=(0, 1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_takeover_margin_with_max_send': _obs_feature_norm(
        mean=-7.178249836,
        std=127.3974152,
        clip_down=-1061,
        clip_up=1123,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.arrival.production_swing_per_ship': _obs_feature_norm(
    #     mean=7.147288322,
    #     std=14.22388268,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.arrival.production_swing_per_ship': _obs_feature_norm(
        mean=17.3116188,
        std=35.17809296,
        clip_down=0.009259259328,
        clip_up=195,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.planet.planet_sun_angle': _obs_feature_norm(
        mean=0.005612774286,
        std=1.797802925,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_time_takeover_ships': _obs_feature_norm(
    #     mean=135.5141296,
    #     std=152.0095978,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_takeover_ships': _obs_feature_norm(
        mean=49.41168594,
        std=71.47599792,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),



    # 'continuous.edge.edge_min_neutralize_bucket_hit_steps': _obs_feature_norm(
    #     mean=11.08959579,
    #     std=3.39849782,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 16),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_neutralize_bucket_hit_steps': _obs_feature_norm(
        mean=29.19917488,
        std=10.89793777,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.arrival.time_step': _obs_feature_norm(
    #     mean=8.5,
    #     std=4.609772205,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.arrival.time_step': _obs_feature_norm(
        mean=20.5,
        std=11.543396,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_neutralize_ships': _obs_feature_norm(
    #     mean=32.94499969,
    #     std=71.06632233,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_neutralize_ships': _obs_feature_norm(
        mean=8.587630272,
        std=9.280441284,
        clip_down=1,
        clip_up=140,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_source_stable_hold_margin_after_min_stable_takeover': _obs_feature_norm(
    #     mean=-130.774704,
    #     std=186.5353699,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, -1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_source_stable_hold_margin_after_min_stable_takeover': _obs_feature_norm(
        mean=-104.1205368,
        std=149.5233307,
        clip_down=-1260,
        clip_up=-1,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.edge.edge_min_takeover_bucket_available': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),


    'continuous.edge.edge_dst_final_owner_is_src_owner_without_action': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),


    'continuous.edge.edge_src_neutral': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    'continuous.arrival.resolution_owner': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),
    

    # 'continuous.edge.edge_source_stable_hold_margin_after_min_takeover': _obs_feature_norm(
    #     mean=-130.3198242,
    #     std=186.1850586,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, -1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_source_stable_hold_margin_after_min_takeover': _obs_feature_norm(
        mean=-103.9634552,
        std=149.3749084,
        clip_down=-1260,
        clip_up=-1,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_min_time_neutralize_bucket': _obs_feature_norm(
    #     mean=33.83995056,
    #     std=26.81046486,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_neutralize_bucket': _obs_feature_norm(
        mean=17.09847832,
        std=13.22268677,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.arrival.hold_valid': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.arrival.takeover_cost': _obs_feature_norm(
    #     mean=97.26792908,
    #     std=154.6291351,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.arrival.takeover_cost': _obs_feature_norm(
        mean=79.21593475,
        std=96.39624786,
        clip_down=1,
        clip_up=1266,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_min_time_stable_takeover_ships': _obs_feature_norm(
    #     mean=135.7391052,
    #     std=152.0838776,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_stable_takeover_ships': _obs_feature_norm(
        mean=49.45559692,
        std=71.53482819,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_time_takeover_bucket': _obs_feature_norm(
    #     mean=38.99108124,
    #     std=27.9874649,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_takeover_bucket': _obs_feature_norm(
        mean=21.73023224,
        std=16.7916317,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.arrival.neutralization_cost': _obs_feature_norm(
    #     mean=13.97770882,
    #     std=21.51285172,
    #     clip_down=1,
    #     clip_up=180,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.arrival.neutralization_cost': _obs_feature_norm(
        mean=13.13112926,
        std=21.75160027,
        clip_down=1,
        clip_up=300,
        spike_values=(10000, 0),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_capture_deadline_slack': _obs_feature_norm(
    #     mean=3.192257881,
    #     std=8.843104362,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_capture_deadline_slack': _obs_feature_norm(
        mean=8.452507973,
        std=23.72919655,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.planet.planet_angular_velocity': _obs_feature_norm(
        mean=0.03852263093,
        std=0.0068338071,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    
    'continuous.arrival.neutralization_valid': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),
    

    # 'continuous.arrival.hold_cost': _obs_feature_norm(
    #     mean=90.54806519,
    #     std=142.5675201,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 1, 4, 3, 1000, 2),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.arrival.hold_cost': _obs_feature_norm(
        mean=65.6517334,
        std=96.64311981,
        clip_down=1,
        clip_up=1056,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    'continuous.arrival.hold_valid': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    'continuous.edge.edge_attack_redundancy_score': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_stable_takeover_ships': _obs_feature_norm(
    #     mean=49.09558105,
    #     std=83.33139038,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_stable_takeover_ships': _obs_feature_norm(
        mean=25.07817078,
        std=33.02511215,
        clip_down=2,
        clip_up=352,
        spike_values=(0, 1),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_min_time_neutralize_bucket_hit_steps': _obs_feature_norm(
    #     mean=9.439317703,
    #     std=4.096652031,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_neutralize_bucket_hit_steps': _obs_feature_norm(
        mean=19.55182838,
        std=10.25128651,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),
        
    # 'continuous.arrival.stable_takeover_cost': _obs_feature_norm(
    #     mean=97.71385956,
    #     std=152.8537903,
    #     clip_down=1,
    #     clip_up=981,
    #     spike_values=(0, 1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.arrival.stable_takeover_cost': _obs_feature_norm(
        mean=79.6499176,
        std=96.67781067,
        clip_down=1,
        clip_up=1267,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),
   

    # 'continuous.edge.edge_arrival_tactical_pressure': _obs_feature_norm(
    #     mean=1.523179054,
    #     std=12.75524998,
    #     clip_down=-52,
    #     clip_up=52,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=True,
    # ),

    'continuous.edge.edge_arrival_tactical_pressure': _obs_feature_norm(
        mean=0.8441126347,
        std=65.97457886,
        clip_down=-694,
        clip_up=633,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.edge.edge_min_neutralize_bucket': _obs_feature_norm(
    #     mean=13.95174122,
    #     std=17.56522369,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_neutralize_bucket': _obs_feature_norm(
        mean=5.13076067,
        std=5.415802956,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    # 'continuous.edge.edge_min_time_neutralize_ships': _obs_feature_norm(
    #     mean=110.957016,
    #     std=140.8632507,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0,),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.edge.edge_min_time_neutralize_ships': _obs_feature_norm(
        mean=32.74618912,
        std=51.20774841,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

    'continuous.edge.edge_snipe_score_at_min_takeover_time': _obs_feature_norm(
        mean=0,
        std=1,
        clip_down=0,
        clip_up=0,
        spike_values=(),
        enabled=False,
        norm_enabled=False,
        clip_enabled=False,
    ),

    # 'continuous.arrival.deny_stable_enemy_cost': _obs_feature_norm(
    #     mean=134.192627,
    #     std=185.6971893,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 1000),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.arrival.deny_stable_enemy_cost': _obs_feature_norm(
        mean=116.2564774,
        std=122.628891,
        clip_down=1,
        clip_up=1491,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=True,
    ),

    # 'continuous.planet.planet_comet_time_before_despawn': _obs_feature_norm(
    #     mean=8.515485764,
    #     std=4.611777782,
    #     clip_down=0,
    #     clip_up=0,
    #     spike_values=(0, 17),
    #     enabled=True,
    #     norm_enabled=True,
    #     clip_enabled=False,
    # ),

    'continuous.planet.planet_comet_time_before_despawn': _obs_feature_norm(
        mean=17.45616531,
        std=9.538769722,
        clip_down=0,
        clip_up=0,
        spike_values=(0,),
        enabled=True,
        norm_enabled=True,
        clip_enabled=False,
    ),

}


ORBIT_IMPALA_OBS_FEATURE_NORMALIZATION: dict[str, Any] = {
    "layout": copy.deepcopy(ORBIT_IMPALA_OBS_FEATURE_LAYOUT),
    "features": {
        **ORBIT_IMPALA_OBS_FEATURE_NORMALIZATION_OVERRIDES,
    },
}
