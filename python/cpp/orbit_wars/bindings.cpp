#include <torch/extension.h>

#include "cpp_env_v2/cpp_env_live_v2.h"
#include "cpp_env_v2/cpp_env_static_cache_v2.h"
#include "io.h"
#include "masks.h"
#include "simulation.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("orbit_wars_format_double_for_reset_trace",
        &orbit_wars_format_double_for_reset_trace);
  m.def("orbit_wars_format_double_two_decimals_for_reset_trace",
        &orbit_wars_format_double_two_decimals_for_reset_trace);
  m.def("orbit_wars_policy_obs_edge_distance", &orbit_wars_policy_obs_edge_distance);
  m.def("orbit_wars_fill_inactive_policy_action_noops",
        &orbit_wars_fill_inactive_policy_action_noops);
  m.def("orbit_wars_fleet_speed", &orbit_cpp_fleet_speed);
  py::class_<CppEnvStaticCacheV2>(m, "CppEnvStaticCacheV2")
      .def(py::init<int32_t, int32_t, double, int32_t, double>(),
           py::arg("num_agents"), py::arg("orbit_instance_id"), py::arg("ship_speed"),
           py::arg("episode_steps"), py::arg("comet_speed"))
      .def("reset", &CppEnvStaticCacheV2::reset, py::arg("angular_velocity"),
           py::arg("planet_rows"), py::arg("planet_count"))
      .def("update_comet_in_noop_cache", &CppEnvStaticCacheV2::update_comet_in_noop_cache,
           py::arg("current_episode_step"), py::arg("path_index"),
           py::arg("comet_internal_id"), py::arg("planet_ids"),
           py::arg("paths_kaggle_yx"))
      .def("noop_trajectory_length", &CppEnvStaticCacheV2::noop_trajectory_length)
      .def("noop_trajectory_planets_tensor",
           &CppEnvStaticCacheV2::noop_trajectory_planets_tensor)
      .def("noop_trajectory_planets_row_tensor",
           &CppEnvStaticCacheV2::noop_trajectory_planets_row_tensor,
           py::arg("step_index"))
      .def("honest_shared_action_mask_limited",
           &CppEnvStaticCacheV2::honest_shared_action_mask_limited,
           py::arg("episode_step"), py::arg("requests"), py::arg("out_action_mask"))
      .def("honest_shared_action_mask_all_geometry",
           &CppEnvStaticCacheV2::honest_shared_action_mask_all_geometry,
           py::arg("episode_step"), py::arg("out_action_mask"))
      .def("honest_shared_action_mask_full_cache_warmup_one",
           &CppEnvStaticCacheV2::honest_shared_action_mask_full_cache_warmup_one,
           py::arg("min_episode_step"), py::arg("current_planet_rows"),
           py::arg("planet_count"))
      .def("honest_shared_action_mask_full_cache_prune_before",
           &CppEnvStaticCacheV2::honest_shared_action_mask_full_cache_prune_before,
           py::arg("min_episode_step"))
      .def("honest_shared_action_mask_full_cache_warmup_stats",
           &CppEnvStaticCacheV2::honest_shared_action_mask_full_cache_warmup_stats)
      .def("send_all_from_external",
           &CppEnvStaticCacheV2::send_all_from_external,
           py::arg("episode_step"), py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("out_action_mask"))
      .def("fill_available_action_mask_from_rows",
           &CppEnvStaticCacheV2::fill_available_action_mask_from_rows,
           py::arg("episode_step"), py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("out_action_mask"))
      .def("fill_policy_obs_from_rows",
           &CppEnvStaticCacheV2::fill_policy_obs_from_rows,
           py::arg("episode_step"),
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"),
           py::arg("orbit_planet_features"),
           py::arg("orbit_planet_mask"),
           py::arg("orbit_planet_pairwise_mask"),
           py::arg("orbit_planet_pairwise_features"),
           py::arg("action_taken_index"),
           py::arg("player_mask"))
      .def("fleet_arrivals_from_rows",
           &CppEnvStaticCacheV2::fleet_arrivals_from_rows,
           py::arg("episode_step"),
           py::arg("fleet_rows"), py::arg("horizon"))
      .def("fleet_arrival_features_from_rows",
           &CppEnvStaticCacheV2::fleet_arrival_features_from_rows,
           py::arg("episode_step"),
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"))
      .def("fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows",
           &CppEnvStaticCacheV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows,
           py::arg("episode_step"),
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"),
           py::arg("orbit_planet_features"),
           py::arg("orbit_planet_pairwise_features"),
           py::arg("available_hit_mask"),
           py::arg("orbit_planet_arrival_features"))
      .def("fleet_arrivals_resolution_from_rows",
           &CppEnvStaticCacheV2::fleet_arrivals_resolution_from_rows,
           py::arg("episode_step"),
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"))
      .def("fleet_hit_traces_from_rows",
           &CppEnvStaticCacheV2::fleet_hit_traces_from_rows,
           py::arg("episode_step"), py::arg("fleet_rows"),
           py::arg("horizon"))
      .def("honest_shared_hit_kind_last",
           &CppEnvStaticCacheV2::honest_shared_hit_kind_last)
      .def("honest_shared_hit_slot_last",
           &CppEnvStaticCacheV2::honest_shared_hit_slot_last)
      .def("honest_shared_hit_steps_last",
           &CppEnvStaticCacheV2::honest_shared_hit_steps_last)
      .def("honest_shared_intercept_fail_reason_last",
           &CppEnvStaticCacheV2::honest_shared_intercept_fail_reason_last)
      .def("honest_shared_send_ships_last",
           &CppEnvStaticCacheV2::honest_shared_send_ships_last)
      .def("honest_shared_dir_last",
           &CppEnvStaticCacheV2::honest_shared_dir_last)
      .def("honest_shared_angle_last_for_action_class",
           &CppEnvStaticCacheV2::honest_shared_angle_last_for_action_class)
      .def("set_wall_profile_enabled",
           &CppEnvStaticCacheV2::set_wall_profile_enabled)
      .def("wall_profile_rows",
           &CppEnvStaticCacheV2::wall_profile_rows);

  py::class_<CppEnvLiveV2>(m, "CppEnvLiveV2")
      .def(py::init<int32_t, int32_t, double, int32_t, double, bool>(),
           py::arg("num_agents"), py::arg("orbit_instance_id"), py::arg("ship_speed"),
           py::arg("episode_steps"), py::arg("comet_speed"),
           py::arg("enable_state_trace") = true)
      .def("reset",
           &CppEnvLiveV2::reset,
           py::arg("angular_velocity"),
           py::arg("planet_rows"), py::arg("planet_count"),
           py::arg("orbit_planet_features"), py::arg("orbit_planet_mask"),
           py::arg("orbit_planet_pairwise_mask"),
           py::arg("orbit_planet_pairwise_features"),
           py::arg("action_taken_index"), py::arg("player_mask"))
      .def("update_comets_from_state", &CppEnvLiveV2::update_comets_from_state,
           py::arg("comet_planet_ids"), py::arg("comet_path_by_planet_id"))
      .def("step", &CppEnvLiveV2::step)
      .def("reset_trace_get", &CppEnvLiveV2::reset_trace_get)
      .def("step_trace_get", &CppEnvLiveV2::step_trace_get)
      .def("set_wall_profile_enabled",
           &CppEnvLiveV2::set_wall_profile_enabled)
      .def("wall_profile_rows",
           &CppEnvLiveV2::wall_profile_rows)
      .def("orbit_episode_terminal", &CppEnvLiveV2::orbit_episode_terminal)
      .def("fleet_ship_total_int_for_owner",
           &CppEnvLiveV2::fleet_ship_total_int_for_owner)
      .def("player_alive_for_owner", &CppEnvLiveV2::player_alive_for_owner)
      .def("planet_count_int_for_owner", &CppEnvLiveV2::planet_count_int_for_owner)
      .def("production_sum_for_owner", &CppEnvLiveV2::production_sum_for_owner)
      .def("game_result_for_owner", &CppEnvLiveV2::game_result_for_owner)
      .def("fleet_delta_for_owner", &CppEnvLiveV2::fleet_delta_for_owner)
      .def("planets_delta_for_owner", &CppEnvLiveV2::planets_delta_for_owner)
      .def("production_delta_for_owner", &CppEnvLiveV2::production_delta_for_owner)
      .def("step_metric_tensors", &CppEnvLiveV2::step_metric_tensors)
      .def("tape_kaggle_planets_rows", &CppEnvLiveV2::tape_kaggle_planets_rows)
      .def("tape_kaggle_fleets_rows", &CppEnvLiveV2::tape_kaggle_fleets_rows)
      .def("angular_velocity", &CppEnvLiveV2::angular_velocity)
      .def("ship_speed", &CppEnvLiveV2::ship_speed)
      .def("episode_step", &CppEnvLiveV2::episode_step)
      .def("kaggle_observation_step", &CppEnvLiveV2::kaggle_observation_step)
      .def("honest_shared_send_all_hit_mask",
           &CppEnvLiveV2::honest_shared_send_all_hit_mask,
           py::arg("out_hit_mask"))
      .def("fill_available_action_mask",
           &CppEnvLiveV2::fill_available_action_mask,
           py::arg("out_action_mask"))
      .def("honest_shared_hit_kind_last",
           &CppEnvLiveV2::honest_shared_hit_kind_last)
      .def("honest_shared_hit_slot_last",
           &CppEnvLiveV2::honest_shared_hit_slot_last)
      .def("honest_shared_hit_steps_last",
           &CppEnvLiveV2::honest_shared_hit_steps_last)
      .def("honest_shared_intercept_fail_reason_last",
           &CppEnvLiveV2::honest_shared_intercept_fail_reason_last)
      .def("honest_shared_send_ships_last",
           &CppEnvLiveV2::honest_shared_send_ships_last)
      .def("honest_shared_dir_last",
           &CppEnvLiveV2::honest_shared_dir_last)
      .def("honest_shared_angle_last_for_action_class",
           &CppEnvLiveV2::honest_shared_angle_last_for_action_class)
      .def("honest_shared_angle",
           &CppEnvLiveV2::honest_shared_angle)
      .def("honest_shared_angle_or_nan",
           &CppEnvLiveV2::honest_shared_angle_or_nan)
      .def("honest_shared_intercept_trace",
           &CppEnvLiveV2::honest_shared_intercept_trace)
      .def("fleet_arrivals_from_state",
           &CppEnvLiveV2::fleet_arrivals_from_state,
           py::arg("horizon"))
      .def("fleet_arrivals_from_rows",
           &CppEnvLiveV2::fleet_arrivals_from_rows,
           py::arg("fleet_rows"), py::arg("horizon"))
      .def("fleet_arrival_features_from_state",
           &CppEnvLiveV2::fleet_arrival_features_from_state,
           py::arg("horizon"))
      .def("fleet_arrival_features_from_rows",
           &CppEnvLiveV2::fleet_arrival_features_from_rows,
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"))
      .def("fleet_arrival_features_and_fill_future_resolution_planet_features_from_state",
           &CppEnvLiveV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_state,
           py::arg("horizon"), py::arg("orbit_planet_features"),
           py::arg("orbit_planet_pairwise_features"),
           py::arg("available_hit_mask"),
           py::arg("orbit_planet_arrival_features"))
      .def("fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows",
           &CppEnvLiveV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows,
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"),
           py::arg("orbit_planet_features"),
           py::arg("orbit_planet_pairwise_features"),
           py::arg("available_hit_mask"),
           py::arg("orbit_planet_arrival_features"))
      .def("fill_policy_obs_from_rows",
           &CppEnvLiveV2::fill_policy_obs_from_rows,
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"),
           py::arg("orbit_planet_features"),
           py::arg("orbit_planet_mask"),
           py::arg("orbit_planet_pairwise_mask"),
           py::arg("orbit_planet_pairwise_features"),
           py::arg("action_taken_index"),
           py::arg("player_mask"))
      .def("fill_future_resolution_planet_features_from_state",
           &CppEnvLiveV2::fill_future_resolution_planet_features_from_state,
           py::arg("horizon"), py::arg("orbit_planet_features"))
      .def("fill_future_resolution_planet_features_from_rows",
           &CppEnvLiveV2::fill_future_resolution_planet_features_from_rows,
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"),
           py::arg("orbit_planet_features"))
      .def("fleet_takeover_cost_features_from_state",
           &CppEnvLiveV2::fleet_takeover_cost_features_from_state,
           py::arg("horizon"))
      .def("fleet_arrivals_resolution",
           &CppEnvLiveV2::fleet_arrivals_resolution,
           py::arg("arrivals"))
      .def("fleet_arrivals_resolution_from_state",
           &CppEnvLiveV2::fleet_arrivals_resolution_from_state,
           py::arg("horizon"))
      .def("fleet_arrivals_resolution_from_rows",
           &CppEnvLiveV2::fleet_arrivals_resolution_from_rows,
           py::arg("fleet_rows"), py::arg("planet_rows"),
           py::arg("planet_count"), py::arg("horizon"))
      .def("fleet_hit_traces_from_state",
           &CppEnvLiveV2::fleet_hit_traces_from_state,
           py::arg("horizon"))
      .def("fleet_hit_traces_from_rows",
           &CppEnvLiveV2::fleet_hit_traces_from_rows,
           py::arg("fleet_rows"), py::arg("horizon"))
      .def("comet_mask_inputs_py", &CppEnvLiveV2::comet_mask_inputs_py)
      .def("noop_trajectory_length", &CppEnvLiveV2::noop_trajectory_length)
      .def("noop_trajectory_planets_tensor",
           &CppEnvLiveV2::noop_trajectory_planets_tensor)
      .def("noop_trajectory_planets_row_tensor",
           &CppEnvLiveV2::noop_trajectory_planets_row_tensor,
           py::arg("step_index"))
      .def("assert_planets_match_noop_cache",
           &CppEnvLiveV2::assert_planets_match_noop_cache);
}
