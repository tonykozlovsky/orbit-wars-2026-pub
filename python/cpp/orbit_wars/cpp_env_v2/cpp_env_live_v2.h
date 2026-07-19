#pragma once

#include "cpp_env_static_cache_v2.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

// Live simulation environment: inherits static cache, adds all dynamic
// simulation state (planets, fleets, owners, episode step, etc.).
// Only used by the RL orbit wars env; all other call sites use
// CppEnvStaticCacheV2 directly.
class CppEnvLiveV2 : public CppEnvStaticCacheV2 {
 public:
  struct OrbitCometGroup {
    int32_t path_index = -1;
    int32_t internal_id = -1;
    std::vector<int32_t> planet_ids;
    std::vector<std::vector<std::pair<double, double>>> paths_kaggle_yx;
  };

  struct PlannedLaunch {
    int32_t player_id = -1;
    int32_t from_planet_id = -1;
    double angle = 0.0;
    int32_t ships = 0;
  };

  CppEnvLiveV2(int32_t num_agents, int32_t orbit_instance_id,
               double ship_speed, int32_t episode_steps, double comet_speed,
               bool enable_state_trace);

  void reset(
      double angular_velocity,
      torch::Tensor planet_rows, int32_t planet_count,
      torch::Tensor orbit_planet_features, torch::Tensor orbit_planet_mask,
      torch::Tensor orbit_planet_pairwise_mask,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor action_taken_index,
      torch::Tensor player_mask);
  void update_comets_from_state(py::iterable comet_planet_ids,
                                py::dict comet_path_by_planet_id);
  void step(torch::Tensor action_classes,
            torch::Tensor orbit_planet_features,
            torch::Tensor orbit_planet_mask,
            torch::Tensor orbit_planet_pairwise_mask,
            torch::Tensor orbit_planet_pairwise_features,
            torch::Tensor action_taken_index,
            torch::Tensor player_mask);

  std::string reset_trace_get() const;
  std::string step_trace_get() const;
  bool orbit_episode_terminal() const;
  int32_t fleet_ship_total_int_for_owner(int32_t owner) const;
  bool player_alive_for_owner(int32_t owner) const;
  int32_t planet_count_int_for_owner(int32_t owner) const;
  double production_sum_for_owner(int32_t owner) const;
  double game_result_for_owner(int32_t owner) const;
  double fleet_delta_for_owner(int32_t owner) const;
  double planets_delta_for_owner(int32_t owner) const;
  double production_delta_for_owner(int32_t owner) const;
  py::tuple step_metric_tensors() const;
  py::list tape_kaggle_planets_rows() const;
  py::list tape_kaggle_fleets_rows() const;
  double angular_velocity() const;
  double ship_speed() const;
  int32_t episode_step() const;
  int32_t kaggle_observation_step() const;
  void honest_shared_send_all_hit_mask(torch::Tensor out_hit_mask) const;
  void fill_available_action_mask(torch::Tensor out_action_mask) const;
  double honest_shared_angle(int32_t src_slot, int32_t dst_slot,
                             int32_t ship_count) const;
  double honest_shared_angle_or_nan(int32_t src_slot, int32_t dst_slot,
                                    int32_t ship_count) const;
  torch::Tensor honest_shared_intercept_trace(int32_t src_slot,
                                              int32_t dst_slot,
                                              int32_t ship_subindex) const;
  torch::Tensor fleet_arrivals_from_state(int32_t horizon) const;
  torch::Tensor fleet_arrivals_from_rows(torch::Tensor fleet_rows,
                                         int32_t horizon) const;
  torch::Tensor fleet_arrival_features_from_state(int32_t horizon) const;
  torch::Tensor fleet_arrival_features_from_rows(torch::Tensor fleet_rows,
                                                 torch::Tensor planet_rows,
                                                 int32_t planet_count,
                                                 int32_t horizon) const;
  void fleet_arrival_features_and_fill_future_resolution_planet_features_from_state(
      int32_t horizon, torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor available_hit_mask,
      torch::Tensor orbit_planet_arrival_features) const;
  void fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      int32_t horizon,
      torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor available_hit_mask,
      torch::Tensor orbit_planet_arrival_features) const;
  void fill_policy_obs_from_rows(
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_mask,
      torch::Tensor orbit_planet_pairwise_mask,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor action_taken_index,
      torch::Tensor player_mask) const;
  void fill_future_resolution_planet_features_from_state(
      int32_t horizon, torch::Tensor orbit_planet_features) const;
  void fill_future_resolution_planet_features_from_rows(
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      int32_t horizon,
      torch::Tensor orbit_planet_features) const;
  torch::Tensor fleet_takeover_cost_features_from_state(int32_t horizon) const;
  py::tuple fleet_arrivals_resolution(torch::Tensor arrivals) const;
  py::tuple fleet_arrivals_resolution_from_state(int32_t horizon) const;
  py::tuple fleet_arrivals_resolution_from_rows(torch::Tensor fleet_rows,
                                                torch::Tensor planet_rows,
                                                int32_t planet_count,
                                                int32_t horizon) const;
  py::list fleet_hit_traces_from_state(int32_t horizon) const;
  py::list fleet_hit_traces_from_rows(torch::Tensor fleet_rows,
                                      int32_t horizon) const;
  py::tuple comet_mask_inputs_py() const;
  void assert_planets_match_noop_cache(int32_t step_index) const;

 private:
  using CombatBySlot = std::array<std::vector<Fleet>, kPlanets>;

  void load_kaggle_comet_groups(
      const SmallPlanetIdSet &comet_planet_ids,
      const CometPathByPlanetId &comet_paths);
  int32_t current_fleet_total_int_for_owner(int32_t owner) const;
  int32_t current_planet_count_int_for_owner(int32_t owner) const;
  double current_production_sum_for_owner(int32_t owner) const;
  void reset_metric_trackers_from_current_state();
  void refresh_delta_metrics_from_current_state();
  void trace_append_to(std::string &buf, const std::string &line);
  void trace_emit_planets_to(const char *stage, const std::vector<Planet> &planets,
                             std::string &buf);
  void trace_emit_fleets_to(const char *stage, const std::vector<Fleet> &fleets,
                            std::string &buf);
  void trace_emit_state_to(const char *stage, const char *block,
                           const std::vector<Planet> &planets,
                           const std::vector<Fleet> &fleets, std::string &buf);
  std::vector<PlannedLaunch> plan_launches_from_classes(
      torch::Tensor action_classes) const;
  void launch_planned_fleets(const std::vector<PlannedLaunch> &launches);
  void launch_fleets_from_kaggle_actions(py::iterable actions_by_player);
  void interpreter_pre_launch();
  void apply_pending_comets_in_pre_launch();
  void rebuild_comet_pid_set();
  void erase_planets_and_trim_comets(const SmallPlanetIdSet &dead);
  void remove_expired_comets_before_launch();
  void interpreter_simulation_after_launch(std::string &trace_buf,
                                           int32_t physics_step);
  double fleet_speed(int32_t ship_count) const;
  const Planet *initial_planet_by_id(int32_t id) const;
  void resolve_combat_keyed(const CombatBySlot &combat);
  void update_done(int32_t physics_step);
  void fill_noop_cache_fixed_planet_row(NoopCachedPlanet *row) const;
  void fill_outputs(torch::Tensor orbit_planet_features,
                    torch::Tensor orbit_planet_mask,
                    torch::Tensor orbit_planet_pairwise_mask,
                    torch::Tensor orbit_planet_pairwise_features,
                    torch::Tensor action_taken_index,
                    torch::Tensor player_mask) const;

  int32_t episode_index_ = 0;
  int32_t episode_step_ = 0;
  int32_t next_fleet_id_ = 0;
  bool done_ = false;
  std::vector<Planet> planets_;
  std::vector<Planet> initial_planets_;
  std::vector<Fleet> fleets_;
  std::vector<OrbitCometGroup> comet_groups_;
  SmallPlanetIdSet comet_pid_set_;
  bool pending_comet_sync_ = false;
  SmallPlanetIdSet pending_comet_planet_ids_;
  CometPathByPlanetId pending_comet_paths_;
  std::vector<int32_t> prev_fleet_total_by_owner_;
  std::vector<int32_t> prev_planet_count_by_owner_;
  std::vector<double> prev_production_sum_by_owner_;
  std::vector<int32_t> last_fleet_delta_;
  std::vector<int32_t> last_planets_delta_;
  std::vector<double> last_production_delta_;
  std::string reset_trace_;
  std::string step_trace_;
  bool enable_state_trace_;
  mutable torch::Tensor dataset_send_action_mask_scratch_;
};
