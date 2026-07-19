#pragma once

#include "../common.h"
#include "../honest_shared_intercept.h"

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

struct DynamicDynamicInterceptCacheEntry {
  uint8_t valid = 0;
  uint8_t fail_reason = 0;
  uint8_t hit_available = 0;
  int16_t hit_kind = orbit_wars_honest::kHitKindNone;
  int16_t hit_slot = orbit_wars_honest::kHitNone;
  int16_t hit_steps = -1;
  float dir_x0 = 0.0f;
  float dir_y0 = 0.0f;
  float turns_to_target = 0.0f;
};

struct HonestFullPairCacheEntry {
  int32_t filled_max_sn = 0;
  std::array<int16_t, kLegacyShipScanClasses> hit_kind{};
  std::array<int16_t, kLegacyShipScanClasses> hit_slot{};
  std::array<int16_t, kLegacyShipScanClasses> hit_steps{};
  std::array<float, kLegacyShipScanClasses> dir_x{};
  std::array<float, kLegacyShipScanClasses> dir_y{};
  std::array<float, kLegacyShipScanClasses> turns{};
  std::array<float, kLegacyShipScanClasses> intercept_ok{};
  std::array<float, kLegacyShipScanClasses> intercept_fail_reason{};
};

// Static cache environment: holds only the minimum static information required
// for mask and feature computation (noop trajectory, slot classifications,
// physics parameters). No dynamic simulation state (planets, fleets, owners,
// step counter). Dynamic information must be supplied externally at call sites.
class CppEnvStaticCacheV2 {
 public:
  CppEnvStaticCacheV2(int32_t num_agents, int32_t orbit_instance_id,
                      double ship_speed, int32_t episode_steps,
                      double comet_speed);

  // Resets the static cache for a new episode.  Parses the planet tensor,
  // then builds the full noop trajectory and slot classification.  Dynamic
  // data is supplied as parameters; no live fields are accessed or stored.
  void reset(double angular_velocity, torch::Tensor planet_rows,
             int32_t planet_count);

  // Lower-level entry point for callers that already hold a parsed planet
  // vector (e.g. CppEnvLiveV2::reset after setting up dynamic state).
  void build_noop_cache(double angular_velocity,
                        const std::vector<Planet> &initial_planets);

  // Rewrites noop comet slots for one spawn event (exactly four comets).
  // Call only on comet-spawn steps; planet_ids and paths_kaggle_yx must each
  // have length 4. Clears comet slots on all frames before writing the new group.
  void update_comet_in_noop_cache(
      int32_t current_episode_step,
      int32_t path_index,
      int32_t comet_internal_id,
      const std::vector<int32_t> &planet_ids,
      const std::vector<std::vector<std::pair<double, double>>> &paths_kaggle_yx);

  int32_t noop_trajectory_length() const;
  torch::Tensor noop_trajectory_planets_tensor() const;
  torch::Tensor noop_trajectory_planets_row_tensor(int32_t step_index) const;

  // Honest geometry mask for explicit (src, dst, max_sn) requests at episode_step.
  // Uses only noop cache and planet_slot_* classification (no live planets/fleets).
  void honest_shared_action_mask_limited(int32_t episode_step,
                                         torch::Tensor requests,
                                         torch::Tensor out_action_mask) const;
  void honest_shared_action_mask_all_geometry(int32_t episode_step,
                                              torch::Tensor out_action_mask) const;
  py::tuple honest_shared_action_mask_full_cache_warmup_one(
      int32_t min_episode_step, torch::Tensor current_planet_rows,
      int32_t planet_count) const;
  int32_t honest_shared_action_mask_full_cache_prune_before(
      int32_t min_episode_step) const;
  py::tuple honest_shared_action_mask_full_cache_warmup_stats() const;
  void send_all_from_external(int32_t episode_step, torch::Tensor fleet_rows,
                              torch::Tensor planet_rows, int32_t planet_count,
                              torch::Tensor out_action_mask) const;
  void fill_available_action_mask_from_rows(
      int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
      int32_t planet_count, torch::Tensor out_action_mask) const;
  void fill_policy_obs_from_rows(
      int32_t episode_step,
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_mask,
      torch::Tensor orbit_planet_pairwise_mask,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor action_taken_index,
      torch::Tensor player_mask) const;
  torch::Tensor fleet_arrivals_from_rows(
      int32_t episode_step, torch::Tensor fleet_rows, int32_t horizon) const;
  torch::Tensor fleet_arrival_features_from_rows(
      int32_t episode_step,
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      int32_t horizon) const;
  void fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
      int32_t episode_step,
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      int32_t horizon,
      torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor available_hit_mask,
      torch::Tensor orbit_planet_arrival_features) const;
  py::tuple fleet_arrivals_resolution_from_rows(
      int32_t episode_step,
      torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
      int32_t horizon) const;
  py::list fleet_hit_traces_from_rows(
      int32_t episode_step, torch::Tensor fleet_rows, int32_t horizon) const;
  torch::Tensor honest_shared_hit_kind_last() const;
  torch::Tensor honest_shared_hit_slot_last() const;
  torch::Tensor honest_shared_hit_steps_last() const;
  torch::Tensor honest_shared_intercept_fail_reason_last() const;
  torch::Tensor honest_shared_send_ships_last() const;
  py::tuple honest_shared_dir_last() const;
  double honest_shared_angle_last_for_action_class(int32_t src_slot,
                                                   int32_t action_class) const;
  void set_wall_profile_enabled(bool enabled);
  py::list wall_profile_rows() const;

  class WallProfileSpan {
   public:
    WallProfileSpan(const CppEnvStaticCacheV2 *env, const char *name);
    ~WallProfileSpan();

   private:
    const CppEnvStaticCacheV2 *env_;
    const char *name_;
    std::string path_;
    std::chrono::steady_clock::time_point t0_;
  };

  struct WallProfileStat {
    std::string path;
    double sum_ms = 0.0;
    int32_t n = 0;
    double last_ms = 0.0;
  };

 protected:
  // Physics constants — fixed for the lifetime of the object.
  int32_t num_agents_;
  int32_t orbit_instance_id_;
  double ship_speed_;
  int32_t episode_steps_;
  double comet_speed_;

  // Per-episode static configuration — set when the noop cache is built,
  // then fixed for the duration of the episode.
  double angular_velocity_ = 0.0;
  std::vector<NoopCachedPlanet> noop_cached_planets_flat_;
  orbit_wars_honest::NoopSpatialGrid noop_spatial_grid_;
  int32_t immutable_planet_prefix_n_ = 0;
  std::array<int32_t, kPlanets> immutable_planet_prefix_ids_{};

  // Slot classification arrays — derived from the planet layout at cache build
  // time. Replaces comet_pid_set_: all is-comet / is-orbiting checks use these
  // slot-indexed arrays instead of an ID set.
  std::array<int32_t, kPlanets> planet_slot_orbiting_{};
  int32_t planet_slot_orbiting_n_ = 0;
  std::array<int32_t, kPlanets> planet_slot_static_{};
  int32_t planet_slot_static_n_ = 0;
  std::array<int32_t, kPlanets> planet_slot_comet_{};
  int32_t planet_slot_comet_n_ = 0;

  // Wall profiler — utility, used from mask functions.
  mutable bool wall_profile_enabled_ = false;
  mutable std::vector<std::string> wall_profile_stack_;
  mutable std::vector<WallProfileStat> wall_profile_stats_;

  void fill_future_resolution_planet_features_from_rows(
      int32_t episode_step,
      torch::Tensor fleet_rows,
      torch::Tensor planet_rows,
      int32_t planet_count,
      int32_t horizon,
      torch::Tensor orbit_planet_features) const;
  py::tuple fleet_arrivals_resolution_from_arrivals_and_rows(
      torch::Tensor arrivals,
      torch::Tensor planet_rows,
      int32_t planet_count) const;
  void send_action_hit_mask_from_state(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      torch::Tensor out_action_mask) const;
  void fill_available_action_mask_from_hit_mask(
      const std::vector<Planet> &planets,
      torch::Tensor send_action_hit_mask,
      torch::Tensor out_action_mask) const;
  void fill_policy_obs_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_mask,
      torch::Tensor orbit_planet_pairwise_mask,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor action_taken_index,
      torch::Tensor player_mask) const;
  torch::Tensor fleet_arrivals_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      int32_t horizon) const;
  torch::Tensor fleet_arrival_features_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      int32_t horizon) const;
  void fleet_arrival_features_and_fill_future_resolution_planet_features_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      int32_t horizon,
      torch::Tensor orbit_planet_features,
      torch::Tensor orbit_planet_pairwise_features,
      torch::Tensor available_hit_mask,
      torch::Tensor orbit_planet_arrival_features) const;
  py::tuple fleet_arrivals_resolution_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      int32_t horizon) const;
  void fill_future_resolution_planet_features_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      int32_t horizon,
      torch::Tensor orbit_planet_features) const;
  torch::Tensor fleet_takeover_cost_features_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      const std::vector<Planet> &planets,
      int32_t horizon) const;
  py::tuple fleet_arrivals_resolution_from_arrivals_and_planets(
      torch::Tensor arrivals,
      const std::vector<Planet> &planets) const;
  py::list fleet_hit_traces_from_state_vectors(
      int32_t episode_step,
      const std::vector<Fleet> &fleets,
      int32_t horizon) const;
  double honest_shared_angle_or_nan_from_state(
      int32_t episode_step,
      const std::vector<Planet> &planets,
      int32_t src_slot,
      int32_t dst_slot,
      int32_t ship_count) const;
  torch::Tensor honest_shared_intercept_trace_from_state(
      int32_t episode_step,
      const std::vector<Planet> &planets,
      int32_t src_slot,
      int32_t dst_slot,
      int32_t ship_subindex) const;

 private:
  // Pre-allocated output scratch tensors for mask computation. Keep private so
  // live simulation cannot depend on mask side effects.
  mutable torch::Tensor last_honest_hit_kind_;
  mutable torch::Tensor last_honest_hit_slot_;
  mutable torch::Tensor last_honest_hit_steps_;
  mutable torch::Tensor last_honest_dir_x_;
  mutable torch::Tensor last_honest_dir_y_;
  mutable torch::Tensor last_honest_turns_;
  mutable torch::Tensor last_honest_intercept_ok_;
  mutable torch::Tensor last_honest_intercept_fail_reason_;
  mutable torch::Tensor last_honest_send_ships_;
  mutable torch::Tensor honest_available_hit_mask_scratch_;
  mutable std::vector<orbit_wars_honest::EdgeInterceptAim> cached_intercept_aims_scratch_;
  mutable std::vector<uint8_t> cached_intercept_aim_valid_scratch_;
  mutable std::vector<uint8_t> static_pair_aim_block_cache_;
  mutable std::vector<uint8_t> static_pair_aim_dynamic_possible_cache_;
  mutable std::vector<uint8_t> static_pair_bucket_aim_terminal_cache_;
  mutable std::vector<uint8_t> static_pair_bucket_aim_fail_reason_cache_;
  std::vector<DynamicDynamicInterceptCacheEntry> dynamic_dynamic_intercept_cache_;
  mutable std::vector<int32_t> honest_full_pair_cache_entry_by_key_;
  mutable std::vector<std::vector<HonestFullPairCacheEntry>> honest_full_pair_cache_entries_by_step_;
  mutable int32_t honest_full_pair_cache_live_entries_ = 0;
  mutable int32_t honest_full_pair_cache_prune_cursor_ = 0;
  mutable std::vector<uint8_t> honest_full_step_src_done_max_sn_;
  mutable int32_t honest_full_warmup_scan_step_ = 0;
  mutable int32_t honest_full_warmup_scan_src_order_idx_ = 0;
  mutable std::array<int32_t, kPlanets> honest_full_src_max_observed_ships_{};
  mutable int32_t honest_full_warmup_extra_sn_ = 0;
  mutable int32_t honest_full_warmup_last_lookahead_steps_ = 0;

  // Rebuilds planet_slot_orbiting/static/comet from noop cache frame 0 and
  // immutable_planet_prefix_n_. Slots 0..prefix_n-1 are classified by orbital
  // radius; comet slots are fixed at prefix_n..prefix_n+3.
  void rebuild_slot_kind_cache();
  void rebuild_static_pair_aim_block_cache();
  void rebuild_dynamic_dynamic_intercept_cache();
  void clear_honest_full_pair_cache();
  bool load_honest_full_pair_cache_entry(
      const orbit_wars_honest::NoopView &noop, int32_t episode_step,
      int32_t remaining_steps, int32_t src, int32_t dst, int32_t max_sn,
      bool apply_comet_overlay,
      const std::array<double, kPlanets> &slot_radius,
      const std::array<int32_t, kPlanets> &slot_planet_id,
      const std::array<int32_t, kPlanets> &slot_comet_internal_id) const;
  void store_honest_full_pair_cache_entry(
      int32_t episode_step, int32_t src, int32_t dst, int32_t min_sn,
      int32_t max_sn) const;
  int32_t prune_honest_full_pair_cache_before(int32_t min_episode_step) const;
  void honest_shared_action_mask_impl(
      int32_t episode_step,
      const int32_t *request_data,
      int32_t request_n,
      bool request_data_has_min_sn,
      bool all_geometry,
      torch::Tensor out_action_mask,
      const char *profile_name,
      bool apply_comet_overlay,
      bool store_full_pair_cache) const;
  void log_failed_interception(
      const orbit_wars_honest::NoopView &noop,
      const NoopCachedPlanet *planets_row,
      int32_t planet_count,
      int32_t episode_step,
      int32_t noop_base_frame,
      int32_t remaining_steps,
      int32_t aim_index,
      int32_t src,
      int32_t dst,
      int32_t sn,
      int32_t ship_count,
      int32_t fail_reason,
      const SmallPlanetIdSet &comet_planet_ids,
      const char *source) const;

  void wall_profile_clear() const;
  std::string wall_profile_begin(const char *name) const;
  void wall_profile_end(const char *name, const std::string &path,
                        std::chrono::steady_clock::time_point t0) const;
};
