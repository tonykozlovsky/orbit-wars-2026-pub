#include "cpp_env_live_v2.h"

#include "../honest_shared_features.h"
#include "../io.h"
#include "../kaggle_integration.h"
#include "../library.h"
#include "../masks.h"
#include "../simulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace {

struct TickPlanetMotion {
  double ox = 0.0;
  double oy = 0.0;
  double nx = 0.0;
  double ny = 0.0;
  bool check = false;
};

}  // namespace

CppEnvLiveV2::CppEnvLiveV2(int32_t num_agents, int32_t orbit_instance_id,
                             double ship_speed, int32_t episode_steps,
                             double comet_speed, bool enable_state_trace)
    : CppEnvStaticCacheV2(num_agents, orbit_instance_id, ship_speed,
                           episode_steps, comet_speed),
      enable_state_trace_(enable_state_trace),
      dataset_send_action_mask_scratch_(torch::zeros(
          {kPlanets, kHitClasses},
          torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU))) {}

void CppEnvLiveV2::reset(
    double angular_velocity,
    torch::Tensor planet_rows, int32_t planet_count,
    torch::Tensor orbit_planet_features, torch::Tensor orbit_planet_mask,
    torch::Tensor orbit_planet_pairwise_mask,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor action_taken_index,
    torch::Tensor player_mask) {
  TORCH_CHECK_DISABLED(std::isfinite(angular_velocity), "angular_velocity");
  reset_trace_.clear();
  step_trace_.clear();
  angular_velocity_ = angular_velocity;
  trace_append_to(reset_trace_, "01\treset");
  trace_append_to(reset_trace_, "02\tangular_velocity\t" +
                                   orbit_wars_reset_trace_fmt_double(angular_velocity_));
  planets_ = orbit_wars_honest::external_planet_rows_tensor_to_vector(
      planet_rows, planet_count, "live v2 reset");
  initial_planets_ = planets_;
  fleets_.clear();
  next_fleet_id_ = 0;
  episode_step_ = 0;
  done_ = false;
  comet_groups_.clear();
  comet_pid_set_.clear();
  pending_comet_sync_ = false;
  pending_comet_planet_ids_.clear();
  pending_comet_paths_.clear();
  immutable_planet_prefix_n_ = static_cast<int32_t>(planets_.size());
  TORCH_CHECK_DISABLED(immutable_planet_prefix_n_ > 0 &&
                  immutable_planet_prefix_n_ <= kPlanets,
              "external immutable planet prefix");
  for (int32_t i = 0; i < immutable_planet_prefix_n_; ++i) {
    immutable_planet_prefix_ids_[static_cast<uint32_t>(i)] =
        planets_[static_cast<uint32_t>(i)].id;
  }
  trace_emit_state_to("30", "reset_init", planets_, fleets_, reset_trace_);
  reset_metric_trackers_from_current_state();
  build_noop_cache(angular_velocity_, initial_planets_);
  fill_outputs(orbit_planet_features, orbit_planet_mask, orbit_planet_pairwise_mask,
               orbit_planet_pairwise_features, action_taken_index, player_mask);
}

int32_t CppEnvLiveV2::current_fleet_total_int_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_,
              "current_fleet_total_int_for_owner: bad owner");
  int32_t total = 0;
  for (const Planet &p : planets_) {
    if (p.owner == owner) {
      total += static_cast<int32_t>(p.ships);
    }
  }
  for (const Fleet &f : fleets_) {
    if (f.owner == owner) {
      total += static_cast<int32_t>(f.ships);
    }
  }
  return total;
}

int32_t CppEnvLiveV2::current_planet_count_int_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_,
              "current_planet_count_int_for_owner: bad owner");
  int32_t total = 0;
  for (const Planet &p : planets_) {
    if (p.owner == owner) {
      total += 1;
    }
  }
  return total;
}

double CppEnvLiveV2::current_production_sum_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_,
              "current_production_sum_for_owner: bad owner");
  double total = 0.0;
  for (const Planet &p : planets_) {
    if (p.owner == owner) {
      total += p.production;
    }
  }
  return total;
}

void CppEnvLiveV2::reset_metric_trackers_from_current_state() {
  prev_fleet_total_by_owner_.assign(static_cast<uint32_t>(num_agents_), 0);
  prev_planet_count_by_owner_.assign(static_cast<uint32_t>(num_agents_), 0);
  prev_production_sum_by_owner_.assign(static_cast<uint32_t>(num_agents_), 0.0);
  last_fleet_delta_.assign(static_cast<uint32_t>(num_agents_), 0);
  last_planets_delta_.assign(static_cast<uint32_t>(num_agents_), 0);
  last_production_delta_.assign(static_cast<uint32_t>(num_agents_), 0.0);
  for (int32_t i = 0; i < num_agents_; ++i) {
    const uint32_t k = static_cast<uint32_t>(i);
    prev_fleet_total_by_owner_[k] = current_fleet_total_int_for_owner(i);
    prev_planet_count_by_owner_[k] = current_planet_count_int_for_owner(i);
    prev_production_sum_by_owner_[k] = current_production_sum_for_owner(i);
  }
}

void CppEnvLiveV2::trace_append_to(std::string &buf, const std::string &line) {
  if (!enable_state_trace_) {
    return;
  }
  buf.append(line);
  buf.push_back('\n');
}

void CppEnvLiveV2::trace_emit_planets_to(const char *stage,
                                         const std::vector<Planet> &planets,
                                         std::string &buf) {
  std::vector<uint32_t> ix(planets.size());
  std::iota(ix.begin(), ix.end(), 0);
  std::sort(ix.begin(), ix.end(),
            [&](uint32_t a, uint32_t b) { return planets[a].id < planets[b].id; });
  for (uint32_t k : ix) {
    const Planet &p = planets[k];
    trace_append_to(buf, std::string(stage) + "\tplanet\t" + std::to_string(p.id) +
                             "\t" + std::to_string(p.owner) + "\t" +
                             orbit_wars_reset_trace_fmt_double(p.x) + "\t" +
                             orbit_wars_reset_trace_fmt_double(p.y) + "\t" +
                             orbit_wars_reset_trace_fmt_double(p.radius) + "\t" +
                             orbit_wars_reset_trace_fmt_double(p.ships) + "\t" +
                             orbit_wars_reset_trace_fmt_double(p.production));
  }
}

void CppEnvLiveV2::trace_emit_fleets_to(const char *stage,
                                        const std::vector<Fleet> &fleets,
                                        std::string &buf) {
  std::vector<uint32_t> ix(fleets.size());
  std::iota(ix.begin(), ix.end(), 0);
  std::sort(ix.begin(), ix.end(),
            [&](uint32_t a, uint32_t b) { return fleets[a].id < fleets[b].id; });
  for (uint32_t k : ix) {
    const Fleet &f = fleets[k];
    trace_append_to(buf, std::string(stage) + "\tfleet\t" + std::to_string(f.id) +
                             "\t" + std::to_string(f.owner) + "\t" +
                             orbit_wars_reset_trace_fmt_double(f.x) + "\t" +
                             orbit_wars_reset_trace_fmt_double(f.y) + "\t" +
                             orbit_wars_reset_trace_fmt_double(f.angle) + "\t" +
                             std::to_string(f.from_planet_id) + "\t" +
                             orbit_wars_reset_trace_fmt_double(f.ships));
  }
}

void CppEnvLiveV2::trace_emit_state_to(const char *stage, const char *block,
                                       const std::vector<Planet> &planets,
                                       const std::vector<Fleet> &fleets,
                                       std::string &buf) {
  trace_emit_planets_to((std::string(stage) + "\t" + block).c_str(), planets, buf);
  trace_emit_fleets_to((std::string(stage) + "\t" + block).c_str(), fleets, buf);
}

void CppEnvLiveV2::fill_outputs(torch::Tensor orbit_planet_features,
                                torch::Tensor orbit_planet_mask,
                                torch::Tensor orbit_planet_pairwise_mask,
                                torch::Tensor orbit_planet_pairwise_features,
                                torch::Tensor action_taken_index,
                                torch::Tensor player_mask) const {
  CppEnvStaticCacheV2::fill_policy_obs_from_state_vectors(
      episode_step_, fleets_, planets_, orbit_planet_features, orbit_planet_mask,
      orbit_planet_pairwise_mask, orbit_planet_pairwise_features,
      action_taken_index, player_mask);
}

void CppEnvLiveV2::load_kaggle_comet_groups(
    const SmallPlanetIdSet &comet_planet_ids,
    const CometPathByPlanetId &comet_paths) {
  planets_.resize(static_cast<uint32_t>(immutable_planet_prefix_n_));
  comet_groups_.clear();
  comet_pid_set_.clear();
  if (comet_planet_ids.empty()) {
    TORCH_CHECK_DISABLED(comet_paths.empty(), "Kaggle comet paths present without comet ids");
    return;
  }
  TORCH_CHECK_DISABLED(static_cast<int32_t>(comet_planet_ids.size()) == 4,
              "Kaggle comet state must expose exactly one four-planet comet group");
  TORCH_CHECK_DISABLED(comet_paths.size() == comet_planet_ids.size(),
              "Kaggle comet paths must match comet ids");
  std::vector<int32_t> ids_sorted;
  ids_sorted.reserve(static_cast<uint32_t>(comet_planet_ids.size()));
  comet_planet_ids.append_sorted_ids(ids_sorted);
  OrbitCometGroup g;
  int32_t path_index = -1;
  for (const int32_t pid : ids_sorted) {
    const CometPathInfo *path = comet_paths.find(pid);
    TORCH_CHECK_DISABLED(path != nullptr, "missing Kaggle comet path for planet ", pid);
    assert(path != nullptr);
    if (path_index < 0) {
      path_index = path->path_index;
    } else {
      TORCH_CHECK_DISABLED(path->path_index == path_index,
                  "Kaggle comet group path_index mismatch");
    }
    g.planet_ids.push_back(pid);
    g.paths_kaggle_yx.push_back(path->path_xy);
  }
  TORCH_CHECK_DISABLED(static_cast<int32_t>(g.planet_ids.size()) == 4,
              "Kaggle comet group must list four planets");
  g.path_index = path_index;
  TORCH_CHECK_DISABLED(path_index >= -1, "Kaggle comet path_index must be >= -1");
  g.internal_id = episode_step_ - static_cast<int32_t>(path_index);
  TORCH_CHECK_DISABLED(g.internal_id >= 0, "Kaggle comet internal id must be non-negative");
  for (uint32_t k = 0; k < ids_sorted.size(); ++k) {
    const int32_t pid = ids_sorted[k];
    const CometPathInfo &ci = comet_paths.at(pid);
    Planet cp;
    cp.id = pid;
    cp.comet_internal_id = g.internal_id;
    cp.owner = -1;
    if (ci.path_index < 0) {
      cp.x = -99.0;
      cp.y = -99.0;
    } else {
      TORCH_CHECK_DISABLED(static_cast<uint32_t>(ci.path_index) < ci.path_xy.size(),
                  "Kaggle comet path_index out of range for planet ", pid);
      cp.x = ci.path_xy[static_cast<uint32_t>(ci.path_index)].first;
      cp.y = ci.path_xy[static_cast<uint32_t>(ci.path_index)].second;
      cp.comet_time_before_despawn =
          static_cast<double>(static_cast<int32_t>(ci.path_xy.size()) -
                              static_cast<int32_t>(ci.path_index));
    }
    cp.radius = kCometRadius;
    cp.ships = ci.ships;
    cp.production = kCometProduction;
    planets_.push_back(cp);
  }
  comet_groups_.push_back(std::move(g));
  rebuild_comet_pid_set();
}

void CppEnvLiveV2::update_comets_from_state(py::iterable comet_planet_ids,
                                            py::dict comet_path_by_planet_id) {
  TORCH_CHECK_DISABLED(!initial_planets_.empty(),
              "update_comets_from_state requires reset first");
  TORCH_CHECK_DISABLED(!pending_comet_sync_,
              "update_comets_from_state called with pending comet sync");
  const SmallPlanetIdSet comet_ids =
      orbit_wars_comet_planet_ids_from_python(comet_planet_ids);
  const CometPathByPlanetId comet_paths =
      orbit_wars_comet_paths_from_python(comet_path_by_planet_id);
  pending_comet_sync_ = true;
  pending_comet_planet_ids_ = comet_ids;
  pending_comet_paths_ = comet_paths;
  if (!pending_comet_planet_ids_.empty()) {
    TORCH_CHECK_DISABLED(static_cast<int32_t>(pending_comet_planet_ids_.size()) == 4,
                "Kaggle comet state must expose exactly one four-planet comet group");
    TORCH_CHECK_DISABLED(pending_comet_paths_.size() == pending_comet_planet_ids_.size(),
                "Kaggle comet paths must match comet ids");
    std::vector<int32_t> ids_sorted;
    ids_sorted.reserve(static_cast<uint32_t>(pending_comet_planet_ids_.size()));
    pending_comet_planet_ids_.append_sorted_ids(ids_sorted);
    std::vector<int32_t> planet_ids;
    std::vector<std::vector<std::pair<double, double>>> paths_kaggle_yx;
    int32_t path_index = -1;
    for (const int32_t pid : ids_sorted) {
      const CometPathInfo *path = pending_comet_paths_.find(pid);
      TORCH_CHECK_DISABLED(path != nullptr,
                  "missing Kaggle comet path for planet ", pid);
      assert(path != nullptr);
      if (path_index < 0) {
        path_index = path->path_index;
      } else {
        TORCH_CHECK_DISABLED(path->path_index == path_index,
                    "Kaggle comet group path_index mismatch");
      }
      planet_ids.push_back(pid);
      paths_kaggle_yx.push_back(path->path_xy);
    }
    TORCH_CHECK_DISABLED(path_index >= -1, "Kaggle comet path_index must be >= -1");
    const int32_t internal_id = episode_step_ - static_cast<int32_t>(path_index);
    TORCH_CHECK_DISABLED(internal_id >= 0, "Kaggle comet internal id must be non-negative");
  } else {
    TORCH_CHECK_DISABLED(pending_comet_paths_.empty(),
                "Kaggle comet paths present without comet ids");
  }
}

void CppEnvLiveV2::fill_noop_cache_fixed_planet_row(NoopCachedPlanet *row) const {
  TORCH_CHECK_DISABLED(row != nullptr, "fill_noop_cache_fixed_planet_row");
  for (int32_t i = 0; i < kPlanets; ++i) {
    row[i] = NoopCachedPlanet{};
  }
  for (int32_t i = 0; i < immutable_planet_prefix_n_; ++i) {
    row[i] = noop_cached_planet_from_planet(planets_[static_cast<uint32_t>(i)]);
  }
  if (comet_groups_.empty()) {
    return;
  }
  TORCH_CHECK_DISABLED(static_cast<int32_t>(planets_.size()) >= immutable_planet_prefix_n_ + 4,
              "fill_noop_cache_fixed_planet_row: missing comet planets");
  for (int32_t k = 0; k < 4; ++k) {
    row[immutable_planet_prefix_n_ + k] =
        noop_cached_planet_from_planet(
            planets_[static_cast<uint32_t>(immutable_planet_prefix_n_ + k)]);
  }
}

void CppEnvLiveV2::assert_planets_match_noop_cache(int32_t step_index) const {
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "assert_planets_match_noop_cache: empty noop cache");
  const int32_t n_frames = noop_trajectory_length();
  TORCH_CHECK_DISABLED(step_index >= 0 && step_index < n_frames,
              "assert_planets_match_noop_cache: bad step_index ", step_index,
              " n_frames=", n_frames);
  const NoopCachedPlanet *cached_row =
      noop_cached_planets_flat_.data() +
      static_cast<uint32_t>(step_index) * static_cast<uint32_t>(kPlanets);
  std::array<NoopCachedPlanet, kPlanets> cur{};
  fill_noop_cache_fixed_planet_row(cur.data());
  for (int32_t i = 0; i < kPlanets; ++i) {
    const NoopCachedPlanet &a = cached_row[static_cast<uint32_t>(i)];
    const NoopCachedPlanet &b = cur[static_cast<uint32_t>(i)];
    TORCH_CHECK_DISABLED(a.id == b.id && a.comet_internal_id == b.comet_internal_id &&
                    a.comet_time_before_despawn == b.comet_time_before_despawn &&
                    a.x == b.x && a.y == b.y && a.radius == b.radius &&
                    a.production == b.production,
                "assert_planets_match_noop_cache: mismatch fixed_slot=", i,
                " step_index=", step_index,
                " episode_step_=", episode_step_,
                " immutable_planet_prefix_n_=", immutable_planet_prefix_n_,
                " slot_below_immutable_prefix=", (i < immutable_planet_prefix_n_),
                " noop_groups=", static_cast<int32_t>(comet_groups_.size()));
  }
}

void CppEnvLiveV2::step(torch::Tensor action_classes,
                        torch::Tensor orbit_planet_features,
                        torch::Tensor orbit_planet_mask,
                        torch::Tensor orbit_planet_pairwise_mask,
                        torch::Tensor orbit_planet_pairwise_features,
                        torch::Tensor action_taken_index,
                        torch::Tensor player_mask) {
  if (wall_profile_enabled_) {
    set_wall_profile_enabled(false);
    set_wall_profile_enabled(true);
  }
  WallProfileSpan step_profile(this, "cpp_env_step");
  TORCH_CHECK_DISABLED(action_classes.device().is_cpu(),
              "action_classes: expected CPU tensor");
  TORCH_CHECK_DISABLED(action_classes.dtype() == torch::kInt32 ||
                  action_classes.dtype() == torch::kInt64,
              "action_classes: expected int32 or int64");
  action_classes = action_classes.to(torch::kInt32).contiguous();
  TORCH_CHECK_DISABLED(action_classes.sizes() == torch::IntArrayRef({num_agents_, kPlanets}));
  step_trace_.clear();
  const int32_t physics_step = episode_step_;
  const int32_t next_episode_step = episode_step_ + 1;
  trace_append_to(step_trace_, "10\tepisode_step\t" + std::to_string(next_episode_step));
  std::vector<PlannedLaunch> planned_launches;
  {
    WallProfileSpan profile(this, "plan_launches_from_classes");
    planned_launches = plan_launches_from_classes(action_classes);
  }
  {
    WallProfileSpan profile(this, "interpreter_pre_launch");
    interpreter_pre_launch();
  }
  {
    WallProfileSpan profile(this, "trace_after_pre_launch");
    trace_emit_state_to("31", "step_after_pre_launch", planets_, fleets_, step_trace_);
  }
  {
    WallProfileSpan profile(this, "launch_planned_fleets");
    launch_planned_fleets(planned_launches);
  }
  {
    WallProfileSpan profile(this, "trace_after_launch");
    trace_emit_state_to("31", "step_after_launch", planets_, fleets_, step_trace_);
  }
  {
    WallProfileSpan profile(this, "interpreter_simulation_after_launch");
    interpreter_simulation_after_launch(step_trace_, physics_step);
  }
  episode_step_ = next_episode_step;
  {
    WallProfileSpan profile(this, "refresh_delta_metrics");
    refresh_delta_metrics_from_current_state();
  }
  {
    WallProfileSpan profile(this, "trace_final_planets");
    trace_emit_planets_to("11", planets_, step_trace_);
  }
  {
    WallProfileSpan profile(this, "fill_outputs");
    fill_outputs(orbit_planet_features, orbit_planet_mask, orbit_planet_pairwise_mask,
                 orbit_planet_pairwise_features, action_taken_index, player_mask);
  }
  {
    WallProfileSpan profile(this, "fill_action_taken_index");
    fill_action_taken_index_from_classes(action_classes, num_agents_, action_taken_index);
  }
}

std::string CppEnvLiveV2::reset_trace_get() const { return reset_trace_; }
std::string CppEnvLiveV2::step_trace_get() const { return step_trace_; }
bool CppEnvLiveV2::orbit_episode_terminal() const { return done_; }
double CppEnvLiveV2::angular_velocity() const { return angular_velocity_; }
double CppEnvLiveV2::ship_speed() const { return ship_speed_; }
int32_t CppEnvLiveV2::episode_step() const { return episode_step_; }
int32_t CppEnvLiveV2::kaggle_observation_step() const { return episode_step_; }

int32_t CppEnvLiveV2::fleet_ship_total_int_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "fleet_ship_total_int_for_owner: bad owner");
  return current_fleet_total_int_for_owner(owner);
}

bool CppEnvLiveV2::player_alive_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "player_alive_for_owner: bad owner");
  return current_planet_count_int_for_owner(owner) > 0 ||
         current_fleet_total_int_for_owner(owner) > 0;
}

int32_t CppEnvLiveV2::planet_count_int_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "planet_count_int_for_owner: bad owner");
  return current_planet_count_int_for_owner(owner);
}

double CppEnvLiveV2::production_sum_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "production_sum_for_owner: bad owner");
  return current_production_sum_for_owner(owner);
}

double CppEnvLiveV2::game_result_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "game_result_for_owner: bad owner");
  if (!done_) {
    return 0.0;
  }
  int32_t alive_count = 0;
  int32_t alive_winner = -1;
  for (int32_t i = 0; i < num_agents_; ++i) {
    const bool alive = player_alive_for_owner(i);
    if (alive) {
      ++alive_count;
      alive_winner = i;
    }
  }
  if (alive_count <= 1) {
    if (alive_count == 0) {
      return -1.0;
    }
    return owner == alive_winner ? 1.0 : -1.0;
  }
  std::vector<int32_t> fleet_by_owner;
  fleet_by_owner.reserve(static_cast<uint32_t>(num_agents_));
  int32_t max_fleet = 0;
  for (int32_t i = 0; i < num_agents_; ++i) {
    const int32_t fleet_i = fleet_ship_total_int_for_owner(i);
    fleet_by_owner.push_back(fleet_i);
    max_fleet = std::max(max_fleet, fleet_i);
  }
  if (max_fleet == 0) {
    return 0.0;
  }
  int32_t winner_count = 0;
  for (int32_t i = 0; i < num_agents_; ++i) {
    if (fleet_by_owner[static_cast<uint32_t>(i)] == max_fleet) {
      ++winner_count;
    }
  }
  TORCH_CHECK_DISABLED(winner_count >= 1 && winner_count <= num_agents_, "winner_count");
  if (winner_count != 1) {
    return 0.0;
  }
  return fleet_by_owner[static_cast<uint32_t>(owner)] == max_fleet ? 1.0 : -1.0;
}

double CppEnvLiveV2::fleet_delta_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "fleet_delta_for_owner: bad owner");
  return static_cast<double>(last_fleet_delta_[static_cast<uint32_t>(owner)]);
}

double CppEnvLiveV2::planets_delta_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "planets_delta_for_owner: bad owner");
  return static_cast<double>(last_planets_delta_[static_cast<uint32_t>(owner)]);
}

double CppEnvLiveV2::production_delta_for_owner(int32_t owner) const {
  TORCH_CHECK_DISABLED(0 <= owner && owner < num_agents_, "production_delta_for_owner: bad owner");
  return last_production_delta_[static_cast<uint32_t>(owner)];
}

py::tuple CppEnvLiveV2::step_metric_tensors() const {
  TORCH_CHECK_DISABLED(static_cast<int32_t>(last_fleet_delta_.size()) == num_agents_,
              "last_fleet_delta size");
  TORCH_CHECK_DISABLED(static_cast<int32_t>(last_planets_delta_.size()) == num_agents_,
              "last_planets_delta size");
  TORCH_CHECK_DISABLED(static_cast<int32_t>(last_production_delta_.size()) == num_agents_,
              "last_production_delta size");
  torch::Tensor fleet_delta =
      torch::empty({num_agents_}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor planets_delta =
      torch::empty({num_agents_}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor production_delta =
      torch::empty({num_agents_}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor fleet_total =
      torch::empty({num_agents_}, torch::TensorOptions().dtype(torch::kInt32));
  float *fleet_delta_p = fleet_delta.data_ptr<float>();
  float *planets_delta_p = planets_delta.data_ptr<float>();
  float *production_delta_p = production_delta.data_ptr<float>();
  int32_t *fleet_total_p = fleet_total.data_ptr<int32_t>();
  for (int32_t i = 0; i < num_agents_; ++i) {
    const uint32_t idx = static_cast<uint32_t>(i);
    fleet_delta_p[i] = static_cast<float>(last_fleet_delta_[idx]);
    planets_delta_p[i] = static_cast<float>(last_planets_delta_[idx]);
    production_delta_p[i] = static_cast<float>(last_production_delta_[idx]);
    fleet_total_p[i] = current_fleet_total_int_for_owner(i);
  }
  return py::make_tuple(fleet_delta, planets_delta, production_delta, fleet_total);
}

py::list CppEnvLiveV2::tape_kaggle_planets_rows() const {
  py::list rows;
  for (const Planet &p : planets_) {
    py::list row;
    row.append(py::cast(p.id));
    row.append(py::cast(p.owner));
    row.append(py::cast(p.x));
    row.append(py::cast(p.y));
    row.append(py::cast(p.radius));
    row.append(py::cast(static_cast<int32_t>(p.ships)));
    row.append(py::cast(p.production));
    rows.append(row);
  }
  return rows;
}

py::list CppEnvLiveV2::tape_kaggle_fleets_rows() const {
  py::list rows;
  for (const Fleet &f : fleets_) {
    py::list row;
    row.append(py::cast(f.id));
    row.append(py::cast(f.owner));
    row.append(py::cast(f.x));
    row.append(py::cast(f.y));
    row.append(py::cast(f.angle));
    row.append(py::cast(f.from_planet_id));
    row.append(py::cast(static_cast<int32_t>(f.ships)));
    rows.append(row);
  }
  return rows;
}

py::tuple CppEnvLiveV2::comet_mask_inputs_py() const {
  py::dict path_dict;
  py::list ids;
  for (const auto &g : comet_groups_) {
    for (uint32_t i = 0; i < g.planet_ids.size(); ++i) {
      const int32_t pid = g.planet_ids[i];
      ids.append(py::cast(pid));
      py::list path_xy;
      for (const auto &pt : g.paths_kaggle_yx[i]) {
        py::list p;
        p.append(pt.first);
        p.append(pt.second);
        path_xy.append(p);
      }
      path_dict[py::int_(pid)] =
          py::make_tuple(static_cast<int32_t>(g.path_index), path_xy);
    }
  }
  return py::make_tuple(ids, path_dict);
}

void CppEnvLiveV2::refresh_delta_metrics_from_current_state() {
  TORCH_CHECK_DISABLED(static_cast<int32_t>(prev_fleet_total_by_owner_.size()) == num_agents_);
  TORCH_CHECK_DISABLED(static_cast<int32_t>(prev_planet_count_by_owner_.size()) == num_agents_);
  TORCH_CHECK_DISABLED(static_cast<int32_t>(prev_production_sum_by_owner_.size()) == num_agents_);
  TORCH_CHECK_DISABLED(static_cast<int32_t>(last_fleet_delta_.size()) == num_agents_);
  TORCH_CHECK_DISABLED(static_cast<int32_t>(last_planets_delta_.size()) == num_agents_);
  TORCH_CHECK_DISABLED(static_cast<int32_t>(last_production_delta_.size()) == num_agents_);
  for (int32_t i = 0; i < num_agents_; ++i) {
    const uint32_t k = static_cast<uint32_t>(i);
    const int32_t cur_fleet = current_fleet_total_int_for_owner(i);
    const int32_t cur_planets = current_planet_count_int_for_owner(i);
    const double cur_production = current_production_sum_for_owner(i);
    last_fleet_delta_[k] = cur_fleet - prev_fleet_total_by_owner_[k];
    last_planets_delta_[k] = cur_planets - prev_planet_count_by_owner_[k];
    last_production_delta_[k] = cur_production - prev_production_sum_by_owner_[k];
    prev_fleet_total_by_owner_[k] = cur_fleet;
    prev_planet_count_by_owner_[k] = cur_planets;
    prev_production_sum_by_owner_[k] = cur_production;
  }
}

std::vector<CppEnvLiveV2::PlannedLaunch> CppEnvLiveV2::plan_launches_from_classes(
    torch::Tensor action_classes) const {
  const auto available = dataset_send_action_mask_scratch_.accessor<int8_t, 2>();
  torch::Tensor send_ships_tensor = honest_shared_send_ships_last();
  const auto send_ships_last = send_ships_tensor.accessor<int32_t, 2>();
  const auto cls = action_classes.accessor<int32_t, 2>();
  const int32_t n = static_cast<int32_t>(planets_.size());
  std::vector<PlannedLaunch> launches;
  for (int32_t player_id = 0; player_id < num_agents_; ++player_id) {
    for (int32_t i = 0; i < n; ++i) {
      const Planet &src = planets_[static_cast<uint32_t>(i)];
      const int32_t c = cls[player_id][i];
      TORCH_CHECK_DISABLED(0 <= c && c < kMoveClasses,
                  "plan_launches_from_classes: action class out of range");
      const int32_t j = c / kMoveClassesPerTarget;
      const int32_t sn = c % kMoveClassesPerTarget;
      if (sn == kMoveClassNoopSubindex && j == i) {
        continue;
      }
      TORCH_CHECK(
          move_subindex_is_send_action(sn),
          "plan_launches_from_classes: move subaction has no execution semantics");
      TORCH_CHECK_DISABLED(move_subindex_is_send_action(sn) && j < n && j != i,
                  "plan_launches_from_classes: invalid non-noop class");
      TORCH_CHECK(j < n && j != i,
                  "plan_launches_from_classes: invalid send destination");
      TORCH_CHECK_DISABLED(src.owner == player_id && src.ships > 0.0,
                  "plan_launches_from_classes: unavailable source planet");
      const int32_t src_ships = static_cast<int32_t>(src.ships);
      TORCH_CHECK_DISABLED(static_cast<double>(src_ships) == src.ships,
                  "plan_launches_from_classes: source ships must be integer");
      const int32_t hit_cls = j * kHitClassesPerTarget + sn;
      TORCH_CHECK_DISABLED(available[i][hit_cls] > 0,
                           "plan_launches_from_classes: unavailable action");
      const int32_t n_send = send_ships_last[i][hit_cls];
      TORCH_CHECK_DISABLED(n_send > 0 && n_send <= src_ships,
                  "plan_launches_from_classes: send ships");
      const double angle = honest_shared_angle_last_for_action_class(i, c);
      TORCH_CHECK_DISABLED(std::isfinite(angle),
                  "plan_launches_from_classes: cached action angle");
      launches.push_back(PlannedLaunch{player_id, src.id, angle, n_send});
    }
  }
  return launches;
}

void CppEnvLiveV2::launch_planned_fleets(const std::vector<PlannedLaunch> &launches) {
  for (const PlannedLaunch &launch : launches) {
    Planet *from_planet = nullptr;
    for (Planet &p : planets_) {
      if (p.id == launch.from_planet_id) {
        from_planet = &p;
        break;
      }
    }
    TORCH_CHECK_DISABLED(from_planet != nullptr,
                "launch_planned_fleets: source planet missing");
    TORCH_CHECK_DISABLED(from_planet->owner == launch.player_id,
                "launch_planned_fleets: source owner mismatch");
    TORCH_CHECK_DISABLED(launch.ships > 0, "launch_planned_fleets: ships");
    TORCH_CHECK_DISABLED(from_planet->ships >= static_cast<double>(launch.ships),
                "launch_planned_fleets: insufficient source ships");
    from_planet->ships -= static_cast<double>(launch.ships);
    fleets_.push_back(Fleet{next_fleet_id_++,
                            launch.player_id,
                            from_planet->x + std::cos(launch.angle) * (from_planet->radius + 0.1),
                            from_planet->y + std::sin(launch.angle) * (from_planet->radius + 0.1),
                            launch.angle,
                            from_planet->id,
                            static_cast<double>(launch.ships)});
  }
}

void CppEnvLiveV2::launch_fleets_from_kaggle_actions(py::iterable actions_by_player) {
  int32_t player_id = 0;
  for (auto seat_obj : actions_by_player) {
    TORCH_CHECK_DISABLED(player_id < num_agents_, "kaggle actions seat overflow");
    py::iterable seat_moves = py::reinterpret_borrow<py::iterable>(seat_obj);
    for (auto move_obj : seat_moves) {
      py::sequence move = py::reinterpret_borrow<py::sequence>(move_obj);
      TORCH_CHECK_DISABLED(py::len(move) == 3, "kaggle move must be [from_planet_id, angle, ships]");
      const int32_t from_planet_id = py::cast<int32_t>(move[0]);
      const double angle = py::cast<double>(move[1]);
      const int32_t ships = py::cast<int32_t>(move[2]);
      TORCH_CHECK_DISABLED(std::isfinite(angle), "kaggle move angle must be finite");
      if (ships <= 0) {
        continue;
      }
      Planet *from_planet = nullptr;
      for (auto &p : planets_) {
        if (p.id == from_planet_id) {
          from_planet = &p;
          break;
        }
      }
      if (from_planet == nullptr) {
        continue;
      }
      if (from_planet->owner != player_id ||
          from_planet->ships < static_cast<double>(ships)) {
        continue;
      }
      from_planet->ships -= static_cast<double>(ships);
      fleets_.push_back(Fleet{next_fleet_id_++,
                              player_id,
                              from_planet->x + std::cos(angle) * (from_planet->radius + 0.1),
                              from_planet->y + std::sin(angle) * (from_planet->radius + 0.1),
                              angle,
                              from_planet->id,
                              static_cast<double>(ships)});
    }
    ++player_id;
  }
  TORCH_CHECK_DISABLED(player_id == num_agents_, "kaggle actions seat count mismatch");
}

void CppEnvLiveV2::interpreter_pre_launch() {
  remove_expired_comets_before_launch();
  apply_pending_comets_in_pre_launch();
}

void CppEnvLiveV2::apply_pending_comets_in_pre_launch() {
  if (!pending_comet_sync_) {
    return;
  }
  load_kaggle_comet_groups(pending_comet_planet_ids_, pending_comet_paths_);
  if (!comet_groups_.empty()) {
    const OrbitCometGroup &g = comet_groups_[0];
    update_comet_in_noop_cache(episode_step_, g.path_index, g.internal_id,
                               g.planet_ids, g.paths_kaggle_yx);
  } else {
    build_noop_cache(angular_velocity_, initial_planets_);
  }
  pending_comet_sync_ = false;
  pending_comet_planet_ids_.clear();
  pending_comet_paths_.clear();
}

void CppEnvLiveV2::rebuild_comet_pid_set() {
  comet_pid_set_.clear();
  for (const auto &g : comet_groups_) {
    for (int32_t pid : g.planet_ids) {
      comet_pid_set_.insert(pid);
    }
  }
}

void CppEnvLiveV2::erase_planets_and_trim_comets(
    const SmallPlanetIdSet &dead) {
  planets_.erase(std::remove_if(planets_.begin(), planets_.end(),
                                [&](const Planet &p) { return dead.contains(p.id); }),
                 planets_.end());
  initial_planets_.erase(
      std::remove_if(initial_planets_.begin(), initial_planets_.end(),
                     [&](const Planet &p) { return dead.contains(p.id); }),
      initial_planets_.end());
  std::vector<OrbitCometGroup> ng;
  for (auto &g : comet_groups_) {
    OrbitCometGroup h;
    h.path_index = g.path_index;
    h.internal_id = g.internal_id;
    for (uint32_t i = 0; i < g.planet_ids.size(); ++i) {
      if (dead.contains(g.planet_ids[i])) {
        continue;
      }
      h.planet_ids.push_back(g.planet_ids[i]);
      h.paths_kaggle_yx.push_back(g.paths_kaggle_yx[i]);
    }
    if (!h.planet_ids.empty()) {
      ng.push_back(std::move(h));
    }
  }
  comet_groups_ = std::move(ng);
  rebuild_comet_pid_set();
}

void CppEnvLiveV2::remove_expired_comets_before_launch() {
  SmallPlanetIdSet expired;
  for (const auto &g : comet_groups_) {
    const int32_t idx = g.path_index;
    for (uint32_t i = 0; i < g.planet_ids.size(); ++i) {
      if (idx >= static_cast<int32_t>(g.paths_kaggle_yx[i].size())) {
        expired.insert(g.planet_ids[i]);
      }
    }
  }
  if (!expired.empty()) {
    erase_planets_and_trim_comets(expired);
  }
}

void CppEnvLiveV2::interpreter_simulation_after_launch(std::string &trace_buf,
                                                       int32_t physics_step) {
  if (done_) {
    return;
  }
  TORCH_CHECK_DISABLED(physics_step >= 0, "interpreter_simulation_after_launch: physics_step");
  assert(planets_.size() <= static_cast<uint32_t>(kPlanets));
  {
    WallProfileSpan profile(this, "production");
    for (Planet &p : planets_) {
      if (p.owner != -1) {
        p.ships += p.production;
      }
    }
  }
  {
    WallProfileSpan profile(this, "trace_after_production");
    trace_emit_state_to("31", "step_after_production", planets_, fleets_, trace_buf);
  }

  std::array<TickPlanetMotion, kPlanets> planet_paths{};
  std::array<uint8_t, kPlanets> planet_path_valid{};
  {
    WallProfileSpan profile(this, "build_planet_paths");
    for (uint32_t pi = 0; pi < planets_.size(); ++pi) {
      Planet &p = planets_[pi];
      if (comet_pid_set_.contains(p.id)) {
        continue;
      }
      const Planet *init = initial_planet_by_id(p.id);
      TickPlanetMotion m;
      m.ox = p.x;
      m.oy = p.y;
      m.nx = p.x;
      m.ny = p.y;
      m.check = true;
      if (init != nullptr) {
        const double dx = init->x - kCenter;
        const double dy = init->y - kCenter;
        const double r_orb = std::sqrt(dx * dx + dy * dy);
        if (r_orb + p.radius < kRotationRadiusLimit) {
          const double initial_angle = std::atan2(dy, dx);
          const double current_angle =
              initial_angle + angular_velocity_ * static_cast<double>(physics_step);
          m.nx = kCenter + r_orb * std::cos(current_angle);
          m.ny = kCenter + r_orb * std::sin(current_angle);
        }
      }
      planet_paths[pi] = m;
      planet_path_valid[pi] = 1;
    }
  }

  std::vector<int32_t> expired_after_tick;
  {
    WallProfileSpan profile(this, "advance_comet_paths");
    for (OrbitCometGroup &gr : comet_groups_) {
      gr.path_index += 1;
      const int32_t idx = gr.path_index;
      for (uint32_t ii = 0; ii < gr.planet_ids.size(); ++ii) {
        const int32_t pid = gr.planet_ids[ii];
        uint32_t planet_slot = planets_.size();
        for (uint32_t pi = 0; pi < planets_.size(); ++pi) {
          if (planets_[pi].id == pid) {
            planet_slot = pi;
            break;
          }
        }
        if (planet_slot == planets_.size()) {
          continue;
        }
        Planet &pl = planets_[planet_slot];
        const auto &p_path = gr.paths_kaggle_yx[ii];
        TickPlanetMotion m;
        m.ox = pl.x;
        m.oy = pl.y;
        if (idx >= static_cast<int32_t>(p_path.size())) {
          expired_after_tick.push_back(pid);
          m.nx = m.ox;
          m.ny = m.oy;
          m.check = true;
        } else {
          m.nx = p_path[static_cast<uint32_t>(idx)].first;
          m.ny = p_path[static_cast<uint32_t>(idx)].second;
          m.check = (m.ox >= 0.0);
          pl.comet_time_before_despawn =
              static_cast<double>(static_cast<int32_t>(p_path.size()) -
                                  static_cast<int32_t>(idx));
        }
        planet_paths[planet_slot] = m;
        planet_path_valid[planet_slot] = 1;
      }
    }
  }

  std::vector<int32_t> remove_fleet;
  CombatBySlot combat;
  auto append_combat = [&](uint32_t planet_slot, const Fleet &f) {
    combat[planet_slot].push_back(f);
  };

  int32_t n_fleet = 0;
  {
    WallProfileSpan profile(this, "fleet_movement_and_hit_checks");
    n_fleet = static_cast<int32_t>(fleets_.size());
    for (int32_t fi = 0; fi < n_fleet; ++fi) {
      Fleet &f = fleets_[fi];
      const double speed = fleet_speed(static_cast<int32_t>(f.ships));
      const double old_x = f.x;
      const double old_y = f.y;
      f.x += std::cos(f.angle) * speed;
      f.y += std::sin(f.angle) * speed;
      bool hit = false;
      for (uint32_t pi = 0; pi < planets_.size(); ++pi) {
        const Planet &p = planets_[pi];
        if (planet_path_valid[pi] == 0 || !planet_paths[pi].check) {
          continue;
        }
        const TickPlanetMotion &m = planet_paths[pi];
        if (orbit_wars_swept_pair_hit(old_x, old_y, f.x, f.y,
                                      m.ox, m.oy, m.nx, m.ny, p.radius)) {
          append_combat(pi, f);
          remove_fleet.push_back(fi);
          hit = true;
          break;
        }
      }
      if (hit) {
        continue;
      }
      if (!(0.0 <= f.x && f.x <= kBoardSize && 0.0 <= f.y && f.y <= kBoardSize)) {
        remove_fleet.push_back(fi);
        continue;
      }
      if (point_to_segment_distance_sq(kCenter, kCenter, old_x, old_y, f.x, f.y) <
          (kSunRadius * kSunRadius)) {
        remove_fleet.push_back(fi);
      }
    }
  }
  {
    WallProfileSpan profile(this, "trace_after_fleet_movement");
    trace_emit_state_to("31", "step_after_fleet_movement", planets_, fleets_, trace_buf);
  }

  {
    WallProfileSpan profile(this, "apply_planet_motion");
    for (uint32_t pi = 0; pi < planets_.size(); ++pi) {
      if (planet_path_valid[pi] != 0) {
        planets_[pi].x = planet_paths[pi].nx;
        planets_[pi].y = planet_paths[pi].ny;
      }
    }
  }
  {
    WallProfileSpan profile(this, "trace_after_planet_motion_apply");
    trace_emit_state_to("31", "step_after_planet_motion_apply", planets_, fleets_, trace_buf);
  }

  {
    WallProfileSpan profile(this, "expired_comet_removal");
    if (!expired_after_tick.empty()) {
      SmallPlanetIdSet es;
      for (int32_t pid : expired_after_tick) {
        es.insert(pid);
      }
      erase_planets_and_trim_comets(es);
    }
  }
  {
    WallProfileSpan profile(this, "trace_after_expired_comet_removal");
    trace_emit_state_to("31", "step_after_expired_comet_removal", planets_, fleets_, trace_buf);
  }

  {
    WallProfileSpan profile(this, "prune_fleets");
    std::sort(remove_fleet.begin(), remove_fleet.end());
    remove_fleet.erase(std::unique(remove_fleet.begin(), remove_fleet.end()), remove_fleet.end());
    std::vector<Fleet> kept;
    for (int32_t fi = 0; fi < n_fleet; ++fi) {
      if (!std::binary_search(remove_fleet.begin(), remove_fleet.end(), fi)) {
        kept.push_back(fleets_[fi]);
      }
    }
    fleets_ = std::move(kept);
  }
  {
    WallProfileSpan profile(this, "trace_after_fleet_prune");
    trace_emit_state_to("31", "step_after_fleet_prune", planets_, fleets_, trace_buf);
    trace_emit_state_to("31", "step_pre_combat", planets_, fleets_, trace_buf);
  }

  {
    WallProfileSpan profile(this, "resolve_combat");
    resolve_combat_keyed(combat);
  }
  {
    WallProfileSpan profile(this, "update_done");
    update_done(physics_step);
  }
  {
    WallProfileSpan profile(this, "trace_after_combat");
    trace_emit_state_to("31", "step_after_combat", planets_, fleets_, trace_buf);
  }
}

double CppEnvLiveV2::fleet_speed(int32_t ship_count) const {
  TORCH_CHECK_DISABLED(ship_count > 0, "ship_count");
  const double speed = 1.0 + (ship_speed_ - 1.0) *
                                 std::pow(std::log(static_cast<double>(ship_count)) /
                                              std::log(1000.0),
                                          1.5);
  return std::min(speed, ship_speed_);
}

const Planet *CppEnvLiveV2::initial_planet_by_id(int32_t id) const {
  for (const Planet &p : initial_planets_) {
    if (p.id == id) {
      return &p;
    }
  }
  return nullptr;
}

void CppEnvLiveV2::resolve_combat_keyed(const CombatBySlot &combat) {
  for (uint32_t pi = 0; pi < planets_.size(); ++pi) {
    Planet &planet = planets_[pi];
    const std::vector<Fleet> &slot_combat = combat[pi];
    if (slot_combat.empty()) {
      continue;
    }
    std::vector<std::pair<int32_t, double>> player_ships_pairs;
    for (const Fleet &f : slot_combat) {
      TORCH_CHECK_DISABLED(0 <= f.owner && f.owner < num_agents_, "fleet owner");
      auto jt = std::find_if(player_ships_pairs.begin(), player_ships_pairs.end(),
                             [&](const std::pair<int32_t, double> &pr) {
                               return pr.first == f.owner;
                             });
      if (jt == player_ships_pairs.end()) {
        player_ships_pairs.push_back({f.owner, f.ships});
      } else {
        jt->second += f.ships;
      }
    }
    std::stable_sort(player_ships_pairs.begin(), player_ships_pairs.end(),
                     [](const std::pair<int32_t, double> &a,
                        const std::pair<int32_t, double> &b) {
                       return a.second > b.second;
                     });
    const int32_t top_player = player_ships_pairs[0].first;
    const double top_ships = player_ships_pairs[0].second;
    double survivor_ships = top_ships;
    int32_t survivor_owner = top_player;
    if (static_cast<int32_t>(player_ships_pairs.size()) > 1) {
      const double second_ships = player_ships_pairs[1].second;
      survivor_ships = top_ships - second_ships;
      if (player_ships_pairs[0].second == player_ships_pairs[1].second) {
        survivor_ships = 0.0;
      }
      survivor_owner = survivor_ships > 0.0 ? top_player : -1;
    }
    if (survivor_ships > 0.0) {
      if (planet.owner == survivor_owner) {
        planet.ships += survivor_ships;
      } else {
        planet.ships -= survivor_ships;
        if (planet.ships < 0.0) {
          planet.owner = survivor_owner;
          planet.ships = std::abs(planet.ships);
        }
      }
    }
  }
}

void CppEnvLiveV2::update_done(int32_t physics_step) {
  if (physics_step >= episode_steps_ - 2) {
    done_ = true;
    return;
  }
  int32_t alive_count = 0;
  for (int32_t i = 0; i < num_agents_; ++i) {
    alive_count += player_alive_for_owner(i) ? 1 : 0;
  }
  done_ = alive_count <= 1;
}

void CppEnvLiveV2::honest_shared_send_all_hit_mask(
    torch::Tensor out_hit_mask) const {
  CppEnvStaticCacheV2::send_action_hit_mask_from_state(
      episode_step_, fleets_, planets_, out_hit_mask);
}

void CppEnvLiveV2::fill_available_action_mask(
    torch::Tensor out_action_mask) const {
  const int32_t planet_count = static_cast<int32_t>(planets_.size());
  TORCH_CHECK_DISABLED(0 < planet_count && planet_count <= kPlanets,
              "available action mask: planet_count");
  CppEnvStaticCacheV2::send_action_hit_mask_from_state(
      episode_step_, fleets_, planets_, dataset_send_action_mask_scratch_);
  CppEnvStaticCacheV2::fill_available_action_mask_from_hit_mask(
      planets_, dataset_send_action_mask_scratch_, out_action_mask);
}

double CppEnvLiveV2::honest_shared_angle(int32_t src_slot, int32_t dst_slot,
                                         int32_t ship_count) const {
  const double angle = honest_shared_angle_or_nan(src_slot, dst_slot, ship_count);
  TORCH_CHECK_DISABLED(std::isfinite(angle), "honest_shared_angle: dynamic intercept failed");
  return angle;
}

double CppEnvLiveV2::honest_shared_angle_or_nan(int32_t src_slot, int32_t dst_slot,
                                                int32_t ship_count) const {
  return CppEnvStaticCacheV2::honest_shared_angle_or_nan_from_state(
      episode_step_, planets_, src_slot, dst_slot, ship_count);
}

torch::Tensor CppEnvLiveV2::honest_shared_intercept_trace(
    int32_t src_slot, int32_t dst_slot, int32_t ship_subindex) const {
  return CppEnvStaticCacheV2::honest_shared_intercept_trace_from_state(
      episode_step_, planets_, src_slot, dst_slot, ship_subindex);
}

torch::Tensor CppEnvLiveV2::fleet_arrivals_from_state(int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_arrivals_from_state_vectors(
      episode_step_, fleets_, horizon);
}

torch::Tensor CppEnvLiveV2::fleet_arrivals_from_rows(torch::Tensor fleet_rows,
                                                     int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_arrivals_from_rows(
      episode_step_, fleet_rows, horizon);
}

torch::Tensor CppEnvLiveV2::fleet_arrival_features_from_state(int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_arrival_features_from_state_vectors(
      episode_step_, fleets_, planets_, horizon);
}

torch::Tensor CppEnvLiveV2::fleet_arrival_features_from_rows(
    torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_arrival_features_from_rows(
      episode_step_, fleet_rows, planet_rows, planet_count, horizon);
}

void CppEnvLiveV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_state(
    int32_t horizon, torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_arrival_features) const {
  CppEnvStaticCacheV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_state_vectors(
      episode_step_, fleets_, planets_, horizon, orbit_planet_features,
      orbit_planet_pairwise_features, available_hit_mask,
      orbit_planet_arrival_features);
}

void CppEnvLiveV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
    torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
    int32_t horizon,
    torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_arrival_features) const {
  CppEnvStaticCacheV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
      episode_step_, fleet_rows, planet_rows, planet_count, horizon,
      orbit_planet_features, orbit_planet_pairwise_features, available_hit_mask,
      orbit_planet_arrival_features);
}

void CppEnvLiveV2::fill_policy_obs_from_rows(
    torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
    torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_mask,
    torch::Tensor orbit_planet_pairwise_mask,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor action_taken_index,
    torch::Tensor player_mask) const {
  CppEnvStaticCacheV2::fill_policy_obs_from_rows(
      episode_step_, fleet_rows, planet_rows, planet_count, orbit_planet_features,
      orbit_planet_mask, orbit_planet_pairwise_mask, orbit_planet_pairwise_features,
      action_taken_index, player_mask);
}

void CppEnvLiveV2::fill_future_resolution_planet_features_from_state(
    int32_t horizon, torch::Tensor orbit_planet_features) const {
  CppEnvStaticCacheV2::fill_future_resolution_planet_features_from_state_vectors(
      episode_step_, fleets_, planets_, horizon, orbit_planet_features);
}

void CppEnvLiveV2::fill_future_resolution_planet_features_from_rows(
    torch::Tensor fleet_rows, torch::Tensor planet_rows, int32_t planet_count,
    int32_t horizon,
    torch::Tensor orbit_planet_features) const {
  CppEnvStaticCacheV2::fill_future_resolution_planet_features_from_rows(
      episode_step_, fleet_rows, planet_rows, planet_count, horizon,
      orbit_planet_features);
}

torch::Tensor CppEnvLiveV2::fleet_takeover_cost_features_from_state(int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_takeover_cost_features_from_state_vectors(
      episode_step_, fleets_, planets_, horizon);
}

py::tuple CppEnvLiveV2::fleet_arrivals_resolution(torch::Tensor arrivals) const {
  return CppEnvStaticCacheV2::fleet_arrivals_resolution_from_arrivals_and_planets(
      arrivals, planets_);
}

py::tuple CppEnvLiveV2::fleet_arrivals_resolution_from_state(int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_arrivals_resolution_from_state_vectors(
      episode_step_, fleets_, planets_, horizon);
}

py::tuple CppEnvLiveV2::fleet_arrivals_resolution_from_rows(
    torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_arrivals_resolution_from_rows(
      episode_step_, fleet_rows, planet_rows, planet_count, horizon);
}

py::list CppEnvLiveV2::fleet_hit_traces_from_state(int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_hit_traces_from_state_vectors(
      episode_step_, fleets_, horizon);
}

py::list CppEnvLiveV2::fleet_hit_traces_from_rows(torch::Tensor fleet_rows,
                                                  int32_t horizon) const {
  return CppEnvStaticCacheV2::fleet_hit_traces_from_rows(
      episode_step_, fleet_rows, horizon);
}
