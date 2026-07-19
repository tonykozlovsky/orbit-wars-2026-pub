#include "cpp_env_static_cache_v2.h"

#include "../masks.h"
#include "../simulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

namespace {

constexpr uint8_t kStaticHitCacheStaticBlocked = 1;
constexpr uint8_t kStaticHitCacheSunBlocked = 2;
constexpr double kStaticSegmentCollisionDistanceSqEpsilon = 1e-7;

std::vector<Planet> planet_rows_tensor_to_vector(torch::Tensor planet_rows,
                                                  int32_t planet_count) {
  TORCH_CHECK_DISABLED(planet_rows.device().is_cpu(), "planet_rows must be CPU");
  TORCH_CHECK_DISABLED(planet_rows.dtype() == torch::kFloat32 ||
                  planet_rows.dtype() == torch::kFloat64,
              "planet_rows must be float32 or float64");
  TORCH_CHECK_DISABLED(planet_rows.sizes() == torch::IntArrayRef({kPlanets, kPlanetRowLen}),
              "planet_rows shape");
  TORCH_CHECK_DISABLED(planet_count > 0 && planet_count <= kPlanets, "planet_count");
  std::vector<Planet> planets;
  planets.reserve(static_cast<uint32_t>(planet_count));
  if (planet_rows.dtype() == torch::kFloat64) {
    const auto pr = planet_rows.accessor<double, 2>();
    for (int32_t i = 0; i < planet_count; ++i) {
      Planet p;
      p.id = static_cast<int32_t>(pr[i][0]);
      p.owner = static_cast<int32_t>(pr[i][1]);
      p.x = pr[i][2];
      p.y = pr[i][3];
      p.radius = pr[i][4];
      p.ships = pr[i][5];
      p.production = pr[i][6];
      TORCH_CHECK_DISABLED(p.id >= 0, "planet id");
      TORCH_CHECK_DISABLED(p.radius > 0.0, "planet radius");
      planets.push_back(p);
    }
  } else {
    const auto pr = planet_rows.accessor<float, 2>();
    for (int32_t i = 0; i < planet_count; ++i) {
      Planet p;
      p.id = static_cast<int32_t>(pr[i][0]);
      p.owner = static_cast<int32_t>(pr[i][1]);
      p.x = static_cast<double>(pr[i][2]);
      p.y = static_cast<double>(pr[i][3]);
      p.radius = static_cast<double>(pr[i][4]);
      p.ships = static_cast<double>(pr[i][5]);
      p.production = static_cast<double>(pr[i][6]);
      TORCH_CHECK_DISABLED(p.id >= 0, "planet id");
      TORCH_CHECK_DISABLED(p.radius > 0.0, "planet radius");
      planets.push_back(p);
    }
  }
  return planets;
}

int32_t noop_frame_count(int32_t episode_steps) {
  TORCH_CHECK_DISABLED(episode_steps > 1, "episode_steps");
  return episode_steps + 1;
}

uint32_t dynamic_dynamic_intercept_cache_index(int32_t aim_index, int32_t src,
                                             int32_t dst, int32_t sn) {
  return (((static_cast<uint32_t>(aim_index) * static_cast<uint32_t>(kPlanets) +
            static_cast<uint32_t>(src)) *
               static_cast<uint32_t>(kPlanets) +
           static_cast<uint32_t>(dst)) *
          static_cast<uint32_t>(kLegacyShipScanClasses)) +
         static_cast<uint32_t>(sn);
}

void apply_orbit_rotation_ticks_inplace(NoopCachedPlanet *p,
                                        double angular_velocity,
                                        int32_t rot_ticks) {
  TORCH_CHECK_DISABLED(p != nullptr, "apply_orbit_rotation_ticks_inplace");
  const double dx = p->x - kCenter;
  const double dy = p->y - kCenter;
  const double r = std::sqrt(dx * dx + dy * dy);
  if (r + p->radius >= kRotationRadiusLimit) {
    return;
  }
  const double initial_angle = std::atan2(dy, dx);
  const double current_angle =
      initial_angle + angular_velocity * static_cast<double>(rot_ticks);
  p->x = kCenter + r * std::cos(current_angle);
  p->y = kCenter + r * std::sin(current_angle);
}

uint32_t static_pair_aim_cache_index(int32_t aim_index, int32_t src,
                                     int32_t dst) {
  return (static_cast<uint32_t>(aim_index) * static_cast<uint32_t>(kPlanets) +
          static_cast<uint32_t>(src)) *
             static_cast<uint32_t>(kPlanets) +
         static_cast<uint32_t>(dst);
}

bool static_intercept_segment_for_aim(const NoopCachedPlanet &src,
                                      const NoopCachedPlanet &dst,
                                      int32_t aim_index, double *out_start_x,
                                      double *out_start_y,
                                      double *out_finish_x,
                                      double *out_finish_y) {
  TORCH_CHECK_DISABLED(out_start_x != nullptr && out_start_y != nullptr &&
                  out_finish_x != nullptr && out_finish_y != nullptr,
              "static intercept segment: outputs");
  const double dx = dst.x - src.x;
  const double dy = dst.y - src.y;
  const double center_dist_sq = dx * dx + dy * dy;
  const double norm = std::sqrt(center_dist_sq);
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    return false;
  }
  const double inv_norm = 1.0 / norm;
  const double ux = dx * inv_norm;
  const double uy = dy * inv_norm;
  const double tangent_dist =
      dst.radius * orbit_wars_honest::kEdgeInterceptTangentRadiusFrac;
  double dir_x = ux;
  double dir_y = uy;
  if (aim_index != orbit_wars_honest::kEdgeInterceptAimCenter) {
    TORCH_CHECK_DISABLED(aim_index == orbit_wars_honest::kEdgeInterceptAimLeft ||
                    aim_index == orbit_wars_honest::kEdgeInterceptAimRight,
                "static intercept segment: bad aim index");
    const double tangent_dist_sq = tangent_dist * tangent_dist;
    if (center_dist_sq <= tangent_dist_sq) {
      return false;
    }
    const double along = std::sqrt(center_dist_sq - tangent_dist_sq);
    const double side =
        (aim_index == orbit_wars_honest::kEdgeInterceptAimLeft) ? 1.0 : -1.0;
    dir_x = (along * ux - side * tangent_dist * uy) * inv_norm;
    dir_y = (along * uy + side * tangent_dist * ux) * inv_norm;
  }
  const double source_offset = src.radius + 0.1;
  const double start_x = src.x + dir_x * source_offset;
  const double start_y = src.y + dir_y * source_offset;
  const double center_from_start_x = dst.x - start_x;
  const double center_from_start_y = dst.y - start_y;
  const double projected =
      center_from_start_x * dir_x + center_from_start_y * dir_y;
  const double center_from_start_sq =
      center_from_start_x * center_from_start_x +
      center_from_start_y * center_from_start_y;
  const double perpendicular_sq =
      center_from_start_sq - projected * projected;
  const double radius_sq = dst.radius * dst.radius;
  if (perpendicular_sq > radius_sq) {
    return false;
  }
  const double hit_distance =
      projected - std::sqrt(radius_sq - perpendicular_sq);
  if (!std::isfinite(hit_distance)) {
    return false;
  }
  *out_start_x = start_x;
  *out_start_y = start_y;
  *out_finish_x = start_x + dir_x * hit_distance;
  *out_finish_y = start_y + dir_y * hit_distance;
  return true;
}

}  // namespace

CppEnvStaticCacheV2::CppEnvStaticCacheV2(int32_t num_agents,
                                          int32_t orbit_instance_id,
                                          double ship_speed,
                                          int32_t episode_steps,
                                          double comet_speed)
    : num_agents_(num_agents),
      orbit_instance_id_(orbit_instance_id),
      ship_speed_(ship_speed),
      episode_steps_(episode_steps),
      comet_speed_(comet_speed) {
  TORCH_CHECK_DISABLED(num_agents_ == 2 || num_agents_ == 4, "num_agents");
  TORCH_CHECK_DISABLED(orbit_instance_id_ >= 0, "orbit_instance_id");
  TORCH_CHECK_DISABLED(ship_speed_ >= 1.0, "ship_speed");
  TORCH_CHECK_DISABLED(episode_steps_ > 0, "episode_steps");
  TORCH_CHECK_DISABLED(comet_speed_ > 0.0, "comet_speed");
  const auto f32_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  const float nan_f = std::numeric_limits<float>::quiet_NaN();
  last_honest_dir_x_ = torch::full({kPlanets, kHitClasses}, nan_f, f32_cpu);
  last_honest_dir_y_ = torch::full({kPlanets, kHitClasses}, nan_f, f32_cpu);
  last_honest_turns_ = torch::full({kPlanets, kHitClasses}, nan_f, f32_cpu);
  last_honest_intercept_ok_ = torch::zeros({kPlanets, kHitClasses}, f32_cpu);
  last_honest_intercept_fail_reason_ = torch::zeros({kPlanets, kHitClasses}, f32_cpu);
  last_honest_send_ships_ = torch::zeros({kPlanets, kHitClasses}, i32_cpu);
  last_honest_hit_kind_ = torch::zeros({kPlanets, kHitClasses}, f32_cpu);
  last_honest_hit_slot_ =
      torch::full({kPlanets, kHitClasses}, static_cast<int32_t>(-1), i32_cpu);
  last_honest_hit_steps_ =
      torch::full({kPlanets, kHitClasses}, static_cast<int32_t>(-1), i32_cpu);
  honest_available_hit_mask_scratch_ =
      torch::zeros({kPlanets, kHitClasses},
                   torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
  cached_intercept_aims_scratch_.resize(static_cast<uint32_t>(kPlanets * kHitClasses));
  cached_intercept_aim_valid_scratch_.resize(static_cast<uint32_t>(kPlanets * kHitClasses));
}

void CppEnvStaticCacheV2::reset(double angular_velocity,
                                  torch::Tensor planet_rows,
                                  int32_t planet_count) {
  const std::vector<Planet> initial_planets =
      planet_rows_tensor_to_vector(planet_rows, planet_count);
  build_noop_cache(angular_velocity, initial_planets);
}

void CppEnvStaticCacheV2::clear_honest_full_pair_cache() {
  const int32_t n_frames = noop_frame_count(episode_steps_);
  honest_full_pair_cache_entry_by_key_.assign(
      static_cast<uint32_t>(n_frames) * static_cast<uint32_t>(kPlanets) *
          static_cast<uint32_t>(kPlanets),
      int32_t{-1});
  honest_full_pair_cache_entries_by_step_.assign(
      static_cast<uint32_t>(n_frames), std::vector<HonestFullPairCacheEntry>{});
  honest_full_pair_cache_live_entries_ = 0;
  honest_full_pair_cache_prune_cursor_ = 0;
  honest_full_step_src_done_max_sn_.assign(
      static_cast<uint32_t>(n_frames) * static_cast<uint32_t>(kPlanets),
      uint8_t{0});
  honest_full_warmup_scan_step_ = 0;
  honest_full_warmup_scan_src_order_idx_ = 0;
  honest_full_src_max_observed_ships_.fill(0);
  honest_full_warmup_extra_sn_ = 0;
  honest_full_warmup_last_lookahead_steps_ = 0;
}

void CppEnvStaticCacheV2::set_wall_profile_enabled(bool enabled) {
  wall_profile_enabled_ = enabled;
  wall_profile_clear();
}

void CppEnvStaticCacheV2::wall_profile_clear() const {
  TORCH_CHECK_DISABLED(wall_profile_stack_.empty(), "wall profile clear with active stack");
  wall_profile_stats_.clear();
}

std::string CppEnvStaticCacheV2::wall_profile_begin(const char *name) const {
  if (!wall_profile_enabled_) {
    return std::string();
  }
  TORCH_CHECK_DISABLED(name != nullptr && name[0] != '\0', "wall profile span name");
  std::string path;
  if (wall_profile_stack_.empty()) {
    path = name;
  } else {
    path = wall_profile_stack_.back() + "/" + name;
  }
  wall_profile_stack_.push_back(path);
  return path;
}

void CppEnvStaticCacheV2::wall_profile_end(
    const char *name, const std::string &path,
    std::chrono::steady_clock::time_point t0) const {
  if (!wall_profile_enabled_) {
    return;
  }
  const auto t1 = std::chrono::steady_clock::now();
  TORCH_CHECK_DISABLED(!wall_profile_stack_.empty(), "wall profile empty stack");
  TORCH_CHECK_DISABLED(wall_profile_stack_.back() == path, "wall profile stack mismatch");
  wall_profile_stack_.pop_back();
  const double dt_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  WallProfileStat *stat = nullptr;
  for (WallProfileStat &candidate : wall_profile_stats_) {
    if (candidate.path == path) {
      stat = &candidate;
      break;
    }
  }
  if (stat == nullptr) {
    wall_profile_stats_.push_back(WallProfileStat{path, 0.0, 0, 0.0});
    stat = &wall_profile_stats_.back();
  }
  stat->sum_ms += dt_ms;
  stat->n += 1;
  stat->last_ms = dt_ms;
  TORCH_CHECK_DISABLED(name != nullptr && name[0] != '\0', "wall profile span name");
}

CppEnvStaticCacheV2::WallProfileSpan::WallProfileSpan(
    const CppEnvStaticCacheV2 *env, const char *name)
    : env_(env),
      name_(name),
      path_(env != nullptr ? env->wall_profile_begin(name) : std::string()) {
  if (env_ != nullptr && env_->wall_profile_enabled_) {
    t0_ = std::chrono::steady_clock::now();
  }
}

CppEnvStaticCacheV2::WallProfileSpan::~WallProfileSpan() {
  if (env_ != nullptr) {
    env_->wall_profile_end(name_, path_, t0_);
  }
}

py::list CppEnvStaticCacheV2::wall_profile_rows() const {
  TORCH_CHECK_DISABLED(wall_profile_stack_.empty(), "wall profile rows with active stack");
  py::list rows;
  for (const WallProfileStat &stat : wall_profile_stats_) {
    rows.append(py::make_tuple(stat.path, stat.sum_ms, stat.n, stat.last_ms));
  }
  return rows;
}

int32_t CppEnvStaticCacheV2::noop_trajectory_length() const {
  TORCH_CHECK_DISABLED(noop_cached_planets_flat_.size() % static_cast<uint32_t>(kPlanets) == 0,
              "noop_trajectory_length: corrupt noop flat buffer");
  return static_cast<int32_t>(noop_cached_planets_flat_.size() /
                              static_cast<uint32_t>(kPlanets));
}

namespace {

torch::Tensor noop_cached_planets_row_to_padded_rows_tensor(const NoopCachedPlanet *row) {
  torch::Tensor t =
      torch::zeros({kPlanets, kPlanetRowLen}, torch::TensorOptions().dtype(torch::kFloat64));
  auto a = t.accessor<double, 2>();
  for (int32_t i = 0; i < kPlanets; ++i) {
    const NoopCachedPlanet &p = row[i];
    if (p.id < 0) {
      continue;
    }
    a[i][0] = static_cast<double>(p.id);
    a[i][1] = -1.0;
    a[i][2] = p.x;
    a[i][3] = p.y;
    a[i][4] = p.radius;
    a[i][5] = 0.0;
    a[i][6] = p.production;
  }
  return t;
}

}  // namespace

torch::Tensor CppEnvStaticCacheV2::noop_trajectory_planets_tensor() const {
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "noop_trajectory_planets_tensor: empty noop cache");
  const int32_t n_frames = noop_trajectory_length();
  std::vector<torch::Tensor> rows;
  rows.reserve(static_cast<uint32_t>(n_frames));
  for (int32_t t = 0; t < n_frames; ++t) {
    const NoopCachedPlanet *row =
        noop_cached_planets_flat_.data() +
        static_cast<uint32_t>(t) * static_cast<uint32_t>(kPlanets);
    rows.push_back(noop_cached_planets_row_to_padded_rows_tensor(row));
  }
  return torch::stack(rows, 0);
}

torch::Tensor CppEnvStaticCacheV2::noop_trajectory_planets_row_tensor(
    int32_t step_index) const {
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "noop_trajectory_planets_row_tensor: empty noop cache");
  const int32_t n_frames = noop_trajectory_length();
  TORCH_CHECK_DISABLED(step_index >= 0 && step_index < n_frames,
              "noop_trajectory_planets_row_tensor: bad step_index ",
              step_index, " n_frames=", n_frames);
  const NoopCachedPlanet *row =
      noop_cached_planets_flat_.data() +
      static_cast<uint32_t>(step_index) * static_cast<uint32_t>(kPlanets);
  return noop_cached_planets_row_to_padded_rows_tensor(row);
}

void CppEnvStaticCacheV2::build_noop_cache(
    double angular_velocity, const std::vector<Planet> &initial_planets) {
  TORCH_CHECK_DISABLED(std::isfinite(angular_velocity), "build_noop_cache: angular_velocity");
  TORCH_CHECK_DISABLED(!initial_planets.empty(), "build_noop_cache: initial_planets empty");
  angular_velocity_ = angular_velocity;
  immutable_planet_prefix_n_ = static_cast<int32_t>(initial_planets.size());
  TORCH_CHECK_DISABLED(immutable_planet_prefix_n_ > 0 &&
                  immutable_planet_prefix_n_ <= kPlanets,
              "build_noop_cache: immutable planet prefix size");
  for (int32_t i = 0; i < immutable_planet_prefix_n_; ++i) {
    immutable_planet_prefix_ids_[static_cast<uint32_t>(i)] =
        initial_planets[static_cast<uint32_t>(i)].id;
  }
  const int32_t n_frames = noop_frame_count(episode_steps_);
  noop_cached_planets_flat_.assign(
      static_cast<uint32_t>(n_frames) * static_cast<uint32_t>(kPlanets),
      NoopCachedPlanet{});
  int32_t rot_ticks = 0;
  for (int32_t t = 0; t < n_frames; ++t) {
    NoopCachedPlanet *row =
        noop_cached_planets_flat_.data() +
        static_cast<uint32_t>(t) * static_cast<uint32_t>(kPlanets);
    for (int32_t i = 0; i < immutable_planet_prefix_n_; ++i) {
      NoopCachedPlanet p =
          noop_cached_planet_from_planet(initial_planets[static_cast<uint32_t>(i)]);
      TORCH_CHECK_DISABLED(p.id == immutable_planet_prefix_ids_[static_cast<uint32_t>(i)],
                  "build_noop_cache: immutable prefix id mismatch");
      if (t == 0) {
        row[i] = p;
      } else {
        apply_orbit_rotation_ticks_inplace(&p, angular_velocity_, rot_ticks);
        row[i] = p;
      }
    }
    if (t > 0) {
      ++rot_ticks;
    }
  }
  // No comet group yet; comet slots remain zeroed. Rebuild slot classification.
  rebuild_slot_kind_cache();
  noop_spatial_grid_ =
      orbit_wars_honest::build_noop_spatial_grid(noop_cached_planets_flat_);
  static_pair_aim_block_cache_.assign(
      static_cast<uint32_t>(orbit_wars_honest::kEdgeInterceptAimCount * kPlanets *
                          kPlanets),
      uint8_t{0});
  static_pair_aim_dynamic_possible_cache_.assign(
      static_pair_aim_block_cache_.size(), uint8_t{0});
  static_pair_bucket_aim_terminal_cache_.assign(
      static_cast<uint32_t>(orbit_wars_honest::kEdgeInterceptAimCount * kPlanets *
                          kPlanets * kLegacyShipScanClasses),
      uint8_t{0});
  static_pair_bucket_aim_fail_reason_cache_.assign(
      static_pair_bucket_aim_terminal_cache_.size(), uint8_t{0});
  clear_honest_full_pair_cache();
  rebuild_static_pair_aim_block_cache();
  rebuild_dynamic_dynamic_intercept_cache();
}

void CppEnvStaticCacheV2::update_comet_in_noop_cache(
    int32_t current_episode_step,
    int32_t path_index,
    int32_t comet_internal_id,
    const std::vector<int32_t> &planet_ids,
    const std::vector<std::vector<std::pair<double, double>>> &paths_kaggle_yx) {
  TORCH_CHECK_DISABLED(current_episode_step >= 0, "update_comet_in_noop_cache: episode_step");
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "update_comet_in_noop_cache: noop cache is empty");
  const int32_t n_frames = noop_frame_count(episode_steps_);
  TORCH_CHECK_DISABLED(noop_trajectory_length() == n_frames,
              "update_comet_in_noop_cache: noop cache length mismatch");
  TORCH_CHECK_DISABLED(immutable_planet_prefix_n_ > 0, "update_comet_in_noop_cache: immutable prefix");
  for (int32_t t = 0; t < n_frames; ++t) {
    NoopCachedPlanet *row =
        noop_cached_planets_flat_.data() +
        static_cast<uint32_t>(t) * static_cast<uint32_t>(kPlanets);
    for (int32_t slot = immutable_planet_prefix_n_; slot < kPlanets; ++slot) {
      row[slot] = NoopCachedPlanet{};
    }
  }
  TORCH_CHECK_DISABLED(static_cast<int32_t>(planet_ids.size()) == 4,
              "update_comet_in_noop_cache: comet group must have 4 planet ids");
  TORCH_CHECK_DISABLED(static_cast<int32_t>(paths_kaggle_yx.size()) == 4,
              "update_comet_in_noop_cache: comet group must have 4 paths");
  TORCH_CHECK_DISABLED(immutable_planet_prefix_n_ + 4 <= kPlanets,
              "update_comet_in_noop_cache: n_prefix+4 exceeds kPlanets");
  for (int32_t t = 0; t < n_frames; ++t) {
    NoopCachedPlanet *row =
        noop_cached_planets_flat_.data() +
        static_cast<uint32_t>(t) * static_cast<uint32_t>(kPlanets);
    for (int32_t k = 0; k < 4; ++k) {
      const int32_t path_idx = path_index + (t - current_episode_step);
      const auto &path = paths_kaggle_yx[static_cast<uint32_t>(k)];
      NoopCachedPlanet cp;
      cp.id = planet_ids[static_cast<uint32_t>(k)];
      cp.comet_internal_id = comet_internal_id;
      if (path_idx < 0) {
        cp.x = -99.0;
        cp.y = -99.0;
      } else {
        if (path_idx >= static_cast<int32_t>(path.size())) {
          continue;
        }
        cp.x = path[static_cast<uint32_t>(path_idx)].first;
        cp.y = path[static_cast<uint32_t>(path_idx)].second;
        cp.comet_time_before_despawn =
            static_cast<double>(static_cast<int32_t>(path.size()) - path_idx);
      }
      cp.radius = kCometRadius;
      cp.production = kCometProduction;
      row[immutable_planet_prefix_n_ + k] = cp;
    }
  }
  noop_spatial_grid_ =
      orbit_wars_honest::build_noop_spatial_grid(noop_cached_planets_flat_);
}

void CppEnvStaticCacheV2::rebuild_slot_kind_cache() {
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "rebuild_slot_kind_cache: empty noop cache");
  TORCH_CHECK_DISABLED(immutable_planet_prefix_n_ > 0 &&
                  immutable_planet_prefix_n_ + 4 <= kPlanets,
              "rebuild_slot_kind_cache: bad immutable_planet_prefix_n_");
  // Classify base planet slots (0..immutable_planet_prefix_n_-1) from
  // noop cache frame 0 using orbital radius.
  planet_slot_orbiting_n_ = 0;
  planet_slot_static_n_ = 0;
  const NoopCachedPlanet *row0 = noop_cached_planets_flat_.data();
  const SmallPlanetIdSet empty_comet_set;
  for (int32_t slot = 0; slot < immutable_planet_prefix_n_; ++slot) {
    const NoopCachedPlanet &p = row0[static_cast<uint32_t>(slot)];
    if (planet_is_rotating_for_mask(p.id, p.x, p.y, p.radius, empty_comet_set)) {
      planet_slot_orbiting_[static_cast<uint32_t>(planet_slot_orbiting_n_++)] = slot;
    } else {
      planet_slot_static_[static_cast<uint32_t>(planet_slot_static_n_++)] = slot;
    }
  }
  // Comet slots are structurally fixed at immutable_planet_prefix_n_ .. +3.
  // Whether a comet is active at a given step is determined from the noop
  // cache (id >= 0), not from this array.
  for (int32_t k = 0; k < 4; ++k) {
    planet_slot_comet_[static_cast<uint32_t>(k)] = immutable_planet_prefix_n_ + k;
  }
  planet_slot_comet_n_ = 4;
}

void CppEnvStaticCacheV2::rebuild_static_pair_aim_block_cache() {
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "static pair block cache: empty noop cache");
  TORCH_CHECK_DISABLED(static_pair_aim_block_cache_.size() ==
                  static_cast<uint32_t>(orbit_wars_honest::kEdgeInterceptAimCount *
                                      kPlanets * kPlanets),
              "static pair block cache: bad size");
  TORCH_CHECK_DISABLED(static_pair_aim_dynamic_possible_cache_.size() ==
                  static_pair_aim_block_cache_.size(),
              "static pair dynamic possible cache: bad size");
  std::fill(static_pair_aim_block_cache_.begin(),
            static_pair_aim_block_cache_.end(), uint8_t{0});
  std::fill(static_pair_aim_dynamic_possible_cache_.begin(),
            static_pair_aim_dynamic_possible_cache_.end(), uint8_t{0});
  const NoopCachedPlanet *row0 = noop_cached_planets_flat_.data();
  double dynamic_outer_radius = 0.0;
  for (int32_t oi = 0; oi < planet_slot_orbiting_n_; ++oi) {
    const int32_t slot = planet_slot_orbiting_[static_cast<uint32_t>(oi)];
    const NoopCachedPlanet &p = row0[static_cast<uint32_t>(slot)];
    const double dx = p.x - kCenter;
    const double dy = p.y - kCenter;
    dynamic_outer_radius =
        std::max(dynamic_outer_radius, std::sqrt(dx * dx + dy * dy) + p.radius);
  }
  const double dynamic_outer_radius_sq =
      dynamic_outer_radius * dynamic_outer_radius;
  for (int32_t ai = 0; ai < orbit_wars_honest::kEdgeInterceptAimCount; ++ai) {
    for (int32_t si = 0; si < planet_slot_static_n_; ++si) {
      const int32_t src = planet_slot_static_[static_cast<uint32_t>(si)];
      const NoopCachedPlanet &ps = row0[static_cast<uint32_t>(src)];
      TORCH_CHECK_DISABLED(ps.id >= 0 && ps.comet_internal_id < 0,
                  "static pair block cache: bad source");
      for (int32_t di = 0; di < planet_slot_static_n_; ++di) {
        const int32_t dst = planet_slot_static_[static_cast<uint32_t>(di)];
        if (dst == src) {
          continue;
        }
        const NoopCachedPlanet &pd = row0[static_cast<uint32_t>(dst)];
        TORCH_CHECK_DISABLED(pd.id >= 0 && pd.comet_internal_id < 0,
                    "static pair block cache: bad target");
        double start_x = 0.0;
        double start_y = 0.0;
        double finish_x = 0.0;
        double finish_y = 0.0;
        if (!static_intercept_segment_for_aim(ps, pd, ai, &start_x, &start_y,
                                             &finish_x, &finish_y)) {
          continue;
        }
        const uint32_t cache_idx = static_pair_aim_cache_index(ai, src, dst);
        if (dynamic_outer_radius > 0.0 &&
            point_to_segment_distance_sq(kCenter, kCenter, start_x, start_y,
                                         finish_x, finish_y) <=
                dynamic_outer_radius_sq) {
          static_pair_aim_dynamic_possible_cache_[cache_idx] = uint8_t{1};
        }
        uint8_t status = 0;
        for (int32_t bi = 0; bi < planet_slot_static_n_; ++bi) {
          const int32_t blocker = planet_slot_static_[static_cast<uint32_t>(bi)];
          if (blocker == dst) {
            continue;
          }
          const NoopCachedPlanet &pb = row0[static_cast<uint32_t>(blocker)];
          const double radius_sq = pb.radius * pb.radius;
          if (point_to_segment_distance_sq(pb.x, pb.y, start_x, start_y,
                                           finish_x, finish_y) <
              radius_sq + kStaticSegmentCollisionDistanceSqEpsilon) {
            status = kStaticHitCacheStaticBlocked;
            break;
          }
        }
        if (status == 0 &&
            point_to_segment_distance_sq(kCenter, kCenter, start_x, start_y,
                                         finish_x, finish_y) <
                kSunRadius * kSunRadius +
                    kStaticSegmentCollisionDistanceSqEpsilon) {
          status = kStaticHitCacheSunBlocked;
        }
        static_pair_aim_block_cache_[cache_idx] = status;
      }
    }
  }
}

void CppEnvStaticCacheV2::rebuild_dynamic_dynamic_intercept_cache() {
  dynamic_dynamic_intercept_cache_.assign(
      static_cast<uint32_t>(orbit_wars_honest::kEdgeInterceptAimCount *
                          kPlanets * kPlanets * kLegacyShipScanClasses),
      DynamicDynamicInterceptCacheEntry{});
  const orbit_wars_honest::NoopView noop =
      orbit_wars_honest::make_noop_view(noop_cached_planets_flat_,
                                        noop_spatial_grid_);
  constexpr int32_t kDynamicDynamicCacheBaseFrame = 1;
  TORCH_CHECK_DISABLED(noop.n_frames > kDynamicDynamicCacheBaseFrame,
              "dynamic-dynamic cache: missing base frame");
  const NoopCachedPlanet *row0 =
      noop.flat + static_cast<uint32_t>(kDynamicDynamicCacheBaseFrame) *
                      static_cast<uint32_t>(kPlanets);
  const SmallPlanetIdSet empty_comet_set;
  std::array<uint8_t, kPlanets> is_dynamic_slot{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  std::array<int32_t, kPlanets> dynamic_slots{};
  std::array<double, kPlanets> slot_radius{};
  int32_t dynamic_slots_n = 0;
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    slot_planet_id[static_cast<uint32_t>(slot)] = -1;
    slot_comet_internal_id[static_cast<uint32_t>(slot)] = -1;
  }
  for (int32_t si = 0; si < planet_slot_orbiting_n_; ++si) {
    const int32_t slot = planet_slot_orbiting_[static_cast<uint32_t>(si)];
    const NoopCachedPlanet &p = row0[static_cast<uint32_t>(slot)];
    is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
    slot_planet_id[static_cast<uint32_t>(slot)] = p.id;
    slot_radius[static_cast<uint32_t>(slot)] = p.radius;
    dynamic_slots[static_cast<uint32_t>(dynamic_slots_n++)] = slot;
  }
  for (int32_t ai = 0; ai < orbit_wars_honest::kEdgeInterceptAimCount; ++ai) {
    for (int32_t si = 0; si < planet_slot_orbiting_n_; ++si) {
      const int32_t src = planet_slot_orbiting_[static_cast<uint32_t>(si)];
      const NoopCachedPlanet &ps = row0[static_cast<uint32_t>(src)];
      TORCH_CHECK_DISABLED(ps.comet_internal_id < 0,
                  "dynamic-dynamic cache: comet source slot");
      for (int32_t di = 0; di < planet_slot_orbiting_n_; ++di) {
        const int32_t dst = planet_slot_orbiting_[static_cast<uint32_t>(di)];
        if (dst == src) {
          continue;
        }
        const NoopCachedPlanet &pd = row0[static_cast<uint32_t>(dst)];
        TORCH_CHECK_DISABLED(pd.comet_internal_id < 0,
                    "dynamic-dynamic cache: comet target slot");
        for (int32_t sn = 1; sn < kLegacyShipScanClasses; ++sn) {
          orbit_wars_honest::EdgeInterceptAim aim;
          const bool valid =
              orbit_wars_honest::edge_intercept_aim_for_ship_count_and_aim_index(
                  noop, ps, dst, pd, empty_comet_set,
                  ship_count_for_legacy_scan_subindex(sn), ship_speed_,
                  kDynamicDynamicCacheBaseFrame, ai, &aim);
          DynamicDynamicInterceptCacheEntry &entry =
              dynamic_dynamic_intercept_cache_[
                  dynamic_dynamic_intercept_cache_index(ai, src, dst, sn)];
          entry.valid = valid ? uint8_t{1} : uint8_t{0};
          entry.fail_reason = static_cast<uint8_t>(aim.fail_reason);
          if (!valid) {
            continue;
          }
          const orbit_wars_honest::EdgeInterceptAimCandidate &candidate =
              aim.candidates[static_cast<uint32_t>(ai)];
          TORCH_CHECK_DISABLED(candidate.valid,
                      "dynamic-dynamic cache: valid aim missing candidate");
          const double dx = candidate.aim_x - ps.x;
          const double dy = candidate.aim_y - ps.y;
          const double norm = std::sqrt(dx * dx + dy * dy);
          TORCH_CHECK_DISABLED(norm > 0.0 && std::isfinite(norm),
                      "dynamic-dynamic cache: zero direction");
          entry.dir_x0 = static_cast<float>(dx / norm);
          entry.dir_y0 = static_cast<float>(dy / norm);
          entry.turns_to_target =
              static_cast<float>(candidate.turns_to_target);
          const orbit_wars_honest::EdgeActionHitWithAim hit_with_aim =
              orbit_wars_honest::edge_action_hit_for_cached_dynamic_dynamic_intercept(
                  noop, kDynamicDynamicCacheBaseFrame,
                  noop.n_frames - kDynamicDynamicCacheBaseFrame, src,
                  ps.radius, dst, ship_count_for_legacy_scan_subindex(sn),
                  ship_speed_, static_cast<double>(entry.dir_x0),
                  static_cast<double>(entry.dir_y0),
                  static_cast<double>(entry.turns_to_target), ai,
                  is_dynamic_slot, slot_planet_id, slot_comet_internal_id,
                  {}, 0, dynamic_slots, dynamic_slots_n, slot_radius);
          TORCH_CHECK_DISABLED(hit_with_aim.has_aim,
                      "dynamic-dynamic cache: hit missing aim");
          entry.hit_available =
              hit_with_aim.hit.available ? uint8_t{1} : uint8_t{0};
          entry.hit_kind = static_cast<int16_t>(hit_with_aim.hit.hit_kind);
          entry.hit_slot = static_cast<int16_t>(hit_with_aim.hit.hit_slot);
          entry.hit_steps = static_cast<int16_t>(hit_with_aim.hit.hit_steps);
        }
      }
    }
  }
}
