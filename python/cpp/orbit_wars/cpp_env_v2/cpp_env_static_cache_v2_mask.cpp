#include "cpp_env_static_cache_v2.h"

#include "../honest_shared_features.h"
#include "../honest_shared_intercept.h"
#include "../library.h"
#include "../masks.h"
#include "../simulation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

using orbit_wars_honest::EdgeActionHit;
using orbit_wars_honest::EdgeActionHitWithAim;
using orbit_wars_honest::EdgeInterceptDebugSegment;
using orbit_wars_honest::EdgeInterceptDebugSolverResult;
using orbit_wars_honest::EdgeInterceptDebugSolverResults;
using orbit_wars_honest::EdgeInterceptDebugTargetPoint;
using orbit_wars_honest::EdgeInterceptAim;
using orbit_wars_honest::EdgeInterceptAimCandidate;
using orbit_wars_honest::NoopView;
using orbit_wars_honest::cached_dynamic_dynamic_comet_overlay_hit;
using orbit_wars_honest::edge_action_hit_for_intercept_aim;
using orbit_wars_honest::edge_action_hit_for_static_checked_intercept_aim;
using orbit_wars_honest::edge_action_hit_with_intercept_aims;
using orbit_wars_honest::edge_intercept_debug_solver_results_for_ship_count_and_aim_index;
using orbit_wars_honest::edge_intercept_debug_trace_for_ship_count_and_aim_index;
using orbit_wars_honest::edge_intercept_aim_for_ship_count;
using orbit_wars_honest::edge_intercept_aim_for_ship_count_and_aim_index;
using orbit_wars_honest::fill_honest_aim_nan_outputs;
using orbit_wars_honest::fleet_arrivals_for_fleets;
using orbit_wars_honest::honest_source_planet_can_launch;
using orbit_wars_honest::kEdgeInterceptAimCenter;
using orbit_wars_honest::kEdgeInterceptAimCount;
using orbit_wars_honest::kEdgeInterceptAimLeft;
using orbit_wars_honest::kEdgeInterceptAimRight;
using orbit_wars_honest::kInterceptFailReasonDynamicSolverNoConverge;
using orbit_wars_honest::kHitKindInterceptionFailed;
using orbit_wars_honest::kHitKindNone;
using orbit_wars_honest::kHitKindStatic;
using orbit_wars_honest::kHitKindSun;
using orbit_wars_honest::kHitKindTarget;
using orbit_wars_honest::kHitKindTimeout;
using orbit_wars_honest::kHitNone;
using orbit_wars_honest::kHitSun;
using orbit_wars_honest::kHonestHitTraceMaxSteps;
using orbit_wars_honest::kHonestInterceptMaxIters;
using orbit_wars_honest::make_noop_view;

namespace {

constexpr uint8_t kStaticHitCacheStaticBlocked = 1;
constexpr uint8_t kStaticHitCacheSunBlocked = 2;
constexpr uint8_t kStaticHitCacheInterceptFailed = 3;
constexpr uint8_t kStaticHitCacheTimeout = 4;
constexpr int32_t kHonestFullCacheWarmupHorizonSteps = 50;
constexpr int32_t kHonestFullCacheWarmupMaxShipBuckets = 1000;
constexpr bool kFailedInterceptionLoggingEnabled = false;
constexpr int32_t kTakeoverCandidateHeavyShipCountBudget = 20;
constexpr int32_t kTakeoverCandidateSpeedSaturationShipCount = 1000;
static_assert(kTakeoverCandidateHeavyShipCountBudget > 1);

int32_t policy_slot_for_compact_agent(int32_t compact_agent, int32_t num_agents) {
  TORCH_CHECK_DISABLED(0 <= compact_agent && compact_agent < num_agents,
              "policy slot: compact_agent");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4,
              "policy slot: num_agents");
  if (num_agents == 4) {
    return compact_agent;
  }
  return compact_agent == 0 ? 0 : 3;
}

struct TakeoverActionCandidate {
  bool available = false;
  int32_t ship_count = 0;
  int32_t target_hit_steps = 0;
  int32_t aim_index = -1;
  EdgeActionHitWithAim hit_with_aim{};
};

struct TakeoverShipInterval {
  int32_t lo = 0;
  int32_t hi = 0;
};

template <size_t N>
void merge_takeover_ship_interval(
    std::array<TakeoverShipInterval, N> &intervals, int32_t &intervals_n,
    int32_t lo, int32_t hi) {
  TORCH_CHECK(lo <= hi, "takeover interval merge: empty interval");
  TORCH_CHECK(0 <= intervals_n && intervals_n <= static_cast<int32_t>(N),
              "takeover interval merge: bad interval count");
  int32_t merged_lo = lo;
  int32_t merged_hi = hi;
  int32_t write_n = 0;
  for (int32_t interval_i = 0; interval_i < intervals_n; ++interval_i) {
    const TakeoverShipInterval interval =
        intervals[static_cast<uint32_t>(interval_i)];
    if (interval.hi < merged_lo - 1 || merged_hi < interval.lo - 1) {
      intervals[static_cast<uint32_t>(write_n)] = interval;
      ++write_n;
      continue;
    }
    merged_lo = std::min<int32_t>(merged_lo, interval.lo);
    merged_hi = std::max<int32_t>(merged_hi, interval.hi);
  }
  TORCH_CHECK(write_n < static_cast<int32_t>(N),
              "takeover interval merge: too many intervals");
  intervals[static_cast<uint32_t>(write_n)] =
      TakeoverShipInterval{merged_lo, merged_hi};
  intervals_n = write_n + 1;
}

bool takeover_ship_count_in_intervals(
    const std::array<TakeoverShipInterval, kPlanetArrivalHorizon> &intervals,
    int32_t intervals_n, int32_t ship_count) {
  TORCH_CHECK(0 <= intervals_n && intervals_n <= kPlanetArrivalHorizon,
              "takeover interval contains: bad interval count");
  for (int32_t interval_i = 0; interval_i < intervals_n; ++interval_i) {
    const TakeoverShipInterval interval =
        intervals[static_cast<uint32_t>(interval_i)];
    if (interval.lo <= ship_count && ship_count <= interval.hi) {
      return true;
    }
  }
  return false;
}

int32_t takeover_ship_interval_hi_from_earlier_required_ships(
    int32_t earlier_required_ships, int32_t source_ships) {
  TORCH_CHECK(source_ships > 0, "takeover interval hi: source ships");
  TORCH_CHECK(earlier_required_ships > 0,
              "takeover interval hi: earlier required ships");
  if (earlier_required_ships >= kTakeoverCandidateSpeedSaturationShipCount) {
    return source_ships;
  }
  return earlier_required_ships - 1;
}

int32_t takeover_ship_count_at_flat_index(
    const std::array<TakeoverShipInterval,
                     kEdgeInterceptAimCount * kPlanetArrivalHorizon>
        &intervals,
    int32_t intervals_n, int32_t flat_index) {
  TORCH_CHECK(0 <= intervals_n &&
                  intervals_n <= kEdgeInterceptAimCount * kPlanetArrivalHorizon,
              "takeover flat ship index: bad interval count");
  TORCH_CHECK(flat_index >= 0, "takeover flat ship index negative");
  int32_t remaining = flat_index;
  for (int32_t interval_i = 0; interval_i < intervals_n; ++interval_i) {
    const TakeoverShipInterval interval =
        intervals[static_cast<uint32_t>(interval_i)];
    const int32_t interval_length = interval.hi - interval.lo + 1;
    TORCH_CHECK(interval_length > 0, "takeover flat interval empty");
    if (remaining < interval_length) {
      return interval.lo + remaining;
    }
    remaining -= interval_length;
  }
  TORCH_CHECK(false, "takeover flat ship index out of range");
  return 0;
}

uint32_t static_pair_aim_cache_index(int32_t aim_index, int32_t src, int32_t dst) {
  return (static_cast<uint32_t>(aim_index) * static_cast<uint32_t>(kPlanets) +
          static_cast<uint32_t>(src)) *
             static_cast<uint32_t>(kPlanets) +
         static_cast<uint32_t>(dst);
}

uint32_t static_pair_bucket_aim_cache_index(int32_t aim_index, int32_t src,
                                          int32_t dst, int32_t sn) {
  return (static_pair_aim_cache_index(aim_index, src, dst) *
          static_cast<uint32_t>(kLegacyShipScanClasses)) +
         static_cast<uint32_t>(sn);
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

uint32_t honest_full_pair_cache_key(int32_t episode_step, int32_t src,
                                    int32_t dst) {
  return (static_cast<uint32_t>(episode_step) * static_cast<uint32_t>(kPlanets) +
          static_cast<uint32_t>(src)) *
             static_cast<uint32_t>(kPlanets) +
         static_cast<uint32_t>(dst);
}

SmallPlanetIdSet active_comet_planet_ids_at_frame(
    const NoopCachedPlanet *planets_row,
    const std::array<int32_t, kPlanets> &comet_slots, int32_t comet_slots_n) {
  SmallPlanetIdSet ids;
  for (int32_t k = 0; k < comet_slots_n; ++k) {
    const int32_t slot = comet_slots[static_cast<uint32_t>(k)];
    TORCH_CHECK_DISABLED(0 <= slot && slot < kPlanets, "active_comet_planet_ids: bad comet slot");
    const int32_t pid = planets_row[static_cast<uint32_t>(slot)].id;
    if (pid >= 0) {
      ids.insert(pid);
    }
  }
  return ids;
}

void fill_mask_slot_arrays_from_cache(
    const std::array<int32_t, kPlanets> &planet_slot_static,
    int32_t planet_slot_static_n,
    const std::array<int32_t, kPlanets> &planet_slot_orbiting,
    int32_t planet_slot_orbiting_n,
    const std::array<int32_t, kPlanets> &planet_slot_comet,
    int32_t planet_slot_comet_n,
    const NoopCachedPlanet *planets_row,
    std::array<int32_t, kPlanets> &static_slots,
    int32_t &static_slots_n,
    std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t &dynamic_slots_n,
    std::array<uint8_t, kPlanets> &is_static_slot,
    std::array<uint8_t, kPlanets> &is_dynamic_slot) {
  static_slots_n = planet_slot_static_n;
  dynamic_slots_n = 0;
  for (int32_t i = 0; i < kPlanets; ++i) {
    is_static_slot[static_cast<uint32_t>(i)] = 0;
    is_dynamic_slot[static_cast<uint32_t>(i)] = 0;
  }
  for (int32_t i = 0; i < static_slots_n; ++i) {
    const int32_t slot = planet_slot_static[static_cast<uint32_t>(i)];
    static_slots[static_cast<uint32_t>(i)] = slot;
    is_static_slot[static_cast<uint32_t>(slot)] = 1;
  }
  for (int32_t i = 0; i < planet_slot_orbiting_n; ++i) {
    const int32_t slot = planet_slot_orbiting[static_cast<uint32_t>(i)];
    dynamic_slots[static_cast<uint32_t>(dynamic_slots_n++)] = slot;
    is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
  }
  for (int32_t k = 0; k < planet_slot_comet_n; ++k) {
    const int32_t slot = planet_slot_comet[static_cast<uint32_t>(k)];
    if (planets_row[static_cast<uint32_t>(slot)].id >= 0) {
      dynamic_slots[static_cast<uint32_t>(dynamic_slots_n++)] = slot;
      is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
    }
  }
}

bool cheap_intercept_candidate_point_for_center(
    double center_x, double center_y, double source_x, double source_y,
    double target_radius, int32_t aim_index, double *out_aim_x,
    double *out_aim_y, double *out_dir_x, double *out_dir_y) {
  const double dx = center_x - source_x;
  const double dy = center_y - source_y;
  const double center_dist_sq = dx * dx + dy * dy;
  const double norm = std::sqrt(center_dist_sq);
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    return false;
  }
  const double inv_norm = 1.0 / norm;
  const double ux = dx * inv_norm;
  const double uy = dy * inv_norm;
  const double tangent_dist =
      target_radius * orbit_wars_honest::kEdgeInterceptTangentRadiusFrac;
  if (aim_index == kEdgeInterceptAimCenter) {
    *out_aim_x = center_x - ux * tangent_dist;
    *out_aim_y = center_y - uy * tangent_dist;
    *out_dir_x = ux;
    *out_dir_y = uy;
    return true;
  }
  TORCH_CHECK_DISABLED(aim_index == kEdgeInterceptAimLeft ||
                           aim_index == kEdgeInterceptAimRight,
                       "cheap takeover candidate point: bad aim index");
  const double tangent_dist_sq = tangent_dist * tangent_dist;
  if (center_dist_sq <= tangent_dist_sq) {
    return false;
  }
  const double along = std::sqrt(center_dist_sq - tangent_dist_sq);
  const double side = aim_index == kEdgeInterceptAimLeft ? 1.0 : -1.0;
  const double dir_x = (along * ux - side * tangent_dist * uy) * inv_norm;
  const double dir_y = (along * uy + side * tangent_dist * ux) * inv_norm;
  *out_aim_x = source_x + dir_x * along;
  *out_aim_y = source_y + dir_y * along;
  *out_dir_x = dir_x;
  *out_dir_y = dir_y;
  return true;
}

int32_t cheap_min_ships_for_required_speed(double required_speed,
                                           int32_t source_ships,
                                           double ship_speed) {
  TORCH_CHECK(source_ships > 0, "cheap min ships: source ships");
  TORCH_CHECK(required_speed >= 0.0 && std::isfinite(required_speed),
              "cheap min ships: required speed");
  const double min_speed = orbit_cpp_fleet_speed(1.0, ship_speed);
  if (required_speed <= min_speed) {
    return 1;
  }
  if (orbit_cpp_fleet_speed(static_cast<double>(source_ships), ship_speed) <
      required_speed) {
    return source_ships + 1;
  }
  TORCH_CHECK(ship_speed > min_speed,
              "cheap min ships: speed curve cannot satisfy required speed");

  constexpr double kSpeedSaturationShipCount = 1000.0;
  const double speed_frac =
      (required_speed - min_speed) / (ship_speed - min_speed);
  TORCH_CHECK(speed_frac > 0.0 && speed_frac <= 1.0 &&
                  std::isfinite(speed_frac),
              "cheap min ships: bad speed fraction");
  const double continuous_ships =
      std::exp(std::log(kSpeedSaturationShipCount) *
               std::pow(speed_frac, 2.0 / 3.0));
  TORCH_CHECK(continuous_ships >= 1.0 &&
                  continuous_ships <= kSpeedSaturationShipCount &&
                  std::isfinite(continuous_ships),
              "cheap min ships: bad inverted ship count");
  int32_t candidate = std::clamp<int32_t>(
      static_cast<int32_t>(std::ceil(continuous_ships)), 1,
      static_cast<int32_t>(kSpeedSaturationShipCount));
  if (candidate > 1 &&
      orbit_cpp_fleet_speed(static_cast<double>(candidate - 1), ship_speed) >=
          required_speed) {
    candidate -= 1;
  }
  if (orbit_cpp_fleet_speed(static_cast<double>(candidate), ship_speed) <
      required_speed) {
    candidate += 1;
  }
  TORCH_CHECK(candidate <= source_ships,
              "cheap min ships: inverted count exceeds source ships");
  return candidate;
}

int32_t cheap_required_ships_for_hit_steps(
    const NoopView &noop, int32_t noop_base_frame,
    const NoopCachedPlanet &src_planet, int32_t dst_slot,
    const NoopCachedPlanet &dst_planet, int32_t aim_index, int32_t hit_steps,
    int32_t source_ships, double ship_speed) {
  TORCH_CHECK_DISABLED(1 <= hit_steps && hit_steps <= kPlanetArrivalHorizon,
                       "cheap takeover required ships: hit_steps");
  const int32_t target_frame = noop_base_frame + hit_steps;
  if (target_frame >= noop.n_frames) {
    return source_ships + 1;
  }
  const NoopCachedPlanet &dst_at_hit =
      noop.flat[static_cast<uint32_t>(target_frame) *
                    static_cast<uint32_t>(kPlanets) +
                static_cast<uint32_t>(dst_slot)];
  if (dst_at_hit.id != dst_planet.id ||
      dst_at_hit.comet_internal_id != dst_planet.comet_internal_id) {
    return source_ships + 1;
  }
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  if (!cheap_intercept_candidate_point_for_center(
          dst_at_hit.x, dst_at_hit.y, src_planet.x, src_planet.y,
          dst_planet.radius, aim_index, &aim_x, &aim_y, &dir_x, &dir_y)) {
    return source_ships + 1;
  }
  const double source_offset = src_planet.radius + 0.1;
  const double start_x = src_planet.x + dir_x * source_offset;
  const double start_y = src_planet.y + dir_y * source_offset;
  const double center_from_start_x = dst_at_hit.x - start_x;
  const double center_from_start_y = dst_at_hit.y - start_y;
  const double projected = center_from_start_x * dir_x + center_from_start_y * dir_y;
  const double center_from_start_sq =
      center_from_start_x * center_from_start_x +
      center_from_start_y * center_from_start_y;
  const double perpendicular_sq =
      center_from_start_sq - projected * projected;
  const double radius_sq = dst_planet.radius * dst_planet.radius;
  if (perpendicular_sq > radius_sq) {
    return source_ships + 1;
  }
  const double hit_distance =
      std::max(0.0, projected - std::sqrt(radius_sq - perpendicular_sq));
  const double required_speed = hit_distance / static_cast<double>(hit_steps);
  return cheap_min_ships_for_required_speed(required_speed, source_ships,
                                            ship_speed);
}

TakeoverActionCandidate best_takeover_action_candidate_for_pair(
    const CppEnvStaticCacheV2 *profile_env, const NoopView &noop,
    int32_t noop_base_frame, int32_t remaining_steps,
    const NoopCachedPlanet &src_planet, int32_t src_slot, int32_t dst_slot,
    const NoopCachedPlanet &dst_planet,
    const SmallPlanetIdSet &comet_planet_ids, int32_t source_ships,
    int32_t player, double ship_speed, torch::Tensor cost_by_time_slot_player,
    const std::array<uint8_t, kPlanets> &is_static_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius) {
  TORCH_CHECK_DISABLED(source_ships > 0,
                       "best takeover candidate: source ships");
  TORCH_CHECK_DISABLED(0 <= player && player < kPlayerAxisSlots,
                       "best takeover candidate: player");
  TORCH_CHECK_DISABLED(cost_by_time_slot_player.device().is_cpu(),
                       "best takeover candidate: cost device");
  TORCH_CHECK_DISABLED(cost_by_time_slot_player.dtype() == torch::kInt32,
                       "best takeover candidate: cost dtype");
  TORCH_CHECK_DISABLED(cost_by_time_slot_player.sizes() ==
                           torch::IntArrayRef({kPlanetArrivalHorizon, kPlanets,
                                               kPlayerAxisSlots}),
                       "best takeover candidate: cost shape");
  const int32_t horizon =
      std::min<int32_t>(kPlanetArrivalHorizon, remaining_steps - 1);
  if (horizon <= 0) {
    return TakeoverActionCandidate{};
  }
  auto costs = cost_by_time_slot_player.accessor<int32_t, 3>();
  TakeoverActionCandidate best;
  std::array<int32_t, kPlanetArrivalHorizon + 1> cost_by_target_hit_steps{};
  bool has_affordable_positive_cost = false;
  for (int32_t target_hit_steps = 1; target_hit_steps <= horizon;
       ++target_hit_steps) {
    const int32_t cost = costs[target_hit_steps - 1][dst_slot][player];
    TORCH_CHECK(cost >= 0, "best takeover candidate: negative cost");
    cost_by_target_hit_steps[static_cast<uint32_t>(target_hit_steps)] = cost;
    has_affordable_positive_cost =
        has_affordable_positive_cost || (cost > 0 && cost <= source_ships);
  }
  if (!has_affordable_positive_cost) {
    return best;
  }
  std::array<std::array<TakeoverShipInterval, kPlanetArrivalHorizon>,
             kEdgeInterceptAimCount>
      ship_intervals_by_aim{};
  std::array<int32_t, kEdgeInterceptAimCount> ship_intervals_n_by_aim{};
  std::array<TakeoverShipInterval,
             kEdgeInterceptAimCount * kPlanetArrivalHorizon>
      global_ship_intervals{};
  int32_t global_ship_intervals_n = 0;
  for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
    std::array<int32_t, kPlanetArrivalHorizon + 1> req_by_hit_steps{};
    for (int32_t hit_steps = 0; hit_steps <= kPlanetArrivalHorizon; ++hit_steps) {
      req_by_hit_steps[static_cast<uint32_t>(hit_steps)] = source_ships + 1;
    }
    for (int32_t hit_steps = 1; hit_steps <= horizon; ++hit_steps) {
      req_by_hit_steps[static_cast<uint32_t>(hit_steps)] =
          cheap_required_ships_for_hit_steps(
              noop, noop_base_frame, src_planet, dst_slot, dst_planet,
              aim_index, hit_steps, source_ships, ship_speed);
    }
    std::array<TakeoverShipInterval, kPlanetArrivalHorizon> &ship_intervals =
        ship_intervals_by_aim[static_cast<uint32_t>(aim_index)];
    int32_t &ship_intervals_n =
        ship_intervals_n_by_aim[static_cast<uint32_t>(aim_index)];
    for (int32_t target_hit_steps = 1; target_hit_steps <= horizon;
         ++target_hit_steps) {
      const int32_t lo_hit_steps =
          std::min<int32_t>(horizon, target_hit_steps + 1);
      int32_t lo = req_by_hit_steps[static_cast<uint32_t>(lo_hit_steps)];
      int32_t hi = source_ships;
      if (target_hit_steps > 2) {
        hi = takeover_ship_interval_hi_from_earlier_required_ships(
            req_by_hit_steps[static_cast<uint32_t>(target_hit_steps - 2)],
            source_ships);
      }
      const int32_t cost =
          cost_by_target_hit_steps[static_cast<uint32_t>(target_hit_steps)];
      if (cost == 0 || cost > source_ships) {
        continue;
      }
      lo = std::max<int32_t>(std::max<int32_t>(lo, cost), 1);
      hi = std::min<int32_t>(hi, source_ships);
      if (lo > hi) {
        continue;
      }
      TORCH_CHECK(cost > 0 && cost <= source_ships && lo >= cost,
                  "takeover interval: range below cost");
      merge_takeover_ship_interval(ship_intervals, ship_intervals_n, lo, hi);
      merge_takeover_ship_interval(global_ship_intervals,
                                   global_ship_intervals_n, lo, hi);
    }
  }
  if (global_ship_intervals_n == 0) {
    return best;
  }
  std::sort(global_ship_intervals.begin(),
            global_ship_intervals.begin() + global_ship_intervals_n,
            [](const TakeoverShipInterval &a, const TakeoverShipInterval &b) {
              return a.lo < b.lo || (a.lo == b.lo && a.hi < b.hi);
            });
  int32_t candidate_ship_counts_n = 0;
  for (int32_t interval_i = 0; interval_i < global_ship_intervals_n;
       ++interval_i) {
    const TakeoverShipInterval interval =
        global_ship_intervals[static_cast<uint32_t>(interval_i)];
    const int32_t interval_length = interval.hi - interval.lo + 1;
    TORCH_CHECK(interval_length > 0, "takeover candidate interval empty");
    TORCH_CHECK(candidate_ship_counts_n <=
                    std::numeric_limits<int32_t>::max() - interval_length,
                "takeover candidate ship count overflow");
    candidate_ship_counts_n += interval_length;
  }
  TORCH_CHECK(candidate_ship_counts_n > 0,
              "takeover candidate ship count list empty");
  const int32_t ship_count_checks =
      std::min<int32_t>(candidate_ship_counts_n,
                        kTakeoverCandidateHeavyShipCountBudget);
  for (int32_t check_i = 0; check_i < ship_count_checks; ++check_i) {
    int32_t candidate_idx = check_i;
    if (candidate_ship_counts_n > kTakeoverCandidateHeavyShipCountBudget) {
      TORCH_CHECK(kTakeoverCandidateHeavyShipCountBudget > 1,
                  "takeover candidate budget must cover endpoints");
      candidate_idx = static_cast<int32_t>(
          (static_cast<int64_t>(check_i) *
           static_cast<int64_t>(candidate_ship_counts_n - 1)) /
          static_cast<int64_t>(kTakeoverCandidateHeavyShipCountBudget - 1));
    }
    const int32_t ship_count = takeover_ship_count_at_flat_index(
        global_ship_intervals, global_ship_intervals_n, candidate_idx);
    for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount;
         ++aim_index) {
      const std::array<TakeoverShipInterval, kPlanetArrivalHorizon>
          &ship_intervals =
              ship_intervals_by_aim[static_cast<uint32_t>(aim_index)];
      const int32_t ship_intervals_n =
          ship_intervals_n_by_aim[static_cast<uint32_t>(aim_index)];
      if (!takeover_ship_count_in_intervals(ship_intervals, ship_intervals_n,
                                            ship_count)) {
        continue;
      }
      EdgeInterceptAim aim;
      const bool valid = edge_intercept_aim_for_ship_count_and_aim_index(
          noop, src_planet, dst_slot, dst_planet, comet_planet_ids,
          ship_count, ship_speed, noop_base_frame, aim_index, &aim);
      if (!valid) {
        continue;
      }
      const EdgeInterceptAimCandidate &intercept =
          aim.candidates[static_cast<uint32_t>(aim_index)];
      if (!intercept.valid) {
        continue;
      }
      const EdgeActionHitWithAim hit_with_aim =
          edge_action_hit_for_intercept_aim(
              noop, noop_base_frame, remaining_steps, src_planet, src_slot,
              dst_slot, ship_count, ship_speed, aim, aim_index,
              is_static_slot[static_cast<uint32_t>(dst_slot)] != 0, false,
              is_static_slot, is_dynamic_slot, slot_planet_id,
              slot_comet_internal_id, static_slots, static_slots_n,
              dynamic_slots, dynamic_slots_n, slot_radius);
      if (!hit_with_aim.has_aim ||
          hit_with_aim.hit.hit_kind != kHitKindTarget ||
          hit_with_aim.hit.hit_slot != dst_slot) {
        continue;
      }
      const int32_t hit_steps = hit_with_aim.hit.hit_steps;
      if (hit_steps < 1 || hit_steps > horizon) {
        continue;
      }
      const int32_t cost =
          cost_by_target_hit_steps[static_cast<uint32_t>(hit_steps)];
      if (cost == 0 || ship_count < cost) {
        continue;
      }
      best.available = true;
      best.ship_count = ship_count;
      best.target_hit_steps = hit_steps;
      best.aim_index = aim_index;
      best.hit_with_aim = hit_with_aim;
      return best;
    }
  }
  return best;
}

}  // namespace

void CppEnvStaticCacheV2::log_failed_interception(
    const NoopView &noop,
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
    const char *source) const {
  assert(kFailedInterceptionLoggingEnabled);
  assert(noop.n_frames > 0);
  assert(planets_row != nullptr);
  assert(0 < planet_count && planet_count <= kPlanets);
  assert(0 <= src && src < planet_count);
  assert(0 <= dst && dst < planet_count);
  assert(src != dst);
  assert(sn > 0);
  assert(ship_count > 0);
  assert(source != nullptr && source[0] != '\0');

  const EdgeInterceptDebugSolverResults solver_results =
      edge_intercept_debug_solver_results_for_ship_count_and_aim_index(
          noop, planets_row[static_cast<uint32_t>(src)], dst,
          planets_row[static_cast<uint32_t>(dst)], comet_planet_ids, ship_count,
          ship_speed_, noop_base_frame, aim_index, kHonestHitTraceMaxSteps);
  if (!(!solver_results.fair_fast.valid && solver_results.fair_slow.valid)) {
    return;
  }

  std::vector<EdgeInterceptDebugTargetPoint> target_path;
  std::vector<EdgeInterceptDebugSegment> segments;
  edge_intercept_debug_trace_for_ship_count_and_aim_index(
      noop, planets_row[static_cast<uint32_t>(src)], dst,
      planets_row[static_cast<uint32_t>(dst)], comet_planet_ids, ship_count,
      ship_speed_, noop_base_frame, aim_index, kHonestHitTraceMaxSteps,
      &target_path, &segments);

  std::filesystem::create_directories("outputs");
  std::ofstream out("outputs/failed_interceptions.txt", std::ios::app);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open outputs/failed_interceptions.txt");
  }

  const NoopCachedPlanet &src_planet = planets_row[static_cast<uint32_t>(src)];
  const NoopCachedPlanet &dst_planet = planets_row[static_cast<uint32_t>(dst)];
  auto write_solver_result = [&](const char *name,
                                 const EdgeInterceptDebugSolverResult &r) {
    out << "\"" << name << "\":{\"valid\":" << (r.valid ? "true" : "false");
    if (r.valid) {
      out << ",\"aim_x\":" << r.aim_x
          << ",\"aim_y\":" << r.aim_y
          << ",\"dir_x\":" << r.dir_x
          << ",\"dir_y\":" << r.dir_y
          << ",\"turns\":" << r.turns;
    }
    out << '}';
  };
  out << std::setprecision(17)
      << "{\"orbit_instance_id\":" << orbit_instance_id_
      << ",\"num_agents\":" << num_agents_
      << ",\"episode_step\":" << episode_step
      << ",\"episode_steps\":" << episode_steps_
      << ",\"noop_base_frame\":" << noop_base_frame
      << ",\"noop_frames\":" << noop.n_frames
      << ",\"remaining_steps\":" << remaining_steps
      << ",\"angular_velocity\":" << angular_velocity_
      << ",\"ship_speed\":" << ship_speed_
      << ",\"comet_speed\":" << comet_speed_
      << ",\"source\":\"" << source << "\""
      << ",\"aim_index\":" << aim_index
      << ",\"src\":" << src
      << ",\"dst\":" << dst
      << ",\"ship_subindex\":" << sn
      << ",\"ship_count\":" << ship_count
      << ",\"fail_reason\":" << fail_reason
      << ",\"solver_results\":{";
  write_solver_result("point", solver_results.point);
  out << ',';
  write_solver_result("bisect", solver_results.bisect);
  out << ',';
  write_solver_result("hybrid", solver_results.hybrid);
  out << ',';
  write_solver_result("fair_fast", solver_results.fair_fast);
  out << ',';
  write_solver_result("fair_slow", solver_results.fair_slow);
  out << "},\"src_planet\":{\"slot\":" << src
      << ",\"id\":" << src_planet.id
      << ",\"x\":" << src_planet.x
      << ",\"y\":" << src_planet.y
      << ",\"radius\":" << src_planet.radius
      << "},\"dst_planet\":{\"slot\":" << dst
      << ",\"id\":" << dst_planet.id
      << ",\"comet_internal_id\":" << dst_planet.comet_internal_id
      << ",\"x\":" << dst_planet.x
      << ",\"y\":" << dst_planet.y
      << ",\"radius\":" << dst_planet.radius
      << "},\"target_path\":[";
  for (uint32_t i = 0; i < target_path.size(); ++i) {
    const EdgeInterceptDebugTargetPoint &p = target_path[i];
    if (i != 0) {
      out << ',';
    }
    out << "{\"step\":" << p.step
        << ",\"x\":" << p.x
        << ",\"y\":" << p.y
        << '}';
  }
  out << "],\"segments\":[";
  for (uint32_t i = 0; i < segments.size(); ++i) {
    const EdgeInterceptDebugSegment &s = segments[i];
    if (i != 0) {
      out << ',';
    }
    out << "{\"iter\":" << s.iter
        << ",\"branch\":" << s.branch
        << ",\"target_center_x\":" << s.target_center_x
        << ",\"target_center_y\":" << s.target_center_y
        << ",\"aim_x\":" << s.aim_x
        << ",\"aim_y\":" << s.aim_y
        << ",\"start_x\":" << s.start_x
        << ",\"start_y\":" << s.start_y
        << ",\"end_x\":" << s.end_x
        << ",\"end_y\":" << s.end_y
        << ",\"turns\":" << s.turns
        << '}';
  }
  out << "]}\n";
  if (!out.good()) {
    throw std::runtime_error("failed to write outputs/failed_interceptions.txt");
  }
}

bool CppEnvStaticCacheV2::load_honest_full_pair_cache_entry(
    const NoopView &noop, int32_t episode_step, int32_t remaining_steps,
    int32_t src, int32_t dst, int32_t max_sn, bool apply_comet_overlay,
    const std::array<double, kPlanets> &slot_radius,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id) const {
  assert(0 <= episode_step && episode_step <= episode_steps_);
  assert(remaining_steps > 0);
  assert(0 <= src && src < kPlanets);
  assert(0 <= dst && dst < kPlanets);
  assert(src != dst);
  assert(1 <= max_sn && max_sn < kLegacyShipScanClasses);
  if (episode_step < honest_full_pair_cache_prune_cursor_) {
    return false;
  }
  const uint32_t key = honest_full_pair_cache_key(episode_step, src, dst);
  if (honest_full_pair_cache_entry_by_key_.empty()) {
    return false;
  }
  assert(key < honest_full_pair_cache_entry_by_key_.size());
  const int32_t entry_idx = honest_full_pair_cache_entry_by_key_[key];
  if (entry_idx < 0) {
    return false;
  }
  assert(episode_step < static_cast<int32_t>(honest_full_pair_cache_entries_by_step_.size()));
  const std::vector<HonestFullPairCacheEntry> &step_entries =
      honest_full_pair_cache_entries_by_step_[static_cast<uint32_t>(episode_step)];
  assert(entry_idx < static_cast<int32_t>(step_entries.size()));
  const HonestFullPairCacheEntry &entry =
      step_entries[static_cast<uint32_t>(entry_idx)];
  if (entry.filled_max_sn < max_sn) {
    return false;
  }
  auto dir_x = last_honest_dir_x_.accessor<float, 2>();
  auto dir_y = last_honest_dir_y_.accessor<float, 2>();
  auto turns = last_honest_turns_.accessor<float, 2>();
  auto intercept_ok = last_honest_intercept_ok_.accessor<float, 2>();
  auto intercept_fail_reason =
      last_honest_intercept_fail_reason_.accessor<float, 2>();
  auto hit_kind = last_honest_hit_kind_.accessor<float, 2>();
  auto hit_slot_last = last_honest_hit_slot_.accessor<int32_t, 2>();
  auto hit_steps_last = last_honest_hit_steps_.accessor<int32_t, 2>();
  const int32_t class_base = dst * kHitClassesPerTarget;
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(episode_step) * static_cast<uint32_t>(kPlanets);
  const double source_radius = planets_row[static_cast<uint32_t>(src)].radius;
  for (int32_t sn = 1; sn <= max_sn; ++sn) {
    const int32_t cls = class_base + sn;
    int32_t cached_hit_kind =
        static_cast<int32_t>(entry.hit_kind[static_cast<uint32_t>(sn)]);
    int32_t cached_hit_slot =
        static_cast<int32_t>(entry.hit_slot[static_cast<uint32_t>(sn)]);
    int32_t cached_hit_steps =
        static_cast<int32_t>(entry.hit_steps[static_cast<uint32_t>(sn)]);
    const float cached_dir_x = entry.dir_x[static_cast<uint32_t>(sn)];
    const float cached_dir_y = entry.dir_y[static_cast<uint32_t>(sn)];
    const float cached_turns = entry.turns[static_cast<uint32_t>(sn)];
    if (apply_comet_overlay && cached_hit_kind == kHitKindTarget) {
      const EdgeActionHit comet_hit = cached_dynamic_dynamic_comet_overlay_hit(
          noop, episode_step, remaining_steps, src, source_radius, dst,
          ship_count_for_legacy_scan_subindex(sn), ship_speed_,
          static_cast<double>(cached_dir_x), static_cast<double>(cached_dir_y),
          static_cast<double>(cached_turns), slot_planet_id,
          slot_comet_internal_id, slot_radius);
      const bool comet_first =
          comet_hit.hit_slot >= 0 &&
          (comet_hit.hit_steps < cached_hit_steps ||
           (comet_hit.hit_steps == cached_hit_steps &&
            comet_hit.hit_slot < cached_hit_slot));
      if (comet_first) {
        cached_hit_kind = comet_hit.hit_kind;
        cached_hit_slot = comet_hit.hit_slot;
        cached_hit_steps = comet_hit.hit_steps;
      }
    }
    hit_kind[src][cls] = static_cast<float>(cached_hit_kind);
    hit_slot_last[src][cls] = cached_hit_slot;
    hit_steps_last[src][cls] = cached_hit_steps;
    dir_x[src][cls] = cached_dir_x;
    dir_y[src][cls] = cached_dir_y;
    turns[src][cls] = cached_turns;
    intercept_ok[src][cls] = entry.intercept_ok[static_cast<uint32_t>(sn)];
    intercept_fail_reason[src][cls] =
        entry.intercept_fail_reason[static_cast<uint32_t>(sn)];
  }
  return true;
}

void CppEnvStaticCacheV2::store_honest_full_pair_cache_entry(
    int32_t episode_step, int32_t src, int32_t dst, int32_t min_sn,
    int32_t max_sn) const {
  assert(0 <= episode_step && episode_step <= episode_steps_);
  assert(0 <= src && src < kPlanets);
  assert(0 <= dst && dst < kPlanets);
  assert(src != dst);
  assert(1 <= min_sn && min_sn <= max_sn);
  assert(1 <= max_sn && max_sn < kLegacyShipScanClasses);
  assert(!honest_full_pair_cache_entry_by_key_.empty());
  assert(episode_step >= honest_full_pair_cache_prune_cursor_);
  const uint32_t key = honest_full_pair_cache_key(episode_step, src, dst);
  assert(key < honest_full_pair_cache_entry_by_key_.size());
  int32_t entry_idx = honest_full_pair_cache_entry_by_key_[key];
  assert(episode_step < static_cast<int32_t>(honest_full_pair_cache_entries_by_step_.size()));
  std::vector<HonestFullPairCacheEntry> &step_entries =
      honest_full_pair_cache_entries_by_step_[static_cast<uint32_t>(episode_step)];
  if (entry_idx < 0) {
    entry_idx = static_cast<int32_t>(step_entries.size());
    honest_full_pair_cache_entry_by_key_[key] = entry_idx;
    step_entries.push_back(HonestFullPairCacheEntry{});
    ++honest_full_pair_cache_live_entries_;
  }
  HonestFullPairCacheEntry &entry =
      step_entries[static_cast<uint32_t>(entry_idx)];
  assert(entry.filled_max_sn + 1 == min_sn);
  auto dir_x = last_honest_dir_x_.accessor<float, 2>();
  auto dir_y = last_honest_dir_y_.accessor<float, 2>();
  auto turns = last_honest_turns_.accessor<float, 2>();
  auto intercept_ok = last_honest_intercept_ok_.accessor<float, 2>();
  auto intercept_fail_reason =
      last_honest_intercept_fail_reason_.accessor<float, 2>();
  auto hit_kind = last_honest_hit_kind_.accessor<float, 2>();
  auto hit_slot_last = last_honest_hit_slot_.accessor<int32_t, 2>();
  auto hit_steps_last = last_honest_hit_steps_.accessor<int32_t, 2>();
  const int32_t class_base = dst * kHitClassesPerTarget;
  for (int32_t sn = min_sn; sn <= max_sn; ++sn) {
    const int32_t cls = class_base + sn;
    entry.hit_kind[static_cast<uint32_t>(sn)] =
        static_cast<int16_t>(static_cast<int32_t>(hit_kind[src][cls]));
    entry.hit_slot[static_cast<uint32_t>(sn)] =
        static_cast<int16_t>(hit_slot_last[src][cls]);
    entry.hit_steps[static_cast<uint32_t>(sn)] =
        static_cast<int16_t>(hit_steps_last[src][cls]);
    entry.dir_x[static_cast<uint32_t>(sn)] = dir_x[src][cls];
    entry.dir_y[static_cast<uint32_t>(sn)] = dir_y[src][cls];
    entry.turns[static_cast<uint32_t>(sn)] = turns[src][cls];
    entry.intercept_ok[static_cast<uint32_t>(sn)] = intercept_ok[src][cls];
    entry.intercept_fail_reason[static_cast<uint32_t>(sn)] =
        intercept_fail_reason[src][cls];
  }
  entry.filled_max_sn = max_sn;
}

int32_t CppEnvStaticCacheV2::prune_honest_full_pair_cache_before(
    int32_t min_episode_step) const {
  assert(0 <= min_episode_step && min_episode_step <= episode_steps_ + 1);
  assert(honest_full_pair_cache_entries_by_step_.size() ==
         static_cast<uint32_t>(episode_steps_ + 1));
  assert(0 <= honest_full_pair_cache_prune_cursor_ &&
         honest_full_pair_cache_prune_cursor_ <= episode_steps_ + 1);
  if (min_episode_step <= honest_full_pair_cache_prune_cursor_) {
    return 0;
  }
  int32_t pruned = 0;
  for (int32_t t = honest_full_pair_cache_prune_cursor_; t < min_episode_step; ++t) {
    std::vector<HonestFullPairCacheEntry> &step_entries =
        honest_full_pair_cache_entries_by_step_[static_cast<uint32_t>(t)];
    pruned += static_cast<int32_t>(step_entries.size());
    std::vector<HonestFullPairCacheEntry>().swap(step_entries);
  }
  honest_full_pair_cache_prune_cursor_ = min_episode_step;
  honest_full_pair_cache_live_entries_ -= pruned;
  assert(honest_full_pair_cache_live_entries_ >= 0);
  return pruned;
}

void CppEnvStaticCacheV2::honest_shared_action_mask_impl(
    int32_t episode_step,
    const int32_t *request_data,
    int32_t request_n,
    bool request_data_has_min_sn,
    bool all_geometry,
    torch::Tensor out_action_mask,
    const char *profile_name,
    bool apply_comet_overlay,
    bool store_full_pair_cache) const {
  WallProfileSpan profile(this, profile_name);
  const int32_t noop_base_frame = episode_step;
  NoopView noop{};
  const NoopCachedPlanet *planets_row = nullptr;
  int32_t n = 0;
  SmallPlanetIdSet comet_planet_ids;
  {
    WallProfileSpan profile_contract(this, "output_contract");
    assert_cpu_int8(out_action_mask, "honest_shared_action_mask_limited out_action_mask");
    TORCH_CHECK_DISABLED(out_action_mask.dim() == 2 && out_action_mask.size(0) == kPlanets &&
                    out_action_mask.size(1) == kHitClasses,
                "honest_shared_action_mask_limited out_action_mask shape");
    TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
                "honest_shared_action_mask_limited: noop cache empty");
    TORCH_CHECK_DISABLED(request_n == 0 || request_data != nullptr,
                "honest_shared_action_mask_limited: missing request data");
  }
  {
    WallProfileSpan profile_noop(this, "noop_frame_and_comets");
    noop = make_noop_view(noop_cached_planets_flat_, noop_spatial_grid_);
    TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
                "honest_shared_action_mask_limited: noop cache missing frame");
    planets_row =
        noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
    while (n < kPlanets && planets_row[static_cast<uint32_t>(n)].id >= 0) {
      ++n;
    }
    comet_planet_ids =
        active_comet_planet_ids_at_frame(planets_row, planet_slot_comet_,
                                         planet_slot_comet_n_);
  }
  {
    WallProfileSpan profile_aim_output_reset(this, "aim_output_reset");
    fill_honest_aim_nan_outputs(
        last_honest_dir_x_, last_honest_dir_y_, last_honest_turns_,
        last_honest_intercept_ok_, last_honest_intercept_fail_reason_);
  }

  {
    WallProfileSpan profile_scratch(this, "hit_scratch_init");
    last_honest_hit_kind_.zero_();
    last_honest_hit_slot_.fill_(kHitNone);
    last_honest_hit_steps_.fill_(-1);
  }

  std::array<double, kPlanets> slot_radius{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  {
    WallProfileSpan profile_slot_metadata(this, "slot_metadata_arrays");
    for (int32_t i = 0; i < kPlanets; ++i) {
      slot_radius[static_cast<uint32_t>(i)] = 0.0;
      slot_planet_id[static_cast<uint32_t>(i)] = -1;
      slot_comet_internal_id[static_cast<uint32_t>(i)] = -1;
    }
    for (int32_t i = 0; i < n; ++i) {
      slot_radius[static_cast<uint32_t>(i)] =
          planets_row[static_cast<uint32_t>(i)].radius;
      slot_planet_id[static_cast<uint32_t>(i)] =
          planets_row[static_cast<uint32_t>(i)].id;
      slot_comet_internal_id[static_cast<uint32_t>(i)] =
          planets_row[static_cast<uint32_t>(i)].comet_internal_id;
    }
  }

  std::array<int32_t, kPlanets * kPlanets> max_sn_by_pair{};
  std::array<int32_t, kPlanets * kPlanets> min_sn_by_pair{};
  {
    WallProfileSpan profile_max_sn(this, "max_sn_by_pair");
    for (int32_t i = 0; i < kPlanets * kPlanets; ++i) {
      max_sn_by_pair[static_cast<uint32_t>(i)] = 0;
      min_sn_by_pair[static_cast<uint32_t>(i)] = 0;
    }
    if (all_geometry) {
      for (int32_t src = 0; src < n; ++src) {
        for (int32_t dst = 0; dst < n; ++dst) {
          if (dst != src) {
            const uint32_t idx = static_cast<uint32_t>(src * kPlanets + dst);
            min_sn_by_pair[idx] = 1;
            max_sn_by_pair[idx] = kLegacyShipScanClasses - 1;
          }
        }
      }
    } else {
      for (int32_t i = 0; i < request_n; ++i) {
        const int32_t stride = request_data_has_min_sn ? 4 : 3;
        const int32_t src = request_data[i * stride + 0];
        const int32_t dst = request_data[i * stride + 1];
        const int32_t min_sn =
            request_data_has_min_sn ? request_data[i * stride + 2] : 1;
        const int32_t max_sn =
            request_data[i * stride + (request_data_has_min_sn ? 3 : 2)];
        TORCH_CHECK_DISABLED(0 <= src && src < n,
                    "honest_shared_action_mask_limited: request src out of range");
        TORCH_CHECK_DISABLED(0 <= dst && dst < n,
                    "honest_shared_action_mask_limited: request dst out of range");
        TORCH_CHECK_DISABLED(0 <= max_sn && max_sn < kLegacyShipScanClasses,
                    "honest_shared_action_mask_limited: request max_sn out of range");
        TORCH_CHECK_DISABLED((max_sn == 0 && min_sn == 1) ||
                             (1 <= min_sn && min_sn <= max_sn),
                    "honest_shared_action_mask_limited: request min_sn out of range");
        const uint32_t idx = static_cast<uint32_t>(src * kPlanets + dst);
        if (max_sn > 0) {
          min_sn_by_pair[idx] =
              max_sn_by_pair[idx] == 0
                  ? min_sn
                  : std::min(min_sn_by_pair[idx], min_sn);
        }
        max_sn_by_pair[idx] = std::max(max_sn_by_pair[idx], max_sn);
      }
    }
  }
  std::array<int32_t, kPlanets * kPlanets> output_max_sn_by_pair =
      max_sn_by_pair;

  if (noop.n_frames > 0) {
    std::array<int32_t, kPlanets> static_slots{};
    std::array<int32_t, kPlanets> dynamic_slots{};
    int32_t static_slots_n = 0;
    int32_t dynamic_slots_n = 0;
    std::array<uint8_t, kPlanets> is_static_slot{};
    std::array<uint8_t, kPlanets> is_dynamic_slot{};
    {
      WallProfileSpan profile_slots(this, "slot_arrays_from_cache");
      fill_mask_slot_arrays_from_cache(
          planet_slot_static_, planet_slot_static_n_, planet_slot_orbiting_,
          planet_slot_orbiting_n_, planet_slot_comet_, planet_slot_comet_n_,
          planets_row, static_slots, static_slots_n, dynamic_slots,
          dynamic_slots_n, is_static_slot, is_dynamic_slot);
    }

    const int32_t remaining_steps = noop.n_frames - noop_base_frame;
    auto dir_x = last_honest_dir_x_.accessor<float, 2>();
    auto dir_y = last_honest_dir_y_.accessor<float, 2>();
    auto turns = last_honest_turns_.accessor<float, 2>();
    auto intercept_ok = last_honest_intercept_ok_.accessor<float, 2>();
    auto intercept_fail_reason =
        last_honest_intercept_fail_reason_.accessor<float, 2>();
    auto hit_kind = last_honest_hit_kind_.accessor<float, 2>();
    auto hit_slot_last = last_honest_hit_slot_.accessor<int32_t, 2>();
    auto hit_steps_last = last_honest_hit_steps_.accessor<int32_t, 2>();

    {
      WallProfileSpan profile_full_pair_cache(this, "full_pair_cache_load");
      const int32_t cache_planet_count = immutable_planet_prefix_n_;
      assert(cache_planet_count > 0 && cache_planet_count <= n);
      for (int32_t src = 0; src < cache_planet_count; ++src) {
        for (int32_t dst = 0; dst < cache_planet_count; ++dst) {
          if (dst == src) {
            continue;
          }
          const uint32_t pair_idx =
              static_cast<uint32_t>(src * kPlanets + dst);
          const int32_t pair_max_sn = max_sn_by_pair[pair_idx];
          if (pair_max_sn <= 0) {
            continue;
          }
          if (load_honest_full_pair_cache_entry(
                  noop, episode_step, remaining_steps, src, dst, pair_max_sn,
                  apply_comet_overlay && !comet_planet_ids.empty(), slot_radius,
                  slot_planet_id, slot_comet_internal_id)) {
            max_sn_by_pair[pair_idx] = 0;
          }
        }
      }
    }

    TORCH_CHECK_DISABLED(
        cached_intercept_aims_scratch_.size() == static_cast<uint32_t>(kPlanets * kHitClasses),
        "honest_shared_action_mask_limited: aim scratch size");
    TORCH_CHECK_DISABLED(static_pair_aim_block_cache_.size() ==
                    static_cast<uint32_t>(kEdgeInterceptAimCount * kPlanets * kPlanets),
                "honest_shared_action_mask_limited: static pair aim cache size");
    TORCH_CHECK_DISABLED(static_pair_aim_dynamic_possible_cache_.size() ==
                    static_pair_aim_block_cache_.size(),
                "honest_shared_action_mask_limited: static pair dynamic possible cache size");
    TORCH_CHECK_DISABLED(static_pair_bucket_aim_terminal_cache_.size() ==
                    static_cast<uint32_t>(kEdgeInterceptAimCount * kPlanets * kPlanets *
                                        kLegacyShipScanClasses),
                "honest_shared_action_mask_limited: static pair bucket cache size");
    TORCH_CHECK_DISABLED(static_pair_bucket_aim_fail_reason_cache_.size() ==
                    static_pair_bucket_aim_terminal_cache_.size(),
                "honest_shared_action_mask_limited: static pair fail reason cache size");
    TORCH_CHECK_DISABLED(dynamic_dynamic_intercept_cache_.size() ==
                    static_cast<uint32_t>(kEdgeInterceptAimCount * kPlanets *
                                        kPlanets * kLegacyShipScanClasses),
                "honest_shared_action_mask_limited: dynamic pair cache size");
    std::vector<EdgeInterceptAim> &cached_intercept_aims = cached_intercept_aims_scratch_;

    // optimization flags

    {
      WallProfileSpan profile_hit_scan(this, "intercept_hit_scan");
      for (int32_t src = 0; src < n; ++src) {
        for (int32_t dst = 0; dst < n; ++dst) {
          const int32_t class_base = dst * kHitClassesPerTarget;
          const int32_t pair_max_sn =
              max_sn_by_pair[static_cast<uint32_t>(src * kPlanets + dst)];
          if (pair_max_sn <= 0) {
            continue;
          }
          const int32_t pair_min_sn =
              min_sn_by_pair[static_cast<uint32_t>(src * kPlanets + dst)];
          assert(1 <= pair_min_sn && pair_min_sn <= pair_max_sn);
          for (int32_t sn0 = pair_max_sn; sn0 >= pair_min_sn; --sn0) {
            const int32_t cls0 = class_base + sn0;
            hit_kind[src][cls0] = static_cast<float>(kHitKindNone);
            hit_slot_last[src][cls0] = kHitNone;
            hit_steps_last[src][cls0] = -1;
          }
        }
      }
      const float nan_v = std::numeric_limits<float>::quiet_NaN();
      for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
        const char *profile_name =
            aim_index == kEdgeInterceptAimCenter
                ? "intercept_hit_center"
                : (aim_index == kEdgeInterceptAimLeft ? "intercept_hit_left"
                                                      : "intercept_hit_right");
        WallProfileSpan profile_hit_side(this, profile_name);
        for (int32_t src = 0; src < n; ++src) {
          const NoopCachedPlanet &ps = planets_row[static_cast<uint32_t>(src)];
          for (int32_t dst = 0; dst < n; ++dst) {
            const int32_t class_base = dst * kHitClassesPerTarget;
            const int32_t pair_max_sn =
                max_sn_by_pair[static_cast<uint32_t>(src * kPlanets + dst)];
            if (pair_max_sn <= 0) {
              continue;
            }
            const int32_t pair_min_sn =
                min_sn_by_pair[static_cast<uint32_t>(src * kPlanets + dst)];
            assert(1 <= pair_min_sn && pair_min_sn <= pair_max_sn);
            const bool source_static = is_static_slot[static_cast<uint32_t>(src)] != 0;
            const bool source_dynamic =
                is_dynamic_slot[static_cast<uint32_t>(src)] != 0;
            const bool source_comet =
                slot_comet_internal_id[static_cast<uint32_t>(src)] >= 0;
            const bool target_comet =
                slot_comet_internal_id[static_cast<uint32_t>(dst)] >= 0;
            const bool target_static = is_static_slot[static_cast<uint32_t>(dst)] != 0;
            const bool target_dynamic =
                is_dynamic_slot[static_cast<uint32_t>(dst)] != 0;
            const bool static_pair =
                source_static && target_static;
            const bool dynamic_pair =
                noop_base_frame >= 1 &&
                source_dynamic &&
                target_dynamic &&
                !source_comet &&
                !target_comet;
            bool blocked_for_remaining_sn = false;
            bool intercept_failed_for_remaining_sn = false;
            bool has_target_hit_for_prior_bucket = false;
            float blocked_hit_kind = static_cast<float>(kHitKindStatic);
            float intercept_fail_reason_for_remaining_sn = 0.0f;
            const char *intercept_fail_source_for_remaining_sn = "remaining_ship_bucket";
            auto record_intercept_failed_if_empty = [&](int32_t cls,
                                                        float fail_reason,
                                                        const char *source) {
              if (static_cast<int32_t>(hit_kind[src][cls]) != kHitKindNone) {
                return;
              }
              if constexpr (kFailedInterceptionLoggingEnabled) {
                const int32_t failed_sn = cls - class_base;
                log_failed_interception(
                    noop, planets_row, n, episode_step, noop_base_frame,
                    remaining_steps, aim_index, src, dst, failed_sn,
                    ship_count_for_legacy_scan_subindex(failed_sn),
                    static_cast<int32_t>(fail_reason), comet_planet_ids,
                    source);
              }
              hit_kind[src][cls] =
                  static_cast<float>(kHitKindInterceptionFailed);
              hit_slot_last[src][cls] = kHitNone;
              hit_steps_last[src][cls] = -1;
              dir_x[src][cls] = nan_v;
              dir_y[src][cls] = nan_v;
              turns[src][cls] = nan_v;
              intercept_ok[src][cls] = -1.0f;
              intercept_fail_reason[src][cls] = fail_reason;
            };
            uint8_t static_pair_cache_status = 0;
            bool static_pair_dynamic_possible = true;
            if (static_pair) {
              const uint32_t pair_aim_cache_idx =
                  static_pair_aim_cache_index(aim_index, src, dst);
              static_pair_cache_status =
                  static_pair_aim_block_cache_[pair_aim_cache_idx];
              static_pair_dynamic_possible =
                  static_pair_aim_dynamic_possible_cache_[pair_aim_cache_idx] != 0;
              if (static_pair_cache_status == kStaticHitCacheStaticBlocked ||
                  static_pair_cache_status == kStaticHitCacheSunBlocked) {
                blocked_for_remaining_sn = true;
                blocked_hit_kind =
                    static_pair_cache_status == kStaticHitCacheStaticBlocked
                        ? static_cast<float>(kHitKindStatic)
                        : static_cast<float>(kHitKindSun);
              }
            }
            for (int32_t sn = pair_max_sn; sn >= pair_min_sn; --sn) {
              const int32_t cls = class_base + sn;
              if (static_cast<int32_t>(hit_kind[src][cls]) == kHitKindTarget) {
                continue;
              }
              if (aim_index != kEdgeInterceptAimCenter &&
                  static_cast<int32_t>(hit_kind[src][cls]) ==
                      kHitKindInterceptionFailed) {
                continue;
              }
              if (blocked_for_remaining_sn) {
                hit_kind[src][cls] = blocked_hit_kind;
                hit_slot_last[src][cls] = kHitNone;
                hit_steps_last[src][cls] = -1;
                dir_x[src][cls] = nan_v;
                dir_y[src][cls] = nan_v;
                turns[src][cls] = nan_v;
                intercept_ok[src][cls] = 0.0f;
                intercept_fail_reason[src][cls] = 0.0f;
                continue;
              }
              if (intercept_failed_for_remaining_sn) {
                record_intercept_failed_if_empty(
                    cls, intercept_fail_reason_for_remaining_sn,
                    intercept_fail_source_for_remaining_sn);
                continue;
              }
              if (static_pair) {
                const uint32_t bucket_cache_idx =
                    static_pair_bucket_aim_cache_index(aim_index, src, dst, sn);
                const uint8_t bucket_cache_status =
                    static_pair_bucket_aim_terminal_cache_[bucket_cache_idx];
                if (bucket_cache_status == kStaticHitCacheInterceptFailed) {
                  record_intercept_failed_if_empty(
                      cls, static_cast<float>(
                               static_pair_bucket_aim_fail_reason_cache_
                                   [bucket_cache_idx]),
                      "static_pair_bucket_cache");
                  continue;
                }
                if (bucket_cache_status == kStaticHitCacheTimeout) {
                  hit_kind[src][cls] = static_cast<float>(kHitKindTimeout);
                  hit_slot_last[src][cls] = kHitNone;
                  hit_steps_last[src][cls] = -1;
                  dir_x[src][cls] = nan_v;
                  dir_y[src][cls] = nan_v;
                  turns[src][cls] = nan_v;
                  intercept_ok[src][cls] = 0.0f;
                  intercept_fail_reason[src][cls] = 0.0f;
                  continue;
                }
              }
              if (dynamic_pair) {
                const DynamicDynamicInterceptCacheEntry &entry =
                    dynamic_dynamic_intercept_cache_[
                        dynamic_dynamic_intercept_cache_index(aim_index, src,
                                                              dst, sn)];
                if (entry.valid == 0) {
                  intercept_failed_for_remaining_sn = true;
                  intercept_fail_reason_for_remaining_sn =
                      static_cast<float>(entry.fail_reason);
                  intercept_fail_source_for_remaining_sn = "dynamic_dynamic_cache_invalid";
                  record_intercept_failed_if_empty(
                      cls, intercept_fail_reason_for_remaining_sn,
                      intercept_fail_source_for_remaining_sn);
                  continue;
                }
                const double cached_turns_to_target =
                    static_cast<double>(entry.turns_to_target);
                if (static_cast<double>(noop_base_frame) +
                        cached_turns_to_target >
                    static_cast<double>(noop.n_frames - 1)) {
                  intercept_failed_for_remaining_sn = true;
                  intercept_fail_reason_for_remaining_sn =
                      static_cast<float>(
                          kInterceptFailReasonDynamicSolverNoConverge);
                  intercept_fail_source_for_remaining_sn = "dynamic_dynamic_cache_timeout";
                  record_intercept_failed_if_empty(
                      cls, intercept_fail_reason_for_remaining_sn,
                      intercept_fail_source_for_remaining_sn);
                  continue;
                }
                const int32_t rot_ticks =
                    noop_base_frame > 0 ? noop_base_frame - 1 : 0;
                const double phase =
                    angular_velocity_ * static_cast<double>(rot_ticks);
                const double c = std::cos(phase);
                const double s = std::sin(phase);
                const double cached_dir_x = static_cast<double>(entry.dir_x0);
                const double cached_dir_y = static_cast<double>(entry.dir_y0);
                const double runtime_dir_x = cached_dir_x * c - cached_dir_y * s;
                const double runtime_dir_y = cached_dir_y * c + cached_dir_x * s;
                EdgeActionHit action_hit{
                    entry.hit_available != 0,
                    static_cast<int32_t>(entry.hit_steps),
                    static_cast<int32_t>(entry.hit_kind),
                    static_cast<int32_t>(entry.hit_slot)};
                if (apply_comet_overlay &&
                    action_hit.hit_kind == kHitKindTarget &&
                    !comet_planet_ids.empty()) {
                  const EdgeActionHit comet_hit =
                      cached_dynamic_dynamic_comet_overlay_hit(
                          noop, noop_base_frame, remaining_steps, src,
                          ps.radius, dst, ship_count_for_legacy_scan_subindex(sn),
                          ship_speed_, runtime_dir_x, runtime_dir_y,
                          cached_turns_to_target, slot_planet_id,
                          slot_comet_internal_id, slot_radius);
                  const bool comet_first =
                      comet_hit.hit_slot >= 0 &&
                      (comet_hit.hit_steps < action_hit.hit_steps ||
                       (comet_hit.hit_steps == action_hit.hit_steps &&
                        comet_hit.hit_slot < action_hit.hit_slot));
                  if (comet_first) {
                    action_hit = comet_hit;
                  }
                }
                hit_kind[src][cls] = static_cast<float>(action_hit.hit_kind);
                hit_slot_last[src][cls] = action_hit.hit_slot;
                hit_steps_last[src][cls] = action_hit.hit_steps;
                dir_x[src][cls] = static_cast<float>(runtime_dir_x);
                dir_y[src][cls] = static_cast<float>(runtime_dir_y);
                turns[src][cls] = static_cast<float>(cached_turns_to_target);
                intercept_ok[src][cls] = 1.0f;
                intercept_fail_reason[src][cls] = 0.0f;
                continue;
              }
              const uint32_t cache_idx =
                  static_cast<uint32_t>(src * kHitClasses + cls);
              EdgeInterceptAim &aim = cached_intercept_aims[cache_idx];
              const bool intercept_valid = 
                  edge_intercept_aim_for_ship_count_and_aim_index(
                      noop, ps, dst, planets_row[static_cast<uint32_t>(dst)],
                      comet_planet_ids, ship_count_for_legacy_scan_subindex(sn),
                      ship_speed_, noop_base_frame, aim_index, &aim);
              if (!intercept_valid) {
                if (static_pair) {
                  static_pair_bucket_aim_terminal_cache_
                      [static_pair_bucket_aim_cache_index(aim_index, src, dst,
                                                          sn)] =
                          kStaticHitCacheInterceptFailed;
                  static_pair_bucket_aim_fail_reason_cache_
                      [static_pair_bucket_aim_cache_index(aim_index, src, dst,
                                                          sn)] =
                          static_cast<uint8_t>(aim.fail_reason);
                }
                intercept_failed_for_remaining_sn = true;
                intercept_fail_reason_for_remaining_sn =
                    static_cast<float>(aim.fail_reason);
                intercept_fail_source_for_remaining_sn = "edge_intercept_aim";
                record_intercept_failed_if_empty(
                    cls, intercept_fail_reason_for_remaining_sn,
                    intercept_fail_source_for_remaining_sn);
                continue;
              }
              EdgeActionHitWithAim action_hit_with_aim;
              if (static_pair && static_pair_cache_status == 0 &&
                  !static_pair_dynamic_possible) {
                const EdgeInterceptAimCandidate &candidate =
                    aim.candidates[static_cast<uint32_t>(aim_index)];
                TORCH_CHECK_DISABLED(candidate.valid,
                            "static-static no dynamic cache: missing candidate");
                const int32_t target_hit_steps =
                    static_cast<int32_t>(std::ceil(candidate.turns_to_target));
                EdgeActionHit action_hit;
                if (!(candidate.turns_to_target <
                      static_cast<double>(remaining_steps)) ||
                    target_hit_steps > kHonestHitTraceMaxSteps) {
                  action_hit =
                      EdgeActionHit{false, -1, kHitKindTimeout, kHitNone};
                } else {
                  TORCH_CHECK_DISABLED(target_hit_steps >= 1,
                              "static-static no dynamic cache: bad hit steps");
                  action_hit =
                      EdgeActionHit{true, target_hit_steps, kHitKindTarget, dst};
                }
                if (!comet_planet_ids.empty()) {
                  const EdgeActionHit comet_hit =
                      cached_dynamic_dynamic_comet_overlay_hit(
                          noop, noop_base_frame, remaining_steps, src,
                          ps.radius, dst, ship_count_for_legacy_scan_subindex(sn),
                          ship_speed_, candidate.dir_x, candidate.dir_y,
                          candidate.turns_to_target, slot_planet_id,
                          slot_comet_internal_id, slot_radius);
                  const bool comet_hit_valid = comet_hit.hit_slot >= 0;
                  const bool comet_first =
                      comet_hit_valid &&
                      (action_hit.hit_kind != kHitKindTarget ||
                       comet_hit.hit_steps < action_hit.hit_steps ||
                       (comet_hit.hit_steps == action_hit.hit_steps &&
                        comet_hit.hit_slot < action_hit.hit_slot));
                  if (comet_first) {
                    action_hit = comet_hit;
                  }
                }
                action_hit_with_aim =
                    EdgeActionHitWithAim{action_hit, true, aim_index,
                                         candidate.dir_x, candidate.dir_y,
                                         candidate.turns_to_target};
              } else {
                action_hit_with_aim =
                    static_pair && static_pair_cache_status == 0
                        ? edge_action_hit_for_static_checked_intercept_aim(
                              noop, noop_base_frame, remaining_steps, ps, src,
                              dst, ship_count_for_legacy_scan_subindex(sn),
                              ship_speed_, aim, aim_index, is_static_slot,
                              is_dynamic_slot, slot_planet_id,
                              slot_comet_internal_id, static_slots,
                              static_slots_n, dynamic_slots, dynamic_slots_n,
                              slot_radius)
                        : edge_action_hit_for_intercept_aim(
                              noop, noop_base_frame, remaining_steps, ps, src,
                              dst, ship_count_for_legacy_scan_subindex(sn),
                              ship_speed_, aim, aim_index, target_static,
                              has_target_hit_for_prior_bucket, is_static_slot,
                              is_dynamic_slot, slot_planet_id,
                              slot_comet_internal_id, static_slots,
                              static_slots_n, dynamic_slots, dynamic_slots_n,
                              slot_radius);
              }
              TORCH_CHECK_DISABLED(action_hit_with_aim.has_aim,
                          "honest_shared_action_mask_limited: hit scan without aim");
              const EdgeActionHit action_hit = action_hit_with_aim.hit;
              const int32_t kind = action_hit.hit_kind;
              const int32_t hit_slot = action_hit.hit_slot;
              hit_kind[src][cls] = static_cast<float>(kind);
              hit_slot_last[src][cls] = hit_slot;
              hit_steps_last[src][cls] = action_hit.hit_steps;
              dir_x[src][cls] = static_cast<float>(action_hit_with_aim.dir_x);
              dir_y[src][cls] = static_cast<float>(action_hit_with_aim.dir_y);
              turns[src][cls] =
                  static_cast<float>(action_hit_with_aim.turns_to_target);
              intercept_ok[src][cls] = 1.0f;
              intercept_fail_reason[src][cls] = 0.0f;
              if (target_static && kind == kHitKindTarget) {
                has_target_hit_for_prior_bucket = true;
              }
              const bool hit_static_non_target =
                  (hit_slot >= 0 && hit_slot < kPlanets &&
                   is_static_slot[static_cast<uint32_t>(hit_slot)] != 0 &&
                   hit_slot != dst);
              if (static_pair && kind == kHitKindTimeout) {
                const uint32_t bucket_cache_idx =
                    static_pair_bucket_aim_cache_index(aim_index, src, dst, sn);
                static_pair_bucket_aim_terminal_cache_[bucket_cache_idx] =
                    kStaticHitCacheTimeout;
              }
              if (target_static && (hit_static_non_target || hit_slot == kHitSun)) {
                if (static_pair) {
                  static_pair_aim_block_cache_[static_pair_aim_cache_index(
                      aim_index, src, dst)] =
                      hit_static_non_target ? kStaticHitCacheStaticBlocked
                                            : kStaticHitCacheSunBlocked;
                }
                blocked_for_remaining_sn = true;
                blocked_hit_kind = hit_static_non_target
                                       ? static_cast<float>(kHitKindStatic)
                                       : static_cast<float>(kHitKindSun);
              }
            }
          }
        }
      }
    }

    if (store_full_pair_cache) {
      WallProfileSpan profile_full_pair_cache_store(this, "full_pair_cache_store");
      assert(!apply_comet_overlay);
      const int32_t cache_planet_count = immutable_planet_prefix_n_;
      assert(cache_planet_count > 0 && cache_planet_count <= n);
      for (int32_t src = 0; src < cache_planet_count; ++src) {
        for (int32_t dst = 0; dst < cache_planet_count; ++dst) {
          if (dst == src) {
            continue;
          }
          const uint32_t pair_idx =
              static_cast<uint32_t>(src * kPlanets + dst);
          const int32_t pair_max_sn = output_max_sn_by_pair[pair_idx];
          if (pair_max_sn > 0 && max_sn_by_pair[pair_idx] > 0) {
            const int32_t pair_min_sn = min_sn_by_pair[pair_idx];
            assert(1 <= pair_min_sn && pair_min_sn <= pair_max_sn);
            store_honest_full_pair_cache_entry(
                episode_step, src, dst, pair_min_sn, pair_max_sn);
          }
        }
      }
    }
  }

  {
    WallProfileSpan profile_output(this, "output_mask");
    out_action_mask.zero_();
    auto available = out_action_mask.accessor<int8_t, 2>();
    auto hit_kind = last_honest_hit_kind_.accessor<float, 2>();
    auto hit_steps = last_honest_hit_steps_.accessor<int32_t, 2>();
    for (int32_t src = 0; src < kPlanets; ++src) {
      available[src][src * kHitClassesPerTarget] = 1;
    }
    for (int32_t src = 0; src < n; ++src) {
      for (int32_t dst = 0; dst < n; ++dst) {
        if (dst == src) {
          continue;
        }
        const int32_t pair_max_sn =
            output_max_sn_by_pair[static_cast<uint32_t>(src * kPlanets + dst)];
        if (pair_max_sn <= 0) {
          continue;
        }
        const int32_t class_base = dst * kHitClassesPerTarget;
        for (int32_t sn = 1; sn <= pair_max_sn; ++sn) {
          const int32_t cls = class_base + sn;
          if (static_cast<int32_t>(hit_kind[src][cls]) == kHitKindTarget) {
            TORCH_CHECK_DISABLED(1 <= hit_steps[src][cls] &&
                        hit_steps[src][cls] <= kHonestHitTraceMaxSteps,
                        "honest_shared_action_mask_limited: target hit_steps out of horizon");
            available[src][cls] = static_cast<int8_t>(hit_steps[src][cls]);
          }
        }
      }
    }
  }
}

void CppEnvStaticCacheV2::honest_shared_action_mask_limited(
    int32_t episode_step, torch::Tensor requests,
    torch::Tensor out_action_mask) const {
  TORCH_CHECK_DISABLED(requests.device().is_cpu(),
              "honest_shared_action_mask_limited requests: expected CPU tensor");
  TORCH_CHECK_DISABLED(requests.dtype() == torch::kInt32 ||
                  requests.dtype() == torch::kInt64,
              "honest_shared_action_mask_limited requests: expected int32 or int64");
  requests = requests.to(torch::kInt32).contiguous();
  TORCH_CHECK_DISABLED(requests.dim() == 2 && requests.size(1) == 3,
              "honest_shared_action_mask_limited requests must be [N,3]");
  honest_shared_action_mask_impl(
      episode_step,
      requests.data_ptr<int32_t>(),
      requests.size(0),
      false,
      false,
      out_action_mask,
      "honest_shared_action_mask_limited",
      true,
      false);
}

void CppEnvStaticCacheV2::honest_shared_action_mask_all_geometry(
    int32_t episode_step, torch::Tensor out_action_mask) const {
  honest_shared_action_mask_impl(
      episode_step,
      nullptr,
      0,
      false,
      true,
      out_action_mask,
      "honest_shared_action_mask_all_geometry",
      true,
      false);
}

py::tuple CppEnvStaticCacheV2::honest_shared_action_mask_full_cache_warmup_one(
    int32_t min_episode_step, torch::Tensor current_planet_rows,
    int32_t planet_count) const {
  WallProfileSpan profile_warmup_one(this, "honest_full_cache_warmup_one");
  assert(0 <= min_episode_step && min_episode_step <= episode_steps_);
  prune_honest_full_pair_cache_before(min_episode_step);
  assert(current_planet_rows.device().is_cpu());
  assert(current_planet_rows.scalar_type() == c10::ScalarType::Double);
  assert(current_planet_rows.dim() == 2);
  assert(current_planet_rows.size(0) == kPlanets);
  assert(current_planet_rows.size(1) == kPlanetRowLen);
  assert(planet_count > 0 && planet_count <= kPlanets);
  assert(!noop_cached_planets_flat_.empty());
  const NoopView noop = make_noop_view(noop_cached_planets_flat_, noop_spatial_grid_);
  assert(noop.n_frames == episode_steps_ + 1);
  assert(!honest_full_pair_cache_entry_by_key_.empty());
  assert(honest_full_step_src_done_max_sn_.size() ==
         static_cast<uint32_t>(noop.n_frames * kPlanets));
  auto current_planets = current_planet_rows.accessor<double, 2>();
  const int32_t n = immutable_planet_prefix_n_;
  assert(n > 0 && n <= planet_count);
  std::array<int32_t, kPlanets> src_order{};
  std::array<int32_t, kPlanets> src_ships{};
  std::array<int32_t, kPlanets> src_base_max_sn{};
  std::array<int32_t, kPlanets> src_max_sn{};
  bool observed_target_increased = false;
  for (int32_t src = 0; src < n; ++src) {
    src_order[static_cast<uint32_t>(src)] = src;
    const double ships_d = current_planets[src][5];
    assert(std::isfinite(ships_d));
    assert(ships_d >= 0.0);
    const int32_t ships = static_cast<int32_t>(ships_d);
    assert(static_cast<double>(ships) == ships_d);
    int32_t &max_observed_ships =
        honest_full_src_max_observed_ships_[static_cast<uint32_t>(src)];
    observed_target_increased = observed_target_increased || ships > max_observed_ships;
    max_observed_ships = std::max(max_observed_ships, ships);
    src_ships[static_cast<uint32_t>(src)] = max_observed_ships;
    src_base_max_sn[static_cast<uint32_t>(src)] =
        max_legacy_scan_subindex_for_available_ships(max_observed_ships);
  }
  std::sort(
      src_order.begin(), src_order.begin() + n,
      [&](int32_t a, int32_t b) {
        return src_ships[static_cast<uint32_t>(a)] >
               src_ships[static_cast<uint32_t>(b)];
      });
  std::vector<int32_t> request_data;
  request_data.reserve(static_cast<uint32_t>(kPlanets * 4));
  if (observed_target_increased ||
      honest_full_warmup_scan_step_ < min_episode_step) {
    honest_full_warmup_scan_step_ = min_episode_step;
    honest_full_warmup_scan_src_order_idx_ = 0;
  }
  assert(0 <= honest_full_warmup_scan_step_ &&
         honest_full_warmup_scan_step_ <= noop.n_frames);
  assert(0 <= honest_full_warmup_scan_src_order_idx_ &&
         honest_full_warmup_scan_src_order_idx_ < n);
  int32_t total_warmed_ship_buckets = 0;
  while (true) {
    bool can_raise_extra_sn = false;
    for (int32_t src = 0; src < n; ++src) {
      const int32_t target_sn = std::min(
          kLegacyShipScanClasses - 1,
          src_base_max_sn[static_cast<uint32_t>(src)] +
              honest_full_warmup_extra_sn_);
      src_max_sn[static_cast<uint32_t>(src)] = target_sn;
      can_raise_extra_sn = can_raise_extra_sn || target_sn < kLegacyShipScanClasses - 1;
    }
    const int32_t search_end =
        can_raise_extra_sn
            ? std::min(noop.n_frames,
                       min_episode_step + 10 * kHonestFullCacheWarmupHorizonSteps)
            : noop.n_frames;
    if (honest_full_warmup_scan_step_ >= search_end) {
      if (!can_raise_extra_sn) {
        honest_full_warmup_scan_step_ = noop.n_frames;
        honest_full_warmup_scan_src_order_idx_ = 0;
        honest_full_warmup_last_lookahead_steps_ = noop.n_frames - min_episode_step;
        return py::make_tuple(total_warmed_ship_buckets, true);
      }
      ++honest_full_warmup_extra_sn_;
      honest_full_warmup_scan_step_ = min_episode_step;
      honest_full_warmup_scan_src_order_idx_ = 0;
      honest_full_warmup_last_lookahead_steps_ = search_end - min_episode_step;
      continue;
    }
    while (honest_full_warmup_scan_step_ < search_end) {
      honest_full_warmup_last_lookahead_steps_ =
          honest_full_warmup_scan_step_ - min_episode_step + 1;
      const int32_t t = honest_full_warmup_scan_step_;
      request_data.clear();
      while (honest_full_warmup_scan_src_order_idx_ < n) {
        {
          WallProfileSpan profile_candidate_search(this, "warmup_candidate_search");
          const int32_t src_i = honest_full_warmup_scan_src_order_idx_;
          const int32_t src = src_order[static_cast<uint32_t>(src_i)];
          const int32_t max_sn = src_max_sn[static_cast<uint32_t>(src)];
          if (max_sn <= 0) {
            ++honest_full_warmup_scan_src_order_idx_;
            continue;
          }
          const uint32_t step_src_idx =
              static_cast<uint32_t>(t * kPlanets + src);
          assert(step_src_idx < honest_full_step_src_done_max_sn_.size());
          const int32_t step_src_done_max_sn =
              static_cast<int32_t>(honest_full_step_src_done_max_sn_[step_src_idx]);
          assert(0 <= step_src_done_max_sn &&
                 step_src_done_max_sn < kLegacyShipScanClasses);
          if (step_src_done_max_sn >= max_sn) {
            ++honest_full_warmup_scan_src_order_idx_;
            continue;
          }
          const NoopCachedPlanet *planets_row =
              noop.flat + static_cast<uint32_t>(t) * static_cast<uint32_t>(kPlanets);
          assert(static_cast<int32_t>(current_planets[src][0]) ==
                 planets_row[static_cast<uint32_t>(src)].id);
          for (int32_t dst = 0; dst < n; ++dst) {
            if (dst == src) {
              continue;
            }
            const uint32_t key = honest_full_pair_cache_key(t, src, dst);
            assert(key < honest_full_pair_cache_entry_by_key_.size());
            const int32_t entry_idx = honest_full_pair_cache_entry_by_key_[key];
            int32_t filled_max_sn = 0;
            if (entry_idx >= 0) {
              assert(t < static_cast<int32_t>(
                             honest_full_pair_cache_entries_by_step_.size()));
              const std::vector<HonestFullPairCacheEntry> &step_entries =
                  honest_full_pair_cache_entries_by_step_[static_cast<uint32_t>(t)];
              assert(entry_idx < static_cast<int32_t>(step_entries.size()));
              filled_max_sn =
                  step_entries[static_cast<uint32_t>(entry_idx)].filled_max_sn;
            }
            assert(0 <= filled_max_sn && filled_max_sn < kLegacyShipScanClasses);
            if (filled_max_sn >= max_sn) {
              continue;
            }
            const int32_t min_sn = filled_max_sn + 1;
            assert(1 <= min_sn && min_sn <= max_sn);
            request_data.push_back(src);
            request_data.push_back(dst);
            request_data.push_back(min_sn);
            request_data.push_back(max_sn);
            total_warmed_ship_buckets += max_sn - min_sn + 1;
          }
          honest_full_step_src_done_max_sn_[step_src_idx] =
              static_cast<uint8_t>(max_sn);
          ++honest_full_warmup_scan_src_order_idx_;
        }
        if (total_warmed_ship_buckets >=
            kHonestFullCacheWarmupMaxShipBuckets) {
          break;
        }
      }
      if (!request_data.empty()) {
        assert(request_data.size() % 4 == 0);
        honest_shared_action_mask_impl(
            t,
            request_data.data(),
            static_cast<int32_t>(request_data.size() / 4),
            true,
            false,
            honest_available_hit_mask_scratch_,
            "honest_shared_action_mask_full_cache_warmup",
            false,
            true);
        if (total_warmed_ship_buckets >=
            kHonestFullCacheWarmupMaxShipBuckets) {
          assert(0 <= honest_full_warmup_scan_src_order_idx_ &&
                 honest_full_warmup_scan_src_order_idx_ <= n);
          if (honest_full_warmup_scan_src_order_idx_ == n) {
            honest_full_warmup_scan_src_order_idx_ = 0;
            ++honest_full_warmup_scan_step_;
          }
          return py::make_tuple(total_warmed_ship_buckets, false);
        }
      }
      honest_full_warmup_scan_src_order_idx_ = 0;
      ++honest_full_warmup_scan_step_;
    }
  }
}

int32_t CppEnvStaticCacheV2::honest_shared_action_mask_full_cache_prune_before(
    int32_t min_episode_step) const {
  return prune_honest_full_pair_cache_before(min_episode_step);
}

py::tuple CppEnvStaticCacheV2::honest_shared_action_mask_full_cache_warmup_stats() const {
  assert(!noop_cached_planets_flat_.empty());
  const NoopView noop = make_noop_view(noop_cached_planets_flat_, noop_spatial_grid_);
  assert(noop.n_frames == episode_steps_ + 1);
  const int32_t cursor = honest_full_warmup_scan_step_;
  assert(0 <= cursor && cursor <= noop.n_frames);
  const int32_t src_total = immutable_planet_prefix_n_;
  assert(src_total > 0 && src_total <= kPlanets);
  const int32_t cursor_done =
      cursor < noop.n_frames ? honest_full_warmup_scan_src_order_idx_
                             : src_total;
  assert(0 <= cursor_done && cursor_done <= src_total);
  const int32_t live_entries = honest_full_pair_cache_live_entries_;
  assert(live_entries >= 0);
  assert(honest_full_warmup_extra_sn_ >= 0);
  assert(honest_full_warmup_last_lookahead_steps_ >= 0);
  return py::make_tuple(
      cursor,
      cursor_done,
      src_total,
      live_entries,
      honest_full_warmup_extra_sn_,
      honest_full_warmup_last_lookahead_steps_);
}

void CppEnvStaticCacheV2::send_action_hit_mask_from_state(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, torch::Tensor out_action_mask) const {
  WallProfileSpan profile(this, "send_action_hit_mask_from_state");
  assert_cpu_int8(out_action_mask, "send_action_hit_mask out_action_mask");
  TORCH_CHECK_DISABLED(out_action_mask.dim() == 2 && out_action_mask.size(0) == kPlanets &&
                  out_action_mask.size(1) == kHitClasses,
              "send_action_hit_mask out_action_mask shape");
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "send_action_hit_mask: noop cache empty");
  const int32_t planet_count = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(planet_count > 0 && planet_count <= kPlanets,
              "send_action_hit_mask: planet_count");
  const int32_t noop_base_frame = episode_step;
  const NoopView noop = make_noop_view(noop_cached_planets_flat_, noop_spatial_grid_);
  TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
              "send_action_hit_mask: noop cache missing frame");
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
  int32_t n = 0;
  while (n < kPlanets && planets_row[static_cast<uint32_t>(n)].id >= 0) {
    ++n;
  }
  TORCH_CHECK_DISABLED(planet_count == n, "send_action_hit_mask: planet_count/noop mismatch");

  std::array<double, kPlanets> slot_radius{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  for (int32_t i = 0; i < kPlanets; ++i) {
    slot_radius[static_cast<uint32_t>(i)] = 0.0;
    slot_planet_id[static_cast<uint32_t>(i)] = -1;
    slot_comet_internal_id[static_cast<uint32_t>(i)] = -1;
  }
  for (int32_t i = 0; i < n; ++i) {
    const NoopCachedPlanet &p = planets_row[static_cast<uint32_t>(i)];
    slot_radius[static_cast<uint32_t>(i)] = p.radius;
    slot_planet_id[static_cast<uint32_t>(i)] = p.id;
    slot_comet_internal_id[static_cast<uint32_t>(i)] = p.comet_internal_id;
  }

  std::array<int32_t, kPlanets> static_slots{};
  std::array<int32_t, kPlanets> dynamic_slots{};
  int32_t static_slots_n = 0;
  int32_t dynamic_slots_n = 0;
  std::array<uint8_t, kPlanets> is_static_slot{};
  std::array<uint8_t, kPlanets> is_dynamic_slot{};
  fill_mask_slot_arrays_from_cache(
      planet_slot_static_, planet_slot_static_n_, planet_slot_orbiting_,
      planet_slot_orbiting_n_, planet_slot_comet_, planet_slot_comet_n_,
      planets_row, static_slots, static_slots_n, dynamic_slots,
      dynamic_slots_n, is_static_slot, is_dynamic_slot);

  const SmallPlanetIdSet comet_planet_ids =
      active_comet_planet_ids_at_frame(planets_row, planet_slot_comet_,
                                       planet_slot_comet_n_);
  const int32_t remaining_steps = noop.n_frames - noop_base_frame;
  TORCH_CHECK_DISABLED(remaining_steps > 0, "send_action_hit_mask: remaining steps");
  torch::Tensor arrivals;
  torch::Tensor takeover_costs;
  torch::Tensor stable_takeover_costs;
  {
    WallProfileSpan profile_costs(this, "send_action_arrivals_and_costs");
    arrivals = fleet_arrivals_for_fleets(
        fleets, kPlanetArrivalHorizon, num_agents_, ship_speed_, noop_base_frame,
        comet_planet_ids, noop_cached_planets_flat_, noop_spatial_grid_);
    takeover_costs =
        orbit_wars_honest::takeover_cost_abs_int64_from_arrivals(
            arrivals, planets, num_agents_);
    stable_takeover_costs =
        orbit_wars_honest::stable_takeover_cost_abs_int64_from_arrivals(
            arrivals, planets, num_agents_);
  }

  {
    WallProfileSpan profile_reset(this, "send_action_output_reset");
    out_action_mask.zero_();
    last_honest_send_ships_.zero_();
  }
  auto available = out_action_mask.accessor<int8_t, 2>();
  auto send_ships_last = last_honest_send_ships_.accessor<int32_t, 2>();
  auto dir_x = last_honest_dir_x_.accessor<float, 2>();
  auto dir_y = last_honest_dir_y_.accessor<float, 2>();
  auto turns = last_honest_turns_.accessor<float, 2>();
  auto hit_kind = last_honest_hit_kind_.accessor<float, 2>();
  auto hit_slot_last = last_honest_hit_slot_.accessor<int32_t, 2>();
  auto hit_steps_last = last_honest_hit_steps_.accessor<int32_t, 2>();
  auto intercept_fail_reason =
      last_honest_intercept_fail_reason_.accessor<float, 2>();
  for (int32_t src = 0; src < kPlanets; ++src) {
    for (int32_t dst = 0; dst < kPlanets; ++dst) {
      if (dst == src) {
        continue;
      }
      for (const int32_t send_subindex : kMoveSendSubindices) {
        const int32_t cls = dst * kHitClassesPerTarget + send_subindex;
        hit_kind[src][cls] = static_cast<float>(kHitKindNone);
        hit_slot_last[src][cls] = kHitNone;
        hit_steps_last[src][cls] = -1;
        intercept_fail_reason[src][cls] = 0.0f;
      }
    }
  }
  for (int32_t src = 0; src < n; ++src) {
    const NoopCachedPlanet &ps = planets_row[static_cast<uint32_t>(src)];
    const Planet &current_src = planets[static_cast<uint32_t>(src)];
    TORCH_CHECK_DISABLED(current_src.id == ps.id,
                "send_action_hit_mask: current/noop source planet id mismatch");
    if (!honest_source_planet_can_launch(current_src)) {
      continue;
    }
    const int32_t source_ship_count = static_cast<int32_t>(current_src.ships);
    TORCH_CHECK_DISABLED(static_cast<double>(source_ship_count) == current_src.ships,
                "send_action_hit_mask: source ships must be integer");
    for (int32_t dst = 0; dst < n; ++dst) {
      if (dst == src) {
        continue;
      }
      const NoopCachedPlanet &pd = planets_row[static_cast<uint32_t>(dst)];
      for (const int32_t send_subindex : kMoveSendSubindices) {
        const int32_t cls = dst * kHitClassesPerTarget + send_subindex;
        if (send_subindex == kMoveClassSendTakeoverSubindex ||
            send_subindex == kMoveClassSendStableTakeoverSubindex) {
          const torch::Tensor costs =
              send_subindex == kMoveClassSendTakeoverSubindex
                  ? takeover_costs
                  : stable_takeover_costs;
          TakeoverActionCandidate candidate;
          {
            const char *profile_name =
                send_subindex == kMoveClassSendTakeoverSubindex
                    ? "send_action_takeover_candidates"
                    : "send_action_stable_takeover_candidates";
            WallProfileSpan profile_takeover(this, profile_name);
            candidate = best_takeover_action_candidate_for_pair(
                this, noop, noop_base_frame, remaining_steps, ps, src, dst,
                pd, comet_planet_ids, source_ship_count, current_src.owner,
                ship_speed_, costs, is_static_slot, is_dynamic_slot,
                slot_planet_id, slot_comet_internal_id, static_slots,
                static_slots_n, dynamic_slots, dynamic_slots_n, slot_radius);
          }
          if (!candidate.available) {
            continue;
          }
          const EdgeActionHit action_hit = candidate.hit_with_aim.hit;
          TORCH_CHECK_DISABLED(action_hit.hit_kind == kHitKindTarget,
                               "takeover candidate: non-target hit");
          TORCH_CHECK_DISABLED(action_hit.hit_slot == dst,
                               "takeover candidate: wrong target slot");
          TORCH_CHECK_DISABLED(1 <= action_hit.hit_steps &&
                                   action_hit.hit_steps <= kHonestHitTraceMaxSteps,
                               "takeover candidate: hit_steps out of horizon");
          hit_kind[src][cls] = static_cast<float>(action_hit.hit_kind);
          hit_slot_last[src][cls] = action_hit.hit_slot;
          hit_steps_last[src][cls] = action_hit.hit_steps;
          intercept_fail_reason[src][cls] = 0.0f;
          dir_x[src][cls] = static_cast<float>(candidate.hit_with_aim.dir_x);
          dir_y[src][cls] = static_cast<float>(candidate.hit_with_aim.dir_y);
          turns[src][cls] =
              static_cast<float>(candidate.hit_with_aim.turns_to_target);
          send_ships_last[src][cls] = candidate.ship_count;
          available[src][cls] = static_cast<int8_t>(action_hit.hit_steps);
          continue;
        }
        const int32_t send_ship_count =
            ship_count_for_move_subindex(source_ship_count, send_subindex);
        bool has_action_hit = false;
        EdgeActionHitWithAim action_hit_with_aim;
        float last_fail_reason = 0.0f;
        {
          const char *profile_name =
              send_subindex == kMoveClassSendAllSubindex
                  ? "send_action_all_candidates"
                  : "send_action_half_candidates";
          WallProfileSpan profile_fixed_send(this, profile_name);
          for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
            EdgeInterceptAim aim;
            const bool valid = edge_intercept_aim_for_ship_count_and_aim_index(
                noop, ps, dst, pd, comet_planet_ids, send_ship_count,
                ship_speed_, noop_base_frame, aim_index, &aim);
            if (!valid) {
              last_fail_reason = static_cast<float>(aim.fail_reason);
              continue;
            }
            const EdgeActionHitWithAim candidate_hit_with_aim =
                edge_action_hit_for_intercept_aim(
                    noop, noop_base_frame, remaining_steps, ps, src, dst,
                    send_ship_count, ship_speed_, aim, aim_index, false, false,
                    is_static_slot, is_dynamic_slot, slot_planet_id,
                    slot_comet_internal_id, static_slots, static_slots_n,
                    dynamic_slots, dynamic_slots_n, slot_radius);
            TORCH_CHECK_DISABLED(candidate_hit_with_aim.has_aim,
                        "send_action_hit_mask: valid aim missing hit");
            if (!has_action_hit) {
              action_hit_with_aim = candidate_hit_with_aim;
              has_action_hit = true;
            }
            if (candidate_hit_with_aim.hit.hit_kind == kHitKindTarget) {
              action_hit_with_aim = candidate_hit_with_aim;
              break;
            }
          }
        }
        if (!has_action_hit) {
          hit_kind[src][cls] = static_cast<float>(kHitKindInterceptionFailed);
          intercept_fail_reason[src][cls] = last_fail_reason;
          continue;
        }
        const EdgeActionHit action_hit = action_hit_with_aim.hit;
        hit_kind[src][cls] = static_cast<float>(action_hit.hit_kind);
        hit_slot_last[src][cls] = action_hit.hit_slot;
        hit_steps_last[src][cls] = action_hit.hit_steps;
        intercept_fail_reason[src][cls] = 0.0f;
        if (action_hit.hit_kind == kHitKindTarget) {
          TORCH_CHECK_DISABLED(action_hit_with_aim.has_aim,
                      "send_action_hit_mask: target hit without aim");
          TORCH_CHECK_DISABLED(1 <= action_hit.hit_steps &&
                      action_hit.hit_steps <= kHonestHitTraceMaxSteps,
                      "send_action_hit_mask: target hit_steps out of horizon");
          dir_x[src][cls] = static_cast<float>(action_hit_with_aim.dir_x);
          dir_y[src][cls] = static_cast<float>(action_hit_with_aim.dir_y);
          turns[src][cls] =
              static_cast<float>(action_hit_with_aim.turns_to_target);
          send_ships_last[src][cls] = send_ship_count;
          available[src][cls] = static_cast<int8_t>(action_hit.hit_steps);
        }
      }
    }
  }
}

void CppEnvStaticCacheV2::send_all_from_external(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, torch::Tensor out_action_mask) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count, "send_all_from_external");
  send_action_hit_mask_from_state(episode_step, fleets, planets, out_action_mask);
}

void CppEnvStaticCacheV2::fill_available_action_mask_from_hit_mask(
    const std::vector<Planet> &planets, torch::Tensor send_action_hit_mask,
    torch::Tensor out_action_mask) const {
  WallProfileSpan profile(this, "action_mask_from_send_all");
  assert_cpu_int8(out_action_mask, "available_action_mask");
  assert_cpu_int8(send_action_hit_mask, "available_action_mask send_action_hit_mask");
  TORCH_CHECK_DISABLED(out_action_mask.sizes() ==
                  torch::IntArrayRef({kPlayerAxisSlots, kPlanets, kMoveClasses}),
              "available_action_mask shape");
  TORCH_CHECK_DISABLED(send_action_hit_mask.sizes() ==
                  torch::IntArrayRef({kPlanets, kHitClasses}),
              "available_action_mask send_action_hit_mask shape");
  const int32_t planet_count = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(0 < planet_count && planet_count <= kPlanets,
              "available action mask: planet_count");
  auto send_action_hit = send_action_hit_mask.accessor<int8_t, 2>();
  auto send_ships_last = last_honest_send_ships_.accessor<int32_t, 2>();
  auto out = out_action_mask.accessor<int8_t, 3>();
  std::fill(out_action_mask.data_ptr<int8_t>(),
            out_action_mask.data_ptr<int8_t>() + out_action_mask.numel(),
            static_cast<int8_t>(0));
  for (int32_t compact_agent = 0; compact_agent < num_agents_;
       ++compact_agent) {
    const int32_t policy_slot =
        policy_slot_for_compact_agent(compact_agent, num_agents_);
    for (int32_t src = 0; src < kPlanets; ++src) {
      out[policy_slot][src][src * kMoveClassesPerTarget + kMoveClassNoopSubindex] = 1;
    }
    for (int32_t src = 0; src < planet_count; ++src) {
      const Planet &planet = planets[static_cast<uint32_t>(src)];
      if (planet.owner != compact_agent || planet.ships <= 0.0) {
        continue;
      }
      for (int32_t dst = 0; dst < planet_count; ++dst) {
        if (dst == src) {
          continue;
        }
        for (const int32_t send_subindex : kMoveSendSubindices) {
          if (send_subindex == kMoveClassSendHalfSubindex &&
              planet.ships < 2.0) {
            continue;
          }
          const int32_t hit_cls = dst * kHitClassesPerTarget + send_subindex;
          const int32_t action_cls = dst * kMoveClassesPerTarget + send_subindex;
          if (send_subindex == kMoveClassSendTakeoverSubindex) {
            const int32_t stable_hit_cls =
                dst * kHitClassesPerTarget + kMoveClassSendStableTakeoverSubindex;
            if (send_action_hit[src][stable_hit_cls] > 0 &&
                send_ships_last[src][hit_cls] == send_ships_last[src][stable_hit_cls]) {
              continue;
            }
          }
          if (send_action_hit[src][hit_cls] > 0) {
            out[policy_slot][src][action_cls] = 1;
          }
        }
      }
    }
  }
}

void CppEnvStaticCacheV2::fill_available_action_mask_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, torch::Tensor out_action_mask) const {
  TORCH_CHECK_DISABLED(0 < planet_count && planet_count <= kPlanets,
              "available action mask: planet_count");
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count, "fill_available_action_mask_from_rows");
  TORCH_CHECK_DISABLED(static_cast<int32_t>(planets.size()) == planet_count,
              "available action mask: planet rows/count mismatch");
  send_action_hit_mask_from_state(
      episode_step, fleets, planets, honest_available_hit_mask_scratch_);
  fill_available_action_mask_from_hit_mask(
      planets, honest_available_hit_mask_scratch_, out_action_mask);
}

double CppEnvStaticCacheV2::honest_shared_angle_or_nan_from_state(
    int32_t episode_step, const std::vector<Planet> &planets,
    int32_t src_slot, int32_t dst_slot, int32_t ship_count) const {
  const int32_t planet_count = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(planet_count > 0 && planet_count <= kPlanets,
              "honest_shared_angle_from_state: planet_count");
  TORCH_CHECK_DISABLED(0 <= src_slot && src_slot < planet_count,
              "honest_shared_angle_from_state: bad src_slot");
  TORCH_CHECK_DISABLED(0 <= dst_slot && dst_slot < planet_count,
              "honest_shared_angle_from_state: bad dst_slot");
  TORCH_CHECK_DISABLED(src_slot != dst_slot,
              "honest_shared_angle_from_state: src_slot == dst_slot");
  TORCH_CHECK_DISABLED(ship_count > 0, "honest_shared_angle_from_state: ship_count");
  TORCH_CHECK_DISABLED(!noop_cached_planets_flat_.empty(),
              "honest_shared_angle_from_state: noop cache empty");
  const int32_t noop_base_frame = episode_step;
  const NoopView noop = make_noop_view(noop_cached_planets_flat_, noop_spatial_grid_);
  TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
              "honest_shared_angle_from_state: noop cache missing frame");
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
  int32_t n = 0;
  while (n < kPlanets && planets_row[static_cast<uint32_t>(n)].id >= 0) {
    ++n;
  }
  TORCH_CHECK_DISABLED(planet_count == n,
              "honest_shared_angle_from_state: planet_count/noop mismatch");
  const NoopCachedPlanet &ps = planets_row[static_cast<uint32_t>(src_slot)];
  const NoopCachedPlanet &pd = planets_row[static_cast<uint32_t>(dst_slot)];
  TORCH_CHECK_DISABLED(ps.id >= 0 && pd.id >= 0,
              "honest_shared_angle_from_state: missing cached slot");
  const Planet &current_src = planets[static_cast<uint32_t>(src_slot)];
  TORCH_CHECK_DISABLED(current_src.id == ps.id,
              "honest_shared_angle_from_state: current/noop source planet id mismatch");
  if (!honest_source_planet_can_launch(current_src)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const SmallPlanetIdSet comet_planet_ids =
      active_comet_planet_ids_at_frame(planets_row, planet_slot_comet_,
                                       planet_slot_comet_n_);
  EdgeInterceptAim aim;
  const bool valid = edge_intercept_aim_for_ship_count(
      noop, ps, dst_slot, pd, comet_planet_ids, ship_count, ship_speed_,
      noop_base_frame, &aim);
  if (!valid) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::array<double, kPlanets> slot_radius{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    slot_planet_id[static_cast<uint32_t>(slot)] = -1;
    slot_comet_internal_id[static_cast<uint32_t>(slot)] = -1;
  }
  for (int32_t slot = 0; slot < n; ++slot) {
    const NoopCachedPlanet &p = planets_row[static_cast<uint32_t>(slot)];
    slot_radius[static_cast<uint32_t>(slot)] = p.radius;
    slot_planet_id[static_cast<uint32_t>(slot)] = p.id;
    slot_comet_internal_id[static_cast<uint32_t>(slot)] = p.comet_internal_id;
  }
  std::array<int32_t, kPlanets> static_slots{};
  std::array<int32_t, kPlanets> dynamic_slots{};
  int32_t static_slots_n = 0;
  int32_t dynamic_slots_n = 0;
  std::array<uint8_t, kPlanets> is_static_slot{};
  std::array<uint8_t, kPlanets> is_dynamic_slot{};
  fill_mask_slot_arrays_from_cache(
      planet_slot_static_, planet_slot_static_n_, planet_slot_orbiting_,
      planet_slot_orbiting_n_, planet_slot_comet_, planet_slot_comet_n_,
      planets_row, static_slots, static_slots_n, dynamic_slots,
      dynamic_slots_n, is_static_slot, is_dynamic_slot);

  const int32_t remaining_steps = noop.n_frames - noop_base_frame;
  TORCH_CHECK_DISABLED(remaining_steps > 0,
              "honest_shared_angle_from_state: remaining steps");
  EdgeActionHitWithAim action_hit_with_aim = edge_action_hit_with_intercept_aims(
      noop, noop_base_frame, remaining_steps, ps, src_slot, dst_slot,
      ship_count, ship_speed_, aim,
      is_static_slot[static_cast<uint32_t>(dst_slot)] != 0, false,
      is_static_slot, is_dynamic_slot, slot_planet_id, slot_comet_internal_id,
      static_slots, static_slots_n, dynamic_slots, dynamic_slots_n,
      slot_radius);
  if (action_hit_with_aim.hit.hit_kind != kHitKindTarget) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  TORCH_CHECK_DISABLED(action_hit_with_aim.has_aim,
              "honest_shared_angle_from_state: target hit without aim");
  return std::atan2(action_hit_with_aim.dir_y, action_hit_with_aim.dir_x);
}

torch::Tensor CppEnvStaticCacheV2::honest_shared_intercept_trace_from_state(
    int32_t episode_step, const std::vector<Planet> &planets,
    int32_t src_slot, int32_t dst_slot, int32_t ship_subindex) const {
  torch::Tensor out =
      torch::full({kHonestInterceptMaxIters, 4},
                  std::numeric_limits<float>::quiet_NaN(),
                  torch::TensorOptions().dtype(torch::kFloat32));
  if (ship_subindex <= 0 || ship_subindex >= kLegacyShipScanClasses) {
    return out;
  }
  const int32_t planet_count = static_cast<int32_t>(planets.size());
  if (planet_count <= 0 || planet_count > kPlanets) {
    return out;
  }
  if (src_slot < 0 || src_slot >= planet_count || dst_slot < 0 ||
      dst_slot >= planet_count || src_slot == dst_slot) {
    return out;
  }
  if (noop_cached_planets_flat_.empty()) {
    return out;
  }
  const int32_t noop_base_frame = episode_step;
  const NoopView noop = make_noop_view(noop_cached_planets_flat_, noop_spatial_grid_);
  if (noop_base_frame < 0 || noop_base_frame >= noop.n_frames) {
    return out;
  }
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
  int32_t n = 0;
  while (n < kPlanets && planets_row[static_cast<uint32_t>(n)].id >= 0) {
    ++n;
  }
  TORCH_CHECK_DISABLED(planet_count == n,
              "honest_shared_intercept_trace_from_state: planet_count/noop mismatch");
  const NoopCachedPlanet &ps = planets_row[static_cast<uint32_t>(src_slot)];
  const NoopCachedPlanet &pd = planets_row[static_cast<uint32_t>(dst_slot)];
  if (ps.id < 0 || pd.id < 0) {
    return out;
  }
  const Planet &current_src = planets[static_cast<uint32_t>(src_slot)];
  TORCH_CHECK_DISABLED(current_src.id == ps.id,
              "honest_shared_intercept_trace_from_state: current/noop source planet id mismatch");
  if (!honest_source_planet_can_launch(current_src)) {
    return out;
  }
  const SmallPlanetIdSet comet_planet_ids =
      active_comet_planet_ids_at_frame(planets_row, planet_slot_comet_,
                                       planet_slot_comet_n_);
  const int32_t ship_count = ship_count_for_legacy_scan_subindex(ship_subindex);
  EdgeInterceptAim aim;
  const bool valid = edge_intercept_aim_for_ship_count(
      noop, ps, dst_slot, pd, comet_planet_ids, ship_count, ship_speed_,
      noop_base_frame, &aim);
  if (!valid) {
    return out;
  }
  auto a = out.accessor<float, 2>();
  for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
    const EdgeInterceptAimCandidate &candidate =
        aim.candidates[static_cast<uint32_t>(aim_index)];
    if (!candidate.valid) {
      continue;
    }
    a[aim_index][0] = static_cast<float>(candidate.aim_x);
    a[aim_index][1] = static_cast<float>(candidate.aim_y);
    a[aim_index][2] = static_cast<float>(candidate.turns_to_target);
    a[aim_index][3] = 1.0f;
  }
  return out;
}

torch::Tensor CppEnvStaticCacheV2::honest_shared_hit_kind_last() const {
  return last_honest_hit_kind_;
}

torch::Tensor CppEnvStaticCacheV2::honest_shared_hit_slot_last() const {
  return last_honest_hit_slot_;
}

torch::Tensor CppEnvStaticCacheV2::honest_shared_hit_steps_last() const {
  return last_honest_hit_steps_;
}

torch::Tensor CppEnvStaticCacheV2::honest_shared_intercept_fail_reason_last() const {
  return last_honest_intercept_fail_reason_;
}

torch::Tensor CppEnvStaticCacheV2::honest_shared_send_ships_last() const {
  return last_honest_send_ships_;
}

py::tuple CppEnvStaticCacheV2::honest_shared_dir_last() const {
  return py::make_tuple(last_honest_dir_x_, last_honest_dir_y_);
}

double CppEnvStaticCacheV2::honest_shared_angle_last_for_action_class(
    int32_t src_slot, int32_t action_class) const {
  TORCH_CHECK_DISABLED(0 <= src_slot && src_slot < kPlanets,
                       "honest_shared_angle_last: src_slot");
  TORCH_CHECK_DISABLED(0 <= action_class && action_class < kMoveClasses,
                       "honest_shared_angle_last: action_class");
  const int32_t dst_slot = action_class / kMoveClassesPerTarget;
  const int32_t move_subindex = action_class % kMoveClassesPerTarget;
  TORCH_CHECK_DISABLED(move_subindex_is_send_action(move_subindex),
                       "honest_shared_angle_last: send subindex");
  const int32_t hit_cls = dst_slot * kHitClassesPerTarget + move_subindex;
  auto dir_x = last_honest_dir_x_.accessor<float, 2>();
  auto dir_y = last_honest_dir_y_.accessor<float, 2>();
  const double x = static_cast<double>(dir_x[src_slot][hit_cls]);
  const double y = static_cast<double>(dir_y[src_slot][hit_cls]);
  TORCH_CHECK_DISABLED(std::isfinite(x) && std::isfinite(y),
                       "honest_shared_angle_last: non-finite cached dir");
  return std::atan2(y, x);
}
