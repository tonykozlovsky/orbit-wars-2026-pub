#pragma once

#include "common.h"

#include <array>
#include <vector>

namespace orbit_wars_honest {

constexpr int32_t kHitNone = -1;
constexpr int32_t kHitSun = -2;
constexpr int32_t kHitOutOfBoard = -3;
constexpr int32_t kHitKindNone = 0;
constexpr int32_t kHitKindTarget = 1;
constexpr int32_t kHitKindStatic = 2;
constexpr int32_t kHitKindDynamic = 3;
constexpr int32_t kHitKindSun = 4;
constexpr int32_t kHitKindOutOfBoard = 5;
constexpr int32_t kHitKindTimeout = 6;
constexpr int32_t kHitKindEndOfGame = 7;
constexpr int32_t kHitKindInterceptionFailed = 8;
constexpr int32_t kHitKindVerifiedTimeout = 9;

constexpr int32_t kInterceptFailReasonNone = 0;
constexpr int32_t kInterceptFailReasonStaticZeroNorm = 1;
constexpr int32_t kInterceptFailReasonStaticBadTurns = 2;
constexpr int32_t kInterceptFailReasonDynamicSeedInvalid = 3;
constexpr int32_t kInterceptFailReasonDynamicSolverNoConverge = 4;
constexpr int32_t kInterceptFailReasonDynamicNonFiniteAim = 5;
constexpr int32_t kInterceptFailReasonDynamicZeroNorm = 6;
constexpr int32_t kInterceptFailReasonDynamicBadTurns = 7;
constexpr int32_t kEdgeInterceptAimCount = 3;
constexpr int32_t kEdgeInterceptAimCenter = 0;
constexpr int32_t kEdgeInterceptAimLeft = 1;
constexpr int32_t kEdgeInterceptAimRight = 2;
constexpr double kEdgeInterceptTangentRadiusFrac = 0.999;

constexpr int32_t kHonestInterceptMaxIters = 32;

constexpr double kHonestInterceptTurnsEpsilon = 0.0001;

constexpr int32_t kHonestHitTraceMaxSteps = kPlanetArrivalHorizon;

struct EdgeActionHit {
  bool available = false;
  int32_t hit_steps = -1;
  int32_t hit_kind = kHitKindNone;
  int32_t hit_slot = kHitNone;
};

struct EdgeActionHitWithAim {
  EdgeActionHit hit{};
  bool has_aim = false;
  int32_t aim_index = -1;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double turns_to_target = 0.0;
};

struct EdgeInterceptAimCandidate {
  bool valid = false;
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double turns_to_target = 0.0;
  int32_t fail_reason = kInterceptFailReasonNone;
};

struct EdgeInterceptAim {
  std::array<EdgeInterceptAimCandidate, kEdgeInterceptAimCount> candidates{};
  int32_t fail_reason = kInterceptFailReasonNone;
};

struct EdgeInterceptDebugTargetPoint {
  int32_t step = 0;
  double x = 0.0;
  double y = 0.0;
};

struct EdgeInterceptDebugSegment {
  int32_t iter = 0;
  int32_t branch = 0;
  double target_center_x = 0.0;
  double target_center_y = 0.0;
  double aim_x = 0.0;
  double aim_y = 0.0;
  double start_x = 0.0;
  double start_y = 0.0;
  double end_x = 0.0;
  double end_y = 0.0;
  double turns = 0.0;
};

struct EdgeInterceptDebugSolverResult {
  bool valid = false;
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double turns = 0.0;
};

struct EdgeInterceptDebugSolverResults {
  EdgeInterceptDebugSolverResult point{};
  EdgeInterceptDebugSolverResult bisect{};
  EdgeInterceptDebugSolverResult hybrid{};
  EdgeInterceptDebugSolverResult fair_fast{};
  EdgeInterceptDebugSolverResult fair_slow{};
};

struct NoopView {
  const NoopCachedPlanet *flat = nullptr;
  int32_t n_frames = 0;
  const NoopSpatialGrid *spatial_grid = nullptr;
};

struct EdgeFrameCollisionMetadata {
  std::array<double, kPlanets> slot_radius{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  std::array<int32_t, kPlanets> static_slots{};
  std::array<int32_t, kPlanets> dynamic_slots{};
  std::array<uint8_t, kPlanets> is_static_slot{};
  std::array<uint8_t, kPlanets> is_dynamic_slot{};
  int32_t static_slots_n = 0;
  int32_t dynamic_slots_n = 0;
};

NoopView make_noop_view(const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
                        const NoopSpatialGrid &spatial_grid);
NoopSpatialGrid build_noop_spatial_grid(
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat);

bool honest_source_planet_can_launch(const Planet &p);
bool edge_intercept_aim_for_ship_count(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count, double ship_speed,
    int32_t noop_base_frame, EdgeInterceptAim *out);
bool edge_intercept_aim_for_ship_count_and_aim_index(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count, double ship_speed,
    int32_t noop_base_frame, int32_t aim_index, EdgeInterceptAim *out);
void edge_intercept_debug_trace_for_ship_count_and_aim_index(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, int32_t noop_base_frame, int32_t aim_index,
    int32_t target_path_steps,
    std::vector<EdgeInterceptDebugTargetPoint> *out_target_path,
    std::vector<EdgeInterceptDebugSegment> *out_segments);
EdgeInterceptDebugSolverResults edge_intercept_debug_solver_results_for_ship_count_and_aim_index(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, int32_t noop_base_frame, int32_t aim_index,
    int32_t target_path_steps);
void fill_honest_aim_nan_outputs(torch::Tensor out_x, torch::Tensor out_y, torch::Tensor out_turns,
                                 torch::Tensor out_intercept_ok,
                                 torch::Tensor out_intercept_fail_reason);
EdgeActionHitWithAim edge_action_hit_for_intercept_aim(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    const NoopCachedPlanet &src, int32_t src_slot, int32_t dst_slot,
    int32_t ship_count, double ship_speed, const EdgeInterceptAim &aim,
    int32_t aim_index, bool target_static,
    bool has_target_hit_for_prior_bucket,
    const std::array<uint8_t, kPlanets> &is_static_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius);
EdgeActionHitWithAim edge_action_hit_for_static_checked_intercept_aim(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    const NoopCachedPlanet &src, int32_t src_slot, int32_t dst_slot,
    int32_t ship_count, double ship_speed, const EdgeInterceptAim &aim,
    int32_t aim_index,
    const std::array<uint8_t, kPlanets> &is_static_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius);
EdgeActionHitWithAim edge_action_hit_for_cached_dynamic_dynamic_intercept(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    int32_t src_slot, double source_radius, int32_t dst_slot,
    int32_t ship_count, double ship_speed, double dir_x, double dir_y,
    double turns_to_target, int32_t aim_index,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius);
EdgeActionHit cached_dynamic_dynamic_comet_overlay_hit(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    int32_t src_slot, double source_radius, int32_t dst_slot,
    int32_t ship_count, double ship_speed, double dir_x, double dir_y,
    double turns_to_target,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<double, kPlanets> &slot_radius);
EdgeActionHitWithAim edge_action_hit_with_intercept_aims(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    const NoopCachedPlanet &src, int32_t src_slot, int32_t dst_slot,
    int32_t ship_count, double ship_speed, const EdgeInterceptAim &aim,
    bool target_static, bool has_target_hit_for_prior_bucket,
    const std::array<uint8_t, kPlanets> &is_static_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius);
std::vector<Fleet> fleet_rows_tensor_to_vector(torch::Tensor fleet_rows);
std::vector<Planet> external_planet_rows_tensor_to_vector(
    torch::Tensor planet_rows, int32_t planet_count, const char *context);
torch::Tensor fleet_arrivals_for_fleets(
    const std::vector<Fleet> &fleets, int32_t horizon, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid);
EdgeFrameCollisionMetadata edge_frame_collision_metadata(
    const NoopView &noop, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids);
int32_t direct_edge_target_hit_steps_with_metadata(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    int32_t src_slot, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, const EdgeFrameCollisionMetadata &metadata);
py::list fleet_hit_traces_for_fleets(
    const std::vector<Fleet> &fleets, int32_t horizon, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid);

}  // namespace orbit_wars_honest
