#include "honest_shared_intercept.h"

#include "library.h"
#include "masks.h"
#include "simulation.h"

#include <algorithm>
#include <cmath>
#include <limits>


namespace orbit_wars_honest {

constexpr double kStaticSegmentCollisionDistanceSqEpsilon = 1e-7;
constexpr int32_t kSunGridSlot = kPlanets;
enum NoopGridSegmentQuery {
  kNoopGridSegmentQueryAll = 0,
  kNoopGridSegmentQueryDynamicAndComet = 1,
  kNoopGridSegmentQueryCometOnly = 2,
};

struct FirstHitResult {
  int32_t hit_slot = kHitNone;
  int32_t hit_planet_id = -1;
  int32_t hit_steps = -1;
  double fleet_x = 0.0;
  double fleet_y = 0.0;
  double object_x = 0.0;
  double object_y = 0.0;
  double object_radius = 0.0;
};

NoopView make_noop_view(const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
                        const NoopSpatialGrid &spatial_grid) {
  TORCH_CHECK_DISABLED(noop_cached_planets_flat.size() % static_cast<uint32_t>(kPlanets) == 0,
              "make_noop_view: corrupt noop flat buffer");
  const int32_t n_frames =
      static_cast<int32_t>(noop_cached_planets_flat.size() / static_cast<uint32_t>(kPlanets));
  TORCH_CHECK_DISABLED(n_frames > 0, "make_noop_view: empty noop cache");
  NoopView view{noop_cached_planets_flat.data(), n_frames, &spatial_grid};
  TORCH_CHECK_DISABLED(view.n_frames == spatial_grid.n_frames,
              "make_noop_view: spatial grid frame mismatch");
  TORCH_CHECK_DISABLED(spatial_grid.cells_per_axis > 0, "make_noop_view: empty spatial grid");
  return view;
}

bool noop_slot_has_expected_planet(const NoopCachedPlanet &p, int32_t expected_planet_id,
                                   int32_t expected_comet_internal_id) {
  return expected_planet_id >= 0 && p.id == expected_planet_id &&
         p.comet_internal_id == expected_comet_internal_id;
}

bool noop_grid_position_is_on_board(double x, double y) {
  return 0.0 <= x && x <= kBoardSize && 0.0 <= y && y <= kBoardSize;
}

int32_t noop_grid_cell_index(const NoopSpatialGrid &grid, double x) {
  int32_t cell = static_cast<int32_t>(std::floor((x - grid.min_coord) / grid.cell_size));
  cell = std::max<int32_t>(0, std::min<int32_t>(grid.cells_per_axis - 1, cell));
  return cell;
}

double point_to_interval_distance_sq(double p, double lo, double hi) {
  if (p < lo) {
    const double d = lo - p;
    return d * d;
  }
  if (p > hi) {
    const double d = p - hi;
    return d * d;
  }
  return 0.0;
}

double point_to_rect_distance_sq(double px, double py, double min_x, double min_y,
                                 double max_x, double max_y) {
  return point_to_interval_distance_sq(px, min_x, max_x) +
         point_to_interval_distance_sq(py, min_y, max_y);
}

bool point_in_rect(double px, double py, double min_x, double min_y, double max_x,
                   double max_y) {
  return min_x <= px && px <= max_x && min_y <= py && py <= max_y;
}

bool segment_intersects_rect(double x0, double y0, double x1, double y1,
                             double min_x, double min_y, double max_x, double max_y) {
  double t0 = 0.0;
  double t1 = 1.0;
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double p[4] = {-dx, dx, -dy, dy};
  const double q[4] = {x0 - min_x, max_x - x0, y0 - min_y, max_y - y0};
  for (int32_t i = 0; i < 4; ++i) {
    if (p[i] == 0.0) {
      if (q[i] < 0.0) {
        return false;
      }
      continue;
    }
    const double t = q[i] / p[i];
    if (p[i] < 0.0) {
      t0 = std::max(t0, t);
    } else {
      t1 = std::min(t1, t);
    }
    if (t0 > t1) {
      return false;
    }
  }
  return true;
}

double segment_to_rect_distance_sq(double x0, double y0, double x1, double y1,
                                   double min_x, double min_y, double max_x,
                                   double max_y) {
  if (point_in_rect(x0, y0, min_x, min_y, max_x, max_y) ||
      point_in_rect(x1, y1, min_x, min_y, max_x, max_y) ||
      segment_intersects_rect(x0, y0, x1, y1, min_x, min_y, max_x, max_y)) {
    return 0.0;
  }
  double best = std::min(point_to_rect_distance_sq(x0, y0, min_x, min_y, max_x, max_y),
                         point_to_rect_distance_sq(x1, y1, min_x, min_y, max_x, max_y));
  best = std::min(best, point_to_segment_distance_sq(min_x, min_y, x0, y0, x1, y1));
  best = std::min(best, point_to_segment_distance_sq(min_x, max_y, x0, y0, x1, y1));
  best = std::min(best, point_to_segment_distance_sq(max_x, min_y, x0, y0, x1, y1));
  best = std::min(best, point_to_segment_distance_sq(max_x, max_y, x0, y0, x1, y1));
  return best;
}

uint32_t noop_grid_cell_flat_index(const NoopSpatialGrid &grid, int32_t frame,
                                 int32_t cell_x, int32_t cell_y) {
  return (static_cast<uint32_t>(frame) * static_cast<uint32_t>(grid.cells_per_axis) +
          static_cast<uint32_t>(cell_y)) *
             static_cast<uint32_t>(grid.cells_per_axis) +
         static_cast<uint32_t>(cell_x);
}

void noop_grid_add_slot_capsule(NoopSpatialGrid *grid, int32_t frame, int32_t slot,
                                double x0, double y0, double x1, double y1,
                                double radius, bool dynamic_slot,
                                bool comet_slot) {
  TORCH_CHECK_DISABLED(grid != nullptr, "noop_grid_add_slot_capsule: grid");
  TORCH_CHECK_DISABLED(0 <= frame && frame < grid->n_frames, "noop_grid_add_slot_capsule: frame");
  TORCH_CHECK_DISABLED(0 <= slot && slot <= kSunGridSlot, "noop_grid_add_slot_capsule: slot");
  TORCH_CHECK_DISABLED(radius > 0.0 && std::isfinite(radius), "noop_grid_add_slot_capsule: radius");
  TORCH_CHECK_DISABLED(!(dynamic_slot && comet_slot),
              "noop_grid_add_slot_capsule: dynamic/comet overlap");
  const int32_t min_cell_x = noop_grid_cell_index(
      *grid, std::min(x0, x1) - radius);
  const int32_t max_cell_x = noop_grid_cell_index(
      *grid, std::max(x0, x1) + radius);
  const int32_t min_cell_y = noop_grid_cell_index(
      *grid, std::min(y0, y1) - radius);
  const int32_t max_cell_y = noop_grid_cell_index(
      *grid, std::max(y0, y1) + radius);
  const uint64_t slot_bit = uint64_t{1} << static_cast<uint32_t>(slot);
  const double radius_sq = radius * radius;
  for (int32_t cy = min_cell_y; cy <= max_cell_y; ++cy) {
    const double cell_min_y = grid->min_coord + static_cast<double>(cy) * grid->cell_size;
    const double cell_max_y = cell_min_y + grid->cell_size;
    for (int32_t cx = min_cell_x; cx <= max_cell_x; ++cx) {
      const double cell_min_x = grid->min_coord + static_cast<double>(cx) * grid->cell_size;
      const double cell_max_x = cell_min_x + grid->cell_size;
      if (segment_to_rect_distance_sq(x0, y0, x1, y1, cell_min_x, cell_min_y,
                                      cell_max_x, cell_max_y) <= radius_sq) {
        const uint32_t cell_idx = noop_grid_cell_flat_index(*grid, frame, cx, cy);
        grid->cell_slot_bits[cell_idx] |= slot_bit;
        if (dynamic_slot) {
          grid->dynamic_cell_slot_bits[cell_idx] |= slot_bit;
        } else if (comet_slot) {
          grid->comet_cell_slot_bits[cell_idx] |= slot_bit;
        } else {
          grid->static_cell_slot_bits[cell_idx] |= slot_bit;
        }
      }
    }
  }
}

NoopSpatialGrid build_noop_spatial_grid(
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat) {
  NoopSpatialGrid grid;
  TORCH_CHECK_DISABLED(noop_cached_planets_flat.size() % static_cast<uint32_t>(kPlanets) == 0,
              "build_noop_spatial_grid: corrupt noop flat buffer");
  grid.n_frames = static_cast<int32_t>(noop_cached_planets_flat.size() /
                                       static_cast<uint32_t>(kPlanets));
  TORCH_CHECK_DISABLED(grid.n_frames > 0, "build_noop_spatial_grid: empty noop cache");
  grid.min_coord = -10.0;
  grid.cell_size = 8.0;
  grid.cells_per_axis = 15;
  grid.cell_slot_bits.assign(static_cast<uint32_t>(grid.n_frames) *
                                 static_cast<uint32_t>(grid.cells_per_axis) *
                                 static_cast<uint32_t>(grid.cells_per_axis),
                            uint64_t{0});
  grid.static_cell_slot_bits.assign(grid.cell_slot_bits.size(), uint64_t{0});
  grid.dynamic_cell_slot_bits.assign(grid.cell_slot_bits.size(), uint64_t{0});
  grid.comet_cell_slot_bits.assign(grid.cell_slot_bits.size(), uint64_t{0});
  std::array<uint8_t, kPlanets> slot_dynamic{};
  for (int32_t frame = 0; frame + 1 < grid.n_frames; ++frame) {
    const NoopCachedPlanet *row =
        noop_cached_planets_flat.data() +
        static_cast<uint32_t>(frame) * static_cast<uint32_t>(kPlanets);
    const NoopCachedPlanet *next_row =
        noop_cached_planets_flat.data() +
        static_cast<uint32_t>(frame + 1) * static_cast<uint32_t>(kPlanets);
    for (int32_t slot = 0; slot < kPlanets; ++slot) {
      const NoopCachedPlanet &p0 = row[static_cast<uint32_t>(slot)];
      if (p0.id < 0) {
        continue;
      }
      if (p0.comet_internal_id >= 0) {
        slot_dynamic[static_cast<uint32_t>(slot)] = 1;
        continue;
      }
      const NoopCachedPlanet &p1 = next_row[static_cast<uint32_t>(slot)];
      if (noop_slot_has_expected_planet(p1, p0.id, p0.comet_internal_id) &&
          (p0.x != p1.x || p0.y != p1.y)) {
        slot_dynamic[static_cast<uint32_t>(slot)] = 1;
      }
    }
  }
  for (int32_t frame = 0; frame < grid.n_frames; ++frame) {
    noop_grid_add_slot_capsule(&grid, frame, kSunGridSlot, kCenter, kCenter,
                               kCenter, kCenter, kSunRadius, false, false);
    const NoopCachedPlanet *row =
        noop_cached_planets_flat.data() +
        static_cast<uint32_t>(frame) * static_cast<uint32_t>(kPlanets);
    for (int32_t slot = 0; slot < kPlanets; ++slot) {
      const NoopCachedPlanet &p0 = row[static_cast<uint32_t>(slot)];
      if (p0.id < 0) {
        continue;
      }
      const NoopCachedPlanet *p1 = &p0;
      if (frame + 1 < grid.n_frames) {
        const NoopCachedPlanet &next_p =
            noop_cached_planets_flat[static_cast<uint32_t>(frame + 1) *
                                         static_cast<uint32_t>(kPlanets) +
                                     static_cast<uint32_t>(slot)];
        if (noop_slot_has_expected_planet(next_p, p0.id, p0.comet_internal_id)) {
          p1 = &next_p;
        }
      }
      const bool comet_slot = p0.comet_internal_id >= 0;
      if (!noop_grid_position_is_on_board(p0.x, p0.y) ||
          !noop_grid_position_is_on_board(p1->x, p1->y)) {
        continue;
      }
      noop_grid_add_slot_capsule(&grid, frame, slot, p0.x, p0.y, p1->x, p1->y,
                                 p0.radius,
                                 slot_dynamic[static_cast<uint32_t>(slot)] != 0 &&
                                     !comet_slot,
                                 comet_slot);
    }
  }
  return grid;
}

bool honest_source_planet_can_launch(const Planet &p) {
  return 0 <= p.owner && p.owner < kPlayerAxisSlots && p.ships > 0.0;
}

bool interp_noop_planet_xy_at_turns(const NoopView &noop, int32_t planet_slot,
                                    int32_t expected_planet_id,
                                    int32_t expected_comet_internal_id,
                                    double absolute_turn,
                                    double *out_x, double *out_y) {
  if (planet_slot < 0 || planet_slot >= kPlanets || absolute_turn < 0.0) {
    return false;
  }
  const int32_t lo = static_cast<int32_t>(std::floor(absolute_turn));
  if (lo < 0 || lo >= noop.n_frames) {
    return false;
  }
  const double frac = absolute_turn - static_cast<double>(lo);
  const uint32_t lo_base =
      static_cast<uint32_t>(lo) * static_cast<uint32_t>(kPlanets) + static_cast<uint32_t>(planet_slot);
  const NoopCachedPlanet &plo = noop.flat[lo_base];
  if (!noop_slot_has_expected_planet(plo, expected_planet_id,
                                     expected_comet_internal_id)) {
    return false;
  }
  if (frac == 0.0) {
    *out_x = plo.x;
    *out_y = plo.y;
    return true;
  }
  const int32_t hi = lo + 1;
  if (hi >= noop.n_frames) {
    return false;
  }
  const uint32_t hi_base =
      static_cast<uint32_t>(hi) * static_cast<uint32_t>(kPlanets) + static_cast<uint32_t>(planet_slot);
  const NoopCachedPlanet &phi = noop.flat[hi_base];
  if (!noop_slot_has_expected_planet(phi, expected_planet_id,
                                     expected_comet_internal_id)) {
    return false;
  }
  *out_x = plo.x + (phi.x - plo.x) * frac;
  *out_y = plo.y + (phi.y - plo.y) * frac;
  return true;
}

bool intercept_candidate_point_for_center(double center_x, double center_y, double source_x,
                                          double source_y, double target_radius,
                                          int32_t aim_index, double *out_aim_x,
                                          double *out_aim_y, double *out_dir_x,
                                          double *out_dir_y) {
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
  const double tangent_dist = target_radius * kEdgeInterceptTangentRadiusFrac;
  if (aim_index == kEdgeInterceptAimCenter) {
    *out_aim_x = center_x - ux * tangent_dist;
    *out_aim_y = center_y - uy * tangent_dist;
    *out_dir_x = ux;
    *out_dir_y = uy;
    return true;
  }
  TORCH_CHECK_DISABLED(aim_index == kEdgeInterceptAimLeft ||
                  aim_index == kEdgeInterceptAimRight,
              "intercept candidate point: bad aim index");

  const double tangent_dist_sq = tangent_dist * tangent_dist;
  if (center_dist_sq <= tangent_dist_sq) {
    return false;
  }
  const double along = std::sqrt(center_dist_sq - tangent_dist_sq);
  const double side = (aim_index == kEdgeInterceptAimLeft) ? 1.0 : -1.0;
  const double dir_x = (along * ux - side * tangent_dist * uy) * inv_norm;
  const double dir_y = (along * uy + side * tangent_dist * ux) * inv_norm;
  *out_aim_x = source_x + dir_x * along;
  *out_aim_y = source_y + dir_y * along;
  *out_dir_x = dir_x;
  *out_dir_y = dir_y;
  return true;
}

bool intercept_turns_for_center_candidate(double center_x, double center_y,
                                          double source_x, double source_y,
                                          double source_radius,
                                          double target_radius, double speed,
                                          int32_t aim_index, double *out_aim_x,
                                          double *out_aim_y, double *out_dir_x,
                                          double *out_dir_y, double *out_turns) {
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  if (!intercept_candidate_point_for_center(center_x, center_y, source_x,
                                           source_y, target_radius, aim_index,
                                           &aim_x, &aim_y, &dir_x, &dir_y)) {
    return false;
  }
  const double source_offset = source_radius + 0.1;
  const double start_x = source_x + dir_x * source_offset;
  const double start_y = source_y + dir_y * source_offset;
  const double center_from_start_x = center_x - start_x;
  const double center_from_start_y = center_y - start_y;
  const double projected = center_from_start_x * dir_x + center_from_start_y * dir_y;
  const double center_from_start_sq =
      center_from_start_x * center_from_start_x +
      center_from_start_y * center_from_start_y;
  const double perpendicular_sq = center_from_start_sq - projected * projected;
  const double radius_sq = target_radius * target_radius;
  if (perpendicular_sq > radius_sq) {
    return false;
  }
  const double hit_distance = projected - std::sqrt(radius_sq - perpendicular_sq);
  const double turns = std::max(0.0, hit_distance / speed);
  if (!std::isfinite(turns)) {
    return false;
  }
  *out_aim_x = aim_x;
  *out_aim_y = aim_y;
  *out_dir_x = dir_x;
  *out_dir_y = dir_y;
  *out_turns = turns;
  return true;
}

bool solve_dynamic_intercept_fixed_point_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    double seed_center_x, double seed_center_y, int32_t aim_index,
    double *out_aim_x, double *out_aim_y, double *out_dir_x,
    double *out_dir_y, double *out_turns) {
  double center_x = seed_center_x;
  double center_y = seed_center_y;
  bool has_prev_turns = false;
  double prev_turns = 0.0;
  for (int32_t iter = 0; iter < kHonestInterceptMaxIters; ++iter) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double turns = 0.0;
    if (!intercept_turns_for_center_candidate(
            center_x, center_y, source_x, source_y, source_radius,
            target_radius, speed, aim_index, &aim_x, &aim_y, &dir_x, &dir_y,
            &turns)) {
      return false;
    }
    const double abs_turn = static_cast<double>(noop_base_frame) + turns;
    double next_center_x = 0.0;
    double next_center_y = 0.0;
    const bool valid = interp_noop_planet_xy_at_turns(
        noop, target_slot, target_planet_id, target_comet_internal_id, abs_turn,
        &next_center_x, &next_center_y);
    if (!valid) {
      return false;
    }
    if (has_prev_turns && std::abs(turns - prev_turns) <= kHonestInterceptTurnsEpsilon) {
      return intercept_turns_for_center_candidate(
          next_center_x, next_center_y, source_x, source_y, source_radius,
          target_radius, speed, aim_index, out_aim_x, out_aim_y, out_dir_x,
          out_dir_y, out_turns);
    }
    prev_turns = turns;
    has_prev_turns = true;
    center_x = next_center_x;
    center_y = next_center_y;
  }
  return false;
}

bool solve_dynamic_intercept_bisect_fixed_point_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    double seed_center_x, double seed_center_y, int32_t aim_index,
    double *out_aim_x, double *out_aim_y, double *out_dir_x,
    double *out_dir_y, double *out_turns) {
  double center_x = seed_center_x;
  double center_y = seed_center_y;
  double turns = 0.0;
  for (int32_t iter = 0; iter < kHonestInterceptMaxIters; ++iter) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double next_turns = 0.0;
    if (!intercept_turns_for_center_candidate(
            center_x, center_y, source_x, source_y, source_radius,
            target_radius, speed, aim_index, &aim_x, &aim_y, &dir_x, &dir_y,
            &next_turns)) {
      return false;
    }
    const double delta_turns = next_turns - turns;
    if (std::abs(delta_turns) <= kHonestInterceptTurnsEpsilon) {
      const double abs_turn = static_cast<double>(noop_base_frame) + next_turns;
      double final_center_x = 0.0;
      double final_center_y = 0.0;
      const bool valid = interp_noop_planet_xy_at_turns(
          noop, target_slot, target_planet_id, target_comet_internal_id, abs_turn,
          &final_center_x, &final_center_y);
      if (!valid) {
        return false;
      }
      return intercept_turns_for_center_candidate(
          final_center_x, final_center_y, source_x, source_y, source_radius,
          target_radius, speed, aim_index, out_aim_x, out_aim_y, out_dir_x,
          out_dir_y, out_turns);
    }
    turns += 0.5 * delta_turns;
    const double abs_turn = static_cast<double>(noop_base_frame) + turns;
    const bool valid = interp_noop_planet_xy_at_turns(
        noop, target_slot, target_planet_id, target_comet_internal_id, abs_turn,
        &center_x, &center_y);
    if (!valid) {
      return false;
    }
  }
  return false;
}

bool finalize_dynamic_intercept_solution_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    double candidate_turns, int32_t aim_index, double *out_aim_x,
    double *out_aim_y, double *out_dir_x, double *out_dir_y, double *out_turns) {
  const double abs_turn = static_cast<double>(noop_base_frame) + candidate_turns;
  double final_center_x = 0.0;
  double final_center_y = 0.0;
  const bool valid = interp_noop_planet_xy_at_turns(
      noop, target_slot, target_planet_id, target_comet_internal_id, abs_turn,
      &final_center_x, &final_center_y);
  if (!valid) {
    return false;
  }
  double final_turns = 0.0;
  if (!intercept_turns_for_center_candidate(
          final_center_x, final_center_y, source_x, source_y, source_radius,
          target_radius, speed, aim_index, out_aim_x, out_aim_y, out_dir_x,
          out_dir_y, &final_turns)) {
    return false;
  }
  if (std::abs(final_turns - candidate_turns) > kHonestInterceptTurnsEpsilon) {
    return false;
  }
  *out_turns = final_turns;
  return true;
}

bool solve_dynamic_intercept_hybrid_fixed_point_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    double seed_center_x, double seed_center_y, int32_t aim_index,
    double *out_aim_x, double *out_aim_y, double *out_dir_x,
    double *out_dir_y, double *out_turns) {
  double turns = 0.0;
  double center_x = seed_center_x;
  double center_y = seed_center_y;
  bool has_prev_error = false;
  double prev_error = 0.0;
  for (int32_t iter = 0; iter < kHonestInterceptMaxIters; ++iter) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double next_turns = 0.0;
    if (!intercept_turns_for_center_candidate(
            center_x, center_y, source_x, source_y, source_radius,
            target_radius, speed, aim_index, &aim_x, &aim_y, &dir_x, &dir_y,
            &next_turns)) {
      return false;
    }

    const double delta_turns = next_turns - turns;
    const double current_error = std::abs(delta_turns);
    if (current_error <= kHonestInterceptTurnsEpsilon) {
      return finalize_dynamic_intercept_solution_noop(
          noop, target_slot, target_planet_id, target_comet_internal_id,
          noop_base_frame, source_x, source_y, source_radius, target_radius,
          speed, next_turns, aim_index, out_aim_x, out_aim_y, out_dir_x,
          out_dir_y, out_turns);
    }
    if (has_prev_error &&
        std::abs(current_error - prev_error) <= kHonestInterceptTurnsEpsilon) {
      return finalize_dynamic_intercept_solution_noop(
          noop, target_slot, target_planet_id, target_comet_internal_id,
          noop_base_frame, source_x, source_y, source_radius, target_radius,
          speed, next_turns, aim_index, out_aim_x, out_aim_y, out_dir_x,
          out_dir_y, out_turns);
    }

    double fixed_point_center_x = 0.0;
    double fixed_point_center_y = 0.0;
    const double fixed_point_abs_turn =
        static_cast<double>(noop_base_frame) + next_turns;
    const bool fixed_point_valid = interp_noop_planet_xy_at_turns(
        noop, target_slot, target_planet_id, target_comet_internal_id,
        fixed_point_abs_turn, &fixed_point_center_x, &fixed_point_center_y);

    double fixed_point_aim_x = 0.0;
    double fixed_point_aim_y = 0.0;
    double fixed_point_dir_x = 0.0;
    double fixed_point_dir_y = 0.0;
    double fixed_point_turns = 0.0;
    const bool fixed_point_candidate_valid =
        fixed_point_valid &&
        intercept_turns_for_center_candidate(
            fixed_point_center_x, fixed_point_center_y, source_x, source_y,
            source_radius, target_radius, speed, aim_index, &fixed_point_aim_x,
            &fixed_point_aim_y, &fixed_point_dir_x, &fixed_point_dir_y,
            &fixed_point_turns);

    if (fixed_point_candidate_valid &&
        std::abs(fixed_point_turns - next_turns) < current_error) {
      prev_error = current_error;
      has_prev_error = true;
      turns = next_turns;
      center_x = fixed_point_center_x;
      center_y = fixed_point_center_y;
      continue;
    }

    prev_error = current_error;
    has_prev_error = true;
    turns += 0.5 * delta_turns;
    const double bisect_abs_turn = static_cast<double>(noop_base_frame) + turns;
    const bool bisect_valid = interp_noop_planet_xy_at_turns(
        noop, target_slot, target_planet_id, target_comet_internal_id,
        bisect_abs_turn, &center_x, &center_y);
    if (!bisect_valid) {
      return false;
    }
  }
  return false;
}

bool solve_dynamic_intercept_fair_fast_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    int32_t aim_index, int32_t target_path_steps, double *out_aim_x,
    double *out_aim_y, double *out_dir_x, double *out_dir_y,
    double *out_turns);

bool solve_dynamic_intercept_noop(const NoopView &noop, int32_t target_slot,
                                  int32_t target_planet_id,
                                  int32_t target_comet_internal_id,
                                  int32_t noop_base_frame, double source_x, double source_y,
                                  double source_radius, double target_radius, double speed,
                                  double seed_center_x, double seed_center_y,
                                  int32_t aim_index, double *out_aim_x,
                                  double *out_aim_y, double *out_dir_x,
                                  double *out_dir_y, double *out_turns) {
  (void)seed_center_x;
  (void)seed_center_y;
  return solve_dynamic_intercept_fair_fast_noop(
      noop, target_slot, target_planet_id, target_comet_internal_id,
      noop_base_frame, source_x, source_y, source_radius, target_radius, speed,
      aim_index, kHonestHitTraceMaxSteps, out_aim_x, out_aim_y, out_dir_x,
      out_dir_y, out_turns);
}

void append_intercept_debug_segment(
    std::vector<EdgeInterceptDebugSegment> &segments, int32_t iter,
    int32_t branch, double source_x, double source_y, double source_radius,
    double speed, double target_center_x, double target_center_y, double aim_x,
    double aim_y, double dir_x, double dir_y, double turns) {
  const double source_offset = source_radius + 0.1;
  const double start_x = source_x + dir_x * source_offset;
  const double start_y = source_y + dir_y * source_offset;
  segments.push_back(EdgeInterceptDebugSegment{
      iter,
      branch,
      target_center_x,
      target_center_y,
      aim_x,
      aim_y,
      start_x,
      start_y,
      start_x + dir_x * speed * turns,
      start_y + dir_y * speed * turns,
      turns});
}

void edge_intercept_debug_trace_for_ship_count_and_aim_index(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, int32_t noop_base_frame, int32_t aim_index,
    int32_t target_path_steps,
    std::vector<EdgeInterceptDebugTargetPoint> *out_target_path,
    std::vector<EdgeInterceptDebugSegment> *out_segments) {
  TORCH_CHECK_DISABLED(out_target_path != nullptr, "debug trace target path");
  TORCH_CHECK_DISABLED(out_segments != nullptr, "debug trace segments");
  TORCH_CHECK_DISABLED(0 <= aim_index && aim_index < kEdgeInterceptAimCount,
              "debug trace aim index");
  out_target_path->clear();
  out_segments->clear();
  const double speed =
      orbit_cpp_fleet_speed(static_cast<double>(ship_count), ship_speed);
  TORCH_CHECK_DISABLED(speed > 0.0 && std::isfinite(speed), "debug trace speed");

  for (int32_t step = 0; step <= target_path_steps; ++step) {
    const double abs_turn = static_cast<double>(noop_base_frame + step);
    double x = 0.0;
    double y = 0.0;
    if (interp_noop_planet_xy_at_turns(
            noop, dst_slot, dst.id, dst.comet_internal_id, abs_turn, &x, &y)) {
      out_target_path->push_back(EdgeInterceptDebugTargetPoint{step, x, y});
    }
  }

  const bool target_is_comet = comet_planet_ids.contains(dst.id);
  const bool target_rotates =
      planet_is_rotating_for_mask(dst.id, dst.x, dst.y, dst.radius, comet_planet_ids);
  if (!target_is_comet && !target_rotates) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double turns = 0.0;
    if (intercept_turns_for_center_candidate(
            dst.x, dst.y, src.x, src.y, src.radius, dst.radius, speed,
            aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns)) {
      append_intercept_debug_segment(
          *out_segments, 0, 0, src.x, src.y, src.radius, speed, dst.x, dst.y,
          aim_x, aim_y, dir_x, dir_y, turns);
    }
    return;
  }

  double seed_center_x = 0.0;
  double seed_center_y = 0.0;
  if (!interp_noop_planet_xy_at_turns(
          noop, dst_slot, dst.id, dst.comet_internal_id,
          static_cast<double>(noop_base_frame), &seed_center_x, &seed_center_y)) {
    return;
  }

  double turns = 0.0;
  double center_x = seed_center_x;
  double center_y = seed_center_y;
  bool has_prev_error = false;
  double prev_error = 0.0;
  for (int32_t iter = 0; iter < kHonestInterceptMaxIters; ++iter) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double next_turns = 0.0;
    if (!intercept_turns_for_center_candidate(
            center_x, center_y, src.x, src.y, src.radius, dst.radius, speed,
            aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &next_turns)) {
      return;
    }
    append_intercept_debug_segment(
        *out_segments, iter, 0, src.x, src.y, src.radius, speed, center_x,
        center_y, aim_x, aim_y, dir_x, dir_y, next_turns);

    const double delta_turns = next_turns - turns;
    const double current_error = std::abs(delta_turns);
    if (current_error <= kHonestInterceptTurnsEpsilon ||
        (has_prev_error &&
         std::abs(current_error - prev_error) <= kHonestInterceptTurnsEpsilon)) {
      const double abs_turn = static_cast<double>(noop_base_frame) + next_turns;
      double final_center_x = 0.0;
      double final_center_y = 0.0;
      if (!interp_noop_planet_xy_at_turns(
              noop, dst_slot, dst.id, dst.comet_internal_id, abs_turn,
              &final_center_x, &final_center_y)) {
        return;
      }
      double final_aim_x = 0.0;
      double final_aim_y = 0.0;
      double final_dir_x = 0.0;
      double final_dir_y = 0.0;
      double final_turns = 0.0;
      if (intercept_turns_for_center_candidate(
              final_center_x, final_center_y, src.x, src.y, src.radius,
              dst.radius, speed, aim_index, &final_aim_x, &final_aim_y,
              &final_dir_x, &final_dir_y, &final_turns)) {
        append_intercept_debug_segment(
            *out_segments, iter, 2, src.x, src.y, src.radius, speed,
            final_center_x, final_center_y, final_aim_x, final_aim_y,
            final_dir_x, final_dir_y, final_turns);
      }
      return;
    }

    double fixed_point_center_x = 0.0;
    double fixed_point_center_y = 0.0;
    const double fixed_point_abs_turn =
        static_cast<double>(noop_base_frame) + next_turns;
    const bool fixed_point_valid = interp_noop_planet_xy_at_turns(
        noop, dst_slot, dst.id, dst.comet_internal_id, fixed_point_abs_turn,
        &fixed_point_center_x, &fixed_point_center_y);

    double fixed_point_aim_x = 0.0;
    double fixed_point_aim_y = 0.0;
    double fixed_point_dir_x = 0.0;
    double fixed_point_dir_y = 0.0;
    double fixed_point_turns = 0.0;
    const bool fixed_point_candidate_valid =
        fixed_point_valid &&
        intercept_turns_for_center_candidate(
            fixed_point_center_x, fixed_point_center_y, src.x, src.y,
            src.radius, dst.radius, speed, aim_index, &fixed_point_aim_x,
            &fixed_point_aim_y, &fixed_point_dir_x, &fixed_point_dir_y,
            &fixed_point_turns);
    if (fixed_point_candidate_valid) {
      append_intercept_debug_segment(
          *out_segments, iter, 1, src.x, src.y, src.radius, speed,
          fixed_point_center_x, fixed_point_center_y, fixed_point_aim_x,
          fixed_point_aim_y, fixed_point_dir_x, fixed_point_dir_y,
          fixed_point_turns);
    }

    if (fixed_point_candidate_valid &&
        std::abs(fixed_point_turns - next_turns) < current_error) {
      prev_error = current_error;
      has_prev_error = true;
      turns = next_turns;
      center_x = fixed_point_center_x;
      center_y = fixed_point_center_y;
      continue;
    }

    prev_error = current_error;
    has_prev_error = true;
    turns += 0.5 * delta_turns;
    const double bisect_abs_turn = static_cast<double>(noop_base_frame) + turns;
    if (!interp_noop_planet_xy_at_turns(
            noop, dst_slot, dst.id, dst.comet_internal_id, bisect_abs_turn,
            &center_x, &center_y)) {
      return;
    }
  }
}

EdgeInterceptDebugSolverResult make_debug_solver_result(
    bool valid, double aim_x, double aim_y, double dir_x, double dir_y,
    double turns) {
  EdgeInterceptDebugSolverResult out;
  out.valid = valid;
  if (valid) {
    out.aim_x = aim_x;
    out.aim_y = aim_y;
    out.dir_x = dir_x;
    out.dir_y = dir_y;
    out.turns = turns;
  }
  return out;
}

bool fair_slow_intercept_error_at_turns(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    int32_t aim_index, double turns, double *out_error, double *out_aim_x,
    double *out_aim_y, double *out_dir_x, double *out_dir_y,
    double *out_travel_turns) {
  double center_x = 0.0;
  double center_y = 0.0;
  const bool center_valid = interp_noop_planet_xy_at_turns(
      noop, target_slot, target_planet_id, target_comet_internal_id,
      static_cast<double>(noop_base_frame) + turns, &center_x, &center_y);
  if (!center_valid) {
    return false;
  }
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double travel_turns = 0.0;
  const bool candidate_valid = intercept_turns_for_center_candidate(
      center_x, center_y, source_x, source_y, source_radius, target_radius,
      speed, aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &travel_turns);
  if (!candidate_valid) {
    return false;
  }
  *out_error = travel_turns - turns;
  *out_aim_x = aim_x;
  *out_aim_y = aim_y;
  *out_dir_x = dir_x;
  *out_dir_y = dir_y;
  *out_travel_turns = travel_turns;
  return true;
}

bool fair_intercept_error_for_center_at_turns(
    double center_x, double center_y, double source_x, double source_y,
    double source_radius, double target_radius, double speed, int32_t aim_index,
    double turns, double *out_error, double *out_aim_x, double *out_aim_y,
    double *out_dir_x, double *out_dir_y, double *out_travel_turns) {
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double travel_turns = 0.0;
  const bool candidate_valid = intercept_turns_for_center_candidate(
      center_x, center_y, source_x, source_y, source_radius, target_radius,
      speed, aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &travel_turns);
  if (!candidate_valid) {
    return false;
  }
  *out_error = travel_turns - turns;
  *out_aim_x = aim_x;
  *out_aim_y = aim_y;
  *out_dir_x = dir_x;
  *out_dir_y = dir_y;
  *out_travel_turns = travel_turns;
  return true;
}

bool solve_dynamic_intercept_fair_slow_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    int32_t aim_index, int32_t target_path_steps, double *out_aim_x,
    double *out_aim_y, double *out_dir_x, double *out_dir_y,
    double *out_turns) {
  constexpr int32_t kFairSlowSamplesPerStep = 16;
  constexpr int32_t kFairSlowBisectIters = 50;
  const int32_t total_samples = target_path_steps * kFairSlowSamplesPerStep;
  bool has_prev = false;
  double prev_t = 0.0;
  double prev_error = 0.0;
  for (int32_t sample = 0; sample <= total_samples; ++sample) {
    const double t = static_cast<double>(sample) /
                     static_cast<double>(kFairSlowSamplesPerStep);
    double error = 0.0;
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double travel_turns = 0.0;
    const bool valid = fair_slow_intercept_error_at_turns(
        noop, target_slot, target_planet_id, target_comet_internal_id,
        noop_base_frame, source_x, source_y, source_radius, target_radius,
        speed, aim_index, t, &error, &aim_x, &aim_y, &dir_x, &dir_y,
        &travel_turns);
    if (!valid) {
      has_prev = false;
      continue;
    }
    if (std::abs(error) <= kHonestInterceptTurnsEpsilon) {
      *out_aim_x = aim_x;
      *out_aim_y = aim_y;
      *out_dir_x = dir_x;
      *out_dir_y = dir_y;
      *out_turns = travel_turns;
      return true;
    }
    if (has_prev &&
        ((prev_error < 0.0 && error > 0.0) ||
         (prev_error > 0.0 && error < 0.0))) {
      assert(t > prev_t);
      double lo = prev_t;
      double hi = t;
      for (int32_t iter = 0; iter < kFairSlowBisectIters; ++iter) {
        const double mid = 0.5 * (lo + hi);
        double mid_error = 0.0;
        double mid_aim_x = 0.0;
        double mid_aim_y = 0.0;
        double mid_dir_x = 0.0;
        double mid_dir_y = 0.0;
        double mid_travel_turns = 0.0;
        const bool mid_valid = fair_slow_intercept_error_at_turns(
            noop, target_slot, target_planet_id, target_comet_internal_id,
            noop_base_frame, source_x, source_y, source_radius, target_radius,
            speed, aim_index, mid, &mid_error, &mid_aim_x, &mid_aim_y,
            &mid_dir_x, &mid_dir_y, &mid_travel_turns);
        if (!mid_valid) {
          return false;
        }
        if (std::abs(mid_error) <= kHonestInterceptTurnsEpsilon) {
          return finalize_dynamic_intercept_solution_noop(
              noop, target_slot, target_planet_id, target_comet_internal_id,
              noop_base_frame, source_x, source_y, source_radius,
              target_radius, speed, mid_travel_turns, aim_index, out_aim_x,
              out_aim_y, out_dir_x, out_dir_y, out_turns);
        }
        if ((prev_error < 0.0 && mid_error > 0.0) ||
            (prev_error > 0.0 && mid_error < 0.0)) {
          hi = mid;
          error = mid_error;
        } else {
          lo = mid;
          prev_error = mid_error;
        }
      }
      const double root_turns = 0.5 * (lo + hi);
      return finalize_dynamic_intercept_solution_noop(
          noop, target_slot, target_planet_id, target_comet_internal_id,
          noop_base_frame, source_x, source_y, source_radius, target_radius,
          speed, root_turns, aim_index, out_aim_x, out_aim_y, out_dir_x,
          out_dir_y, out_turns);
    }
    has_prev = true;
    prev_t = t;
    prev_error = error;
  }
  return false;
}

bool solve_dynamic_intercept_fair_fast_noop(
    const NoopView &noop, int32_t target_slot, int32_t target_planet_id,
    int32_t target_comet_internal_id, int32_t noop_base_frame, double source_x,
    double source_y, double source_radius, double target_radius, double speed,
    int32_t aim_index, int32_t target_path_steps, double *out_aim_x,
    double *out_aim_y, double *out_dir_x, double *out_dir_y,
    double *out_turns) {
  bool has_prev = false;
  double prev_t = 0.0;
  double prev_error = 0.0;
  double prev_center_x = 0.0;
  double prev_center_y = 0.0;
  for (int32_t step = 0; step <= target_path_steps; ++step) {
    const double t = static_cast<double>(step);
    double center_x = 0.0;
    double center_y = 0.0;
    const bool center_valid = interp_noop_planet_xy_at_turns(
        noop, target_slot, target_planet_id, target_comet_internal_id,
        static_cast<double>(noop_base_frame) + t, &center_x, &center_y);
    if (!center_valid) {
      has_prev = false;
      continue;
    }
    double error = 0.0;
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double travel_turns = 0.0;
    const bool valid = fair_intercept_error_for_center_at_turns(
        center_x, center_y, source_x, source_y, source_radius, target_radius,
        speed, aim_index, t, &error, &aim_x, &aim_y, &dir_x, &dir_y,
        &travel_turns);
    if (!valid) {
      has_prev = false;
      continue;
    }
    if (std::abs(error) <= kHonestInterceptTurnsEpsilon) {
      *out_aim_x = aim_x;
      *out_aim_y = aim_y;
      *out_dir_x = dir_x;
      *out_dir_y = dir_y;
      *out_turns = travel_turns;
      return true;
    }
    if (has_prev &&
        ((prev_error < 0.0 && error > 0.0) ||
         (prev_error > 0.0 && error < 0.0))) {
      assert(t > prev_t);
      double lo = prev_t;
      double hi = t;
      double lo_error = prev_error;
      const double lo_center_x0 = prev_center_x;
      const double lo_center_y0 = prev_center_y;
      const double hi_center_x0 = center_x;
      const double hi_center_y0 = center_y;
      for (int32_t iter = 0; iter < kHonestInterceptMaxIters; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double mid_alpha = (mid - prev_t) / (t - prev_t);
        const double mid_center_x =
            lo_center_x0 + (hi_center_x0 - lo_center_x0) * mid_alpha;
        const double mid_center_y =
            lo_center_y0 + (hi_center_y0 - lo_center_y0) * mid_alpha;
        double mid_error = 0.0;
        double mid_aim_x = 0.0;
        double mid_aim_y = 0.0;
        double mid_dir_x = 0.0;
        double mid_dir_y = 0.0;
        double mid_travel_turns = 0.0;
        const bool mid_valid = fair_intercept_error_for_center_at_turns(
            mid_center_x, mid_center_y, source_x, source_y, source_radius,
            target_radius, speed, aim_index, mid, &mid_error, &mid_aim_x,
            &mid_aim_y, &mid_dir_x, &mid_dir_y, &mid_travel_turns);
        if (!mid_valid) {
          return false;
        }
        if (std::abs(mid_error) <= kHonestInterceptTurnsEpsilon) {
          *out_aim_x = mid_aim_x;
          *out_aim_y = mid_aim_y;
          *out_dir_x = mid_dir_x;
          *out_dir_y = mid_dir_y;
          *out_turns = mid_travel_turns;
          return true;
        }
        if ((lo_error < 0.0 && mid_error > 0.0) ||
            (lo_error > 0.0 && mid_error < 0.0)) {
          hi = mid;
        } else {
          lo = mid;
          lo_error = mid_error;
        }
      }
      const double root_turns = 0.5 * (lo + hi);
      const double root_alpha = (root_turns - prev_t) / (t - prev_t);
      const double root_center_x =
          lo_center_x0 + (hi_center_x0 - lo_center_x0) * root_alpha;
      const double root_center_y =
          lo_center_y0 + (hi_center_y0 - lo_center_y0) * root_alpha;
      double root_error = 0.0;
      double root_travel_turns = 0.0;
      if (!fair_intercept_error_for_center_at_turns(
              root_center_x, root_center_y, source_x, source_y, source_radius,
              target_radius, speed, aim_index, root_turns, &root_error,
              out_aim_x, out_aim_y, out_dir_x, out_dir_y,
              &root_travel_turns)) {
        return false;
      }
      if (std::abs(root_error) > kHonestInterceptTurnsEpsilon) {
        return false;
      }
      *out_turns = root_travel_turns;
      return true;
    }
    has_prev = true;
    prev_t = t;
    prev_error = error;
    prev_center_x = center_x;
    prev_center_y = center_y;
  }
  return false;
}

EdgeInterceptDebugSolverResults edge_intercept_debug_solver_results_for_ship_count_and_aim_index(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, int32_t noop_base_frame, int32_t aim_index,
    int32_t target_path_steps) {
  TORCH_CHECK_DISABLED(0 <= aim_index && aim_index < kEdgeInterceptAimCount,
              "debug solver results aim index");
  const double speed =
      orbit_cpp_fleet_speed(static_cast<double>(ship_count), ship_speed);
  TORCH_CHECK_DISABLED(speed > 0.0 && std::isfinite(speed), "debug solver results speed");
  EdgeInterceptDebugSolverResults out;
  const bool target_is_comet = comet_planet_ids.contains(dst.id);
  const bool target_rotates =
      planet_is_rotating_for_mask(dst.id, dst.x, dst.y, dst.radius, comet_planet_ids);
  if (!target_is_comet && !target_rotates) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double turns = 0.0;
    const bool valid = intercept_turns_for_center_candidate(
        dst.x, dst.y, src.x, src.y, src.radius, dst.radius, speed, aim_index,
        &aim_x, &aim_y, &dir_x, &dir_y, &turns);
    out.point = make_debug_solver_result(valid, aim_x, aim_y, dir_x, dir_y, turns);
    out.bisect = out.point;
    out.hybrid = out.point;
    out.fair_fast = out.point;
    out.fair_slow = out.point;
    return out;
  }

  double seed_center_x = 0.0;
  double seed_center_y = 0.0;
  const bool seed_valid = interp_noop_planet_xy_at_turns(
      noop, dst_slot, dst.id, dst.comet_internal_id,
      static_cast<double>(noop_base_frame), &seed_center_x, &seed_center_y);
  if (!seed_valid) {
    return out;
  }

  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double turns = 0.0;
  bool valid = solve_dynamic_intercept_fixed_point_noop(
      noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
      src.y, src.radius, dst.radius, speed, seed_center_x, seed_center_y,
      aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns);
  out.point = make_debug_solver_result(valid, aim_x, aim_y, dir_x, dir_y, turns);

  valid = solve_dynamic_intercept_bisect_fixed_point_noop(
      noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
      src.y, src.radius, dst.radius, speed, seed_center_x, seed_center_y,
      aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns);
  out.bisect = make_debug_solver_result(valid, aim_x, aim_y, dir_x, dir_y, turns);

  valid = solve_dynamic_intercept_hybrid_fixed_point_noop(
      noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
      src.y, src.radius, dst.radius, speed, seed_center_x, seed_center_y,
      aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns);
  out.hybrid = make_debug_solver_result(valid, aim_x, aim_y, dir_x, dir_y, turns);

  valid = solve_dynamic_intercept_fair_fast_noop(
      noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
      src.y, src.radius, dst.radius, speed, aim_index, target_path_steps,
      &aim_x, &aim_y, &dir_x, &dir_y, &turns);
  out.fair_fast = make_debug_solver_result(valid, aim_x, aim_y, dir_x, dir_y, turns);

  valid = solve_dynamic_intercept_fair_slow_noop(
      noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
      src.y, src.radius, dst.radius, speed, aim_index, target_path_steps,
      &aim_x, &aim_y, &dir_x, &dir_y, &turns);
  out.fair_slow = make_debug_solver_result(valid, aim_x, aim_y, dir_x, dir_y, turns);
  return out;
}

bool edge_intercept_candidate_is_valid(const EdgeInterceptAimCandidate &candidate) {
  return candidate.valid && std::isfinite(candidate.aim_x) &&
         std::isfinite(candidate.aim_y) &&
         std::isfinite(candidate.dir_x) &&
         std::isfinite(candidate.dir_y) &&
         std::isfinite(candidate.turns_to_target) &&
         candidate.turns_to_target >= 0.0;
}

bool edge_intercept_aim_has_valid_candidate(const EdgeInterceptAim &aim) {
  for (const EdgeInterceptAimCandidate &candidate : aim.candidates) {
    if (edge_intercept_candidate_is_valid(candidate)) {
      return true;
    }
  }
  return false;
}

void set_edge_intercept_candidate_failure(EdgeInterceptAimCandidate &candidate,
                                          int32_t fail_reason) {
  candidate.valid = false;
  candidate.fail_reason = fail_reason;
}

void set_edge_intercept_candidate_success(EdgeInterceptAimCandidate &candidate,
                                          double aim_x, double aim_y,
                                          double dir_x, double dir_y,
                                          double turns_to_target) {
  candidate.valid = true;
  candidate.aim_x = aim_x;
  candidate.aim_y = aim_y;
  candidate.dir_x = dir_x;
  candidate.dir_y = dir_y;
  candidate.turns_to_target = turns_to_target;
  candidate.fail_reason = kInterceptFailReasonNone;
}

bool edge_intercept_aim_for_ship_count(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, int32_t noop_base_frame, EdgeInterceptAim *out) {
  TORCH_CHECK_DISABLED(out != nullptr, "edge intercept aim: output");
  *out = EdgeInterceptAim{};
  out->fail_reason = kInterceptFailReasonNone;
  TORCH_CHECK_DISABLED(ship_count > 0, "edge intercept aim: ship_count");
  const double speed =
      orbit_cpp_fleet_speed(static_cast<double>(ship_count), ship_speed);
  TORCH_CHECK_DISABLED(speed > 0.0 && std::isfinite(speed), "edge intercept aim: speed");
  const bool target_is_comet = comet_planet_ids.contains(dst.id);
  const bool target_rotates =
      planet_is_rotating_for_mask(dst.id, dst.x, dst.y, dst.radius, comet_planet_ids);
  if (!target_is_comet && !target_rotates) {
    for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
      EdgeInterceptAimCandidate &candidate =
          out->candidates[static_cast<uint32_t>(aim_index)];
      double aim_x = 0.0;
      double aim_y = 0.0;
      double dir_x = 0.0;
      double dir_y = 0.0;
      double turns_to_target = 0.0;
      if (!intercept_turns_for_center_candidate(
              dst.x, dst.y, src.x, src.y, src.radius, dst.radius, speed,
              aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns_to_target)) {
        set_edge_intercept_candidate_failure(candidate,
                                             kInterceptFailReasonStaticZeroNorm);
        if (aim_index == kEdgeInterceptAimCenter) {
          out->fail_reason = kInterceptFailReasonStaticZeroNorm;
          return false;
        }
        continue;
      }
      if (!std::isfinite(turns_to_target)) {
        set_edge_intercept_candidate_failure(candidate,
                                             kInterceptFailReasonStaticBadTurns);
        if (aim_index == kEdgeInterceptAimCenter) {
          out->fail_reason = kInterceptFailReasonStaticBadTurns;
          return false;
        }
        continue;
      }
      set_edge_intercept_candidate_success(candidate, aim_x, aim_y,
                                           dir_x, dir_y, turns_to_target);
    }
    if (!edge_intercept_aim_has_valid_candidate(*out)) {
      out->fail_reason = kInterceptFailReasonStaticZeroNorm;
      return false;
    }
    return true;
  }

  double seed_center_x = 0.0;
  double seed_center_y = 0.0;
  bool valid = interp_noop_planet_xy_at_turns(
      noop, dst_slot, dst.id, dst.comet_internal_id,
      static_cast<double>(noop_base_frame), &seed_center_x, &seed_center_y);
  if (!valid) {
    out->fail_reason = kInterceptFailReasonDynamicSeedInvalid;
    return false;
  }
  for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
    EdgeInterceptAimCandidate &candidate =
        out->candidates[static_cast<uint32_t>(aim_index)];
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double turns_to_target = 0.0;
    valid = solve_dynamic_intercept_noop(
        noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
        src.y, src.radius, dst.radius, speed, seed_center_x, seed_center_y,
        aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns_to_target);
    if (!valid) {
      set_edge_intercept_candidate_failure(
          candidate, kInterceptFailReasonDynamicSolverNoConverge);
      if (aim_index == kEdgeInterceptAimCenter) {
        out->fail_reason = kInterceptFailReasonDynamicSolverNoConverge;
        return false;
      }
      continue;
    }
    if (!std::isfinite(aim_x) || !std::isfinite(aim_y)) {
      set_edge_intercept_candidate_failure(
          candidate, kInterceptFailReasonDynamicNonFiniteAim);
      if (aim_index == kEdgeInterceptAimCenter) {
        out->fail_reason = kInterceptFailReasonDynamicNonFiniteAim;
        return false;
      }
      continue;
    }
    if (!std::isfinite(turns_to_target) || turns_to_target < 0.0) {
      set_edge_intercept_candidate_failure(candidate,
                                           kInterceptFailReasonDynamicBadTurns);
      if (aim_index == kEdgeInterceptAimCenter) {
        out->fail_reason = kInterceptFailReasonDynamicBadTurns;
        return false;
      }
      continue;
    }
    set_edge_intercept_candidate_success(candidate, aim_x, aim_y,
                                         dir_x, dir_y, turns_to_target);
  }
  if (!edge_intercept_aim_has_valid_candidate(*out)) {
    out->fail_reason = kInterceptFailReasonDynamicSolverNoConverge;
    return false;
  }
  return true;
}

bool edge_intercept_aim_for_ship_count_and_aim_index(
    const NoopView &noop, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, int32_t noop_base_frame, int32_t aim_index,
    EdgeInterceptAim *out) {
  TORCH_CHECK_DISABLED(out != nullptr, "edge intercept aim single: output");
  TORCH_CHECK_DISABLED(0 <= aim_index && aim_index < kEdgeInterceptAimCount,
              "edge intercept aim single: aim index");
  *out = EdgeInterceptAim{};
  out->fail_reason = kInterceptFailReasonNone;
  TORCH_CHECK_DISABLED(ship_count > 0, "edge intercept aim single: ship_count");
  const double speed =
      orbit_cpp_fleet_speed(static_cast<double>(ship_count), ship_speed);
  TORCH_CHECK_DISABLED(speed > 0.0 && std::isfinite(speed), "edge intercept aim single: speed");
  EdgeInterceptAimCandidate &candidate =
      out->candidates[static_cast<uint32_t>(aim_index)];
  const bool target_is_comet = comet_planet_ids.contains(dst.id);
  const bool target_rotates =
      planet_is_rotating_for_mask(dst.id, dst.x, dst.y, dst.radius, comet_planet_ids);
  if (!target_is_comet && !target_rotates) {
    double aim_x = 0.0;
    double aim_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double turns_to_target = 0.0;
    if (!intercept_turns_for_center_candidate(
            dst.x, dst.y, src.x, src.y, src.radius, dst.radius, speed,
            aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns_to_target)) {
      set_edge_intercept_candidate_failure(candidate,
                                           kInterceptFailReasonStaticZeroNorm);
      out->fail_reason = kInterceptFailReasonStaticZeroNorm;
      return false;
    }
    if (!std::isfinite(turns_to_target)) {
      set_edge_intercept_candidate_failure(candidate,
                                           kInterceptFailReasonStaticBadTurns);
      out->fail_reason = kInterceptFailReasonStaticBadTurns;
      return false;
    }
    set_edge_intercept_candidate_success(candidate, aim_x, aim_y,
                                         dir_x, dir_y, turns_to_target);
    return true;
  }

  double seed_center_x = 0.0;
  double seed_center_y = 0.0;
  bool valid = interp_noop_planet_xy_at_turns(
      noop, dst_slot, dst.id, dst.comet_internal_id,
      static_cast<double>(noop_base_frame), &seed_center_x, &seed_center_y);
  if (!valid) {
    out->fail_reason = kInterceptFailReasonDynamicSeedInvalid;
    return false;
  }
  double aim_x = 0.0;
  double aim_y = 0.0;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double turns_to_target = 0.0;
  valid = solve_dynamic_intercept_noop(
      noop, dst_slot, dst.id, dst.comet_internal_id, noop_base_frame, src.x,
      src.y, src.radius, dst.radius, speed, seed_center_x, seed_center_y,
      aim_index, &aim_x, &aim_y, &dir_x, &dir_y, &turns_to_target);
  if (!valid) {
    set_edge_intercept_candidate_failure(
        candidate, kInterceptFailReasonDynamicSolverNoConverge);
    out->fail_reason = kInterceptFailReasonDynamicSolverNoConverge;
    return false;
  }
  if (!std::isfinite(aim_x) || !std::isfinite(aim_y)) {
    set_edge_intercept_candidate_failure(
        candidate, kInterceptFailReasonDynamicNonFiniteAim);
    out->fail_reason = kInterceptFailReasonDynamicNonFiniteAim;
    return false;
  }
  if (!std::isfinite(turns_to_target) || turns_to_target < 0.0) {
    set_edge_intercept_candidate_failure(candidate,
                                         kInterceptFailReasonDynamicBadTurns);
    out->fail_reason = kInterceptFailReasonDynamicBadTurns;
    return false;
  }
  set_edge_intercept_candidate_success(candidate, aim_x, aim_y,
                                       dir_x, dir_y, turns_to_target);
  return true;
}

// int32_t simulate_first_hit_slot_from_position_linear(
//     const NoopView &noop, int32_t base_frame, double start_x, double start_y,
//     double fleet_speed, double dir_x, double dir_y, int32_t max_steps,
//     bool skip_static_and_sun_checks, int32_t target_slot,
//     const std::array<int32_t, kPlanets> &slot_planet_id,
//     const std::array<int32_t, kPlanets> &slot_comet_internal_id,
//     const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
//     const std::array<int32_t, kPlanets> &dynamic_slots, int32_t dynamic_slots_n,
//     const std::array<double, kPlanets> &slot_radius, int32_t &hit_steps,
//     FirstHitResult *hit_result) {
//   hit_steps = -1;
//   if (base_frame < 0 || base_frame >= noop.n_frames || max_steps <= 0) {
//     return kHitNone;
//   }
//   double sx = start_x;
//   double sy = start_y;
//   const double c = dir_x;
//   const double s = dir_y;
//   std::array<uint8_t, kPlanets> is_dynamic_slot{};
//   for (int32_t kk = 0; kk < dynamic_slots_n; ++kk) {
//     const int32_t slot = dynamic_slots[static_cast<uint32_t>(kk)];
//     is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
//   }
//   int32_t simulated_steps = 0;
//   for (int32_t frame = base_frame; frame < noop.n_frames && simulated_steps < max_steps;
//        ++frame, ++simulated_steps) {
//     const double old_x = sx;
//     const double old_y = sy;
//     sx += c * fleet_speed;
//     sy += s * fleet_speed;

//     for (int32_t slot = 0; slot < kPlanets; ++slot) {
//       if (skip_static_and_sun_checks && slot != target_slot &&
//           is_dynamic_slot[static_cast<uint32_t>(slot)] == 0) {
//         continue;
//       }
//       const uint32_t idx = static_cast<uint32_t>(frame) * static_cast<uint32_t>(kPlanets) +
//                          static_cast<uint32_t>(slot);
//       const NoopCachedPlanet &p = noop.flat[idx];
//       if (!noop_slot_has_expected_planet(
//               p, slot_planet_id[static_cast<uint32_t>(slot)],
//               slot_comet_internal_id[static_cast<uint32_t>(slot)])) {
//         continue;
//       }
//       const double r = slot_radius[static_cast<uint32_t>(slot)];
//       if (r <= 0.0) {
//         continue;
//       }
//       const int32_t candidate_hit_steps = simulated_steps + 1;
//       if (is_dynamic_slot[static_cast<uint32_t>(slot)] != 0) {
//         const NoopCachedPlanet *p1 = &p;
//         if (frame + 1 < noop.n_frames) {
//           const uint32_t idx_next = static_cast<uint32_t>(frame + 1) * static_cast<uint32_t>(kPlanets) +
//                                   static_cast<uint32_t>(slot);
//           const NoopCachedPlanet &next_p = noop.flat[idx_next];
//           if (noop_slot_has_expected_planet(
//                   next_p, slot_planet_id[static_cast<uint32_t>(slot)],
//                   slot_comet_internal_id[static_cast<uint32_t>(slot)])) {
//             p1 = &next_p;
//           } else if (slot_comet_internal_id[static_cast<uint32_t>(slot)] < 0) {
//             continue;
//           }
//         } else if (slot_comet_internal_id[static_cast<uint32_t>(slot)] < 0) {
//           continue;
//         }
//         if (orbit_wars_swept_pair_hit(old_x, old_y, sx, sy, p.x, p.y, p1->x, p1->y, r)) {
//           hit_steps = candidate_hit_steps;
//           if (hit_result != nullptr) {
//             hit_result->hit_slot = slot;
//             hit_result->hit_planet_id = p1->id;
//             hit_result->hit_steps = candidate_hit_steps;
//             hit_result->fleet_x = sx;
//             hit_result->fleet_y = sy;
//             hit_result->object_x = p1->x;
//             hit_result->object_y = p1->y;
//             hit_result->object_radius = r;
//           }
//           return slot;
//         }
//         continue;
//       }
//       if (point_to_segment_distance_sq(p.x, p.y, old_x, old_y, sx, sy) < (r * r)) {
//         hit_steps = candidate_hit_steps;
//         if (hit_result != nullptr) {
//           hit_result->hit_slot = slot;
//           hit_result->hit_planet_id = p.id;
//           hit_result->hit_steps = candidate_hit_steps;
//           hit_result->fleet_x = sx;
//           hit_result->fleet_y = sy;
//           hit_result->object_x = p.x;
//           hit_result->object_y = p.y;
//           hit_result->object_radius = r;
//         }
//         return slot;
//       }
//     }

//     if (!(0.0 <= sx && sx <= kBoardSize && 0.0 <= sy && sy <= kBoardSize)) {
//       hit_steps = simulated_steps + 1;
//       return kHitOutOfBoard;
//     }
//   }
//   return kHitNone;
// }

uint64_t noop_grid_query_segment_slot_bits(const NoopSpatialGrid &grid, int32_t frame,
                                           double x0, double y0, double x1,
                                           double y1,
                                           NoopGridSegmentQuery query) {
  TORCH_CHECK_DISABLED(0 <= frame && frame < grid.n_frames, "noop_grid_query_segment_slots: frame");
  const int32_t min_cell_x = noop_grid_cell_index(grid, std::min(x0, x1));
  const int32_t max_cell_x = noop_grid_cell_index(grid, std::max(x0, x1));
  const int32_t min_cell_y = noop_grid_cell_index(grid, std::min(y0, y1));
  const int32_t max_cell_y = noop_grid_cell_index(grid, std::max(y0, y1));
  uint64_t slot_bits = 0;
  for (int32_t cy = min_cell_y; cy <= max_cell_y; ++cy) {
    for (int32_t cx = min_cell_x; cx <= max_cell_x; ++cx) {
      const uint32_t cell_idx = noop_grid_cell_flat_index(grid, frame, cx, cy);
      switch (query) {
        case kNoopGridSegmentQueryAll:
          slot_bits |= grid.cell_slot_bits[cell_idx];
          break;
        case kNoopGridSegmentQueryDynamicAndComet:
          slot_bits |= grid.dynamic_cell_slot_bits[cell_idx];
          slot_bits |= grid.comet_cell_slot_bits[cell_idx];
          break;
        case kNoopGridSegmentQueryCometOnly:
          slot_bits |= grid.comet_cell_slot_bits[cell_idx];
          break;
      }
    }
  }
  return slot_bits;
}

uint64_t noop_grid_query_segment_static_slot_bits(const NoopSpatialGrid &grid, int32_t frame,
                                                  double x0, double y0, double x1,
                                                  double y1) {
  TORCH_CHECK_DISABLED(0 <= frame && frame < grid.n_frames,
              "noop_grid_query_segment_static_slots: frame");
  const int32_t min_cell_x = noop_grid_cell_index(grid, std::min(x0, x1));
  const int32_t max_cell_x = noop_grid_cell_index(grid, std::max(x0, x1));
  const int32_t min_cell_y = noop_grid_cell_index(grid, std::min(y0, y1));
  const int32_t max_cell_y = noop_grid_cell_index(grid, std::max(y0, y1));
  uint64_t slot_bits = 0;
  for (int32_t cy = min_cell_y; cy <= max_cell_y; ++cy) {
    for (int32_t cx = min_cell_x; cx <= max_cell_x; ++cx) {
      slot_bits |= grid.static_cell_slot_bits[noop_grid_cell_flat_index(grid, frame, cx, cy)];
    }
  }
  return slot_bits;
}

int32_t first_static_blocker_on_segment(
    const NoopView &noop, int32_t frame, double x0, double y0, double x1,
    double y1, int32_t target_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<double, kPlanets> &slot_radius, int32_t *out_hit_steps) {
  TORCH_CHECK_DISABLED(noop.spatial_grid != nullptr, "first_static_blocker_on_segment: grid");
  TORCH_CHECK_DISABLED(out_hit_steps != nullptr, "first_static_blocker_on_segment: output");
  *out_hit_steps = -1;
  uint64_t slot_bits = noop_grid_query_segment_static_slot_bits(
      *noop.spatial_grid, frame, x0, y0, x1, y1);
  while (slot_bits != 0) {
    const int32_t slot = static_cast<int32_t>(__builtin_ctzll(slot_bits));
    slot_bits &= slot_bits - 1;
    if (slot == target_slot) {
      continue;
    }
    if (slot == kSunGridSlot) {
      if (point_to_segment_distance_sq(kCenter, kCenter, x0, y0, x1, y1) <
          kSunRadius * kSunRadius + kStaticSegmentCollisionDistanceSqEpsilon) {
        *out_hit_steps = 1;
        return kHitSun;
      }
      continue;
    }
    const uint32_t idx = static_cast<uint32_t>(frame) * static_cast<uint32_t>(kPlanets) +
                       static_cast<uint32_t>(slot);
    const NoopCachedPlanet &p = noop.flat[idx];
    if (!noop_slot_has_expected_planet(
            p, slot_planet_id[static_cast<uint32_t>(slot)],
            slot_comet_internal_id[static_cast<uint32_t>(slot)])) {
      continue;
    }
    const double r = slot_radius[static_cast<uint32_t>(slot)];
    if (r <= 0.0) {
      continue;
    }
    if (point_to_segment_distance_sq(p.x, p.y, x0, y0, x1, y1) <
        r * r + kStaticSegmentCollisionDistanceSqEpsilon) {
      *out_hit_steps = 1;
      return slot;
    }
  }
  return kHitNone;
}

int32_t simulate_first_hit_slot_from_position_grid(
    const NoopView &noop, int32_t base_frame, double start_x, double start_y,
    double fleet_speed, double dir_x, double dir_y, int32_t max_steps,
    bool skip_static_and_sun_checks, int32_t target_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots, int32_t dynamic_slots_n,
    const std::array<double, kPlanets> &slot_radius, int32_t &hit_steps,
    FirstHitResult *hit_result, bool comet_only = false) {
  TORCH_CHECK_DISABLED(noop.spatial_grid != nullptr, "simulate_first_hit_slot_from_position_grid: grid");
  hit_steps = -1;
  if (base_frame < 0 || base_frame >= noop.n_frames || max_steps <= 0) {
    return kHitNone;
  }
  double sx = start_x;
  double sy = start_y;
  const double c = dir_x;
  const double s = dir_y;
  int32_t simulated_steps = 0;
  for (int32_t frame = base_frame; frame < noop.n_frames && simulated_steps < max_steps;
       ++frame, ++simulated_steps) {
    const double old_x = sx;
    const double old_y = sy;
    sx += c * fleet_speed;
    sy += s * fleet_speed;

    const NoopGridSegmentQuery query =
        comet_only ? kNoopGridSegmentQueryCometOnly
                   : (skip_static_and_sun_checks
                          ? kNoopGridSegmentQueryDynamicAndComet
                          : kNoopGridSegmentQueryAll);
    uint64_t slot_bits = noop_grid_query_segment_slot_bits(
        *noop.spatial_grid, frame, old_x, old_y, sx, sy, query);
    if (skip_static_and_sun_checks && !comet_only) {
      slot_bits |= uint64_t{1} << static_cast<uint32_t>(target_slot);
    }
    while (slot_bits != 0) {
      const int32_t slot =
          static_cast<int32_t>(__builtin_ctzll(slot_bits));
      slot_bits &= slot_bits - 1;
      if (slot == kSunGridSlot) {
        const int32_t candidate_hit_steps = simulated_steps + 1;
        if (point_to_segment_distance_sq(kCenter, kCenter, old_x, old_y, sx, sy) <
            (kSunRadius * kSunRadius)) {
          hit_steps = candidate_hit_steps;
          return kHitSun;
        }
        continue;
      }
      if (skip_static_and_sun_checks && slot != target_slot &&
          is_dynamic_slot[static_cast<uint32_t>(slot)] == 0) {
        continue;
      }
      const uint32_t idx = static_cast<uint32_t>(frame) * static_cast<uint32_t>(kPlanets) +
                         static_cast<uint32_t>(slot);
      const NoopCachedPlanet &p = noop.flat[idx];
      if (!noop_slot_has_expected_planet(
              p, slot_planet_id[static_cast<uint32_t>(slot)],
              slot_comet_internal_id[static_cast<uint32_t>(slot)])) {
        continue;
      }
      const double r = slot_radius[static_cast<uint32_t>(slot)];
      if (r <= 0.0) {
        continue;
      }
      const int32_t candidate_hit_steps = simulated_steps + 1;
      if (is_dynamic_slot[static_cast<uint32_t>(slot)] != 0) {
        const NoopCachedPlanet *p1 = &p;
        if (frame + 1 < noop.n_frames) {
          const uint32_t idx_next = static_cast<uint32_t>(frame + 1) * static_cast<uint32_t>(kPlanets) +
                                  static_cast<uint32_t>(slot);
          const NoopCachedPlanet &next_p = noop.flat[idx_next];
          if (noop_slot_has_expected_planet(
                  next_p, slot_planet_id[static_cast<uint32_t>(slot)],
                  slot_comet_internal_id[static_cast<uint32_t>(slot)])) {
            p1 = &next_p;
          } else if (slot_comet_internal_id[static_cast<uint32_t>(slot)] < 0) {
            continue;
          }
        } else if (slot_comet_internal_id[static_cast<uint32_t>(slot)] < 0) {
          continue;
        }
        if (orbit_wars_swept_pair_hit(old_x, old_y, sx, sy, p.x, p.y, p1->x, p1->y, r)) {
          hit_steps = candidate_hit_steps;
          if (hit_result != nullptr) {
            hit_result->hit_slot = slot;
            hit_result->hit_planet_id = p1->id;
            hit_result->hit_steps = candidate_hit_steps;
            hit_result->fleet_x = sx;
            hit_result->fleet_y = sy;
            hit_result->object_x = p1->x;
            hit_result->object_y = p1->y;
            hit_result->object_radius = r;
          }
          return slot;
        }
        continue;
      }
      if (point_to_segment_distance_sq(p.x, p.y, old_x, old_y, sx, sy) < (r * r)) {
        hit_steps = candidate_hit_steps;
        if (hit_result != nullptr) {
          hit_result->hit_slot = slot;
          hit_result->hit_planet_id = p.id;
          hit_result->hit_steps = candidate_hit_steps;
          hit_result->fleet_x = sx;
          hit_result->fleet_y = sy;
          hit_result->object_x = p.x;
          hit_result->object_y = p.y;
          hit_result->object_radius = r;
        }
        return slot;
      }
    }

    if (!(0.0 <= sx && sx <= kBoardSize && 0.0 <= sy && sy <= kBoardSize)) {
      hit_steps = simulated_steps + 1;
      return kHitOutOfBoard;
    }
  }
  return kHitNone;
}

int32_t simulate_first_hit_slot_from_position(
    const NoopView &noop, int32_t base_frame, double start_x, double start_y,
    double fleet_speed, double dir_x, double dir_y, int32_t max_steps,
    bool skip_static_and_sun_checks, int32_t target_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots, int32_t dynamic_slots_n,
    const std::array<double, kPlanets> &slot_radius, int32_t &hit_steps,
    FirstHitResult *hit_result, bool comet_only = false) {
  TORCH_CHECK_DISABLED(noop.spatial_grid != nullptr,
              "simulate_first_hit_slot_from_position: spatial grid required");
  int32_t grid_hit_steps = -1;
  const int32_t grid_hit_slot = simulate_first_hit_slot_from_position_grid(
      noop, base_frame, start_x, start_y, fleet_speed, dir_x, dir_y, max_steps,
      skip_static_and_sun_checks, target_slot, is_dynamic_slot, slot_planet_id,
      slot_comet_internal_id, static_slots, static_slots_n, dynamic_slots,
      dynamic_slots_n, slot_radius, grid_hit_steps, hit_result, comet_only);
  hit_steps = grid_hit_steps;
  return grid_hit_slot;

  // int32_t linear_hit_steps = -1;
  // const int32_t linear_hit_slot = simulate_first_hit_slot_from_position_linear(
  //     noop, base_frame, start_x, start_y, fleet_speed, dir_x, dir_y, max_steps,
  //     skip_static_and_sun_checks, target_slot, slot_planet_id,
  //     slot_comet_internal_id, static_slots, static_slots_n, dynamic_slots,
  //     dynamic_slots_n, slot_radius, linear_hit_steps, hit_result);
  // TORCH_CHECK_DISABLED(grid_hit_slot == linear_hit_slot && grid_hit_steps == linear_hit_steps,
  //             "intercept spatial grid mismatch: grid slot=", grid_hit_slot,
  //             " grid steps=", grid_hit_steps, " linear slot=", linear_hit_slot,
  //             " linear steps=", linear_hit_steps);
  // hit_steps = linear_hit_steps;
  // return linear_hit_slot;
}

int32_t simulate_first_hit_slot_from_dir(const NoopView &noop, int32_t base_frame,
                                         int32_t src_slot, double source_radius, double fleet_speed,
                                         double dir_x, double dir_y, int32_t max_steps,
                                         bool skip_static_and_sun_checks,
                                         int32_t target_slot,
                                         const std::array<uint8_t, kPlanets>
                                             &is_dynamic_slot,
                                         const std::array<int32_t, kPlanets> &slot_planet_id,
                                         const std::array<int32_t, kPlanets>
                                             &slot_comet_internal_id,
                                         const std::array<int32_t, kPlanets> &static_slots,
                                         int32_t static_slots_n,
                                         const std::array<int32_t, kPlanets> &dynamic_slots,
                                         int32_t dynamic_slots_n,
                                         const std::array<double, kPlanets> &slot_radius,
                                         int32_t &hit_steps) {
  hit_steps = -1;
  if (base_frame < 0 || base_frame >= noop.n_frames || max_steps <= 0) {
    return kHitNone;
  }
  TORCH_CHECK_DISABLED(slot_planet_id[static_cast<uint32_t>(src_slot)] >= 0,
              "simulate_first_hit_slot_from_dir: missing source id");
  const uint32_t src_idx = static_cast<uint32_t>(base_frame) * static_cast<uint32_t>(kPlanets) +
                         static_cast<uint32_t>(src_slot);
  if (!noop_slot_has_expected_planet(
          noop.flat[src_idx], slot_planet_id[static_cast<uint32_t>(src_slot)],
          slot_comet_internal_id[static_cast<uint32_t>(src_slot)])) {
    return kHitNone;
  }
  double sx = noop.flat[src_idx].x;
  double sy = noop.flat[src_idx].y;
  const double c = dir_x;
  const double s = dir_y;
  const double start_offset = source_radius + 0.1;
  sx += c * start_offset;
  sy += s * start_offset;
  return simulate_first_hit_slot_from_position(
      noop, base_frame, sx, sy, fleet_speed, dir_x, dir_y, max_steps,
      skip_static_and_sun_checks, target_slot, is_dynamic_slot, slot_planet_id,
      slot_comet_internal_id, static_slots, static_slots_n, dynamic_slots,
      dynamic_slots_n, slot_radius, hit_steps, nullptr);
}

void fill_honest_aim_nan_outputs(torch::Tensor out_x, torch::Tensor out_y, torch::Tensor out_turns,
                                 torch::Tensor out_intercept_ok,
                                 torch::Tensor out_intercept_fail_reason) {
  TORCH_CHECK_DISABLED(out_x.dim() == 2 && out_x.size(0) == kPlanets && out_x.size(1) == kHitClasses,
              "honest aim out_x shape");
  TORCH_CHECK_DISABLED(out_y.sizes() == out_x.sizes() && out_turns.sizes() == out_x.sizes() &&
                  out_intercept_ok.sizes() == out_x.sizes() &&
                  out_intercept_fail_reason.sizes() == out_x.sizes(),
              "honest aim outputs shape mismatch");
  TORCH_CHECK_DISABLED(out_x.dtype() == torch::kFloat32 && !out_x.is_cuda(), "honest aim out_x dtype/device");
  TORCH_CHECK_DISABLED(out_y.dtype() == torch::kFloat32 && !out_y.is_cuda(), "honest aim out_y dtype/device");
  TORCH_CHECK_DISABLED(out_turns.dtype() == torch::kFloat32 && !out_turns.is_cuda(),
              "honest aim out_turns dtype/device");
  TORCH_CHECK_DISABLED(out_intercept_ok.dtype() == torch::kFloat32 && !out_intercept_ok.is_cuda(),
              "honest aim out_intercept_ok dtype/device");
  TORCH_CHECK_DISABLED(out_intercept_fail_reason.dtype() == torch::kFloat32 &&
                  !out_intercept_fail_reason.is_cuda(),
              "honest aim out_intercept_fail_reason dtype/device");
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  out_x.fill_(nan_v);
  out_y.fill_(nan_v);
  out_turns.fill_(nan_v);
  out_intercept_ok.zero_();
  out_intercept_fail_reason.zero_();
}

EdgeActionHit edge_action_hit_from_dir(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    int32_t src_slot, double source_radius, int32_t dst_slot,
    int32_t ship_count, double ship_speed, double dir_x, double dir_y,
    double turns_to_target, bool check_static_blockers, bool /*target_static*/,
    bool /*has_target_hit_for_prior_bucket*/,
    const std::array<uint8_t, kPlanets> &is_static_slot,
    const std::array<uint8_t, kPlanets> &is_dynamic_slot,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<int32_t, kPlanets> &static_slots, int32_t static_slots_n,
    const std::array<int32_t, kPlanets> &dynamic_slots,
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius) {


  TORCH_CHECK_DISABLED(remaining_steps > 0, "edge action hit: remaining_steps");
  TORCH_CHECK_DISABLED(ship_count > 0, "edge action hit: ship count");
  const double speed =
      orbit_cpp_fleet_speed(static_cast<double>(ship_count), ship_speed);
  TORCH_CHECK_DISABLED(speed > 0.0 && std::isfinite(speed), "edge action hit: speed");
  if (!std::isfinite(dir_x) || !std::isfinite(dir_y) ||
      !std::isfinite(turns_to_target) || turns_to_target < 0.0) {
    return EdgeActionHit{};
  }
  if (!(turns_to_target < static_cast<double>(remaining_steps))) {
    return EdgeActionHit{false, -1, kHitKindTimeout, kHitNone};
  }
  const int32_t target_hit_steps =
      static_cast<int32_t>(std::ceil(turns_to_target));
  if (target_hit_steps > kHonestHitTraceMaxSteps) {
    return EdgeActionHit{false, -1, kHitKindTimeout, kHitNone};
  }

  const int32_t max_steps =
      std::min<int32_t>(kHonestHitTraceMaxSteps,
                        std::min<int32_t>(
                            remaining_steps,
                            target_hit_steps + 1));
  TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
              "edge action hit: noop base frame");
  TORCH_CHECK_DISABLED(slot_planet_id[static_cast<uint32_t>(src_slot)] >= 0,
              "edge action hit: missing source id");
  const uint32_t src_idx =
      static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets) +
      static_cast<uint32_t>(src_slot);
  const NoopCachedPlanet &src = noop.flat[src_idx];
  TORCH_CHECK_DISABLED(noop_slot_has_expected_planet(
                  src, slot_planet_id[static_cast<uint32_t>(src_slot)],
                  slot_comet_internal_id[static_cast<uint32_t>(src_slot)]),
              "edge action hit: source slot mismatch");
  const double start_offset = source_radius + 0.1;
  const double start_x = src.x + dir_x * start_offset;
  const double start_y = src.y + dir_y * start_offset;
  const double finish_x = start_x + dir_x * speed * turns_to_target;
  const double finish_y = start_y + dir_y * speed * turns_to_target;
  if (check_static_blockers) {
    int32_t static_blocker_hit_steps = -1;
    const int32_t static_blocker_slot = first_static_blocker_on_segment(
        noop, noop_base_frame, start_x, start_y, finish_x, finish_y,
        dst_slot, slot_planet_id, slot_comet_internal_id, slot_radius,
        &static_blocker_hit_steps);
    if (static_blocker_slot == kHitSun) {
      return EdgeActionHit{false, static_blocker_hit_steps, kHitKindSun, kHitSun};
    }
    if (static_blocker_slot != kHitNone) {
      return EdgeActionHit{false, static_blocker_hit_steps, kHitKindStatic,
                           static_blocker_slot};
    }
  } else if (point_to_segment_distance_sq(kCenter, kCenter, start_x, start_y,
                                          finish_x, finish_y) <
             kSunRadius * kSunRadius +
                 kStaticSegmentCollisionDistanceSqEpsilon) {
    return EdgeActionHit{false, 1, kHitKindSun, kHitSun};
  }

  int32_t hit_steps = -1;
  int32_t kind = kHitKindNone;
  const int32_t hit_slot = simulate_first_hit_slot_from_position(
      noop, noop_base_frame, start_x, start_y, speed, dir_x, dir_y,
      max_steps, true, dst_slot, is_dynamic_slot, slot_planet_id,
      slot_comet_internal_id, static_slots, static_slots_n, dynamic_slots,
      dynamic_slots_n, slot_radius, hit_steps, nullptr);

  if (hit_slot == dst_slot) {
    kind = kHitKindTarget;
  } else if (hit_slot >= 0 && hit_slot < kPlanets &&
             is_static_slot[static_cast<uint32_t>(hit_slot)] != 0) {
    kind = kHitKindStatic;
  } else if (hit_slot >= 0 && hit_slot < kPlanets &&
             is_dynamic_slot[static_cast<uint32_t>(hit_slot)] != 0) {
    kind = kHitKindDynamic;
  } else if (hit_slot == kHitNone) {
    kind = (max_steps == remaining_steps) ? kHitKindEndOfGame : kHitKindTimeout;
  } else if (hit_slot == kHitSun) {
    kind = kHitKindSun;
  } else if (hit_slot == kHitOutOfBoard) {
    kind = kHitKindOutOfBoard;
  }

  if (kind == kHitKindTarget) {
    TORCH_CHECK_DISABLED(hit_steps >= 1, "edge action hit: target hit without steps");
    return EdgeActionHit{true, hit_steps, kind, hit_slot};
  }
  return EdgeActionHit{false, hit_steps, kind, hit_slot};
}

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
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius) {
  TORCH_CHECK_DISABLED(0 <= aim_index && aim_index < kEdgeInterceptAimCount,
              "cached dynamic-dynamic hit: aim index");
  if (!std::isfinite(dir_x) || !std::isfinite(dir_y) ||
      !std::isfinite(turns_to_target) || turns_to_target < 0.0) {
    return EdgeActionHitWithAim{};
  }
  //const double dir_norm = std::sqrt(dir_x * dir_x + dir_y * dir_y);
  //TORCH_CHECK_DISABLED(std::abs(dir_norm - 1.0) < 1e-6,
  //            "cached dynamic-dynamic hit: non-unit direction");
  const std::array<uint8_t, kPlanets> no_static_slots{};
  const EdgeActionHit action_hit = edge_action_hit_from_dir(
      noop, noop_base_frame, remaining_steps, src_slot, source_radius,
      dst_slot, ship_count, ship_speed, dir_x, dir_y, turns_to_target,
      false, false, false, no_static_slots, is_dynamic_slot, slot_planet_id,
      slot_comet_internal_id, static_slots, static_slots_n, dynamic_slots,
      dynamic_slots_n, slot_radius);
  return EdgeActionHitWithAim{
      action_hit,
      true, aim_index, dir_x, dir_y, turns_to_target};
}

EdgeActionHit cached_dynamic_dynamic_comet_overlay_hit(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    int32_t src_slot, double source_radius, int32_t dst_slot,
    int32_t ship_count, double ship_speed, double dir_x, double dir_y,
    double turns_to_target,
    const std::array<int32_t, kPlanets> &slot_planet_id,
    const std::array<int32_t, kPlanets> &slot_comet_internal_id,
    const std::array<double, kPlanets> &slot_radius) {
  TORCH_CHECK_DISABLED(remaining_steps > 0,
              "cached dynamic-dynamic comet overlay: remaining_steps");
  TORCH_CHECK_DISABLED(ship_count > 0,
              "cached dynamic-dynamic comet overlay: ship count");
  TORCH_CHECK_DISABLED(noop.spatial_grid != nullptr,
              "cached dynamic-dynamic comet overlay: grid");
  TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
              "cached dynamic-dynamic comet overlay: noop base frame");
  TORCH_CHECK_DISABLED(slot_planet_id[static_cast<uint32_t>(src_slot)] >= 0,
              "cached dynamic-dynamic comet overlay: missing source id");
  TORCH_CHECK_DISABLED(slot_comet_internal_id[static_cast<uint32_t>(dst_slot)] < 0,
              "cached dynamic-dynamic comet overlay: comet target");
  if (!std::isfinite(dir_x) || !std::isfinite(dir_y) ||
      !std::isfinite(turns_to_target) || turns_to_target < 0.0) {
    return EdgeActionHit{};
  }
  const double speed =
      orbit_cpp_fleet_speed(static_cast<double>(ship_count), ship_speed);
  TORCH_CHECK_DISABLED(speed > 0.0 && std::isfinite(speed),
              "cached dynamic-dynamic comet overlay: speed");
  const int32_t target_hit_steps =
      static_cast<int32_t>(std::ceil(turns_to_target));
  const int32_t max_steps =
      std::min<int32_t>(kHonestHitTraceMaxSteps,
                        std::min<int32_t>(
                            remaining_steps,
                            target_hit_steps + 1));
  const uint32_t src_idx =
      static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets) +
      static_cast<uint32_t>(src_slot);
  const NoopCachedPlanet &src = noop.flat[src_idx];
  TORCH_CHECK_DISABLED(noop_slot_has_expected_planet(
                  src, slot_planet_id[static_cast<uint32_t>(src_slot)],
                  slot_comet_internal_id[static_cast<uint32_t>(src_slot)]),
              "cached dynamic-dynamic comet overlay: source slot mismatch");
  const std::array<uint8_t, kPlanets> no_static_slots{};
  std::array<uint8_t, kPlanets> comet_slots{};
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    comet_slots[static_cast<uint32_t>(slot)] =
        slot_comet_internal_id[static_cast<uint32_t>(slot)] >= 0 ? uint8_t{1}
                                                               : uint8_t{0};
  }
  int32_t hit_steps = -1;
  const int32_t hit_slot = simulate_first_hit_slot_from_position(
      noop, noop_base_frame, src.x + dir_x * (source_radius + 0.1),
      src.y + dir_y * (source_radius + 0.1), speed, dir_x, dir_y,
      max_steps, true, dst_slot, comet_slots, slot_planet_id,
      slot_comet_internal_id, {}, 0, {}, 0, slot_radius, hit_steps, nullptr,
      true);
  if (hit_slot >= 0 && hit_slot < kPlanets) {
    return EdgeActionHit{false, hit_steps, kHitKindDynamic, hit_slot};
  }
  return EdgeActionHit{};
}

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
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius) {
  TORCH_CHECK_DISABLED(0 <= aim_index && aim_index < kEdgeInterceptAimCount,
              "edge action intercept aim: aim index");
  const EdgeInterceptAimCandidate &candidate =
      aim.candidates[static_cast<uint32_t>(aim_index)];
  if (!edge_intercept_candidate_is_valid(candidate)) {
    return EdgeActionHitWithAim{};
  }
  EdgeActionHit action_hit = edge_action_hit_from_dir(
      noop, noop_base_frame, remaining_steps, src_slot, src.radius, dst_slot,
      ship_count, ship_speed, candidate.dir_x, candidate.dir_y,
      candidate.turns_to_target, true, target_static,
      has_target_hit_for_prior_bucket, is_static_slot, is_dynamic_slot,
      slot_planet_id, slot_comet_internal_id, static_slots, static_slots_n,
      dynamic_slots, dynamic_slots_n, slot_radius);
  return EdgeActionHitWithAim{action_hit, true, aim_index, candidate.dir_x,
                              candidate.dir_y, candidate.turns_to_target};
}

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
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius) {
  TORCH_CHECK_DISABLED(0 <= aim_index && aim_index < kEdgeInterceptAimCount,
              "static checked intercept aim: aim index");
  const EdgeInterceptAimCandidate &candidate =
      aim.candidates[static_cast<uint32_t>(aim_index)];
  if (!edge_intercept_candidate_is_valid(candidate)) {
    return EdgeActionHitWithAim{};
  }
  EdgeActionHit action_hit = edge_action_hit_from_dir(
      noop, noop_base_frame, remaining_steps, src_slot, src.radius, dst_slot,
      ship_count, ship_speed, candidate.dir_x, candidate.dir_y,
      candidate.turns_to_target, false, true, false, is_static_slot,
      is_dynamic_slot, slot_planet_id, slot_comet_internal_id, static_slots,
      static_slots_n, dynamic_slots, dynamic_slots_n, slot_radius);
  return EdgeActionHitWithAim{action_hit, true, aim_index, candidate.dir_x,
                              candidate.dir_y, candidate.turns_to_target};
}

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
    int32_t dynamic_slots_n, const std::array<double, kPlanets> &slot_radius) {
  bool has_first_result = false;
  EdgeActionHitWithAim first_result{};
  for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
    EdgeActionHitWithAim action_hit_with_aim = edge_action_hit_for_intercept_aim(
        noop, noop_base_frame, remaining_steps, src, src_slot, dst_slot,
        ship_count, ship_speed, aim, aim_index, target_static,
        has_target_hit_for_prior_bucket, is_static_slot, is_dynamic_slot,
        slot_planet_id, slot_comet_internal_id, static_slots, static_slots_n,
        dynamic_slots, dynamic_slots_n, slot_radius);
    if (!action_hit_with_aim.has_aim) {
      continue;
    }
    if (!has_first_result) {
      first_result = action_hit_with_aim;
      has_first_result = true;
    }
    if (action_hit_with_aim.hit.hit_kind == kHitKindTarget) {
      return action_hit_with_aim;
    }
  }
  TORCH_CHECK_DISABLED(has_first_result, "edge action intercept aim: no valid candidates");
  return first_result;
}

std::vector<Fleet> fleet_rows_tensor_to_vector(torch::Tensor fleet_rows) {
  TORCH_CHECK_DISABLED(fleet_rows.device().is_cpu(), "fleet_rows must be CPU");
  TORCH_CHECK_DISABLED(fleet_rows.dtype() == torch::kFloat32 || fleet_rows.dtype() == torch::kFloat64,
              "fleet_rows must be float32 or float64");
  TORCH_CHECK_DISABLED(fleet_rows.dim() == 2 && fleet_rows.size(1) == 7,
              "fleet_rows must be [N,7]");
  fleet_rows = fleet_rows.contiguous();
  const int32_t n = fleet_rows.size(0);
  std::vector<Fleet> fleets;
  fleets.reserve(static_cast<uint32_t>(n));
  if (fleet_rows.dtype() == torch::kFloat64) {
    const auto a = fleet_rows.accessor<double, 2>();
    for (int32_t i = 0; i < n; ++i) {
      Fleet f;
      f.id = static_cast<int32_t>(a[i][0]);
      f.owner = static_cast<int32_t>(a[i][1]);
      f.x = a[i][2];
      f.y = a[i][3];
      f.angle = a[i][4];
      f.from_planet_id = static_cast<int32_t>(a[i][5]);
      f.ships = a[i][6];
      fleets.push_back(f);
    }
  } else {
    const auto a = fleet_rows.accessor<float, 2>();
    for (int32_t i = 0; i < n; ++i) {
      Fleet f;
      f.id = static_cast<int32_t>(a[i][0]);
      f.owner = static_cast<int32_t>(a[i][1]);
      f.x = static_cast<double>(a[i][2]);
      f.y = static_cast<double>(a[i][3]);
      f.angle = static_cast<double>(a[i][4]);
      f.from_planet_id = static_cast<int32_t>(a[i][5]);
      f.ships = static_cast<double>(a[i][6]);
      fleets.push_back(f);
    }
  }
  return fleets;
}

std::vector<Planet> external_planet_rows_tensor_to_vector(
    torch::Tensor planet_rows, int32_t planet_count, const char *context) {
  TORCH_CHECK_DISABLED(planet_rows.device().is_cpu(), context, ": planet_rows must be CPU");
  TORCH_CHECK_DISABLED(planet_rows.dtype() == torch::kFloat32 ||
                  planet_rows.dtype() == torch::kFloat64,
              context, ": planet_rows must be float32 or float64");
  TORCH_CHECK_DISABLED(planet_rows.sizes() == torch::IntArrayRef({kPlanets, kPlanetRowLen}),
              context, ": planet_rows must be [kPlanets,7]");
  TORCH_CHECK_DISABLED(0 <= planet_count && planet_count <= kPlanets,
              context, ": planet_count");
  planet_rows = planet_rows.contiguous();
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
      TORCH_CHECK_DISABLED(p.id >= 0, context, ": planet id");
      TORCH_CHECK_DISABLED(p.radius > 0.0, context, ": planet radius");
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
      TORCH_CHECK_DISABLED(p.id >= 0, context, ": planet id");
      TORCH_CHECK_DISABLED(p.radius > 0.0, context, ": planet radius");
      planets.push_back(p);
    }
  }
  return planets;
}

torch::Tensor fleet_arrivals_for_fleets_from_noop(
    const std::vector<Fleet> &fleets, int32_t horizon, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const NoopView &noop) {
  TORCH_CHECK_DISABLED(horizon > 0, "fleet arrivals horizon must be positive");
  TORCH_CHECK_DISABLED(num_agents > 0 && num_agents <= kPlayerAxisSlots, "fleet arrivals num_agents");
  TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
              "fleet arrivals: noop cache missing current frame");
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
  int32_t n = 0;
  while (n < kPlanets && planets_row[static_cast<uint32_t>(n)].id >= 0) {
    ++n;
  }
  std::array<int32_t, kPlanets> dynamic_slots{};
  std::array<int32_t, kPlanets> static_slots{};
  std::array<uint8_t, kPlanets> is_dynamic_slot{};
  int32_t dynamic_slots_n = 0;
  int32_t static_slots_n = 0;
  std::array<double, kPlanets> slot_radius{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  for (int32_t i = 0; i < kPlanets; ++i) {
    slot_radius[static_cast<uint32_t>(i)] = 0.0;
    slot_planet_id[static_cast<uint32_t>(i)] = -1;
    slot_comet_internal_id[static_cast<uint32_t>(i)] = -1;
  }
  for (int32_t slot = 0; slot < n; ++slot) {
    const NoopCachedPlanet &p = planets_row[static_cast<uint32_t>(slot)];
    slot_radius[static_cast<uint32_t>(slot)] = p.radius;
    slot_planet_id[static_cast<uint32_t>(slot)] = p.id;
    slot_comet_internal_id[static_cast<uint32_t>(slot)] = p.comet_internal_id;
    const bool is_comet = comet_planet_ids.contains(p.id);
    const bool rotates = planet_is_rotating_for_mask(p.id, p.x, p.y, p.radius, comet_planet_ids);
    if (is_comet || rotates) {
      dynamic_slots[static_cast<uint32_t>(dynamic_slots_n)] = slot;
      is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
      ++dynamic_slots_n;
    } else {
      static_slots[static_cast<uint32_t>(static_slots_n)] = slot;
      ++static_slots_n;
    }
  }

  torch::Tensor arrivals =
      torch::zeros({horizon, kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto out = arrivals.accessor<float, 3>();
  const int32_t remaining_steps = noop.n_frames - noop_base_frame;
  const int32_t max_steps = std::min<int32_t>(horizon, remaining_steps);
  for (const Fleet &f : fleets) {
    TORCH_CHECK_DISABLED(0 <= f.owner && f.owner < num_agents, "fleet arrivals: bad fleet owner");
    TORCH_CHECK_DISABLED(std::isfinite(f.x) && std::isfinite(f.y) && std::isfinite(f.angle),
                "fleet arrivals: non-finite fleet geometry");
    TORCH_CHECK_DISABLED(f.ships > 0.0 && std::isfinite(f.ships),
                "fleet arrivals: bad fleet ships");
    const double speed = orbit_cpp_fleet_speed(f.ships, ship_speed);
    TORCH_CHECK_DISABLED(speed > 0.0, "fleet arrivals: bad fleet speed");
    const double dir_x = std::cos(f.angle);
    const double dir_y = std::sin(f.angle);
    int32_t hit_steps = -1;
    const int32_t hit_slot = simulate_first_hit_slot_from_position(
        noop, noop_base_frame, f.x, f.y, speed, dir_x, dir_y, max_steps,
        false, -1, is_dynamic_slot, slot_planet_id, slot_comet_internal_id,
        static_slots, static_slots_n, dynamic_slots, dynamic_slots_n,
        slot_radius, hit_steps, nullptr);
    if (hit_slot >= 0 && hit_slot < kPlanets && hit_steps >= 1 && hit_steps <= horizon) {
      out[hit_steps - 1][hit_slot][f.owner] += static_cast<float>(f.ships);
    }
  }
  return arrivals;
}

torch::Tensor fleet_arrivals_for_fleets(
    const std::vector<Fleet> &fleets, int32_t horizon, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid) {
  const NoopView noop = make_noop_view(noop_cached_planets_flat, noop_spatial_grid);
  return fleet_arrivals_for_fleets_from_noop(
      fleets, horizon, num_agents, ship_speed, noop_base_frame, comet_planet_ids,
      noop);
}

EdgeFrameCollisionMetadata edge_frame_collision_metadata(
    const NoopView &noop, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids) {
  EdgeFrameCollisionMetadata metadata;
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    metadata.slot_radius[static_cast<uint32_t>(slot)] = 0.0;
    metadata.slot_planet_id[static_cast<uint32_t>(slot)] = -1;
    metadata.slot_comet_internal_id[static_cast<uint32_t>(slot)] = -1;
    metadata.is_static_slot[static_cast<uint32_t>(slot)] = 0;
    metadata.is_dynamic_slot[static_cast<uint32_t>(slot)] = 0;
  }
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    const NoopCachedPlanet &p = planets_row[static_cast<uint32_t>(slot)];
    if (p.id < 0) {
      continue;
    }
    metadata.slot_radius[static_cast<uint32_t>(slot)] = p.radius;
    metadata.slot_planet_id[static_cast<uint32_t>(slot)] = p.id;
    metadata.slot_comet_internal_id[static_cast<uint32_t>(slot)] = p.comet_internal_id;
    const bool is_comet = comet_planet_ids.contains(p.id);
    const bool rotates =
        planet_is_rotating_for_mask(p.id, p.x, p.y, p.radius, comet_planet_ids);
    if (is_comet || rotates) {
      metadata.dynamic_slots[static_cast<uint32_t>(metadata.dynamic_slots_n)] = slot;
      metadata.is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
      ++metadata.dynamic_slots_n;
    } else {
      metadata.static_slots[static_cast<uint32_t>(metadata.static_slots_n)] = slot;
      metadata.is_static_slot[static_cast<uint32_t>(slot)] = 1;
      ++metadata.static_slots_n;
    }
  }
  return metadata;
}

int32_t direct_edge_target_hit_steps_with_metadata(
    const NoopView &noop, int32_t noop_base_frame, int32_t remaining_steps,
    int32_t src_slot, const NoopCachedPlanet &src, int32_t dst_slot,
    const NoopCachedPlanet &dst,
    const SmallPlanetIdSet &comet_planet_ids, int32_t ship_count,
    double ship_speed, const EdgeFrameCollisionMetadata &metadata) {
  TORCH_CHECK_DISABLED(ship_count > 0, "edge resolution features: ship_count");
  const bool target_static =
      metadata.is_static_slot[static_cast<uint32_t>(dst_slot)] != 0;
  for (int32_t aim_index = 0; aim_index < kEdgeInterceptAimCount; ++aim_index) {
    EdgeInterceptAim aim;
    const bool valid = edge_intercept_aim_for_ship_count_and_aim_index(
        noop, src, dst_slot, dst, comet_planet_ids, ship_count, ship_speed,
        noop_base_frame, aim_index, &aim);
    if (!valid) {
      continue;
    }
    const EdgeActionHitWithAim action_hit_with_aim =
        edge_action_hit_for_intercept_aim(
            noop, noop_base_frame, remaining_steps, src, src_slot, dst_slot,
            ship_count, ship_speed, aim, aim_index, target_static, false,
            metadata.is_static_slot, metadata.is_dynamic_slot,
            metadata.slot_planet_id, metadata.slot_comet_internal_id,
            metadata.static_slots, metadata.static_slots_n,
            metadata.dynamic_slots, metadata.dynamic_slots_n,
            metadata.slot_radius);
    TORCH_CHECK_DISABLED(action_hit_with_aim.has_aim,
                         "edge resolution features: valid aim missing hit");
    if (action_hit_with_aim.hit.hit_kind == kHitKindTarget) {
      return action_hit_with_aim.hit.hit_steps;
    }
  }
  return -1;
}

py::list fleet_hit_traces_for_fleets(
    const std::vector<Fleet> &fleets, int32_t horizon, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid) {
  TORCH_CHECK_DISABLED(horizon > 0, "fleet hit traces horizon must be positive");
  TORCH_CHECK_DISABLED(num_agents > 0 && num_agents <= kPlayerAxisSlots, "fleet hit traces num_agents");
  const NoopView noop = make_noop_view(noop_cached_planets_flat, noop_spatial_grid);
  TORCH_CHECK_DISABLED(noop_base_frame >= 0 && noop_base_frame < noop.n_frames,
              "fleet hit traces: noop cache missing current frame");
  const NoopCachedPlanet *planets_row =
      noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
  int32_t n = 0;
  while (n < kPlanets && planets_row[static_cast<uint32_t>(n)].id >= 0) {
    ++n;
  }
  std::array<int32_t, kPlanets> dynamic_slots{};
  std::array<int32_t, kPlanets> static_slots{};
  std::array<uint8_t, kPlanets> is_dynamic_slot{};
  int32_t dynamic_slots_n = 0;
  int32_t static_slots_n = 0;
  std::array<double, kPlanets> slot_radius{};
  std::array<int32_t, kPlanets> slot_planet_id{};
  std::array<int32_t, kPlanets> slot_comet_internal_id{};
  for (int32_t i = 0; i < kPlanets; ++i) {
    slot_radius[static_cast<uint32_t>(i)] = 0.0;
    slot_planet_id[static_cast<uint32_t>(i)] = -1;
    slot_comet_internal_id[static_cast<uint32_t>(i)] = -1;
  }
  for (int32_t slot = 0; slot < n; ++slot) {
    const NoopCachedPlanet &p = planets_row[static_cast<uint32_t>(slot)];
    slot_radius[static_cast<uint32_t>(slot)] = p.radius;
    slot_planet_id[static_cast<uint32_t>(slot)] = p.id;
    slot_comet_internal_id[static_cast<uint32_t>(slot)] = p.comet_internal_id;
    const bool is_comet = comet_planet_ids.contains(p.id);
    const bool rotates = planet_is_rotating_for_mask(p.id, p.x, p.y, p.radius, comet_planet_ids);
    if (is_comet || rotates) {
      dynamic_slots[static_cast<uint32_t>(dynamic_slots_n)] = slot;
      is_dynamic_slot[static_cast<uint32_t>(slot)] = 1;
      ++dynamic_slots_n;
    } else {
      static_slots[static_cast<uint32_t>(static_slots_n)] = slot;
      ++static_slots_n;
    }
  }

  py::list traces;
  const int32_t remaining_steps = noop.n_frames - noop_base_frame;
  const int32_t max_steps = std::min<int32_t>(horizon, remaining_steps);
  for (const Fleet &f : fleets) {
    TORCH_CHECK_DISABLED(0 <= f.owner && f.owner < num_agents, "fleet hit traces: bad fleet owner");
    TORCH_CHECK_DISABLED(std::isfinite(f.x) && std::isfinite(f.y) && std::isfinite(f.angle),
                "fleet hit traces: non-finite fleet geometry");
    TORCH_CHECK_DISABLED(f.ships > 0.0 && std::isfinite(f.ships),
                "fleet hit traces: bad fleet ships");
    const double speed = orbit_cpp_fleet_speed(f.ships, ship_speed);
    TORCH_CHECK_DISABLED(speed > 0.0, "fleet hit traces: bad fleet speed");
    const double dir_x = std::cos(f.angle);
    const double dir_y = std::sin(f.angle);
    int32_t hit_steps = -1;
    FirstHitResult hit;
    const int32_t hit_slot = simulate_first_hit_slot_from_position(
        noop, noop_base_frame, f.x, f.y, speed, dir_x, dir_y, max_steps,
        false, -1, is_dynamic_slot, slot_planet_id, slot_comet_internal_id,
        static_slots, static_slots_n, dynamic_slots, dynamic_slots_n,
        slot_radius, hit_steps, &hit);
    if (hit_slot < 0 || hit_steps < 1 || hit_steps > horizon) {
      continue;
    }
    py::dict row;
    row["fleet_id"] = py::cast(f.id);
    row["owner"] = py::cast(f.owner);
    row["ships"] = py::cast(f.ships);
    row["x0"] = py::cast(f.x);
    row["y0"] = py::cast(f.y);
    row["x1"] = py::cast(hit.fleet_x);
    row["y1"] = py::cast(hit.fleet_y);
    row["hit_slot"] = py::cast(hit.hit_slot);
    row["hit_planet_id"] = py::cast(hit.hit_planet_id);
    row["hit_steps"] = py::cast(hit.hit_steps);
    row["object_x"] = py::cast(hit.object_x);
    row["object_y"] = py::cast(hit.object_y);
    row["object_radius"] = py::cast(hit.object_radius);
    traces.append(row);
  }
  return traces;
}

}  // namespace orbit_wars_honest
