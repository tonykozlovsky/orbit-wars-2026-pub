#include "simulation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

constexpr int32_t kFleetSpeedSaturationShipCount = 1000;

const std::array<double, kFleetSpeedSaturationShipCount> &fleet_speed_factor_table() {
  static const std::array<double, kFleetSpeedSaturationShipCount> table = [] {
    std::array<double, kFleetSpeedSaturationShipCount> values{};
    values[0] = 0.0;
    for (int32_t ship_count = 1; ship_count < kFleetSpeedSaturationShipCount; ++ship_count) {
      values[static_cast<uint32_t>(ship_count)] =
          std::pow(std::log(static_cast<double>(ship_count)) / std::log(1000.0), 1.5);
    }
    return values;
  }();
  return table;
}

}  // namespace

double distance_xy_sq(double x0, double y0, double x1, double y1) {
  const double dx = x0 - x1;
  const double dy = y0 - y1;
  return dx * dx + dy * dy;
}

double point_to_segment_distance_sq(double px, double py, double vx, double vy, double wx,
                                    double wy) {
  const double dx = wx - vx;
  const double dy = wy - vy;
  const double l2 = dx * dx + dy * dy;
  if (l2 == 0.0) {
    return distance_xy_sq(px, py, vx, vy);
  }
  double t = ((px - vx) * dx + (py - vy) * dy) / l2;
  t = std::max(0.0, std::min(1.0, t));
  const double qx = vx + t * dx;
  const double qy = vy + t * dy;
  return distance_xy_sq(px, py, qx, qy);
}

bool orbit_wars_swept_pair_hit(double ax, double ay, double bx, double by, double p0x,
                               double p0y, double p1x, double p1y, double r) {
  const double d0x = ax - p0x;
  const double d0y = ay - p0y;
  const double dvx = (bx - ax) - (p1x - p0x);
  const double dvy = (by - ay) - (p1y - p0y);
  const double a = dvx * dvx + dvy * dvy;
  const double b = 2.0 * (d0x * dvx + d0y * dvy);
  const double c = d0x * d0x + d0y * d0y - r * r;
  if (a < 1e-12) {
    return c <= 0.0;
  }
  const double q1 = a + b + c;
  if (c <= 0.0 || q1 <= 0.0) {
    return true;
  }
  if (b >= 0.0 || b <= -2.0 * a) {
    return false;
  }
  return b * b >= 4.0 * a * c;
}

double orbit_cpp_fleet_speed(double ship_count, double ship_speed_max) {
  TORCH_CHECK_DISABLED(ship_count > 0.0, "ship_count");
  const int32_t ship_count_i = static_cast<int32_t>(ship_count);
  assert(ship_count == static_cast<double>(ship_count_i));
  if (ship_count_i >= kFleetSpeedSaturationShipCount) {
    return ship_speed_max;
  }
  const double speed =
      1.0 + (ship_speed_max - 1.0) *
                fleet_speed_factor_table()[static_cast<uint32_t>(ship_count_i)];
  return std::min(speed, ship_speed_max);
}
