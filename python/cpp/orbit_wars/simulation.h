#pragma once

#include "common.h"

double distance_xy_sq(double x0, double y0, double x1, double y1);
double point_to_segment_distance_sq(double px, double py, double vx, double vy, double wx,
                                    double wy);
bool orbit_wars_swept_pair_hit(double ax, double ay, double bx, double by, double p0x,
                               double p0y, double p1x, double p1y, double r);
double orbit_cpp_fleet_speed(double ship_count, double ship_speed_max);
