#pragma once

#include "common.h"

#include <array>
#include <string>
#include <vector>

std::string orbit_wars_format_double_for_reset_trace(double v);
std::string orbit_wars_format_double_two_decimals_for_reset_trace(double v);
std::string orbit_wars_reset_trace_fmt_double(double v);
std::string orbit_wars_trace_planets_full_digest(const std::vector<Planet> &planets);
std::string orbit_wars_trace_combat_digest(
    const std::array<std::vector<Fleet>, kPlanets> &combat,
    const std::vector<Planet> &planets);
