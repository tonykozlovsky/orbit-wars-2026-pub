#include "io.h"

#include <algorithm>
#include <cstdio>
#include <numeric>

std::string orbit_wars_reset_trace_fmt_double(double v) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

std::string orbit_wars_format_double_for_reset_trace(double v) {
  return orbit_wars_reset_trace_fmt_double(v);
}

std::string orbit_wars_format_double_two_decimals_for_reset_trace(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f", v);
  return std::string(buf);
}

std::string orbit_wars_trace_planets_full_digest(const std::vector<Planet> &planets) {
  std::vector<uint32_t> ix(planets.size());
  std::iota(ix.begin(), ix.end(), 0);
  std::sort(ix.begin(), ix.end(),
            [&](uint32_t a, uint32_t b) { return planets[a].id < planets[b].id; });
  std::string out;
  for (uint32_t k = 0; k < ix.size(); ++k) {
    if (k > 0) {
      out.push_back(';');
    }
    const Planet &p = planets[ix[k]];
    out += std::to_string(p.id) + "|" + std::to_string(p.owner) + "|" +
           orbit_wars_reset_trace_fmt_double(p.x) + "|" +
           orbit_wars_reset_trace_fmt_double(p.y) + "|" +
           orbit_wars_reset_trace_fmt_double(p.radius) + "|" +
           orbit_wars_reset_trace_fmt_double(p.ships) + "|" +
           orbit_wars_reset_trace_fmt_double(p.production);
  }
  return out;
}

std::string orbit_wars_trace_combat_digest(
    const std::array<std::vector<Fleet>, kPlanets> &combat,
    const std::vector<Planet> &planets) {
  assert(planets.size() <= static_cast<uint32_t>(kPlanets));
  std::vector<uint32_t> slots;
  slots.reserve(planets.size());
  for (uint32_t slot = 0; slot < planets.size(); ++slot) {
    if (!combat[slot].empty()) {
      slots.push_back(slot);
    }
  }
  std::sort(slots.begin(), slots.end(),
            [&](uint32_t a, uint32_t b) { return planets[a].id < planets[b].id; });
  std::string out;
  for (uint32_t i = 0; i < slots.size(); ++i) {
    if (i > 0) {
      out.push_back(';');
    }
    const uint32_t slot = slots[i];
    const int32_t pid = planets[slot].id;
    std::vector<Fleet> sorted = combat[slot];
    std::sort(sorted.begin(), sorted.end(),
              [](const Fleet &a, const Fleet &b) { return a.id < b.id; });
    out += std::to_string(pid) + ":";
    for (uint32_t u = 0; u < sorted.size(); ++u) {
      if (u > 0) {
        out.push_back(',');
      }
      const Fleet &f = sorted[u];
      out += std::to_string(f.id) + "/" + std::to_string(f.owner) + "/" +
             orbit_wars_reset_trace_fmt_double(f.ships);
    }
  }
  return out;
}
