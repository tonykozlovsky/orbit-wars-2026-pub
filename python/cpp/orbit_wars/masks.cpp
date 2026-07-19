#include "masks.h"

#include "library.h"
#include "simulation.h"

#include <algorithm>
#include <array>
#include <cmath>

float orbit_wars_policy_obs_edge_distance(double x0, double y0, double x1, double y1) {
  const float xf0 = static_cast<float>(x0);
  const float yf0 = static_cast<float>(y0);
  const float xf1 = static_cast<float>(x1);
  const float yf1 = static_cast<float>(y1);
  const double dx = static_cast<double>(xf0) - static_cast<double>(xf1);
  const double dy = static_cast<double>(yf0) - static_cast<double>(yf1);
  return static_cast<float>(std::sqrt(dx * dx + dy * dy));
}

static float orbit_policy_orbit_radius_float32(double x, double y) {
  const float xf = static_cast<float>(x);
  const float yf = static_cast<float>(y);
  const float c = static_cast<float>(kCenter);
  return std::sqrt((xf - c) * (xf - c) + (yf - c) * (yf - c));
}

static float orbit_policy_sun_angle_float32(double x, double y) {
  const float xf = static_cast<float>(x);
  const float yf = static_cast<float>(y);
  const float c = static_cast<float>(kCenter);
  return std::atan2(yf - c, xf - c);
}

static float orbit_policy_ship_count_feature(double ships) {
  TORCH_CHECK_DISABLED(std::isfinite(ships), "policy obs ship count: non-finite");
  TORCH_CHECK_DISABLED(ships >= 0.0, "policy obs ship count: negative");
  return static_cast<float>(ships);
}

static int32_t policy_geometry_position(int32_t player_slot, int32_t num_players) {
  TORCH_CHECK_DISABLED(0 <= player_slot && player_slot < kPlayerAxisSlots, "player_slot");
  TORCH_CHECK_DISABLED(num_players == 2 || num_players == 4, "num_players");
  if (num_players == 2) {
    TORCH_CHECK_DISABLED(player_slot < num_players, "2p player_slot < num_players");
    return player_slot == 0 ? 0 : 2;
  }
  TORCH_CHECK_DISABLED(player_slot < num_players, "4p player_slot < num_players");
  const int32_t pos = kPlayerPositionById4p[player_slot];
  TORCH_CHECK_DISABLED(0 <= pos && pos < kPlayerAxisSlots, "4p player geometry position");
  return pos;
}

static int32_t policy_slot_for_compact_agent(int32_t compact_agent, int32_t num_players) {
  TORCH_CHECK_DISABLED(0 <= compact_agent && compact_agent < num_players, "compact_agent");
  TORCH_CHECK_DISABLED(num_players == 2 || num_players == 4, "num_players");
  if (num_players == 4) {
    return compact_agent;
  }
  return compact_agent == 0 ? 0 : 3;
}

static std::pair<float, float> orbit_policy_player_relative_xy_float32(
    double x, double y, int32_t player_position) {
  const float xf = static_cast<float>(x);
  const float yf = static_cast<float>(y);
  const float b = static_cast<float>(kBoardSize);
  switch (player_position) {
    case 0:
      return {xf, yf};
    case 1:
      return {yf, b - xf};
    case 2:
      return {b - xf, b - yf};
    case 3:
      return {b - yf, xf};
  }
  TORCH_CHECK_DISABLED(false, "player_position");
  return {0.0f, 0.0f};
}

constexpr std::array<int32_t, kLegacyShipScanClasses> make_legacy_scan_ship_count_by_subindex() {
  std::array<int32_t, kLegacyShipScanClasses> counts{};
  int32_t cur = 0;
  int32_t idx = 1;
  for (int32_t g = 0; g < kLegacyShipScanBlocks; ++g) {
    const int32_t step = g + 1;
    const int32_t start = cur + step;
    for (int32_t i = 0; i < kLegacyShipScanClassesPerBlock; ++i) {
      counts[static_cast<uint32_t>(idx)] = start + step * i;
      ++idx;
    }
    cur = start + step * (kLegacyShipScanClassesPerBlock - 1);
  }
  return counts;
}

constexpr std::array<int32_t, kLegacyShipScanClasses> kLegacyScanShipCountBySubindex =
    make_legacy_scan_ship_count_by_subindex();

int32_t ship_count_for_legacy_scan_subindex(int32_t sn) {
  TORCH_CHECK_DISABLED(0 <= sn && sn < kLegacyShipScanClasses, "ship subindex");
  return kLegacyScanShipCountBySubindex[static_cast<uint32_t>(sn)];
}

int32_t max_legacy_scan_subindex_for_available_ships(int32_t ships) {
  TORCH_CHECK_DISABLED(ships >= 0, "ships");
  if (ships == 0) {
    return 0;
  }
  const auto first_larger = std::upper_bound(
      kLegacyScanShipCountBySubindex.begin() + 1, kLegacyScanShipCountBySubindex.end(),
      ships);
  return static_cast<int32_t>(
             first_larger - kLegacyScanShipCountBySubindex.begin()) -
         1;
}

bool planet_is_rotating_for_mask(int32_t planet_id, double x, double y, double radius,
                                 const SmallPlanetIdSet &comet_planet_ids) {
  if (comet_planet_ids.contains(planet_id)) {
    return false;
  }
  TORCH_CHECK_DISABLED(radius > 0.0, "radius");
  const float orbital_r = orbit_policy_orbit_radius_float32(x, y);
  const float rad = static_cast<float>(radius);
  const float lim = static_cast<float>(kRotationRadiusLimit);
  return orbital_r + rad < lim;
}

static void self_enemy_player_order(int32_t player_slot, int32_t num_players,
                                    int32_t out[kPlayerAxisSlots]) {
  TORCH_CHECK_DISABLED(0 <= player_slot && player_slot < kPlayerAxisSlots, "player_slot");
  TORCH_CHECK_DISABLED(num_players == 2 || num_players == 4, "num_players");
  if (num_players == 2) {
    TORCH_CHECK_DISABLED(player_slot < num_players, "2p player_slot < num_players");
    out[0] = player_slot;
    out[1] = 1 - player_slot;
    out[2] = -1;
    out[3] = -1;
    return;
  }
  TORCH_CHECK_DISABLED(player_slot < num_players, "4p player_slot < num_players");
  const int32_t self_pos = kPlayerPositionById4p[player_slot];
  for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
    const int32_t pos = (self_pos + player_block) % kPlayerAxisSlots;
    out[player_block] = kPlayerIdByPosition4p[pos];
  }
}

static int32_t fleet_ship_count_for_player(const std::vector<Planet> &planets,
                                           const std::vector<Fleet> &fleets,
                                           int32_t player_id) {
  int32_t n = 0;
  for (const Planet &p : planets) {
    if (p.owner == player_id) {
      n += static_cast<int32_t>(p.ships);
    }
  }
  for (const Fleet &f : fleets) {
    if (f.owner == player_id) {
      n += static_cast<int32_t>(f.ships);
    }
  }
  return n;
}

void fill_policy_obs_from_state(int32_t episode_step_scalar,
                                double angular_velocity,
                                const std::vector<std::vector<Planet>> &planets_by_seat,
                                const std::vector<std::vector<Fleet>> &fleets_by_seat,
                                int32_t num_agents,
                                const SmallPlanetIdSet &comet_planet_ids,
                                torch::Tensor orbit_planet_features,
                                torch::Tensor orbit_planet_mask,
                                torch::Tensor orbit_planet_pairwise_mask,
                                torch::Tensor orbit_planet_pairwise_features,
                                torch::Tensor action_taken_index,
                                torch::Tensor player_mask) {
  TORCH_CHECK_DISABLED(orbit_planet_features.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, kPlanetFeatures}));
  TORCH_CHECK_DISABLED(orbit_planet_mask.sizes() == torch::IntArrayRef({kPlayerAxisSlots, kPlanets}));
  TORCH_CHECK_DISABLED(orbit_planet_pairwise_mask.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPairwise}));
  TORCH_CHECK_DISABLED(orbit_planet_pairwise_features.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPairwise, kEdgeFeatures}));
  TORCH_CHECK_DISABLED(action_taken_index.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, 1}));
  TORCH_CHECK_DISABLED(player_mask.sizes() == torch::IntArrayRef({kPlayerAxisSlots}));
  assert_cpu_float(orbit_planet_features, "orbit_planet_features");
  assert_cpu_float(orbit_planet_mask, "orbit_planet_mask");
  assert_cpu_float(orbit_planet_pairwise_mask, "orbit_planet_pairwise_mask");
  assert_cpu_float(orbit_planet_pairwise_features, "orbit_planet_pairwise_features");
  TORCH_CHECK_DISABLED(action_taken_index.device().is_cpu(),
              "action_taken_index: expected CPU tensor");
  TORCH_CHECK_DISABLED(action_taken_index.dtype() == torch::kInt32,
              "action_taken_index: expected int32");
  assert_cpu_float(player_mask, "player_mask");
  TORCH_CHECK_DISABLED(static_cast<int32_t>(planets_by_seat.size()) == num_agents, "planets_by_seat");
  TORCH_CHECK_DISABLED(static_cast<int32_t>(fleets_by_seat.size()) == num_agents, "fleets_by_seat");
  TORCH_CHECK_DISABLED(episode_step_scalar >= 0 && episode_step_scalar <= kPlanetEpisodeStepMax,
              "episode_step_scalar");

  orbit_planet_features.zero_();
  orbit_planet_mask.zero_();
  orbit_planet_pairwise_mask.zero_();
  orbit_planet_pairwise_features.zero_();
  action_taken_index.zero_();
  player_mask.zero_();

  auto planet_features = orbit_planet_features.accessor<float, 3>();
  auto planet_mask = orbit_planet_mask.accessor<float, 2>();
  auto pairwise_mask = orbit_planet_pairwise_mask.accessor<float, 2>();
  auto edge_features = orbit_planet_pairwise_features.accessor<float, 3>();
  auto pmask = player_mask.accessor<float, 1>();

  int32_t num_players = 0;
  for (int32_t pidx = 0; pidx < num_agents; ++pidx) {
    const int32_t n = static_cast<int32_t>(planets_by_seat[pidx].size());
    TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, "planet_count");
    if (n > 0) {
      ++num_players;
    }
  }
  TORCH_CHECK_DISABLED(num_players == 2 || num_players == 4, "num_players");

  for (int32_t pidx = 0; pidx < num_players; ++pidx) {
    const int32_t policy_slot = policy_slot_for_compact_agent(pidx, num_players);
    const auto &planets = planets_by_seat[pidx];
    const auto &fleets = fleets_by_seat[pidx];
    const int32_t n = static_cast<int32_t>(planets.size());
    const int32_t geometry_pos = policy_geometry_position(pidx, num_players);
    float fleet_abs[kPlayerAxisSlots] = {};
    for (int32_t owner_id = 0; owner_id < kPlayerAxisSlots; ++owner_id) {
      fleet_abs[owner_id] =
          static_cast<float>(fleet_ship_count_for_player(planets, fleets, owner_id));
    }
    bool owns_planet = false;
    for (const Planet &planet : planets) {
      if (planet.owner == pidx) {
        owns_planet = true;
      }
    }
    const bool player_alive = owns_planet || fleet_abs[pidx] > 0.0f;
    pmask[policy_slot] = player_alive ? 1.0f : 0.0f;
    for (int32_t i = 0; i < n; ++i) {
      const Planet &planet = planets[i];
      const int32_t owner = planet.owner;
      const float ships = orbit_policy_ship_count_feature(planet.ships);
      const float production = static_cast<float>(planet.production);
      const bool neutral_owned = owner == -1;

      planet_mask[policy_slot][i] = 1.0f;
      const auto rel_xy =
          orbit_policy_player_relative_xy_float32(planet.x, planet.y, geometry_pos);
      planet_features[policy_slot][i][kPlanetBaseFeatureX] = rel_xy.first;
      planet_features[policy_slot][i][kPlanetBaseFeatureY] = rel_xy.second;
      planet_features[policy_slot][i][kPlanetBaseFeatureNeutralShips] = neutral_owned ? ships : 0.0f;
      planet_features[policy_slot][i][kPlanetBaseFeatureEpisodeStep] =
          static_cast<float>(episode_step_scalar);
      const int32_t pid = planet.id;
      const bool is_comet = comet_planet_ids.contains(pid);
      const bool rotates =
          planet_is_rotating_for_mask(pid, planet.x, planet.y, planet.radius, comet_planet_ids);
      planet_features[policy_slot][i][kPlanetBaseFeatureIsStatic] =
          (!is_comet && !rotates) ? 1.0f : 0.0f;
      planet_features[policy_slot][i][kPlanetBaseFeatureIsDynamic] =
          (!is_comet && rotates) ? 1.0f : 0.0f;
      planet_features[policy_slot][i][kPlanetBaseFeatureIsComet] = is_comet ? 1.0f : 0.0f;
      planet_features[policy_slot][i][kPlanetBaseFeatureCometTimeBeforeDespawn] =
          is_comet ? static_cast<float>(planet.comet_time_before_despawn)
                   : 0.0f;
      planet_features[policy_slot][i][kPlanetBaseFeatureRadius] = static_cast<float>(planet.radius);
      planet_features[policy_slot][i][kPlanetBaseFeaturePlanetProduction] = production;
      const float orbit_r = orbit_policy_orbit_radius_float32(planet.x, planet.y);
      planet_features[policy_slot][i][kPlanetBaseFeatureOrbitRadius] = orbit_r;
      planet_features[policy_slot][i][kPlanetBaseFeatureAngularVelocity] =
          (!is_comet && rotates) ? static_cast<float>(angular_velocity) : 0.0f;
      planet_features[policy_slot][i][kPlanetBaseFeatureSunAngle] =
          orbit_policy_sun_angle_float32(rel_xy.first, rel_xy.second);

      int32_t order[kPlayerAxisSlots];
      self_enemy_player_order(pidx, num_players, order);
      for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
        const int32_t owner_id = order[player_block];
        if (owner_id < 0) {
          continue;
        }
        const int32_t planet_base =
            kPlanetPlayerFeatureOffset + player_block * kPlanetPlayerFeaturesPerPlayer;
        const bool owned = owner == owner_id;
        planet_features[policy_slot][i][planet_base + kPlanetPlayerFeatureShips] =
            owned ? ships : 0.0f;
        planet_features[policy_slot][i][planet_base + kPlanetPlayerFeatureTotalFleetFrac] =
            owned ? fleet_abs[owner_id] / 1000.0f : 0.0f;
        planet_features[policy_slot][i][planet_base + kPlanetPlayerFeatureProduction] =
            owned ? production : 0.0f;
      }
    }

    int32_t order[kPlayerAxisSlots];
    self_enemy_player_order(pidx, num_players, order);
    for (int32_t src = 0; src < n; ++src) {
      const Planet &src_planet = planets[src];
      const int32_t src_owner = src_planet.owner;
      const bool src_neutral = src_owner == -1;
      const double src_x = src_planet.x;
      const double src_y = src_planet.y;
      for (int32_t dst = 0; dst < n; ++dst) {
        const Planet &dst_planet = planets[dst];
        const int32_t dst_owner = dst_planet.owner;
        const bool dst_neutral = dst_owner == -1;
        const double dst_x = dst_planet.x;
        const double dst_y = dst_planet.y;
        const int32_t eidx = src * kPlanets + dst;

        pairwise_mask[policy_slot][eidx] = 1.0f;
        edge_features[policy_slot][eidx][kEdgeBaseFeatureDistance] =
            orbit_wars_policy_obs_edge_distance(src_x, src_y, dst_x, dst_y);
        edge_features[policy_slot][eidx][kEdgeBaseFeatureSrcNeutral] =
            src_neutral ? 1.0f : 0.0f;
        edge_features[policy_slot][eidx][kEdgeBaseFeatureDstNeutral] =
            dst_neutral ? 1.0f : 0.0f;

        for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
          const int32_t owner_id = order[player_block];
          if (owner_id < 0) {
            continue;
          }
          const int32_t edge_base =
              kEdgePlayerFeatureOffset + player_block * kEdgePlayerFeaturesPerPlayer;
          edge_features[policy_slot][eidx][edge_base + kEdgePlayerFeatureSrcOwned] =
              src_owner == owner_id ? 1.0f : 0.0f;
          edge_features[policy_slot][eidx][edge_base + kEdgePlayerFeatureDstOwned] =
              dst_owner == owner_id ? 1.0f : 0.0f;
        }
      }
    }
  }
}

void fill_action_taken_index_from_classes(torch::Tensor action_classes, int32_t num_agents,
                                          torch::Tensor action_taken_index) {
  TORCH_CHECK_DISABLED(action_classes.device().is_cpu(),
              "action_classes: expected CPU tensor");
  TORCH_CHECK_DISABLED(action_classes.dtype() == torch::kInt32 ||
                  action_classes.dtype() == torch::kInt64,
              "action_classes: expected int32 or int64");
  action_classes = action_classes.to(torch::kInt32).contiguous();
  TORCH_CHECK_DISABLED(action_classes.sizes() == torch::IntArrayRef({num_agents, kPlanets}));
  TORCH_CHECK_DISABLED(action_taken_index.device().is_cpu(),
              "action_taken_index: expected CPU tensor");
  TORCH_CHECK_DISABLED(action_taken_index.dtype() == torch::kInt32,
              "action_taken_index: expected int32");
  TORCH_CHECK_DISABLED(action_taken_index.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, 1}));
  const auto classes = action_classes.accessor<int32_t, 2>();
  auto taken = action_taken_index.accessor<int32_t, 3>();
  for (int32_t seat = 0; seat < num_agents; ++seat) {
    const int32_t policy_slot = policy_slot_for_compact_agent(seat, num_agents);
    for (int32_t planet = 0; planet < kPlanets; ++planet) {
      const int32_t cls = classes[seat][planet];
      TORCH_CHECK_DISABLED(0 <= cls && cls < kMoveClasses, "action class");
      taken[policy_slot][planet][0] = cls;
    }
  }
}

void orbit_wars_fill_inactive_policy_action_noops(torch::Tensor available_action_mask,
                                                  torch::Tensor action_taken_index,
                                                  int32_t num_agents) {
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4, "num_agents");
  assert_cpu_int8(available_action_mask, "available_action_mask");
  TORCH_CHECK_DISABLED(action_taken_index.device().is_cpu(),
              "action_taken_index: expected CPU tensor");
  TORCH_CHECK_DISABLED(action_taken_index.dtype() == torch::kInt32,
              "action_taken_index: expected int32");
  TORCH_CHECK_DISABLED(available_action_mask.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, kMoveClasses}));
  TORCH_CHECK_DISABLED(action_taken_index.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, 1}));
  if (num_agents == 4) {
    return;
  }
  auto avail = available_action_mask.accessor<int8_t, 3>();
  auto taken = action_taken_index.accessor<int32_t, 3>();
  const int32_t inactive_slots[2] = {1, 2};
  for (int32_t si = 0; si < 2; ++si) {
    const int32_t slot = inactive_slots[si];
    for (int32_t src = 0; src < kPlanets; ++src) {
      for (int32_t cls = 0; cls < kMoveClasses; ++cls) {
        avail[slot][src][cls] = 0;
      }
    }
    for (int32_t src = 0; src < kPlanets; ++src) {
      const int32_t noop_class = src * kMoveClassesPerTarget;
      avail[slot][src][noop_class] = 1;
      taken[slot][src][0] = noop_class;
    }
  }
}





