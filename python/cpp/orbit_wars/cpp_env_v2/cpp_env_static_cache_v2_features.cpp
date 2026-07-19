#include "cpp_env_static_cache_v2.h"

#include "../honest_shared_features.h"
#include "../honest_shared_intercept.h"
#include "../masks.h"

#include <algorithm>
#include <cmath>

namespace {

const NoopCachedPlanet *noop_row_at_episode_step(
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    int32_t episode_step) {
  TORCH_CHECK_DISABLED(noop_cached_planets_flat.size() % static_cast<uint32_t>(kPlanets) == 0,
              "static cache features: corrupt noop flat buffer");
  const int32_t n_frames =
      static_cast<int32_t>(noop_cached_planets_flat.size() / static_cast<uint32_t>(kPlanets));
  TORCH_CHECK_DISABLED(episode_step >= 0 && episode_step < n_frames,
              "static cache features: noop cache missing episode_step");
  return noop_cached_planets_flat.data() +
         static_cast<uint32_t>(episode_step) * static_cast<uint32_t>(kPlanets);
}

SmallPlanetIdSet active_comet_planet_ids_for_slots(
    const NoopCachedPlanet *planets_row,
    const std::array<int32_t, kPlanets> &planet_slot_comet,
    int32_t planet_slot_comet_n) {
  SmallPlanetIdSet out;
  for (int32_t i = 0; i < planet_slot_comet_n; ++i) {
    const int32_t slot = planet_slot_comet[static_cast<uint32_t>(i)];
    TORCH_CHECK_DISABLED(0 <= slot && slot < kPlanets,
                "static cache features: bad comet slot");
    const int32_t pid = planets_row[static_cast<uint32_t>(slot)].id;
    if (pid >= 0) {
      out.insert(pid);
    }
  }
  return out;
}

py::tuple resolve_fleet_arrivals_for_planets(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const char *context) {
  TORCH_CHECK_DISABLED(arrivals.device().is_cpu(), context, ": arrivals must be CPU");
  TORCH_CHECK_DISABLED(arrivals.dtype() == torch::kFloat32,
              context, ": arrivals must be float32");
  TORCH_CHECK_DISABLED(arrivals.dim() == 3, context, ": arrivals dim");
  const int32_t horizon = arrivals.size(0);
  TORCH_CHECK_DISABLED(horizon > 0, context, ": horizon must be positive");
  TORCH_CHECK_DISABLED(arrivals.size(1) == kPlanets, context, ": planet slots");
  TORCH_CHECK_DISABLED(arrivals.size(2) == kPlayerAxisSlots, context, ": player slots");
  TORCH_CHECK_DISABLED(0 < num_agents && num_agents <= kPlayerAxisSlots,
              context, ": num_agents");

  std::array<int32_t, kPlanets> owners{};
  std::array<double, kPlanets> ships{};
  std::array<double, kPlanets> productions{};
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    owners[static_cast<uint32_t>(slot)] = -2;
    ships[static_cast<uint32_t>(slot)] = 0.0;
    productions[static_cast<uint32_t>(slot)] = 0.0;
  }
  const int32_t n = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, context, ": planet count");
  for (int32_t slot = 0; slot < n; ++slot) {
    const Planet &p = planets[static_cast<uint32_t>(slot)];
    owners[static_cast<uint32_t>(slot)] = p.owner;
    ships[static_cast<uint32_t>(slot)] = p.ships;
    productions[static_cast<uint32_t>(slot)] = p.production;
  }

  torch::Tensor arrivals_c = arrivals.contiguous();
  const auto in = arrivals_c.accessor<float, 3>();
  torch::Tensor out_owners =
      torch::full({horizon, kPlanets}, -2,
                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  torch::Tensor out_ships =
      torch::zeros({horizon, kPlanets},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto owner_out = out_owners.accessor<int32_t, 2>();
  auto ships_out = out_ships.accessor<float, 2>();

  for (int32_t t = 0; t < horizon; ++t) {
    for (int32_t slot = 0; slot < n; ++slot) {
      const uint32_t idx = static_cast<uint32_t>(slot);
      if (owners[idx] != -1) {
        ships[idx] += productions[idx];
      }
    }
    for (int32_t slot = n; slot < kPlanets; ++slot) {
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        TORCH_CHECK_DISABLED(in[t][slot][owner] == 0.0f,
                    context, ": arrival on inactive planet slot");
      }
    }
    for (int32_t slot = 0; slot < n; ++slot) {
      std::vector<std::pair<int32_t, double>> player_ships_pairs;
      player_ships_pairs.reserve(static_cast<uint32_t>(num_agents));
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        const double arrival_ships = static_cast<double>(in[t][slot][owner]);
        if (arrival_ships > 0.0) {
          player_ships_pairs.push_back({owner, arrival_ships});
        }
      }
      if (!player_ships_pairs.empty()) {
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
          const uint32_t idx = static_cast<uint32_t>(slot);
          if (owners[idx] == survivor_owner) {
            ships[idx] += survivor_ships;
          } else {
            ships[idx] -= survivor_ships;
            if (ships[idx] < 0.0) {
              owners[idx] = survivor_owner;
              ships[idx] = std::abs(ships[idx]);
            }
          }
        }
      }
    }
    for (int32_t slot = 0; slot < n; ++slot) {
      const uint32_t idx = static_cast<uint32_t>(slot);
      owner_out[t][slot] = owners[idx];
      ships_out[t][slot] = static_cast<float>(ships[idx]);
    }
  }
  return py::make_tuple(out_owners, out_ships);
}

}  // namespace

void CppEnvStaticCacheV2::fill_policy_obs_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_mask, torch::Tensor orbit_planet_pairwise_mask,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor action_taken_index, torch::Tensor player_mask) const {
  const NoopCachedPlanet *planets_row =
      noop_row_at_episode_step(noop_cached_planets_flat_, episode_step);
  const SmallPlanetIdSet comet_planet_ids =
      active_comet_planet_ids_for_slots(planets_row, planet_slot_comet_,
                                        planet_slot_comet_n_);
  std::vector<Planet> planets_with_comet_lifetime = planets;
  for (Planet &planet : planets_with_comet_lifetime) {
    planet.comet_time_before_despawn = 0.0;
    if (!comet_planet_ids.contains(planet.id)) {
      continue;
    }
    bool found_comet_slot = false;
    for (int32_t i = 0; i < planet_slot_comet_n_; ++i) {
      const int32_t slot = planet_slot_comet_[static_cast<uint32_t>(i)];
      TORCH_CHECK_DISABLED(0 <= slot && slot < kPlanets,
                  "static cache features: bad comet slot");
      const NoopCachedPlanet &cached = planets_row[static_cast<uint32_t>(slot)];
      if (cached.id == planet.id) {
        planet.comet_time_before_despawn = cached.comet_time_before_despawn;
        found_comet_slot = true;
        break;
      }
    }
    TORCH_CHECK_DISABLED(found_comet_slot,
                "static cache features: missing active comet lifetime");
  }
  std::vector<std::vector<Planet>> planets_by_seat;
  std::vector<std::vector<Fleet>> fleets_by_seat;
  planets_by_seat.reserve(static_cast<uint32_t>(num_agents_));
  fleets_by_seat.reserve(static_cast<uint32_t>(num_agents_));
  for (int32_t seat = 0; seat < num_agents_; ++seat) {
    planets_by_seat.push_back(planets_with_comet_lifetime);
    fleets_by_seat.push_back(fleets);
  }
  fill_policy_obs_from_state(
      episode_step, angular_velocity_, planets_by_seat, fleets_by_seat,
      num_agents_, comet_planet_ids, orbit_planet_features, orbit_planet_mask,
      orbit_planet_pairwise_mask, orbit_planet_pairwise_features,
      action_taken_index, player_mask);
}

void CppEnvStaticCacheV2::fill_policy_obs_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_mask, torch::Tensor orbit_planet_pairwise_mask,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor action_taken_index, torch::Tensor player_mask) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count, "static cache fill_policy_obs_from_rows");
  fill_policy_obs_from_state_vectors(
      episode_step, fleets, planets, orbit_planet_features, orbit_planet_mask,
      orbit_planet_pairwise_mask, orbit_planet_pairwise_features,
      action_taken_index, player_mask);
}

torch::Tensor CppEnvStaticCacheV2::fleet_arrivals_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    int32_t horizon) const {
  if (wall_profile_enabled_) {
    wall_profile_clear();
  }
  WallProfileSpan profile(this, "fleet_arrivals_from_state_vectors");
  const NoopCachedPlanet *planets_row;
  {
    WallProfileSpan profile_noop(this, "noop_row");
    planets_row = noop_row_at_episode_step(noop_cached_planets_flat_, episode_step);
  }
  SmallPlanetIdSet comet_planet_ids;
  {
    WallProfileSpan profile_comets(this, "active_comet_ids");
    comet_planet_ids =
        active_comet_planet_ids_for_slots(planets_row, planet_slot_comet_,
                                          planet_slot_comet_n_);
  }
  {
    WallProfileSpan profile_arrivals(this, "fleet_arrivals_for_fleets");
    return orbit_wars_honest::fleet_arrivals_for_fleets(
        fleets, horizon, num_agents_, ship_speed_, episode_step, comet_planet_ids,
        noop_cached_planets_flat_, noop_spatial_grid_);
  }
}

torch::Tensor CppEnvStaticCacheV2::fleet_arrivals_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, int32_t horizon) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  return fleet_arrivals_from_state_vectors(episode_step, fleets, horizon);
}

torch::Tensor CppEnvStaticCacheV2::fleet_arrival_features_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, int32_t horizon) const {
  if (wall_profile_enabled_) {
    wall_profile_clear();
  }
  WallProfileSpan profile(this, "arrival_features_from_state_vectors");
  const NoopCachedPlanet *planets_row;
  {
    WallProfileSpan profile_noop(this, "noop_row");
    planets_row = noop_row_at_episode_step(noop_cached_planets_flat_, episode_step);
  }
  SmallPlanetIdSet comet_planet_ids;
  {
    WallProfileSpan profile_comets(this, "active_comet_ids");
    comet_planet_ids =
        active_comet_planet_ids_for_slots(planets_row, planet_slot_comet_,
                                          planet_slot_comet_n_);
  }
  torch::Tensor arrivals;
  {
    WallProfileSpan profile_arrivals(this, "fleet_arrivals_for_fleets");
    arrivals = orbit_wars_honest::fleet_arrivals_for_fleets(
        fleets, horizon, num_agents_, ship_speed_, episode_step, comet_planet_ids,
        noop_cached_planets_flat_, noop_spatial_grid_);
  }
  {
    WallProfileSpan profile_temporal_cube(this, "player_centric_temporal_planet_feature_cube");
    return orbit_wars_honest::player_centric_temporal_planet_feature_cube_from_arrivals(
        arrivals, planets, num_agents_, "static cache arrival features from rows",
        this);
  }
}

torch::Tensor CppEnvStaticCacheV2::fleet_arrival_features_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, int32_t horizon) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count,
          "static cache fleet_arrival_features_from_rows");
  return fleet_arrival_features_from_state_vectors(
      episode_step, fleets, planets, horizon);
}

void CppEnvStaticCacheV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, int32_t horizon,
    torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_arrival_features) const {
  if (wall_profile_enabled_) {
    wall_profile_clear();
  }
  WallProfileSpan profile(this, "arrival_features_with_resolution_from_state_vectors");
  const NoopCachedPlanet *planets_row;
  {
    WallProfileSpan profile_noop(this, "noop_row");
    planets_row = noop_row_at_episode_step(noop_cached_planets_flat_, episode_step);
  }
  SmallPlanetIdSet comet_planet_ids;
  {
    WallProfileSpan profile_comets(this, "active_comet_ids");
    comet_planet_ids =
        active_comet_planet_ids_for_slots(planets_row, planet_slot_comet_,
                                          planet_slot_comet_n_);
  }
  torch::Tensor arrivals;
  {
    WallProfileSpan profile_arrivals(this, "fleet_arrivals_for_fleets");
    arrivals = orbit_wars_honest::fleet_arrivals_for_fleets(
        fleets, horizon, num_agents_, ship_speed_, episode_step, comet_planet_ids,
        noop_cached_planets_flat_, noop_spatial_grid_);
  }
  {
    WallProfileSpan profile_planet_features(this, "fill_future_resolution_planet_features");
    orbit_wars_honest::fill_future_resolution_planet_features_from_arrivals(
        arrivals, planets, num_agents_, orbit_planet_features);
  }
  {
    WallProfileSpan profile_edge_features(this, "fill_future_resolution_edge_features");
    orbit_wars_honest::fill_future_resolution_edge_features_from_arrivals(
        arrivals, planets, num_agents_, ship_speed_, episode_step,
        comet_planet_ids, noop_cached_planets_flat_, noop_spatial_grid_,
        available_hit_mask,
        orbit_planet_pairwise_features, this);
  }
  {
    WallProfileSpan profile_temporal_cube(this, "player_centric_temporal_planet_feature_cube");
    orbit_wars_honest::fill_player_centric_temporal_planet_feature_cube_from_arrivals(
        arrivals, planets, num_agents_,
        "static cache arrival features with resolution planet features from rows",
        this, orbit_planet_arrival_features);
  }
}

void CppEnvStaticCacheV2::fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, int32_t horizon,
    torch::Tensor orbit_planet_features,
    torch::Tensor orbit_planet_pairwise_features,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_arrival_features) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count,
          "static cache fleet_arrival_features_and_fill_future_resolution_planet_features_from_rows");
  fleet_arrival_features_and_fill_future_resolution_planet_features_from_state_vectors(
      episode_step, fleets, planets, horizon, orbit_planet_features,
      orbit_planet_pairwise_features, available_hit_mask,
      orbit_planet_arrival_features);
}

py::tuple CppEnvStaticCacheV2::fleet_arrivals_resolution_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, int32_t horizon) const {
  const NoopCachedPlanet *planets_row =
      noop_row_at_episode_step(noop_cached_planets_flat_, episode_step);
  const SmallPlanetIdSet comet_planet_ids =
      active_comet_planet_ids_for_slots(planets_row, planet_slot_comet_,
                                        planet_slot_comet_n_);
  torch::Tensor arrivals = orbit_wars_honest::fleet_arrivals_for_fleets(
      fleets, horizon, num_agents_, ship_speed_, episode_step, comet_planet_ids,
      noop_cached_planets_flat_, noop_spatial_grid_);
  return resolve_fleet_arrivals_for_planets(
      arrivals, planets, num_agents_,
      "static cache fleet_arrivals_resolution_from_rows");
}

py::tuple CppEnvStaticCacheV2::fleet_arrivals_resolution_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, int32_t horizon) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count,
          "static cache fleet_arrivals_resolution_from_rows");
  return fleet_arrivals_resolution_from_state_vectors(
      episode_step, fleets, planets, horizon);
}

void CppEnvStaticCacheV2::fill_future_resolution_planet_features_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, int32_t horizon,
    torch::Tensor orbit_planet_features) const {
  torch::Tensor arrivals =
      fleet_arrivals_from_state_vectors(episode_step, fleets, horizon);
  orbit_wars_honest::fill_future_resolution_planet_features_from_arrivals(
      arrivals, planets, num_agents_, orbit_planet_features);
}

void CppEnvStaticCacheV2::fill_future_resolution_planet_features_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, torch::Tensor planet_rows,
    int32_t planet_count, int32_t horizon,
    torch::Tensor orbit_planet_features) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count,
          "static cache fill_future_resolution_planet_features_from_rows");
  fill_future_resolution_planet_features_from_state_vectors(
      episode_step, fleets, planets, horizon, orbit_planet_features);
}

torch::Tensor CppEnvStaticCacheV2::fleet_takeover_cost_features_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    const std::vector<Planet> &planets, int32_t horizon) const {
  torch::Tensor arrivals =
      fleet_arrivals_from_state_vectors(episode_step, fleets, horizon).contiguous();
  torch::Tensor abs_features =
      orbit_wars_honest::takeover_cost_abs_features_from_arrivals(
          arrivals, planets, num_agents_);
  return orbit_wars_honest::player_centric_temporal_planet_features_from_abs(
      abs_features, num_agents_, "static cache takeover features from rows");
}

py::tuple CppEnvStaticCacheV2::fleet_arrivals_resolution_from_arrivals_and_planets(
    torch::Tensor arrivals, const std::vector<Planet> &planets) const {
  return resolve_fleet_arrivals_for_planets(
      arrivals, planets, num_agents_,
      "static cache fleet_arrivals_resolution_from_arrivals_and_rows");
}

py::tuple CppEnvStaticCacheV2::fleet_arrivals_resolution_from_arrivals_and_rows(
    torch::Tensor arrivals, torch::Tensor planet_rows,
    int32_t planet_count) const {
  const std::vector<Planet> planets =
      orbit_wars_honest::external_planet_rows_tensor_to_vector(
          planet_rows, planet_count,
          "static cache fleet_arrivals_resolution_from_arrivals_and_rows");
  return fleet_arrivals_resolution_from_arrivals_and_planets(arrivals, planets);
}

py::list CppEnvStaticCacheV2::fleet_hit_traces_from_state_vectors(
    int32_t episode_step, const std::vector<Fleet> &fleets,
    int32_t horizon) const {
  const NoopCachedPlanet *planets_row =
      noop_row_at_episode_step(noop_cached_planets_flat_, episode_step);
  const SmallPlanetIdSet comet_planet_ids =
      active_comet_planet_ids_for_slots(planets_row, planet_slot_comet_,
                                        planet_slot_comet_n_);
  return orbit_wars_honest::fleet_hit_traces_for_fleets(
      fleets, horizon, num_agents_, ship_speed_, episode_step, comet_planet_ids,
      noop_cached_planets_flat_, noop_spatial_grid_);
}

py::list CppEnvStaticCacheV2::fleet_hit_traces_from_rows(
    int32_t episode_step, torch::Tensor fleet_rows, int32_t horizon) const {
  const std::vector<Fleet> fleets =
      orbit_wars_honest::fleet_rows_tensor_to_vector(fleet_rows);
  return fleet_hit_traces_from_state_vectors(episode_step, fleets, horizon);
}
