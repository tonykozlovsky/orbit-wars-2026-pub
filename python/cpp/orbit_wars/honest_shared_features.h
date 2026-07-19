#pragma once

#include "common.h"
#include "honest_shared_intercept.h"

#include <vector>

class CppEnvStaticCacheV2;

namespace orbit_wars_honest {

struct ArrivalSurvivor {
  int32_t owner = -1;
  double ships = 0.0;
};

struct ArrivalSurvivorInt {
  int32_t owner = -1;
  int32_t ships = 0;
};

struct ArrivalResolutionStep {
  int32_t t = -1;
  int32_t slot = -1;
  int32_t pre_owner = -2;
  double pre_ships = 0.0;
  double arrivals[kPlayerAxisSlots] = {};
  ArrivalSurvivor survivor;
  int32_t post_owner = -2;
  double post_ships = 0.0;
};

struct StableTakeoverWork {
  int32_t horizon = 0;
  int32_t n = 0;
  std::vector<int32_t> pre_owners;
  std::vector<int32_t> pre_ships;
  std::vector<int32_t> required_after;
  std::array<int32_t, kPlanets> productions{};
};

void fill_future_resolution_planet_features_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    torch::Tensor orbit_planet_features);
void fill_future_resolution_edge_features_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_pairwise_features,
    const CppEnvStaticCacheV2 *profile_env);
torch::Tensor player_centric_temporal_planet_feature_cube_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const char *name, const CppEnvStaticCacheV2 *profile_env);
void fill_player_centric_temporal_planet_feature_cube_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const char *name, const CppEnvStaticCacheV2 *profile_env,
    torch::Tensor out);
torch::Tensor player_centric_temporal_planet_features_from_abs(
    torch::Tensor features_abs, int32_t num_agents, const char *name);
torch::Tensor takeover_cost_abs_features_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets,
    int32_t num_agents);
torch::Tensor takeover_cost_abs_int64_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets,
    int32_t num_agents);
StableTakeoverWork build_stable_takeover_work(torch::Tensor arrivals,
                                               const std::vector<Planet> &planets,
                                               int32_t num_agents,
                                               const char *name);
torch::Tensor stable_takeover_cost_abs_int64_from_work(
    torch::Tensor arrivals, const StableTakeoverWork &work, int32_t num_agents,
    const char *name);
torch::Tensor stable_takeover_cost_abs_int64_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets,
    int32_t num_agents);

}  // namespace orbit_wars_honest
