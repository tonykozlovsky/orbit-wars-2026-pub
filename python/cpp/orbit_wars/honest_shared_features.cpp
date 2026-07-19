#include "honest_shared_features.h"
#include "honest_shared_intercept.h"

#include "cpp_env_v2/cpp_env_static_cache_v2.h"
#include "library.h"
#include "masks.h"
#include "simulation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>


namespace orbit_wars_honest {

void self_enemy_player_order_for_features(int32_t player_slot, int32_t num_agents,
                                          int32_t out[kPlayerAxisSlots]) {
  TORCH_CHECK_DISABLED(0 <= player_slot && player_slot < kPlayerAxisSlots,
              "takeover features: player_slot");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4, "takeover features: num_agents");
  for (int32_t i = 0; i < kPlayerAxisSlots; ++i) {
    out[i] = -1;
  }
  if (num_agents == 2) {
    if (player_slot == 0) {
      out[0] = 0;
      out[1] = 1;
    } else if (player_slot == 3) {
      out[0] = 1;
      out[1] = 0;
    }
    return;
  }
  const int32_t self_pos = kPlayerPositionById4p[player_slot];
  for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
    const int32_t pos = (self_pos + player_block) % kPlayerAxisSlots;
    out[player_block] = kPlayerIdByPosition4p[pos];
  }
}

int32_t policy_geometry_position_for_compact_player(int32_t player_slot,
                                                    int32_t num_agents) {
  TORCH_CHECK_DISABLED(0 <= player_slot && player_slot < kPlayerAxisSlots,
              "policy geometry: player_slot");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4,
              "policy geometry: num_agents");
  if (num_agents == 2) {
    TORCH_CHECK_DISABLED(player_slot < num_agents, "policy geometry: 2p compact slot");
    return player_slot == 0 ? 0 : 2;
  }
  TORCH_CHECK_DISABLED(player_slot < num_agents, "policy geometry: 4p compact slot");
  const int32_t pos = kPlayerPositionById4p[player_slot];
  TORCH_CHECK_DISABLED(0 <= pos && pos < kPlayerAxisSlots,
              "policy geometry: 4p position");
  return pos;
}

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

std::pair<float, float> policy_geometry_rotate_vector_to_player_frame(
    float dx, float dy, int32_t player_position) {
  switch (player_position) {
    case 0:
      return {dx, dy};
    case 1:
      return {dy, -dx};
    case 2:
      return {-dx, -dy};
    case 3:
      return {-dy, dx};
  }
  TORCH_CHECK_DISABLED(false, "policy geometry: player_position");
}

ArrivalSurvivor arrival_survivor_from_ships(const double arrivals[kPlayerAxisSlots],
                                            int32_t num_agents) {
  double top = 0.0;
  double second = 0.0;
  int32_t top_owner = -1;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    const double ships = arrivals[owner];
    TORCH_CHECK_DISABLED(ships >= 0.0 && std::isfinite(ships),
                "takeover features: bad arrival ships");
    if (ships > top) {
      second = top;
      top = ships;
      top_owner = owner;
    } else if (ships > second) {
      second = ships;
    }
  }
  if (top <= 0.0 || top == second) {
    return ArrivalSurvivor{};
  }
  return ArrivalSurvivor{top_owner, top - second};
}

ArrivalSurvivorInt arrival_survivor_from_int_ships(
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents) {
  int32_t top = 0;
  int32_t second = 0;
  int32_t top_owner = -1;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    const int32_t ships = arrivals[owner];
    TORCH_CHECK_DISABLED(ships >= 0, "stable takeover features: bad arrival ships");
    if (ships > top) {
      second = top;
      top = ships;
      top_owner = owner;
    } else if (ships > second) {
      second = ships;
    }
  }
  if (top <= 0 || top == second) {
    return ArrivalSurvivorInt{};
  }
  return ArrivalSurvivorInt{top_owner, top - second};
}

void apply_arrival_survivor(int32_t &owner, double &ships,
                            const ArrivalSurvivor &survivor) {
  TORCH_CHECK_DISABLED(owner >= -1 && owner < kPlayerAxisSlots,
              "takeover features: bad planet owner");
  TORCH_CHECK_DISABLED(ships >= 0.0 && std::isfinite(ships),
              "takeover features: bad planet ships");
  TORCH_CHECK_DISABLED(survivor.ships >= 0.0 && std::isfinite(survivor.ships),
              "takeover features: bad survivor ships");
  if (survivor.ships <= 0.0) {
    return;
  }
  TORCH_CHECK_DISABLED(0 <= survivor.owner && survivor.owner < kPlayerAxisSlots,
              "takeover features: bad survivor owner");
  if (owner == survivor.owner) {
    ships += survivor.ships;
    return;
  }
  ships -= survivor.ships;
  if (ships < 0.0) {
    owner = survivor.owner;
    ships = std::abs(ships);
  }
}

int32_t min_int_strictly_greater_than(double value) {
  TORCH_CHECK_DISABLED(std::isfinite(value), "takeover features: non-finite strict threshold");
  if (value < 0.0) {
    return 0;
  }
  TORCH_CHECK_DISABLED(value < static_cast<double>(std::numeric_limits<int32_t>::max() - 1),
              "takeover features: strict threshold too large");
  return static_cast<int32_t>(std::floor(value)) + 1;
}

int32_t min_int_greater_or_equal(double value) {
  TORCH_CHECK_DISABLED(std::isfinite(value), "takeover features: non-finite threshold");
  if (value <= 0.0) {
    return 0;
  }
  TORCH_CHECK_DISABLED(value <= static_cast<double>(std::numeric_limits<int32_t>::max()),
              "takeover features: threshold too large");
  return static_cast<int32_t>(std::ceil(value));
}

int32_t exact_int64_ship_count(double value, const char *name) {
  TORCH_CHECK_DISABLED(std::isfinite(value), name, ": non-finite ship count");
  TORCH_CHECK_DISABLED(value >= 0.0, name, ": negative ship count");
  const double rounded = std::round(value);
  TORCH_CHECK_DISABLED(std::abs(value - rounded) <= 1e-6, name, ": non-integer ship count");
  TORCH_CHECK_DISABLED(rounded <= static_cast<double>(std::numeric_limits<int32_t>::max()),
              name, ": ship count too large");
  return static_cast<int32_t>(rounded);
}

int32_t takeover_cost_ships(int32_t pre_owner, double pre_ships,
                            const double arrivals[kPlayerAxisSlots], int32_t num_agents,
                            int32_t player) {
  TORCH_CHECK_DISABLED(0 <= player && player < num_agents, "takeover features: player");
  int32_t no_add_owner = pre_owner;
  double no_add_ships = pre_ships;
  apply_arrival_survivor(no_add_owner, no_add_ships,
                         arrival_survivor_from_ships(arrivals, num_agents));
  if (no_add_owner == player) {
    return 0;
  }

  const double player_arrivals = arrivals[player];
  double other_max = 0.0;
  double other_second = 0.0;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    if (owner == player) {
      continue;
    }
    const double ships = arrivals[owner];
    if (ships > other_max) {
      other_second = other_max;
      other_max = ships;
    } else if (ships > other_second) {
      other_second = ships;
    }
  }

  double capture_threshold = other_max - player_arrivals;
  if (pre_owner != player) {
    capture_threshold += pre_ships;
  }
  int32_t best = min_int_strictly_greater_than(capture_threshold);

  if (pre_owner == player) {
    const double defend_threshold = other_max - pre_ships - player_arrivals;
    if (other_second >= other_max - pre_ships) {
      best = 0;
    } else {
      best = std::min(best, min_int_greater_or_equal(defend_threshold));
    }
  }
  TORCH_CHECK_DISABLED(best >= 0, "takeover features: negative cost");
  return best;
}

int32_t stable_takeover_cost_ships(int32_t pre_owner, int32_t pre_ships,
                                   const int32_t arrivals[kPlayerAxisSlots],
                                   int32_t num_agents, int32_t player,
                                   int32_t required_post_ships) {
  TORCH_CHECK_DISABLED(0 <= player && player < num_agents, "stable takeover features: player");
  TORCH_CHECK_DISABLED(required_post_ships >= 0, "stable takeover features: bad required ships");
  int32_t no_add_owner = pre_owner;
  int32_t no_add_ships = pre_ships;
  const ArrivalSurvivorInt no_add_survivor =
      arrival_survivor_from_int_ships(arrivals, num_agents);
  if (no_add_survivor.ships > 0) {
    if (no_add_owner == no_add_survivor.owner) {
      no_add_ships += no_add_survivor.ships;
    } else {
      no_add_ships -= no_add_survivor.ships;
      if (no_add_ships < 0) {
        no_add_owner = no_add_survivor.owner;
        no_add_ships = std::abs(no_add_ships);
      }
    }
  }
  if (no_add_owner == player && no_add_ships >= required_post_ships) {
    return 0;
  }

  const int32_t player_arrivals = arrivals[player];
  int32_t other_max = 0;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    if (owner == player) {
      continue;
    }
    other_max = std::max(other_max, arrivals[owner]);
  }

  int32_t cost = 0;
  if (pre_owner == player) {
    cost = std::max<int32_t>(
        0, other_max + required_post_ships - pre_ships - player_arrivals);
  } else if (required_post_ships > 0) {
    cost = std::max<int32_t>(
        0, other_max + pre_ships + required_post_ships - player_arrivals);
  } else {
    const int32_t threshold = other_max + pre_ships - player_arrivals;
    cost = threshold < 0 ? 0 : threshold + 1;
  }
  TORCH_CHECK_DISABLED(cost >= 0, "stable takeover features: negative cost");
  return cost;
}

int32_t raw_ship_cost_int64(int32_t cost, const char *name) {
  TORCH_CHECK_DISABLED(cost >= 0, name, ": negative ship cost");
  return cost;
}

float raw_ship_cost_feature(int32_t cost, const char *name) {
  const int32_t raw = raw_ship_cost_int64(cost, name);
  TORCH_CHECK_DISABLED(raw >= 0, name, ": bad raw ship cost");
  return static_cast<float>(raw);
}

float raw_signed_ship_margin_feature(double ships) {
  TORCH_CHECK_DISABLED(std::isfinite(ships), "resolution features: non-finite ship margin");
  return static_cast<float>(ships);
}

void assert_abs_temporal_player_planet_features(torch::Tensor features_abs,
                                                int32_t horizon, const char *name) {
  TORCH_CHECK_DISABLED(features_abs.device().is_cpu(), name, ": expected CPU tensor");
  TORCH_CHECK_DISABLED(features_abs.dtype() == torch::kFloat32, name, ": expected float32");
  TORCH_CHECK_DISABLED(features_abs.sizes() ==
                  torch::IntArrayRef({horizon, kPlanets, kPlayerAxisSlots}),
              name, ": expected [horizon, planets, player_axis]");
}

template <typename Fn>
void for_each_arrival_resolution_step(torch::Tensor arrivals,
                                      const std::vector<Planet> &planets,
                                      int32_t num_agents, const char *name, Fn fn) {
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4, name, ": num_agents");
  const int32_t horizon = arrivals.size(0);
  assert_abs_temporal_player_planet_features(arrivals, horizon, name);
  const int32_t n = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, name, ": planet count");
  torch::Tensor arrivals_c = arrivals.contiguous();
  auto in = arrivals_c.accessor<float, 3>();

  std::array<int32_t, kPlanets> owners{};
  std::array<double, kPlanets> ships{};
  std::array<double, kPlanets> productions{};
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    owners[static_cast<uint32_t>(slot)] = -2;
    ships[static_cast<uint32_t>(slot)] = 0.0;
    productions[static_cast<uint32_t>(slot)] = 0.0;
  }
  for (int32_t slot = 0; slot < n; ++slot) {
    const Planet &p = planets[static_cast<uint32_t>(slot)];
    TORCH_CHECK_DISABLED(p.owner >= -1 && p.owner < kPlayerAxisSlots,
                name, ": bad planet owner");
    TORCH_CHECK_DISABLED(p.ships >= 0.0 && std::isfinite(p.ships),
                name, ": bad planet ships");
    TORCH_CHECK_DISABLED(p.production >= 0.0 && std::isfinite(p.production),
                name, ": bad planet production");
    owners[static_cast<uint32_t>(slot)] = p.owner;
    ships[static_cast<uint32_t>(slot)] = p.ships;
    productions[static_cast<uint32_t>(slot)] = p.production;
  }

  for (int32_t t = 0; t < horizon; ++t) {
    for (int32_t slot = 0; slot < n; ++slot) {
      const uint32_t idx = static_cast<uint32_t>(slot);
      if (owners[idx] != -1) {
        ships[idx] += productions[idx];
        TORCH_CHECK_DISABLED(std::isfinite(ships[idx]), name, ": non-finite ships");
      }
    }
    for (int32_t slot = n; slot < kPlanets; ++slot) {
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        TORCH_CHECK_DISABLED(in[t][slot][owner] == 0.0f,
                    name, ": arrival on inactive planet slot");
      }
    }
    for (int32_t slot = 0; slot < n; ++slot) {
      const uint32_t idx = static_cast<uint32_t>(slot);
      ArrivalResolutionStep step;
      step.t = t;
      step.slot = slot;
      step.pre_owner = owners[idx];
      step.pre_ships = ships[idx];
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        step.arrivals[owner] = static_cast<double>(in[t][slot][owner]);
        TORCH_CHECK_DISABLED(step.arrivals[owner] >= 0.0 && std::isfinite(step.arrivals[owner]),
                    name, ": bad step arrival");
      }
      step.survivor = arrival_survivor_from_ships(step.arrivals, num_agents);
      step.post_owner = step.pre_owner;
      step.post_ships = step.pre_ships;
      apply_arrival_survivor(step.post_owner, step.post_ships, step.survivor);
      TORCH_CHECK_DISABLED(step.post_owner >= -1 && step.post_owner < kPlayerAxisSlots,
                  name, ": bad post owner");
      TORCH_CHECK_DISABLED(step.post_ships >= 0.0 && std::isfinite(step.post_ships),
                  name, ": bad post ships");
      fn(step);
      owners[idx] = step.post_owner;
      ships[idx] = step.post_ships;
    }
  }
}

uint32_t temporal_planet_idx(int32_t t, int32_t slot) {
  return static_cast<uint32_t>(t * kPlanets + slot);
}

uint32_t temporal_planet_owner_idx(int32_t t, int32_t slot, int32_t owner) {
  return static_cast<uint32_t>((t * kPlanets + slot) * kPlayerAxisSlots + owner);
}

StableTakeoverWork build_stable_takeover_work(torch::Tensor arrivals,
                                               const std::vector<Planet> &planets,
                                               int32_t num_agents, const char *name);

torch::Tensor stable_takeover_cost_abs_int64_from_work(torch::Tensor arrivals,
                                                       const StableTakeoverWork &work,
                                                       int32_t num_agents, const char *name);

void compute_extra_temporal_player_features_abs(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const StableTakeoverWork &work, torch::Tensor stable_takeover_raw,
    torch::Tensor neutralization_raw, torch::Tensor deny_stable_enemy_raw,
    torch::Tensor battle_tie_distance_raw, torch::Tensor production_swing_per_ship_abs,
    torch::Tensor arrival_leverage_abs, const CppEnvStaticCacheV2 *profile_env);

void post_state_after_added_arrival_int(
    int32_t pre_owner, int32_t pre_ships,
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents, int32_t player,
    int32_t added_ships, int32_t &post_owner, int32_t &post_ships);
int32_t neutralization_cost_ships(int32_t pre_owner, int32_t pre_ships,
                                  const int32_t arrivals[kPlayerAxisSlots],
                                  int32_t num_agents, int32_t player);
int32_t battle_tie_distance_ships(const int32_t arrivals[kPlayerAxisSlots],
                                  int32_t num_agents, int32_t player);
float local_player_owner_margin_feature(int32_t owner, int32_t ships,
                                        int32_t num_agents, int32_t player);
int32_t deny_stable_enemy_cost_ships(
    int32_t pre_owner, int32_t pre_ships,
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents, int32_t player,
    const StableTakeoverWork &work, int32_t t, int32_t slot);

void self_enemy_player_order_for_compact_planet_features(
    int32_t player_slot, int32_t num_agents, int32_t out[kPlayerAxisSlots]) {
  TORCH_CHECK_DISABLED(0 <= player_slot && player_slot < num_agents,
              "resolution planet features: player_slot");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4,
              "resolution planet features: num_agents");
  for (int32_t i = 0; i < kPlayerAxisSlots; ++i) {
    out[i] = -1;
  }
  if (num_agents == 2) {
    out[0] = player_slot;
    out[1] = 1 - player_slot;
    return;
  }
  const int32_t self_pos = kPlayerPositionById4p[player_slot];
  for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
    const int32_t pos = (self_pos + player_block) % kPlayerAxisSlots;
    out[player_block] = kPlayerIdByPosition4p[pos];
  }
}

torch::Tensor player_centric_temporal_planet_features_from_abs(
    torch::Tensor features_abs, int32_t num_agents, const char *name) {
  const int32_t horizon = features_abs.size(0);
  TORCH_CHECK_DISABLED(horizon > 0, name, ": horizon must be positive");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4, name, ": num_agents");
  torch::Tensor features_abs_c = features_abs.contiguous();
  assert_abs_temporal_player_planet_features(features_abs_c, horizon, name);

  torch::Tensor out =
      torch::zeros({kPlayerAxisSlots, kPlanets, horizon, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto in = features_abs_c.accessor<float, 3>();
  auto out_features = out.accessor<float, 4>();

  int32_t orders[kPlayerAxisSlots][kPlayerAxisSlots];
  for (int32_t pidx = 0; pidx < kPlayerAxisSlots; ++pidx) {
    self_enemy_player_order_for_features(pidx, num_agents, orders[pidx]);
  }
  for (int32_t t = 0; t < horizon; ++t) {
    for (int32_t slot = 0; slot < kPlanets; ++slot) {
      for (int32_t owner = 0; owner < kPlayerAxisSlots; ++owner) {
        const float v = in[t][slot][owner];
        TORCH_CHECK_DISABLED(std::isfinite(static_cast<double>(v)), name, ": non-finite feature");
        if (owner >= num_agents) {
          TORCH_CHECK_DISABLED(v == 0.0f, name, ": inactive owner feature must be zero");
        }
      }
    }
  }
  for (int32_t pidx = 0; pidx < kPlayerAxisSlots; ++pidx) {
    for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
      const int32_t owner_id = orders[pidx][player_block];
      if (owner_id < 0) {
        continue;
      }
      for (int32_t slot = 0; slot < kPlanets; ++slot) {
        for (int32_t t = 0; t < horizon; ++t) {
          out_features[pidx][slot][t][player_block] = in[t][slot][owner_id];
        }
      }
    }
  }
  return out;
}

void fill_player_centric_temporal_planet_feature_cube_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const char *name, const CppEnvStaticCacheV2 *profile_env,
    torch::Tensor out) {
  const int32_t horizon = arrivals.size(0);
  TORCH_CHECK_DISABLED(horizon > 0, name, ": horizon must be positive");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4, name, ": num_agents");
  TORCH_CHECK_DISABLED(out.device().is_cpu(), name, ": out must be CPU");
  TORCH_CHECK_DISABLED(out.dtype() == torch::kFloat32, name, ": out dtype");
  TORCH_CHECK_DISABLED(out.is_contiguous(), name, ": out must be contiguous");
  TORCH_CHECK_DISABLED(out.sizes() ==
                  torch::IntArrayRef({kPlayerAxisSlots, kPlanets, horizon,
                                      kPlayerAxisSlots,
                                      kTemporalPlanetFeatures}),
              name, ": out shape");
  torch::Tensor arrivals_c;
  {
    CppEnvStaticCacheV2::WallProfileSpan profile(profile_env, "cube_contiguous");
    arrivals_c = arrivals.contiguous();
    assert_abs_temporal_player_planet_features(arrivals_c, horizon, name);
  }

  auto features = out.accessor<float, 5>();
  StableTakeoverWork stable_work;
  {
    CppEnvStaticCacheV2::WallProfileSpan profile(profile_env, "cube_build_stable_takeover_work");
    stable_work = build_stable_takeover_work(
        arrivals_c, planets, num_agents, "extra temporal feature work");
  }
  torch::Tensor stable_takeover_raw;
  {
    CppEnvStaticCacheV2::WallProfileSpan profile(profile_env, "cube_stable_takeover_cost");
    stable_takeover_raw = stable_takeover_cost_abs_int64_from_work(
        arrivals_c, stable_work, num_agents, "extra temporal stable takeover");
  }
  auto stable_takeover = stable_takeover_raw.accessor<int32_t, 3>();

  int32_t orders[kPlayerAxisSlots][kPlayerAxisSlots];
  for (int32_t pidx = 0; pidx < kPlayerAxisSlots; ++pidx) {
    self_enemy_player_order_for_features(pidx, num_agents, orders[pidx]);
  }

  CppEnvStaticCacheV2::WallProfileSpan fill_profile(profile_env, "cube_fill_player_centric_features");
  for_each_arrival_resolution_step(
      arrivals_c, planets, num_agents, name,
      [&](const ArrivalResolutionStep &step) {
        const float time_step = static_cast<float>(step.t + 1);
        TORCH_CHECK_DISABLED(time_step >= 1.0f && time_step <= static_cast<float>(horizon),
                    name, ": bad temporal time step");
        double arrival_by_owner[kPlayerAxisSlots] = {};
        float takeover_by_owner[kPlayerAxisSlots] = {};
        float hold_by_owner[kPlayerAxisSlots] = {};
        float hold_valid_by_owner[kPlayerAxisSlots] = {};
        float neutralization_by_owner[kPlayerAxisSlots] = {};
        float neutralization_valid_by_owner[kPlayerAxisSlots] = {};
        float battle_tie_by_owner[kPlayerAxisSlots] = {};
        float battle_tie_valid_by_owner[kPlayerAxisSlots] = {};
        float deny_stable_enemy_by_owner[kPlayerAxisSlots] = {};
        float production_swing_by_owner[kPlayerAxisSlots] = {};
        float leverage_by_owner[kPlayerAxisSlots] = {};
        int32_t step_arrivals_int[kPlayerAxisSlots] = {};
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          step_arrivals_int[owner] = exact_int64_ship_count(
              step.arrivals[owner], "extra temporal arrival ships");
        }
        const uint32_t state_idx = temporal_planet_idx(step.t, step.slot);
        const int32_t pre_owner_int = stable_work.pre_owners[state_idx];
        const int32_t pre_ships_int = stable_work.pre_ships[state_idx];
        int32_t no_add_owner = -2;
        int32_t no_add_ships = 0;
        post_state_after_added_arrival_int(
            pre_owner_int, pre_ships_int, step_arrivals_int, num_agents, 0, 0,
            no_add_owner, no_add_ships);
        const double future_production_steps =
            static_cast<double>(horizon - step.t - 1);
        const double production =
            static_cast<double>(stable_work.productions[static_cast<uint32_t>(step.slot)]);
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          const double arrival_ships = step.arrivals[owner];
          TORCH_CHECK_DISABLED(arrival_ships >= 0.0 && std::isfinite(arrival_ships),
                      name, ": bad arrival feature");
          arrival_by_owner[owner] = arrival_ships;
          const int32_t takeover_cost = takeover_cost_ships(
              step.pre_owner, step.pre_ships, step.arrivals, num_agents, owner);
          takeover_by_owner[owner] =
              raw_ship_cost_feature(takeover_cost, "temporal takeover cost");
          if (step.post_owner == owner) {
            hold_by_owner[owner] = raw_ship_cost_feature(
                stable_takeover[step.t][step.slot][owner],
                "temporal hold cost");
            hold_valid_by_owner[owner] = 1.0f;
          } else {
            hold_by_owner[owner] = 0.0f;
            hold_valid_by_owner[owner] = 0.0f;
          }
          const int32_t neutralization_cost =
              neutralization_cost_ships(
                  pre_owner_int, pre_ships_int, step_arrivals_int, num_agents,
                  owner);
          TORCH_CHECK_DISABLED(neutralization_cost >= 0,
                      "temporal neutralization cost: negative cost");
          neutralization_by_owner[owner] =
              static_cast<float>(neutralization_cost);
          neutralization_valid_by_owner[owner] = 1.0f;
          const int32_t battle_tie_distance =
              battle_tie_distance_ships(
                  step_arrivals_int, num_agents, owner);
          TORCH_CHECK_DISABLED(battle_tie_distance >= 0,
                      "temporal battle tie distance: negative cost");
          battle_tie_by_owner[owner] = static_cast<float>(battle_tie_distance);
          battle_tie_valid_by_owner[owner] = 1.0f;
          deny_stable_enemy_by_owner[owner] =
              raw_ship_cost_feature(
                  deny_stable_enemy_cost_ships(
                      pre_owner_int, pre_ships_int, step_arrivals_int, num_agents,
                      owner, stable_work, step.t, step.slot),
                  "temporal deny stable enemy cost");
          const int32_t stable_cost = stable_takeover[step.t][step.slot][owner];
          const double denom =
              static_cast<double>(std::max<int32_t>(stable_cost, 1));
          const double swing =
              production * future_production_steps / denom;
          TORCH_CHECK_DISABLED(std::isfinite(swing),
                      "production swing per ship: non-finite");
          production_swing_by_owner[owner] = static_cast<float>(swing);
          int32_t plus_one_owner = -2;
          int32_t plus_one_ships = 0;
          post_state_after_added_arrival_int(
              pre_owner_int, pre_ships_int, step_arrivals_int, num_agents,
              owner, 1, plus_one_owner, plus_one_ships);
          const float base_margin = local_player_owner_margin_feature(
              no_add_owner, no_add_ships, num_agents, owner);
          const float plus_one_margin = local_player_owner_margin_feature(
              plus_one_owner, plus_one_ships, num_agents, owner);
          const double delta =
              std::abs(static_cast<double>(plus_one_margin) -
                       static_cast<double>(base_margin));
          leverage_by_owner[owner] = static_cast<float>(delta);
        }
        for (int32_t pidx = 0; pidx < kPlayerAxisSlots; ++pidx) {
          for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
            const int32_t owner_id = orders[pidx][player_block];
            if (owner_id < 0) {
              continue;
            }
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureArrivalShips] =
                static_cast<float>(arrival_by_owner[owner_id]);
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureTakeoverCost] =
                takeover_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureResolutionOwner] =
                step.post_owner == owner_id ? 1.0f : 0.0f;
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureResolutionShips] =
                step.post_owner == owner_id ? static_cast<float>(step.post_ships) : 0.0f;
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureTimeStep] = time_step;
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureStableTakeoverCost] =
                raw_ship_cost_feature(
                    stable_takeover[step.t][step.slot][owner_id],
                    "temporal stable takeover cost");
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureHoldCost] =
                hold_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureHoldValid] =
                hold_valid_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureNeutralizationCost] =
                neutralization_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureNeutralizationValid] =
                neutralization_valid_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureDenyStableEnemyCost] =
                deny_stable_enemy_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureBattleTieDistance] =
                battle_tie_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureBattleTieValid] =
                battle_tie_valid_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureProductionSwingPerShip] =
                production_swing_by_owner[owner_id];
            features[pidx][step.slot][step.t][player_block]
                    [kTemporalPlanetFeatureArrivalLeverage] =
                leverage_by_owner[owner_id];
          }
        }
      });
}

torch::Tensor player_centric_temporal_planet_feature_cube_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const char *name, const CppEnvStaticCacheV2 *profile_env) {
  const int32_t horizon = arrivals.size(0);
  torch::Tensor out;
  {
    CppEnvStaticCacheV2::WallProfileSpan profile(profile_env, "cube_alloc");
    out = torch::zeros({kPlayerAxisSlots, kPlanets, horizon, kPlayerAxisSlots,
                        kTemporalPlanetFeatures},
                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  }
  fill_player_centric_temporal_planet_feature_cube_from_arrivals(
      arrivals, planets, num_agents, name, profile_env, out);
  return out;
}

torch::Tensor takeover_cost_abs_features_from_arrivals(torch::Tensor arrivals,
                                                       const std::vector<Planet> &planets,
                                                       int32_t num_agents) {
  const int32_t horizon = arrivals.size(0);
  torch::Tensor out =
      torch::zeros({horizon, kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto abs_features = out.accessor<float, 3>();

  for_each_arrival_resolution_step(
      arrivals, planets, num_agents, "takeover abs features",
      [&](const ArrivalResolutionStep &step) {
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          const int32_t cost = takeover_cost_ships(
              step.pre_owner, step.pre_ships, step.arrivals, num_agents, owner);
          abs_features[step.t][step.slot][owner] =
              raw_ship_cost_feature(cost, "takeover abs features");
        }
      });
  return out;
}

torch::Tensor takeover_cost_abs_int64_from_arrivals(torch::Tensor arrivals,
                                                    const std::vector<Planet> &planets,
                                                    int32_t num_agents) {
  const int32_t horizon = arrivals.size(0);
  torch::Tensor out =
      torch::zeros({horizon, kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  auto abs_features = out.accessor<int32_t, 3>();

  for_each_arrival_resolution_step(
      arrivals, planets, num_agents, "takeover raw features",
      [&](const ArrivalResolutionStep &step) {
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          abs_features[step.t][step.slot][owner] = takeover_cost_ships(
              step.pre_owner, step.pre_ships, step.arrivals, num_agents, owner);
        }
      });
  return out;
}

StableTakeoverWork build_stable_takeover_work(torch::Tensor arrivals,
                                               const std::vector<Planet> &planets,
                                               int32_t num_agents,
                                               const char *name) {
  const int32_t horizon = arrivals.size(0);
  TORCH_CHECK_DISABLED(horizon > 0, name, ": horizon must be positive");
  torch::Tensor arrivals_c = arrivals.contiguous();
  assert_abs_temporal_player_planet_features(arrivals_c, horizon, name);
  const int32_t n = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, name, ": planet count");

  StableTakeoverWork work;
  work.horizon = horizon;
  work.n = n;
  work.pre_owners.assign(static_cast<uint32_t>(horizon * kPlanets), -2);
  work.pre_ships.assign(static_cast<uint32_t>(horizon * kPlanets), 0);
  work.required_after.assign(
      static_cast<uint32_t>(horizon * kPlanets * kPlayerAxisSlots), 0);

  for_each_arrival_resolution_step(
      arrivals_c, planets, num_agents, name,
      [&](const ArrivalResolutionStep &step) {
        const uint32_t idx = temporal_planet_idx(step.t, step.slot);
        work.pre_owners[idx] = step.pre_owner;
        work.pre_ships[idx] = exact_int64_ship_count(step.pre_ships, name);
      });

  for (int32_t slot = 0; slot < n; ++slot) {
    const Planet &p = planets[static_cast<uint32_t>(slot)];
    work.productions[static_cast<uint32_t>(slot)] =
        exact_int64_ship_count(p.production, name);
  }

  auto in = arrivals_c.accessor<float, 3>();
  for (int32_t t = horizon - 2; t >= 0; --t) {
    const int32_t next_t = t + 1;
    for (int32_t slot = 0; slot < n; ++slot) {
      int32_t step_arrivals[kPlayerAxisSlots] = {};
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        step_arrivals[owner] = exact_int64_ship_count(
            static_cast<double>(in[next_t][slot][owner]), name);
      }
      const ArrivalSurvivorInt survivor =
          arrival_survivor_from_int_ships(step_arrivals, num_agents);
      const int32_t production = work.productions[static_cast<uint32_t>(slot)];
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        int32_t required =
            work.required_after[temporal_planet_owner_idx(next_t, slot, owner)] -
            production;
        if (survivor.owner == owner) {
          required -= survivor.ships;
        } else if (survivor.owner >= 0) {
          required += survivor.ships;
        }
        required = std::max<int32_t>(0, required);
        work.required_after[temporal_planet_owner_idx(t, slot, owner)] = required;
      }
    }
  }

  return work;
}

torch::Tensor stable_takeover_cost_abs_int64_from_work(
    torch::Tensor arrivals, const StableTakeoverWork &work, int32_t num_agents,
    const char *name) {
  const int32_t horizon = arrivals.size(0);
  TORCH_CHECK_DISABLED(horizon == work.horizon, name, ": horizon mismatch");
  torch::Tensor arrivals_c = arrivals.contiguous();
  assert_abs_temporal_player_planet_features(arrivals_c, horizon, name);

  torch::Tensor out =
      torch::zeros({horizon, kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

  auto costs = out.accessor<int32_t, 3>();
  auto in = arrivals_c.accessor<float, 3>();
  for (int32_t t = 0; t < horizon; ++t) {
    for (int32_t slot = 0; slot < work.n; ++slot) {
      int32_t step_arrivals[kPlayerAxisSlots] = {};
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        step_arrivals[owner] = exact_int64_ship_count(
            static_cast<double>(in[t][slot][owner]),
            "stable takeover arrival ships");
      }
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        const uint32_t idx = temporal_planet_idx(t, slot);
        costs[t][slot][owner] = stable_takeover_cost_ships(
            work.pre_owners[idx], work.pre_ships[idx], step_arrivals,
            num_agents, owner,
            work.required_after[temporal_planet_owner_idx(t, slot, owner)]);
      }
    }
  }
  return out;
}

torch::Tensor stable_takeover_cost_abs_int64_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents) {
  torch::Tensor arrivals_c = arrivals.contiguous();
  StableTakeoverWork work = build_stable_takeover_work(
      arrivals_c, planets, num_agents, "stable takeover features");
  return stable_takeover_cost_abs_int64_from_work(
      arrivals_c, work, num_agents, "stable takeover features");
}

void post_state_after_added_arrival_int(
    int32_t pre_owner, int32_t pre_ships,
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents, int32_t player,
    int32_t added_ships, int32_t &post_owner, int32_t &post_ships) {
  TORCH_CHECK_DISABLED(0 <= player && player < num_agents, "extra temporal features: player");
  TORCH_CHECK_DISABLED(added_ships >= 0, "extra temporal features: negative added ships");
  int32_t modified[kPlayerAxisSlots] = {};
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    modified[owner] = arrivals[owner];
  }
  modified[player] += added_ships;
  TORCH_CHECK_DISABLED(modified[player] >= arrivals[player],
              "extra temporal features: added ships overflow");
  post_owner = pre_owner;
  post_ships = pre_ships;
  const ArrivalSurvivorInt survivor =
      arrival_survivor_from_int_ships(modified, num_agents);
  if (survivor.ships <= 0) {
    return;
  }
  if (post_owner == survivor.owner) {
    post_ships += survivor.ships;
  } else {
    post_ships -= survivor.ships;
    if (post_ships < 0) {
      post_owner = survivor.owner;
      post_ships = std::abs(post_ships);
    }
  }
  TORCH_CHECK_DISABLED(post_owner >= -1 && post_owner < kPlayerAxisSlots,
              "extra temporal features: bad post owner");
  TORCH_CHECK_DISABLED(post_ships >= 0, "extra temporal features: bad post ships");
}

void add_temporal_cost_candidate(std::vector<int32_t> &candidates, int32_t value) {
  const int32_t cap = static_cast<int32_t>(kFleetNormalizer);
  for (int32_t delta = -1; delta <= 1; ++delta) {
    const int32_t v = value + delta;
    if (0 <= v && v <= cap) {
      candidates.push_back(v);
    }
  }
}

template <typename Fn>
int32_t min_added_ship_cost_from_candidates(std::vector<int32_t> candidates,
                                            Fn accept) {
  const int32_t cap = static_cast<int32_t>(kFleetNormalizer);
  candidates.push_back(0);
  candidates.push_back(cap);
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  for (int32_t added : candidates) {
    TORCH_CHECK_DISABLED(0 <= added && added <= cap,
                "extra temporal features: candidate out of range");
    if (accept(added)) {
      return added;
    }
  }
  return cap;
}

void add_arrival_threshold_candidates(std::vector<int32_t> &candidates,
                                      const int32_t arrivals[kPlayerAxisSlots],
                                      int32_t num_agents, int32_t player,
                                      int32_t pre_ships,
                                      int32_t required_ships) {
  const int32_t player_arrivals = arrivals[player];
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    if (owner == player) {
      continue;
    }
    const int32_t other = arrivals[owner];
    add_temporal_cost_candidate(candidates, other - player_arrivals);
    add_temporal_cost_candidate(candidates, other + pre_ships - player_arrivals);
    add_temporal_cost_candidate(candidates,
                                other + pre_ships + required_ships -
                                    player_arrivals);
  }
}

void add_neutralization_threshold_candidates(
    std::vector<int32_t> &candidates,
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents,
    int32_t player, int32_t pre_ships) {
  const int32_t player_arrivals = arrivals[player];
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    if (owner == player) {
      continue;
    }
    const int32_t other = arrivals[owner];
    add_temporal_cost_candidate(candidates,
                                other - pre_ships - player_arrivals);
    add_temporal_cost_candidate(candidates, other - player_arrivals);
    add_temporal_cost_candidate(candidates,
                                other + pre_ships - player_arrivals);
  }
}

void add_deny_stable_enemy_threshold_candidates(
    std::vector<int32_t> &candidates,
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents,
    int32_t player, int32_t pre_owner, int32_t pre_ships,
    int32_t enemy, int32_t required_ships) {
  const int32_t player_arrivals = arrivals[player];
  int32_t other_max = 0;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    if (owner == player) {
      continue;
    }
    const int32_t other = arrivals[owner];
    other_max = std::max(other_max, other);
    add_temporal_cost_candidate(candidates, other - player_arrivals);
  }

  add_temporal_cost_candidate(candidates,
                              other_max + pre_ships + 1 - player_arrivals);

  const int32_t enemy_arrivals = arrivals[enemy];
  if (pre_owner == enemy) {
    add_temporal_cost_candidate(
        candidates,
        enemy_arrivals + pre_ships - required_ships + 1 - player_arrivals);
  } else {
    add_temporal_cost_candidate(
        candidates,
        enemy_arrivals - pre_ships - required_ships + 1 - player_arrivals);
  }
}

int32_t neutralization_cost_ships(int32_t pre_owner, int32_t pre_ships,
                                 const int32_t arrivals[kPlayerAxisSlots],
                                 int32_t num_agents, int32_t player) {
  int32_t no_add_owner = -2;
  int32_t no_add_ships = 0;
  post_state_after_added_arrival_int(pre_owner, pre_ships, arrivals, num_agents,
                                     player, 0, no_add_owner, no_add_ships);
  if (no_add_owner == -1) {
    return 0;
  }
  if (pre_owner != -1) {
    return static_cast<int32_t>(kFleetNormalizer);
  }
  std::vector<int32_t> candidates;
  add_neutralization_threshold_candidates(candidates, arrivals, num_agents,
                                          player, pre_ships);
  add_arrival_threshold_candidates(candidates, arrivals, num_agents, player,
                                   pre_ships, 0);
  return min_added_ship_cost_from_candidates(
      candidates, [&](int32_t added) {
        int32_t post_owner = -2;
        int32_t post_ships = 0;
        post_state_after_added_arrival_int(pre_owner, pre_ships, arrivals,
                                           num_agents, player, added,
                                           post_owner, post_ships);
        return post_owner == -1;
      });
}

int32_t battle_tie_distance_ships(const int32_t arrivals[kPlayerAxisSlots],
                                  int32_t num_agents, int32_t player) {
  int32_t top = 0;
  int32_t top_count = 0;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    const int32_t ships = arrivals[owner];
    if (ships > top) {
      top = ships;
      top_count = 1;
    } else if (ships == top) {
      top_count += 1;
    }
  }
  if (top <= 0) {
    return static_cast<int32_t>(kFleetNormalizer);
  }
  if (arrivals[player] == top && top_count >= 2) {
    return 0;
  }
  if (arrivals[player] < top) {
    return raw_ship_cost_int64(top - arrivals[player],
                               "battle tie distance");
  }
  return static_cast<int32_t>(kFleetNormalizer);
}

float snipe_window_score_feature(
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents, int32_t player,
    int32_t takeover_cost, int32_t post_owner) {
  if (post_owner == player) {
    return 0.0f;
  }
  int32_t enemy_top = 0;
  int32_t enemy_second = 0;
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    if (owner == player) {
      continue;
    }
    const int32_t ships = arrivals[owner];
    if (ships > enemy_top) {
      enemy_second = enemy_top;
      enemy_top = ships;
    } else if (ships > enemy_second) {
      enemy_second = ships;
    }
  }
  if (enemy_second <= 0) {
    return 0.0f;
  }
  TORCH_CHECK_DISABLED(takeover_cost >= 0, "snipe window score: negative takeover cost");
  const int32_t score = std::max<int32_t>(0, enemy_second - takeover_cost);
  return raw_ship_cost_feature(score, "snipe window score");
}

float local_player_owner_margin_feature(int32_t owner, int32_t ships,
                                        int32_t num_agents, int32_t player) {
  TORCH_CHECK_DISABLED(ships >= 0, "arrival leverage: bad ships");
  if (owner == player) {
    return raw_signed_ship_margin_feature(static_cast<double>(ships));
  }
  double zero_arrivals[kPlayerAxisSlots] = {};
  const int32_t cost = takeover_cost_ships(
      owner, static_cast<double>(ships), zero_arrivals, num_agents, player);
  return -raw_ship_cost_feature(cost, "arrival leverage margin");
}

int32_t deny_stable_enemy_cost_ships(
    int32_t pre_owner, int32_t pre_ships,
    const int32_t arrivals[kPlayerAxisSlots], int32_t num_agents, int32_t player,
    const StableTakeoverWork &work, int32_t t, int32_t slot) {
  auto enemy_stable_after = [&](int32_t added) {
    int32_t post_owner = -2;
    int32_t post_ships = 0;
    post_state_after_added_arrival_int(pre_owner, pre_ships, arrivals,
                                       num_agents, player, added, post_owner,
                                       post_ships);
    for (int32_t enemy = 0; enemy < num_agents; ++enemy) {
      if (enemy == player) {
        continue;
      }
      const int32_t required =
          work.required_after[temporal_planet_owner_idx(t, slot, enemy)];
      if (post_owner == enemy && post_ships >= required) {
        return true;
      }
    }
    return false;
  };
  if (!enemy_stable_after(0)) {
    return 0;
  }

  std::vector<int32_t> candidates;
  for (int32_t enemy = 0; enemy < num_agents; ++enemy) {
    if (enemy == player) {
      continue;
    }
    const int32_t required =
        work.required_after[temporal_planet_owner_idx(t, slot, enemy)];
    add_deny_stable_enemy_threshold_candidates(
        candidates, arrivals, num_agents, player, pre_owner, pre_ships,
        enemy, required);
    add_arrival_threshold_candidates(candidates, arrivals, num_agents, player,
                                     pre_ships, required);
  }
  return min_added_ship_cost_from_candidates(
      candidates, [&](int32_t added) { return !enemy_stable_after(added); });
}

void compute_extra_temporal_player_features_abs(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    const StableTakeoverWork &work, torch::Tensor stable_takeover_raw,
    torch::Tensor neutralization_raw, torch::Tensor deny_stable_enemy_raw,
    torch::Tensor battle_tie_distance_raw, torch::Tensor production_swing_per_ship_abs,
    torch::Tensor arrival_leverage_abs, const CppEnvStaticCacheV2 *profile_env) {
  const int32_t horizon = arrivals.size(0);
  torch::Tensor arrivals_c = arrivals.contiguous();
  assert_abs_temporal_player_planet_features(arrivals_c, horizon,
                                             "extra temporal features arrivals");
  auto in = arrivals_c.accessor<float, 3>();
  auto stable_takeover = stable_takeover_raw.accessor<int32_t, 3>();
  auto neutralization = neutralization_raw.accessor<int32_t, 3>();
  auto deny_stable_enemy = deny_stable_enemy_raw.accessor<int32_t, 3>();
  auto battle_tie = battle_tie_distance_raw.accessor<int32_t, 3>();
  auto production_swing = production_swing_per_ship_abs.accessor<float, 3>();
  auto leverage = arrival_leverage_abs.accessor<float, 3>();

  CppEnvStaticCacheV2::WallProfileSpan profile(profile_env, "extra_temporal_t_slot_owner_loop");
  for (int32_t t = 0; t < horizon; ++t) {
    const double future_production_steps =
        static_cast<double>(horizon - t - 1);
    for (int32_t slot = 0; slot < work.n; ++slot) {
      int32_t step_arrivals[kPlayerAxisSlots] = {};
      double step_arrivals_double[kPlayerAxisSlots] = {};
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        step_arrivals[owner] = exact_int64_ship_count(
            static_cast<double>(in[t][slot][owner]),
            "extra temporal arrival ships");
        step_arrivals_double[owner] =
            static_cast<double>(step_arrivals[owner]);
      }
      const uint32_t state_idx = temporal_planet_idx(t, slot);
      const int32_t pre_owner = work.pre_owners[state_idx];
      const int32_t pre_ships = work.pre_ships[state_idx];
      int32_t no_add_owner = -2;
      int32_t no_add_ships = 0;
      post_state_after_added_arrival_int(pre_owner, pre_ships, step_arrivals,
                                         num_agents, 0, 0, no_add_owner,
                                         no_add_ships);
      const double production =
          static_cast<double>(work.productions[static_cast<uint32_t>(slot)]);
      for (int32_t owner = 0; owner < num_agents; ++owner) {
        neutralization[t][slot][owner] = neutralization_cost_ships(
            pre_owner, pre_ships, step_arrivals, num_agents, owner);
        deny_stable_enemy[t][slot][owner] = deny_stable_enemy_cost_ships(
            pre_owner, pre_ships, step_arrivals, num_agents, owner, work, t,
            slot);
        battle_tie[t][slot][owner] =
            battle_tie_distance_ships(step_arrivals, num_agents, owner);
        const int32_t stable_cost = stable_takeover[t][slot][owner];
        const double denom =
            static_cast<double>(std::max<int32_t>(stable_cost, 1));
        const double swing =
            production * future_production_steps / denom;
        TORCH_CHECK_DISABLED(std::isfinite(swing),
                    "production swing per ship: non-finite");
        production_swing[t][slot][owner] = static_cast<float>(swing);

        int32_t plus_one_owner = -2;
        int32_t plus_one_ships = 0;
        post_state_after_added_arrival_int(pre_owner, pre_ships, step_arrivals,
                                           num_agents, owner, 1,
                                           plus_one_owner, plus_one_ships);
        const float base_margin = local_player_owner_margin_feature(
            no_add_owner, no_add_ships, num_agents, owner);
        const float plus_one_margin = local_player_owner_margin_feature(
            plus_one_owner, plus_one_ships, num_agents, owner);
        const double delta =
            std::abs(static_cast<double>(plus_one_margin) -
                     static_cast<double>(base_margin));
        leverage[t][slot][owner] = static_cast<float>(delta);
      }
    }
  }
}

void compute_future_resolution_player_scalar_features_abs(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    torch::Tensor owner_survival_margin_abs, torch::Tensor flip_time_abs,
    torch::Tensor stable_flip_time_abs, torch::Tensor owner_churn_abs,
    torch::Tensor last_decisive_battle_step_abs,
    torch::Tensor post_horizon_owner_margin_abs) {
  const int32_t horizon = arrivals.size(0);
  TORCH_CHECK_DISABLED(horizon > 0, "resolution scalar features: horizon must be positive");
  torch::Tensor arrivals_c = arrivals.contiguous();
  assert_abs_temporal_player_planet_features(arrivals_c, horizon,
                                             "resolution scalar features arrivals");
  TORCH_CHECK_DISABLED(owner_survival_margin_abs.device().is_cpu(),
              "resolution scalar features: survival expected CPU");
  TORCH_CHECK_DISABLED(owner_survival_margin_abs.dtype() == torch::kFloat32,
              "resolution scalar features: survival expected float32");
  TORCH_CHECK_DISABLED(owner_survival_margin_abs.sizes() ==
                  torch::IntArrayRef({kPlanets, kPlayerAxisSlots}),
              "resolution scalar features: survival shape");
  TORCH_CHECK_DISABLED(flip_time_abs.device().is_cpu(),
              "resolution scalar features: flip expected CPU");
  TORCH_CHECK_DISABLED(flip_time_abs.dtype() == torch::kFloat32,
              "resolution scalar features: flip expected float32");
  TORCH_CHECK_DISABLED(flip_time_abs.sizes() ==
                  torch::IntArrayRef({kPlanets, kPlayerAxisSlots}),
              "resolution scalar features: flip shape");
  TORCH_CHECK_DISABLED(stable_flip_time_abs.device().is_cpu(),
              "resolution scalar features: stable flip expected CPU");
  TORCH_CHECK_DISABLED(stable_flip_time_abs.dtype() == torch::kFloat32,
              "resolution scalar features: stable flip expected float32");
  TORCH_CHECK_DISABLED(stable_flip_time_abs.sizes() ==
                  torch::IntArrayRef({kPlanets, kPlayerAxisSlots}),
              "resolution scalar features: stable flip shape");
  TORCH_CHECK_DISABLED(owner_churn_abs.device().is_cpu(),
              "resolution scalar features: churn expected CPU");
  TORCH_CHECK_DISABLED(owner_churn_abs.dtype() == torch::kFloat32,
              "resolution scalar features: churn expected float32");
  TORCH_CHECK_DISABLED(owner_churn_abs.sizes() ==
                  torch::IntArrayRef({kPlanets, kPlayerAxisSlots}),
              "resolution scalar features: churn shape");
  TORCH_CHECK_DISABLED(last_decisive_battle_step_abs.device().is_cpu(),
              "resolution scalar features: last decisive expected CPU");
  TORCH_CHECK_DISABLED(last_decisive_battle_step_abs.dtype() == torch::kFloat32,
              "resolution scalar features: last decisive expected float32");
  TORCH_CHECK_DISABLED(last_decisive_battle_step_abs.sizes() ==
                  torch::IntArrayRef({kPlanets, kPlayerAxisSlots}),
              "resolution scalar features: last decisive shape");
  TORCH_CHECK_DISABLED(post_horizon_owner_margin_abs.device().is_cpu(),
              "resolution scalar features: post margin expected CPU");
  TORCH_CHECK_DISABLED(post_horizon_owner_margin_abs.dtype() == torch::kFloat32,
              "resolution scalar features: post margin expected float32");
  TORCH_CHECK_DISABLED(post_horizon_owner_margin_abs.sizes() ==
                  torch::IntArrayRef({kPlanets, kPlayerAxisSlots}),
              "resolution scalar features: post margin shape");
  TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4,
              "resolution scalar features: num_agents");
  const int32_t n = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, "resolution scalar features: planet count");

  owner_survival_margin_abs.zero_();
  flip_time_abs.zero_();
  stable_flip_time_abs.zero_();
  owner_churn_abs.zero_();
  last_decisive_battle_step_abs.zero_();
  post_horizon_owner_margin_abs.zero_();
  auto survival = owner_survival_margin_abs.accessor<float, 2>();
  auto flip = flip_time_abs.accessor<float, 2>();
  auto stable_flip = stable_flip_time_abs.accessor<float, 2>();
  auto churn = owner_churn_abs.accessor<float, 2>();
  auto last_decisive = last_decisive_battle_step_abs.accessor<float, 2>();
  auto post_margin = post_horizon_owner_margin_abs.accessor<float, 2>();

  std::array<int32_t, kPlanets> current_owners{};
  std::array<double, kPlanets> min_owned_ships{};
  std::array<uint8_t, kPlanets> current_owner_lost{};
  std::vector<int32_t> post_owners(static_cast<uint32_t>(horizon * kPlanets), -2);
  std::vector<double> post_ships(static_cast<uint32_t>(horizon * kPlanets), 0.0);
  auto post_idx = [](int32_t t, int32_t slot) {
    return static_cast<uint32_t>(t * kPlanets + slot);
  };
  for (int32_t slot = 0; slot < kPlanets; ++slot) {
    current_owners[static_cast<uint32_t>(slot)] = -2;
    min_owned_ships[static_cast<uint32_t>(slot)] =
        std::numeric_limits<double>::infinity();
    current_owner_lost[static_cast<uint32_t>(slot)] = 0;
  }
  for (int32_t slot = 0; slot < n; ++slot) {
    const Planet &p = planets[static_cast<uint32_t>(slot)];
    TORCH_CHECK_DISABLED(p.owner >= -1 && p.owner < kPlayerAxisSlots,
                "resolution scalar features: bad current owner");
    TORCH_CHECK_DISABLED(p.owner < num_agents,
                "resolution scalar features: current owner outside active players");
    current_owners[static_cast<uint32_t>(slot)] = p.owner;
    for (int32_t owner = 0; owner < num_agents; ++owner) {
      flip[slot][owner] = static_cast<float>(horizon + 1);
      stable_flip[slot][owner] = static_cast<float>(horizon + 1);
    }
    if (p.owner >= 0) {
      flip[slot][p.owner] = 0.0f;
    }
  }

  for_each_arrival_resolution_step(
      arrivals_c, planets, num_agents, "resolution scalar features states",
      [&](const ArrivalResolutionStep &step) {
        const int32_t post_owner = step.post_owner;
        post_owners[post_idx(step.t, step.slot)] = post_owner;
        post_ships[post_idx(step.t, step.slot)] = step.post_ships;
        if (post_owner >= 0 &&
            flip[step.slot][post_owner] == static_cast<float>(horizon + 1)) {
          flip[step.slot][post_owner] = static_cast<float>(step.t + 1);
        }
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          const bool pre_owned = step.pre_owner == owner;
          const bool post_owned = post_owner == owner;
          if (pre_owned != post_owned) {
            churn[step.slot][owner] += 1.0f;
            last_decisive[step.slot][owner] =
                static_cast<float>(step.t + 1);
          }
        }

        const int32_t current_owner =
            current_owners[static_cast<uint32_t>(step.slot)];
        if (current_owner >= 0) {
          if (post_owner == current_owner) {
            double &min_ships =
                min_owned_ships[static_cast<uint32_t>(step.slot)];
            min_ships = std::min(min_ships, step.post_ships);
          } else {
            current_owner_lost[static_cast<uint32_t>(step.slot)] = 1;
          }
        }
      });

  torch::Tensor stable_takeover_raw =
      stable_takeover_cost_abs_int64_from_arrivals(arrivals_c, planets, num_agents);
  auto stable_takeover = stable_takeover_raw.accessor<int32_t, 3>();
  for (int32_t slot = 0; slot < n; ++slot) {
    const int32_t current_owner = current_owners[static_cast<uint32_t>(slot)];
    if (current_owner < 0) {
      continue;
    }
    const int32_t hold_cost_now = stable_takeover[0][slot][current_owner];
    if (hold_cost_now > 0) {
      survival[slot][current_owner] =
          -raw_ship_cost_feature(hold_cost_now, "owner survival hold cost");
      continue;
    }
    TORCH_CHECK_DISABLED(current_owner_lost[static_cast<uint32_t>(slot)] == 0,
                "resolution scalar features: zero hold cost but current owner lost");
    const double min_ships = min_owned_ships[static_cast<uint32_t>(slot)];
    TORCH_CHECK_DISABLED(std::isfinite(min_ships),
                "resolution scalar features: missing survival ship margin");
    survival[slot][current_owner] = raw_signed_ship_margin_feature(min_ships);
  }

  double zero_arrivals[kPlayerAxisSlots] = {};
  for (int32_t slot = 0; slot < n; ++slot) {
    int32_t suffix_owner = -2;
    for (int32_t t = horizon - 1; t >= 0; --t) {
      const int32_t owner = post_owners[post_idx(t, slot)];
      if (t == horizon - 1) {
        suffix_owner = owner;
      } else if (suffix_owner != owner) {
        suffix_owner = -2;
      }
      if (suffix_owner >= 0) {
        stable_flip[slot][suffix_owner] = static_cast<float>(t + 1);
      }
    }
    const int32_t current_owner = current_owners[static_cast<uint32_t>(slot)];
    if (current_owner >= 0 && suffix_owner == current_owner) {
      stable_flip[slot][current_owner] = 0.0f;
    }

    const int32_t final_owner = post_owners[post_idx(horizon - 1, slot)];
    const double final_ships = post_ships[post_idx(horizon - 1, slot)];
    TORCH_CHECK_DISABLED(final_owner >= -1 && final_owner < kPlayerAxisSlots,
                "resolution scalar features: bad final owner");
    TORCH_CHECK_DISABLED(final_ships >= 0.0 && std::isfinite(final_ships),
                "resolution scalar features: bad final ships");
    for (int32_t owner = 0; owner < num_agents; ++owner) {
      if (final_owner == owner) {
        post_margin[slot][owner] = raw_signed_ship_margin_feature(final_ships);
      } else {
        const int32_t cost =
            takeover_cost_ships(final_owner, final_ships, zero_arrivals,
                                num_agents, owner);
        post_margin[slot][owner] =
            -raw_ship_cost_feature(cost, "post horizon owner margin");
      }
      TORCH_CHECK_DISABLED(0.0f <= churn[slot][owner] &&
                      churn[slot][owner] <= static_cast<float>(horizon),
                  "resolution scalar features: churn out of range");
      TORCH_CHECK_DISABLED(0.0f <= last_decisive[slot][owner] &&
                      last_decisive[slot][owner] <= static_cast<float>(horizon),
                  "resolution scalar features: last decisive out of range");
      TORCH_CHECK_DISABLED(0.0f <= stable_flip[slot][owner] &&
                      stable_flip[slot][owner] <= static_cast<float>(horizon + 1),
                  "resolution scalar features: stable flip out of range");
    }
  }
}

void fill_future_resolution_planet_features_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    torch::Tensor orbit_planet_features) {
  TORCH_CHECK_DISABLED(orbit_planet_features.device().is_cpu(),
              "resolution planet features: expected CPU tensor");
  TORCH_CHECK_DISABLED(orbit_planet_features.dtype() == torch::kFloat32,
              "resolution planet features: expected float32 tensor");
  TORCH_CHECK_DISABLED(orbit_planet_features.is_contiguous(),
              "resolution planet features: expected contiguous tensor");
  TORCH_CHECK_DISABLED(orbit_planet_features.sizes() ==
                  torch::IntArrayRef({kPlayerAxisSlots, kPlanets, kPlanetFeatures}),
              "resolution planet features: expected [players, planets, features]");

  torch::Tensor survival_abs =
      torch::zeros({kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  torch::Tensor flip_abs =
      torch::zeros({kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  torch::Tensor stable_flip_abs =
      torch::zeros({kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  torch::Tensor owner_churn_abs =
      torch::zeros({kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  torch::Tensor last_decisive_battle_step_abs =
      torch::zeros({kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  torch::Tensor post_horizon_owner_margin_abs =
      torch::zeros({kPlanets, kPlayerAxisSlots},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  compute_future_resolution_player_scalar_features_abs(
      arrivals, planets, num_agents, survival_abs, flip_abs, stable_flip_abs,
      owner_churn_abs, last_decisive_battle_step_abs,
      post_horizon_owner_margin_abs);
  auto survival = survival_abs.accessor<float, 2>();
  auto flip = flip_abs.accessor<float, 2>();
  auto stable_flip = stable_flip_abs.accessor<float, 2>();
  auto owner_churn = owner_churn_abs.accessor<float, 2>();
  auto last_decisive =
      last_decisive_battle_step_abs.accessor<float, 2>();
  auto post_margin = post_horizon_owner_margin_abs.accessor<float, 2>();
  auto planet_features = orbit_planet_features.accessor<float, 3>();

  for (int32_t pidx = 0; pidx < kPlayerAxisSlots; ++pidx) {
    for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
      const int32_t base =
          kPlanetPlayerFeatureOffset + player_block * kPlanetPlayerFeaturesPerPlayer;
      for (int32_t slot = 0; slot < kPlanets; ++slot) {
        planet_features[pidx][slot][base + kPlanetPlayerFeatureOwnerSurvivalMargin] =
            0.0f;
        planet_features[pidx][slot][base + kPlanetPlayerFeatureFlipTime] = 0.0f;
        planet_features[pidx][slot][base + kPlanetPlayerFeatureStableFlipTime] =
            0.0f;
        planet_features[pidx][slot][base + kPlanetPlayerFeatureOwnerChurn] =
            0.0f;
        planet_features[pidx][slot][base + kPlanetPlayerFeatureLastDecisiveBattleStep] =
            0.0f;
        planet_features[pidx][slot][base + kPlanetPlayerFeaturePostHorizonOwnerMargin] =
            0.0f;
      }
    }
  }

  const int32_t n = static_cast<int32_t>(planets.size());
  for (int32_t pidx = 0; pidx < num_agents; ++pidx) {
    const int32_t policy_slot = policy_slot_for_compact_agent(pidx, num_agents);
    int32_t order[kPlayerAxisSlots];
    self_enemy_player_order_for_compact_planet_features(pidx, num_agents, order);
    for (int32_t player_block = 0; player_block < kPlayerAxisSlots; ++player_block) {
      const int32_t owner_id = order[player_block];
      if (owner_id < 0) {
        continue;
      }
      const int32_t base =
          kPlanetPlayerFeatureOffset + player_block * kPlanetPlayerFeaturesPerPlayer;
      for (int32_t slot = 0; slot < n; ++slot) {
        planet_features[policy_slot][slot][base + kPlanetPlayerFeatureOwnerSurvivalMargin] =
            survival[slot][owner_id];
        planet_features[policy_slot][slot][base + kPlanetPlayerFeatureFlipTime] =
            flip[slot][owner_id];
        planet_features[policy_slot][slot][base + kPlanetPlayerFeatureStableFlipTime] =
            stable_flip[slot][owner_id];
        planet_features[policy_slot][slot][base + kPlanetPlayerFeatureOwnerChurn] =
            owner_churn[slot][owner_id];
        planet_features[policy_slot][slot][base + kPlanetPlayerFeatureLastDecisiveBattleStep] =
            last_decisive[slot][owner_id];
        planet_features[policy_slot][slot][base + kPlanetPlayerFeaturePostHorizonOwnerMargin] =
            post_margin[slot][owner_id];
      }
    }
  }
}

void edge_costs_for_arrival_step(
    torch::Tensor arrivals, const StableTakeoverWork &work,
    torch::Tensor stable_takeover_raw, int32_t num_agents, int32_t t,
    int32_t dst_slot, int32_t player, int32_t &takeover_cost,
    int32_t &stable_takeover_cost, int32_t &neutralization_cost) {
  auto in = arrivals.accessor<float, 3>();
  auto stable = stable_takeover_raw.accessor<int32_t, 3>();
  int32_t step_arrivals[kPlayerAxisSlots] = {};
  double step_arrivals_double[kPlayerAxisSlots] = {};
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    step_arrivals[owner] = exact_int64_ship_count(
        static_cast<double>(in[t][dst_slot][owner]),
        "edge resolution features: arrival ships");
    step_arrivals_double[owner] = static_cast<double>(step_arrivals[owner]);
  }
  const uint32_t idx = temporal_planet_idx(t, dst_slot);
  const int32_t pre_owner = work.pre_owners[idx];
  const int32_t pre_ships = work.pre_ships[idx];
  takeover_cost =
      takeover_cost_ships(pre_owner, static_cast<double>(pre_ships),
                          step_arrivals_double, num_agents, player);
  stable_takeover_cost = stable[t][dst_slot][player];
  neutralization_cost =
      neutralization_cost_ships(pre_owner, pre_ships, step_arrivals,
                                num_agents, player);
  TORCH_CHECK_DISABLED(takeover_cost >= 0 && stable_takeover_cost >= 0 &&
                  neutralization_cost >= 0,
              "edge resolution features: negative cost");
}

struct EdgeResolutionSummary {
  std::vector<int32_t> final_owners;
  std::vector<float> post_horizon_owner_margin;
  std::vector<float> stable_flip_time;
};

float raw_positive_feature(double value, const char *name) {
  TORCH_CHECK_DISABLED(std::isfinite(value), name, ": non-finite positive feature");
  TORCH_CHECK_DISABLED(value >= 0.0, name, ": negative positive feature");
  return static_cast<float>(value);
}

float edge_roi_feature(double production, int32_t remaining_steps,
                       int32_t send_ships, const char *name) {
  TORCH_CHECK_DISABLED(production >= 0.0 && std::isfinite(production),
              name, ": bad production");
  TORCH_CHECK_DISABLED(remaining_steps >= 0, name, ": bad remaining steps");
  TORCH_CHECK_DISABLED(send_ships > 0, name, ": bad send ships");
  const double value =
      production * static_cast<double>(remaining_steps) /
      static_cast<double>(send_ships);
  return raw_positive_feature(value, name);
}

EdgeResolutionSummary edge_resolution_summary_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets,
    int32_t num_agents, const char *name) {
  const int32_t horizon = arrivals.size(0);
  TORCH_CHECK_DISABLED(horizon > 0, name, ": horizon must be positive");
  const int32_t n = static_cast<int32_t>(planets.size());
  TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, name, ": planet count");

  std::vector<int32_t> post_owners(
      static_cast<uint32_t>(horizon * kPlanets), -2);
  std::vector<double> post_ships(
      static_cast<uint32_t>(horizon * kPlanets), 0.0);
  auto post_idx = [](int32_t t, int32_t slot) -> uint32_t {
    return static_cast<uint32_t>(t * kPlanets + slot);
  };

  for_each_arrival_resolution_step(
      arrivals, planets, num_agents, name,
      [&](const ArrivalResolutionStep &step) {
        post_owners[post_idx(step.t, step.slot)] = step.post_owner;
        post_ships[post_idx(step.t, step.slot)] = step.post_ships;
      });

  EdgeResolutionSummary out;
  out.final_owners.assign(static_cast<uint32_t>(kPlanets), -2);
  out.post_horizon_owner_margin.assign(
      static_cast<uint32_t>(kPlanets * kPlayerAxisSlots), 0.0f);
  out.stable_flip_time.assign(
      static_cast<uint32_t>(kPlanets * kPlayerAxisSlots),
      static_cast<float>(horizon + 1));
  auto owner_idx = [](int32_t slot, int32_t owner) -> uint32_t {
    return static_cast<uint32_t>(slot * kPlayerAxisSlots + owner);
  };

  double zero_arrivals[kPlayerAxisSlots] = {};
  for (int32_t slot = 0; slot < n; ++slot) {
    int32_t suffix_owner = -2;
    for (int32_t t = horizon - 1; t >= 0; --t) {
      const int32_t owner = post_owners[post_idx(t, slot)];
      if (t == horizon - 1) {
        suffix_owner = owner;
      } else if (suffix_owner != owner) {
        suffix_owner = -2;
      }
      if (suffix_owner >= 0) {
        out.stable_flip_time[owner_idx(slot, suffix_owner)] =
            static_cast<float>(t + 1);
      }
    }
    if (planets[static_cast<uint32_t>(slot)].owner >= 0 &&
        suffix_owner == planets[static_cast<uint32_t>(slot)].owner) {
      out.stable_flip_time[owner_idx(
          slot, planets[static_cast<uint32_t>(slot)].owner)] = 0.0f;
    }

    const int32_t final_owner = post_owners[post_idx(horizon - 1, slot)];
    const double final_ships = post_ships[post_idx(horizon - 1, slot)];
    TORCH_CHECK_DISABLED(final_owner >= -1 && final_owner < kPlayerAxisSlots,
                name, ": bad final owner");
    TORCH_CHECK_DISABLED(final_ships >= 0.0 && std::isfinite(final_ships),
                name, ": bad final ships");
    out.final_owners[static_cast<uint32_t>(slot)] = final_owner;
    for (int32_t owner = 0; owner < num_agents; ++owner) {
      if (final_owner == owner) {
        out.post_horizon_owner_margin[owner_idx(slot, owner)] =
            raw_signed_ship_margin_feature(final_ships);
      } else {
        const int32_t cost =
            takeover_cost_ships(final_owner, final_ships, zero_arrivals,
                                num_agents, owner);
        out.post_horizon_owner_margin[owner_idx(slot, owner)] =
            -raw_ship_cost_feature(cost, name);
      }
      const float stable_flip =
          out.stable_flip_time[owner_idx(slot, owner)];
      TORCH_CHECK_DISABLED(0.0f <= stable_flip &&
                      stable_flip <= static_cast<float>(horizon + 1),
                  name, ": stable flip out of range");
    }
  }
  return out;
}

float source_stable_hold_margin_after_send(
    torch::Tensor arrivals, const StableTakeoverWork &work, int32_t num_agents,
    int32_t src_slot, int32_t player, int32_t send_ships) {
  TORCH_CHECK_DISABLED(0 <= src_slot && src_slot < work.n,
              "edge source hold margin: src_slot");
  TORCH_CHECK_DISABLED(0 <= player && player < num_agents,
              "edge source hold margin: player");
  TORCH_CHECK_DISABLED(send_ships >= 0, "edge source hold margin: send ships");
  auto in = arrivals.accessor<float, 3>();
  int32_t step_arrivals[kPlayerAxisSlots] = {};
  for (int32_t owner = 0; owner < num_agents; ++owner) {
    step_arrivals[owner] = exact_int64_ship_count(
        static_cast<double>(in[0][src_slot][owner]),
        "edge source hold margin arrivals");
  }
  const uint32_t idx = temporal_planet_idx(0, src_slot);
  TORCH_CHECK_DISABLED(work.pre_owners[idx] == player,
              "edge source hold margin: source owner mismatch");
  TORCH_CHECK_DISABLED(work.pre_ships[idx] >= send_ships,
              "edge source hold margin: send exceeds source ships");
  const int32_t cost = stable_takeover_cost_ships(
      player, work.pre_ships[idx] - send_ships, step_arrivals, num_agents,
      player, work.required_after[temporal_planet_owner_idx(0, src_slot, player)]);
  return cost == 0 ? 0.0f
                   : -raw_ship_cost_feature(cost, "edge source hold margin");
}

float dst_motion_angle_to_src_dst_feature(
    const Planet &src, const Planet &dst,
    const SmallPlanetIdSet &comet_planet_ids) {
  if (!planet_is_rotating_for_mask(dst.id, dst.x, dst.y, dst.radius,
                                   comet_planet_ids)) {
    return 0.0f;
  }
  const double edge_x = dst.x - src.x;
  const double edge_y = dst.y - src.y;
  const double edge_norm = std::hypot(edge_x, edge_y);
  if (edge_norm == 0.0) {
    return 0.0f;
  }
  const double radius_x = dst.x - kCenter;
  const double radius_y = dst.y - kCenter;
  const double motion_x = -radius_y;
  const double motion_y = radius_x;
  const double motion_norm = std::hypot(motion_x, motion_y);
  TORCH_CHECK_DISABLED(motion_norm > 0.0,
              "edge dst motion angle: dynamic planet at rotation center");
  const double cos_angle =
      std::max(-1.0, std::min(1.0, (motion_x * edge_x + motion_y * edge_y) /
                                       (motion_norm * edge_norm)));
  return static_cast<float>(std::acos(cos_angle));
}

void edge_relative_velocity_features(
    const NoopCachedPlanet *noop_planets_row,
    const NoopCachedPlanet *noop_next_planets_row, int32_t src_slot,
    int32_t dst_slot, float &velocity_dx, float &velocity_dy,
    float &closing_speed) {
  const NoopCachedPlanet &src_now =
      noop_planets_row[static_cast<uint32_t>(src_slot)];
  const NoopCachedPlanet &dst_now =
      noop_planets_row[static_cast<uint32_t>(dst_slot)];
  const NoopCachedPlanet &src_next =
      noop_next_planets_row[static_cast<uint32_t>(src_slot)];
  const NoopCachedPlanet &dst_next =
      noop_next_planets_row[static_cast<uint32_t>(dst_slot)];
  if (src_now.id != src_next.id || dst_now.id != dst_next.id) {
    velocity_dx = 0.0f;
    velocity_dy = 0.0f;
    closing_speed = 0.0f;
    return;
  }
  const double src_vx = src_next.x - src_now.x;
  const double src_vy = src_next.y - src_now.y;
  const double dst_vx = dst_next.x - dst_now.x;
  const double dst_vy = dst_next.y - dst_now.y;
  const double dx = dst_vx - src_vx;
  const double dy = dst_vy - src_vy;
  TORCH_CHECK_DISABLED(std::isfinite(dx) && std::isfinite(dy),
              "edge relative velocity: non-finite value");
  const double edge_x = dst_now.x - src_now.x;
  const double edge_y = dst_now.y - src_now.y;
  const double edge_norm = std::hypot(edge_x, edge_y);
  if (edge_norm == 0.0) {
    velocity_dx = 0.0f;
    velocity_dy = 0.0f;
    closing_speed = 0.0f;
    return;
  }
  const double closing = -(dx * edge_x + dy * edge_y) / edge_norm;
  TORCH_CHECK_DISABLED(std::isfinite(closing),
              "edge relative velocity: non-finite closing speed");
  velocity_dx = static_cast<float>(dx);
  velocity_dy = static_cast<float>(dy);
  closing_speed = static_cast<float>(closing);
}

template <typename ProfileEnv>
void fill_future_resolution_edge_features_from_arrivals_impl(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_pairwise_features, const ProfileEnv *profile_env) {
  const int32_t horizon = arrivals.size(0);
  // Edge bucket features that depended on available_hit_mask are deprecated and
  // intentionally remain at their zero defaults. Max-send features below are
  // not deprecated: they are computed independently from direct send-all geometry.
  (void)available_hit_mask;
  torch::Tensor arrivals_c;
  int32_t n = 0;
  NoopView noop;
  const NoopCachedPlanet *noop_planets_row = nullptr;
  const NoopCachedPlanet *noop_next_planets_row = nullptr;
  StableTakeoverWork work;
  torch::Tensor stable_takeover_raw;
  EdgeResolutionSummary resolution_summary;
  {
    typename ProfileEnv::WallProfileSpan profile(profile_env, "edge_prep");
    arrivals_c = arrivals.contiguous();
    assert_abs_temporal_player_planet_features(
        arrivals_c, horizon, "edge resolution features arrivals");
    TORCH_CHECK_DISABLED(num_agents == 2 || num_agents == 4,
                "edge resolution features: num_agents");
    TORCH_CHECK_DISABLED(orbit_planet_pairwise_features.device().is_cpu(),
                "edge resolution features: pairwise features must be CPU");
    TORCH_CHECK_DISABLED(orbit_planet_pairwise_features.dtype() == torch::kFloat32,
                "edge resolution features: pairwise features dtype");
    TORCH_CHECK_DISABLED(orbit_planet_pairwise_features.sizes() ==
                    torch::IntArrayRef({kPlayerAxisSlots, kPairwise,
                                        kEdgeFeatures}),
                "edge resolution features: pairwise features shape");
    n = static_cast<int32_t>(planets.size());
    TORCH_CHECK_DISABLED(0 <= n && n <= kPlanets, "edge resolution features: planet count");
    noop = make_noop_view(noop_cached_planets_flat, noop_spatial_grid);
    TORCH_CHECK_DISABLED(0 <= noop_base_frame && noop_base_frame < noop.n_frames,
                "edge resolution features: noop cache missing current frame");
    TORCH_CHECK_DISABLED(noop_base_frame + 1 < noop.n_frames,
                "edge resolution features: noop cache missing next frame");
    noop_planets_row =
        noop.flat + static_cast<uint32_t>(noop_base_frame) * static_cast<uint32_t>(kPlanets);
    noop_next_planets_row =
        noop.flat + static_cast<uint32_t>(noop_base_frame + 1) *
                        static_cast<uint32_t>(kPlanets);
    for (int32_t slot = 0; slot < n; ++slot) {
      TORCH_CHECK_DISABLED(planets[static_cast<uint32_t>(slot)].id ==
                      noop_planets_row[static_cast<uint32_t>(slot)].id,
                  "edge resolution features: current/noop planet id mismatch");
    }
    {
      typename ProfileEnv::WallProfileSpan profile(profile_env, "edge_build_stable_takeover_work");
      work = build_stable_takeover_work(
          arrivals_c, planets, num_agents, "edge resolution stable work");
    }
    {
      typename ProfileEnv::WallProfileSpan profile(profile_env, "edge_stable_takeover_cost");
      stable_takeover_raw = stable_takeover_cost_abs_int64_from_work(
          arrivals_c, work, num_agents, "edge resolution stable takeover");
    }
    {
      typename ProfileEnv::WallProfileSpan profile(profile_env, "edge_resolution_summary");
      resolution_summary = edge_resolution_summary_from_arrivals(
          arrivals_c, planets, num_agents, "edge resolution summary");
    }
  }
  auto edge_features = orbit_planet_pairwise_features.accessor<float, 3>();
  auto arrivals_acc = arrivals_c.accessor<float, 3>();
  auto stable_takeover = stable_takeover_raw.accessor<int32_t, 3>();
  auto owner_summary_idx = [](int32_t slot, int32_t owner) -> uint32_t {
    return static_cast<uint32_t>(slot * kPlayerAxisSlots + owner);
  };
  auto edge_cost_idx = [](int32_t t, int32_t slot, int32_t player) -> uint32_t {
    return static_cast<uint32_t>((t * kPlanets + slot) * kPlayerAxisSlots + player);
  };
  const uint32_t edge_cost_cache_size =
      static_cast<uint32_t>(horizon * kPlanets * kPlayerAxisSlots);
  std::vector<uint8_t> edge_cost_ready(edge_cost_cache_size, 0);
  std::vector<int32_t> edge_takeover_cost_cache(edge_cost_cache_size, 0);
  std::vector<int32_t> edge_stable_cost_cache(edge_cost_cache_size, 0);
  std::vector<int32_t> edge_neutralize_cost_cache(edge_cost_cache_size, 0);
  auto edge_costs_for_arrival_step_cached =
      [&](int32_t t, int32_t dst_slot, int32_t player, int32_t &takeover_cost,
          int32_t &stable_takeover_cost, int32_t &neutralization_cost) {
        const uint32_t idx = edge_cost_idx(t, dst_slot, player);
        if (edge_cost_ready[idx] == 0) {
          int32_t step_arrivals[kPlayerAxisSlots] = {};
          double step_arrivals_double[kPlayerAxisSlots] = {};
          for (int32_t owner = 0; owner < num_agents; ++owner) {
            step_arrivals[owner] = exact_int64_ship_count(
                static_cast<double>(arrivals_acc[t][dst_slot][owner]),
                "edge resolution features: arrival ships");
            step_arrivals_double[owner] =
                static_cast<double>(step_arrivals[owner]);
          }
          const uint32_t state_idx = temporal_planet_idx(t, dst_slot);
          const int32_t pre_owner = work.pre_owners[state_idx];
          const int32_t pre_ships = work.pre_ships[state_idx];
          edge_takeover_cost_cache[idx] =
              takeover_cost_ships(pre_owner, static_cast<double>(pre_ships),
                                  step_arrivals_double, num_agents, player);
          edge_stable_cost_cache[idx] = stable_takeover[t][dst_slot][player];
          edge_neutralize_cost_cache[idx] =
              neutralization_cost_ships(pre_owner, pre_ships, step_arrivals,
                                        num_agents, player);
          TORCH_CHECK_DISABLED(
              edge_takeover_cost_cache[idx] >= 0 &&
                  edge_stable_cost_cache[idx] >= 0 &&
                  edge_neutralize_cost_cache[idx] >= 0,
              "edge resolution features: negative cost");
          edge_cost_ready[idx] = 1;
        }
        takeover_cost = edge_takeover_cost_cache[idx];
        stable_takeover_cost = edge_stable_cost_cache[idx];
        neutralization_cost = edge_neutralize_cost_cache[idx];
      };
  const EdgeFrameCollisionMetadata edge_collision_metadata =
      edge_frame_collision_metadata(noop, noop_base_frame, comet_planet_ids);
  const int32_t edge_remaining_steps = noop.n_frames - noop_base_frame;

  auto set_edge_base = [&](int32_t eidx, int32_t feature, float value) {
    TORCH_CHECK_DISABLED(0 <= eidx && eidx < kPairwise, "edge resolution features: eidx");
    TORCH_CHECK_DISABLED(0 <= feature && feature < kEdgePlayerFeatureOffset,
                "edge resolution features: feature");
    TORCH_CHECK_DISABLED(std::isfinite(static_cast<double>(value)),
                "edge resolution features: non-finite base value");
    for (int32_t pidx = 0; pidx < num_agents; ++pidx) {
      const int32_t policy_slot = policy_slot_for_compact_agent(pidx, num_agents);
      edge_features[policy_slot][eidx][feature] = value;
    }
  };
  {
    typename ProfileEnv::WallProfileSpan profile(profile_env, "edge_init_defaults");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      const Planet &src = planets[static_cast<uint32_t>(src_slot)];
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const Planet &dst = planets[static_cast<uint32_t>(dst_slot)];
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        set_edge_base(eidx, kEdgeBaseFeatureDistance,
                    orbit_wars_policy_obs_edge_distance(src.x, src.y, dst.x, dst.y));
      set_edge_base(eidx, kEdgeBaseFeatureSrcNeutral,
                    src.owner == -1 ? 1.0f : 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureDstNeutral,
                    dst.owner == -1 ? 1.0f : 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverShips, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverShips, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverShips, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeStableTakeoverBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeStableTakeoverShips, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeShips, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeShips, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureTakeoverMarginWithMaxSend, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureStableMarginWithMaxSend, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucketAvailable, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucketHitSteps, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucketAvailable, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucketHitSteps, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketAvailable, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketHitSteps, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeStableTakeoverBucketAvailable, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeStableTakeoverBucketHitSteps, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeBucketAvailable, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeBucketHitSteps, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeBucketAvailable, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeBucketHitSteps, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureNeutralizeMarginWithMaxSend, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketRoi, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureMaxSendStableRoi, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureSourceStableHoldMarginAfterMinTakeover, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureSourceStableHoldMarginAfterMinStableTakeover, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureCaptureDeadlineSlack, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureArrivalTacticalPressure, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureSnipeScoreAtMinTakeoverTime, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureOverkillWithMinStableBucket, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureTimeToHitWithMaxSend, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureIsAvailableWithMaxSend, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureDstMotionAngleToSrcDst,
                    dst_motion_angle_to_src_dst_feature(src, dst,
                                                        comet_planet_ids));
      float velocity_dx = 0.0f;
      float velocity_dy = 0.0f;
      float closing_speed = 0.0f;
      edge_relative_velocity_features(noop_planets_row, noop_next_planets_row,
                                      src_slot, dst_slot, velocity_dx,
                                      velocity_dy, closing_speed);
      for (int32_t pidx = 0; pidx < num_agents; ++pidx) {
        const int32_t policy_slot = policy_slot_for_compact_agent(pidx, num_agents);
        const int32_t geometry_pos =
            policy_geometry_position_for_compact_player(pidx, num_agents);
        const auto rel_velocity =
            policy_geometry_rotate_vector_to_player_frame(
                velocity_dx, velocity_dy, geometry_pos);
        edge_features[policy_slot][eidx][kEdgeBaseFeatureVelocityDx] =
            rel_velocity.first;
        edge_features[policy_slot][eidx][kEdgeBaseFeatureVelocityDy] =
            rel_velocity.second;
      }
      set_edge_base(eidx, kEdgeBaseFeatureClosingSpeed, closing_speed);
      set_edge_base(eidx, kEdgeBaseFeatureStableCaptureVsCurrentOwnerValue, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureDstFinalOwnerIsSrcOwnerWithoutAction, 0.0f);
      set_edge_base(eidx, kEdgeBaseFeatureAttackRedundancyScore, 0.0f);
      }
    }
  }

  struct EdgeResolutionEdgeCache {
    bool active = false;
    int32_t player = -1;
    int32_t src_ships = 0;
    int32_t min_takeover_sn = 0;
    int32_t min_takeover_hit_steps = 0;
    int32_t min_time_takeover_sn = 0;
    int32_t min_time_takeover_hit_steps = 0;
    int32_t min_stable_takeover_sn = 0;
    int32_t min_stable_takeover_hit_steps = 0;
    int32_t min_time_stable_takeover_sn = 0;
    int32_t min_time_stable_takeover_hit_steps = 0;
    int32_t min_takeover_exact_ships = 0;
    int32_t min_takeover_exact_hit_steps = 0;
    int32_t min_stable_takeover_exact_ships = 0;
    int32_t min_stable_takeover_exact_hit_steps = 0;
    int32_t half_exact_ships = 0;
    int32_t half_exact_hit_steps = 0;
    int32_t min_neutralize_sn = 0;
    int32_t min_neutralize_hit_steps = 0;
    int32_t min_time_neutralize_sn = 0;
    int32_t min_time_neutralize_hit_steps = 0;
    int32_t max_send_hit_steps = 0;
    int32_t max_send_takeover_cost = 0;
    int32_t max_send_stable_cost = 0;
    int32_t max_send_neutralize_cost = 0;
    std::array<int32_t, kLegacyShipScanClasses> hit_steps_by_sn{};
    std::array<int32_t, kLegacyShipScanClasses> send_ships_by_sn{};
  };
  std::array<EdgeResolutionEdgeCache, kPairwise> edge_resolution_cache{};

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_collect_source_edges");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      const Planet &src = planets[static_cast<uint32_t>(src_slot)];
      const int32_t player = src.owner;
      if (player < 0 || player >= num_agents) {
        continue;
      }
      const int32_t src_ships =
          exact_int64_ship_count(src.ships, "edge resolution source ships");
      if (src_ships <= 0) {
        continue;
      }
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        if (src_slot == dst_slot) {
          continue;
        }
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        cache.active = true;
        cache.player = player;
        cache.src_ships = src_ships;
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_collect_exact_takeover_candidates_from_mask");
    TORCH_CHECK(profile_env != nullptr,
                "edge exact takeover features: profile env required");
    torch::Tensor hit_kind_last = profile_env->honest_shared_hit_kind_last();
    torch::Tensor hit_slot_last = profile_env->honest_shared_hit_slot_last();
    torch::Tensor hit_steps_last = profile_env->honest_shared_hit_steps_last();
    torch::Tensor send_ships_last = profile_env->honest_shared_send_ships_last();
    TORCH_CHECK(hit_kind_last.device().is_cpu() &&
                    hit_kind_last.dtype() == torch::kFloat32 &&
                    hit_kind_last.sizes() ==
                        torch::IntArrayRef({kPlanets, kHitClasses}),
                "edge exact takeover features: hit kind tensor");
    TORCH_CHECK(hit_slot_last.device().is_cpu() &&
                    hit_slot_last.dtype() == torch::kInt32 &&
                    hit_slot_last.sizes() ==
                        torch::IntArrayRef({kPlanets, kHitClasses}),
                "edge exact takeover features: hit slot tensor");
    TORCH_CHECK(hit_steps_last.device().is_cpu() &&
                    hit_steps_last.dtype() == torch::kInt32 &&
                    hit_steps_last.sizes() ==
                        torch::IntArrayRef({kPlanets, kHitClasses}),
                "edge exact takeover features: hit steps tensor");
    TORCH_CHECK(send_ships_last.device().is_cpu() &&
                    send_ships_last.dtype() == torch::kInt32 &&
                    send_ships_last.sizes() ==
                        torch::IntArrayRef({kPlanets, kHitClasses}),
                "edge exact takeover features: send ships tensor");
    auto hit_kind = hit_kind_last.accessor<float, 2>();
    auto hit_slot = hit_slot_last.accessor<int32_t, 2>();
    auto hit_steps = hit_steps_last.accessor<int32_t, 2>();
    auto send_ships = send_ships_last.accessor<int32_t, 2>();
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active) {
          continue;
        }
        const int32_t takeover_cls =
            dst_slot * kHitClassesPerTarget + kMoveClassSendTakeoverSubindex;
        const int32_t stable_cls =
            dst_slot * kHitClassesPerTarget +
            kMoveClassSendStableTakeoverSubindex;
        const int32_t half_cls =
            dst_slot * kHitClassesPerTarget + kMoveClassSendHalfSubindex;
        const auto collect_exact_candidate = [&](int32_t cls,
                                                int32_t &out_ships,
                                                int32_t &out_hit_steps) {
          if (static_cast<int32_t>(hit_kind[src_slot][cls]) != kHitKindTarget) {
            return;
          }
          TORCH_CHECK(hit_slot[src_slot][cls] == dst_slot,
                      "edge exact takeover features: target slot mismatch");
          const int32_t steps = hit_steps[src_slot][cls];
          const int32_t ships = send_ships[src_slot][cls];
          TORCH_CHECK(1 <= steps && steps <= horizon,
                      "edge exact takeover features: hit steps");
          TORCH_CHECK(1 <= ships && ships <= cache.src_ships,
                      "edge exact takeover features: send ships");
          out_ships = ships;
          out_hit_steps = steps;
        };
        {
          typename ProfileEnv::WallProfileSpan profile_collect_takeover(
              profile_env, "edge_collect_exact_takeover_from_mask");
          collect_exact_candidate(takeover_cls, cache.min_takeover_exact_ships,
                                  cache.min_takeover_exact_hit_steps);
        }
        {
          typename ProfileEnv::WallProfileSpan profile_collect_stable(
              profile_env, "edge_collect_exact_stable_takeover_from_mask");
          collect_exact_candidate(stable_cls,
                                  cache.min_stable_takeover_exact_ships,
                                  cache.min_stable_takeover_exact_hit_steps);
        }
        {
          typename ProfileEnv::WallProfileSpan profile_collect_half(
              profile_env, "edge_collect_exact_half_from_mask");
          collect_exact_candidate(half_cls, cache.half_exact_ships,
                                  cache.half_exact_hit_steps);
        }
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_bucket_candidate_costs");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active) {
          continue;
        }
        const int32_t player = cache.player;
        for (int32_t sn = 1; sn < kLegacyShipScanClasses; ++sn) {
          const int32_t hit_steps =
              cache.hit_steps_by_sn[static_cast<uint32_t>(sn)];
          if (hit_steps <= 0) {
            continue;
          }
          const int32_t send_ships =
              cache.send_ships_by_sn[static_cast<uint32_t>(sn)];
          TORCH_CHECK_DISABLED(send_ships > 0,
                      "edge bucket candidate costs: send_ships");
          const int32_t t = hit_steps - 1;
          int32_t takeover_cost = 0;
          int32_t stable_cost = 0;
          int32_t neutralize_cost = 0;
          edge_costs_for_arrival_step_cached(
              t, dst_slot, player, takeover_cost, stable_cost, neutralize_cost);
          if (send_ships >= takeover_cost) {
            if (cache.min_takeover_sn == 0) {
              cache.min_takeover_sn = sn;
              cache.min_takeover_hit_steps = hit_steps;
            }
            if (cache.min_time_takeover_hit_steps == 0 ||
                hit_steps < cache.min_time_takeover_hit_steps) {
              cache.min_time_takeover_sn = sn;
              cache.min_time_takeover_hit_steps = hit_steps;
            }
          }
          if (send_ships >= stable_cost) {
            if (cache.min_stable_takeover_sn == 0) {
              cache.min_stable_takeover_sn = sn;
              cache.min_stable_takeover_hit_steps = hit_steps;
            }
            if (cache.min_time_stable_takeover_hit_steps == 0 ||
                hit_steps < cache.min_time_stable_takeover_hit_steps) {
              cache.min_time_stable_takeover_sn = sn;
              cache.min_time_stable_takeover_hit_steps = hit_steps;
            }
          }
          if (send_ships >= neutralize_cost) {
            if (cache.min_neutralize_sn == 0) {
              cache.min_neutralize_sn = sn;
              cache.min_neutralize_hit_steps = hit_steps;
            }
            if (cache.min_time_neutralize_hit_steps == 0 ||
                hit_steps < cache.min_time_neutralize_hit_steps) {
              cache.min_time_neutralize_sn = sn;
              cache.min_time_neutralize_hit_steps = hit_steps;
            }
          }
        }
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_bucket_resolution_features");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        const EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active) {
          continue;
        }
        const Planet &dst = planets[static_cast<uint32_t>(dst_slot)];
        const int32_t player = cache.player;

        if (cache.min_takeover_sn > 0) {
          TORCH_CHECK_DISABLED(0 < cache.min_takeover_sn &&
                          cache.min_takeover_sn < kLegacyShipScanClasses,
                      "edge min takeover: ship_subindex");
          TORCH_CHECK_DISABLED(1 <= cache.min_takeover_hit_steps &&
                          cache.min_takeover_hit_steps <= horizon,
                      "edge min takeover: hit_steps");
          const int32_t hit_steps = cache.min_takeover_hit_steps;
          const int32_t send_ships =
              cache.send_ships_by_sn[static_cast<uint32_t>(cache.min_takeover_sn)];
          TORCH_CHECK_DISABLED(send_ships > 0,
                               "edge min takeover: cached send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucket,
                        static_cast<float>(cache.min_takeover_sn));
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucketHitSteps,
                        static_cast<float>(hit_steps));
          set_edge_base(
              eidx, kEdgeBaseFeatureSourceStableHoldMarginAfterMinTakeover,
              source_stable_hold_margin_after_send(
                  arrivals_c, work, num_agents, src_slot, player, send_ships));

          const int32_t t = hit_steps - 1;
          int32_t step_arrivals[kPlayerAxisSlots] = {};
          double step_arrivals_double[kPlayerAxisSlots] = {};
          for (int32_t owner = 0; owner < num_agents; ++owner) {
            step_arrivals[owner] = exact_int64_ship_count(
                static_cast<double>(arrivals_acc[t][dst_slot][owner]),
                "edge min takeover arrival ships");
            step_arrivals_double[owner] =
                static_cast<double>(step_arrivals[owner]);
          }
          const uint32_t state_idx = temporal_planet_idx(t, dst_slot);
          const int32_t takeover_cost = takeover_cost_ships(
              work.pre_owners[state_idx],
              static_cast<double>(work.pre_ships[state_idx]),
              step_arrivals_double, num_agents, player);
          int32_t no_action_owner = -2;
          int32_t no_action_ships = 0;
          post_state_after_added_arrival_int(
              work.pre_owners[state_idx], work.pre_ships[state_idx],
              step_arrivals, num_agents, player, 0, no_action_owner,
              no_action_ships);
          set_edge_base(eidx, kEdgeBaseFeatureSnipeScoreAtMinTakeoverTime,
                        snipe_window_score_feature(
                            step_arrivals, num_agents, player, takeover_cost,
                            no_action_owner));
        }

        if (cache.min_time_takeover_sn > 0) {
          TORCH_CHECK_DISABLED(cache.min_time_takeover_sn >= cache.min_takeover_sn &&
                          cache.min_takeover_sn > 0,
                      "edge min time takeover: bucket below min takeover bucket");
          TORCH_CHECK_DISABLED(1 <= cache.min_time_takeover_hit_steps &&
                          cache.min_time_takeover_hit_steps <= horizon,
                      "edge min time takeover: hit_steps");
          const int32_t send_ships =
              cache.send_ships_by_sn[
                  static_cast<uint32_t>(cache.min_time_takeover_sn)];
          TORCH_CHECK_DISABLED(send_ships > 0,
                               "edge min time takeover: cached send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucket,
                        static_cast<float>(cache.min_time_takeover_sn));
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucketHitSteps,
                        static_cast<float>(cache.min_time_takeover_hit_steps));
        }

        if (cache.min_stable_takeover_sn > 0) {
          TORCH_CHECK_DISABLED(0 < cache.min_stable_takeover_sn &&
                          cache.min_stable_takeover_sn < kLegacyShipScanClasses,
                      "edge min stable takeover: ship_subindex");
          TORCH_CHECK_DISABLED(1 <= cache.min_stable_takeover_hit_steps &&
                          cache.min_stable_takeover_hit_steps <= horizon,
                      "edge min stable takeover: hit_steps");
          const int32_t hit_steps = cache.min_stable_takeover_hit_steps;
          const int32_t send_ships =
              cache.send_ships_by_sn[
                  static_cast<uint32_t>(cache.min_stable_takeover_sn)];
          TORCH_CHECK_DISABLED(send_ships > 0,
                               "edge min stable takeover: cached send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucket,
                        static_cast<float>(cache.min_stable_takeover_sn));
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketHitSteps,
                        static_cast<float>(hit_steps));
          set_edge_base(
              eidx,
              kEdgeBaseFeatureSourceStableHoldMarginAfterMinStableTakeover,
              source_stable_hold_margin_after_send(
                  arrivals_c, work, num_agents, src_slot, player, send_ships));
          const int32_t future_steps =
              std::max<int32_t>(0, horizon - hit_steps);
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketRoi,
                        edge_roi_feature(dst.production, future_steps,
                                         send_ships,
                                         "edge min stable takeover roi"));
          const int32_t t = hit_steps - 1;
          int32_t takeover_cost = 0;
          int32_t stable_cost = 0;
          int32_t neutralize_cost = 0;
          edge_costs_for_arrival_step_cached(
              t, dst_slot, player, takeover_cost, stable_cost, neutralize_cost);
          set_edge_base(
              eidx, kEdgeBaseFeatureOverkillWithMinStableBucket,
              raw_signed_ship_margin_feature(
                  static_cast<double>(send_ships - stable_cost)));
          const float no_action_margin =
              resolution_summary.post_horizon_owner_margin[owner_summary_idx(
                  dst_slot, player)];
          const double value =
              no_action_margin < 0.0f
                  ? -static_cast<double>(no_action_margin) +
                        dst.production * static_cast<double>(future_steps)
                  : 0.0;
          set_edge_base(eidx,
                        kEdgeBaseFeatureStableCaptureVsCurrentOwnerValue,
                        raw_positive_feature(
                            value,
                            "edge stable capture vs current owner value"));
        }

        if (cache.min_time_stable_takeover_sn > 0) {
          TORCH_CHECK_DISABLED(
              cache.min_time_stable_takeover_sn >=
                      cache.min_stable_takeover_sn &&
                  cache.min_stable_takeover_sn > 0,
              "edge min time stable takeover: bucket below min stable takeover bucket");
          TORCH_CHECK_DISABLED(1 <= cache.min_time_stable_takeover_hit_steps &&
                          cache.min_time_stable_takeover_hit_steps <= horizon,
                      "edge min time stable takeover: hit_steps");
          const int32_t send_ships =
              cache.send_ships_by_sn[
                  static_cast<uint32_t>(cache.min_time_stable_takeover_sn)];
          TORCH_CHECK_DISABLED(
              send_ships > 0,
              "edge min time stable takeover: cached send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeStableTakeoverBucket,
                        static_cast<float>(cache.min_time_stable_takeover_sn));
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeStableTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(
              eidx, kEdgeBaseFeatureMinTimeStableTakeoverBucketAvailable,
              1.0f);
          set_edge_base(
              eidx, kEdgeBaseFeatureMinTimeStableTakeoverBucketHitSteps,
              static_cast<float>(cache.min_time_stable_takeover_hit_steps));
        }

        if (cache.min_neutralize_sn > 0) {
          TORCH_CHECK_DISABLED(0 < cache.min_neutralize_sn &&
                          cache.min_neutralize_sn < kLegacyShipScanClasses,
                      "edge min neutralize: ship_subindex");
          TORCH_CHECK_DISABLED(1 <= cache.min_neutralize_hit_steps &&
                          cache.min_neutralize_hit_steps <= horizon,
                      "edge min neutralize: hit_steps");
          const int32_t send_ships =
              cache.send_ships_by_sn[
                  static_cast<uint32_t>(cache.min_neutralize_sn)];
          TORCH_CHECK_DISABLED(send_ships > 0,
                               "edge min neutralize: cached send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeBucket,
                        static_cast<float>(cache.min_neutralize_sn));
          set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinNeutralizeBucketHitSteps,
                        static_cast<float>(cache.min_neutralize_hit_steps));
        }

        if (cache.min_time_neutralize_sn > 0) {
          TORCH_CHECK_DISABLED(
              cache.min_time_neutralize_sn >= cache.min_neutralize_sn &&
                  cache.min_neutralize_sn > 0,
              "edge min time neutralize: bucket below min neutralize bucket");
          TORCH_CHECK_DISABLED(1 <= cache.min_time_neutralize_hit_steps &&
                          cache.min_time_neutralize_hit_steps <= horizon,
                      "edge min time neutralize: hit_steps");
          const int32_t send_ships =
              cache.send_ships_by_sn[
                  static_cast<uint32_t>(cache.min_time_neutralize_sn)];
          TORCH_CHECK_DISABLED(send_ships > 0,
                               "edge min time neutralize: cached send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeBucket,
                        static_cast<float>(cache.min_time_neutralize_sn));
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeNeutralizeBucketHitSteps,
                        static_cast<float>(cache.min_time_neutralize_hit_steps));
        }
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_exact_takeover_resolution_features");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        const EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active) {
          continue;
        }
        const Planet &dst = planets[static_cast<uint32_t>(dst_slot)];
        const int32_t player = cache.player;
        if (cache.min_takeover_exact_ships > 0) {
          typename ProfileEnv::WallProfileSpan profile_takeover(
              profile_env, "edge_exact_takeover_resolution_takeover");
          const int32_t send_ships = cache.min_takeover_exact_ships;
          const int32_t hit_steps = cache.min_takeover_exact_hit_steps;
          TORCH_CHECK(1 <= hit_steps && hit_steps <= horizon,
                      "edge exact min takeover: hit_steps");
          TORCH_CHECK(send_ships > 0,
                      "edge exact min takeover: send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinTakeoverBucketHitSteps,
                        static_cast<float>(hit_steps));
          set_edge_base(
              eidx, kEdgeBaseFeatureSourceStableHoldMarginAfterMinTakeover,
              source_stable_hold_margin_after_send(
                  arrivals_c, work, num_agents, src_slot, player, send_ships));
        }
        if (cache.half_exact_ships > 0) {
          typename ProfileEnv::WallProfileSpan profile_half(
              profile_env, "edge_exact_half_resolution_features");
          const int32_t send_ships = cache.half_exact_ships;
          const int32_t hit_steps = cache.half_exact_hit_steps;
          TORCH_CHECK(1 <= hit_steps && hit_steps <= horizon,
                      "edge exact half: hit_steps");
          TORCH_CHECK(send_ships > 0, "edge exact half: send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinTimeTakeoverBucketHitSteps,
                        static_cast<float>(hit_steps));
        }
        if (cache.min_stable_takeover_exact_ships > 0) {
          typename ProfileEnv::WallProfileSpan profile_stable_takeover(
              profile_env, "edge_exact_takeover_resolution_stable_takeover");
          const int32_t send_ships = cache.min_stable_takeover_exact_ships;
          const int32_t hit_steps = cache.min_stable_takeover_exact_hit_steps;
          TORCH_CHECK(1 <= hit_steps && hit_steps <= horizon,
                      "edge exact min stable takeover: hit_steps");
          TORCH_CHECK(send_ships > 0,
                      "edge exact min stable takeover: send_ships");
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverShips,
                        static_cast<float>(send_ships));
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketAvailable,
                        1.0f);
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketHitSteps,
                        static_cast<float>(hit_steps));
          set_edge_base(
              eidx,
              kEdgeBaseFeatureSourceStableHoldMarginAfterMinStableTakeover,
              source_stable_hold_margin_after_send(
                  arrivals_c, work, num_agents, src_slot, player, send_ships));
          const int32_t future_steps =
              std::max<int32_t>(0, horizon - hit_steps);
          set_edge_base(eidx, kEdgeBaseFeatureMinStableTakeoverBucketRoi,
                        edge_roi_feature(dst.production, future_steps,
                                         send_ships,
                                         "edge exact min stable takeover roi"));
          int32_t takeover_cost = 0;
          int32_t stable_cost = 0;
          int32_t neutralize_cost = 0;
          edge_costs_for_arrival_step_cached(
              hit_steps - 1, dst_slot, player, takeover_cost, stable_cost,
              neutralize_cost);
          set_edge_base(
              eidx, kEdgeBaseFeatureOverkillWithMinStableBucket,
              raw_signed_ship_margin_feature(
                  static_cast<double>(send_ships - stable_cost)));
          const float no_action_margin =
              resolution_summary.post_horizon_owner_margin[owner_summary_idx(
                  dst_slot, player)];
          const double value =
              no_action_margin < 0.0f
                  ? -static_cast<double>(no_action_margin) +
                        dst.production * static_cast<double>(future_steps)
                  : 0.0;
          set_edge_base(eidx,
                        kEdgeBaseFeatureStableCaptureVsCurrentOwnerValue,
                        raw_positive_feature(
                            value,
                            "edge exact stable capture vs current owner value"));
        }
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_collect_max_send_candidates");
    // Max-send features do not use available_hit_mask. They evaluate the
    // current source ship count directly against target geometry.
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active) {
          continue;
        }
        const int32_t max_send_ships = cache.src_ships;
        const NoopCachedPlanet &src =
            noop_planets_row[static_cast<uint32_t>(src_slot)];
        const NoopCachedPlanet &dst =
            noop_planets_row[static_cast<uint32_t>(dst_slot)];
        const int32_t hit_steps = direct_edge_target_hit_steps_with_metadata(
            noop, noop_base_frame, edge_remaining_steps, src_slot, src, dst_slot,
            dst, comet_planet_ids, max_send_ships, ship_speed,
            edge_collision_metadata);
        if (hit_steps < 1 || hit_steps > horizon) {
          continue;
        }
        const int32_t t = hit_steps - 1;
        cache.max_send_hit_steps = hit_steps;
        edge_costs_for_arrival_step_cached(
            t, dst_slot, cache.player, cache.max_send_takeover_cost,
            cache.max_send_stable_cost, cache.max_send_neutralize_cost);
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_max_send_margin_features");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        const EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active || cache.max_send_hit_steps == 0) {
          continue;
        }
        const Planet &dst = planets[static_cast<uint32_t>(dst_slot)];
        const int32_t max_send_ships = cache.src_ships;
        const int32_t hit_steps = cache.max_send_hit_steps;
        set_edge_base(eidx, kEdgeBaseFeatureTimeToHitWithMaxSend,
                      static_cast<float>(hit_steps));
        set_edge_base(eidx, kEdgeBaseFeatureIsAvailableWithMaxSend, 1.0f);
        set_edge_base(
            eidx, kEdgeBaseFeatureTakeoverMarginWithMaxSend,
            raw_signed_ship_margin_feature(
                static_cast<double>(max_send_ships -
                                    cache.max_send_takeover_cost)));
        set_edge_base(
            eidx, kEdgeBaseFeatureStableMarginWithMaxSend,
            raw_signed_ship_margin_feature(
                static_cast<double>(max_send_ships -
                                    cache.max_send_stable_cost)));
        const int32_t neutralize_pre_owner =
            work.pre_owners[temporal_planet_idx(hit_steps - 1, dst_slot)];
        float neutralize_margin_with_max_send = 0.0f;
        if (neutralize_pre_owner == -1) {
          neutralize_margin_with_max_send = raw_signed_ship_margin_feature(
              static_cast<double>(max_send_ships -
                                  cache.max_send_neutralize_cost));
        } else {
          TORCH_CHECK_DISABLED(
              cache.max_send_neutralize_cost ==
                  static_cast<int32_t>(kFleetNormalizer),
              "edge max send neutralize margin: owned target cost sentinel");
        }
        set_edge_base(
            eidx, kEdgeBaseFeatureNeutralizeMarginWithMaxSend,
            neutralize_margin_with_max_send);
        if (max_send_ships >= cache.max_send_stable_cost) {
          const int32_t future_steps =
              std::max<int32_t>(0, horizon - hit_steps);
          set_edge_base(eidx, kEdgeBaseFeatureMaxSendStableRoi,
                        edge_roi_feature(dst.production, future_steps,
                                         max_send_ships,
                                         "edge max send stable roi"));
        }
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_tactical_pressure_features");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        const EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active || cache.max_send_hit_steps == 0) {
          continue;
        }
        const int32_t player = cache.player;
        const int32_t hit_steps = cache.max_send_hit_steps;
        const int32_t t = hit_steps - 1;
        int32_t max_enemy_arrivals = 0;
        const int32_t self_arrivals = exact_int64_ship_count(
            static_cast<double>(arrivals_acc[t][dst_slot][player]),
            "edge tactical pressure self arrivals");
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          if (owner == player) {
            continue;
          }
          max_enemy_arrivals = std::max(
              max_enemy_arrivals,
              exact_int64_ship_count(
                  static_cast<double>(arrivals_acc[t][dst_slot][owner]),
                  "edge tactical pressure enemy arrivals"));
        }
        set_edge_base(
            eidx, kEdgeBaseFeatureArrivalTacticalPressure,
            raw_signed_ship_margin_feature(
                static_cast<double>(max_enemy_arrivals - self_arrivals)));
        float enemy_deadline = static_cast<float>(horizon + 1);
        for (int32_t owner = 0; owner < num_agents; ++owner) {
          if (owner == player) {
            continue;
          }
          enemy_deadline = std::min(
              enemy_deadline,
              resolution_summary.stable_flip_time[owner_summary_idx(dst_slot,
                                                                     owner)]);
        }
        set_edge_base(eidx, kEdgeBaseFeatureCaptureDeadlineSlack,
                      raw_signed_ship_margin_feature(
                          static_cast<double>(enemy_deadline) -
                          static_cast<double>(hit_steps)));
      }
    }
  }

  {
    typename ProfileEnv::WallProfileSpan profile(
        profile_env, "edge_resolution_summary_flags");
    for (int32_t src_slot = 0; src_slot < n; ++src_slot) {
      for (int32_t dst_slot = 0; dst_slot < n; ++dst_slot) {
        const int32_t eidx = src_slot * kPlanets + dst_slot;
        const EdgeResolutionEdgeCache &cache =
            edge_resolution_cache[static_cast<uint32_t>(eidx)];
        if (!cache.active) {
          continue;
        }
        const int32_t player = cache.player;
        if (resolution_summary.final_owners[static_cast<uint32_t>(dst_slot)] ==
            player) {
          set_edge_base(
              eidx, kEdgeBaseFeatureDstFinalOwnerIsSrcOwnerWithoutAction, 1.0f);
        }
        if (resolution_summary.stable_flip_time[owner_summary_idx(dst_slot,
                                                                  player)] ==
            0.0f) {
          set_edge_base(eidx, kEdgeBaseFeatureAttackRedundancyScore, 1.0f);
        }
      }
    }
  }
}

void fill_future_resolution_edge_features_from_arrivals(
    torch::Tensor arrivals, const std::vector<Planet> &planets, int32_t num_agents,
    double ship_speed, int32_t noop_base_frame,
    const SmallPlanetIdSet &comet_planet_ids,
    const std::vector<NoopCachedPlanet> &noop_cached_planets_flat,
    const NoopSpatialGrid &noop_spatial_grid,
    torch::Tensor available_hit_mask,
    torch::Tensor orbit_planet_pairwise_features,
    const CppEnvStaticCacheV2 *profile_env) {
  fill_future_resolution_edge_features_from_arrivals_impl<CppEnvStaticCacheV2>(
      arrivals, planets, num_agents, ship_speed, noop_base_frame,
      comet_planet_ids, noop_cached_planets_flat, noop_spatial_grid,
      available_hit_mask, orbit_planet_pairwise_features, profile_env);
}

}  // namespace orbit_wars_honest
