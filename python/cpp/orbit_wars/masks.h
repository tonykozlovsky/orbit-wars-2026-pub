#pragma once

#include "common.h"

#include <vector>

float orbit_wars_policy_obs_edge_distance(double x0, double y0, double x1, double y1);
int32_t ship_count_for_legacy_scan_subindex(int32_t sn);
int32_t max_legacy_scan_subindex_for_available_ships(int32_t ships);
bool planet_is_rotating_for_mask(int32_t planet_id, double x, double y, double radius,
                                 const SmallPlanetIdSet &comet_planet_ids);
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
                                torch::Tensor player_mask);
void fill_action_taken_index_from_classes(torch::Tensor action_classes, int32_t num_agents,
                                          torch::Tensor action_taken_index);
void orbit_wars_fill_inactive_policy_action_noops(torch::Tensor available_action_mask,
                                                  torch::Tensor action_taken_index,
                                                  int32_t num_agents);
