#pragma once

#include "common.h"

void assert_cpu_float(const torch::Tensor &t, const char *name);
void assert_cpu_bool(const torch::Tensor &t, const char *name);
void assert_cpu_int8(const torch::Tensor &t, const char *name);
void assert_cpu_int64(const torch::Tensor &t, const char *name);
void assert_workspace_shapes(const torch::Tensor &orbit_planet_features,
                             const torch::Tensor &orbit_planet_mask,
                             const torch::Tensor &orbit_planet_pairwise_mask,
                             const torch::Tensor &orbit_planet_pairwise_features,
                             const torch::Tensor &available_action_mask,
                             const torch::Tensor &action_taken_index,
                             const torch::Tensor &player_mask);
