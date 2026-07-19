#include "library.h"

static_assert(kMoveClassesPerTarget == 5, "Orbit move action space must expose five subactions per target");

void assert_cpu_float(const torch::Tensor &t, const char *name) {
  TORCH_CHECK_DISABLED(t.device().is_cpu(), name, ": expected CPU tensor");
  TORCH_CHECK_DISABLED(t.is_contiguous(), name, ": expected contiguous tensor");
  TORCH_CHECK_DISABLED(t.scalar_type() == c10::ScalarType::Float, name, ": expected float32");
}

void assert_cpu_bool(const torch::Tensor &t, const char *name) {
  TORCH_CHECK_DISABLED(t.device().is_cpu(), name, ": expected CPU tensor");
  TORCH_CHECK_DISABLED(t.is_contiguous(), name, ": expected contiguous tensor");
  TORCH_CHECK_DISABLED(t.scalar_type() == c10::ScalarType::Bool, name, ": expected bool");
}

void assert_cpu_int8(const torch::Tensor &t, const char *name) {
  TORCH_CHECK_DISABLED(t.device().is_cpu(), name, ": expected CPU tensor");
  TORCH_CHECK_DISABLED(t.is_contiguous(), name, ": expected contiguous tensor");
  TORCH_CHECK_DISABLED(t.scalar_type() == c10::ScalarType::Char, name, ": expected int8");
}

void assert_cpu_int64(const torch::Tensor &t, const char *name) {
  TORCH_CHECK_DISABLED(t.device().is_cpu(), name, ": expected CPU tensor");
  TORCH_CHECK_DISABLED(t.is_contiguous(), name, ": expected contiguous tensor");
  TORCH_CHECK_DISABLED(t.scalar_type() == c10::ScalarType::Long, name, ": expected int64");
}

void assert_workspace_shapes(const torch::Tensor &orbit_planet_features,
                             const torch::Tensor &orbit_planet_mask,
                             const torch::Tensor &orbit_planet_pairwise_mask,
                             const torch::Tensor &orbit_planet_pairwise_features,
                             const torch::Tensor &available_action_mask,
                             const torch::Tensor &action_taken_index,
                             const torch::Tensor &player_mask) {
  TORCH_CHECK_DISABLED(orbit_planet_features.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, kPlanetFeatures}));
  TORCH_CHECK_DISABLED(orbit_planet_mask.sizes() == torch::IntArrayRef({kPlayerAxisSlots, kPlanets}));
  TORCH_CHECK_DISABLED(orbit_planet_pairwise_mask.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPairwise}));
  TORCH_CHECK_DISABLED(orbit_planet_pairwise_features.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPairwise, kEdgeFeatures}));
  TORCH_CHECK_DISABLED(available_action_mask.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, kMoveClasses}));
  TORCH_CHECK_DISABLED(action_taken_index.sizes() ==
              torch::IntArrayRef({kPlayerAxisSlots, kPlanets, 1}));
  TORCH_CHECK_DISABLED(player_mask.sizes() == torch::IntArrayRef({kPlayerAxisSlots}));
}
