#pragma once

#include <torch/extension.h>
#define TORCH_CHECK_DISABLED(...) ((void)0)


#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

constexpr int32_t kPlayerAxisSlots = 4;
constexpr int32_t kPlanets = 44;
constexpr int32_t kPlanetRowLen = 7;

constexpr int32_t kPlanetArrivalHorizon = 40; // KEK

constexpr int32_t kPlanetTemporalStepMax = kPlanetArrivalHorizon + 1;
constexpr int32_t kPlanetFeatures = 49;
constexpr int32_t kTemporalPlanetFeatures = 15;
constexpr int32_t kPairwise = 1936;
constexpr int32_t kEdgeFeatures = 55;
constexpr int32_t kMoveClassNoopSubindex = 0;
constexpr int32_t kMoveClassSendAllSubindex = 1;
constexpr int32_t kMoveClassSendHalfSubindex = 2;
constexpr int32_t kMoveClassSendTakeoverSubindex = 3;
constexpr int32_t kMoveClassSendStableTakeoverSubindex = 4;
constexpr int32_t kMoveClassesPerTarget = 5;
constexpr std::array<int32_t, 4> kMoveSendSubindices = {
    kMoveClassSendAllSubindex,
    kMoveClassSendHalfSubindex,
    kMoveClassSendTakeoverSubindex,
    kMoveClassSendStableTakeoverSubindex,
};
static_assert(kMoveClassesPerTarget == 5);
constexpr int32_t kMoveClasses = kPlanets * kMoveClassesPerTarget;
constexpr int32_t kPlanetEpisodeStepMax = 500;
constexpr double kFleetNormalizer = 10000.0;

constexpr int32_t kPlanetBaseFeatureX = 0;
constexpr int32_t kPlanetBaseFeatureY = 1;
constexpr int32_t kPlanetBaseFeatureNeutralShips = 2;
constexpr int32_t kPlanetBaseFeatureEpisodeStep = 3;
constexpr int32_t kPlanetBaseFeatureIsStatic = 4;
constexpr int32_t kPlanetBaseFeatureIsDynamic = 5;
constexpr int32_t kPlanetBaseFeatureIsComet = 6;
constexpr int32_t kPlanetBaseFeatureCometTimeBeforeDespawn = 7;
constexpr int32_t kPlanetBaseFeatureRadius = 8;
constexpr int32_t kPlanetBaseFeaturePlanetProduction = 9;
constexpr int32_t kPlanetBaseFeatureOrbitRadius = 10;
constexpr int32_t kPlanetBaseFeatureAngularVelocity = 11;
constexpr int32_t kPlanetBaseFeatureSunAngle = 12;
constexpr int32_t kPlanetPlayerFeatureOffset = 13;
constexpr int32_t kPlanetPlayerFeaturesPerPlayer = 9;
constexpr int32_t kPlanetPlayerFeatureShips = 0;
constexpr int32_t kPlanetPlayerFeatureTotalFleetFrac = 1;
constexpr int32_t kPlanetPlayerFeatureProduction = 2;
constexpr int32_t kPlanetPlayerFeatureOwnerSurvivalMargin = 3;
constexpr int32_t kPlanetPlayerFeatureFlipTime = 4;
constexpr int32_t kPlanetPlayerFeatureStableFlipTime = 5;
constexpr int32_t kPlanetPlayerFeatureOwnerChurn = 6;
constexpr int32_t kPlanetPlayerFeatureLastDecisiveBattleStep = 7;
constexpr int32_t kPlanetPlayerFeaturePostHorizonOwnerMargin = 8;
static_assert(kPlanetFeatures ==
              kPlanetPlayerFeatureOffset +
                  kPlayerAxisSlots * kPlanetPlayerFeaturesPerPlayer);

constexpr int32_t kTemporalPlanetFeatureArrivalShips = 0;
constexpr int32_t kTemporalPlanetFeatureTakeoverCost = 1;
constexpr int32_t kTemporalPlanetFeatureResolutionOwner = 2;
constexpr int32_t kTemporalPlanetFeatureResolutionShips = 3;
constexpr int32_t kTemporalPlanetFeatureTimeStep = 4;
constexpr int32_t kTemporalPlanetFeatureStableTakeoverCost = 5;
constexpr int32_t kTemporalPlanetFeatureHoldCost = 6;
constexpr int32_t kTemporalPlanetFeatureHoldValid = 7;
constexpr int32_t kTemporalPlanetFeatureNeutralizationCost = 8;
constexpr int32_t kTemporalPlanetFeatureNeutralizationValid = 9;
constexpr int32_t kTemporalPlanetFeatureDenyStableEnemyCost = 10;
constexpr int32_t kTemporalPlanetFeatureBattleTieDistance = 11;
constexpr int32_t kTemporalPlanetFeatureBattleTieValid = 12;
constexpr int32_t kTemporalPlanetFeatureProductionSwingPerShip = 13;
constexpr int32_t kTemporalPlanetFeatureArrivalLeverage = 14;

constexpr int32_t kEdgeBaseFeatureDistance = 0;
constexpr int32_t kEdgeBaseFeatureSrcNeutral = 1;
constexpr int32_t kEdgeBaseFeatureDstNeutral = 2;
constexpr int32_t kEdgeBaseFeatureMinTakeoverBucket = 3;
constexpr int32_t kEdgeBaseFeatureMinTakeoverShips = 4;
constexpr int32_t kEdgeBaseFeatureMinTakeoverBucketAvailable = 5;
constexpr int32_t kEdgeBaseFeatureMinTakeoverBucketHitSteps = 6;
constexpr int32_t kEdgeBaseFeatureMinTimeTakeoverBucket = 7;
constexpr int32_t kEdgeBaseFeatureMinTimeTakeoverShips = 8;
constexpr int32_t kEdgeBaseFeatureMinTimeTakeoverBucketAvailable = 9;
constexpr int32_t kEdgeBaseFeatureMinTimeTakeoverBucketHitSteps = 10;
constexpr int32_t kEdgeBaseFeatureMinStableTakeoverBucket = 11;
constexpr int32_t kEdgeBaseFeatureMinStableTakeoverShips = 12;
constexpr int32_t kEdgeBaseFeatureMinStableTakeoverBucketAvailable = 13;
constexpr int32_t kEdgeBaseFeatureMinStableTakeoverBucketHitSteps = 14;
constexpr int32_t kEdgeBaseFeatureMinTimeStableTakeoverBucket = 15;
constexpr int32_t kEdgeBaseFeatureMinTimeStableTakeoverShips = 16;
constexpr int32_t kEdgeBaseFeatureMinTimeStableTakeoverBucketAvailable = 17;
constexpr int32_t kEdgeBaseFeatureMinTimeStableTakeoverBucketHitSteps = 18;
constexpr int32_t kEdgeBaseFeatureMinNeutralizeBucket = 19;
constexpr int32_t kEdgeBaseFeatureMinNeutralizeShips = 20;
constexpr int32_t kEdgeBaseFeatureMinNeutralizeBucketAvailable = 21;
constexpr int32_t kEdgeBaseFeatureMinNeutralizeBucketHitSteps = 22;
constexpr int32_t kEdgeBaseFeatureMinTimeNeutralizeBucket = 23;
constexpr int32_t kEdgeBaseFeatureMinTimeNeutralizeShips = 24;
constexpr int32_t kEdgeBaseFeatureMinTimeNeutralizeBucketAvailable = 25;
constexpr int32_t kEdgeBaseFeatureMinTimeNeutralizeBucketHitSteps = 26;
constexpr int32_t kEdgeBaseFeatureTakeoverMarginWithMaxSend = 27;
constexpr int32_t kEdgeBaseFeatureStableMarginWithMaxSend = 28;
constexpr int32_t kEdgeBaseFeatureNeutralizeMarginWithMaxSend = 29;
constexpr int32_t kEdgeBaseFeatureTimeToHitWithMaxSend = 30;
constexpr int32_t kEdgeBaseFeatureIsAvailableWithMaxSend = 31;
constexpr int32_t kEdgeBaseFeatureDstMotionAngleToSrcDst = 32;
constexpr int32_t kEdgeBaseFeatureVelocityDx = 33;
constexpr int32_t kEdgeBaseFeatureVelocityDy = 34;
constexpr int32_t kEdgeBaseFeatureClosingSpeed = 35;
constexpr int32_t kEdgeBaseFeatureMinStableTakeoverBucketRoi = 36;
constexpr int32_t kEdgeBaseFeatureMaxSendStableRoi = 37;
constexpr int32_t kEdgeBaseFeatureSourceStableHoldMarginAfterMinTakeover = 38;
constexpr int32_t kEdgeBaseFeatureSourceStableHoldMarginAfterMinStableTakeover = 39;
constexpr int32_t kEdgeBaseFeatureCaptureDeadlineSlack = 40;
constexpr int32_t kEdgeBaseFeatureArrivalTacticalPressure = 41;
constexpr int32_t kEdgeBaseFeatureSnipeScoreAtMinTakeoverTime = 42;
constexpr int32_t kEdgeBaseFeatureOverkillWithMinStableBucket = 43;
constexpr int32_t kEdgeBaseFeatureStableCaptureVsCurrentOwnerValue = 44;
constexpr int32_t kEdgeBaseFeatureDstFinalOwnerIsSrcOwnerWithoutAction = 45;
constexpr int32_t kEdgeBaseFeatureAttackRedundancyScore = 46;
constexpr int32_t kEdgePlayerFeatureOffset = 47;
constexpr int32_t kEdgePlayerFeaturesPerPlayer = 2;
constexpr int32_t kEdgePlayerFeatureSrcOwned = 0;
constexpr int32_t kEdgePlayerFeatureDstOwned = 1;
static_assert(kEdgeFeatures ==
              kEdgePlayerFeatureOffset +
                  kPlayerAxisSlots * kEdgePlayerFeaturesPerPlayer);

constexpr int32_t kPlayerPositionById4p[kPlayerAxisSlots] = {0, 1, 3, 2};
constexpr int32_t kPlayerIdByPosition4p[kPlayerAxisSlots] = {0, 1, 3, 2};
constexpr int32_t kLegacyShipScanClasses = 101;
constexpr int32_t kHitClassesPerTarget = kLegacyShipScanClasses + 1;
constexpr int32_t kHitClasses = kPlanets * kHitClassesPerTarget;
constexpr int32_t kLegacyShipScanClassesPerBlock = 10;
constexpr int32_t kLegacyShipScanBlocks = 10;
constexpr double kBoardSize = 100.0;
constexpr double kCenter = 50.0;
constexpr double kSunRadius = 10.0;
constexpr double kRotationRadiusLimit = 50.0;
constexpr double kPlanetClearance = 7.0;
constexpr double kPi = 3.14159265358979323846;
constexpr int32_t kMinPlanetGroups = 5;
constexpr int32_t kMaxPlanetGroups = 10;
constexpr int32_t kMinStaticGroups = 3;
constexpr double kCometRadius = 1.0;
constexpr double kCometProduction = 1.0;

inline bool move_subindex_is_send_action(int32_t subindex) {
  for (const int32_t send_subindex : kMoveSendSubindices) {
    if (subindex == send_subindex) {
      return true;
    }
  }
  return false;
}

inline int32_t ship_count_for_move_subindex(int32_t source_ships,
                                            int32_t subindex) {
  TORCH_CHECK(source_ships > 0, "source_ships must be positive");
  if (subindex == kMoveClassSendAllSubindex) {
    return source_ships;
  }
  if (subindex == kMoveClassSendHalfSubindex) {
    return (source_ships + 1) / 2;
  }
  TORCH_CHECK(false, "unsupported move subindex: ", subindex);
  return 0;
}

struct Planet {
  int32_t id = -1;
  int32_t comet_internal_id = -1;
  double comet_time_before_despawn = 0.0;
  int32_t owner = -1;
  double x = 0.0;
  double y = 0.0;
  double radius = 0.0;
  double ships = 0.0;
  double production = 0.0;
};

struct NoopCachedPlanet {
  int32_t id = -1;
  int32_t comet_internal_id = -1;
  double comet_time_before_despawn = 0.0;
  double x = 0.0;
  double y = 0.0;
  double radius = 0.0;
  double production = 0.0;
};

namespace orbit_wars_honest {

struct NoopSpatialGrid {
  int32_t n_frames = 0;
  int32_t cells_per_axis = 0;
  double min_coord = 0.0;
  double cell_size = 0.0;
  std::vector<uint64_t> cell_slot_bits;
  std::vector<uint64_t> static_cell_slot_bits;
  std::vector<uint64_t> dynamic_cell_slot_bits;
  std::vector<uint64_t> comet_cell_slot_bits;
};

}  // namespace orbit_wars_honest

inline NoopCachedPlanet noop_cached_planet_from_planet(const Planet &p) {
  NoopCachedPlanet out;
  out.id = p.id;
  out.comet_internal_id = p.comet_internal_id;
  out.comet_time_before_despawn = p.comet_time_before_despawn;
  out.x = p.x;
  out.y = p.y;
  out.radius = p.radius;
  out.production = p.production;
  return out;
}

struct Fleet {
  int32_t id = -1;
  int32_t owner = -1;
  double x = 0.0;
  double y = 0.0;
  double angle = 0.0;
  int32_t from_planet_id = -1;
  double ships = 0.0;
};

struct CometPathInfo {
  int32_t path_index = 0;
  std::vector<std::pair<double, double>> path_xy;
  double ships = 0.0;
};

struct SmallPlanetIdSet {
  std::array<uint8_t, kPlanets> present{};
  int32_t n = 0;

  void clear() {
    present.fill(0);
    n = 0;
  }

  bool empty() const {
    return n == 0;
  }

  int32_t size() const {
    return n;
  }

  bool contains(int32_t id) const {
    assert(0 <= id && id < kPlanets);
    return present[static_cast<uint32_t>(id)] != 0;
  }

  void insert(int32_t id) {
    assert(0 <= id && id < kPlanets);
    const uint32_t idx = static_cast<uint32_t>(id);
    if (present[idx] != 0) {
      return;
    }
    present[idx] = 1;
    ++n;
  }

  void append_sorted_ids(std::vector<int32_t> &out) const {
    for (int32_t id = 0; id < kPlanets; ++id) {
      if (present[static_cast<uint32_t>(id)] != 0) {
        out.push_back(id);
      }
    }
  }
};

struct CometPathByPlanetId {
  std::array<uint8_t, kPlanets> present{};
  std::array<CometPathInfo, kPlanets> paths{};
  int32_t n = 0;

  void clear() {
    present.fill(0);
    n = 0;
  }

  bool empty() const {
    return n == 0;
  }

  int32_t size() const {
    return n;
  }

  const CometPathInfo *find(int32_t planet_id) const {
    assert(0 <= planet_id && planet_id < kPlanets);
    const uint32_t idx = static_cast<uint32_t>(planet_id);
    if (present[idx] != 0) {
      return &paths[idx];
    }
    return nullptr;
  }

  const CometPathInfo &at(int32_t planet_id) const {
    const CometPathInfo *path = find(planet_id);
    assert(path != nullptr);
    return *path;
  }

  void insert(int32_t planet_id, const CometPathInfo &path) {
    assert(0 <= planet_id && planet_id < kPlanets);
    const uint32_t idx = static_cast<uint32_t>(planet_id);
    if (present[idx] == 0) {
      present[idx] = 1;
      ++n;
    }
    paths[idx] = path;
  }
};
