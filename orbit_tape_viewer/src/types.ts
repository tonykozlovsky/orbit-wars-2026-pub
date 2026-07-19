export type DebugTapeLine = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  w_m: number;
  color: [number, number, number, number];
  layer: string;
};

export type DebugTapePoint = {
  planet_id?: number;
  fleet_id?: number;
  x: number;
  y: number;
  r_m: number;
  color: [number, number, number, number];
  layer: string;
};

export type DebugTapeText = {
  x: number;
  y: number;
  text: string;
  size_px: number;
  color: [number, number, number, number];
  layer: string;
};

export type DebugTapeActionEdge = {
  source_id: number;
  target_id: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  available: boolean;
};

export type DebugTapeFleetArrivalTrace = {
  fleet_id: number;
  owner: number;
  ships: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  hit_slot: number;
  hit_planet_id: number;
  hit_steps: number;
  object_x: number;
  object_y: number;
  object_radius: number;
};

export type DebugTapeFleetArrivalResolution = {
  planet_id: number;
  hit_slot: number;
  steps: DebugTapeFleetArrivalResolutionStep[];
};

export type DebugTapeFleetArrivalResolutionStep = {
  hit_steps: number;
  owner: number;
  ships: number;
  arrivals: DebugTapeFleetArrivalResolutionArrival[];
};

export type DebugTapeFleetArrivalResolutionArrival = {
  owner: number;
  ships: number;
};

export type DebugTapeOrbitPlanetFeaturePack = {
  version: number;
  horizon: number;
  player_axis_slots: number;
  num_agents: number;
  planet_ids: number[];
  features: DebugTapeOrbitPlanetFeature[];
  edge_features?: DebugTapeOrbitEdgeFeature[];
};

export type DebugTapeOrbitPlanetFeature = {
  name: string;
  kind: "player_temporal" | "planet_temporal" | "player_scalar";
  dtype: "int64" | "float32";
  unit: string;
  values: number[][][] | number[][];
  policy_transform?: Record<string, unknown>;
};

export type DebugTapeOrbitEdgeFeature = {
  name: string;
  kind: "edge_scalar" | "edge_player_scalar" | "edge_string";
  dtype: "int64" | "float32" | "string";
  unit: string;
  values: number[][] | number[][][] | string[][];
};

/** Schema version 2: orbit tapes (lines, points, texts). Optional learner value head per real player. */
export type DebugTapeFrame = {
  version: 2;
  lines: DebugTapeLine[];
  points: DebugTapePoint[];
  texts: DebugTapeText[];
  action_edges?: DebugTapeActionEdge[];
  fleet_arrival_traces?: DebugTapeFleetArrivalTrace[];
  fleet_arrival_resolution?: DebugTapeFleetArrivalResolution[];
  orbit_planet_feature_pack?: DebugTapeOrbitPlanetFeaturePack | null;
  player_value_baseline?: number[] | null;
  /** True when fleet is empty: baseline is frozen at last step with ships (matches learner `player_mask`=0). */
  player_value_baseline_eliminated?: boolean[] | null;
  player_value_baseline_failed?: boolean[] | null;
  player_supervised_heads?: Record<
    string,
    {
      prediction: number[];
      target: number[];
      eliminated?: boolean[];
    }
  > | null;
};

export type DebugTapeListItem = {
  id: string;
  frame_count: number;
};

export type DebugTapeListResponse = {
  default_tape_id: string;
  tapes: DebugTapeListItem[];
};
