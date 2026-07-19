import type { DebugTapeFrame, DebugTapeListResponse } from "../types";

const BACKEND_BASE_URL = "";

let tapeListPromise: Promise<DebugTapeListResponse> | null = null;

function assertOrbitTapeFrameV2(frame: unknown): asserts frame is DebugTapeFrame {
  if (typeof frame !== "object" || frame === null) {
    throw new Error("tape frame must be a JSON object");
  }
  const f = frame as Record<string, unknown>;
  if (f.version !== 2) {
    throw new Error(`tape frame version must be 2 (orbit schema), got ${String(f.version)}`);
  }
  if (!Array.isArray(f.lines) || !Array.isArray(f.points) || !Array.isArray(f.texts)) {
    throw new Error("tape frame must include lines[], points[], texts[] arrays");
  }
  if (f.action_edges !== undefined && f.action_edges !== null && !Array.isArray(f.action_edges)) {
    throw new Error("tape frame action_edges must be an array/null when present");
  }
  if (
    f.fleet_arrival_traces !== undefined &&
    f.fleet_arrival_traces !== null &&
    !Array.isArray(f.fleet_arrival_traces)
  ) {
    throw new Error("tape frame fleet_arrival_traces must be an array/null when present");
  }
  if (
    f.fleet_arrival_resolution !== undefined &&
    f.fleet_arrival_resolution !== null &&
    !Array.isArray(f.fleet_arrival_resolution)
  ) {
    throw new Error("tape frame fleet_arrival_resolution must be an array/null when present");
  }
  if (
    f.orbit_planet_feature_pack !== undefined &&
    f.orbit_planet_feature_pack !== null &&
    (typeof f.orbit_planet_feature_pack !== "object" || Array.isArray(f.orbit_planet_feature_pack))
  ) {
    throw new Error("tape frame orbit_planet_feature_pack must be an object/null when present");
  }
  const pvb = f.player_value_baseline;
  if (pvb !== undefined && pvb !== null) {
    if (!Array.isArray(pvb) || !pvb.every((x) => typeof x === "number" && Number.isFinite(x))) {
      throw new Error("tape frame player_value_baseline must be a number[] or null/omitted");
    }
  }
  const elim = f.player_value_baseline_eliminated;
  if (elim !== undefined && elim !== null) {
    if (!Array.isArray(elim) || !elim.every((x) => typeof x === "boolean")) {
      throw new Error(
        "tape frame player_value_baseline_eliminated must be a boolean[] or null/omitted"
      );
    }
    if (
      pvb !== undefined &&
      pvb !== null &&
      Array.isArray(pvb) &&
      elim.length !== pvb.length
    ) {
      throw new Error(
        "tape frame player_value_baseline_eliminated length must match player_value_baseline"
      );
    }
  }
  const failed = f.player_value_baseline_failed;
  if (failed !== undefined && failed !== null) {
    if (!Array.isArray(failed) || !failed.every((x) => typeof x === "boolean")) {
      throw new Error("tape frame player_value_baseline_failed must be a boolean[] or null/omitted");
    }
    if (
      pvb !== undefined &&
      pvb !== null &&
      Array.isArray(pvb) &&
      failed.length !== pvb.length
    ) {
      throw new Error("tape frame player_value_baseline_failed length must match player_value_baseline");
    }
  }
  const psh = f.player_supervised_heads;
  if (psh !== undefined && psh !== null) {
    if (typeof psh !== "object" || Array.isArray(psh)) {
      throw new Error("tape frame player_supervised_heads must be an object when present");
    }
    for (const [headName, payload] of Object.entries(psh as Record<string, unknown>)) {
      if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
        throw new Error(`tape frame player_supervised_heads.${headName} must be an object`);
      }
      const p = payload as Record<string, unknown>;
      const pred = p.prediction;
      const tgt = p.target;
      if (
        !Array.isArray(pred) ||
        !pred.every((x) => typeof x === "number" && Number.isFinite(x))
      ) {
        throw new Error(
          `tape frame player_supervised_heads.${headName}.prediction must be number[]`
        );
      }
      if (
        !Array.isArray(tgt) ||
        !tgt.every((x) => typeof x === "number" && Number.isFinite(x))
      ) {
        throw new Error(
          `tape frame player_supervised_heads.${headName}.target must be number[]`
        );
      }
      if (pred.length !== tgt.length) {
        throw new Error(
          `tape frame player_supervised_heads.${headName}: prediction and target lengths must match`
        );
      }
      const elimH = p.eliminated;
      if (elimH !== undefined && elimH !== null) {
        if (!Array.isArray(elimH) || !elimH.every((x) => typeof x === "boolean")) {
          throw new Error(
            `tape frame player_supervised_heads.${headName}.eliminated must be boolean[]`
          );
        }
        if (elimH.length !== pred.length) {
          throw new Error(
            `tape frame player_supervised_heads.${headName}.eliminated length must match prediction`
          );
        }
      }
    }
  }
}

function backendUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (BACKEND_BASE_URL.length === 0) {
    return normalizedPath;
  }
  const base = BACKEND_BASE_URL.endsWith("/") ? BACKEND_BASE_URL.slice(0, -1) : BACKEND_BASE_URL;
  return `${base}${normalizedPath}`;
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(backendUrl(path));
  if (!response.ok) {
    throw new Error(`backend ${path} failed (${response.status})`);
  }
  return (await response.json()) as T;
}

async function fetchJsonWithInit<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(backendUrl(path), init);
  if (!response.ok) {
    throw new Error(`backend ${path} failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export function clearTapeListCache(): void {
  tapeListPromise = null;
}

export async function loadTapeList(forceRefresh: boolean = false): Promise<DebugTapeListResponse> {
  if (forceRefresh || tapeListPromise === null) {
    tapeListPromise = fetchJson<DebugTapeListResponse>("/api/tapes");
  }
  return tapeListPromise;
}

export async function loadTapeFrame(
  tapeId: string,
  frameIndex: number,
  signal: AbortSignal
): Promise<DebugTapeFrame> {
  const query = new URLSearchParams({
    tape: tapeId,
    index: String(frameIndex)
  });
  const frame = await fetchJsonWithInit<DebugTapeFrame>(`/api/tapes/frame?${query.toString()}`, {
    signal
  });
  assertOrbitTapeFrameV2(frame);
  return {
    ...frame,
    action_edges: Array.isArray(frame.action_edges) ? frame.action_edges : [],
    fleet_arrival_traces: Array.isArray(frame.fleet_arrival_traces)
      ? frame.fleet_arrival_traces
      : [],
    fleet_arrival_resolution: Array.isArray(frame.fleet_arrival_resolution)
      ? frame.fleet_arrival_resolution
      : [],
    orbit_planet_feature_pack:
      frame.orbit_planet_feature_pack !== undefined ? frame.orbit_planet_feature_pack : null
  };
}
