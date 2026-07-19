import type { CSSProperties } from "react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrthographicView } from "@deck.gl/core";
import { LineLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import type {
  DebugTapeActionEdge,
  DebugTapeFrame,
  DebugTapeFleetArrivalTrace,
  DebugTapeLine,
  DebugTapeListItem,
  DebugTapeOrbitEdgeFeature,
  DebugTapeOrbitPlanetFeature,
  DebugTapePoint,
  DebugTapeText
} from "../types";
import { clearTapeListCache, loadTapeFrame, loadTapeList } from "./tapeApi";

const VIEW = new OrthographicView({ id: "orbit-tape-ortho" });

const THEME = {
  pageBg: "#f1f5f9",
  toolbarBg: "#f8fafc",
  toolbarBorder: "#e2e8f0",
  text: "#0f172a",
  muted: "#64748b",
  controlBorder: "#cbd5e1",
  controlBg: "#ffffff"
} as const;

/** Matches ``_orbit_tape_owner_rgba`` palette (players 0..3). */
const ORBIT_PLAYER_VALUE_COLORS = ["#DC2626", "#2563EB", "#16A34A", "#D97706"] as const;

/** Fixed orthographic extent per axis: [-5, 105] world units (span 110); view centered at 50. */
const WORLD_MIN = -5;
const WORLD_MAX = 105;
const WORLD_SPAN = WORLD_MAX - WORLD_MIN;
const WORLD_VIEW_CENTER = (WORLD_MIN + WORLD_MAX) / 2;
const ORBIT_SUN_RADIUS = 10;
const ACTION_EDGE_ALLOWED: [number, number, number, number] = [22, 163, 74, 220];
const ACTION_EDGE_BLOCKED: [number, number, number, number] = [220, 38, 38, 170];
const FLEET_ARRIVAL_TRACE_COLOR: [number, number, number, number] = [15, 23, 42, 230];
const FLEET_ARRIVAL_OBJECT_COLOR: [number, number, number, number] = [250, 204, 21, 95];
const FLEET_ARRIVAL_OBJECT_STROKE: [number, number, number, number] = [15, 23, 42, 235];
const FLEET_HOVER_PICK_RADIUS_M = 0.75;
const ORBIT_PLANET_SUPERVISED_LAYER = "orbit_planet_supervised_label";
/**
 * Frontend-only tuning knobs for per-planet supervised labels.
 * Keep these in one place so we can quickly tune readability without backend changes.
 */
const ORBIT_PLANET_SUPERVISED_WORLD_SHIFT_X = 1;
const ORBIT_PLANET_SUPERVISED_WORLD_SHIFT_Y = -5;
const ORBIT_PLANET_SUPERVISED_PIXEL_SHIFT_X = 0;
const ORBIT_PLANET_SUPERVISED_PIXEL_SHIFT_Y = 0;
const ORBIT_PLANET_SUPERVISED_STACK_GAP_PX = 15;
const ORBIT_PLANET_SUPERVISED_SIZE_MULTIPLIER = 1.5;
const ORBIT_PLANET_SUPERVISED_FINAL_POLICY_SIZE_MULTIPLIER = 1.2;
const ORBIT_PLANET_SUPERVISED_CLUSTER_SPLIT_GAP_M = 0.22;
const ORBIT_PLANET_SUPERVISED_SOURCE_X_OFFSET_M = 0.12;
const ORBIT_PLANET_SUPERVISED_SOURCE_X_MATCH_EPS_M = 0.03;
const ORBIT_PLANET_SUPERVISED_TEXT_COLOR: [number, number, number, number] = [0, 0, 0, 255];
const ORBIT_HUD_LAYER = "orbit_hud";
const ORBIT_HUD_TEXT_COLOR: [number, number, number, number] = [0, 0, 0, 255];
const ORBIT_HUD_WORLD_SHIFT_X = -2;
const ORBIT_HUD_WORLD_SHIFT_Y = -2;
const ORBIT_PLAYER_BASELINE_OVERLAY_LEFT_PX = 6;
const ORBIT_PLAYER_BASELINE_OVERLAY_TOP_PX = 4;
const ORBIT_SUN_POINT = [
  {
    x: WORLD_VIEW_CENTER,
    y: WORLD_VIEW_CENTER,
    r_m: ORBIT_SUN_RADIUS,
    color: [255, 184, 0, 235] as [number, number, number, number],
    layer: "orbit_sun"
  }
] satisfies DebugTapePoint[];


function clampFrameIndex(index: number, frameCount: number): number {
  if (frameCount <= 0) {
    return 0;
  }
  if (index < 0) {
    return 0;
  }
  if (index >= frameCount) {
    return frameCount - 1;
  }
  return index;
}

function tapeOptionLabel(id: string): string {
  const sep = "__";
  const i = id.indexOf(sep);
  if (i === -1) {
    return id;
  }
  return `${id.slice(0, i)} / ${id.slice(i + sep.length)}`;
}

function fixedOrbitViewState(width: number, height: number): ViewState {
  const z = Math.log2(Math.min(width, height) / WORLD_SPAN);
  return { target: [WORLD_VIEW_CENTER, WORLD_VIEW_CENTER, 0], zoom: z };
}

type ViewState = {
  target: [number, number, number];
  zoom: number;
};

type RenderTapeText = DebugTapeText & {
  render_x: number;
  render_y: number;
  pixel_offset: [number, number];
  render_size_px: number;
};

type SupervisedDisplayLine = {
  text: string;
  size_multiplier: number;
};

type HoveredPlanetFeatureRow = {
  hitSteps: number;
  values: Array<{ playerIndex: number; value: number }> | null;
  value: number | null;
};

type HoveredEdgeFeatureRow = {
  dstPlanetId: number;
  value: number | null;
  text: string | null;
  values: Array<{ playerIndex: number; value: number }> | null;
};

function formatFeatureValue(feature: DebugTapeOrbitPlanetFeature | DebugTapeOrbitEdgeFeature, value: number): string {
  if (feature.dtype === "int64" || Number.isInteger(value)) {
    return String(value);
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}

function formatPlanetId(planetId: number): string {
  return `p${planetId}`;
}

const FINAL_POLICY_SUPERVISED_PREFIX = "final_policy: PRED ";
const FINAL_POLICY_SUPERVISED_GT_SEPARATOR = " GT ";
const FINAL_POLICY_PRED_ROW_RE =
  /dst=\d+ cls=\S+ p=[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?/g;
const FINAL_POLICY_GT_ROW_RE = /^dst=\d+ cls=\S+$/;

function supervisedDisplayLines(text: string): SupervisedDisplayLine[] {
  if (!text.startsWith(FINAL_POLICY_SUPERVISED_PREFIX)) {
    return [{ text, size_multiplier: 1 }];
  }
  const body = text.slice(FINAL_POLICY_SUPERVISED_PREFIX.length);
  const gtIndex = body.indexOf(FINAL_POLICY_SUPERVISED_GT_SEPARATOR);
  if (gtIndex < 0) {
    throw new Error(`final_policy supervised text must include GT separator: ${text}`);
  }
  const predText = body.slice(0, gtIndex).trim();
  const gtText = body.slice(gtIndex + FINAL_POLICY_SUPERVISED_GT_SEPARATOR.length).trim();
  const predRows = Array.from(predText.matchAll(FINAL_POLICY_PRED_ROW_RE), (match) => match[0]);
  if (predRows.length === 0 || predRows.join(" ") !== predText) {
    throw new Error(`final_policy supervised PRED text has unexpected format: ${text}`);
  }
  if (!FINAL_POLICY_GT_ROW_RE.test(gtText)) {
    throw new Error(`final_policy supervised GT text has unexpected format: ${text}`);
  }
  return [
    { text: "final_policy:", size_multiplier: ORBIT_PLANET_SUPERVISED_FINAL_POLICY_SIZE_MULTIPLIER },
    ...predRows.map((row) => ({
      text: `PRED ${row}`,
      size_multiplier: ORBIT_PLANET_SUPERVISED_FINAL_POLICY_SIZE_MULTIPLIER
    })),
    { text: `GT ${gtText}`, size_multiplier: ORBIT_PLANET_SUPERVISED_FINAL_POLICY_SIZE_MULTIPLIER }
  ];
}

export function OrbitTapeViewer() {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [deckSize, setDeckSize] = useState({ width: 0, height: 0 });
  const [viewState, setViewState] = useState<ViewState>({
    target: [WORLD_VIEW_CENTER, WORLD_VIEW_CENTER, 0],
    zoom: 0
  });
  const [tapes, setTapes] = useState<DebugTapeListItem[]>([]);
  const [tapeId, setTapeId] = useState("");
  const [tapeErr, setTapeErr] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameInput, setFrameInput] = useState("0");
  const [frame, setFrame] = useState<DebugTapeFrame | null>(null);
  const [status, setStatus] = useState("loading tapes…");
  const [hoveredPlanetId, setHoveredPlanetId] = useState<number | null>(null);
  const [hoveredFleetId, setHoveredFleetId] = useState<number | null>(null);
  const [selectedPlanetFeatureName, setSelectedPlanetFeatureName] = useState("");
  const [selectedEdgeFeatureName, setSelectedEdgeFeatureName] = useState("");

  const containerRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (el === null) {
      return;
    }
    const r = el.getBoundingClientRect();
    const w = Math.floor(r.width);
    const h = Math.floor(r.height);
    if (w >= 1 && h >= 1) {
      setSize({ width: w, height: h });
    }
  }, []);

  const frameCount = useMemo(() => {
    const row = tapes.find((t) => t.id === tapeId);
    return row?.frame_count ?? 0;
  }, [tapes, tapeId]);

  useEffect(() => {
    const el = containerRef.current;
    if (el === null) {
      return;
    }
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      const w = Math.floor(cr.width);
      const h = Math.floor(cr.height);
      if (w < 1 || h < 1) {
        return;
      }
      setSize({ width: w, height: h });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    setTapeErr("");
    setStatus("loading tapes…");
    loadTapeList(false)
      .then((res) => {
        if (cancelled) {
          return;
        }
        setTapes(res.tapes);
        if (res.tapes.length === 0) {
          setTapeId("");
          setFrame(null);
          setTapeErr("no tapes (check backend --tapes-root)");
          setStatus("");
          return;
        }
        setTapeId((prev) => {
          if (prev.length > 0 && res.tapes.some((t) => t.id === prev)) {
            return prev;
          }
          if (res.default_tape_id.length > 0) {
            return res.default_tape_id;
          }
          return res.tapes[0].id;
        });
        setStatus("");
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setTapeErr(String(e));
          setStatus("");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (tapeId.length === 0 || frameCount <= 0) {
      setFrame(null);
      return;
    }
    const idx = clampFrameIndex(frameIndex, frameCount);
    if (idx !== frameIndex) {
      setFrameIndex(idx);
      setFrameInput(String(idx));
      return;
    }
    const ac = new AbortController();
    loadTapeFrame(tapeId, idx, ac.signal)
      .then((f) => {
        setFrame(f);
        setHoveredPlanetId(null);
        setHoveredFleetId(null);
        setTapeErr("");
        setFrameInput(String(idx));
      })
      .catch((e: unknown) => {
        if (!ac.signal.aborted) {
          setTapeErr(String(e));
        }
      });
    return () => {
      ac.abort();
    };
  }, [tapeId, frameIndex, frameCount]);

  useEffect(() => {
    if (size.width < 1 || size.height < 1) {
      setDeckSize({ width: 0, height: 0 });
      return;
    }
    const raf = window.requestAnimationFrame(() => {
      setDeckSize(size);
    });
    setViewState(fixedOrbitViewState(size.width, size.height));
    return () => window.cancelAnimationFrame(raf);
  }, [size.width, size.height]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") {
        return;
      }
      const t = ev.target as HTMLElement | null;
      if (t !== null && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) {
        return;
      }
      if (frameCount <= 0) {
        return;
      }
      ev.preventDefault();
      setFrameIndex((prev) => clampFrameIndex(prev + (ev.key === "ArrowRight" ? 1 : -1), frameCount));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frameCount]);

  const layers = useMemo(() => {
    if (frame === null) {
      return [];
    }
    const hoveredPlanetPoint =
      hoveredPlanetId === null
        ? null
        : frame.points.find(
            (p) => p.layer === "orbit_planet" && typeof p.planet_id === "number" && p.planet_id === hoveredPlanetId
          ) ?? null;
    const hoveredSupervisedAnchorX =
      hoveredPlanetPoint === null
        ? null
        : hoveredPlanetPoint.x + hoveredPlanetPoint.r_m + ORBIT_PLANET_SUPERVISED_SOURCE_X_OFFSET_M;
    const hoveredSupervisedColorKey =
      hoveredPlanetPoint === null ? null : hoveredPlanetPoint.color.slice(0, 3).join(",");
    const regularTexts: DebugTapeText[] = [];
    const supervisedTextsRaw: DebugTapeText[] = [];
    for (const t of frame.texts) {
      if (t.layer === ORBIT_PLANET_SUPERVISED_LAYER) {
        if (hoveredSupervisedAnchorX === null || hoveredSupervisedColorKey === null) {
          continue;
        }
        if (t.color.slice(0, 3).join(",") !== hoveredSupervisedColorKey) {
          continue;
        }
        if (Math.abs(t.x - hoveredSupervisedAnchorX) > ORBIT_PLANET_SUPERVISED_SOURCE_X_MATCH_EPS_M) {
          continue;
        }
        supervisedTextsRaw.push(t);
      } else {
        regularTexts.push(t);
      }
    }
    const supervisedTexts: RenderTapeText[] = [];
    const textsByXColor = new Map<string, DebugTapeText[]>();
    for (const t of supervisedTextsRaw) {
      const key = `${t.color.join(",")}|${t.x.toFixed(3)}`;
      const bucket = textsByXColor.get(key);
      if (bucket === undefined) {
        textsByXColor.set(key, [t]);
      } else {
        bucket.push(t);
      }
    }
    for (const bucket of textsByXColor.values()) {
      bucket.sort((a, b) => a.y - b.y);
      const clusters: DebugTapeText[][] = [];
      for (const t of bucket) {
        const prevCluster = clusters[clusters.length - 1];
        if (prevCluster === undefined) {
          clusters.push([t]);
          continue;
        }
        const prev = prevCluster[prevCluster.length - 1];
        if (Math.abs(t.y - prev.y) > ORBIT_PLANET_SUPERVISED_CLUSTER_SPLIT_GAP_M) {
          clusters.push([t]);
        } else {
          prevCluster.push(t);
        }
      }
      for (const cluster of clusters) {
        const baseY = cluster[0].y;
        let row = 0;
        for (const t of cluster) {
          const displayLines = supervisedDisplayLines(t.text);
          for (const line of displayLines) {
            supervisedTexts.push({
              ...t,
              text: line.text,
              render_x: t.x + ORBIT_PLANET_SUPERVISED_WORLD_SHIFT_X,
              render_y: baseY + ORBIT_PLANET_SUPERVISED_WORLD_SHIFT_Y,
              pixel_offset: [
                ORBIT_PLANET_SUPERVISED_PIXEL_SHIFT_X,
                ORBIT_PLANET_SUPERVISED_PIXEL_SHIFT_Y + row * ORBIT_PLANET_SUPERVISED_STACK_GAP_PX
              ],
              render_size_px:
                t.size_px * ORBIT_PLANET_SUPERVISED_SIZE_MULTIPLIER * line.size_multiplier
            });
            row += 1;
          }
        }
      }
    }
    const hoveredActionEdges =
      hoveredPlanetId === null
        ? []
        : (frame.action_edges ?? []).filter((edge) => edge.source_id === hoveredPlanetId);
    const hoveredFleetTrace =
      hoveredFleetId === null
        ? null
        : (frame.fleet_arrival_traces ?? []).find((trace) => trace.fleet_id === hoveredFleetId) ?? null;
    const hoveredFleetTraceLine = hoveredFleetTrace === null ? [] : [hoveredFleetTrace];
    const hoveredFleetTraceObject =
      hoveredFleetTrace === null
        ? []
        : [
            {
              x: hoveredFleetTrace.object_x,
              y: hoveredFleetTrace.object_y,
              r_m: hoveredFleetTrace.object_radius,
              color: FLEET_ARRIVAL_OBJECT_COLOR,
              layer: "orbit_fleet_arrival_hit_object"
            } satisfies DebugTapePoint
          ];
    const fleetHoverPickPoints = frame.points.filter(
      (p) => p.layer === "orbit_fleet" && typeof p.fleet_id === "number"
    );
    return [
      new ScatterplotLayer<DebugTapePoint>({
        id: "orbit-tape-sun",
        data: ORBIT_SUN_POINT,
        getPosition: (d) => [d.x, d.y],
        getFillColor: (d) => [...d.color],
        getRadius: (d) => d.r_m,
        radiusUnits: "common",
        radiusMinPixels: 1,
        stroked: true,
        getLineColor: [255, 215, 0, 255],
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        filled: true,
        pickable: false
      }),
      new LineLayer<DebugTapeLine>({
        id: "orbit-tape-lines",
        data: frame.lines,
        getSourcePosition: (d) => [d.x0, d.y0],
        getTargetPosition: (d) => [d.x1, d.y1],
        getColor: (d) => [...d.color],
        getWidth: (d) => d.w_m,
        widthUnits: "common",
        widthMinPixels: 0.5,
        pickable: false
      }),
      new LineLayer<DebugTapeActionEdge>({
        id: "orbit-tape-action-edges",
        data: hoveredActionEdges,
        getSourcePosition: (d) => [d.x0, d.y0],
        getTargetPosition: (d) => [d.x1, d.y1],
        getColor: (d) => (d.available ? ACTION_EDGE_ALLOWED : ACTION_EDGE_BLOCKED),
        getWidth: (d) => (d.available ? 0.12 : 0.08),
        widthUnits: "common",
        widthMinPixels: 1.25,
        pickable: false
      }),
      new LineLayer<DebugTapeFleetArrivalTrace>({
        id: "orbit-tape-fleet-arrival-trace",
        data: hoveredFleetTraceLine,
        getSourcePosition: (d) => [d.x0, d.y0],
        getTargetPosition: (d) => [d.x1, d.y1],
        getColor: () => [...FLEET_ARRIVAL_TRACE_COLOR],
        getWidth: () => 0.12,
        widthUnits: "common",
        widthMinPixels: 1.5,
        pickable: false
      }),
      new ScatterplotLayer<DebugTapePoint>({
        id: "orbit-tape-fleet-arrival-hit-object",
        data: hoveredFleetTraceObject,
        getPosition: (d) => [d.x, d.y],
        getFillColor: (d) => [...d.color],
        getRadius: (d) => d.r_m,
        radiusUnits: "common",
        radiusMinPixels: 1,
        stroked: true,
        getLineColor: () => [...FLEET_ARRIVAL_OBJECT_STROKE],
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        filled: true,
        pickable: false
      }),
      new ScatterplotLayer<DebugTapePoint>({
        id: "orbit-tape-points",
        data: frame.points,
        getPosition: (d) => [d.x, d.y],
        getFillColor: (d) => [...d.color],
        getRadius: (d) => d.r_m,
        radiusUnits: "common",
        radiusMinPixels: 1,
        stroked: false,
        filled: true,
        pickable: true,
        onHover: (info) => {
          const p = info.object;
          if (p?.layer === "orbit_planet" && typeof p.planet_id === "number") {
            setHoveredPlanetId(p.planet_id);
            setHoveredFleetId(null);
          } else if (p?.layer === "orbit_fleet" && typeof p.fleet_id === "number") {
            setHoveredFleetId(p.fleet_id);
            setHoveredPlanetId(null);
          } else {
            setHoveredPlanetId(null);
            setHoveredFleetId(null);
          }
        }
      }),
      new ScatterplotLayer<DebugTapePoint>({
        id: "orbit-tape-fleet-hover-pick-points",
        data: fleetHoverPickPoints,
        getPosition: (d) => [d.x, d.y],
        getFillColor: () => [255, 255, 255, 1],
        getRadius: () => FLEET_HOVER_PICK_RADIUS_M,
        radiusUnits: "common",
        radiusMinPixels: 8,
        stroked: false,
        filled: true,
        pickable: true,
        onHover: (info) => {
          const p = info.object;
          if (p?.layer === "orbit_fleet" && typeof p.fleet_id === "number") {
            setHoveredFleetId(p.fleet_id);
            setHoveredPlanetId(null);
          } else {
            setHoveredFleetId(null);
          }
        }
      }),
      new TextLayer<DebugTapeText>({
        id: "orbit-tape-texts",
        data: regularTexts,
        getPosition: (d) =>
          d.layer === ORBIT_HUD_LAYER
            ? [d.x + ORBIT_HUD_WORLD_SHIFT_X, d.y + ORBIT_HUD_WORLD_SHIFT_Y]
            : [d.x, d.y],
        getText: (d) => d.text,
        getSize: (d) => d.size_px,
        getColor: (d) =>
          d.layer === ORBIT_HUD_LAYER ? [...ORBIT_HUD_TEXT_COLOR] : [...d.color],
        getTextAnchor: "start",
        getAlignmentBaseline: "top",
        billboard: true,
        fontFamily: "ui-monospace, monospace",
        pickable: false
      }),
      new TextLayer<RenderTapeText>({
        id: "orbit-tape-supervised-texts",
        data: supervisedTexts,
        getPosition: (d) => [d.render_x, d.render_y],
        getPixelOffset: (d) => [...d.pixel_offset],
        getText: (d) => d.text,
        getSize: (d) => d.render_size_px,
        getColor: () => [...ORBIT_PLANET_SUPERVISED_TEXT_COLOR],
        getTextAnchor: "start",
        getAlignmentBaseline: "top",
        billboard: true,
        fontFamily: "ui-monospace, monospace",
        fontWeight: 800,
        pickable: false
      })
    ];
  }, [frame, hoveredFleetId, hoveredPlanetId]);

  const applyFrameIndex = (next: number) => {
    const clamped = clampFrameIndex(next, frameCount);
    setFrameIndex(clamped);
    setFrameInput(String(clamped));
  };

  const baselineOverlayRows = useMemo(() => {
    const baseline = frame?.player_value_baseline ?? [];
    const baselineElim = frame?.player_value_baseline_eliminated ?? [];
    const baselineFailed = frame?.player_value_baseline_failed ?? [];
    return baseline.map((value, i) => ({
      playerIndex: i,
      value,
      struck: i < baselineElim.length ? Boolean(baselineElim[i]) : false,
      failed: i < baselineFailed.length ? Boolean(baselineFailed[i]) : false
    }));
  }, [frame]);

  const planetFeatureOptions = useMemo(() => frame?.orbit_planet_feature_pack?.features ?? [], [frame]);
  const edgeFeatureOptions = useMemo(() => frame?.orbit_planet_feature_pack?.edge_features ?? [], [frame]);

  const selectedPlanetFeature = useMemo<DebugTapeOrbitPlanetFeature | null>(() => {
    if (planetFeatureOptions.length === 0) {
      return null;
    }
    return (
      planetFeatureOptions.find((feature) => feature.name === selectedPlanetFeatureName) ??
      planetFeatureOptions[0]
    );
  }, [planetFeatureOptions, selectedPlanetFeatureName]);

  const selectedEdgeFeature = useMemo<DebugTapeOrbitEdgeFeature | null>(() => {
    if (edgeFeatureOptions.length === 0) {
      return null;
    }
    return edgeFeatureOptions.find((feature) => feature.name === selectedEdgeFeatureName) ?? edgeFeatureOptions[0];
  }, [edgeFeatureOptions, selectedEdgeFeatureName]);

  const hoveredPlanetFeatureRows = useMemo<HoveredPlanetFeatureRow[]>(() => {
    const pack = frame?.orbit_planet_feature_pack ?? null;
    const feature = selectedPlanetFeature;
    if (pack === null || feature === null || hoveredPlanetId === null) {
      return [];
    }
    const slot = pack.planet_ids.indexOf(hoveredPlanetId);
    if (slot < 0) {
      return [];
    }
    const horizon = Math.max(0, Math.trunc(pack.horizon));
    if (feature.kind === "player_temporal") {
      const values = feature.values as number[][][];
      const byOwner = values[slot] ?? [];
      return Array.from({ length: horizon }, (_, t) => ({
        hitSteps: t + 1,
        value: null,
        values: Array.from({ length: pack.player_axis_slots }, (_unused, playerIndex) => ({
          playerIndex,
          value: byOwner[playerIndex]?.[t] ?? 0
        }))
      }));
    }
    if (feature.kind === "player_scalar") {
      const values = feature.values as number[][];
      const byOwner = values[slot] ?? [];
      return [
        {
          hitSteps: 0,
          value: null,
          values: Array.from({ length: pack.player_axis_slots }, (_unused, playerIndex) => ({
            playerIndex,
            value: byOwner[playerIndex] ?? 0
          }))
        }
      ];
    }
    const values = feature.values as number[][];
    const byStep = values[slot] ?? [];
    return Array.from({ length: horizon }, (_, t) => ({
      hitSteps: t + 1,
      value: byStep[t] ?? 0,
      values: null
    }));
  }, [frame, hoveredPlanetId, selectedPlanetFeature]);

  const hoveredEdgeFeatureRows = useMemo<HoveredEdgeFeatureRow[]>(() => {
    const pack = frame?.orbit_planet_feature_pack ?? null;
    const feature = selectedEdgeFeature;
    if (pack === null || feature === null || hoveredPlanetId === null) {
      return [];
    }
    const srcSlot = pack.planet_ids.indexOf(hoveredPlanetId);
    if (srcSlot < 0) {
      return [];
    }
    const values = feature.values;
    const srcRows = values[srcSlot] ?? [];
    return pack.planet_ids
      .map((dstPlanetId, dstSlot) => ({ dstPlanetId, dstSlot }))
      .filter((row) => row.dstSlot !== srcSlot)
      .map(({ dstPlanetId, dstSlot }) => {
        if (feature.kind === "edge_string") {
          const rowValues = srcRows as string[];
          return {
            dstPlanetId,
            value: null,
            text: rowValues[dstSlot] ?? "",
            values: null
          };
        }
        if (feature.kind === "edge_scalar") {
          const rowValues = srcRows as number[];
          return {
            dstPlanetId,
            value: rowValues[dstSlot] ?? 0,
            text: null,
            values: null
          };
        }
        const byOwner = (srcRows as number[][])[dstSlot] ?? [];
        return {
          dstPlanetId,
          value: null,
          text: null,
          values: Array.from({ length: pack.player_axis_slots }, (_unused, playerIndex) => ({
            playerIndex,
            value: byOwner[playerIndex] ?? 0
          }))
        };
      })
      .filter((row) => {
        if (feature.kind !== "edge_string") {
          return true;
        }
        const text = row.text ?? "";
        return text.trim().length > 0;
      });
  }, [frame, hoveredPlanetId, selectedEdgeFeature]);

  const reloadTapes = () => {
    clearTapeListCache();
    setTapeErr("");
    setStatus("reloading…");
    loadTapeList(true)
      .then((res) => {
        setTapes(res.tapes);
        if (res.tapes.length === 0) {
          setTapeId("");
          setFrame(null);
          setTapeErr("no tapes");
          setStatus("");
          return;
        }
        setTapeId((prev) => {
          if (prev.length > 0 && res.tapes.some((t) => t.id === prev)) {
            return prev;
          }
          if (res.default_tape_id.length > 0) {
            return res.default_tape_id;
          }
          return res.tapes[0].id;
        });
        setFrameIndex(0);
        setFrameInput("0");
        setStatus("");
      })
      .catch((e: unknown) => {
        setTapeErr(String(e));
        setStatus("");
      });
  };

  const controlStyle: CSSProperties = {
    height: 32,
    borderRadius: 8,
    border: `1px solid ${THEME.controlBorder}`,
    padding: "0 10px",
    background: THEME.controlBg,
    color: THEME.text
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        background: THEME.pageBg,
        color: THEME.text,
        fontFamily: "system-ui, sans-serif"
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          alignItems: "center",
          padding: "8px 10px",
          borderBottom: `1px solid ${THEME.toolbarBorder}`,
          background: THEME.toolbarBg
        }}
      >
        <strong style={{ marginRight: 4 }}>orbit tape</strong>
        <select
          style={controlStyle}
          value={tapeId}
          disabled={tapes.length === 0}
          onChange={(ev) => {
            setTapeId(ev.target.value);
            setFrameIndex(0);
            setFrameInput("0");
          }}
        >
          {tapes.length === 0 ? <option value="">—</option> : null}
          {tapes.map((t) => (
            <option key={t.id} value={t.id}>
              {tapeOptionLabel(t.id)} ({t.frame_count})
            </option>
          ))}
        </select>
        <button type="button" style={{ ...controlStyle, cursor: "pointer" }} onClick={reloadTapes}>
          reload
        </button>
        <button
          type="button"
          style={{ ...controlStyle, cursor: frameCount <= 0 ? "not-allowed" : "pointer" }}
          disabled={frameCount <= 0}
          onClick={() => {
            applyFrameIndex(frameIndex - 1);
          }}
        >
          ←
        </button>
        <input
          style={{ ...controlStyle, width: 72, cursor: frameCount <= 0 ? "not-allowed" : "text" }}
          type="number"
          min={0}
          max={Math.max(0, frameCount - 1)}
          step={1}
          value={frameInput}
          disabled={frameCount <= 0}
          onChange={(ev) => {
            setFrameInput(ev.target.value);
          }}
          onBlur={(ev) => {
            const n = Number.parseInt(ev.target.value, 10);
            if (Number.isNaN(n)) {
              setFrameInput(String(frameIndex));
              return;
            }
            applyFrameIndex(n);
          }}
          onKeyDown={(ev) => {
            if (ev.key === "Enter") {
              const n = Number.parseInt(frameInput, 10);
              if (Number.isNaN(n)) {
                setFrameInput(String(frameIndex));
                return;
              }
              applyFrameIndex(n);
            }
          }}
        />
        <button
          type="button"
          style={{ ...controlStyle, cursor: frameCount <= 0 ? "not-allowed" : "pointer" }}
          disabled={frameCount <= 0}
          onClick={() => {
            applyFrameIndex(frameIndex + 1);
          }}
        >
          →
        </button>
        <span style={{ color: THEME.muted }}>
          / {Math.max(0, frameCount - 1)}
        </span>
        <select
          style={controlStyle}
          value={selectedPlanetFeature?.name ?? ""}
          disabled={planetFeatureOptions.length === 0}
          onChange={(ev) => {
            setSelectedPlanetFeatureName(ev.target.value);
          }}
        >
          {planetFeatureOptions.length === 0 ? <option value="">no planet features</option> : null}
          {planetFeatureOptions.map((feature) => (
            <option key={feature.name} value={feature.name}>
              {feature.name}
            </option>
          ))}
        </select>
        <select
          style={controlStyle}
          value={selectedEdgeFeature?.name ?? ""}
          disabled={edgeFeatureOptions.length === 0}
          onChange={(ev) => {
            setSelectedEdgeFeatureName(ev.target.value);
          }}
        >
          {edgeFeatureOptions.length === 0 ? <option value="">no edge features</option> : null}
          {edgeFeatureOptions.map((feature) => (
            <option key={feature.name} value={feature.name}>
              {feature.name}
            </option>
          ))}
        </select>
        <span style={{ color: THEME.muted, fontSize: 13 }}>
          {status}
          {tapeErr.length > 0 ? ` · ${tapeErr}` : ""}
        </span>
      </div>
      <div
        ref={containerRef}
        style={{
          flex: 1,
          minHeight: 0,
          minWidth: 0,
          position: "relative",
          width: "100%",
          height: "100%"
        }}
      >
        {deckSize.width >= 1 && deckSize.height >= 1 ? (
          <>
            <DeckGL
              width={deckSize.width}
              height={deckSize.height}
              views={VIEW}
              controller={false}
              viewState={viewState}
              layers={layers}
              getCursor={() => (hoveredPlanetId === null && hoveredFleetId === null ? "default" : "crosshair")}
            />
            <div
              style={{
                position: "absolute",
                left: ORBIT_PLAYER_BASELINE_OVERLAY_LEFT_PX,
                top: ORBIT_PLAYER_BASELINE_OVERLAY_TOP_PX,
                zIndex: 2,
                pointerEvents: "none",
                display: "flex",
                flexDirection: "column",
                gap: 6,
                textShadow: "0 1px 2px rgba(0,0,0,0.45)"
              }}
            >
              {baselineOverlayRows.map((row) => {
                const i = row.playerIndex;
                return (
                  <div
                    key={`player-baseline-overlay-${i}`}
                    style={{
                      fontSize: 16,
                      fontWeight: 700,
                      fontFamily: "ui-monospace, monospace",
                      color: ORBIT_PLAYER_VALUE_COLORS[i] ?? THEME.text,
                      lineHeight: 1.1,
                      textDecoration: row.struck ? "line-through" : undefined,
                      opacity: row.struck ? 0.72 : undefined,
                      whiteSpace: "pre"
                    }}
                  >
                    {`P${i} value: ${row.value.toFixed(4)}${row.failed ? " FAILED" : ""}`}
                  </div>
                );
              })}
            </div>
            {hoveredPlanetId !== null && selectedPlanetFeature !== null && hoveredPlanetFeatureRows.length > 0 ? (
              <div
                style={{
                  position: "absolute",
                  right: 12,
                  top: 12,
                  zIndex: 3,
                  pointerEvents: "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 13,
                  lineHeight: 1.35,
                  color: THEME.text,
                  maxHeight: "70%",
                  maxWidth: 520
                }}
              >
                <div
                  style={{
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "rgba(255, 255, 255, 0.96)",
                    border: "1px solid rgba(100, 116, 139, 0.75)",
                    boxShadow: "0 10px 24px rgba(15, 23, 42, 0.16)",
                    maxHeight: "58vh",
                    overflowY: "auto"
                  }}
                >
                  <div style={{ fontWeight: 900, marginBottom: 6 }}>
                    {selectedPlanetFeature.name} {formatPlanetId(hoveredPlanetId)}
                    <span style={{ color: THEME.muted, fontWeight: 700 }}>
                      {` · ${selectedPlanetFeature.dtype} ${selectedPlanetFeature.unit}`}
                    </span>
                  </div>
                  {hoveredPlanetFeatureRows.map((row) => (
                    <div
                      key={`planet-feature-${hoveredPlanetId}-${selectedPlanetFeature.name}-${row.hitSteps}`}
                      style={{ whiteSpace: "pre", display: "flex", gap: 10 }}
                    >
                      <span style={{ color: THEME.muted, fontWeight: 700, minWidth: 42 }}>
                        {row.hitSteps === 0 ? "now:" : `+${row.hitSteps}:`}
                      </span>
                      {row.values === null ? (
                        <span style={{ fontWeight: 900 }}>
                          {formatFeatureValue(selectedPlanetFeature, row.value ?? 0)}
                        </span>
                      ) : (
                        <span>
                          {row.values.map((v, i) => (
                            <span
                              key={`planet-feature-${hoveredPlanetId}-${selectedPlanetFeature.name}-${row.hitSteps}-p${v.playerIndex}`}
                              style={{
                                color: ORBIT_PLAYER_VALUE_COLORS[v.playerIndex] ?? THEME.text,
                                fontWeight: 900,
                                marginLeft: i === 0 ? 0 : 8
                              }}
                            >
                              {`P${v.playerIndex} ${formatFeatureValue(selectedPlanetFeature, v.value)}`}
                            </span>
                          ))}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {hoveredPlanetId !== null && selectedEdgeFeature !== null && hoveredEdgeFeatureRows.length > 0 ? (
              <div
                style={{
                  position: "absolute",
                  left: 12,
                  top: 12,
                  zIndex: 3,
                  pointerEvents: "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 13,
                  lineHeight: 1.35,
                  color: THEME.text,
                  maxHeight: "70%",
                  maxWidth: 620
                }}
              >
                <div
                  style={{
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "rgba(255, 255, 255, 0.96)",
                    border: "1px solid rgba(100, 116, 139, 0.75)",
                    boxShadow: "0 10px 24px rgba(15, 23, 42, 0.16)",
                    maxHeight: "58vh",
                    overflowY: "auto"
                  }}
                >
                  <div style={{ fontWeight: 900, marginBottom: 6 }}>
                    {selectedEdgeFeature.name} src {formatPlanetId(hoveredPlanetId)}
                    <span style={{ color: THEME.muted, fontWeight: 700 }}>
                      {` · ${selectedEdgeFeature.dtype} ${selectedEdgeFeature.unit}`}
                    </span>
                  </div>
                  {hoveredEdgeFeatureRows.map((row) => (
                    <div
                      key={`edge-feature-${hoveredPlanetId}-${selectedEdgeFeature.name}-${row.dstPlanetId}`}
                      style={{ whiteSpace: "pre", display: "flex", gap: 10 }}
                    >
                      <span style={{ color: THEME.muted, fontWeight: 700, minWidth: 52 }}>
                        {`${formatPlanetId(row.dstPlanetId)}:`}
                      </span>
                      <span>
                        {row.text !== null ? (
                          <span
                            style={{
                              color: THEME.text,
                              fontWeight: 700,
                              fontFamily: "ui-monospace, monospace",
                              fontSize: 11,
                              letterSpacing: 0.02
                            }}
                          >
                            {row.text}
                          </span>
                        ) : row.values === null ? (
                          <span style={{ color: THEME.text, fontWeight: 900 }}>
                            {formatFeatureValue(selectedEdgeFeature, row.value ?? 0)}
                          </span>
                        ) : (
                          row.values.map((v, i) => (
                            <span
                              key={`edge-feature-${hoveredPlanetId}-${selectedEdgeFeature.name}-${row.dstPlanetId}-p${v.playerIndex}`}
                              style={{
                                color: ORBIT_PLAYER_VALUE_COLORS[v.playerIndex] ?? THEME.text,
                                fontWeight: 900,
                                marginLeft: i === 0 ? 0 : 8
                              }}
                            >
                              {`P${v.playerIndex} ${formatFeatureValue(selectedEdgeFeature, v.value)}`}
                            </span>
                          ))
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}
