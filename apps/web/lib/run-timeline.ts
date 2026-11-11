import type { Experiment, Step } from "./experiment-schema";
import type { WallStep } from "./live-feed-data";

// Geometry for the run timelines (saved worker spans and live wall-clock steps). Positions
// are shares of one window on one clock, so a bar never mixes clocks.

const NS_PER_SECOND = 1e9;
const MS_PER_SECOND = 1e3;
/** Tick spacings tried in each power of ten, smallest first. */
const TICK_STEPS = [1, 2, 2.5, 5] as const;
const DEFAULT_MAX_TICKS = 6;

export type TimelineSegment = {
  /** `wait` is measured queue delay (hatched); `work` is execution wall time (solid). */
  kind: "wait" | "work";
  /** Length as a share of the timeline window. */
  share: number;
};

export type TimelineTick = { seconds: number; share: number };

/**
 * Evenly spaced ticks from 0 to `spanSeconds` at a round spacing (1, 2, 2.5 or 5 × 10ⁿ),
 * at most `maxTicks` intervals. A non-positive or non-finite span has only the origin.
 */
export function timelineTicks(
  spanSeconds: number,
  maxTicks = DEFAULT_MAX_TICKS,
): TimelineTick[] {
  if (!Number.isFinite(spanSeconds) || spanSeconds <= 0)
    return [{ seconds: 0, share: 0 }];
  const rough = spanSeconds / Math.max(1, maxTicks);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const spacing =
    TICK_STEPS.map((step) => step * magnitude).find((step) => step >= rough) ??
    10 * magnitude;
  const ticks: TimelineTick[] = [];
  // The epsilon keeps a tick that lands on the window end despite float error.
  for (let i = 0; i * spacing <= spanSeconds * (1 + 1e-9); i++) {
    const seconds = Number((i * spacing).toPrecision(12));
    ticks.push({ seconds, share: seconds / spanSeconds });
  }
  return ticks;
}

/** Tick text with only the decimals its spacing needs ("0 s", "0.5 s", "2.5 s"). */
export function tickLabel(tick: TimelineTick, ticks: readonly TimelineTick[]) {
  const spacing = ticks[1]?.seconds ?? 1;
  const decimals = Math.min(3, String(spacing).split(".")[1]?.length ?? 0);
  return `${tick.seconds.toFixed(decimals)} s`;
}

export type SavedPhase = "baseline" | "clean" | "contended";

export type SavedTimelineRow = {
  step: Step;
  phase: SavedPhase;
  /** Submission time as a share of the window. */
  startShare: number;
  segments: TimelineSegment[];
};

export type SavedTimeline = {
  /** Wall-clock submission of the first step, epoch nanoseconds. */
  originNs: number;
  spanSeconds: number;
  ticks: TimelineTick[];
  rows: SavedTimelineRow[];
};

export const SAVED_PHASE_LABELS: Record<SavedPhase, string> = {
  baseline: "Baseline repeat",
  clean: "Clean evaluation",
  contended: "Contended evaluation",
};

/**
 * The saved run's worker spans in submission order on one wall clock (the worker records
 * submit, start and finish with `time.time_ns()`). Queue is submit→start, execution is
 * start→finish, exactly as recorded.
 */
export function savedTimeline(run: Experiment): SavedTimeline {
  const phased: [Step, SavedPhase][] = [
    ...run.baseline.steps.map((step): [Step, SavedPhase] => [step, "baseline"]),
    [run.clean, "clean"],
    [run.contended, "contended"],
  ];
  phased.sort(([a], [b]) => a.submitted_ns - b.submitted_ns);
  const originNs = Math.min(...phased.map(([step]) => step.submitted_ns));
  const endNs = Math.max(...phased.map(([step]) => step.finished_ns));
  // A zero-length window would divide by zero; one nanosecond keeps every share finite.
  const spanNs = Math.max(endNs - originNs, 1);
  return {
    originNs,
    spanSeconds: spanNs / NS_PER_SECOND,
    ticks: timelineTicks(spanNs / NS_PER_SECOND),
    rows: phased.map(([step, phase]) => ({
      step,
      phase,
      startShare: (step.submitted_ns - originNs) / spanNs,
      segments: [
        { kind: "wait", share: (step.started_ns - step.submitted_ns) / spanNs },
        { kind: "work", share: (step.finished_ns - step.started_ns) / spanNs },
      ],
    })),
  };
}

/** Seconds from the timeline origin to an epoch-nanosecond wall-clock reading. */
export const secondsSinceOrigin = (timestampNs: number, originNs: number) =>
  (timestampNs - originNs) / NS_PER_SECOND;

export type WallWindow = { originMs: number; endMs: number };

/** The window spanned by live steps' producer wall-clock events; null with no steps. */
export function wallWindow(steps: readonly WallStep[]): WallWindow | null {
  if (!steps.length) return null;
  return {
    originMs: Math.min(...steps.map((step) => step.startMs)),
    endMs: Math.max(...steps.map((step) => step.endMs ?? step.startMs)),
  };
}

export const wallSpanSeconds = (window: WallWindow) =>
  Math.max((window.endMs - window.originMs) / MS_PER_SECOND, 0.001);

export const wallSecondsInto = (atMs: number, window: WallWindow) =>
  (atMs - window.originMs) / MS_PER_SECOND;

export type WallTimelineRow = {
  step: WallStep;
  startShare: number;
  /** One solid segment: the wall window covers the step's running time, not its queue. */
  segments: TimelineSegment[];
  /** Still running: the bar reaches the window end and has no duration yet. */
  open: boolean;
  /** Seconds from its running to its terminal event; null while open. */
  wallSeconds: number | null;
};

/** Live steps placed on their producer wall clock, in start order. */
export function wallTimeline(steps: readonly WallStep[]) {
  const window = wallWindow(steps);
  if (!window) return null;
  const span = wallSpanSeconds(window);
  const rows: WallTimelineRow[] = [...steps]
    .sort((a, b) => a.startMs - b.startMs)
    .map((step) => {
      const start = wallSecondsInto(step.startMs, window);
      const end =
        step.endMs === null ? span : wallSecondsInto(step.endMs, window);
      return {
        step,
        startShare: start / span,
        segments: [{ kind: "work", share: (end - start) / span }],
        open: step.endMs === null,
        wallSeconds: step.endMs === null ? null : end - start,
      };
    });
  return { window, spanSeconds: span, ticks: timelineTicks(span), rows };
}
