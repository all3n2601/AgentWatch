import {
  terminalStates,
  type LiveRunView,
  type LiveStep,
  type StepState,
} from "./step-events";

/** Server-owned run state; distinct from step states and unknown when null. */
export type RunStatus = { state: StepState; reason: string | null } | null;
export type RunOutcome =
  "active" | "completed" | "failed" | "cancelled" | "unknown";
export type FeedFreshness = "never" | "fresh" | "stale" | "closed";

/** How long an open run may go without a newly received record before it is stale. */
export const LIVE_FEED_STALE_AFTER_MS = 30_000;

const isTerminal = (step: LiveStep) => terminalStates.includes(step.state);
const plural = (count: number, noun: string) =>
  `${count} ${noun}${count === 1 ? "" : "s"}`;
const isAre = (count: number) => (count === 1 ? "is" : "are");

export const activeSteps = (view: LiveRunView) =>
  view.steps.filter((step) => !isTerminal(step));
export const failedSteps = (view: LiveRunView) =>
  view.steps.filter(
    (step) => step.state === "failed" || step.state === "cancelled",
  );

export function runOutcome(run: RunStatus): RunOutcome {
  if (!run) return "unknown";
  return run.state === "queued" || run.state === "running"
    ? "active"
    : run.state;
}

/**
 * Uses browser receipt times on both sides: producer `observed_at` is a
 * different clock and cannot measure how long the feed has been quiet.
 */
export function feedFreshness({
  lastReceivedAtMs,
  nowMs,
  outcome,
  staleAfterMs = LIVE_FEED_STALE_AFTER_MS,
}: {
  lastReceivedAtMs: number | null;
  nowMs: number;
  outcome: RunOutcome;
  staleAfterMs?: number;
}): FeedFreshness {
  if (
    outcome === "completed" ||
    outcome === "failed" ||
    outcome === "cancelled"
  )
    return "closed";
  if (lastReceivedAtMs === null) return "never";
  return nowMs - lastReceivedAtMs > staleAfterMs ? "stale" : "fresh";
}

export type TimingTotal = {
  seconds: number;
  measuredSteps: number;
  unmeasuredSteps: number;
};
export type CpuTotal = {
  /** Null unless every terminal step has measured CPU time in one scope. */
  seconds: number | null;
  scope: LiveStep["cpuScope"];
  measuredSteps: number;
  unmeasuredSteps: number;
};
export type StepTotals = {
  /** True only when the sums below are final run totals. */
  final: boolean;
  reasons: string[];
  queue: TimingTotal;
  execution: TimingTotal;
  duration: TimingTotal;
  cpu: CpuTotal;
};

function sumOf(values: (number | null)[]): TimingTotal {
  const measured = values.filter((v): v is number => v !== null);
  return {
    seconds: measured.reduce((total, v) => total + v, 0),
    measuredSteps: measured.length,
    unmeasuredSteps: values.length - measured.length,
  };
}

function cpuTotalOf(steps: LiveStep[]): CpuTotal {
  const measured = steps.filter((step) => step.cpuSeconds !== null);
  const scopes = new Set(measured.map((step) => step.cpuScope));
  const summable =
    measured.length > 0 &&
    measured.length === steps.length &&
    scopes.size === 1;
  return {
    seconds: summable
      ? measured.reduce((total, step) => total + step.cpuSeconds!, 0)
      : null,
    scope: summable ? measured[0].cpuScope : null,
    measuredSteps: measured.length,
    unmeasuredSteps: steps.length - measured.length,
  };
}

/**
 * Sums the timings of terminal steps. The sums are partial unless the run is
 * closed, every reported step is terminal, and no revision or record is missing.
 * Per-step durations are summed (each is within one clock domain); ticks from
 * different steps are never subtracted from each other.
 */
export function stepTotals(view: LiveRunView, run: RunStatus): StepTotals {
  const outcome = runOutcome(run);
  const unfinished = activeSteps(view).length;
  const reasons = [
    outcome === "unknown" && "Run state has not been reported",
    outcome === "active" && "Run is still active",
    view.steps.length === 0 &&
      view.gaps.length === 0 &&
      "No steps have been reported",
    unfinished > 0 &&
      `${plural(unfinished, "step")} ${isAre(unfinished)} still queued or running`,
    view.gaps.length > 0 &&
      `${plural(view.gaps.length, "step")} ${isAre(view.gaps.length)} waiting for a missing revision`,
    view.unreadableRecords > 0 &&
      `${plural(view.unreadableRecords, "record")} could not be read`,
  ].filter((reason): reason is string => !!reason);
  const terminal = view.steps.filter(isTerminal);
  return {
    final: reasons.length === 0,
    reasons,
    queue: sumOf(terminal.map((s) => s.timings?.queueSeconds ?? null)),
    execution: sumOf(terminal.map((s) => s.timings?.executionSeconds ?? null)),
    duration: sumOf(terminal.map((s) => s.timings?.durationSeconds ?? null)),
    cpu: cpuTotalOf(terminal),
  };
}
