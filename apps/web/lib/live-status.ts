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
 * A record receipt anchored to one browser's monotonic clock: the record's age
 * when the anchor was taken. Server and producer wall clocks never enter it, so
 * clock offsets between machines cannot distort feed age.
 */
export type FeedReceipt = { ageAtAnchorMs: number; anchorMonotonicMs: number };

/** Age of the latest received record at a later reading of the same clock. */
export function receiptAgeMs(
  receipt: FeedReceipt | null,
  nowMonotonicMs: number,
): number | null {
  if (!receipt) return null;
  return (
    receipt.ageAtAnchorMs +
    Math.max(0, nowMonotonicMs - receipt.anchorMonotonicMs)
  );
}

/** Wall time of receipt on the viewer's clock, derived from the same age. */
export const receivedAtWallMs = (ageMs: number, browserWallNowMs: number) =>
  browserWallNowMs - ageMs;

const SECOND_MS = 1_000;
const MINUTE_MS = 60 * SECOND_MS;
const HOUR_MS = 60 * MINUTE_MS;
const JUST_NOW_MS = 5 * SECOND_MS;

export function formatReceiptAge(ageMs: number) {
  if (ageMs < JUST_NOW_MS) return "Just now";
  if (ageMs < MINUTE_MS) return `${Math.floor(ageMs / SECOND_MS)} s ago`;
  if (ageMs < HOUR_MS) return `${Math.floor(ageMs / MINUTE_MS)} min ago`;
  return `${Math.floor(ageMs / HOUR_MS)} h ago`;
}

export function feedFreshness({
  ageMs,
  outcome,
  staleAfterMs = LIVE_FEED_STALE_AFTER_MS,
}: {
  ageMs: number | null;
  outcome: RunOutcome;
  staleAfterMs?: number;
}): FeedFreshness {
  if (
    outcome === "completed" ||
    outcome === "failed" ||
    outcome === "cancelled"
  )
    return "closed";
  if (ageMs === null) return "never";
  return ageMs > staleAfterMs ? "stale" : "fresh";
}

/** One reading of the viewer's clocks; null until the browser has mounted. */
export type BrowserClockReading = {
  anchorMonotonicMs: number;
  nowMonotonicMs: number;
  nowWallMs: number;
} | null;

/**
 * The display clock must keep ticking whenever a receipt age is shown, even
 * for a closed run whose freshness no longer changes.
 */
export const feedClockTicks = (receivedAgoMs: number | null) =>
  receivedAgoMs !== null;

export type FeedDisplay = {
  freshness: FeedFreshness;
  ageMs: number | null;
  ageText: string | null;
  /** Receipt time on the viewer's wall clock; null before mount or receipt. */
  receivedAtWallMs: number | null;
};

/**
 * Derives freshness and the "last record received" text from one receipt age.
 * `receivedAgoMs` is the age when the browser received the data, or null when
 * nothing has been received.
 */
export function feedDisplay({
  receivedAgoMs,
  outcome,
  clock,
}: {
  receivedAgoMs: number | null;
  outcome: RunOutcome;
  clock: BrowserClockReading;
}): FeedDisplay {
  const ageMs =
    receivedAgoMs === null
      ? null
      : clock
        ? receiptAgeMs(
            {
              ageAtAnchorMs: receivedAgoMs,
              anchorMonotonicMs: clock.anchorMonotonicMs,
            },
            clock.nowMonotonicMs,
          )
        : receivedAgoMs;
  return {
    freshness: feedFreshness({ ageMs, outcome }),
    ageMs,
    ageText: ageMs === null ? null : formatReceiptAge(ageMs),
    receivedAtWallMs:
      ageMs === null || !clock
        ? null
        : receivedAtWallMs(ageMs, clock.nowWallMs),
  };
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
