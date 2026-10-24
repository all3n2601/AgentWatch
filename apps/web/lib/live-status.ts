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

const NS_PER_SECOND = 1e9;

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
export type SpanTotal = {
  /**
   * Wall time covered by at least one terminal step, so overlapping parent and child windows
   * count once. Null when the steps use more than one clock domain.
   */
  seconds: number | null;
  measuredSteps: number;
  clockDomains: number;
};
export type StepTotals = {
  /** True only when the sums below are final run totals. */
  final: boolean;
  reasons: string[];
  /** Queue, execution and CPU sums cover leaf steps; a parent's window contains its children. */
  queue: TimingTotal;
  execution: TimingTotal;
  cpu: CpuTotal;
  /** Terminal parent steps left out of the sums because their children are counted. */
  parentSteps: number;
  duration: SpanTotal;
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

/** Length of the union of [submitted, finished] windows; every step is in one clock domain. */
function coveredSeconds(steps: LiveStep[]) {
  const windows = steps
    .flatMap((step) =>
      step.finishedTickNs === null
        ? []
        : [[step.submittedTickNs, step.finishedTickNs] as const],
    )
    .sort((a, b) => a[0] - b[0]);
  let coveredNs = 0;
  let open: [number, number] | null = null;
  for (const [start, end] of windows) {
    if (open && start <= open[1]) open[1] = Math.max(open[1], end);
    else {
      if (open) coveredNs += open[1] - open[0];
      open = [start, end];
    }
  }
  if (open) coveredNs += open[1] - open[0];
  return coveredNs / NS_PER_SECOND;
}

function spanOf(steps: LiveStep[]): SpanTotal {
  const clockDomains = new Set(steps.map((step) => step.clockId)).size;
  return {
    seconds: clockDomains > 1 ? null : coveredSeconds(steps),
    measuredSteps: steps.length,
    clockDomains,
  };
}

/**
 * Totals the timings of terminal steps. The totals are partial unless the run is
 * closed, every reported step is terminal, and no revision or record is missing.
 * Queue, execution and CPU time are summed over leaf steps only, because a
 * parent's window already contains its children's. The duration is the wall
 * time covered by any step, compared only within one clock domain.
 */
export function stepTotals(view: LiveRunView, run: RunStatus): StepTotals {
  const outcome = runOutcome(run);
  const unfinished = activeSteps(view).length;
  const expired = view.gaps.filter((gap) => gap.expired).length;
  const waiting = view.gaps.length - expired;
  const reasons = [
    outcome === "unknown" && "Run state has not been reported",
    outcome === "active" && "Run is still active",
    view.steps.length === 0 &&
      view.gaps.length === 0 &&
      "No steps have been reported",
    unfinished > 0 &&
      `${plural(unfinished, "step")} ${isAre(unfinished)} still queued or running`,
    waiting > 0 &&
      `${plural(waiting, "step")} ${isAre(waiting)} waiting for a missing revision`,
    expired > 0 &&
      `${plural(expired, "step")} ${expired === 1 ? "has" : "have"} expired revisions that will never be applied`,
    view.unreadableRecords > 0 &&
      `${plural(view.unreadableRecords, "record")} could not be read`,
  ].filter((reason): reason is string => !!reason);
  const terminal = view.steps.filter(isTerminal);
  const parents = new Set(view.steps.map((step) => step.parentStepId));
  const leaves = terminal.filter((step) => !parents.has(step.stepId));
  return {
    final: reasons.length === 0,
    reasons,
    queue: sumOf(leaves.map((s) => s.timings?.queueSeconds ?? null)),
    execution: sumOf(leaves.map((s) => s.timings?.executionSeconds ?? null)),
    cpu: cpuTotalOf(leaves),
    parentSteps: terminal.length - leaves.length,
    duration: spanOf(terminal),
  };
}
