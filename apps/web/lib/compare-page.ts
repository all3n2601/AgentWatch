import {
  duration,
  evidenceUrl,
  runLabel,
  scopedRuns,
  stepsOf,
} from "./dashboard-data";
import type { DashboardData, Experiment, Step } from "./experiment-schema";
import { formatQuantity } from "./measurement-format";
import type { Route } from "next";
import { routes } from "./routes";
import {
  compareConfiguration,
  configurationOf,
  policyUnknown,
  POLICY_NOT_RECORDED_WARNING,
  type Compatibility,
} from "./run-compatibility";

/**
 * Pure model for the Compare page (`/compare?run=&baseline=`): which saved runs
 * were requested, whether their configurations agree (lib/run-compatibility.ts),
 * and the observed timing differences between them. Every difference is
 * candidate minus baseline and every ratio is candidate / baseline; neither is a
 * causal claim. When a recorded setting differs, no difference is calculated.
 */

export type CompareRole = "candidate" | "baseline";

/** The Compare page's tabs, in display order; the first is the default. */
export const COMPARE_TABS = ["summary", "configuration", "steps"] as const;
export type CompareTab = (typeof COMPARE_TABS)[number];
/** Query key holding the open tab (TabbedView's default `view`). */
export const COMPARE_TAB_PARAM = "view";

/** The tab a `?view=` value names, or undefined for anything else. */
export function compareTab(
  value: string | null | undefined,
): CompareTab | undefined {
  return COMPARE_TABS.find((tab) => tab === value);
}

/**
 * `/compare?run=&baseline=` that keeps the open tab, so choosing or swapping
 * runs does not jump back to the summary. Only a known, non-default tab is kept.
 */
export function compareHref(
  run: string | undefined,
  baseline: string | undefined,
  view: string | null | undefined,
): Route {
  const href = routes.compare(run, baseline);
  const tab = compareTab(view);
  if (!tab || tab === COMPARE_TABS[0]) return href;
  const separator = href.includes("?") ? "&" : "?";
  // A computed href, typed as routes.ts does for its own builders.
  return `${href}${separator}${COMPARE_TAB_PARAM}=${tab}` as Route;
}

export type RunOption = {
  id: string;
  /** "coding · qwen3:1.7b · 67991452 · 2026-11-11 00:20 UTC" */
  label: string;
};

/** What the page knows about one requested slot. */
export type RunSlot =
  | { status: "unselected" }
  /** An id was requested but no loaded run has it (invalid or unknown). */
  | { status: "not-found"; requestedId: string }
  | { status: "selected"; run: RunSummary };

export type RunSummary = {
  id: string;
  kind: Experiment["kind"];
  label: string;
  /** Timezone-aware save time, when the run came from saved history. */
  storedAt: string | null;
  persisted: boolean;
  passed: boolean;
  /** Evidence export for this run; local-fallback runs export the local artifact. */
  evidenceHref: string;
};

/** One observed timing, candidate against baseline. */
export type TimingDelta = {
  baselineSeconds: number | null;
  candidateSeconds: number | null;
  /** Candidate minus baseline; null when either side is missing. */
  deltaSeconds: number | null;
  /** Candidate / baseline; null when either side is missing or the baseline is 0. */
  ratio: number | null;
};

export type MetricDelta = TimingDelta & {
  id: string;
  label: string;
  /** What the number covers ("7 steps", "clean step"). */
  scope: string;
};

export type StepTiming = {
  durationSeconds: number;
  queueSeconds: number;
  executionSeconds: number;
  cpuSeconds: number;
  queueFraction: number;
};

export type AlignedStep = {
  stepId: string;
  label: string;
  baseline: StepTiming | null;
  candidate: StepTiming | null;
  duration: TimingDelta;
};

/** The observed total-time difference, stated without better or worse. */
export type Headline = {
  title: string;
  deltaSeconds: number;
  ratio: number | null;
};

/** A quality or coverage reading per run, shown side by side and never differenced. */
export type QualityRow = {
  id: string;
  label: string;
  baseline: string;
  candidate: string;
};

export type Comparison = {
  candidate: RunSummary;
  baseline: RunSummary;
  compatibility: Compatibility;
  /** False when a recorded setting differs: deltas and ratios are then null. */
  differencesShown: boolean;
  metrics: MetricDelta[];
  steps: AlignedStep[];
  largestStepDifferences: AlignedStep[];
  headline: Headline | null;
  quality: QualityRow[];
  interpretation: string;
  /** Recorded run limitations, deduplicated, in first-seen order. */
  limitations: string[];
};

export type ComparePageModel = {
  source: DashboardData["source"];
  warnings: string[];
  loadedRuns: number;
  candidateOptions: RunOption[];
  baselineOptions: RunOption[];
  candidate: RunSlot;
  baseline: RunSlot;
  /** Both ids name the same run. */
  sameRun: boolean;
  comparison: Comparison | null;
};

const LARGEST_DIFFERENCES_SHOWN = 3;
const SHORT_ID_LENGTH = 8;

/** Why throughput and tail latency are not shown for saved runs. */
export const SINGLE_RUN_LIMITATION =
  "Throughput and p95 task latency are not recorded: a saved run is one task, so it has no batch throughput or per-task tail latency.";

/** "2026-11-11 00:20 UTC"; formatted in UTC so server and client agree. */
export function formatStoredAt(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const time = Date.parse(iso);
  if (!Number.isFinite(time)) return null;
  return `${new Date(time).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

export function summarizeRun(run: Experiment): RunSummary {
  return {
    id: run.run_id,
    kind: run.kind,
    label: runLabel(run),
    storedAt: run.stored_at ?? null,
    persisted: run.persisted === true,
    passed: run.checks.passed,
    evidenceHref: evidenceUrl(run),
  };
}

export function runOptionLabel(run: Experiment): string {
  return [
    run.kind,
    runLabel(run),
    run.run_id.slice(0, SHORT_ID_LENGTH),
    formatStoredAt(run.stored_at) ?? "local artifact",
  ].join(" · ");
}

/** "Baseline repeat 1", "Clean step", "Contended step"; unknown ids unchanged. */
export function stepLabel(stepId: string): string {
  if (stepId === "clean") return "Clean step";
  if (stepId === "contended") return "Contended step";
  const repeat = /^baseline-(\d+)$/.exec(stepId);
  if (repeat) return `Baseline repeat ${Number(repeat[1]) + 1}`;
  return stepId;
}

const isFiniteNumber = (value: number | null): value is number =>
  value !== null && Number.isFinite(value);

export function timingDelta(
  baselineSeconds: number | null,
  candidateSeconds: number | null,
): TimingDelta {
  const baseline = isFiniteNumber(baselineSeconds) ? baselineSeconds : null;
  const candidate = isFiniteNumber(candidateSeconds) ? candidateSeconds : null;
  if (baseline === null || candidate === null)
    return {
      baselineSeconds: baseline,
      candidateSeconds: candidate,
      deltaSeconds: null,
      ratio: null,
    };
  return {
    baselineSeconds: baseline,
    candidateSeconds: candidate,
    deltaSeconds: candidate - baseline,
    ratio: baseline > 0 ? candidate / baseline : null,
  };
}

const sum = (steps: Step[], pick: (step: Step) => number) =>
  steps.reduce((total, step) => total + pick(step), 0);

/** Model inference wall time; null without agent evidence (never 0). */
function inferenceSeconds(run: Experiment): number | null {
  if (!run.agent) return null;
  return run.agent.inference.reduce(
    (total, call) => total + call.wall_seconds,
    0,
  );
}

const stepCount = (baseline: Experiment, candidate: Experiment) => {
  const a = stepsOf(baseline).length;
  const b = stepsOf(candidate).length;
  return a === b ? `${a} steps` : `${a} vs ${b} steps`;
};

/**
 * Totals first, then the clean and contended steps' split, then inference.
 * Inference is listed only when at least one run recorded agent evidence; the
 * side without it stays unavailable rather than 0 s.
 */
export function metricDeltas(
  baseline: Experiment,
  candidate: Experiment,
): MetricDelta[] {
  const all = stepCount(baseline, candidate);
  const metric = (
    id: string,
    label: string,
    scope: string,
    pick: (run: Experiment) => number | null,
  ): MetricDelta => ({
    id,
    label,
    scope,
    ...timingDelta(pick(baseline), pick(candidate)),
  });
  const metrics = [
    metric("total", "Total step time", all, (run) =>
      sum(stepsOf(run), (s) => s.duration_seconds),
    ),
    metric("queue", "Worker queue delay", all, (run) =>
      sum(stepsOf(run), (s) => s.queue_seconds),
    ),
    metric("execution", "Execution wall time", all, (run) =>
      sum(stepsOf(run), (s) => s.execution_seconds),
    ),
    metric("cpu", "Process CPU time", all, (run) =>
      sum(stepsOf(run), (s) => s.cpu_seconds),
    ),
    metric(
      "clean-queue",
      "Clean queue",
      "clean step",
      (run) => run.clean.queue_seconds,
    ),
    metric(
      "clean-execution",
      "Clean execution",
      "clean step",
      (run) => run.clean.execution_seconds,
    ),
    metric(
      "contended-queue",
      "Contended queue",
      "contended step",
      (run) => run.contended.queue_seconds,
    ),
    metric(
      "contended-execution",
      "Contended execution",
      "contended step",
      (run) => run.contended.execution_seconds,
    ),
    metric(
      "inference",
      "Model inference wall time",
      "agent model calls",
      inferenceSeconds,
    ),
  ];
  return baseline.agent || candidate.agent
    ? metrics
    : metrics.filter((m) => m.id !== "inference");
}

const stepTiming = (step: Step): StepTiming => ({
  durationSeconds: step.duration_seconds,
  queueSeconds: step.queue_seconds,
  executionSeconds: step.execution_seconds,
  cpuSeconds: step.cpu_seconds,
  queueFraction: step.queue_fraction,
});

/**
 * Pairs steps by `step_id` in baseline order, then appends candidate-only
 * steps. A step recorded on one side only keeps a null for the other, so a
 * missing step is never compared as zero.
 */
export function alignSteps(
  baseline: Experiment,
  candidate: Experiment,
): AlignedStep[] {
  const baseSteps = stepsOf(baseline);
  const candSteps = stepsOf(candidate);
  const byId = (steps: Step[]) => new Map(steps.map((s) => [s.step_id, s]));
  const baseById = byId(baseSteps);
  const candById = byId(candSteps);
  const ids = [
    ...new Set([
      ...baseSteps.map((s) => s.step_id),
      ...candSteps.map((s) => s.step_id),
    ]),
  ];
  return ids.map((stepId) => {
    const base = baseById.get(stepId);
    const cand = candById.get(stepId);
    return {
      stepId,
      label: stepLabel(stepId),
      baseline: base ? stepTiming(base) : null,
      candidate: cand ? stepTiming(cand) : null,
      duration: timingDelta(
        base?.duration_seconds ?? null,
        cand?.duration_seconds ?? null,
      ),
    };
  });
}

/** Steps present on both sides, ordered by absolute duration difference. */
export function largestDifferences(
  steps: readonly AlignedStep[],
  limit = LARGEST_DIFFERENCES_SHOWN,
): AlignedStep[] {
  const magnitude = (step: AlignedStep) =>
    Math.abs(step.duration.deltaSeconds ?? 0);
  return steps
    .filter((step) => step.duration.deltaSeconds !== null)
    .sort((a, b) => magnitude(b) - magnitude(a))
    .slice(0, limit);
}

/** "+24.85 s", "−840 μs", "±0 ms", with the dashboard's duration precision. */
export const signedDuration = (seconds: number) =>
  `${seconds > 0 ? "+" : seconds < 0 ? "−" : "±"}${duration(Math.abs(seconds))}`;

const deltaPhrase = (metric: MetricDelta) =>
  metric.deltaSeconds === null
    ? "unavailable"
    : `${signedDuration(metric.deltaSeconds)}${metric.ratio === null ? "" : ` (${formatQuantity(metric.ratio, "ratio")})`}`;

/** The observed total difference; null without a total measured on both runs. */
export function headline(total: MetricDelta): Headline | null {
  if (total.deltaSeconds === null) return null;
  const ratio = total.ratio !== null && total.ratio > 0 ? total.ratio : null;
  return {
    title: `Total step time ${signedDuration(total.deltaSeconds)}${ratio === null ? "" : ` (${formatQuantity(ratio, "ratio")} of baseline)`}`,
    deltaSeconds: total.deltaSeconds,
    ratio,
  };
}

const telemetryText = (step: Step) =>
  step.telemetry
    ? `${formatQuantity(step.telemetry.sample_count, "count")} samples · ${formatQuantity(step.telemetry.valid_pressure_samples, "count")} valid pressure readings (${step.telemetry.status})`
    : "No telemetry recorded";

/**
 * Quality and coverage of each run. Throughput and p95 task latency need a
 * batch of tasks, which saved runs do not record, so they are not derived.
 */
export function qualityRows(
  baseline: Experiment,
  candidate: Experiment,
): QualityRow[] {
  const row = (
    id: string,
    label: string,
    text: (run: Experiment) => string,
  ): QualityRow => ({
    id,
    label,
    baseline: text(baseline),
    candidate: text(candidate),
  });
  return [
    row("throughput", "Throughput", () => "Not recorded"),
    row("p95", "p95 task latency", () => "Not recorded"),
    row("checks", "Acceptance checks", (run) =>
      run.checks.passed ? "Passed" : "Failed",
    ),
    row(
      "agent",
      "Agent outcome",
      (run) => run.agent?.status ?? "No agent evidence",
    ),
    row("clean-telemetry", "Clean step telemetry", (run) =>
      telemetryText(run.clean),
    ),
    row("contended-telemetry", "Contended step telemetry", (run) =>
      telemetryText(run.contended),
    ),
  ];
}

/** What was observed, never why, and what the configuration check allows. */
export function interpretation(
  compatibility: Compatibility,
  metrics: readonly MetricDelta[],
): string {
  if (compatibility.verdict === "differs")
    return `Recorded settings differ (${compatibility.differing.join(", ").toLowerCase()}), so no difference or ratio is calculated. Each run's own values are listed side by side.`;
  const total = metrics.find((m) => m.id === "total");
  const queue = metrics.find((m) => m.id === "queue");
  const observed = `Observed total step time ${total ? deltaPhrase(total) : "unavailable"} and worker queue delay ${queue ? deltaPhrase(queue) : "unavailable"}, candidate minus baseline.`;
  const caveat =
    "These are two uncontrolled runs: the difference is observed, not causal. Worker queue delay reflects worker-capacity contention, not host CPU saturation.";
  if (compatibility.verdict === "unverified") {
    const unknown = policyUnknown(compatibility)
      ? `${POLICY_NOT_RECORDED_WARNING}.`
      : `Not recorded on both runs: ${compatibility.unknown.join(", ").toLowerCase()}; full compatibility cannot be verified.`;
    return `${observed} Every recorded setting matches. ${unknown} ${caveat}`;
  }
  return `${observed} The recorded settings match. ${caveat}`;
}

const withoutDifference = <T extends TimingDelta>(delta: T): T => ({
  ...delta,
  deltaSeconds: null,
  ratio: null,
});

export function buildComparison(
  baseline: Experiment,
  candidate: Experiment,
): Comparison {
  const compatibility = compareConfiguration(
    configurationOf(baseline),
    configurationOf(candidate),
  );
  // A recorded setting that differs makes any difference cross-configuration,
  // so none is calculated; each run's values stay visible.
  const differencesShown = compatibility.verdict !== "differs";
  const measured = metricDeltas(baseline, candidate);
  const metrics = differencesShown ? measured : measured.map(withoutDifference);
  const aligned = alignSteps(baseline, candidate);
  const steps = differencesShown
    ? aligned
    : aligned.map((step) => ({
        ...step,
        duration: withoutDifference(step.duration),
      }));
  const total = metrics.find((m) => m.id === "total");
  return {
    candidate: summarizeRun(candidate),
    baseline: summarizeRun(baseline),
    compatibility,
    differencesShown,
    metrics,
    steps,
    largestStepDifferences: differencesShown ? largestDifferences(steps) : [],
    headline: differencesShown && total ? headline(total) : null,
    quality: qualityRows(baseline, candidate),
    interpretation: interpretation(compatibility, measured),
    limitations: [
      ...new Set([
        ...candidate.limitations,
        ...baseline.limitations,
        SINGLE_RUN_LIMITATION,
      ]),
    ],
  };
}

function resolveSlot(
  runs: Experiment[],
  requestedId: string | undefined,
): {
  slot: RunSlot;
  run?: Experiment;
} {
  const id = requestedId?.trim();
  if (!id) return { slot: { status: "unselected" } };
  // Only the exact requested id counts; another run is never substituted.
  const run = runs.find((item) => item.run_id === id);
  return run
    ? { slot: { status: "selected", run: summarizeRun(run) }, run }
    : { slot: { status: "not-found", requestedId: id } };
}

/** The whole page model from loaded history and the two requested ids. */
export function buildComparePage(
  data: DashboardData,
  requested: { run?: string; baseline?: string },
): ComparePageModel {
  const runs = scopedRuns(data.experiments, "all");
  const candidate = resolveSlot(runs, requested.run);
  const baseline = resolveSlot(runs, requested.baseline);
  const sameRun = candidate.run !== undefined && candidate.run === baseline.run;
  const options = (exclude?: Experiment) =>
    runs
      .filter((run) => run !== exclude)
      .map((run) => ({ id: run.run_id, label: runOptionLabel(run) }));
  return {
    source: data.source,
    warnings: data.warnings,
    loadedRuns: runs.length,
    candidateOptions: options(sameRun ? undefined : baseline.run),
    baselineOptions: options(sameRun ? undefined : candidate.run),
    candidate: candidate.slot,
    baseline: baseline.slot,
    sameRun,
    comparison:
      candidate.run && baseline.run && !sameRun
        ? buildComparison(baseline.run, candidate.run)
        : null,
  };
}
