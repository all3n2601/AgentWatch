import {
  queueOf,
  executionOf,
  runLabel,
  runTime,
  stepsOf,
} from "./dashboard-data";
import type { Experiment } from "./experiment-schema";
import type { LiveRunSnapshot } from "./live-api-schema";
import { liveRunSummaries, type LiveRunSummary } from "./live-feed-data";
import type { LiveRunPage } from "./live-api-schema";

export type RunSource = "live" | "saved";
export type RunState =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "passed"
  | "inconclusive";
/** Coarser than a state, so one filter reads the same for live and saved runs. */
export type StateGroup = "in_progress" | "done" | "needs_attention";

/** One row of the Runs list, whichever store the run came from. Absent values stay null. */
export type UnifiedRun = {
  id: string;
  source: RunSource;
  title: string;
  kind: string;
  state: RunState;
  /** When the run entered the store (live) or finished (saved); never a start time. */
  at: string;
  steps: number | null;
  /** Queue share of profiled step time; only saved runs carry it in the list. */
  waitingShare: number | null;
  topResource: string | null;
};

const GROUPS: Record<RunState, StateGroup> = {
  queued: "in_progress",
  running: "in_progress",
  completed: "done",
  passed: "done",
  failed: "needs_attention",
  cancelled: "needs_attention",
  inconclusive: "needs_attention",
};
export const stateGroup = (state: RunState): StateGroup => GROUPS[state];

export function fromLive(run: LiveRunSummary): UnifiedRun {
  return {
    id: run.runId,
    source: "live",
    title: run.taskId,
    kind: "agent",
    state: run.state,
    at: run.createdAt,
    steps: null,
    waitingShare: null,
    topResource: null,
  };
}

export const fromLivePage = (page: LiveRunPage) =>
  liveRunSummaries(page).map(fromLive);

function topAttributedResource(run: Experiment): string | null {
  const counts = new Map<string, number>();
  for (const step of stepsOf(run))
    if (step.resource !== "none")
      counts.set(step.resource, (counts.get(step.resource) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
}

export function fromSaved(run: Experiment): UnifiedRun {
  const queue = queueOf(run);
  const execution = executionOf(run);
  return {
    id: run.run_id,
    source: "saved",
    title: runLabel(run),
    kind: run.agent ? "agent" : run.kind,
    state: run.checks.passed ? "passed" : "inconclusive",
    at: new Date(runTime(run)).toISOString(),
    steps: stepsOf(run).length,
    waitingShare: queue + execution > 0 ? queue / (queue + execution) : null,
    topResource: topAttributedResource(run),
  };
}

export type RunFilters = {
  source: RunSource | "all";
  group: StateGroup | "all";
  kind: string;
  query: string;
};

export function filterRuns(runs: UnifiedRun[], filters: RunFilters) {
  const needle = filters.query.trim().toLowerCase();
  return runs.filter(
    (run) =>
      (filters.source === "all" || run.source === filters.source) &&
      (filters.group === "all" || stateGroup(run.state) === filters.group) &&
      (filters.kind === "all" || run.kind === filters.kind) &&
      (needle === "" ||
        `${run.id} ${run.title} ${run.kind}`.toLowerCase().includes(needle)),
  );
}

/** Newest first; ties by id so paging never reorders rows between renders. */
export const byNewest = (a: UnifiedRun, b: UnifiedRun) =>
  b.at.localeCompare(a.at) || a.id.localeCompare(b.id);

export type DiagnosisStep = {
  stepId: string;
  type: string;
  state: string;
  durationSeconds: number | null;
  queueSeconds: number | null;
  executionSeconds: number | null;
  waitingShare: number | null;
  /** The resource class involved, and what that claim rests on. */
  resource: string | null;
  basis: "attributed" | "ran_on" | "none";
};

const bySlowestWaiting = (a: DiagnosisStep, b: DiagnosisStep) => {
  if (a.queueSeconds === null) return b.queueSeconds === null ? 0 : 1;
  if (b.queueSeconds === null) return -1;
  return b.queueSeconds - a.queueSeconds;
};

/** Saved-run steps carry a rule attribution recorded when the run was saved. */
export function diagnoseSaved(run: Experiment): DiagnosisStep[] {
  return stepsOf(run)
    .map((step): DiagnosisStep => ({
      stepId: step.step_id,
      type: "tool",
      state: "completed",
      durationSeconds: step.queue_seconds + step.execution_seconds,
      queueSeconds: step.queue_seconds,
      executionSeconds: step.execution_seconds,
      waitingShare: step.queue_fraction,
      resource: step.resource === "none" ? null : step.resource,
      basis: step.resource === "none" ? "none" : "attributed",
    }))
    .sort(bySlowestWaiting);
}

/**
 * Live steps record the resource class they ran on, which is not an attribution of waiting:
 * no per-step attribution is stored for live runs, so the basis says exactly that.
 */
export function diagnoseLive(snapshot: LiveRunSnapshot): DiagnosisStep[] {
  const classes = new Map(
    snapshot.registration.resources.map((resource) => [
      resource.resource_id,
      resource.resource_class,
    ]),
  );
  return snapshot.steps
    .map((view): DiagnosisStep => ({
      stepId: view.event.step_id,
      type: view.event.step_type,
      state: view.event.state,
      durationSeconds: view.timings.duration_seconds,
      queueSeconds: view.timings.queue_seconds,
      executionSeconds: view.timings.execution_seconds,
      waitingShare: view.timings.queue_fraction,
      resource: view.event.resource_id
        ? (classes.get(view.event.resource_id) ?? null)
        : null,
      basis: view.event.resource_id ? "ran_on" : "none",
    }))
    .sort(bySlowestWaiting);
}

/** One run for its own page, from whichever store holds it. */
export type RunView =
  | { source: "live"; snapshot: LiveRunSnapshot }
  | { source: "saved"; run: Experiment };

export const diagnoseView = (view: RunView) =>
  view.source === "live"
    ? diagnoseLive(view.snapshot)
    : diagnoseSaved(view.run);
