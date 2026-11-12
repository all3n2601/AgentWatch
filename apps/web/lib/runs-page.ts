import { z } from "zod";
import { evidenceUrl, stepsOf } from "./dashboard-data";
import { experimentSchema, type Experiment } from "./experiment-schema";
import type { LiveRunSummary } from "./live-feed-data";
import {
  compareConfiguration,
  configurationOf,
  type Compatibility,
  type RecordedConfiguration,
} from "./run-compatibility";
import type { Tone } from "./tone";
import {
  filterRuns,
  fromLive,
  fromSaved,
  stateGroup,
  type RunFilters,
  type RunState,
  type UnifiedRun,
} from "./unified-runs";

/**
 * Data for the Runs page (`/runs`): one row per live or saved run, plus the
 * saved-run evidence the preview panel shows. Live runs carry no step timings
 * in the list API, so their evidence is null and the page says so.
 */

/** Longest `?q=` honoured; longer text is cut rather than rejected. */
export const MAX_QUERY_LENGTH = 200;
/** Rows per page of the list. */
export const RUNS_PAGE_SIZE = 25;

export type TelemetryCoverage = "full" | "limited" | "none";

/** What a saved run recorded, summarised for its row and preview. */
export type SavedRunEvidence = {
  workload: string;
  /** The agent model, or null for tool-only workloads. */
  model: string | null;
  /** False for a local artifact read because run history was unavailable. */
  persisted: boolean;
  stepCount: number;
  baselineSteps: number;
  /** Sums over every profiled step (baselines, clean and contended). */
  queueSeconds: number;
  executionSeconds: number;
  /** Steps the recorded rule attributed to a resource. */
  attributedSteps: number;
  ruleVersions: string[];
  /** Steps whose host telemetry reported status "available". */
  telemetrySteps: number;
  telemetry: TelemetryCoverage;
  cleanSeconds: number;
  contendedSeconds: number;
  contendedQueueSeconds: number;
  checks: { id: string; passed: boolean }[];
  workerCapacity: number;
  iterations: number;
  /** The settings the shared compatibility check compares (lib/run-compatibility.ts). */
  configuration: RecordedConfiguration;
  evidenceHref: string;
};

export type RunsPageRow = UnifiedRun & {
  /** Unique across both stores; live and saved ids are not guaranteed disjoint. */
  key: string;
  evidence: SavedRunEvidence | null;
};

const TELEMETRY_AVAILABLE = "available";

function telemetryCoverage(present: number, total: number): TelemetryCoverage {
  if (present === 0) return "none";
  return present === total ? "full" : "limited";
}

export function savedRow(run: Experiment): RunsPageRow {
  const steps = stepsOf(run);
  const telemetrySteps = steps.filter(
    (step) => step.telemetry?.status === TELEMETRY_AVAILABLE,
  ).length;
  return {
    ...fromSaved(run),
    key: `saved:${run.run_id}`,
    evidence: {
      workload: run.experiment.workload,
      model: run.agent?.model ?? null,
      persisted: run.persisted === true,
      stepCount: steps.length,
      baselineSteps: run.baseline.steps.length,
      queueSeconds: steps.reduce((sum, step) => sum + step.queue_seconds, 0),
      executionSeconds: steps.reduce(
        (sum, step) => sum + step.execution_seconds,
        0,
      ),
      attributedSteps: steps.filter((step) => step.resource !== "none").length,
      ruleVersions: [...new Set(steps.map((step) => step.rule_version))],
      telemetrySteps,
      telemetry: telemetryCoverage(telemetrySteps, steps.length),
      cleanSeconds: run.clean.duration_seconds,
      contendedSeconds: run.contended.duration_seconds,
      contendedQueueSeconds: run.contended.queue_seconds,
      checks: Object.entries(run.checks)
        .filter(([id]) => id !== "passed")
        .map(([id, passed]) => ({ id, passed })),
      workerCapacity: run.experiment.worker_capacity,
      iterations: run.experiment.iterations,
      configuration: configurationOf(run),
      evidenceHref: evidenceUrl(run),
    },
  };
}

export const liveRow = (run: LiveRunSummary): RunsPageRow => ({
  ...fromLive(run),
  key: `live:${run.runId}`,
  evidence: null,
});

/**
 * Contended over clean step duration for one saved run: an observed timing
 * difference between two single steps, not a causal measurement.
 */
export function contendedOverClean(evidence: SavedRunEvidence): number | null {
  return evidence.cleanSeconds > 0
    ? evidence.contendedSeconds / evidence.cleanSeconds
    : null;
}

export type RunsPageFilters = RunFilters & {
  /** A top attributed resource, "unattributed" for saved runs without one, or "all". */
  resource: string;
};

export const UNATTRIBUTED = "unattributed";

export const DEFAULT_FILTERS: RunsPageFilters = {
  source: "all",
  group: "all",
  kind: "all",
  resource: "all",
  query: "",
};

function matchesResource(row: RunsPageRow, resource: string) {
  if (resource === "all") return true;
  // Live rows carry no attribution in the list, so no resource filter holds for them.
  if (row.evidence === null) return false;
  return resource === UNATTRIBUTED
    ? row.topResource === null
    : row.topResource === resource;
}

function matchesQuery(row: RunsPageRow, needle: string) {
  if (needle === "") return true;
  const text = [
    row.id,
    row.title,
    row.kind,
    row.evidence?.workload,
    row.evidence?.model,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return text.includes(needle);
}

/**
 * Source, state and kind use the shared `filterRuns` rules; search also looks
 * at a saved run's workload and model, and the resource filter is page-local.
 */
export function filterRows(
  rows: readonly RunsPageRow[],
  filters: RunsPageFilters,
): RunsPageRow[] {
  const structural = new Set<UnifiedRun>(
    filterRuns([...rows], { ...filters, query: "" }),
  );
  const needle = filters.query.trim().toLowerCase();
  return rows.filter(
    (row) =>
      structural.has(row) &&
      matchesResource(row, filters.resource) &&
      matchesQuery(row, needle),
  );
}

/** Newest first; ties by key so paging never reorders rows between renders. */
export const byNewestRow = (a: RunsPageRow, b: RunsPageRow) =>
  b.at.localeCompare(a.at) || a.key.localeCompare(b.key);

/** Saved rows from an older history page, skipping runs already loaded. */
export function mergeOlderSaved(
  loaded: readonly RunsPageRow[],
  older: readonly RunsPageRow[],
): RunsPageRow[] {
  const keys = new Set(loaded.map((row) => row.key));
  return [...loaded, ...older.filter((row) => !keys.has(row.key))];
}

export type RunsSummary = {
  total: number;
  done: number;
  needsAttention: number;
  inProgress: number;
};

export function summarizeRows(rows: readonly RunsPageRow[]): RunsSummary {
  const summary = {
    total: rows.length,
    done: 0,
    needsAttention: 0,
    inProgress: 0,
  };
  for (const row of rows) {
    const group = stateGroup(row.state);
    if (group === "done") summary.done += 1;
    else if (group === "needs_attention") summary.needsAttention += 1;
    else summary.inProgress += 1;
  }
  return summary;
}

/** Distinct values for a filter, so the menu never offers an option with no runs. */
export function filterOptions(rows: readonly RunsPageRow[]) {
  const kinds = [...new Set(rows.map((row) => row.kind))].sort();
  const resources = [
    ...new Set(
      rows.flatMap((row) =>
        row.evidence && row.topResource ? [row.topResource] : [],
      ),
    ),
  ].sort();
  const unattributed = rows.some(
    (row) => row.evidence !== null && row.topResource === null,
  );
  return { kinds, resources, unattributed };
}

export const STATE_PRESENTATION: Record<
  RunState,
  { label: string; tone: Tone }
> = {
  queued: { label: "Queued", tone: "info" },
  running: { label: "Running", tone: "info" },
  completed: { label: "Completed", tone: "success" },
  passed: { label: "Passed", tone: "success" },
  failed: { label: "Failed", tone: "danger" },
  cancelled: { label: "Cancelled", tone: "warning" },
  inconclusive: { label: "Inconclusive", tone: "warning" },
};

export const TELEMETRY_PRESENTATION: Record<
  TelemetryCoverage,
  { label: string; tone: Tone }
> = {
  full: { label: "Full", tone: "success" },
  limited: { label: "Limited", tone: "warning" },
  none: { label: "None", tone: "caution" },
};

export type CompareSelection =
  | { status: "incomplete"; selected: number }
  | { status: "incompatible"; reason: string }
  | {
      status: "ready";
      /** The older run, so differences read as "candidate versus baseline". */
      baseline: RunsPageRow;
      candidate: RunsPageRow;
      /** The same check the Compare page applies to these two runs. */
      compatibility: Compatibility;
    };

/**
 * Two saved runs, older as baseline, checked with the Compare page's rule. A pair
 * whose settings differ can still be opened: Compare then lists the differing
 * settings and calculates no difference.
 */
export function compareSelection(
  selected: readonly RunsPageRow[],
): CompareSelection {
  if (selected.length !== 2)
    return { status: "incomplete", selected: selected.length };
  const [older, newer] = [...selected].sort(
    (a, b) => a.at.localeCompare(b.at) || a.key.localeCompare(b.key),
  );
  if (older.evidence === null || newer.evidence === null)
    return {
      status: "incompatible",
      reason:
        "Only saved runs can be compared; live runs have no saved timings.",
    };
  return {
    status: "ready",
    baseline: older,
    candidate: newer,
    compatibility: compareConfiguration(
      older.evidence.configuration,
      newer.evidence.configuration,
    ),
  };
}

/** Selecting a run for comparison, keeping at most two. */
export function toggleCompare(
  selected: readonly string[],
  key: string,
): string[] {
  if (selected.includes(key)) return selected.filter((item) => item !== key);
  return selected.length >= 2 ? [...selected] : [...selected, key];
}

export type RunsQuery = {
  query: string;
  /** Live-run cursor from `?liveBefore=`; null for the newest page. */
  liveBefore: number | null;
  liveBeforeError?: string;
};

const first = (value: string | string[] | undefined) =>
  Array.isArray(value) ? value[0] : value;

export function parseRunsQuery(
  params: Record<string, string | string[] | undefined>,
): RunsQuery {
  const query = (first(params.q) ?? "").slice(0, MAX_QUERY_LENGTH);
  const rawCursor = first(params.liveBefore);
  if (rawCursor === undefined || rawCursor === "")
    return { query, liveBefore: null };
  const cursor = Number(rawCursor);
  return Number.isSafeInteger(cursor) && cursor > 0
    ? { query, liveBefore: cursor }
    : { query, liveBefore: null, liveBeforeError: "Invalid live run cursor." };
}

/** Tabs of the Runs page, kept in `?view=`; the first is the default. */
export const RUNS_VIEWS = ["all", "compare"] as const;
export type RunsView = (typeof RUNS_VIEWS)[number];

/** The tab named in `?view=`; unknown or missing values open the run list. */
export function parseRunsView(
  params: Record<string, string | string[] | undefined>,
): RunsView {
  const view = first(params.view);
  return RUNS_VIEWS.find((option) => option === view) ?? RUNS_VIEWS[0];
}

export type LiveRunsResult =
  | { status: "ready"; rows: RunsPageRow[]; nextCursor: number | null }
  | { status: "unavailable"; reason: string };

export type RunsPageData = {
  saved: {
    rows: RunsPageRow[];
    /** "local" means run history was unreachable and local artifacts are shown. */
    source: "database" | "local";
    warnings: string[];
    nextCursor: number | null;
  };
  live: LiveRunsResult;
  /** The live-run cursor this page was read at; null for the newest page. */
  liveBefore: number | null;
};

const olderHistorySchema = z.object({
  experiments: z.array(
    experimentSchema.extend({
      kind: z.enum(["coding", "cpu"]),
      persisted: z.boolean().optional(),
      stored_at: z.string().optional(),
    }),
  ),
  nextCursor: z.number().nullable().optional(),
});

/**
 * Validates an older page from `/api/experiments?before=` before it reaches
 * the list. A malformed page is an error, never an empty result.
 */
export function parseOlderSaved(
  body: unknown,
):
  { ok: true; rows: RunsPageRow[]; nextCursor: number | null } | { ok: false } {
  const parsed = olderHistorySchema.safeParse(body);
  if (!parsed.success) return { ok: false };
  return {
    ok: true,
    rows: parsed.data.experiments.map(savedRow),
    nextCursor: parsed.data.nextCursor ?? null,
  };
}
