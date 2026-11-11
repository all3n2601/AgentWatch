import type { Manifest } from "./dataset-schema";
import type { DatasetRun } from "./dataset-runs";

export const LABEL_RESOURCES = [
  { id: "none", label: "No excess wait", color: "var(--series-slate)" },
  { id: "cpu_sandbox", label: "CPU sandbox", color: "var(--series-amber)" },
  { id: "inference", label: "Inference", color: "var(--series-violet)" },
  { id: "retrieval", label: "Retrieval", color: "var(--series-sky)" },
  { id: "mixed", label: "Mixed", color: "var(--series-rose)" },
] as const;

/** Display names for the manifest's machine values; unknown values keep their raw name. */
const NAMES: Record<string, string> = {
  agent_replay: "Agent replay",
  coding: "Coding tool",
  cpu: "CPU task",
  baseline: "Baseline",
  clean: "Clean control",
  contended: "Contended",
  present: "Samples overlap the step",
  no_overlap: "Samples missed the step",
  empty: "No samples",
  baseline_reference: "Baseline reference",
  clean_within_noise: "Clean, within noise",
  clean_exceeded_noise: "Clean, exceeded noise",
  contention_exceeded_threshold: "Contention, exceeded threshold",
  contention_below_threshold: "Contention, below threshold",
  mixed_confirmed: "Mixed, confirmed",
  mixed_partial: "Mixed, partly confirmed",
  mixed_joint_only: "Mixed, joint effect only",
  mixed_below_threshold: "Mixed, below threshold",
};
export const displayName = (value: string) =>
  NAMES[value] ?? value.replaceAll("_", " ");

/** A replay's condition: which contention, if any, was injected around the measured steps. */
const REPLAY_CONDITIONS: Record<string, string> = {
  baseline: "Baseline",
  clean: "Clean control",
  cpu: "CPU contention",
  inference: "Inference contention",
  retrieval: "Retrieval contention",
  mixed: "Mixed contention",
};
export const replayConditionName = (value: string) =>
  REPLAY_CONDITIONS[value] ?? displayName(value);

export const count = (n: number) => n.toLocaleString("en-US");
export const sharePercent = (part: number, whole: number) =>
  whole === 0
    ? "0%"
    : `${((part / whole) * 100).toFixed(part / whole < 0.1 ? 1 : 0)}%`;

export type Segment = { id: string; label: string; value: number };

/** Counts as segments, largest first, so a legend and its bar agree. */
export function segments(counts: Record<string, number>): Segment[] {
  return Object.entries(counts)
    .map(([id, value]) => ({ id, label: displayName(id), value }))
    .sort((a, b) => b.value - a.value);
}

/** Label counts in the fixed class order; classes absent from the counts stay at zero rows. */
export function labelSeries(counts: Record<string, number>) {
  return LABEL_RESOURCES.map((entry) => ({
    ...entry,
    value: counts[entry.id] ?? 0,
  }));
}

export type MissingColumn = { column: string; missing: number; share: number };

/**
 * Columns with at least one empty value, most-empty first. Empty means not recorded for that
 * row (many columns apply to one workload kind only); it is never charted as a measured zero.
 */
export function missingColumns(
  manifest: Manifest,
  limit = 12,
): MissingColumn[] {
  return Object.entries(manifest.missing)
    .filter(([, missing]) => missing > 0)
    .map(([column, missing]) => ({
      column,
      missing,
      share: missing / manifest.rows,
    }))
    .sort((a, b) => b.missing - a.missing || a.column.localeCompare(b.column))
    .slice(0, limit);
}

const CONDITION_ORDER = ["clean", "cpu", "retrieval", "inference", "mixed"];

export type ConditionTiming = {
  id: string;
  label: string;
  runs: number;
  medianSeconds: number;
};

const median = (sorted: number[]) => {
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Median total step time per replay condition, within the replay task with the most runs so
 * every bar compares the same workload. Observed medians, not a controlled comparison.
 */
export function medianStepTimeByCondition(runs: DatasetRun[]): {
  taskGroup: string;
  conditions: ConditionTiming[];
} | null {
  const replays = runs.filter((run) => run.replayCondition !== null);
  const sizes = new Map<string, number>();
  for (const run of replays)
    sizes.set(run.taskGroup, (sizes.get(run.taskGroup) ?? 0) + 1);
  const taskGroup = [...sizes.entries()].sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  )[0]?.[0];
  if (!taskGroup) return null;
  const conditions = CONDITION_ORDER.flatMap((id) => {
    const times = replays
      .filter(
        (run) => run.taskGroup === taskGroup && run.replayCondition === id,
      )
      .map((run) => run.durationSeconds)
      .sort((a, b) => a - b);
    return times.length
      ? [
          {
            id,
            label: replayConditionName(id),
            runs: times.length,
            medianSeconds: median(times),
          },
        ]
      : [];
  });
  return { taskGroup, conditions };
}
