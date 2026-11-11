import { z } from "zod";

/** Parquet int64 columns arrive as bigint; the dataset's counts are far below 2^53. */
const count = z.preprocess(
  (value) => (typeof value === "bigint" ? Number(value) : value),
  z.number().int().nonnegative(),
);
const seconds = z.number().finite().nonnegative();

/** The `steps.parquet` columns the run views read. */
export const DATASET_COLUMNS = [
  "run_id",
  "step_id",
  "kind",
  "workload",
  "task_group",
  "experiment",
  "condition",
  "replay_condition",
  "replay_index",
  "step_type",
  "agent_model",
  "injection_method",
  "recorded_at",
  "duration_seconds",
  "queue_seconds",
  "execution_seconds",
  "cpu_seconds",
  "label_resource",
  "label_excess_fraction",
  "telemetry_status",
  "run_passed",
] as const;

export const stepRowSchema = z.object({
  run_id: z.string().uuid(),
  step_id: z.string(),
  kind: z.string(),
  workload: z.string(),
  task_group: z.string(),
  experiment: z.string().nullable(),
  condition: z.string(),
  replay_condition: z.string().nullable(),
  replay_index: count.nullable(),
  step_type: z.string().nullable(),
  agent_model: z.string().nullable(),
  injection_method: z.string(),
  recorded_at: z.date().nullable(),
  duration_seconds: seconds,
  queue_seconds: seconds,
  execution_seconds: seconds,
  cpu_seconds: seconds.nullable(),
  label_resource: z.string(),
  label_excess_fraction: z.number().finite().nullable(),
  telemetry_status: z.string(),
  run_passed: z.boolean().nullable(),
});
export type StepRow = z.infer<typeof stepRowSchema>;

/** One run of the release: a replay of an agent task, or a coding/CPU experiment. */
export type DatasetRun = {
  runId: string;
  kind: string;
  workload: string;
  taskGroup: string;
  experiment: string | null;
  replayCondition: string | null;
  replayIndex: number | null;
  injectionMethod: string;
  agentModel: string | null;
  recordedAt: string | null;
  passed: boolean | null;
  steps: number;
  durationSeconds: number;
  queueSeconds: number;
  executionSeconds: number;
  /** Steps whose label names a resource, i.e. anything other than "none". */
  excessSteps: number;
  labels: Record<string, number>;
  /** Steps per call type (`model_call`, `tool`, `retrieval`); steps without one are not counted. */
  stepTypes: Record<string, number>;
  /** Steps per call type whose label names a resource (excess wait), like `excessSteps`. */
  excessByStepType: Record<string, number>;
  /** The most frequent non-"none" label; "none" when no step shows excess wait. */
  topLabel: string;
  telemetryPresent: number;
};

export function parseStepRows(rows: unknown[]): StepRow[] {
  return rows.map((row) => stepRowSchema.parse(row));
}

const NO_EXCESS = "none";

function topLabel(labels: Record<string, number>) {
  const ranked = Object.entries(labels)
    .filter(([label]) => label !== NO_EXCESS)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return ranked[0]?.[0] ?? NO_EXCESS;
}

/** Newest first; runs without a recorded time (coding/CPU experiments) follow, by id. */
const byRecency = (a: DatasetRun, b: DatasetRun) =>
  (b.recordedAt ?? "").localeCompare(a.recordedAt ?? "") ||
  a.runId.localeCompare(b.runId);

export function summarizeRuns(rows: StepRow[]): DatasetRun[] {
  const runs = new Map<string, DatasetRun>();
  for (const row of rows) {
    let run = runs.get(row.run_id);
    if (!run) {
      run = {
        runId: row.run_id,
        kind: row.kind,
        workload: row.workload,
        taskGroup: row.task_group,
        experiment: row.experiment,
        replayCondition: row.replay_condition,
        replayIndex: row.replay_index,
        injectionMethod: row.injection_method,
        agentModel: row.agent_model,
        recordedAt: row.recorded_at?.toISOString() ?? null,
        passed: row.run_passed,
        steps: 0,
        durationSeconds: 0,
        queueSeconds: 0,
        executionSeconds: 0,
        excessSteps: 0,
        labels: {},
        stepTypes: {},
        excessByStepType: {},
        topLabel: NO_EXCESS,
        telemetryPresent: 0,
      };
      runs.set(row.run_id, run);
    }
    run.steps += 1;
    run.durationSeconds += row.duration_seconds;
    run.queueSeconds += row.queue_seconds;
    run.executionSeconds += row.execution_seconds;
    run.labels[row.label_resource] = (run.labels[row.label_resource] ?? 0) + 1;
    if (row.step_type !== null) {
      run.stepTypes[row.step_type] = (run.stepTypes[row.step_type] ?? 0) + 1;
      if (row.label_resource !== NO_EXCESS)
        run.excessByStepType[row.step_type] =
          (run.excessByStepType[row.step_type] ?? 0) + 1;
    }
    if (row.label_resource !== NO_EXCESS) run.excessSteps += 1;
    if (row.telemetry_status === "present") run.telemetryPresent += 1;
    // A run's time is its first recorded step; steps of one run are recorded together.
    if (row.recorded_at) {
      const iso = row.recorded_at.toISOString();
      if (!run.recordedAt || iso < run.recordedAt) run.recordedAt = iso;
    }
  }
  const summaries = [...runs.values()];
  for (const run of summaries) run.topLabel = topLabel(run.labels);
  return summaries.sort(byRecency);
}

export type ExperimentSummary = {
  experiment: string;
  taskGroup: string;
  replays: number;
  recordedAt: string | null;
  /** Median total step time per replay condition; absent conditions are omitted, not zero. */
  medians: Record<string, number>;
};

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** Replay runs grouped by the experiment that recorded them, newest first. */
export function summarizeExperiments(runs: DatasetRun[]): ExperimentSummary[] {
  const groups = new Map<string, DatasetRun[]>();
  for (const run of runs)
    if (run.experiment !== null)
      groups.set(run.experiment, [...(groups.get(run.experiment) ?? []), run]);
  return [...groups.entries()]
    .map(([experiment, members]): ExperimentSummary => {
      const byCondition = new Map<string, number[]>();
      for (const run of members)
        if (run.replayCondition !== null)
          byCondition.set(run.replayCondition, [
            ...(byCondition.get(run.replayCondition) ?? []),
            run.durationSeconds,
          ]);
      const times = members.flatMap((run) =>
        run.recordedAt === null ? [] : [run.recordedAt],
      );
      return {
        experiment,
        taskGroup: members[0].taskGroup,
        replays: members.length,
        recordedAt: times.length ? times.sort()[0] : null,
        medians: Object.fromEntries(
          [...byCondition].map(([condition, values]) => [
            condition,
            median(values),
          ]),
        ),
      };
    })
    .sort(
      (a, b) =>
        (b.recordedAt ?? "").localeCompare(a.recordedAt ?? "") ||
        a.experiment.localeCompare(b.experiment),
    );
}

export type ConditionRatio = {
  condition: string;
  ratio: number;
  experiments: number;
};

const INJECTED = ["cpu", "retrieval", "inference", "mixed"];

/**
 * Per injected condition, the median across experiments of (condition median / clean median).
 * Ratios, not seconds, because the experiments replay tasks of different length. Experiments
 * missing either side are left out of that condition; observed, not a controlled comparison.
 */
export function conditionRatios(
  experiments: ExperimentSummary[],
): ConditionRatio[] {
  return INJECTED.flatMap((condition) => {
    const ratios = experiments.flatMap((experiment) => {
      const clean = experiment.medians.clean;
      const injected = experiment.medians[condition];
      return clean && injected !== undefined ? [injected / clean] : [];
    });
    return ratios.length
      ? [{ condition, ratio: median(ratios), experiments: ratios.length }]
      : [];
  });
}
