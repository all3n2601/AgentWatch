import {
  conditionRatios,
  summarizeExperiments,
  type DatasetRun,
} from "./dataset-runs";
import type { DatasetSnapshot } from "./dataset-schema";
import { LABEL_RESOURCES, replayConditionName } from "./dataset-view";

export type Ready = Extract<DatasetSnapshot, { status: "ready" }>;

/** One replay run as a dot: x is its replay order within its experiment, y its total time. */
export type Dot = {
  id: string;
  /** The live run's UUID after an import, when an import map says so; the dot links to it. */
  liveId: string | null;
  x: number;
  y: number;
  condition: string;
  ratio: number | null;
};

export type WorstRun = {
  runId: string;
  liveId: string | null;
  task: string;
  condition: string;
  seconds: number;
  cleanSeconds: number;
  ratio: number;
  queueSeconds: number;
  slowSteps: number;
  steps: number;
};

export type OverviewModel = {
  release: string;
  runs: number;
  replayRuns: number;
  steps: number;
  excessSteps: number;
  telemetryPresent: number;
  telemetryMissed: number;
  experiments: number;
  slowest: WorstRun | null;
  dots: Dot[];
  conditions: {
    id: string;
    label: string;
    ratio: number;
    experiments: number;
  }[];
  labelMix: { id: string; label: string; value: number; color: string }[];
  worst: WorstRun[];
  tasks: { task: string; ratios: Record<string, number> }[];
  health: {
    gate: string;
    anomalies: number;
    trainRows: number;
    testRows: number;
    validationRows: number;
  };
};

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

const REPLAY_CONDITIONS = ["cpu", "retrieval", "inference", "mixed"];
const WORST_COUNT = 6;

/**
 * The Overview's numbers, computed once on the server. "Time lost" is each run's total time over
 * the median total time of the clean replays of the same experiment, so every ratio compares a
 * run with its own baseline. Observed under injected contention, not a controlled comparison.
 */
export function buildOverviewModel(
  snapshot: Ready,
  importMap: Record<string, string> = {},
): OverviewModel {
  const { manifest, split, validation, runs } = snapshot;
  const replays = runs.filter(
    (run) => run.replayCondition !== null && run.experiment !== null,
  );
  const cleanTimes = new Map<string, number[]>();
  for (const run of replays)
    if (run.replayCondition === "clean")
      cleanTimes.set(run.experiment!, [
        ...(cleanTimes.get(run.experiment!) ?? []),
        run.durationSeconds,
      ]);
  const cleanMedian = new Map(
    [...cleanTimes].map(([experiment, times]) => [experiment, median(times)]),
  );

  const ratioOf = (run: DatasetRun) => {
    const clean = cleanMedian.get(run.experiment ?? "");
    return clean ? run.durationSeconds / clean : null;
  };
  const toWorst = (run: DatasetRun, ratio: number): WorstRun => ({
    runId: run.runId,
    liveId: importMap[run.runId] ?? null,
    task: run.taskGroup,
    condition: run.replayCondition!,
    seconds: run.durationSeconds,
    cleanSeconds: cleanMedian.get(run.experiment!)!,
    ratio,
    queueSeconds: run.queueSeconds,
    slowSteps: run.excessSteps,
    steps: run.steps,
  });

  const injected = replays
    .filter((run) => REPLAY_CONDITIONS.includes(run.replayCondition!))
    .flatMap((run) => {
      const ratio = ratioOf(run);
      return ratio === null ? [] : [toWorst(run, ratio)];
    })
    .sort((a, b) => b.ratio - a.ratio || a.runId.localeCompare(b.runId));

  const experiments = summarizeExperiments(runs);
  const taskRatios = new Map<string, Map<string, number[]>>();
  for (const experiment of experiments) {
    const clean = experiment.medians.clean;
    if (!clean) continue;
    const byCondition = taskRatios.get(experiment.taskGroup) ?? new Map();
    for (const condition of REPLAY_CONDITIONS) {
      const value = experiment.medians[condition];
      if (value !== undefined)
        byCondition.set(condition, [
          ...(byCondition.get(condition) ?? []),
          value / clean,
        ]);
    }
    taskRatios.set(experiment.taskGroup, byCondition);
  }

  const status = manifest.telemetry_status;
  return {
    release: manifest.dataset_version,
    runs: runs.length,
    replayRuns: replays.length,
    steps: manifest.rows,
    excessSteps: manifest.rows - (manifest.label_resource.none ?? 0),
    telemetryPresent: status.present ?? 0,
    telemetryMissed: status.no_overlap ?? 0,
    experiments: experiments.length,
    slowest: injected[0] ?? null,
    dots: replays
      .filter((run) => run.replayIndex !== null)
      .map((run) => ({
        id: run.runId,
        liveId: importMap[run.runId] ?? null,
        x: run.replayIndex!,
        y: run.durationSeconds,
        condition: run.replayCondition!,
        ratio: ratioOf(run),
      })),
    conditions: conditionRatios(experiments).map((entry) => ({
      id: entry.condition,
      label: replayConditionName(entry.condition),
      ratio: entry.ratio,
      experiments: entry.experiments,
    })),
    labelMix: LABEL_RESOURCES.filter((entry) => entry.id !== "none").map(
      (entry) => ({
        id: entry.id,
        label: entry.label,
        value: manifest.label_resource[entry.id] ?? 0,
        color: entry.color,
      }),
    ),
    worst: injected.slice(0, WORST_COUNT),
    tasks: [...taskRatios]
      .map(([task, byCondition]) => ({
        task,
        ratios: Object.fromEntries(
          [...byCondition].map(([condition, values]) => [
            condition,
            median(values),
          ]),
        ),
      }))
      .sort((a, b) => a.task.localeCompare(b.task)),
    health: {
      gate: validation.status,
      anomalies: validation.anomalies.length,
      trainRows: split.splits.train?.rows ?? 0,
      testRows: split.splits.test?.rows ?? 0,
      validationRows: split.splits.validation?.rows ?? 0,
    },
  };
}
