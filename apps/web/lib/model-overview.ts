import {
  conditionRatios,
  summarizeExperiments,
  type DatasetRun,
} from "./dataset-runs";
import { LABEL_RESOURCES, displayName } from "./dataset-view";
import type { Ready } from "./overview-model";

/** Call types in the order the Overview lists them; others follow, largest first. */
const CALL_TYPES = [
  { id: "model_call", label: "Model calls" },
  { id: "tool", label: "Tool calls" },
  { id: "retrieval", label: "Retrieval calls" },
] as const;

/** Short condition names for "Largest observed slowdown · Mixed 8.6×". */
const CONDITION_NAMES: Record<string, string> = {
  cpu: "CPU",
  inference: "Inference",
  retrieval: "Retrieval",
  mixed: "Mixed",
};

/** Call type served by each pipeline stage in the campus scene. */
export const STAGE_CALL_TYPES = {
  model: "model_call",
  retrieval: "retrieval",
  cpuTool: "tool",
} as const;
export type PipelineStage = keyof typeof STAGE_CALL_TYPES;

export type ModelOption = { id: string; name: string; runs: number };

export type FailedRun = {
  runId: string;
  /** The live run's UUID when an import map links it; only then can the run page open it. */
  liveId: string | null;
  task: string;
  recordedAt: string | null;
};

export type ModelOverview = {
  /** Full release id, e.g. "steps-v1.3+aafae73eb211". */
  release: string;
  /** Release name without the content hash, e.g. "steps-v1.3". */
  releaseName: string;
  model: ModelOption;
  models: ModelOption[];
  runs: number;
  steps: number;
  outcomes: { passed: number; failed: number; unrecorded: number };
  /** Steps whose host samples overlap the step. */
  telemetryPresent: number;
  callTypes: { id: string; label: string; value: number }[];
  /** Steps labelled with each resource's excess wait; steps without excess are left out. */
  resourceLabels: { id: string; label: string; value: number }[];
  /**
   * The injected condition with the highest median ratio of its replays' total time over the
   * same experiment's clean replays. Observed under injected contention, not causal.
   */
  largestSlowdown: {
    condition: string;
    label: string;
    ratio: number;
    experiments: number;
  } | null;
  /**
   * The pipeline stages the campus scene draws, each with its steps and the steps
   * labelled with excess wait. Stage = call type: model calls run on the model
   * runtime, retrieval calls on retrieval, tool calls on the CPU tool.
   */
  stages: Record<PipelineStage, { calls: number; excessSteps: number }>;
  /** Failed runs, newest first. */
  failedRuns: FailedRun[];
  /** Runs in the release with no agent model (coding and CPU experiments). */
  runsWithoutModel: number;
};

/** "qwen3:1.7b" → "Qwen3 1.7B"; identifiers in any other shape are shown as recorded. */
export function modelDisplayName(id: string): string {
  const match = /^([a-z][\w.-]*):([\w.]+)$/i.exec(id);
  if (!match) return id;
  const [, family, size] = match;
  return `${family[0].toUpperCase()}${family.slice(1)} ${size.toUpperCase()}`;
}

const sum = (runs: DatasetRun[], pick: (run: DatasetRun) => number) =>
  runs.reduce((total, run) => total + pick(run), 0);

function addCounts(
  runs: DatasetRun[],
  pick: (run: DatasetRun) => Record<string, number>,
) {
  const totals: Record<string, number> = {};
  for (const run of runs)
    for (const [key, value] of Object.entries(pick(run)))
      totals[key] = (totals[key] ?? 0) + value;
  return totals;
}

/**
 * The Overview for one agent model's population of runs. Without `requestedModel`, or when it
 * names no model in the release, the model with the most runs is shown. Null when no run in the
 * release records a model.
 */
export function buildModelOverview(
  snapshot: Ready,
  importMap: Record<string, string> = {},
  requestedModel?: string,
): ModelOverview | null {
  const byModel = new Map<string, DatasetRun[]>();
  for (const run of snapshot.runs)
    if (run.agentModel !== null)
      byModel.set(run.agentModel, [
        ...(byModel.get(run.agentModel) ?? []),
        run,
      ]);

  const models = [...byModel]
    .map(([id, runs]) => ({
      id,
      name: modelDisplayName(id),
      runs: runs.length,
    }))
    .sort((a, b) => b.runs - a.runs || a.id.localeCompare(b.id));
  if (models.length === 0) return null;

  const model =
    models.find((option) => option.id === requestedModel) ?? models[0];
  const runs = byModel.get(model.id)!;

  const stepTypes = addCounts(runs, (run) => run.stepTypes);
  const excessByType = addCounts(runs, (run) => run.excessByStepType);
  const stages = Object.fromEntries(
    Object.entries(STAGE_CALL_TYPES).map(([stage, type]) => [
      stage,
      { calls: stepTypes[type] ?? 0, excessSteps: excessByType[type] ?? 0 },
    ]),
  ) as ModelOverview["stages"];
  const known = new Set<string>(CALL_TYPES.map((type) => type.id));
  const callTypes = [
    ...CALL_TYPES.map((type) => ({ ...type, value: stepTypes[type.id] ?? 0 })),
    ...Object.entries(stepTypes)
      .filter(([id]) => !known.has(id))
      .map(([id, value]) => ({ id, label: displayName(id), value }))
      .sort((a, b) => b.value - a.value),
  ];

  const labels = addCounts(runs, (run) => run.labels);
  const resourceLabels = LABEL_RESOURCES.filter((entry) => entry.id !== "none")
    .map((entry) => ({
      id: entry.id,
      label: entry.label,
      value: labels[entry.id] ?? 0,
    }))
    .sort((a, b) => b.value - a.value);

  const ratios = conditionRatios(summarizeExperiments(runs));
  const top = [...ratios].sort((a, b) => b.ratio - a.ratio)[0];

  const release = snapshot.manifest.dataset_version;
  return {
    release,
    releaseName: release.split("+")[0],
    model,
    models,
    runs: runs.length,
    steps: sum(runs, (run) => run.steps),
    outcomes: {
      passed: runs.filter((run) => run.passed === true).length,
      failed: runs.filter((run) => run.passed === false).length,
      unrecorded: runs.filter((run) => run.passed === null).length,
    },
    telemetryPresent: sum(runs, (run) => run.telemetryPresent),
    callTypes,
    stages,
    resourceLabels,
    largestSlowdown: top
      ? {
          condition: top.condition,
          label: CONDITION_NAMES[top.condition] ?? displayName(top.condition),
          ratio: top.ratio,
          experiments: top.experiments,
        }
      : null,
    // `snapshot.runs` is newest first, and filtering keeps that order.
    failedRuns: runs
      .filter((run) => run.passed === false)
      .map((run) => ({
        runId: run.runId,
        liveId: importMap[run.runId] ?? null,
        task: run.taskGroup,
        recordedAt: run.recordedAt,
      })),
    runsWithoutModel: snapshot.runs.filter((run) => run.agentModel === null)
      .length,
  };
}
