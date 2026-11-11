import { executionOf, queueOf } from "./dashboard-data";
import type { Experiment, Step } from "./experiment-schema";

// Whether two saved runs may be compared, and the measurements each one recorded. Runs whose
// workload settings differ, or whose settings are unknown, are never compared: a timing
// difference between them would mix configuration with performance.

export type SettingDifference = {
  setting: string;
  baseline: string;
  candidate: string;
};
export type Comparability = {
  comparable: boolean;
  /** Why the pair cannot be compared regardless of settings. */
  reasons: string[];
  /** Settings that differ or are not recorded on one or both runs. */
  differences: SettingDifference[];
};

/** A recorded setting; `undefined` means the run did not record it. */
type Setting = string | number | null | undefined;

const NOT_RECORDED = "not recorded";

/**
 * Settings that define a saved run's workload. `tests_per_step` is null by contract for a
 * workload without tests, so null there is a value; any absent setting is unknown.
 */
const SETTINGS: [string, (run: Experiment) => Setting][] = [
  ["Workload kind", (run) => run.kind],
  ["Workload", (run) => run.experiment.workload],
  ["Workload checksum", (run) => run.clean.checksum],
  ["Iterations per step", (run) => run.experiment.iterations],
  [
    "Competing iterations (contention)",
    (run) => run.experiment.competing_iterations,
  ],
  ["Worker capacity", (run) => run.experiment.worker_capacity],
  ["Tests per step", (run) => run.experiment.tests_per_step],
  ["Baseline repeats", (run) => run.experiment.baseline_repeats],
  ["Agent model", (run) => (run.agent ? run.agent.model : "no agent")],
];

function settingText(value: Setting) {
  if (value === undefined) return NOT_RECORDED;
  if (value === null) return "not applicable";
  return String(value);
}

export function comparability(
  baseline: Experiment,
  candidate: Experiment,
): Comparability {
  const reasons: string[] = [];
  if (baseline.run_id === candidate.run_id)
    reasons.push("The same run is selected as baseline and candidate.");
  for (const [label, run] of [
    ["Baseline", baseline],
    ["Candidate", candidate],
  ] as const)
    if (!run.persisted)
      reasons.push(
        `${label} is a local artifact, not a saved run, so its UUID is not tied to recorded history.`,
      );
  const differences = SETTINGS.flatMap(([setting, value]) => {
    const a = value(baseline);
    const b = value(candidate);
    return a === undefined || b === undefined || a !== b
      ? [{ setting, baseline: settingText(a), candidate: settingText(b) }]
      : [];
  });
  return {
    comparable: reasons.length === 0 && differences.length === 0,
    reasons,
    differences,
  };
}

/** Sample coverage of one step; null when the step recorded no telemetry at all. */
function coverageOf(step: Step) {
  return step.telemetry
    ? {
        samples: step.telemetry.sample_count,
        validPressureSamples: step.telemetry.valid_pressure_samples,
        status: step.telemetry.status,
      }
    : null;
}

/** Why throughput and tail latency are not shown for saved runs. */
export const SINGLE_RUN_LIMITATION =
  "Not recorded: a saved run is one task, so it has no batch throughput or per-task tail latency.";

/**
 * What one saved run measured. Throughput and p95 task latency need a batch of tasks, which
 * saved runs do not record, so they are null rather than derived from a few steps.
 */
export function runMeasurements(run: Experiment) {
  return {
    cleanStepSeconds: run.clean.duration_seconds,
    contendedStepSeconds: run.contended.duration_seconds,
    cleanQueueSeconds: run.clean.queue_seconds,
    cleanExecutionSeconds: run.clean.execution_seconds,
    contendedQueueSeconds: run.contended.queue_seconds,
    contendedExecutionSeconds: run.contended.execution_seconds,
    totalExecutionSeconds: executionOf(run),
    totalQueueSeconds: queueOf(run),
    throughputTasksPerSecond: null,
    p95TaskLatencySeconds: null,
    checksPassed: run.checks.passed,
    agentStatus: run.agent?.status ?? null,
    coverage: {
      clean: coverageOf(run.clean),
      contended: coverageOf(run.contended),
    },
    limitations: [...run.limitations, ...(run.agent?.limitations ?? [])],
  };
}

/**
 * Candidate minus baseline. The relative change is null when either value is missing or the
 * baseline is zero, since no percentage of zero exists.
 */
export function observedDifference(
  baselineSeconds: number | null,
  candidateSeconds: number | null,
) {
  if (baselineSeconds === null || candidateSeconds === null) return null;
  const seconds = candidateSeconds - baselineSeconds;
  return {
    seconds,
    relative: baselineSeconds > 0 ? seconds / baselineSeconds : null,
  };
}
