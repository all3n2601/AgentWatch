import type { Experiment } from "./experiment-schema";
export const pages = [
  "overview",
  "runs",
  "traces",
  "agents",
  "resources",
  "compare",
  "settings",
] as const;
export type Page = (typeof pages)[number];
export const stepsOf = (run: Experiment) => [
  ...run.baseline.steps,
  run.clean,
  run.contended,
];
export const runTime = (run: Experiment) => run.contended.finished_ns / 1e6;
export const runLabel = (run: Experiment) =>
  run.agent?.model ??
  (run.kind === "coding" ? "Python test tool" : "CPU replay");
export const queueOf = (run: Experiment) =>
  stepsOf(run).reduce((n, s) => n + s.queue_seconds, 0);
export const executionOf = (run: Experiment) =>
  stepsOf(run).reduce((n, s) => n + s.execution_seconds, 0);
export const duration = (n: number) =>
  n === 0
    ? "0 ms"
    : n < 0.001
      ? `${(n * 1e6).toFixed(0)} μs`
      : n < 1
        ? `${(n * 1000).toFixed(0)} ms`
        : `${n.toFixed(2)} s`;
export const percent = (n: number) => `${(n * 100).toFixed(1)}%`;
export function scopedRuns(
  runs: Experiment[],
  range: string,
  now = Date.now(),
) {
  const cutoff =
    range === "24h"
      ? now - 86400000
      : range === "7d"
        ? now - 7 * 86400000
        : -Infinity;
  return runs
    .filter((run) => runTime(run) >= cutoff)
    .sort((a, b) => runTime(b) - runTime(a));
}
export function aggregate(runs: Experiment[]) {
  const steps = runs.flatMap(stepsOf);
  const queue = steps.reduce((n, s) => n + s.queue_seconds, 0);
  const execution = steps.reduce((n, s) => n + s.execution_seconds, 0);
  const calls = runs.flatMap((r) => r.agent?.inference ?? []);
  return {
    runs: runs.length,
    steps: steps.length,
    passed: runs.filter((r) => r.checks.passed).length,
    queue,
    execution,
    queueFraction: queue + execution ? queue / (queue + execution) : 0,
    attributed: steps.filter((s) => s.resource !== "none").length,
    agentRuns: runs.filter((r) => r.agent).length,
    modelCalls: calls.length,
    inferenceWall: calls.reduce((n, c) => n + c.wall_seconds, 0),
    outputTokens:
      calls.length && calls.every((c) => c.metrics.eval_count !== undefined)
        ? calls.reduce((n, c) => n + c.metrics.eval_count!, 0)
        : null,
  };
}
export function matchingWorkload(a: Experiment, b: Experiment) {
  return (
    a.clean.checksum === b.clean.checksum &&
    a.experiment.iterations === b.experiment.iterations &&
    a.experiment.worker_capacity === b.experiment.worker_capacity
  );
}
export const evidenceUrl = (run: Experiment, file = "results.json") =>
  `/api/experiments/${run.kind}/artifact?file=${file}${run.persisted ? `&run_id=${run.run_id}` : ""}`;
