import { randomUUID } from "node:crypto";
import type { SavedRun } from "./fixture-control";

// Builders for synthetic saved runs (/v1/runs) served by the fixture API in browser tests.
// Shaped like the worker demo's results.json; never recorded evidence.

export const newSavedRunId = () => randomUUID();

type Options = {
  sequence: number;
  iterations?: number;
  /** Contention intensity; null omits it, as older saved runs do. Defaults to 8 × iterations. */
  competingIterations?: number | null;
  cleanSeconds?: number;
  passed?: boolean;
};

function step(
  stepId: string,
  endMs: number,
  queueSeconds: number,
  executionSeconds: number,
) {
  const finishedNs = endMs * 1e6;
  const startedNs = finishedNs - executionSeconds * 1e9;
  return {
    step_id: stepId,
    submitted_ns: startedNs - queueSeconds * 1e9,
    started_ns: startedNs,
    finished_ns: finishedNs,
    duration_seconds: queueSeconds + executionSeconds,
    queue_seconds: queueSeconds,
    execution_seconds: executionSeconds,
    cpu_seconds: executionSeconds * 0.9,
    queue_fraction: queueSeconds / (queueSeconds + executionSeconds),
    resource: "cpu_sandbox",
    checksum: "e2e-checksum",
    evidence: "synthetic browser-test fixture",
    rule_version: "v1",
    threshold_seconds: 0.01,
  };
}

/** A saved coding run; `sequence` orders history and sets its recorded time. */
export function savedRun(
  runId: string,
  {
    sequence,
    iterations = 4,
    competingIterations = iterations * 8,
    cleanSeconds = 3,
    passed = true,
  }: Options,
): SavedRun {
  const endMs = Date.parse("2026-11-01T12:00:00Z") + sequence * 3_600_000;
  return {
    run_id: runId,
    kind: "coding",
    created_at: new Date(endMs).toISOString(),
    sequence,
    result: {
      run_id: runId,
      experiment: {
        worker_capacity: 1,
        iterations,
        ...(competingIterations !== null && {
          competing_iterations: competingIterations,
        }),
        baseline_repeats: 3,
        workload: "fixed Python project: run three tests",
        tests_per_step: 3,
      },
      baseline: {
        execution_cv: 0.01,
        queue_threshold_seconds: 0.01,
        steps: [0, 1, 2].map((i) =>
          step(`baseline-${i}`, endMs - 60_000 + i, 0.5, 2),
        ),
      },
      clean: step("clean", endMs - 30_000, 1, cleanSeconds - 1),
      contended: step("contended", endMs, 2, 3),
      checks: { passed, stable_baseline: passed },
      limitations: [`Synthetic browser-test run ${sequence}`],
    },
  };
}
