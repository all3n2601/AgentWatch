import assert from "node:assert/strict";
import test from "node:test";
import { aggregate, matchingWorkload, scopedRuns } from "../lib/dashboard-data";
import type { Experiment } from "../lib/experiment-schema";
function run(id: string, time: number): Experiment {
  const step = {
    step_id: "clean",
    submitted_ns: 0,
    started_ns: 1e9,
    finished_ns: 2e9,
    duration_seconds: 2,
    queue_seconds: 1,
    execution_seconds: 1,
    cpu_seconds: 0.9,
    queue_fraction: 0.5,
    resource: "cpu_sandbox" as const,
    checksum: "abc",
    evidence: "timestamps",
    rule_version: "v1",
    threshold_seconds: 0.01,
  };
  return {
    run_id: id,
    kind: "coding",
    experiment: {
      worker_capacity: 1,
      iterations: 1,
      baseline_repeats: 3,
      workload: "tests",
      tests_per_step: 3,
    },
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [step, step, step],
    },
    clean: step,
    contended: { ...step, finished_ns: time * 1e6 },
    checks: { passed: true },
    limitations: [],
  };
}
test("workspace totals sum measured tool steps and keep inference time separate", () => {
  const a = run("a", 1000);
  a.agent = {
    model: "qwen3",
    status: "passed",
    messages: [],
    limitations: [],
    inference: [
      { request_index: 0, wall_seconds: 8, metrics: { eval_count: 0 } },
    ],
  };
  const stats = aggregate([a, run("b", 2000)]);
  assert.equal(stats.steps, 10);
  assert.equal(stats.queue, 10);
  assert.equal(stats.execution, 10);
  assert.equal(stats.inferenceWall, 8);
  assert.equal(stats.queueFraction, 0.5);
  assert.equal(stats.outputTokens, 0);
});
test("missing token metrics remain unavailable instead of becoming zero", () => {
  const a = run("a", 0);
  a.agent = {
    model: "qwen3",
    status: "passed",
    messages: [],
    limitations: [],
    inference: [{ request_index: 0, wall_seconds: 8, metrics: {} }],
  };
  assert.equal(aggregate([a]).outputTokens, null);
  assert.equal(aggregate([]).outputTokens, null);
  assert.equal(aggregate([]).queueFraction, 0);
});
test("time scopes use recorded completion time and preserve source order", () => {
  const now = 10 * 86400000;
  const input = [
    run("older", now - 8 * 86400000),
    run("day", now - 86400000),
    run("latest", now),
  ];
  assert.deepEqual(
    scopedRuns(input, "24h", now).map((r) => r.run_id),
    ["latest", "day"],
  );
  assert.equal(scopedRuns(input, "7d", now).length, 2);
  assert.equal(scopedRuns(input, "all", now).length, 3);
  assert.equal(input[0].run_id, "older");
});
test("matching comparisons require identical output and capacity settings", () => {
  const a = run("a", 0),
    b = run("b", 1);
  assert.equal(matchingWorkload(a, b), true);
  b.experiment.worker_capacity = 2;
  assert.equal(matchingWorkload(a, b), false);
  b.experiment.worker_capacity = 1;
  b.clean = { ...b.clean, checksum: "different" };
  assert.equal(matchingWorkload(a, b), false);
});
