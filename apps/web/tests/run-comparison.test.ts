import assert from "node:assert/strict";
import test from "node:test";
import { experimentSchema, type Experiment } from "../lib/experiment-schema";
import {
  comparability,
  observedDifference,
  runMeasurements,
} from "../lib/run-comparison";

const step = {
  step_id: "clean",
  submitted_ns: 0,
  started_ns: 1e9,
  finished_ns: 3e9,
  duration_seconds: 3,
  queue_seconds: 1,
  execution_seconds: 2,
  cpu_seconds: 1.8,
  queue_fraction: 1 / 3,
  resource: "cpu_sandbox" as const,
  checksum: "abc",
  evidence: "timestamps",
  rule_version: "v1",
  threshold_seconds: 0.01,
};

/** A saved coding run with the demo's recorded settings. */
function savedRun(id: string): Experiment {
  return {
    run_id: id,
    kind: "coding",
    persisted: true,
    experiment: {
      worker_capacity: 1,
      iterations: 4,
      competing_iterations: 32,
      baseline_repeats: 3,
      workload: "fixed Python project: run three tests",
      tests_per_step: 3,
    },
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [step, step, step].map((s, i) => ({ ...s, step_id: `b${i}` })),
    },
    clean: step,
    contended: { ...step, step_id: "contended" },
    checks: { passed: true },
    limitations: ["Coding workload is a trusted test tool fixture"],
  };
}

const BASE = "0b7d1f0e-1c33-4f2e-9a43-6b7c2d0e5a10";
const CAND = "5a3e9c11-7d2b-4c8a-b1f0-2e6d9a4c7b21";

test("saved runs with identical settings are comparable", () => {
  assert.deepEqual(comparability(savedRun(BASE), savedRun(CAND)), {
    comparable: true,
    reasons: [],
    differences: [],
  });
});

test("every differing workload setting blocks the comparison and is listed", () => {
  const cases: [string, (run: Experiment) => void, string, string][] = [
    ["Workload kind", (r) => (r.kind = "cpu"), "coding", "cpu"],
    [
      "Workload",
      (r) => (r.experiment.workload = "fixed SHA-256 CPU task"),
      "fixed Python project: run three tests",
      "fixed SHA-256 CPU task",
    ],
    [
      "Workload checksum",
      (r) => (r.clean = { ...r.clean, checksum: "def" }),
      "abc",
      "def",
    ],
    ["Iterations per step", (r) => (r.experiment.iterations = 8), "4", "8"],
    [
      "Competing iterations (contention)",
      (r) => (r.experiment.competing_iterations = 64),
      "32",
      "64",
    ],
    ["Worker capacity", (r) => (r.experiment.worker_capacity = 2), "1", "2"],
    ["Tests per step", (r) => (r.experiment.tests_per_step = 1), "3", "1"],
    ["Baseline repeats", (r) => (r.experiment.baseline_repeats = 5), "3", "5"],
  ];
  for (const [setting, change, baseline, candidate] of cases) {
    const other = savedRun(CAND);
    change(other);
    const result = comparability(savedRun(BASE), other);
    assert.equal(result.comparable, false, setting);
    assert.deepEqual(result.differences, [{ setting, baseline, candidate }]);
  }
});

test("a setting not recorded on either run is unknown, never a match", () => {
  const a = savedRun(BASE);
  const b = savedRun(CAND);
  delete a.experiment.competing_iterations;
  delete b.experiment.competing_iterations;
  const result = comparability(a, b);
  assert.equal(result.comparable, false);
  assert.deepEqual(result.differences, [
    {
      setting: "Competing iterations (contention)",
      baseline: "not recorded",
      candidate: "not recorded",
    },
  ]);
});

test("tests per step is not applicable on both CPU runs, but differs from a recorded count", () => {
  const cpu = (id: string) => {
    const run = savedRun(id);
    run.experiment.tests_per_step = null;
    return run;
  };
  assert.equal(comparability(cpu(BASE), cpu(CAND)).comparable, true);
  assert.deepEqual(comparability(cpu(BASE), savedRun(CAND)).differences, [
    { setting: "Tests per step", baseline: "not applicable", candidate: "3" },
  ]);
});

test("a different or missing agent model blocks the comparison", () => {
  const agent = (model: string): Experiment["agent"] => ({
    model,
    status: "passed",
    messages: [],
    limitations: [],
    inference: [],
  });
  const a = savedRun(BASE);
  const b = savedRun(CAND);
  a.agent = agent("qwen3:1.7b");
  assert.deepEqual(comparability(a, b).differences, [
    { setting: "Agent model", baseline: "qwen3:1.7b", candidate: "no agent" },
  ]);
  b.agent = agent("qwen3:8b");
  assert.equal(comparability(a, b).comparable, false);
  b.agent = agent("qwen3:1.7b");
  assert.equal(comparability(a, b).comparable, true);
});

test("local artifacts and a run compared with itself are refused", () => {
  const local = savedRun(CAND);
  local.persisted = false;
  const result = comparability(savedRun(BASE), local);
  assert.equal(result.comparable, false);
  assert.deepEqual(result.differences, []);
  assert.match(result.reasons[0], /^Candidate is a local artifact/);

  const self = comparability(savedRun(BASE), savedRun(BASE));
  assert.equal(self.comparable, false);
  assert.match(self.reasons[0], /same run/);
});

test("measurements keep unrecorded batch metrics and telemetry null, not zero", () => {
  const run = savedRun(BASE);
  run.contended = {
    ...run.contended,
    telemetry: {
      sample_count: 0,
      valid_pressure_samples: 0,
      max_cpu_pressure_avg10_percent: null,
      status: "unsupported",
    },
  };
  run.agent = {
    model: "qwen3:1.7b",
    status: "inconclusive",
    messages: [],
    limitations: ["Model calls ran on CPU"],
    inference: [],
  };
  const measured = runMeasurements(run);
  assert.equal(measured.cleanStepSeconds, 3);
  assert.equal(measured.totalExecutionSeconds, 10);
  assert.equal(measured.totalQueueSeconds, 5);
  assert.equal(measured.throughputTasksPerSecond, null);
  assert.equal(measured.p95TaskLatencySeconds, null);
  assert.equal(measured.coverage.clean, null);
  assert.deepEqual(measured.coverage.contended, {
    samples: 0,
    validPressureSamples: 0,
    status: "unsupported",
  });
  assert.equal(measured.agentStatus, "inconclusive");
  assert.deepEqual(measured.limitations, [
    "Coding workload is a trusted test tool fixture",
    "Model calls ran on CPU",
  ]);
  assert.equal(runMeasurements(savedRun(CAND)).agentStatus, null);
});

test("observed differences have no relative change from a zero or missing baseline", () => {
  assert.deepEqual(observedDifference(2, 3), { seconds: 1, relative: 0.5 });
  assert.deepEqual(observedDifference(0, 1), { seconds: 1, relative: null });
  assert.deepEqual(observedDifference(0, 0), { seconds: 0, relative: null });
  assert.equal(observedDifference(null, 1), null);
  assert.equal(observedDifference(1, null), null);
});

test("the saved-run schema keeps the recorded contention intensity", () => {
  // `kind` and `persisted` are added after parsing; the schema strips unknown keys.
  const result = savedRun(BASE);
  const parsed = experimentSchema.parse(result);
  assert.equal(parsed.experiment.competing_iterations, 32);
  delete result.experiment.competing_iterations;
  assert.equal(
    experimentSchema.parse(result).experiment.competing_iterations,
    undefined,
  );
});
