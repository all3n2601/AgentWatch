import assert from "node:assert/strict";
import test from "node:test";
import { experimentSchema, type Experiment } from "../lib/experiment-schema";
import {
  comparability,
  measurementRows,
  observedDifference,
  runMeasurements,
  type Comparability,
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
const POLICY = "Concurrency policy";

/** Differences other than the policy, which saved runs never record. */
const workloadDifferences = (result: Comparability) =>
  result.differences.filter(({ setting }) => setting !== POLICY);

test("the unrecorded policy blocks saved runs whose workload settings all match", () => {
  const result = comparability(savedRun(BASE), savedRun(CAND));
  assert.equal(result.comparable, false);
  assert.deepEqual(result.reasons, []);
  assert.deepEqual(result.differences, [
    { setting: POLICY, baseline: "not recorded", candidate: "not recorded" },
  ]);
  assert.deepEqual(
    result.settings.map(({ setting, status }) => [setting, status]),
    [
      ["Workload kind", "same"],
      ["Workload", "same"],
      ["Workload checksum", "same"],
      ["Iterations per step", "same"],
      ["Competing iterations (contention)", "same"],
      ["Worker capacity", "same"],
      ["Tests per step", "same"],
      ["Baseline repeats", "same"],
      ["Agent model", "same"],
      [POLICY, "unknown"],
    ],
  );
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
    assert.deepEqual(workloadDifferences(result), [
      { setting, baseline, candidate },
    ]);
    assert.equal(
      result.settings.find((row) => row.setting === setting)?.status,
      "differs",
    );
  }
});

test("a setting not recorded on either run is unknown, never a match", () => {
  const a = savedRun(BASE);
  const b = savedRun(CAND);
  delete a.experiment.competing_iterations;
  delete b.experiment.competing_iterations;
  const result = comparability(a, b);
  assert.equal(result.comparable, false);
  assert.deepEqual(workloadDifferences(result), [
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
  assert.deepEqual(
    workloadDifferences(comparability(cpu(BASE), cpu(CAND))),
    [],
  );
  assert.deepEqual(
    workloadDifferences(comparability(cpu(BASE), savedRun(CAND))),
    [{ setting: "Tests per step", baseline: "not applicable", candidate: "3" }],
  );
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
  assert.deepEqual(workloadDifferences(comparability(a, b)), [
    { setting: "Agent model", baseline: "qwen3:1.7b", candidate: "no agent" },
  ]);
  b.agent = agent("qwen3:8b");
  assert.equal(workloadDifferences(comparability(a, b)).length, 1);
  b.agent = agent("qwen3:1.7b");
  assert.deepEqual(workloadDifferences(comparability(a, b)), []);
});

test("local artifacts and a run compared with itself are refused", () => {
  const local = savedRun(CAND);
  local.persisted = false;
  const result = comparability(savedRun(BASE), local);
  assert.equal(result.comparable, false);
  assert.deepEqual(workloadDifferences(result), []);
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

test("blocked pairs show each run's measurements without any difference", () => {
  const candidate = savedRun(CAND);
  candidate.clean = {
    ...candidate.clean,
    duration_seconds: 4.5,
    execution_seconds: 3.5,
  };
  const rows = measurementRows(savedRun(BASE), candidate, false);
  assert.ok(rows.every((row) => row.difference === undefined));
  const clean = rows.find(
    (row) => row.measurement === "Clean step completion time",
  );
  assert.deepEqual(clean, {
    measurement: "Clean step completion time",
    baseline: "3.00 s",
    candidate: "4.50 s",
  });
  assert.deepEqual(
    rows.find((row) => row.measurement === "Throughput"),
    {
      measurement: "Throughput",
      baseline: "Not recorded",
      candidate: "Not recorded",
    },
  );
});

test("comparable pairs show signed timing differences, never for quality or coverage", () => {
  const candidate = savedRun(CAND);
  candidate.clean = {
    ...candidate.clean,
    duration_seconds: 4.5,
    execution_seconds: 3.5,
  };
  candidate.checks = { passed: false };
  const rows = measurementRows(savedRun(BASE), candidate, true);
  assert.deepEqual(
    rows.find((row) => row.measurement === "Clean step completion time")
      ?.difference,
    { seconds: "+1.50 s", relative: "+50.0%" },
  );
  assert.deepEqual(
    rows.find((row) => row.measurement === "Clean queue")?.difference,
    { seconds: "0 ms", relative: "0.0%" },
  );
  for (const measurement of [
    "Throughput",
    "p95 task latency",
    "Acceptance checks",
    "Agent outcome",
    "Clean step coverage",
  ])
    assert.equal(
      rows.find((row) => row.measurement === measurement)?.difference,
      undefined,
      measurement,
    );
  assert.equal(
    rows.find((row) => row.measurement === "Acceptance checks")?.candidate,
    "Failed",
  );
  assert.equal(
    rows.find((row) => row.measurement === "Clean step coverage")?.baseline,
    "No telemetry recorded",
  );
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
