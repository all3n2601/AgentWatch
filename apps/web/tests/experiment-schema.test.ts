import assert from "node:assert/strict";
import test from "node:test";
import { experimentSchema } from "../lib/experiment-schema";

function fixture() {
  const step = {
    step_id: "clean",
    submitted_ns: 0,
    started_ns: 1_000_000,
    finished_ns: 1_000_000_000,
    duration_seconds: 1,
    queue_seconds: 0.001,
    execution_seconds: 0.999,
    cpu_seconds: 0.9,
    queue_fraction: 0.001,
    resource: "none",
    checksum: "abc",
    evidence: "worker timestamps",
    rule_version: "cpu-queue-v1",
    threshold_seconds: 0.01,
    telemetry: {
      sample_count: 1,
      valid_pressure_samples: 0,
      max_cpu_pressure_avg10_percent: null as number | null,
      status: "unavailable_or_no_samples",
    },
  };
  return {
    run_id: "run",
    experiment: {
      worker_capacity: 1,
      iterations: 1000,
      baseline_repeats: 3,
      workload: "tests",
      tests_per_step: 3,
    },
    baseline: {
      execution_cv: 0.02,
      queue_threshold_seconds: 0.01,
      steps: [step, step, step],
    },
    clean: step,
    contended: step,
    checks: { passed: true },
    limitations: [],
  };
}

test("unavailable telemetry stays null rather than appearing as zero pressure", () => {
  const result = experimentSchema.parse(fixture());
  assert.equal(result.clean.telemetry?.max_cpu_pressure_avg10_percent, null);
  const recordedZero = fixture();
  recordedZero.clean.telemetry.max_cpu_pressure_avg10_percent = 0;
  assert.equal(
    experimentSchema.parse(recordedZero).clean.telemetry
      ?.max_cpu_pressure_avg10_percent,
    0,
  );
});

test("invalid queue proportions and unknown resource labels are rejected", () => {
  const data = fixture();
  data.clean.queue_fraction = 2;
  assert.equal(experimentSchema.safeParse(data).success, false);
  data.clean.queue_fraction = 0.001;
  data.clean.resource = "invented";
  assert.equal(experimentSchema.safeParse(data).success, false);
});

test("reversed timestamps and inconsistent time breakdowns cannot reach charts", () => {
  const data = fixture();
  data.clean.started_ns = data.clean.finished_ns + 1;
  assert.equal(experimentSchema.safeParse(data).success, false);
  data.clean.started_ns = 1_000_000;
  data.clean.execution_seconds = 2;
  assert.equal(experimentSchema.safeParse(data).success, false);
});

test("empty baselines and missing acceptance status are rejected", () => {
  const data = fixture();
  data.baseline.steps = [];
  assert.equal(experimentSchema.safeParse(data).success, false);
  assert.equal(
    experimentSchema.safeParse({ ...fixture(), checks: {} }).success,
    false,
  );
});
