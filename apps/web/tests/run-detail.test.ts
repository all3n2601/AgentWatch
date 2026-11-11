import assert from "node:assert/strict";
import test from "node:test";
import type { Experiment, Step } from "../lib/experiment-schema";
import type { LiveRunSnapshot } from "../lib/live-api-schema";
import {
  buildRunDetail,
  contextSegments,
  formatTiming,
  formatUtc,
  headlineFor,
  rowBarShares,
  type EvidenceRow,
} from "../lib/run-detail";
import { RUN_ID, runSnapshot, stepEvent, stepView } from "./live-fixtures";

// Synthetic runs for tests only; they are not recorded-run evidence.
const SECOND_NS = 1e9;
const START_NS = Date.UTC(2026, 10, 11, 10, 0, 0) * 1e6;

function step(
  id: string,
  queue: number,
  execution: number,
  overrides: Partial<Step> = {},
): Step {
  return {
    step_id: id,
    submitted_ns: START_NS,
    started_ns: START_NS + queue * SECOND_NS,
    finished_ns: START_NS + (queue + execution) * SECOND_NS,
    duration_seconds: queue + execution,
    queue_seconds: queue,
    execution_seconds: execution,
    cpu_seconds: execution * 0.9,
    queue_fraction: queue / (queue + execution),
    resource: "none",
    checksum: "abc",
    evidence: "direct worker enqueue/start timestamps",
    rule_version: "cpu-queue-v1",
    threshold_seconds: 0.01,
    ...overrides,
  };
}

function savedRun(): Experiment {
  return {
    run_id: "11111111-1111-4111-8111-111111111111",
    kind: "cpu",
    experiment: {
      worker_capacity: 1,
      iterations: 1,
      baseline_repeats: 3,
      workload: "fixed task",
      tests_per_step: null,
    },
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [
        step("baseline-0", 0.001, 0.2),
        step("baseline-1", 0.002, 0.2),
        step("baseline-2", 0.003, 0.2),
      ],
    },
    clean: step("clean", 0.002, 0.2, {
      telemetry: {
        sample_count: 2,
        valid_pressure_samples: 0,
        max_cpu_pressure_avg10_percent: 0,
        status: "unavailable_or_no_samples",
      },
    }),
    contended: step("contended", 1.5, 0.2, {
      resource: "cpu_sandbox",
      telemetry: {
        sample_count: 7,
        valid_pressure_samples: 7,
        max_cpu_pressure_avg10_percent: 0,
        status: "available",
      },
    }),
    checks: { passed: true },
    limitations: ["CPU worker queue only; not host saturation"],
  };
}

const row = (rows: EvidenceRow[], id: string) =>
  rows.find((entry) => entry.id === id);

test("saved runs rank by queue delay and label steps by their role", () => {
  const model = buildRunDetail({ source: "saved", run: savedRun() });
  assert.deepEqual(
    model.steps.map((s) => s.label),
    [
      "Contended",
      "Baseline repeat 3 of 3",
      "Baseline repeat 2 of 3",
      "Clean control",
      "Baseline repeat 1 of 3",
    ],
  );
  assert.equal(model.steps[0].basis, "attributed");
  assert.equal(model.steps[0].flag.label, "Excess wait");
  assert.equal(model.steps[1].flag.label, "No excess wait");
  assert.equal(model.totals.attributedSteps, 1);
  assert.equal(model.totals.totalSteps, 5);
  assert.ok(Math.abs(model.contendedOverClean! - 1.7 / 0.202) < 1e-9);
  assert.equal(model.limitations.length, 1);
});

test("a zero pressure value without a valid sample is unavailable, not zero", () => {
  const model = buildRunDetail({ source: "saved", run: savedRun() });
  const clean = model.steps.find((s) => s.stepId === "clean")!;
  const contended = model.steps.find((s) => s.stepId === "contended")!;
  assert.equal(row(clean.evidence, "pressure")?.value, null);
  assert.equal(row(contended.evidence, "pressure")?.value, 0);
  assert.equal(row(clean.evidence, "samples")?.value, 2);
  assert.equal(model.telemetry.label, "Telemetry limited");
  assert.equal(
    model.telemetry.detail,
    "1 of 5 steps have a valid CPU-pressure sample",
  );
});

test("saved step timestamps are shown in UTC with milliseconds", () => {
  const model = buildRunDetail({ source: "saved", run: savedRun() });
  const contended = model.steps[0];
  assert.equal(
    row(contended.evidence, "submitted")?.value,
    "2026-11-11 10:00:00.000 UTC",
  );
  assert.equal(
    row(contended.evidence, "started")?.value,
    "2026-11-11 10:00:01.500 UTC",
  );
  assert.equal(formatUtc("not a time"), "not a time");
});

test("live steps say where they ran and never claim an attribution", () => {
  const slowView = stepView(
    stepEvent("slow", 3, "completed", {
      submitted: 0,
      started: 2,
      finished: 5,
      cpu: 1.2,
    }),
  );
  const slow = {
    ...slowView,
    event: { ...slowView.event, resource_id: "tool-host-a" },
  };
  const unmeasured = stepEvent("unstarted", 4, "queued", { submitted: 1 });
  const snapshot = runSnapshot({
    state: "running",
    steps: [stepView(unmeasured), slow],
    unfinished_steps: ["unstarted"],
  }) as unknown as LiveRunSnapshot;
  const model = buildRunDetail({ source: "live", snapshot });
  assert.equal(model.runId, RUN_ID);
  assert.deepEqual(
    model.steps.map((s) => s.stepId),
    ["slow", "unstarted"],
  );
  const [first, second] = model.steps;
  assert.equal(first.basis, "ran_on");
  assert.equal(first.resourceLabel, "CPU sandbox");
  assert.match(first.basisLabel, /not an attribution/);
  assert.equal(
    row(first.evidence, "attribution")?.value,
    "None recorded for live runs",
  );
  assert.equal(row(first.evidence, "cpu")?.label, "CPU time (process)");
  assert.equal(second.queueSeconds, null);
  assert.equal(second.flag.label, "Queued");
  assert.equal(model.totals.attributedSteps, null);
  assert.equal(model.totals.measuredQueueSteps, 1);
  assert.equal(model.totals.queueSeconds, 2);
  assert.equal(model.contendedOverClean, null);
  assert.ok(model.notes.some((note) => note.includes("1 step(s)")));
});

test("the headline separates missing, zero and positive queue delay", () => {
  const live = (queue: number | null) =>
    buildRunDetail({
      source: "live",
      snapshot: runSnapshot({
        steps: [
          stepView(
            stepEvent("only", 1, "completed", {
              submitted: 0,
              started: queue ?? undefined,
              finished: queue === null ? undefined : queue + 1,
            }),
          ),
        ],
      }) as unknown as LiveRunSnapshot,
    });
  assert.equal(
    live(null).headline?.title,
    "No step has a measured queue delay",
  );
  assert.equal(live(0).headline?.title, "No step waited in queue");
  assert.equal(live(0).headline?.tone, "success");
  const waited = live(0.25).headline!;
  assert.equal(waited.title, "only waited longest: 250 ms in queue");
  assert.match(waited.detail, /Observed, not causal/);
  assert.equal(headlineFor([], null), null);
});

test("timings pick ms or s and never round a measured value to zero", () => {
  assert.equal(formatTiming(null).text, "—");
  assert.equal(formatTiming(null).hasValue, false);
  assert.equal(formatTiming(0).text, "0 ms");
  assert.equal(formatTiming(0.0004).text, "< 1 ms");
  assert.equal(formatTiming(0.069).text, "69 ms");
  assert.equal(formatTiming(2.18).text, "2.18 s");
  assert.equal(formatTiming(Number.NaN).hasValue, false);
});

test("bars and the selected-step split refuse missing readings", () => {
  assert.deepEqual(rowBarShares({ queueSeconds: 1, executionSeconds: 3 }, 8), {
    queue: 0.125,
    execution: 0.375,
  });
  assert.equal(
    rowBarShares({ queueSeconds: null, executionSeconds: 3 }, 8),
    null,
  );
  assert.equal(rowBarShares({ queueSeconds: 1, executionSeconds: 3 }, 0), null);
  assert.equal(
    contextSegments({ queueSeconds: 0.5, executionSeconds: null }),
    null,
  );
  const small = contextSegments({
    queueSeconds: 0.069,
    executionSeconds: 0.2,
  })!;
  assert.equal(small.unit, "ms");
  assert.deepEqual(
    small.segments.map((s) => [s.id, Math.round(s.value), s.kind]),
    [
      ["queue", 69, "wait"],
      ["execution", 200, "work"],
    ],
  );
  assert.equal(
    contextSegments({ queueSeconds: 0.5, executionSeconds: 2 })!.unit,
    "s",
  );
});
