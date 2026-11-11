import assert from "node:assert/strict";
import test from "node:test";
import type { Experiment, Step } from "../lib/experiment-schema";
import {
  parseSamples,
  pressureSeries,
  stepWindow,
  summarizePressure,
  unavailableSignals,
} from "../lib/run-samples";
import {
  savedTimeline,
  tickLabel,
  timelineTicks,
  wallTimeline,
} from "../lib/run-timeline";

// Synthetic runs and samples for tests only; they are not recorded-run evidence.
const SECOND_NS = 1e9;
const START_NS = Date.UTC(2026, 10, 11, 10, 0, 0) * 1e6;

function step(
  id: string,
  submittedAt: number,
  queue: number,
  execution: number,
  overrides: Partial<Step> = {},
): Step {
  const submitted = START_NS + submittedAt * SECOND_NS;
  return {
    step_id: id,
    submitted_ns: submitted,
    started_ns: submitted + queue * SECOND_NS,
    finished_ns: submitted + (queue + execution) * SECOND_NS,
    duration_seconds: queue + execution,
    queue_seconds: queue,
    execution_seconds: execution,
    cpu_seconds: execution,
    queue_fraction: queue / (queue + execution),
    resource: "none",
    checksum: "abc",
    evidence: "direct worker enqueue/start timestamps",
    rule_version: "cpu-queue-v1",
    threshold_seconds: 0.01,
    ...overrides,
  };
}

function run(): Experiment {
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
      // Deliberately out of submission order: the timeline sorts by submission.
      steps: [
        step("baseline-1", 0.5, 0, 0.5),
        step("baseline-0", 0, 0, 0.5),
        step("baseline-2", 1, 0, 0.5),
      ],
    },
    clean: step("clean", 1.5, 0, 0.5),
    contended: step("contended", 2, 1.5, 0.5, { resource: "cpu_sandbox" }),
    checks: { passed: true },
    limitations: [],
  };
}

test("timelineTicks picks a round spacing and ends at or before the span", () => {
  assert.deepEqual(
    timelineTicks(4).map((tick) => tick.seconds),
    [0, 1, 2, 3, 4],
  );
  assert.deepEqual(
    timelineTicks(2.4).map((tick) => tick.seconds),
    [0, 0.5, 1, 1.5, 2],
  );
  const ticks = timelineTicks(2.4);
  assert.equal(ticks.at(-1)?.share, 2 / 2.4);
  assert.deepEqual(timelineTicks(0), [{ seconds: 0, share: 0 }]);
  assert.deepEqual(timelineTicks(Number.NaN), [{ seconds: 0, share: 0 }]);
});

test("tick labels carry only the decimals their spacing needs", () => {
  const halves = timelineTicks(2.4);
  assert.deepEqual(
    halves.map((tick) => tickLabel(tick, halves)),
    ["0.0 s", "0.5 s", "1.0 s", "1.5 s", "2.0 s"],
  );
  const whole = timelineTicks(4);
  assert.equal(tickLabel(whole[2], whole), "2 s");
});

test("savedTimeline orders spans by submission and splits queue from execution", () => {
  const timeline = savedTimeline(run());
  assert.deepEqual(
    timeline.rows.map((row) => row.step.step_id),
    ["baseline-0", "baseline-1", "baseline-2", "clean", "contended"],
  );
  assert.equal(timeline.originNs, START_NS);
  assert.equal(timeline.spanSeconds, 4);
  const contended = timeline.rows.at(-1)!;
  assert.equal(contended.phase, "contended");
  assert.equal(contended.startShare, 0.5);
  assert.deepEqual(contended.segments, [
    { kind: "wait", share: 1.5 / 4 },
    { kind: "work", share: 0.5 / 4 },
  ]);
  // Every bar ends inside the window.
  for (const row of timeline.rows) {
    const end =
      row.startShare + row.segments.reduce((sum, s) => sum + s.share, 0);
    assert.ok(end <= 1 + 1e-12, `${row.step.step_id} ends inside the window`);
  }
});

test("wallTimeline leaves running steps open with no duration", () => {
  const timeline = wallTimeline([
    {
      stepId: "b",
      stepType: "tool",
      state: "running",
      startMs: 1_500,
      endMs: null,
    },
    {
      stepId: "a",
      stepType: "model_call",
      state: "completed",
      startMs: 1_000,
      endMs: 2_000,
    },
  ]);
  assert.ok(timeline);
  assert.deepEqual(
    timeline.rows.map((row) => row.step.stepId),
    ["a", "b"],
  );
  assert.equal(timeline.rows[0].wallSeconds, 1);
  assert.equal(timeline.rows[1].open, true);
  assert.equal(timeline.rows[1].wallSeconds, null);
  assert.equal(timeline.rows[1].startShare, 0.5);
  assert.equal(wallTimeline([]), null);
});

const sampleLine = (atSeconds: number, avg10: number | null, missing = []) =>
  JSON.stringify({
    timestamp_ns: START_NS + atSeconds * SECOND_NS,
    cpu_pressure: avg10 === null ? null : { some: { avg10 } },
    cgroup_cpu_stat: null,
    unavailable: avg10 === null ? ["cpu_pressure", ...missing] : missing,
  });

test("parseSamples rejects the whole file on one malformed line", () => {
  const good = [sampleLine(0, 1), "", sampleLine(1, 2)].join("\n");
  const parsed = parseSamples(good);
  assert.ok(parsed.ok);
  assert.equal(parsed.samples.length, 2);

  const notJson = parseSamples(`${sampleLine(0, 1)}\n{oops`);
  assert.deepEqual(notJson, {
    ok: false,
    message: "samples.jsonl line 2 is not JSON.",
  });
  const wrongShape = parseSamples(JSON.stringify({ timestamp_ns: -1 }));
  assert.equal(wrongShape.ok, false);
});

test("missing pressure readings stay gaps and never lower the summary", () => {
  const parsed = parseSamples(
    [sampleLine(2, 4), sampleLine(0, null), sampleLine(1, 2)].join("\n"),
  );
  assert.ok(parsed.ok);
  const points = pressureSeries(parsed.samples, START_NS);
  assert.deepEqual(points, [
    { seconds: 0, pressure: null },
    { seconds: 1, pressure: 2 },
    { seconds: 2, pressure: 4 },
  ]);
  assert.deepEqual(summarizePressure(points), {
    readings: 2,
    peak: 4,
    peakSeconds: 2,
    mean: 3,
  });
  assert.equal(summarizePressure([{ seconds: 0, pressure: null }]), null);
  assert.deepEqual(unavailableSignals(parsed.samples), [
    { signal: "cpu_pressure", count: 1 },
  ]);
});

test("stepWindow places a step on the same origin as the samples", () => {
  const contended = run().contended;
  assert.deepEqual(stepWindow(contended, START_NS), { from: 2, to: 4 });
});

test("formatSignal sizes precision to the reading and keeps % attached", async () => {
  const { formatSignal, latestReading } = await import("../lib/signal-format");
  assert.equal(formatSignal(35.1, "%"), "35.1%");
  assert.equal(formatSignal(8.514, "ms"), "8.51 ms");
  assert.equal(formatSignal(245.6, "W"), "246 W");
  assert.equal(formatSignal(0, "ms"), "0.00 ms");
  // Axis ticks drop trailing zeros and abbreviate thousands.
  assert.equal(formatSignal(25, "%", true), "25%");
  assert.equal(formatSignal(15, "", true), "15");
  assert.equal(formatSignal(12_500, "", true), "12.5K");
  assert.equal(formatSignal(Number.NaN, "ms"), "—");
  // The latest reading skips trailing gaps; all-missing has none.
  assert.deepEqual(
    latestReading([{ value: 1 }, { value: 2 }, { value: null }]),
    { value: 2, index: 1 },
  );
  assert.equal(latestReading([{ value: null }]), null);
});
