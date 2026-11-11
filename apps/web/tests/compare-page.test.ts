import assert from "node:assert/strict";
import test from "node:test";
import {
  alignSteps,
  buildComparePage,
  compareConfiguration,
  compareHref,
  compareTab,
  formatStoredAt,
  headline,
  interpretation,
  largestDifferences,
  metricDeltas,
  stepLabel,
  timingDelta,
} from "../lib/compare-page";
import type { DashboardData, Experiment, Step } from "../lib/experiment-schema";

// Synthetic runs for tests only; they are not recorded-run evidence.
function step(stepId: string, queue: number, execution: number): Step {
  return {
    step_id: stepId,
    submitted_ns: 0,
    started_ns: Math.round(queue * 1e9),
    finished_ns: Math.round((queue + execution) * 1e9),
    duration_seconds: queue + execution,
    queue_seconds: queue,
    execution_seconds: execution,
    cpu_seconds: execution * 0.9,
    queue_fraction: queue / (queue + execution),
    resource: "none",
    checksum: "abc",
    evidence: "timestamps",
    rule_version: "cpu-queue-v1",
    threshold_seconds: 0.01,
  };
}

function run(
  id: string,
  overrides: Partial<Experiment> = {},
  scale = 1,
): Experiment {
  return {
    run_id: id,
    kind: "coding",
    experiment: {
      worker_capacity: 1,
      iterations: 600000,
      baseline_repeats: 3,
      workload: "fixed Python project",
      tests_per_step: 3,
    },
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [0, 1, 2].map((i) => step(`baseline-${i}`, 0.001, 0.5 * scale)),
    },
    clean: step("clean", 0.001, 0.5 * scale),
    contended: step("contended", 0.4 * scale, 0.6 * scale),
    checks: { passed: true },
    limitations: ["CPU worker queue only; not host saturation"],
    persisted: true,
    stored_at: "2026-11-11T00:20:44.413890+00:00",
    ...overrides,
  };
}

const data = (experiments: Experiment[]): DashboardData => ({
  experiments,
  warnings: [],
  source: "database",
});

test("timingDelta keeps missing readings distinct from zero", () => {
  assert.deepEqual(timingDelta(null, 2), {
    baselineSeconds: null,
    candidateSeconds: 2,
    deltaSeconds: null,
    ratio: null,
  });
  assert.equal(timingDelta(Number.NaN, 2).deltaSeconds, null);
  // A measured zero baseline gives a delta but no ratio.
  assert.deepEqual(timingDelta(0, 0.5), {
    baselineSeconds: 0,
    candidateSeconds: 0.5,
    deltaSeconds: 0.5,
    ratio: null,
  });
  assert.equal(timingDelta(2, 5).ratio, 2.5);
});

test("compareConfiguration matches identical settings", () => {
  const result = compareConfiguration(run("a"), run("b"));
  assert.equal(result.allMatch, true);
  assert.equal(result.matchingWorkload, true);
  assert.deepEqual(result.differing, []);
});

test("compareConfiguration exposes each differing setting", () => {
  const candidate = run("b", {
    kind: "cpu",
    experiment: {
      worker_capacity: 2,
      iterations: 600000,
      baseline_repeats: 3,
      workload: "fixed SHA-256 CPU task",
      tests_per_step: null,
    },
    clean: { ...step("clean", 0.001, 0.5), checksum: "def" },
  });
  const result = compareConfiguration(run("a"), candidate);
  assert.equal(result.sameKind, false);
  assert.equal(result.matchingWorkload, false);
  assert.deepEqual(result.differing, [
    "Workload kind",
    "Workload",
    "Worker capacity",
    "Tests per step",
    "Clean output checksum",
  ]);
  const capacity = result.checks.find((c) => c.id === "capacity");
  assert.equal(capacity?.baseline, "1 worker");
  assert.equal(capacity?.candidate, "2 workers");
});

test("checksum compares full values though it displays shortened", () => {
  const long = (middle: string) =>
    `319bfd57${middle}11b5a258d8e9e9ae651e56bb3ef84`;
  const a = run("a", {
    clean: { ...step("clean", 0, 1), checksum: long("aa") },
  });
  const b = run("b", {
    clean: { ...step("clean", 0, 1), checksum: long("bb") },
  });
  const check = compareConfiguration(a, b).checks.find(
    (c) => c.id === "checksum",
  );
  assert.equal(check?.baseline, check?.candidate);
  assert.equal(check?.matches, false);
  assert.equal(check?.baselineTitle, long("aa"));
});

test("agent model mismatch when only one run has agent evidence", () => {
  const agent = {
    model: "qwen3:1.7b",
    status: "passed" as const,
    limitations: [],
    messages: [],
    inference: [
      { request_index: 0, wall_seconds: 1.5, metrics: {} },
      { request_index: 1, wall_seconds: 0.5, metrics: {} },
    ],
  };
  const withAgent = run("b", { agent });
  const result = compareConfiguration(run("a"), withAgent);
  assert.deepEqual(result.differing, ["Agent model"]);
  const inference = metricDeltas(run("a"), withAgent).find(
    (m) => m.id === "inference",
  );
  // No agent evidence is unavailable, never a measured 0 s.
  assert.equal(inference?.baselineSeconds, null);
  assert.equal(inference?.candidateSeconds, 2);
  assert.equal(inference?.deltaSeconds, null);
});

test("metricDeltas totals every step and subtracts candidate minus baseline", () => {
  const metrics = metricDeltas(run("a"), run("b", {}, 2));
  const total = metrics.find((m) => m.id === "total");
  // Baseline: 3 × 0.501 + 0.501 + 1.0 = 3.004; candidate doubles execution and contended queue.
  assert.ok(total);
  assert.ok(Math.abs((total.baselineSeconds ?? 0) - 3.004) < 1e-9);
  assert.ok(Math.abs((total.candidateSeconds ?? 0) - 6.004) < 1e-9);
  assert.ok(Math.abs((total.deltaSeconds ?? 0) - 3) < 1e-9);
  assert.equal(total.scope, "5 steps");
  // Neither run has agent evidence, so there is no inference row at all.
  assert.equal(
    metrics.some((m) => m.id === "inference"),
    false,
  );
  const queue = metrics.find((m) => m.id === "contended-queue");
  assert.ok(Math.abs((queue?.deltaSeconds ?? 0) - 0.4) < 1e-9);
});

test("alignSteps pairs by step id and keeps one-sided steps", () => {
  const candidate = run("b", {
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [0, 1, 2, 3].map((i) => step(`baseline-${i}`, 0.001, 0.5)),
    },
  });
  const baseline = run("a", {
    baseline: {
      execution_cv: 0.01,
      queue_threshold_seconds: 0.01,
      steps: [
        step("baseline-0", 0.001, 0.5),
        step("baseline-1", 0.001, 0.5),
        step("baseline-x", 0.001, 0.5),
      ],
    },
  });
  const aligned = alignSteps(baseline, candidate);
  assert.deepEqual(
    aligned.map((s) => s.stepId),
    [
      "baseline-0",
      "baseline-1",
      "baseline-x",
      "clean",
      "contended",
      "baseline-2",
      "baseline-3",
    ],
  );
  const onlyBaseline = aligned.find((s) => s.stepId === "baseline-x");
  assert.equal(onlyBaseline?.candidate, null);
  assert.equal(onlyBaseline?.duration.deltaSeconds, null);
  const onlyCandidate = aligned.find((s) => s.stepId === "baseline-3");
  assert.equal(onlyCandidate?.baseline, null);
  assert.equal(onlyCandidate?.duration.candidateSeconds, 0.501);
  // Missing steps never rank as a difference.
  assert.ok(
    largestDifferences(aligned, 10).every((s) => s.baseline && s.candidate),
  );
});

test("largestDifferences ranks by absolute duration change", () => {
  const faster = run("b", { contended: step("contended", 0.05, 0.6) });
  const ranked = largestDifferences(alignSteps(run("a"), faster));
  assert.equal(ranked[0]?.stepId, "contended");
  assert.ok((ranked[0]?.duration.deltaSeconds ?? 0) < 0);
  assert.equal(ranked.length, 3);
});

test("stepLabel names recorded step ids", () => {
  assert.equal(stepLabel("baseline-0"), "Baseline repeat 1");
  assert.equal(stepLabel("clean"), "Clean step");
  assert.equal(stepLabel("contended"), "Contended step");
  assert.equal(stepLabel("custom"), "custom");
});

test("headline states direction at display precision", () => {
  const metric = (ratio: number | null) => ({
    id: "total",
    label: "Total",
    scope: "5 steps",
    baselineSeconds: 1,
    candidateSeconds: ratio,
    deltaSeconds: ratio === null ? null : ratio - 1,
    ratio,
  });
  assert.equal(headline(metric(11.4))?.title, "Candidate is 11.4× slower");
  assert.equal(headline(metric(0.5))?.title, "Candidate is 2.0× faster");
  assert.equal(headline(metric(1.02))?.direction, "same");
  assert.equal(headline(metric(null)), null);
});

test("interpretation is observational and names differing settings", () => {
  const same = run("a");
  const metrics = metricDeltas(same, run("b", {}, 2));
  const matched = interpretation(compareConfiguration(same, run("b")), metrics);
  assert.match(matched, /observed, not causal/);
  assert.doesNotMatch(matched, /because|caused/i);
  const differing = interpretation(
    compareConfiguration(
      same,
      run("b", { experiment: { ...same.experiment, worker_capacity: 4 } }),
    ),
    metrics,
  );
  assert.match(differing, /Settings differ \(worker capacity\)/);
  const otherKind = interpretation(
    compareConfiguration(same, run("b", { kind: "cpu" })),
    metrics,
  );
  assert.match(otherKind, /not comparable/);
});

test("compareTab accepts only the page's tab ids", () => {
  assert.equal(compareTab("steps"), "steps");
  assert.equal(compareTab("configuration"), "configuration");
  assert.equal(compareTab("summary"), "summary");
  assert.equal(compareTab("timeline"), undefined);
  assert.equal(compareTab(""), undefined);
  assert.equal(compareTab(null), undefined);
});

test("compareHref keeps a known non-default tab with the chosen runs", () => {
  assert.equal(
    compareHref("cand", "base", "steps"),
    "/compare?run=cand&baseline=base&view=steps",
  );
  // The default tab and unknown values are dropped rather than carried along.
  assert.equal(
    compareHref("cand", "base", "summary"),
    "/compare?run=cand&baseline=base",
  );
  assert.equal(compareHref("cand", undefined, "bogus"), "/compare?run=cand");
  assert.equal(
    compareHref(undefined, undefined, "configuration"),
    "/compare?view=configuration",
  );
  assert.equal(compareHref(undefined, undefined, null), "/compare");
});

test("buildComparePage uses only the requested ids", () => {
  const newest = run("new", { stored_at: "2026-11-12T00:00:00+00:00" });
  const page = buildComparePage(data([newest, run("a"), run("b")]), {
    run: "a",
    baseline: "missing-id",
  });
  assert.equal(page.candidate.status, "selected");
  assert.deepEqual(page.baseline, {
    status: "not-found",
    requestedId: "missing-id",
  });
  // The newest run is never substituted for a missing baseline.
  assert.equal(page.comparison, null);
});

test("buildComparePage leaves an unchosen baseline unchosen", () => {
  const page = buildComparePage(data([run("a"), run("b")]), { run: "a" });
  assert.deepEqual(page.baseline, { status: "unselected" });
  assert.equal(page.comparison, null);
  assert.deepEqual(
    page.baselineOptions.map((o) => o.id),
    ["b"],
  );
});

test("buildComparePage refuses comparing a run with itself", () => {
  const page = buildComparePage(data([run("a"), run("b")]), {
    run: "a",
    baseline: "a",
  });
  assert.equal(page.sameRun, true);
  assert.equal(page.comparison, null);
});

test("buildComparePage compares two chosen runs", () => {
  const page = buildComparePage(data([run("a"), run("b", {}, 2)]), {
    run: "b",
    baseline: "a",
  });
  assert.equal(page.comparison?.candidate.id, "b");
  assert.equal(page.comparison?.baseline.id, "a");
  assert.equal(page.comparison?.headline?.direction, "slower");
  assert.equal(
    page.comparison?.baseline.evidenceHref,
    "/api/experiments/coding/artifact?file=results.json&run_id=a",
  );
  assert.deepEqual(page.comparison?.limitations, [
    "CPU worker queue only; not host saturation",
  ]);
});

test("formatStoredAt formats in UTC and rejects bad input", () => {
  assert.equal(
    formatStoredAt("2026-11-11T00:20:44.413890+00:00"),
    "2026-11-11 00:20 UTC",
  );
  assert.equal(formatStoredAt("not a date"), null);
  assert.equal(formatStoredAt(undefined), null);
});
