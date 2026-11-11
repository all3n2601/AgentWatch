import assert from "node:assert/strict";
import test from "node:test";
import {
  campusStages,
  campusTraffic,
  excessTone,
  failureTone,
  spreadFlags,
} from "../lib/campus-status";
import type { ModelOverview } from "../lib/model-overview";

// Synthetic overview for tests only; it is not recorded-run evidence.
const overview = (overrides: Partial<ModelOverview> = {}): ModelOverview => ({
  release: "steps-test+abc",
  releaseName: "steps-test",
  model: { id: "qwen3:1.7b", name: "Qwen3 1.7B", runs: 100 },
  models: [],
  runs: 100,
  steps: 1000,
  outcomes: { passed: 90, failed: 10, unrecorded: 0 },
  telemetryPresent: 50,
  callTypes: [],
  resourceLabels: [],
  stages: {
    model: { calls: 500, excessSteps: 50 },
    retrieval: { calls: 100, excessSteps: 54 },
    cpuTool: { calls: 400, excessSteps: 0 },
  },
  largestSlowdown: null,
  failedRuns: [],
  runsWithoutModel: 0,
  ...overrides,
});

test("excess and failure tones follow the declared thresholds", () => {
  assert.equal(excessTone(0), "success");
  assert.equal(excessTone(0.05), "warning");
  assert.equal(excessTone(0.15), "caution");
  assert.equal(excessTone(0.35), "danger");
  assert.equal(failureTone(0), "success");
  assert.equal(failureTone(0.01), "warning");
  assert.equal(failureTone(0.05), "caution");
  assert.equal(failureTone(0.15), "danger");
});

test("stage status comes from each stage's own calls and excess-wait steps", () => {
  const stages = campusStages(overview());
  assert.equal(stages.retrieval.tone, "danger");
  assert.match(stages.retrieval.summary, /^54\.0% of retrieval calls/);
  assert.equal(stages.model.tone, "warning");
  assert.equal(stages.model.name, "Qwen3 1.7B runtime");
  // A measured zero is a healthy stage, not a missing one.
  assert.equal(stages.cpuTool.tone, "success");
  assert.equal(stages.complete.tone, "caution");
  assert.equal(stages.complete.summary, "10 of 100 runs failed");
});

test("a stage without steps has no tone instead of looking healthy", () => {
  const stages = campusStages(
    overview({
      stages: {
        model: { calls: 0, excessSteps: 0 },
        retrieval: { calls: 0, excessSteps: 0 },
        cpuTool: { calls: 0, excessSteps: 0 },
      },
      outcomes: { passed: 0, failed: 0, unrecorded: 5 },
    }),
  );
  assert.equal(stages.model.tone, null);
  assert.equal(stages.complete.tone, null);
  assert.ok(
    stages.complete.facts.some((fact) => fact.label === "Outcome not recorded"),
  );
});

test("traffic shares match the stage counts, and unrecorded outcomes are not counted as passed", () => {
  const traffic = campusTraffic(
    overview({ outcomes: { passed: 90, failed: 10, unrecorded: 900 } }),
  );
  assert.equal(traffic.retrieval.flagShare, 0.54);
  assert.equal(traffic.tool.flagShare, 0);
  assert.equal(traffic.egress.flagShare, 0.1);
  assert.equal(traffic.ingress.flagShare, 0);
});

test("spreadFlags flags the rounded share, evenly, and keeps a rare flag visible", () => {
  const half = spreadFlags(10, 0.5);
  assert.equal(half.filter(Boolean).length, 5);
  assert.deepEqual(half.slice(0, 4), [false, true, false, true]);
  assert.equal(spreadFlags(40, 0.022).filter(Boolean).length, 1);
  assert.equal(spreadFlags(10, 0).filter(Boolean).length, 0);
  assert.equal(spreadFlags(10, 1).filter(Boolean).length, 10);
  assert.deepEqual(spreadFlags(0, 0.5), []);
  assert.equal(spreadFlags(10, Number.NaN).filter(Boolean).length, 0);
});
