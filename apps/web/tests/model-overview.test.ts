import assert from "node:assert/strict";
import test from "node:test";
import type { DatasetRun } from "../lib/dataset-runs";
import { buildModelOverview, modelDisplayName } from "../lib/model-overview";
import type { Ready } from "../lib/overview-model";

// Synthetic runs for tests only; they are not recorded-run evidence.
const run = (
  runId: string,
  overrides: Partial<DatasetRun> = {},
): DatasetRun => ({
  runId,
  kind: "agent_replay",
  workload: "replay",
  taskGroup: "task-a",
  experiment: "exp-1",
  replayCondition: "clean",
  replayIndex: 1,
  injectionMethod: "none",
  agentModel: "qwen3:1.7b",
  recordedAt: null,
  passed: true,
  steps: 3,
  durationSeconds: 2,
  queueSeconds: 0.1,
  executionSeconds: 1.9,
  excessSteps: 0,
  labels: { none: 3 },
  stepTypes: { model_call: 2, tool: 1 },
  excessByStepType: {},
  topLabel: "none",
  telemetryPresent: 1,
  ...overrides,
});

const snapshot = (runs: DatasetRun[]): Ready =>
  ({
    status: "ready",
    runs,
    manifest: { dataset_version: "steps-v9.9+abc123" },
  }) as unknown as Ready;

const runs = [
  run("clean-1", { durationSeconds: 2, recordedAt: "2026-01-02T00:00:00Z" }),
  run("clean-2", { durationSeconds: 4 }),
  run("mixed-1", {
    replayCondition: "mixed",
    durationSeconds: 30,
    passed: false,
    recordedAt: "2026-01-03T00:00:00Z",
    labels: { none: 1, mixed: 2 },
    stepTypes: { model_call: 1, retrieval: 1, planner: 1 },
    excessByStepType: { retrieval: 1, planner: 1 },
    telemetryPresent: 0,
  }),
  run("cpu-1", { replayCondition: "cpu", durationSeconds: 6, passed: null }),
  run("other-model", { agentModel: "llama3:8b", passed: false }),
  run("coding", { agentModel: null, kind: "coding", experiment: null }),
];

test("buildModelOverview defaults to the model with the most runs and counts only its runs", () => {
  const overview = buildModelOverview(snapshot(runs))!;
  assert.equal(overview.model.id, "qwen3:1.7b");
  assert.equal(overview.model.name, "Qwen3 1.7B");
  assert.deepEqual(
    overview.models.map((m) => [m.id, m.runs]),
    [
      ["qwen3:1.7b", 4],
      ["llama3:8b", 1],
    ],
  );
  assert.equal(overview.runs, 4);
  assert.equal(overview.steps, 12);
  assert.equal(overview.telemetryPresent, 3);
  assert.equal(overview.runsWithoutModel, 1);
  assert.equal(overview.releaseName, "steps-v9.9");
});

test("unrecorded outcomes are kept apart from passed and failed", () => {
  const overview = buildModelOverview(snapshot(runs))!;
  assert.deepEqual(overview.outcomes, { passed: 2, failed: 1, unrecorded: 1 });
});

test("call types list the known types first, then any others, and resource labels skip no-excess", () => {
  const overview = buildModelOverview(snapshot(runs))!;
  assert.deepEqual(
    overview.callTypes.map((c) => [c.id, c.value]),
    [
      ["model_call", 7],
      ["tool", 3],
      ["retrieval", 1],
      ["planner", 1],
    ],
  );
  assert.ok(overview.resourceLabels.every((label) => label.id !== "none"));
  assert.equal(
    overview.resourceLabels.find((label) => label.id === "mixed")?.value,
    2,
  );
  // A resource with no labelled steps is a measured zero, not missing.
  assert.equal(
    overview.resourceLabels.find((label) => label.id === "inference")?.value,
    0,
  );
});

test("largest slowdown is the injected condition with the highest ratio over its own clean median", () => {
  const overview = buildModelOverview(snapshot(runs))!;
  // Clean median of exp-1 is 3 s; mixed is 30 s (10×), cpu 6 s (2×).
  assert.deepEqual(overview.largestSlowdown, {
    condition: "mixed",
    label: "Mixed",
    ratio: 10,
    experiments: 1,
  });
});

test("largest slowdown is null without a clean baseline", () => {
  const noClean = runs.filter((r) => r.replayCondition !== "clean");
  assert.equal(buildModelOverview(snapshot(noClean))!.largestSlowdown, null);
});

test("failed runs link to a live run only when the import map names one", () => {
  const live = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
  const overview = buildModelOverview(snapshot(runs), { "mixed-1": live })!;
  assert.deepEqual(overview.failedRuns, [
    {
      runId: "mixed-1",
      liveId: live,
      task: "task-a",
      recordedAt: "2026-01-03T00:00:00Z",
    },
  ]);
  assert.equal(buildModelOverview(snapshot(runs))!.failedRuns[0].liveId, null);
});

test("a requested model is honoured, an unknown one falls back to the default", () => {
  assert.equal(
    buildModelOverview(snapshot(runs), {}, "llama3:8b")!.model.id,
    "llama3:8b",
  );
  assert.equal(
    buildModelOverview(snapshot(runs), {}, "nope")!.model.id,
    "qwen3:1.7b",
  );
});

test("a release with no model-tagged runs has no model overview", () => {
  assert.equal(buildModelOverview(snapshot([runs[5]])), null);
});

test("modelDisplayName formats family:size ids and leaves other ids as recorded", () => {
  assert.equal(modelDisplayName("qwen3:0.6b"), "Qwen3 0.6B");
  assert.equal(modelDisplayName("gpt-4o"), "gpt-4o");
});

test("stages pair each call type's steps with its excess-wait steps", () => {
  const overview = buildModelOverview(snapshot(runs))!;
  assert.deepEqual(overview.stages, {
    model: { calls: 7, excessSteps: 0 },
    retrieval: { calls: 1, excessSteps: 1 },
    cpuTool: { calls: 3, excessSteps: 0 },
  });
});
