import assert from "node:assert/strict";
import test from "node:test";
import {
  conditionRatios,
  summarizeExperiments,
  type DatasetRun,
} from "../lib/dataset-runs";
import {
  byNewest,
  diagnoseLive,
  filterRuns,
  fromLive,
  stateGroup,
  type UnifiedRun,
} from "../lib/unified-runs";
import type { LiveRunSnapshot } from "../lib/live-api-schema";

// Synthetic rows for tests only; they are not recorded-run evidence.
const run = (overrides: Partial<UnifiedRun> = {}): UnifiedRun => ({
  id: "11111111-1111-4111-8111-111111111111",
  source: "live",
  title: "fix-window-count-r1",
  kind: "agent",
  state: "completed",
  at: "2026-11-05T17:00:00.000Z",
  steps: null,
  waitingShare: null,
  topResource: null,
  ...overrides,
});

test("states fold into the three groups the Runs filter offers", () => {
  assert.equal(stateGroup("running"), "in_progress");
  assert.equal(stateGroup("passed"), "done");
  assert.equal(stateGroup("completed"), "done");
  assert.equal(stateGroup("inconclusive"), "needs_attention");
  assert.equal(stateGroup("cancelled"), "needs_attention");
});

test("filters combine and ignore case in the search", () => {
  const runs = [
    run(),
    run({
      id: "2",
      source: "saved",
      title: "Python test tool",
      kind: "coding",
      state: "inconclusive",
    }),
    run({
      id: "3",
      source: "saved",
      title: "CPU replay",
      kind: "cpu",
      state: "passed",
    }),
  ];
  const all = { source: "all", group: "all", kind: "all", query: "" } as const;
  assert.equal(filterRuns(runs, all).length, 3);
  assert.deepEqual(
    filterRuns(runs, { ...all, source: "saved", group: "needs_attention" }).map(
      (r) => r.id,
    ),
    ["2"],
  );
  assert.deepEqual(
    filterRuns(runs, { ...all, query: "PYTHON" }).map((r) => r.id),
    ["2"],
  );
});

test("newest first, with the id as a stable tie-break", () => {
  const sorted = [
    run({ id: "b", at: "2026-01-01T00:00:00.000Z" }),
    run({ id: "a", at: "2026-01-01T00:00:00.000Z" }),
    run({ id: "c", at: "2026-02-01T00:00:00.000Z" }),
  ].sort(byNewest);
  assert.deepEqual(
    sorted.map((r) => r.id),
    ["c", "a", "b"],
  );
});

test("a live summary becomes a run with no invented waiting share or step count", () => {
  const live = fromLive({
    runId: "11111111-1111-4111-8111-111111111111",
    taskId: "fix-mean-empty-r1",
    state: "running",
    createdAt: "2026-11-05T17:00:00.000Z",
  });
  assert.equal(live.source, "live");
  assert.equal(live.waitingShare, null);
  assert.equal(live.steps, null);
});

const snapshot = {
  registration: {
    resources: [
      {
        resource_id: "tool-host",
        service_id: "s",
        resource_class: "cpu_sandbox",
      },
    ],
  },
  steps: [
    {
      event: {
        step_id: "fast",
        step_type: "tool",
        state: "completed",
        resource_id: "tool-host",
      },
      timings: {
        duration_seconds: 1,
        queue_seconds: 0.1,
        execution_seconds: 0.9,
        queue_fraction: 0.1,
      },
    },
    {
      event: {
        step_id: "unmeasured",
        step_type: "other",
        state: "failed",
        resource_id: null,
      },
      timings: {
        duration_seconds: 0.001,
        queue_seconds: null,
        execution_seconds: null,
        queue_fraction: null,
      },
    },
    {
      event: {
        step_id: "slow",
        step_type: "model_call",
        state: "completed",
        resource_id: "tool-host",
      },
      timings: {
        duration_seconds: 5,
        queue_seconds: 2,
        execution_seconds: 3,
        queue_fraction: 0.4,
      },
    },
  ],
} as unknown as LiveRunSnapshot;

test("live diagnosis ranks by waiting, puts unmeasured steps last, and never calls a resource an attribution", () => {
  const steps = diagnoseLive(snapshot);
  assert.deepEqual(
    steps.map((s) => s.stepId),
    ["slow", "fast", "unmeasured"],
  );
  assert.equal(steps[0].resource, "cpu_sandbox");
  assert.equal(steps[0].basis, "ran_on");
  assert.equal(steps[2].queueSeconds, null);
  assert.equal(steps[2].basis, "none");
});

const dataset = (overrides: Partial<DatasetRun>): DatasetRun => ({
  runId: "r",
  kind: "agent_replay",
  workload: "w",
  taskGroup: "task-a",
  experiment: "exp-1",
  replayCondition: "clean",
  replayIndex: 1,
  injectionMethod: "none",
  agentModel: null,
  recordedAt: "2026-11-05T10:00:00.000Z",
  passed: true,
  steps: 7,
  durationSeconds: 2,
  queueSeconds: 0.1,
  executionSeconds: 1.9,
  excessSteps: 0,
  labels: {},
  stepTypes: {},
  excessByStepType: {},
  topLabel: "none",
  telemetryPresent: 0,
  ...overrides,
});

test("experiments group replays, take per-condition medians and skip non-replay runs", () => {
  const summaries = summarizeExperiments([
    dataset({ durationSeconds: 1 }),
    dataset({ durationSeconds: 3 }),
    dataset({ durationSeconds: 2 }),
    dataset({ replayCondition: "inference", durationSeconds: 10 }),
    dataset({ experiment: "exp-2", recordedAt: "2026-11-06T10:00:00.000Z" }),
    dataset({ experiment: null, kind: "coding" }),
  ]);
  assert.deepEqual(
    summaries.map((s) => s.experiment),
    ["exp-2", "exp-1"],
  );
  const first = summaries[1];
  assert.equal(first.replays, 4);
  assert.deepEqual(first.medians, { clean: 2, inference: 10 });
  assert.equal(summaries[0].medians.cpu, undefined);
});

test("condition ratios compare each experiment to its own clean median, not across tasks", () => {
  const ratios = conditionRatios([
    {
      experiment: "a",
      taskGroup: "t1",
      replays: 3,
      recordedAt: null,
      medians: { clean: 2, inference: 10 },
    },
    {
      experiment: "b",
      taskGroup: "t2",
      replays: 3,
      recordedAt: null,
      medians: { clean: 10, inference: 30, cpu: 11 },
    },
    {
      experiment: "c",
      taskGroup: "t2",
      replays: 1,
      recordedAt: null,
      medians: { inference: 99 },
    },
  ]);
  assert.deepEqual(ratios, [
    { condition: "cpu", ratio: 1.1, experiments: 1 },
    { condition: "inference", ratio: 4, experiments: 2 },
  ]);
  assert.deepEqual(conditionRatios([]), []);
});
