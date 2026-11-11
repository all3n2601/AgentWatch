import assert from "node:assert/strict";
import test from "node:test";
import type { DatasetRun } from "../lib/dataset-runs";
import {
  buildExperimentsPage,
  checkTally,
  coefficientOfVariation,
  conditionReading,
  ratioAxis,
  shortExperimentId,
  type ExperimentsPageInput,
  type ReplayStep,
} from "../lib/experiments-page";

// Synthetic runs for tests only; they are not recorded-run evidence.
const run = (
  runId: string,
  overrides: Partial<DatasetRun> = {},
): DatasetRun => ({
  runId,
  kind: "agent_replay",
  workload: "replay",
  taskGroup: "task-a",
  experiment: "replay-task-a-r1-1",
  replayCondition: "clean",
  replayIndex: 1,
  injectionMethod: "none",
  agentModel: "model-x",
  recordedAt: null,
  passed: true,
  steps: 2,
  durationSeconds: 2,
  queueSeconds: 0,
  executionSeconds: 2,
  excessSteps: 0,
  labels: { none: 2 },
  stepTypes: {},
  excessByStepType: {},
  topLabel: "none",
  telemetryPresent: 1,
  ...overrides,
});

/** One experiment: baseline at 1–2, injected cpu at 3, clean control at 4, mixed at 5. */
function experiment(
  id: string,
  taskGroup: string,
  clean: [number, number, number],
  cpu: number,
  mixed: number,
  extra: Partial<DatasetRun> = {},
): DatasetRun[] {
  const base = { experiment: id, taskGroup, ...extra };
  return [
    run(`${id}-1`, { ...base, replayIndex: 1, durationSeconds: clean[0] }),
    run(`${id}-2`, { ...base, replayIndex: 2, durationSeconds: clean[1] }),
    run(`${id}-3`, {
      ...base,
      replayIndex: 3,
      replayCondition: "cpu",
      injectionMethod: "cpu-spin",
      durationSeconds: cpu,
      excessSteps: 1,
    }),
    run(`${id}-4`, { ...base, replayIndex: 4, durationSeconds: clean[2] }),
    run(`${id}-5`, {
      ...base,
      replayIndex: 5,
      replayCondition: "mixed",
      injectionMethod: "cpu-spin+model",
      durationSeconds: mixed,
      excessSteps: 2,
    }),
  ];
}

const ROLES: Record<number, string> = {
  1: "baseline",
  2: "baseline",
  3: "contended",
  4: "clean_control",
  5: "contended",
};
const stepsFor = (id: string, controlBasis = "clean_within_noise") =>
  [1, 2, 3, 4, 5].map((index): ReplayStep => ({
    experiment: id,
    replayIndex: index,
    replayRole: ROLES[index],
    labelBasis:
      ROLES[index] === "baseline"
        ? "baseline_reference"
        : ROLES[index] === "clean_control"
          ? controlBasis
          : "contention_exceeded_threshold",
  }));

const runs = [
  ...experiment("replay-task-a-r1-1", "task-a", [2, 2, 2], 3, 20),
  ...experiment("replay-task-a-r1-2", "task-a", [4, 4, 4], 4, 24),
  ...experiment("replay-task-b-r1-3", "task-b", [1, 1, 1], 2, 9),
  // A coding run without an experiment is not a replay and must be ignored.
  run("coding-1", { kind: "coding", experiment: null, replayCondition: null }),
];

const input = (
  overrides: Partial<ExperimentsPageInput> = {},
): ExperimentsPageInput => ({
  release: "steps-v9.9+abc123",
  runs,
  validation: {
    status: "passed",
    validatedAt: "2026-01-01T00:00:00Z",
    gateVersion: "0.1.0",
  },
  replaySteps: [
    ...stepsFor("replay-task-a-r1-1"),
    ...stepsFor("replay-task-a-r1-2", "clean_exceeded_noise"),
    ...stepsFor("replay-task-b-r1-3"),
  ],
  build: {
    buildId: "build-1",
    experiments: 4,
    rejections: [
      {
        origin: "experiment replay-task-a-r1-9",
        reason:
          "protocol_checks_failed: failed or missing checks stable_baseline",
        stage: "replay_label",
      },
      {
        origin: "experiment replay-task-a-r1-1 replay 01 step s1",
        reason: "baseline_reference_missing: fewer than 2 usable copies",
        stage: "replay_label",
      },
    ],
  },
  ...overrides,
});

const check = (page: ReturnType<typeof buildExperimentsPage>, id: string) => {
  const found = page?.checks.find((c) => c.id === id);
  assert.ok(found, `check ${id} present`);
  return found;
};

test("scope counts only replay experiments", () => {
  const page = buildExperimentsPage(input())!;
  assert.equal(page.experiments, 3);
  assert.equal(page.replayRuns, 15);
  assert.equal(page.replaySteps, 30);
  assert.equal(page.releaseName, "steps-v9.9");
  assert.deepEqual(page.models, ["model-x"]);
});

test("condition ratios use each experiment's own clean median", () => {
  const page = buildExperimentsPage(input())!;
  const mixed = page.conditions.find((c) => c.id === "mixed")!;
  // Ratios 10, 6 and 9: median 9, range 6–10.
  assert.equal(mixed.medianRatio, 9);
  assert.equal(mixed.minRatio, 6);
  assert.equal(mixed.maxRatio, 10);
  assert.equal(mixed.replays, 3);
  assert.equal(mixed.excessSteps, 6);
  assert.deepEqual(mixed.injectionMethods, ["cpu-spin+model"]);
  assert.equal(page.largest?.id, "mixed");
  // Conditions never replayed are left out, not shown as 1× or 0×.
  assert.deepEqual(
    page.conditions.map((c) => c.id),
    ["cpu", "mixed"],
  );
});

test("experiment rows keep missing conditions as null", () => {
  const page = buildExperimentsPage(input())!;
  const taskA = page.groups.find((g) => g.taskGroup === "task-a")!;
  assert.equal(page.groups[0].taskGroup, "task-a");
  const row = taskA.experiments.find((r) => r.id === "replay-task-a-r1-1")!;
  assert.equal(row.shortId, "r1-1");
  assert.equal(row.ratios.cpu, 1.5);
  assert.equal(row.ratios.mixed, 10);
  assert.equal(row.ratios.inference, null);
  assert.equal(row.ratios.retrieval, null);
  assert.equal(row.cleanMedianSeconds, 2);
  assert.equal(row.cleanCv, 0);
});

test("replay sequence follows the modal order with roles from steps", () => {
  const page = buildExperimentsPage(input())!;
  const sequence = page.sequence!;
  assert.equal(sequence.matching, 3);
  assert.deepEqual(
    sequence.slots.map((s) => [s.index, s.condition, s.role]),
    [
      [1, "clean", "baseline"],
      [2, "clean", "baseline"],
      [3, "cpu", "contended"],
      [4, "clean", "clean_control"],
      [5, "mixed", "contended"],
    ],
  );
  assert.equal(sequence.slots[4].medianRatio, 9);
  assert.deepEqual(page.roleCounts, [
    { role: "baseline", replays: 2 },
    { role: "clean_control", replays: 1 },
    { role: "contended", replays: 2 },
  ]);
});

test("experiments with a different order are counted out, not merged", () => {
  const reordered = runs.map((r) =>
    r.experiment === "replay-task-b-r1-3" && r.replayIndex === 3
      ? { ...r, replayIndex: 6 }
      : r,
  );
  const page = buildExperimentsPage(input({ runs: reordered }))!;
  assert.equal(page.sequence?.matching, 2);
});

test("recorded checks report failures and missing evidence distinctly", () => {
  const page = buildExperimentsPage(input())!;
  const protocol = check(page, "protocol");
  assert.equal(protocol.state, "warn");
  assert.match(protocol.label, /stable baseline/);
  assert.match(protocol.detail, /3 of 4 recorded experiments passed/);
  assert.equal(check(page, "step-labels").state, "warn");
  const noise = check(page, "clean-noise");
  assert.equal(noise.state, "warn");
  assert.match(noise.detail, /1 of 3 clean-control steps exceeded/);
  assert.equal(check(page, "quality-gate").state, "pass");
  assert.equal(check(page, "structure").state, "warn");
  // Nothing in the release records these, so they are never shown as passed.
  assert.equal(check(page, "telemetry").state, "unknown");
  assert.equal(check(page, "injection").state, "unknown");
});

test("missing build report and step columns read as not checked", () => {
  const page = buildExperimentsPage(input({ build: null, replaySteps: null }))!;
  assert.equal(check(page, "protocol").state, "unknown");
  assert.equal(check(page, "clean-noise").state, "unknown");
  assert.equal(
    page.checks.some((c) => c.id === "step-labels"),
    false,
  );
  assert.equal(page.labelOutcomes, null);
  assert.equal(page.sequence?.slots[0].role, null);
});

test("failed and unrecorded task outcomes are not counted as passed", () => {
  const withOutcomes = runs.map((r) =>
    r.runId === "replay-task-a-r1-1-5"
      ? { ...r, passed: false }
      : r.runId === "replay-task-b-r1-3-2"
        ? { ...r, passed: null }
        : r,
  );
  const page = buildExperimentsPage(input({ runs: withOutcomes }))!;
  const outcome = check(page, "task-outcome");
  assert.equal(outcome.state, "warn");
  assert.match(outcome.detail, /13 of 15 replay runs passed/);
  assert.match(outcome.detail, /1 have no recorded outcome/);
  const rowA = page.groups[0].experiments.find(
    (r) => r.id === "replay-task-a-r1-1",
  )!;
  assert.equal(rowA.failedReplays, 1);
  const rowB = page.groups[1].experiments[0];
  assert.equal(rowB.unrecordedReplays, 1);
});

test("a failing or unknown quality gate is never a pass", () => {
  const failed = buildExperimentsPage(
    input({
      validation: { status: "failed", validatedAt: "x", gateVersion: "1" },
    }),
  )!;
  assert.equal(check(failed, "quality-gate").state, "fail");
  const odd = buildExperimentsPage(
    input({
      validation: { status: "pending", validatedAt: "x", gateVersion: "1" },
    }),
  )!;
  assert.equal(check(odd, "quality-gate").state, "unknown");
});

test("a release without replay experiments has no page", () => {
  assert.equal(
    buildExperimentsPage(input({ runs: [runs[runs.length - 1]] })),
    null,
  );
});

test("reading names the two largest slowdowns and skips non-slowdowns", () => {
  const page = buildExperimentsPage(input())!;
  assert.match(
    page.reading!,
    /^Mixed and CPU contention show the largest observed slowdowns: 9\.0× and 1\.5×/,
  );
  assert.equal(
    conditionReading(page.conditions.map((c) => ({ ...c, medianRatio: 0.9 }))),
    null,
  );
  const [cpu, mixed] = page.conditions;
  assert.match(
    conditionReading([
      { ...mixed, medianRatio: 9 },
      { ...cpu, shortLabel: "Inference", medianRatio: 5 },
    ])!,
    /^Mixed and inference contention/,
  );
});

test("helpers", () => {
  assert.equal(coefficientOfVariation([1]), null);
  assert.equal(coefficientOfVariation([0, 0]), null);
  assert.ok(Math.abs(coefficientOfVariation([1, 3])! - Math.SQRT2 / 2) < 1e-12);
  assert.equal(shortExperimentId("other-id", "task-a"), "other-id");
  assert.deepEqual(ratioAxis(8.6), { max: 10, ticks: [0, 2, 4, 6, 8, 10] });
  assert.deepEqual(ratioAxis(Number.NaN), {
    max: 1,
    ticks: [0, 0.5, 1],
  });
});

test("check tally counts only passes", () => {
  const check = (id: string, state: "pass" | "warn" | "fail" | "unknown") => ({
    id,
    label: id,
    state,
    detail: "",
  });
  assert.deepEqual(checkTally([]), { passed: 0, total: 0 });
  assert.deepEqual(
    checkTally([
      check("a", "pass"),
      check("b", "warn"),
      check("c", "fail"),
      check("d", "unknown"),
      check("e", "pass"),
    ]),
    { passed: 2, total: 5 },
  );
});
