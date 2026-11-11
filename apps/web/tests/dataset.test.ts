import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { parquetWriteBuffer } from "hyparquet-writer";
import { loadDatasetSnapshot, readStepRows } from "../lib/dataset-load";
import { parseStepRows, summarizeRuns } from "../lib/dataset-runs";
import type { Manifest } from "../lib/dataset-schema";
import {
  labelSeries,
  medianStepTimeByCondition,
  missingColumns,
  segments,
  sharePercent,
} from "../lib/dataset-view";

const RELEASE = "steps-v9.9+fixture";
const OLDER_RELEASE = "steps-v9.8+older";

const manifest = (release = RELEASE) => ({
  dataset_version: release,
  schema_version: "steps-v9.9",
  release_format: 3,
  rows: 10,
  runs: 4,
  tasks: 2,
  kind: { agent_replay: 8, cpu: 2 },
  condition: { baseline: 6, clean: 2, contended: 2 },
  label_resource: { none: 7, inference: 3 },
  label_basis: { baseline_reference: 6, clean_within_noise: 2 },
  telemetry_status: { present: 1, empty: 9 },
  missing: { cpu_seconds: 4, run_id: 0, agent_model: 2 },
  label_version: "labels-fixture",
});
const splitPart = {
  rows: 5,
  runs: 2,
  groups: 1,
  label_resource: {},
  condition: {},
  injection_method: {},
};
const split = {
  split_id: "split-fixture",
  split_method: "split-v2",
  contract: {
    version: "eval-contract-v2",
    status: "accepted",
    group_key: "task_group",
  },
  class_balance: {
    status: "met",
    per_class: { none: 7 },
    clean: 2,
    mixed: 0,
    minimums: { clean: 1, mixed: 1, per_class: 1 },
  },
  splits: {
    train: splitPart,
    test: splitPart,
    validation: { ...splitPart, rows: 0 },
  },
  design_encoding: {
    status: "report_only",
    rows: 10,
    features: [{ feature: "queue_seconds", auc: 0.7 }],
  },
  leakage: {
    status: "passed",
    max_single_feature_auc: 0.98,
    features: [{ class: "retrieval", feature: "queue_seconds", auc: 0.9 }],
  },
  held_out_injection_methods: {
    status: "met",
    methods: ["a"],
    only_in_test: ["a"],
  },
};
const validation = {
  status: "passed",
  validated_at: "2026-11-07T07:33:56+00:00",
  examples: 10,
  anomalies: [],
  drift: [],
  gate_version: "0.1.0",
  tfdv_version: "1.21.0",
};

const RUN_A = "11111111-1111-4111-8111-111111111111";
const RUN_B = "22222222-2222-4222-8222-222222222222";

/** Synthetic rows for tests only; they are not recorded-run evidence. */
const stepRow = (overrides: Record<string, unknown> = {}) => ({
  run_id: RUN_A,
  step_id: "model-call-1",
  kind: "agent_replay",
  workload: "agent replay fixture",
  task_group: "fixture-task",
  experiment: "replay-fixture",
  condition: "baseline",
  replay_condition: "clean",
  replay_index: 1,
  step_type: "model_call",
  agent_model: "fixture-model",
  injection_method: "none",
  recorded_at: new Date("2026-11-05T17:48:31.500Z"),
  duration_seconds: 0.5,
  queue_seconds: 0.1,
  execution_seconds: 0.4,
  cpu_seconds: null,
  label_resource: "none",
  label_excess_fraction: 0,
  telemetry_status: "empty",
  run_passed: true,
  ...overrides,
});

const fixtureRows = () => [
  stepRow(),
  stepRow({
    step_id: "tool-2",
    step_type: "tool",
    label_resource: "inference",
    label_excess_fraction: 0.6,
    telemetry_status: "present",
    recorded_at: new Date("2026-11-05T17:48:30.000Z"),
  }),
  stepRow({
    step_id: "tool-3",
    label_resource: "inference",
    duration_seconds: 1,
    queue_seconds: 0.2,
    execution_seconds: 0.8,
  }),
  stepRow({
    run_id: RUN_B,
    step_id: "baseline-0",
    kind: "coding",
    workload: "fixed Python project",
    experiment: null,
    replay_condition: null,
    replay_index: null,
    step_type: null,
    agent_model: null,
    recorded_at: null,
    cpu_seconds: 0.25,
    run_passed: false,
  }),
];

function stepsParquet(rows: ReturnType<typeof fixtureRows>) {
  const text = (name: keyof ReturnType<typeof stepRow>) => ({
    name,
    data: rows.map((row) => row[name] as string | null),
    type: "STRING" as const,
    nullable: true,
  });
  const number = (
    name:
      | "duration_seconds"
      | "queue_seconds"
      | "execution_seconds"
      | "cpu_seconds"
      | "label_excess_fraction",
  ) => ({
    name,
    data: rows.map((row) => row[name] as number | null),
    type: "DOUBLE" as const,
    nullable: true,
  });
  return parquetWriteBuffer({
    columnData: [
      text("run_id"),
      text("step_id"),
      text("kind"),
      text("workload"),
      text("task_group"),
      text("experiment"),
      text("condition"),
      text("replay_condition"),
      {
        name: "replay_index",
        data: rows.map((row) =>
          row.replay_index === null ? null : BigInt(row.replay_index),
        ),
        type: "INT64" as const,
        nullable: true,
      },
      text("step_type"),
      text("agent_model"),
      text("injection_method"),
      {
        name: "recorded_at",
        data: rows.map((row) => row.recorded_at),
        type: "TIMESTAMP" as const,
        nullable: true,
      },
      number("duration_seconds"),
      number("queue_seconds"),
      number("execution_seconds"),
      number("cpu_seconds"),
      text("label_resource"),
      number("label_excess_fraction"),
      text("telemetry_status"),
      {
        name: "run_passed",
        data: rows.map((row) => row.run_passed),
        type: "BOOLEAN" as const,
        nullable: true,
      },
    ],
  });
}

async function writeJson(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(value));
}

async function writeRelease(root: string, release: string, buildName: string) {
  await writeJson(path.join(root, "builds", buildName), {
    dataset_version: release,
  });
  await writeJson(path.join(root, release, "manifest.json"), {
    ...manifest(release),
    rows: fixtureRows().length,
    runs: 2,
  });
  await writeFile(
    path.join(root, release, "steps.parquet"),
    new Uint8Array(stepsParquet(fixtureRows())),
  );
  await writeJson(path.join(root, "splits", release, "split-v2+x.json"), split);
  await writeJson(
    path.join(
      root,
      "validations",
      release,
      "20261107T000000Z-aaaa",
      "result.json",
    ),
    validation,
  );
}

async function withRoot(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(tmpdir(), "agentwatch-dataset-"));
  try {
    await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

test("loads the release named by the newest build report", async () => {
  await withRoot(async (root) => {
    await writeRelease(root, OLDER_RELEASE, "20261101T000000Z-aaaa.json");
    await writeRelease(root, RELEASE, "20261107T000000Z-bbbb.json");
    const snapshot = await loadDatasetSnapshot(root);
    assert.equal(snapshot.status, "ready");
    if (snapshot.status === "ready") {
      assert.equal(snapshot.manifest.dataset_version, RELEASE);
      assert.equal(snapshot.split.splits.validation.rows, 0);
      assert.equal(snapshot.validation.status, "passed");
      assert.deepEqual(
        snapshot.runs.map((run) => run.runId),
        [RUN_A, RUN_B],
      );
    }
  });
});

test("a missing dataset is unavailable with a fetch hint, not an empty dataset", async () => {
  await withRoot(async (root) => {
    const snapshot = await loadDatasetSnapshot(path.join(root, "absent"));
    assert.equal(snapshot.status, "unavailable");
    if (snapshot.status === "unavailable") {
      assert.match(snapshot.reason, /not found/);
      assert.match(snapshot.reason, /make pull-dataset/);
    }
  });
});

test("a manifest in the wrong format is rejected, not charted", async () => {
  await withRoot(async (root) => {
    await writeRelease(root, RELEASE, "20261107T000000Z-bbbb.json");
    await writeJson(path.join(root, RELEASE, "manifest.json"), {
      ...manifest(),
      rows: "many",
    });
    const snapshot = await loadDatasetSnapshot(root);
    assert.equal(snapshot.status, "unavailable");
    if (snapshot.status === "unavailable") {
      assert.match(snapshot.reason, /expected format/);
    }
  });
});

test("labelSeries keeps the class order and reports absent classes as zero rows", () => {
  const series = labelSeries({ none: 7, inference: 3 });
  assert.deepEqual(
    series.map((entry) => [entry.id, entry.value]),
    [
      ["none", 7],
      ["cpu_sandbox", 0],
      ["inference", 3],
      ["retrieval", 0],
      ["mixed", 0],
    ],
  );
});

test("segments sort largest first and name known values", () => {
  assert.deepEqual(segments({ cpu: 2, agent_replay: 8, other_kind: 1 }), [
    { id: "agent_replay", label: "Agent replay", value: 8 },
    { id: "cpu", label: "CPU task", value: 2 },
    { id: "other_kind", label: "other kind", value: 1 },
  ]);
});

test("missingColumns drops complete columns and ranks by empty rows", () => {
  const columns = missingColumns(manifest() as Manifest);
  assert.deepEqual(
    columns.map((c) => [c.column, c.missing, c.share]),
    [
      ["cpu_seconds", 4, 0.4],
      ["agent_model", 2, 0.2],
    ],
  );
});

test("sharePercent handles an empty whole", () => {
  assert.equal(sharePercent(0, 0), "0%");
  assert.equal(sharePercent(1, 3), "33%");
  assert.equal(sharePercent(1, 100), "1.0%");
});

test("summarizeRuns totals each run's steps and keeps absent times and readings absent", () => {
  const [replay, coding] = summarizeRuns(parseStepRows(fixtureRows()));
  assert.equal(replay.runId, RUN_A);
  assert.equal(replay.steps, 3);
  assert.ok(Math.abs(replay.durationSeconds - 2) < 1e-9);
  assert.ok(Math.abs(replay.queueSeconds - 0.4) < 1e-9);
  assert.equal(replay.excessSteps, 2);
  assert.equal(replay.topLabel, "inference");
  assert.deepEqual(replay.labels, { none: 1, inference: 2 });
  assert.deepEqual(replay.stepTypes, { model_call: 2, tool: 1 });
  // tool-2 and tool-3 (a model call by default) carry the inference label.
  assert.deepEqual(replay.excessByStepType, { model_call: 1, tool: 1 });
  assert.equal(replay.telemetryPresent, 1);
  assert.equal(replay.recordedAt, "2026-11-05T17:48:30.000Z");
  assert.equal(coding.runId, RUN_B);
  assert.equal(coding.recordedAt, null);
  assert.equal(coding.passed, false);
  assert.equal(coding.topLabel, "none");
  // Steps without a call type are not counted under an invented one.
  assert.deepEqual(coding.stepTypes, {});
  assert.deepEqual(coding.excessByStepType, {});
});

test("a row with a negative duration is rejected, not summed", () => {
  assert.throws(() => parseStepRows([stepRow({ duration_seconds: -1 })]));
});

test("readStepRows reads a written release file and converts int64 columns", async () => {
  await withRoot(async (root) => {
    await writeRelease(root, RELEASE, "20261107T000000Z-bbbb.json");
    const rows = await readStepRows(root, RELEASE);
    assert.equal(rows.length, 4);
    assert.equal(rows[0].replay_index, 1);
    assert.equal(rows[3].recorded_at, null);
  });
});

test("a manifest whose counts disagree with the step file is unavailable", async () => {
  await withRoot(async (root) => {
    await writeRelease(root, RELEASE, "20261107T000000Z-bbbb.json");
    await writeJson(path.join(root, RELEASE, "manifest.json"), {
      ...manifest(),
      rows: 4,
      runs: 3,
    });
    const snapshot = await loadDatasetSnapshot(root);
    assert.equal(snapshot.status, "unavailable");
    if (snapshot.status === "unavailable") {
      assert.match(snapshot.reason, /disagree/);
    }
  });
});

test("medianStepTimeByCondition compares one replay task and ignores non-replay runs", () => {
  const replay = (
    runId: string,
    condition: string,
    taskGroup: string,
    seconds: number,
  ) => ({
    ...summarizeRuns(parseStepRows([stepRow({ run_id: runId })]))[0],
    replayCondition: condition,
    taskGroup,
    durationSeconds: seconds,
  });
  const ids = [1, 2, 3, 4, 5, 6].map(
    (n) =>
      `${String(n).repeat(8)}-${String(n).repeat(4)}-4${String(n).repeat(3)}-8${String(n).repeat(3)}-${String(n).repeat(12)}`,
  );
  const runs = [
    replay(ids[0], "clean", "task-a", 1),
    replay(ids[1], "clean", "task-a", 3),
    replay(ids[2], "clean", "task-a", 2),
    replay(ids[3], "inference", "task-a", 10),
    replay(ids[4], "clean", "task-b", 99),
    ...summarizeRuns(
      parseStepRows([stepRow({ run_id: ids[5], replay_condition: null })]),
    ),
  ];
  const result = medianStepTimeByCondition(runs);
  assert.equal(result?.taskGroup, "task-a");
  assert.deepEqual(
    result?.conditions.map((c) => [c.id, c.runs, c.medianSeconds]),
    [
      ["clean", 3, 2],
      ["inference", 1, 10],
    ],
  );
  assert.equal(medianStepTimeByCondition([]), null);
});
