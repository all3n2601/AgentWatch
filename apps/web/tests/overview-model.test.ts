import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import type { DatasetRun } from "../lib/dataset-runs";
import { loadImportMap } from "../lib/import-map";
import { buildOverviewModel, type Ready } from "../lib/overview-model";

// Synthetic runs for tests only; they are not recorded-run evidence.
const run = (
  runId: string,
  condition: string,
  seconds: number,
  overrides: Partial<DatasetRun> = {},
): DatasetRun =>
  ({
    runId,
    kind: "agent_replay",
    taskGroup: "task-a",
    experiment: "exp-1",
    replayCondition: condition,
    replayIndex: 1,
    durationSeconds: seconds,
    queueSeconds: 0.1,
    excessSteps: 0,
    steps: 7,
    ...overrides,
  }) as DatasetRun;

const snapshot = (runs: DatasetRun[]): Ready =>
  ({
    status: "ready",
    runs,
    manifest: {
      dataset_version: "steps-test",
      rows: 100,
      label_resource: { none: 80, inference: 20 },
      telemetry_status: { present: 13, no_overlap: 30, empty: 57 },
    },
    split: { splits: { train: { rows: 60 }, test: { rows: 40 } } },
    validation: { status: "passed", anomalies: [] },
  }) as unknown as Ready;

const A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
const LIVE = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";

const runs = [
  run("c1", "clean", 1, { replayIndex: 1 }),
  run("c2", "clean", 2, { replayIndex: 2 }),
  run("c3", "clean", 3, { replayIndex: 3 }),
  run(A, "inference", 10, { replayIndex: 6, excessSteps: 3 }),
  run(B, "mixed", 20, { replayIndex: 7, excessSteps: 5 }),
  run("other", "clean", 50, { experiment: "exp-2", replayIndex: 1 }),
  {
    ...run("coding", "clean", 4),
    replayCondition: null,
    experiment: null,
  } as DatasetRun,
];

test("ratios compare each run with the median clean time of its own experiment", () => {
  const model = buildOverviewModel(snapshot(runs), { [B]: LIVE });
  assert.equal(model.slowest?.runId, B);
  assert.equal(model.slowest?.ratio, 10); // 20 s over the clean median of 2 s
  assert.equal(model.slowest?.liveId, LIVE);
  assert.deepEqual(
    model.worst.map((w) => [w.runId, w.ratio, w.liveId]),
    [
      [B, 10, LIVE],
      [A, 5, null],
    ],
  );
});

test("dots are replay runs only, with their order and an unlinked live id when unmapped", () => {
  const model = buildOverviewModel(snapshot(runs));
  assert.equal(model.replayRuns, 6);
  assert.equal(model.dots.length, 6);
  const mixed = model.dots.find((dot) => dot.id === B)!;
  assert.deepEqual(
    [mixed.x, mixed.y, mixed.condition, mixed.liveId],
    [7, 20, "mixed", null],
  );
  assert.equal(
    model.dots.some((dot) => dot.id === "coding"),
    false,
  );
});

test("headline counts come from the manifest and nothing is invented for missing parts", () => {
  const model = buildOverviewModel(snapshot(runs));
  assert.equal(model.steps, 100);
  assert.equal(model.excessSteps, 20);
  assert.equal(model.telemetryPresent, 13);
  assert.equal(model.telemetryMissed, 30);
  assert.deepEqual(model.health, {
    gate: "passed",
    anomalies: 0,
    trainRows: 60,
    testRows: 40,
    validationRows: 0,
  });
  assert.equal(model.labelMix.find((m) => m.id === "inference")?.value, 20);
  assert.equal(model.labelMix.find((m) => m.id === "retrieval")?.value, 0);
  assert.equal(buildOverviewModel(snapshot([])).slowest, null);
});

test("per-task ratios use the experiments of that task only", () => {
  const model = buildOverviewModel(snapshot(runs));
  assert.deepEqual(model.tasks, [
    { task: "task-a", ratios: { inference: 5, mixed: 10 } },
  ]);
});

test("the import map is optional and untrusted", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "agentwatch-map-"));
  try {
    assert.deepEqual(await loadImportMap(path.join(dir, "absent.json")), {});
    const file = path.join(dir, "map.json");
    await writeFile(file, JSON.stringify({ [A]: LIVE }));
    assert.deepEqual(await loadImportMap(file), { [A]: LIVE });
    await writeFile(file, JSON.stringify({ [A]: "not-a-uuid" }));
    assert.deepEqual(await loadImportMap(file), {});
    await writeFile(file, "{broken");
    assert.deepEqual(await loadImportMap(file), {});
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
