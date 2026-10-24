import assert from "node:assert/strict";
import test from "node:test";
import { runSnapshotSchema } from "../lib/live-api-schema";
import {
  resourceSample,
  runSnapshot,
  savedRecord,
  stepEvent,
  stepView as fixtureStepView,
} from "./live-fixtures";
import {
  liveFeedRun,
  liveSampleSeries,
  parseGpuSamples,
  summarizeSeries,
  wallSteps,
} from "../lib/live-feed-data";

const at = (second: number) =>
  `2026-10-20T23:30:${String(second).padStart(2, "0")}Z`;
const ms = (second: number) => Date.parse(at(second));

const step = (
  step_id: string,
  sequence: number,
  state: string,
  second: number,
  step_type = "model_call",
) => ({ step_id, step_type, sequence, state, observed_at: at(second) });

test("wall steps span running to terminal observations regardless of arrival order", () => {
  const steps = wallSteps([
    step("model-call-1", 2, "completed", 7),
    step("model-call-1", 0, "queued", 1),
    step("model-call-1", 1, "running", 3),
    step("tool-1", 0, "queued", 8, "tool"),
    step("tool-1", 1, "failed", 9, "tool"),
  ]);
  assert.deepEqual(steps, [
    {
      stepId: "model-call-1",
      stepType: "model_call",
      state: "completed",
      startMs: ms(3),
      endMs: ms(7),
    },
    {
      stepId: "tool-1",
      stepType: "tool",
      state: "failed",
      startMs: ms(8),
      endMs: ms(9),
    },
  ]);
});

test("unfinished steps keep a null end and unreadable events are skipped", () => {
  const steps = wallSteps([
    step("agent-run", 0, "queued", 1, "other"),
    step("agent-run", 1, "running", 2, "other"),
    { step_id: "broken", sequence: 0, state: "bogus" },
  ]);
  assert.equal(steps.length, 1);
  assert.equal(steps[0].endMs, null);
  assert.equal(steps[0].state, "running");
});

const sample = (
  metric: string,
  second: number,
  value: number | null,
  availability = "measured",
  resource_id = "tool-host-a",
) => ({
  observed_at: at(second),
  resource_id,
  service_id: `${resource_id}-service`,
  resource_class: "cpu_sandbox",
  metric,
  unit: metric.includes("psi") ? "percent" : "microseconds",
  value,
  availability,
});

test("unavailable CPU readings stay gaps instead of becoming zero", () => {
  const [psi] = liveSampleSeries([
    sample("cpu.psi.some.avg10.v1", 1, 0.5),
    sample("cpu.psi.some.avg10.v1", 2, null, "unavailable"),
  ]);
  assert.equal(psi.label, "CPU pressure (PSI some, avg10)");
  assert.deepEqual(
    psi.points.map((p) => p.value),
    [0.5, null],
  );
  assert.equal(psi.unavailable, 1);
});

test("live NVML GPU samples are labeled, converted to MiB and keep gaps", () => {
  const gpu = (
    metric: string,
    second: number,
    value: number | null,
    unit: string,
  ) => ({
    ...sample(
      metric,
      second,
      value,
      value === null ? "unsupported" : "measured",
    ),
    resource_class: "inference",
    unit,
  });
  const series = liveSampleSeries([
    gpu("gpu.nvml.memory.used_bytes.v1", 1, 1_702_593_821, "bytes"),
    gpu("gpu.nvml.memory.used_bytes.v1", 2, null, "bytes"),
    gpu("gpu.nvml.utilization.gpu.v1", 1, 0, "percent"),
  ]);
  const memory = series.find(
    (s) => s.metric === "gpu.nvml.memory.used_bytes.v1",
  );
  const utilization = series.find(
    (s) => s.metric === "gpu.nvml.utilization.gpu.v1",
  );
  assert.equal(memory?.label, "GPU memory used (NVML)");
  assert.equal(memory?.unit, "MiB");
  assert.equal(memory?.source, "inference live samples");
  assert.deepEqual(
    memory?.points.map((p) => p.value),
    [1_702_593_821 / (1024 * 1024), null],
  );
  assert.equal(memory?.unavailable, 1);
  // A measured idle GPU is a zero reading, not a gap.
  assert.deepEqual(
    utilization?.points.map((p) => p.value),
    [0],
  );
});

test("cumulative throttling becomes per-sample milliseconds and skips gaps", () => {
  const [throttled] = liveSampleSeries([
    sample("cpu.cgroup.throttled_usec.v1", 3, 5_000),
    sample("cpu.cgroup.throttled_usec.v1", 1, 1_000),
    sample("cpu.cgroup.throttled_usec.v1", 2, null, "unavailable"),
  ]);
  assert.equal(throttled.unit, "ms");
  // The first reading has no predecessor; the gap neither resets nor zeroes the counter.
  assert.deepEqual(
    throttled.points.map((p) => p.value),
    [null, null, 4],
  );
});

test("two resources with interleaved samples stay separate, labeled series", () => {
  const THROTTLED = "cpu.cgroup.throttled_usec.v1";
  const PSI = "cpu.psi.some.avg10.v1";
  const series = liveSampleSeries([
    sample(THROTTLED, 1, 1_000, "measured", "host-a"),
    sample(THROTTLED, 2, 50_000, "measured", "host-b"),
    sample(PSI, 2, 0.5, "measured", "host-a"),
    sample(THROTTLED, 3, 3_000, "measured", "host-a"),
    sample(PSI, 3, null, "unsupported", "host-b"),
    sample(THROTTLED, 4, 51_000, "measured", "host-b"),
  ]);
  const find = (metric: string, resource: string) =>
    series.find(
      (s) => s.metric === metric && s.resource?.resourceId === resource,
    );
  assert.equal(series.length, 4);
  assert.equal(new Set(series.map((s) => s.key)).size, 4);

  // Each counter is differenced only against its own resource's previous reading.
  assert.deepEqual(
    find(THROTTLED, "host-a")?.points.map((p) => p.value),
    [null, 2],
  );
  assert.deepEqual(
    find(THROTTLED, "host-b")?.points.map((p) => p.value),
    [null, 1],
  );
  assert.deepEqual(find(THROTTLED, "host-b")?.resource, {
    resourceId: "host-b",
    serviceId: "host-b-service",
  });

  // Availability is counted per resource; one host's unsupported reading is not another's.
  assert.equal(find(PSI, "host-a")?.unavailable, 0);
  assert.deepEqual(find(PSI, "host-a")?.notMeasured, {});
  assert.deepEqual(find(PSI, "host-b")?.notMeasured, {
    unsupported: { count: 1, reason: null },
  });
  assert.equal(find(PSI, "host-b")?.label, "CPU pressure (PSI some, avg10)");
});

test("not-measured readings are counted by state with the latest reason", () => {
  const [psi] = liveSampleSeries([
    { ...sample("cpu.psi.some.avg10.v1", 1, null, "missing"), reason: "late" },
    { ...sample("cpu.psi.some.avg10.v1", 2, null, "missing"), reason: null },
    {
      ...sample("cpu.psi.some.avg10.v1", 3, null, "unsupported"),
      reason: "no PSI on macOS",
    },
    sample("cpu.psi.some.avg10.v1", 4, 2),
  ]);
  assert.deepEqual(psi.notMeasured, {
    missing: { count: 2, reason: "late" },
    unsupported: { count: 1, reason: "no PSI on macOS" },
  });
  assert.equal(psi.unavailable, 3);
  assert.deepEqual(
    psi.points.map((p) => p.value),
    [null, null, null, 2],
  );
});

const header =
  "timestamp, index, name, pstate, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W]";
const row = (time: string, util: string, power: string) =>
  `2026/10/20 ${time}, 0, GPU, P0, ${util}, 1 %, 2520 MiB, 8151 MiB, ${power}`;
const meta =
  "nvidia_smi_timestamp_utc_offset=-0400 (local)\nnvidia_smi_timestamp_utc_offset_end=-0400\n";

test("GPU samples are placed in UTC using the probe's recorded offset", () => {
  const gpu = parseGpuSamples(
    [
      header,
      row("19:30:33.500", "57 %", "38.75 W"),
      row("19:30:33.770", "[N/A]", "40 W"),
    ].join("\n"),
    meta,
    "gpu-agent-r1",
  );
  assert.ok("series" in gpu);
  const util = gpu.series.find((s) => s.key === "gpu.util")!;
  assert.equal(util.points[0].atMs, Date.parse("2026-10-20T23:30:33.500Z"));
  assert.deepEqual(
    util.points.map((p) => p.value),
    [57, null],
  );
  assert.equal(util.unavailable, 1);
  assert.equal(
    gpu.series.find((s) => s.key === "gpu.power")!.points[1].value,
    40,
  );
});

test("GPU samples are refused when the offset is missing or changed", () => {
  const csv = [header, row("19:30:33.500", "57 %", "38 W")].join("\n");
  assert.ok("missing" in parseGpuSamples(csv, "probe_started_utc=x\n", "p"));
  assert.ok(
    "missing" in
      parseGpuSamples(
        csv,
        "nvidia_smi_timestamp_utc_offset=-0400\nnvidia_smi_timestamp_utc_offset_end=-0500\n",
        "p",
      ),
  );
});

test("series summaries ignore gaps and report nothing when unmeasured", () => {
  const series = {
    key: "k",
    metric: "m",
    label: "l",
    resource: null,
    unit: "%",
    source: "s",
    unavailable: 1,
    notMeasured: null,
    points: [
      { atMs: 0, value: 2 },
      { atMs: 1, value: null },
      { atMs: 2, value: 4 },
    ],
  };
  assert.deepEqual(summarizeSeries(series), {
    min: 2,
    max: 4,
    mean: 3,
    measured: 2,
  });
  assert.equal(
    summarizeSeries({ ...series, points: [{ atMs: 0, value: null }] }),
    null,
  );
});

test("the live view draws applied records only and counts every record received", () => {
  const queued = stepEvent("tool-1", 0, "queued", { second: 1 });
  const early = stepEvent("tool-1", 2, "completed", {
    started: 2,
    finished: 3,
    second: 3,
  });
  const snapshot = runSnapshotSchema.parse(
    runSnapshot({
      steps: [fixtureStepView(queued)],
      records: [
        savedRecord(queued, 1),
        savedRecord(early, 2, "pending"),
        savedRecord(resourceSample("s-1", { second: 2 }), 3),
      ],
    }),
  );
  const run = liveFeedRun(snapshot, snapshot.records, null);
  assert.equal(run.recordCount, 3);
  assert.deepEqual(
    run.wallSteps.map((s) => [s.stepId, s.state, s.endMs]),
    [["tool-1", "queued", null]],
    "the pending completion is not drawn",
  );
  assert.equal(run.liveSeries.length, 1);
  assert.equal(run.replay.view.gaps.length, 0);
  assert.equal(run.replay.receivedAgoMs, null);
  assert.equal(run.gpu, null);
});
