import assert from "node:assert/strict";
import test from "node:test";
import {
  latestReceiptAgeMs,
  liveSampleSeries,
  parseGpuSamples,
  summarizeSeries,
  wallSteps,
  type LiveRecord,
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
) => ({
  observed_at: at(second),
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

test("CPU stall and usage counters become per-sample ms, with resets as gaps", () => {
  const counter = (metric: string, second: number, value: number | null) => ({
    ...sample(metric, second, value),
    unit: "microseconds",
  });
  const series = liveSampleSeries([
    counter("cpu.psi.some.total_usec.v1", 1, 499_359_020),
    counter("cpu.psi.some.total_usec.v1", 2, 499_361_020),
    counter("cpu.psi.some.total_usec.v1", 3, 100),
    counter("cpu.cgroup.usage_usec.v1", 1, 1_000),
    counter("cpu.cgroup.usage_usec.v1", 2, 251_000),
  ]);
  const stall = series.find((s) => s.key === "cpu.psi.some.total_usec.v1");
  const usage = series.find((s) => s.key === "cpu.cgroup.usage_usec.v1");
  assert.equal(stall?.label, "CPU stall per sample (PSI some, host)");
  assert.equal(stall?.unit, "ms");
  // The first reading has no predecessor and the reset is a gap, never negative time.
  assert.deepEqual(
    stall?.points.map((p) => p.value),
    [null, 2, null],
  );
  assert.deepEqual(
    usage?.points.map((p) => p.value),
    [null, 250],
  );
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
  const memory = series.find((s) => s.key === "gpu.nvml.memory.used_bytes.v1");
  const utilization = series.find(
    (s) => s.key === "gpu.nvml.utilization.gpu.v1",
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

test("receipt age uses the newest record and is null with none", () => {
  const record = (second: number): LiveRecord => ({
    sequence: second,
    disposition: "applied",
    received_at: at(second),
    payload: { record_type: "step_event" },
  });
  assert.equal(latestReceiptAgeMs([], ms(10)), null);
  assert.equal(latestReceiptAgeMs([record(2), record(5)], ms(10)), 5_000);
  assert.equal(latestReceiptAgeMs([record(5)], ms(1)), 0);
});

test("series summaries ignore gaps and report nothing when unmeasured", () => {
  const series = {
    key: "k",
    label: "l",
    unit: "%",
    source: "s",
    unavailable: 1,
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
