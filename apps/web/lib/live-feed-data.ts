import { z } from "zod";
import type { LiveReplay } from "./live-replay";
import { stepStates } from "./step-events";

// Pure parsing for the live-runs prototype: B01 read-API responses and the A01 nvidia-smi
// diagnostic files become chart-ready series. I/O lives in live-feed.ts.

export const runStateSchema = z.enum(stepStates);
export type LiveRecord = {
  sequence: number;
  disposition: string;
  received_at: string;
  payload: { record_type: string };
};
const sampleSchema = z.object({
  observed_at: z.iso.datetime({ offset: true }),
  resource_id: z.string(),
  service_id: z.string(),
  resource_class: z.string(),
  metric: z.string(),
  unit: z.string(),
  value: z.number().finite().nullable(),
  availability: z.string(),
  reason: z.string().nullable().optional(),
});
const stepWallSchema = z.object({
  step_id: z.string(),
  step_type: z.string(),
  sequence: z.number().int().nonnegative(),
  state: runStateSchema,
  observed_at: z.iso.datetime({ offset: true }),
});

export type LiveRunSummary = {
  runId: string;
  taskId: string;
  state: z.infer<typeof runStateSchema>;
  createdAt: string;
};

/**
 * A step's producer wall-clock window (observed_at of its running and terminal events).
 * Approximate: used only to place steps beside wall-clock resource samples on a chart.
 */
export type WallStep = {
  stepId: string;
  stepType: string;
  state: string;
  startMs: number;
  endMs: number | null;
};
export type SeriesPoint = { atMs: number; value: number | null };
/** Readings that were not measured, by reported availability, with the latest reason. */
export type NotMeasured = Record<
  string,
  { count: number; reason: string | null }
>;
export type MetricSeries = {
  /** Unique per series: live samples are keyed by resource, service and metric. */
  key: string;
  metric: string;
  label: string;
  /** The registered resource and service the readings belong to; null for diagnostics. */
  resource: { resourceId: string; serviceId: string } | null;
  unit: string;
  source: string;
  points: SeriesPoint[];
  unavailable: number;
  /** Null when the source does not report availability (the nvidia-smi diagnostic). */
  notMeasured: NotMeasured | null;
};
export type GpuEvidence =
  { series: MetricSeries[]; evidenceDir: string } | { missing: string };
export type LiveFeedRun = {
  replay: LiveReplay;
  resources: { resourceId: string; resourceClass: string }[];
  wallSteps: WallStep[];
  liveSeries: MetricSeries[];
  gpu: GpuEvidence;
  recordCount: number;
};
export type LiveFeedData = {
  runs: LiveRunSummary[];
  selected: LiveFeedRun | null;
  requested: string | null;
  error: string | null;
};

const terminal = new Set(["completed", "failed", "cancelled"]);

export function wallSteps(payloads: readonly unknown[]): WallStep[] {
  const events = payloads
    .map((raw) => stepWallSchema.safeParse(raw))
    .flatMap((parsed) => (parsed.success ? [parsed.data] : []))
    .sort((a, b) => a.sequence - b.sequence);
  const byStep = new Map<string, WallStep>();
  for (const { step_id, step_type, state, observed_at } of events) {
    const atMs = Date.parse(observed_at);
    const step = byStep.get(step_id) ?? {
      stepId: step_id,
      stepType: step_type,
      state,
      startMs: atMs,
      endMs: null,
    };
    if (state === "running") step.startMs = atMs;
    if (terminal.has(state)) step.endMs = atMs;
    step.state = state;
    byStep.set(step_id, step);
  }
  return [...byStep.values()].sort((a, b) => a.startMs - b.startMs);
}

const MICROSECONDS_PER_MILLISECOND = 1000;
const BYTES_PER_MEBIBYTE = 1024 * 1024;
/**
 * Labels for the registered whole-device NVML gauges (A04). Exact versioned names: a future
 * version may change meaning, so it keeps its raw name until it is described here.
 */
const GPU_METRIC_DISPLAY: Record<
  string,
  { label: string; unit: string; divisor?: number }
> = {
  "gpu.nvml.utilization.gpu.v1": { label: "GPU utilization (NVML)", unit: "%" },
  "gpu.nvml.utilization.memory.v1": {
    label: "GPU memory busy (NVML)",
    unit: "%",
  },
  "gpu.nvml.memory.used_bytes.v1": {
    label: "GPU memory used (NVML)",
    unit: "MiB",
    divisor: BYTES_PER_MEBIBYTE,
  },
  "gpu.nvml.power.usage_watts.v1": {
    label: "GPU power draw (NVML)",
    unit: "W",
  },
  "gpu.nvml.clock.sm_mhz.v1": { label: "GPU SM clock (NVML)", unit: "MHz" },
};

/**
 * Live samples per resource, service and metric, so readings from different resources are
 * never merged (or differenced) into one series. Unavailable readings stay gaps, never zero.
 */
export function liveSampleSeries(payloads: readonly unknown[]): MetricSeries[] {
  const series = new Map<string, MetricSeries>();
  for (const raw of payloads) {
    const sample = sampleSchema.safeParse(raw);
    if (!sample.success) continue;
    const {
      metric,
      unit,
      value,
      availability,
      reason,
      resource_id,
      service_id,
      resource_class,
      observed_at,
    } = sample.data;
    const key = JSON.stringify([resource_id, service_id, metric]);
    const entry: MetricSeries = series.get(key) ?? {
      key,
      metric,
      label: metric,
      resource: { resourceId: resource_id, serviceId: service_id },
      unit,
      source: `${resource_class} live samples`,
      points: [],
      unavailable: 0,
      notMeasured: {},
    };
    const measured = availability === "measured" && value !== null;
    if (!measured) {
      entry.unavailable += 1;
      const state =
        availability === "measured" ? "measured without a value" : availability;
      const notMeasured = entry.notMeasured ?? {};
      const counted = notMeasured[state] ?? { count: 0, reason: null };
      notMeasured[state] = {
        count: counted.count + 1,
        reason: reason ?? counted.reason,
      };
      entry.notMeasured = notMeasured;
    }
    entry.points.push({
      atMs: Date.parse(observed_at),
      value: measured ? value : null,
    });
    series.set(key, entry);
  }
  for (const entry of series.values()) {
    entry.points.sort((a, b) => a.atMs - b.atMs);
    if (entry.metric.startsWith("cpu.cgroup.throttled_usec")) {
      // A cumulative counter: plot the increase since the previous measured reading.
      entry.label = "CPU throttled per sample";
      entry.unit = "ms";
      let previous: number | null = null;
      entry.points = entry.points.map(({ atMs, value }) => {
        const delta =
          value === null || previous === null
            ? null
            : (value - previous) / MICROSECONDS_PER_MILLISECOND;
        if (value !== null) previous = value;
        return { atMs, value: delta };
      });
    } else if (entry.metric.startsWith("cpu.psi.some.avg10")) {
      entry.label = "CPU pressure (PSI some, avg10)";
      entry.unit = "%";
    } else if (entry.metric in GPU_METRIC_DISPLAY) {
      const { label, unit, divisor = 1 } = GPU_METRIC_DISPLAY[entry.metric];
      entry.label = label;
      entry.unit = unit;
      entry.points = entry.points.map(({ atMs, value }) => ({
        atMs,
        value: value === null ? null : value / divisor,
      }));
    }
  }
  return [...series.values()];
}

const GPU_COLUMNS = [
  {
    column: "utilization.gpu [%]",
    key: "gpu.util",
    label: "GPU utilization",
    unit: "%",
  },
  {
    column: "power.draw [W]",
    key: "gpu.power",
    label: "GPU power draw",
    unit: "W",
  },
  {
    column: "memory.used [MiB]",
    key: "gpu.mem",
    label: "GPU memory used",
    unit: "MiB",
  },
] as const;
const UTC_OFFSET =
  /nvidia_smi_timestamp_utc_offset(?:_end)?=([+-]\d{2})(\d{2})/g;

/**
 * Parses an A01 probe's gpu-samples.csv. nvidia-smi writes local time without an offset, so
 * samples are placed only when probe-meta.txt records one offset for the whole run.
 */
export function parseGpuSamples(
  csv: string,
  meta: string,
  evidenceDir: string,
): GpuEvidence {
  const offsets = [...meta.matchAll(UTC_OFFSET)].map((m) => `${m[1]}:${m[2]}`);
  if (!offsets.length || offsets.some((o) => o !== offsets[0]))
    return {
      missing:
        "The GPU sampler's UTC offset is unknown or changed during the run.",
    };
  const [header = "", ...rows] = csv.trim().split(/\r?\n/);
  const columns = header.split(",").map((c) => c.trim());
  const timeIndex = columns.indexOf("timestamp");
  if (timeIndex < 0)
    return { missing: "gpu-samples.csv has no timestamp column." };
  const series: MetricSeries[] = GPU_COLUMNS.map(({ key, label, unit }) => ({
    key,
    metric: key,
    label,
    resource: null,
    unit,
    source: "nvidia-smi diagnostic (not ingested)",
    points: [],
    unavailable: 0,
    notMeasured: null,
  }));
  for (const row of rows) {
    const cells = row.split(",").map((c) => c.trim());
    const stamp = cells[timeIndex]?.replaceAll("/", "-").replace(" ", "T");
    const atMs = Date.parse(`${stamp}${offsets[0]}`);
    if (!Number.isFinite(atMs)) continue;
    GPU_COLUMNS.forEach(({ column }, i) => {
      const index = columns.indexOf(column);
      const value =
        index < 0 ? Number.NaN : Number.parseFloat(cells[index] ?? "");
      const measured = Number.isFinite(value);
      if (!measured) series[i].unavailable += 1;
      series[i].points.push({ atMs, value: measured ? value : null });
    });
  }
  return { series, evidenceDir };
}

/** Wall-clock receipt age of the newest record, or null when nothing was received. */
export function latestReceiptAgeMs(
  records: readonly LiveRecord[],
  nowMs: number,
) {
  if (!records.length) return null;
  const latest = Math.max(...records.map((r) => Date.parse(r.received_at)));
  return Math.max(0, nowMs - latest);
}

export type SeriesSummary = {
  min: number;
  max: number;
  mean: number;
  measured: number;
};

/** Text summary of a series' measured values, for readers who cannot use the chart. */
export function summarizeSeries(series: MetricSeries): SeriesSummary | null {
  const values = series.points.flatMap((p) =>
    p.value === null ? [] : [p.value],
  );
  if (!values.length) return null;
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((sum, v) => sum + v, 0) / values.length,
    measured: values.length,
  };
}
