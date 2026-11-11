import { z } from "zod";
import type { Step } from "./experiment-schema";
import { secondsSinceOrigin } from "./run-timeline";

// A saved run's host sampler readings (samples.jsonl). The sampler stamps each reading with
// `time.time_ns()`, the same wall clock as the worker's span timestamps, so readings can be
// placed on the run timeline.

export const sampleSchema = z.object({
  timestamp_ns: z.number().finite().nonnegative(),
  cpu_pressure: z
    .record(z.string(), z.record(z.string(), z.number().finite().nonnegative()))
    .nullable(),
  cgroup_cpu_stat: z
    .record(z.string(), z.number().finite().nonnegative())
    .nullable(),
  unavailable: z.array(z.string()),
});
export type HostSample = z.infer<typeof sampleSchema>;

export type ParsedSamples =
  { ok: true; samples: HostSample[] } | { ok: false; message: string };

/** Parses samples.jsonl; one malformed line rejects the file rather than dropping readings. */
export function parseSamples(raw: string): ParsedSamples {
  const samples: HostSample[] = [];
  const lines = raw.split("\n");
  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    let json: unknown;
    try {
      json = JSON.parse(line);
    } catch {
      return {
        ok: false,
        message: `samples.jsonl line ${index + 1} is not JSON.`,
      };
    }
    const parsed = sampleSchema.safeParse(json);
    if (!parsed.success)
      return {
        ok: false,
        message: `samples.jsonl line ${index + 1} does not match the sample format.`,
      };
    samples.push(parsed.data);
  }
  return { ok: true, samples };
}

/** PSI `some avg10` in percent, or null when the reading was not taken. */
export const pressureOf = (sample: HostSample) =>
  sample.cpu_pressure?.some?.avg10 ?? null;

export type PressurePoint = { seconds: number; pressure: number | null };

/** Pressure readings in seconds since the timeline origin; missing readings stay null. */
export function pressureSeries(
  samples: readonly HostSample[],
  originNs: number,
): PressurePoint[] {
  return samples
    .map((sample) => ({
      seconds: secondsSinceOrigin(sample.timestamp_ns, originNs),
      pressure: pressureOf(sample),
    }))
    .sort((a, b) => a.seconds - b.seconds);
}

export type PressureSummary = {
  readings: number;
  peak: number;
  peakSeconds: number;
  mean: number;
};

/** Peak and mean of the measured readings only; null when none was measured. */
export function summarizePressure(
  points: readonly PressurePoint[],
): PressureSummary | null {
  const measured = points.filter(
    (point): point is { seconds: number; pressure: number } =>
      point.pressure !== null,
  );
  if (!measured.length) return null;
  const peak = measured.reduce((best, point) =>
    point.pressure > best.pressure ? point : best,
  );
  return {
    readings: measured.length,
    peak: peak.pressure,
    peakSeconds: peak.seconds,
    mean:
      measured.reduce((sum, point) => sum + point.pressure, 0) /
      measured.length,
  };
}

/** Each signal the sampler reported as unavailable, with how many readings lacked it. */
export function unavailableSignals(samples: readonly HostSample[]) {
  const counts = new Map<string, number>();
  for (const sample of samples)
    for (const signal of new Set(sample.unavailable))
      counts.set(signal, (counts.get(signal) ?? 0) + 1);
  return [...counts].map(([signal, count]) => ({ signal, count }));
}

/** A step's submit-to-finish window in seconds since the timeline origin. */
export const stepWindow = (step: Step, originNs: number) => ({
  from: secondsSinceOrigin(step.submitted_ns, originNs),
  to: secondsSinceOrigin(step.finished_ns, originNs),
});
