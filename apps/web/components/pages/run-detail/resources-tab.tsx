"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  Boxes,
  Gauge,
  Layers,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { chartAxisTick, chartTooltipStyle } from "@/components/chart-kit";
import { Button } from "@/components/ui/button";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { KeyValueList } from "@/components/widgets/evidence-lists";
import { StatStrip } from "@/components/widgets/stat";
import {
  ErrorState,
  Skeleton,
  UnavailableState,
} from "@/components/widgets/state-views";
import { evidenceUrl } from "@/lib/dashboard-data";
import type { Experiment, Step } from "@/lib/experiment-schema";
import { formatQuantity, MISSING_VALUE } from "@/lib/measurement-format";
import {
  parseSamples,
  pressureOf,
  pressureSeries,
  stepWindow,
  summarizePressure,
  unavailableSignals,
  type HostSample,
  type PressurePoint,
  type PressureSummary,
} from "@/lib/run-samples";
import { savedTimeline } from "@/lib/run-timeline";
import { cn } from "@/lib/utils";
import {
  FILL_PANEL,
  PANEL_SCROLL,
  TAB_GRID,
} from "@/components/shell/panel-layout";

type SamplesState =
  | { status: "loading" }
  | { status: "ready"; samples: HostSample[] }
  | { status: "error"; message: string };

/** Reads this run's samples.jsonl by its UUID; a new run never shows the previous one's. */
function useRecordedSamples(run: Experiment) {
  const url = evidenceUrl(run, "samples.jsonl");
  const [state, setState] = useState<{ url: string; value: SamplesState }>({
    url,
    value: { status: "loading" },
  });
  useEffect(() => {
    const abort = new AbortController();
    fetch(url, { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            `The recorded samples could not be read (HTTP ${response.status}).`,
          );
        const parsed = parseSamples(await response.text());
        setState({
          url,
          value: parsed.ok
            ? { status: "ready", samples: parsed.samples }
            : { status: "error", message: parsed.message },
        });
      })
      .catch((error: unknown) => {
        if (abort.signal.aborted) return;
        setState({
          url,
          value: {
            status: "error",
            message:
              error instanceof Error
                ? error.message
                : "The recorded samples could not be read.",
          },
        });
      });
    return () => abort.abort();
  }, [url]);
  return state.url === url ? state.value : ({ status: "loading" } as const);
}

const percent = (value: number | null) =>
  value === null ? MISSING_VALUE : formatQuantity(value, "percent");

const BANDS = [
  { phase: "clean", label: "Clean window", fill: "var(--aw-success-solid)" },
  {
    phase: "contended",
    label: "Contended window",
    fill: "var(--aw-warning-solid)",
  },
] as const;

function PressureChart({
  run,
  originNs,
  points,
  summary,
}: {
  run: Experiment;
  originNs: number;
  points: PressurePoint[];
  summary: PressureSummary | null;
}) {
  if (!summary)
    return (
      <UnavailableState
        title="No CPU pressure readings"
        reason="Linux PSI was not available on the host for this run, so no pressure is drawn. Measured queue delay is still on the Timeline tab."
        className="fit:flex-1"
      />
    );
  return (
    <>
      <div
        className="h-64 min-w-0 fit:h-auto fit:min-h-40 fit:flex-1"
        aria-hidden
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={points}
            margin={{ top: 8, right: 12, bottom: 0, left: -12 }}
          >
            <CartesianGrid vertical={false} stroke="var(--aw-line)" />
            {BANDS.map((band) => {
              const window = stepWindow(run[band.phase], originNs);
              return (
                <ReferenceArea
                  key={band.phase}
                  x1={window.from}
                  x2={window.to}
                  fill={band.fill}
                  fillOpacity={0.14}
                  ifOverflow="extendDomain"
                />
              );
            })}
            <XAxis
              dataKey="seconds"
              type="number"
              domain={["dataMin", "dataMax"]}
              unit=" s"
              tickFormatter={(value: number) => value.toFixed(1)}
              tick={chartAxisTick}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              unit="%"
              tick={chartAxisTick}
              axisLine={false}
              tickLine={false}
              allowDecimals
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              labelFormatter={(value) => `${Number(value).toFixed(2)} s`}
              formatter={(value) => [
                value === null || value === undefined
                  ? "Not measured"
                  : percent(Number(value)),
                "PSI some avg10",
              ]}
            />
            <Area
              dataKey="pressure"
              type="monotone"
              stroke="var(--aw-brand-strong)"
              strokeWidth={2}
              fill="var(--aw-brand)"
              fillOpacity={0.16}
              connectNulls={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-ink fit:shrink-0">
        <span className="flex items-center gap-2">
          <i aria-hidden className="h-0.5 w-4 rounded bg-brand-strong" />
          PSI some avg10
        </span>
        {BANDS.map((band) => (
          <span key={band.phase} className="flex items-center gap-2">
            <i
              aria-hidden
              className="size-3.5 rounded-[4px] opacity-40"
              style={{ background: band.fill }}
            />
            {band.label}
          </span>
        ))}
        <Button asChild variant="glass" size="pill" className="ml-auto">
          <a href={evidenceUrl(run, "samples.jsonl")}>
            <ArrowDownToLine aria-hidden />
            samples.jsonl
          </a>
        </Button>
      </div>
      <p className="mt-2 text-xs text-ink-muted fit:shrink-0">
        {summary.peak === 0
          ? `All ${summary.readings} readings were ${percent(0)}.`
          : `${summary.readings} readings. Peak ${percent(summary.peak)} at ${summary.peakSeconds.toFixed(2)} s, mean ${percent(summary.mean)}.`}
        Host-wide signal: it shows pressure on the machine, not which step
        caused it. Gaps are readings that were not taken, never zero.
      </p>
    </>
  );
}

function EvaluationWindow({ title, step }: { title: string; step: Step }) {
  const telemetry = step.telemetry;
  return (
    <InsetCard>
      <h3 className="mb-1 text-sm font-semibold text-ink">{title}</h3>
      <p className="mb-2 text-xs text-ink-muted">{step.step_id}</p>
      {telemetry ? (
        <KeyValueList
          dense
          items={[
            {
              id: "samples",
              label: "Samples in window",
              value: formatQuantity(telemetry.sample_count, "count"),
            },
            {
              id: "valid",
              label: "With CPU pressure",
              value: formatQuantity(telemetry.valid_pressure_samples, "count"),
            },
            {
              id: "max",
              label: "Peak avg10",
              value: percent(telemetry.max_cpu_pressure_avg10_percent),
            },
            { id: "status", label: "Telemetry", value: telemetry.status },
          ]}
        />
      ) : (
        <p className="text-sm text-ink-muted">
          No telemetry summary was recorded for this step.
        </p>
      )}
    </InsetCard>
  );
}

function Coverage({
  run,
  samples,
}: {
  run: Experiment;
  samples: HostSample[] | null;
}) {
  const missing = samples ? unavailableSignals(samples) : [];
  return (
    <GlassPanel
      as="section"
      aria-labelledby="run-resource-coverage"
      className={FILL_PANEL}
    >
      <PanelHeader
        titleId="run-resource-coverage"
        icon={Layers}
        title="Pressure during each evaluation"
        description="Readings inside each step's submit-to-finish window."
        className="fit:shrink-0"
      />
      <div
        tabIndex={0}
        role="group"
        aria-label="Telemetry coverage by evaluation"
        className={cn(PANEL_SCROLL, "space-y-3")}
      >
        <EvaluationWindow title="Clean evaluation" step={run.clean} />
        <EvaluationWindow title="Contended evaluation" step={run.contended} />
        {missing.length > 0 && (
          <InsetCard>
            <h3 className="mb-1 text-sm font-semibold text-ink">
              Signals not available
            </h3>
            <ul className="space-y-1 text-sm text-ink-muted">
              {missing.map(({ signal, count }) => (
                <li key={signal} className="[overflow-wrap:anywhere]">
                  {signal}{" "}
                  <span className="tabular-nums">
                    ({count} of {samples?.length} readings)
                  </span>
                </li>
              ))}
            </ul>
          </InsetCard>
        )}
      </div>
    </GlassPanel>
  );
}

/**
 * Host sampler readings for a saved run: counts, CPU pressure across the run with the clean
 * and contended windows marked, and coverage per evaluation.
 */
export function ResourcesTab({ run }: { run: Experiment }) {
  const state = useRecordedSamples(run);
  const samples = state.status === "ready" ? state.samples : null;
  const count = (filter: (sample: HostSample) => boolean) =>
    samples ? formatQuantity(samples.filter(filter).length, "count") : "…";
  const { originNs } = useMemo(() => savedTimeline(run), [run]);
  const points = useMemo(
    () => (samples ? pressureSeries(samples, originNs) : null),
    [samples, originNs],
  );
  const summary = points ? summarizePressure(points) : null;

  return (
    <div className="space-y-4 fit:flex fit:h-full fit:min-h-0 fit:flex-col fit:gap-3 fit:space-y-0">
      <StatStrip
        label="Recorded host samples"
        className="fit:shrink-0"
        items={[
          {
            id: "samples",
            icon: Activity,
            value: count(() => true),
            label: "readings collected",
          },
          {
            id: "psi",
            icon: Gauge,
            value: count((sample) => pressureOf(sample) !== null),
            label: "with CPU pressure",
          },
          {
            id: "cgroup",
            icon: Boxes,
            value: count((sample) => sample.cgroup_cpu_stat !== null),
            label: "with cgroup CPU",
          },
          {
            id: "peak",
            icon: TrendingUp,
            value: samples ? percent(summary?.peak ?? null) : "…",
            label: "peak avg10",
          },
        ]}
      />
      <div className={cn(TAB_GRID, "fit:min-h-0 fit:flex-1")}>
        <GlassPanel
          as="section"
          aria-labelledby="run-pressure"
          className={FILL_PANEL}
        >
          <PanelHeader
            titleId="run-pressure"
            icon={Gauge}
            title="CPU pressure over the run"
            description="Linux PSI, share of time tasks waited for CPU, from the first submission."
            className="fit:shrink-0"
          />
          {state.status === "loading" && (
            <Skeleton
              label="Reading recorded samples"
              className="h-64 fit:h-auto fit:flex-1"
            />
          )}
          {state.status === "error" && (
            <ErrorState
              title="Samples unavailable"
              detail={state.message}
              className="fit:flex-1"
            />
          )}
          {points && (
            <PressureChart
              run={run}
              originNs={originNs}
              points={points}
              summary={summary}
            />
          )}
        </GlassPanel>
        <Coverage run={run} samples={samples} />
      </div>
    </div>
  );
}
