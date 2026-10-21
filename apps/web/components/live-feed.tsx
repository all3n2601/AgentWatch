"use client";
import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Database, Unplug } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { PanelHeading } from "@/components/run-views";
import { ReplayView, ScenarioPicker } from "@/components/live-run";
import { duration } from "@/lib/dashboard-data";
import {
  summarizeSeries,
  type LiveFeedData,
  type LiveFeedRun,
  type MetricSeries,
  type WallStep,
} from "@/lib/live-feed-data";
import type { LivePageData } from "@/lib/live-replay";
import { cn } from "@/lib/utils";

// Prototype view of recorded live runs; see lib/live-feed.ts for its data sources.

const ACTIVE_REFRESH_INTERVAL_MS = 2_000;
const stepColors: Record<string, string> = {
  model_call: "#a78bfa",
  tool: "#34d399",
  retrieval: "#38bdf8",
  other: "#64748b",
};
const seriesColors: Record<string, string> = {
  "gpu.util": "#a78bfa",
  "gpu.power": "#fbbf24",
};
const CPU_SERIES_COLOR = "#34d399";
const tooltipStyle = {
  background: "#171d27",
  border: "1px solid #303948",
  borderRadius: 8,
  fontSize: 12,
};

function useRefreshWhileActive(active: boolean) {
  const router = useRouter();
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(
      () => router.refresh(),
      ACTIVE_REFRESH_INTERVAL_MS,
    );
    return () => clearInterval(timer);
  }, [active, router]);
}

function RunPicker({
  feed,
  current,
}: {
  feed: LiveFeedData;
  current?: string;
}) {
  if (!feed.runs.length) return null;
  return (
    <nav aria-label="Recorded live runs">
      <p className="mb-2 text-[11px] text-muted-foreground">
        Live runs from the API ({feed.runs.length}):
      </p>
      <ul className="flex flex-wrap gap-2">
        {feed.runs.map((run) => (
          <li key={run.runId}>
            <Link
              href={`/live?liveRun=${run.runId}`}
              aria-current={run.runId === current ? "page" : undefined}
              className={cn(
                "inline-flex min-h-11 flex-col justify-center rounded-md border px-3 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-primary",
                run.runId === current
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="font-mono">{run.runId.slice(0, 8)}</span>
              <span className="text-[10px]">
                {run.taskId} · {run.state} ·{" "}
                <time dateTime={run.createdAt} suppressHydrationWarning>
                  {new Date(run.createdAt).toLocaleTimeString()}
                </time>
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

type Window = { originMs: number; endMs: number };

function runWindow(steps: WallStep[]): Window | null {
  if (!steps.length) return null;
  return {
    originMs: Math.min(...steps.map((s) => s.startMs)),
    endMs: Math.max(...steps.map((s) => s.endMs ?? s.startMs)),
  };
}

const secondsInto = (atMs: number, w: Window) => (atMs - w.originMs) / 1000;
const spanSeconds = (w: Window) =>
  Math.max((w.endMs - w.originMs) / 1000, 0.001);

function Timeline({ steps, window }: { steps: WallStep[]; window: Window }) {
  const span = spanSeconds(window);
  return (
    <div>
      <ul
        className="mb-3 flex flex-wrap gap-4 text-[11px] text-muted-foreground"
        aria-label="Step types"
      >
        {Object.entries(stepColors).map(([type, color]) => (
          <li key={type} className="flex items-center gap-1.5">
            <span
              className="size-2 rounded-full"
              style={{ background: color }}
              aria-hidden
            />
            {type.replace("_", " ")}
          </li>
        ))}
      </ul>
      <ol className="space-y-1" aria-label="Step timeline">
        {steps.map((step) => {
          const start = secondsInto(step.startMs, window);
          const end =
            step.endMs === null ? span : secondsInto(step.endMs, window);
          return (
            <li
              key={step.stepId}
              className="flex h-6 items-center gap-2 text-[11px]"
            >
              <span
                className="w-24 shrink-0 truncate font-mono text-muted-foreground sm:w-40"
                title={step.stepId}
              >
                {step.stepId}
              </span>
              <span
                className="relative h-3 flex-1 rounded-sm bg-white/[0.03]"
                aria-hidden
              >
                <span
                  className={cn(
                    "absolute inset-y-0 rounded-sm",
                    step.endMs === null && "motion-safe:animate-pulse",
                  )}
                  style={{
                    left: `${(start / span) * 100}%`,
                    width: `${Math.max(((end - start) / span) * 100, 0.4)}%`,
                    background: stepColors[step.stepType] ?? stepColors.other,
                    opacity: step.state === "failed" ? 0.45 : 1,
                  }}
                />
              </span>
              <span className="w-28 shrink-0 text-right text-muted-foreground">
                {step.endMs === null ? "in progress" : duration(end - start)}
                <span className="sr-only">
                  , {step.stepType.replace("_", " ")}, {step.state}, starts at{" "}
                  {start.toFixed(2)} s
                </span>
                {step.state === "failed" && (
                  <span className="ml-1 text-rose-200">failed</span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-2 text-[11px] text-muted-foreground">
        0 s → {span.toFixed(1)} s on the producer&apos;s wall clock
        (observed_at). The step details below use the runner&apos;s monotonic
        ticks.
      </p>
    </div>
  );
}

function SeriesSummaryText({ series }: { series: MetricSeries }) {
  const summary = summarizeSeries(series);
  if (!summary)
    return (
      <p className="mt-2 text-[11px] text-muted-foreground">
        No measured values.
      </p>
    );
  const f = (v: number) => `${v.toFixed(2)} ${series.unit}`;
  return (
    <dl className="mt-2 grid grid-cols-3 gap-2 text-[11px]">
      {(
        [
          ["Min", summary.min],
          ["Mean", summary.mean],
          ["Max", summary.max],
        ] as const
      ).map(([label, value]) => (
        <div key={label}>
          <dt className="text-muted-foreground">{label}</dt>
          <dd className="metric-number">{f(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function MetricChart({
  series,
  window,
  modelCalls,
  color,
}: {
  series: MetricSeries;
  window: Window;
  modelCalls: WallStep[];
  color: string;
}) {
  const data = series.points.map((p) => ({
    t: Number(secondsInto(p.atMs, window).toFixed(3)),
    value: p.value,
  }));
  const measured = data.filter((p) => p.value !== null).length;
  return (
    <Card className="min-w-0 gap-0 p-4">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-2">
        <h3 className="text-xs font-semibold">
          {series.label}{" "}
          <span className="font-normal text-muted-foreground">
            ({series.unit})
          </span>
        </h3>
        <span className="text-[10px] text-muted-foreground">
          {series.source} · {measured} measured
          {series.unavailable > 0 && ` · ${series.unavailable} unavailable`}
        </span>
      </div>
      <div className="h-36" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 4, right: 8, bottom: 0, left: -16 }}
          >
            <CartesianGrid vertical={false} stroke="#273140" />
            {modelCalls.map((call) => (
              <ReferenceArea
                key={call.stepId}
                x1={secondsInto(call.startMs, window)}
                x2={secondsInto(call.endMs ?? window.endMs, window)}
                fill="#a78bfa"
                fillOpacity={0.12}
                ifOverflow="hidden"
              />
            ))}
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, Number(spanSeconds(window).toFixed(3))]}
              allowDataOverflow
              unit="s"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10 }}
            />
            <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(t) => `${Number(t).toFixed(2)} s`}
              formatter={(v) => [
                v === null || v === undefined
                  ? "unavailable"
                  : `${Number(v).toFixed(2)} ${series.unit}`,
                series.label,
              ]}
            />
            <Line
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={data.length < 40 ? { r: 2 } : false}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <SeriesSummaryText series={series} />
    </Card>
  );
}

function Resources({ run }: { run: LiveFeedRun }) {
  const window = runWindow(run.wallSteps);
  if (!window) return null;
  const modelCalls = run.wallSteps.filter((s) => s.stepType === "model_call");
  const gpu = "series" in run.gpu ? run.gpu : null;
  const charts = [
    ...(gpu?.series.filter((s) => s.key !== "gpu.mem") ?? []),
    ...run.cpuSeries,
  ];
  return (
    <Card className="gap-0 p-5 sm:p-6">
      <PanelHeading
        title="Step timeline and resource signals"
        detail="Shaded bands mark model-call steps. CPU signals are ingested live samples; GPU signals come from the A01 nvidia-smi diagnostic, which is not ingested and is aligned only by wall clock."
      />
      <Timeline steps={run.wallSteps} window={window} />
      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {charts.map((s) => (
          <MetricChart
            key={s.key}
            series={s}
            window={window}
            modelCalls={modelCalls}
            color={seriesColors[s.key] ?? CPU_SERIES_COLOR}
          />
        ))}
      </div>
      {"missing" in run.gpu && (
        <p className="mt-4 text-xs text-muted-foreground">
          GPU: not measured for this run. {run.gpu.missing}
        </p>
      )}
      <p className="mt-4 text-[11px] break-words text-muted-foreground">
        Registered resources:{" "}
        {run.resources
          .map((r) => `${r.resourceClass} (${r.resourceId})`)
          .join(", ")}
        . Only the CPU sandbox reports live samples; inference and retrieval
        have none yet.
      </p>
    </Card>
  );
}

export function LiveFeed({
  feed,
  scenarios,
}: {
  feed: LiveFeedData;
  scenarios: LivePageData["scenarios"];
}) {
  const selected = feed.selected;
  const runState = selected?.replay.run?.state;
  useRefreshWhileActive(runState === "running" || runState === "queued");
  return (
    <div className="space-y-6">
      {feed.error ? (
        <Card
          className="items-center gap-3 border-dashed p-8 text-center"
          role="alert"
        >
          <Unplug className="size-7 text-muted-foreground" aria-hidden />
          <h2 className="text-base font-semibold">Live feed unavailable</h2>
          <p className="max-w-lg text-sm break-words text-muted-foreground">
            {feed.error}
          </p>
        </Card>
      ) : (
        <>
          <div
            role="note"
            className="flex items-start gap-3 rounded-lg border border-sky-400/30 bg-sky-400/10 p-3 text-xs leading-5 text-sky-100"
          >
            <Database className="mt-0.5 size-4 shrink-0" aria-hidden />
            <p>
              <strong className="font-semibold tracking-wide uppercase">
                Live API
              </strong>{" "}
              — recorded runs read from <code>/v1/live/runs</code>
              {selected && ` · ${selected.recordCount} records`}. Prototype
              view; refreshes every 2 s while a run is active.
            </p>
          </div>
          <RunPicker
            feed={feed}
            current={selected?.replay.id ?? feed.requested ?? undefined}
          />
          {selected ? (
            <>
              <Resources run={selected} />
              <ReplayView replay={selected.replay} />
            </>
          ) : (
            <Card className="items-center gap-3 border-dashed p-8 text-center">
              <h2 className="text-base font-semibold">
                {feed.requested ? "Live run not found" : "No live runs yet"}
              </h2>
              <p className="max-w-lg text-sm break-words text-muted-foreground">
                {feed.requested
                  ? `The live API has no run ${feed.requested}.`
                  : "Start one with make agent-run; it appears here while it runs."}
              </p>
            </Card>
          )}
        </>
      )}
      <ScenarioPicker scenarios={scenarios} />
    </div>
  );
}
