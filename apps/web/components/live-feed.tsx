"use client";
import { useMemo } from "react";
import Link from "next/link";
import {
  CircleAlert,
  Database,
  Download,
  LoaderCircle,
  RadioTower,
  RefreshCw,
  Unplug,
} from "lucide-react";
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
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PanelHeading } from "@/components/run-views";
import { ReplayView, ScenarioPicker } from "@/components/live-run";
import {
  useGpuEvidence,
  useRunFeed,
  useRunList,
} from "@/components/use-live-feed";
import { duration } from "@/lib/dashboard-data";
import { isLiveRunId } from "@/lib/live-api";
import {
  liveFeedRun,
  liveRunSummaries,
  summarizeSeries,
  type LiveFeedRun,
  type LiveRunSummary,
  type MetricSeries,
  type WallStep,
} from "@/lib/live-feed-data";
import {
  LIVE_POLL_INTERVAL_MS,
  runClosed,
  type FeedConnection,
  type RunFeed,
  type RunListFeed,
} from "@/lib/live-poll";
import type { LivePageData } from "@/lib/live-replay";
import { cn } from "@/lib/utils";

// Live runs read from the B01 live API through this app's routes, polled in the browser.

const stepColors: Record<string, string> = {
  model_call: "#a78bfa",
  tool: "#34d399",
  retrieval: "#38bdf8",
  other: "#64748b",
};
const seriesColors: Record<string, string> = {
  "gpu.util": "#a78bfa",
  "gpu.power": "#fbbf24",
  "gpu.nvml.utilization.gpu.v1": "#a78bfa",
  "gpu.nvml.power.usage_watts.v1": "#fbbf24",
};
const CPU_SERIES_COLOR = "#34d399";
const tooltipStyle = {
  background: "#171d27",
  border: "1px solid #303948",
  borderRadius: 8,
  fontSize: 12,
};

const seconds = (ms: number) => `${Math.max(1, Math.round(ms / 1000))} s`;

function RunPicker({
  runs,
  list,
  current,
}: {
  runs: LiveRunSummary[];
  list: RunListFeed;
  current: string | null;
}) {
  const { connection } = list;
  return (
    <nav aria-label="Recorded live runs">
      <p className="mb-2 text-[11px] text-muted-foreground">
        {connection.status === "connecting"
          ? "Loading live runs…"
          : `Newest live runs from the API (${runs.length}):`}
        {connection.status === "retrying" &&
          ` The run list could not be refreshed (${connection.message}); retrying in ${seconds(connection.retryInMs)}.`}
        {connection.status === "stopped" &&
          ` The run list could not be read: ${connection.message}`}
      </p>
      {runs.length > 0 && (
        <ul className="flex flex-wrap gap-2">
          {runs.map((run) => (
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
      )}
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
          {series.notMeasured
            ? Object.entries(series.notMeasured).map(
                ([state, { count }]) => ` · ${count} ${state}`,
              )
            : series.unavailable > 0 && ` · ${series.unavailable} unavailable`}
        </span>
      </div>
      {series.resource && (
        <p className="mb-2 text-[10px] break-all text-muted-foreground">
          Resource{" "}
          <span className="font-mono">{series.resource.resourceId}</span> ·
          service <span className="font-mono">{series.resource.serviceId}</span>
        </p>
      )}
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
      {series.notMeasured &&
        Object.entries(series.notMeasured).map(([state, { reason }]) => (
          <p
            key={state}
            className="mt-1 text-[11px] break-words text-muted-foreground"
          >
            Not measured ({state}){reason ? `: ${reason}` : ""}. Shown as gaps,
            never as zero.
          </p>
        ))}
    </Card>
  );
}

function Resources({ run }: { run: LiveFeedRun }) {
  const window = runWindow(run.wallSteps);
  if (!window) return null;
  const modelCalls = run.wallSteps.filter((s) => s.stepType === "model_call");
  const gpu = run.gpu && "series" in run.gpu ? run.gpu : null;
  const liveGpu = run.liveSeries.some((s) => s.metric.startsWith("gpu."));
  const charts = [
    ...(gpu?.series.filter((s) => s.key !== "gpu.mem") ?? []),
    ...run.liveSeries,
  ];
  return (
    <Card className="gap-0 p-5 sm:p-6">
      <PanelHeading
        title="Step timeline and resource signals"
        detail="Shaded bands mark model-call steps. CPU and NVML GPU signals are ingested live samples of the whole host or device, not of one step. Signals labeled nvidia-smi come from the A01 diagnostic, which is not ingested and is aligned only by wall clock."
      />
      <Timeline steps={run.wallSteps} window={window} />
      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {charts.map((s) => (
          <MetricChart
            key={s.key}
            series={s}
            window={window}
            modelCalls={modelCalls}
            color={seriesColors[s.metric] ?? CPU_SERIES_COLOR}
          />
        ))}
      </div>
      {run.gpu && "missing" in run.gpu && !liveGpu && (
        <p className="mt-4 text-xs text-muted-foreground">
          GPU: not measured for this run. {run.gpu.missing}
        </p>
      )}
      <p className="mt-4 text-[11px] break-words text-muted-foreground">
        Registered resources:{" "}
        {run.resources
          .map((r) => `${r.resourceClass} (${r.resourceId})`)
          .join(", ")}
        . Every live sample a resource reported is charted above; retrieval has
        no registered live metric yet.
      </p>
    </Card>
  );
}

function connectionText(connection: FeedConnection, closed: boolean) {
  switch (connection.status) {
    case "connecting":
      return { label: "Connecting", detail: "Reading the selected run…" };
    case "connected":
      return closed
        ? {
            label: "Polling stopped",
            detail:
              "The run is closed. Late samples or step evidence can still arrive.",
          }
        : {
            label: "Connected",
            detail: `Reading new records every ${seconds(LIVE_POLL_INTERVAL_MS)}.`,
          };
    case "retrying":
      return {
        label: `Reconnecting (attempt ${connection.attempt})`,
        detail: `${connection.message} Retrying in ${seconds(connection.retryInMs)}; the last data received stays shown.`,
      };
    case "stopped":
      return { label: "Stopped", detail: connection.message };
  }
}

function ConnectionStatus({
  feed,
  onReadAgain,
}: {
  feed: RunFeed;
  onReadAgain: () => void;
}) {
  const closed = runClosed(feed);
  const { connection } = feed;
  const text = connectionText(connection, closed);
  const Icon =
    connection.status === "retrying" || connection.status === "stopped"
      ? CircleAlert
      : connection.status === "connecting"
        ? LoaderCircle
        : RadioTower;
  const canReadAgain =
    connection.status === "retrying" ||
    (connection.status === "connected" && closed);
  return (
    <Card className="flex-row flex-wrap items-center gap-3 p-4">
      <Icon
        className={cn(
          "size-4 shrink-0",
          connection.status === "retrying"
            ? "text-amber-300"
            : "text-muted-foreground",
        )}
        aria-hidden
      />
      <div
        className="min-w-0 flex-1 text-xs"
        role="status"
        aria-live="polite"
        aria-atomic
      >
        <span className="sr-only">Live API connection: </span>
        <span
          className={cn(
            "font-semibold",
            connection.status === "retrying" && "text-amber-200",
          )}
        >
          {text.label}
        </span>
        <span className="text-muted-foreground"> — {text.detail}</span>
      </div>
      {canReadAgain && (
        <Button
          variant="outline"
          size="sm"
          className="min-h-11"
          onClick={onReadAgain}
        >
          <RefreshCw className="size-3.5" aria-hidden />
          {closed ? "Check for late evidence" : "Retry now"}
        </Button>
      )}
    </Card>
  );
}

function EmptyState({
  title,
  detail,
  alert = false,
}: {
  title: string;
  detail: string;
  alert?: boolean;
}) {
  return (
    <Card
      className="items-center gap-3 border-dashed p-8 text-center"
      role={alert ? "alert" : undefined}
    >
      <Unplug className="size-7 text-muted-foreground" aria-hidden />
      <h2 className="text-base font-semibold">{title}</h2>
      <p className="max-w-lg text-sm break-words text-muted-foreground">
        {detail}
      </p>
    </Card>
  );
}

function LiveEvidence({ runId }: { runId: string }) {
  return (
    <Card className="gap-0 p-5 sm:p-6">
      <PanelHeading
        title="Evidence for this run"
        detail="What can be downloaded for the selected live run."
      />
      <dl className="grid grid-cols-1 gap-4 text-xs sm:grid-cols-2">
        <div>
          <dt className="font-semibold">Saved evidence</dt>
          <dd className="mt-1 leading-5 text-muted-foreground">
            Not available for live runs. Live records are kept separately from
            saved run history, and agent runs report only to the live API, so
            this run has no saved-run file.
          </dd>
        </div>
        <div>
          <dt className="font-semibold">Live records</dt>
          <dd className="mt-1 leading-5 text-muted-foreground">
            Every record the live API holds for run{" "}
            <span className="font-mono break-all">{runId}</span>, read page by
            page with its gaps and pending or expired records unchanged. Run
            state, steps and gaps are as of the time of reading.
          </dd>
          <dd className="mt-3">
            <Button asChild variant="outline" className="min-h-11">
              <a href={`/api/live/runs/${encodeURIComponent(runId)}/export`}>
                <Download className="size-4" aria-hidden />
                Download live records (JSON)
              </a>
            </Button>
          </dd>
        </div>
      </dl>
    </Card>
  );
}

function SelectedRun({
  feed,
  onReadAgain,
}: {
  feed: RunFeed;
  onReadAgain: () => void;
}) {
  const { snapshot, records, connection } = feed;
  const gpu = useGpuEvidence(snapshot ? feed.runId : null, runClosed(feed));
  const run: LiveFeedRun | null = useMemo(
    () => (snapshot ? liveFeedRun(snapshot, records, gpu) : null),
    [snapshot, records, gpu],
  );
  const notFound =
    connection.status === "stopped" && connection.kind === "not_found";
  return (
    <>
      <ConnectionStatus feed={feed} onReadAgain={onReadAgain} />
      {notFound && (
        <EmptyState
          title="Live run not found"
          detail={`The live API has no run ${feed.runId}. No other run is shown in its place.`}
          alert
        />
      )}
      {run ? (
        <>
          <p className="text-[11px] text-muted-foreground">
            {run.recordCount} record{run.recordCount === 1 ? "" : "s"} received
            for run <span className="font-mono break-all">{feed.runId}</span>.
          </p>
          <Resources run={run} />
          <ReplayView replay={run.replay} receipt={feed.receipt} />
          <LiveEvidence runId={feed.runId} />
        </>
      ) : (
        !notFound &&
        connection.status !== "connecting" && (
          <EmptyState
            title="Live run unavailable"
            detail={`No data has been received for run ${feed.runId} yet.`}
          />
        )
      )}
    </>
  );
}

export function LiveFeed({
  requested,
  scenarios,
}: {
  /** The `?liveRun=` UUID, or null to follow the newest run when the list first loads. */
  requested: string | null;
  scenarios: LivePageData["scenarios"];
}) {
  const { list, firstNewest } = useRunList();
  const requestedValid = requested === null || isLiveRunId(requested);
  // An explicit selection is read directly by UUID, so it need not be on the newest page.
  const selectedId = requested ?? firstNewest;
  const { feed, readAgain } = useRunFeed(requestedValid ? selectedId : null);
  const runs = useMemo(
    () => (list.page ? liveRunSummaries(list.page) : []),
    [list.page],
  );
  return (
    <div className="space-y-6">
      <div
        role="note"
        className="flex items-start gap-3 rounded-lg border border-sky-400/30 bg-sky-400/10 p-3 text-xs leading-5 text-sky-100"
      >
        <Database className="mt-0.5 size-4 shrink-0" aria-hidden />
        <p>
          <strong className="font-semibold tracking-wide uppercase">
            Live API
          </strong>{" "}
          — recorded runs read from <code>/v1/live/runs</code> while this page
          is open. Polling pauses while the tab is hidden.
        </p>
      </div>
      <RunPicker runs={runs} list={list} current={selectedId} />
      {!requestedValid ? (
        <EmptyState
          title="Not a live run ID"
          detail={`“${requested}” is not a live run UUID. Choose a run above.`}
          alert
        />
      ) : feed ? (
        <SelectedRun feed={feed} onReadAgain={readAgain} />
      ) : (
        list.connection.status === "connected" && (
          <EmptyState
            title="No live runs yet"
            detail="Start one with make agent-run; it appears here while it runs."
          />
        )
      )}
      {list.connection.status !== "connected" && !feed && requestedValid && (
        <EmptyState
          title={
            list.connection.status === "connecting"
              ? "Connecting to the live API"
              : "Live API unavailable"
          }
          detail={
            list.connection.status === "connecting"
              ? "Reading the newest live runs…"
              : "No run is selected and the run list cannot be read yet. It is retried automatically."
          }
          alert={list.connection.status !== "connecting"}
        />
      )}
      <ScenarioPicker scenarios={scenarios} />
    </div>
  );
}
