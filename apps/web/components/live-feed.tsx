"use client";
import { routes } from "@/lib/routes";
import { useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  CircleAlert,
  Download,
  ExternalLink,
  FileSearch,
  FolderArchive,
  LoaderCircle,
  RadioTower,
  RefreshCw,
  Timer,
  Waypoints,
} from "lucide-react";
import {
  LiveGaps,
  LiveRunStatus,
  liveStepItems,
  StepStateBadge,
  stepTypeLabel,
} from "@/components/live-run";
import {
  FILL_PANEL,
  PANEL_SCROLL,
  TAB_GRID,
} from "@/components/shell/panel-layout";
import { TabbedView } from "@/components/shell/tabbed-view";
import { Button } from "@/components/ui/button";
import { CopyChip } from "@/components/ui/copy-chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge, TONE_DOT } from "@/components/ui/status-badge";
import { KeyValueList } from "@/components/widgets/evidence-lists";
import {
  EmptyState,
  ErrorState,
  Skeleton,
  UnavailableState,
} from "@/components/widgets/state-views";
import {
  TimelineChart,
  type TimelineChartRow,
} from "@/components/widgets/timeline-chart";
import { useLiveRun, useRunList } from "@/components/use-live-feed";
import {
  BAND_COLOR,
  SignalChart,
  type SignalBand,
} from "@/components/widgets/signal-chart";
import { formatSignal } from "@/lib/signal-format";
import { duration } from "@/lib/dashboard-data";
import { isLiveRunId } from "@/lib/live-api";
import {
  liveRunSummaries,
  summarizeSeries,
  type LiveFeedRun,
  type LiveRunSummary,
  type MetricSeries,
} from "@/lib/live-feed-data";
import {
  LIVE_POLL_INTERVAL_MS,
  runClosed,
  type FeedConnection,
  type LateCheck,
  type RunFeed,
  type RunListFeed,
} from "@/lib/live-poll";
import { wallSecondsInto, wallTimeline } from "@/lib/run-timeline";
import type { StepState } from "@/lib/live-view";
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

// Live runs read from the B01 live API through this app's routes, polled in the browser.

const seriesColors: Record<string, string> = {
  "gpu.util": "var(--series-violet)",
  "gpu.power": "var(--series-amber)",
  "gpu.nvml.utilization.gpu.v1": "var(--series-violet)",
  "gpu.nvml.power.usage_watts.v1": "var(--series-amber)",
};
const CPU_SERIES_COLOR = "var(--aw-brand-strong)";

const seconds = (ms: number) => `${Math.max(1, Math.round(ms / 1000))} s`;

const RUN_STATE_TONES: Record<string, Tone> = {
  queued: "info",
  running: "info",
  completed: "success",
  failed: "danger",
  cancelled: "warning",
};

/** The live timeline gives the chart more width than the step detail beside it. */
const WIDE_MAIN = "xl:grid-cols-[minmax(0,1.7fr)_minmax(19rem,1fr)]";

/** A selected live run as every tab reads it. */
export type LiveRunState = ReturnType<typeof useLiveRun>;

function connectionText(connection: FeedConnection, closed: boolean) {
  switch (connection.status) {
    case "connecting":
      return {
        label: "Connecting",
        detail: "Reading the selected run…",
        tone: "info" as Tone,
      };
    case "connected":
      return closed
        ? {
            label: "Polling stopped",
            detail:
              "The run is closed. Late samples or step evidence can still arrive.",
            tone: "info" as Tone,
          }
        : {
            label: "Connected",
            detail: `Reading new records every ${seconds(LIVE_POLL_INTERVAL_MS)}.`,
            tone: "success" as Tone,
          };
    case "retrying":
      return {
        label: `Reconnecting (attempt ${connection.attempt})`,
        detail: `${connection.message} Retrying in ${seconds(connection.retryInMs)}; the last data received stays shown.`,
        tone: "warning" as Tone,
      };
    case "stopped":
      return {
        label: "Stopped",
        detail: connection.message,
        tone: "danger" as Tone,
      };
  }
}

function lateCheckText(check: LateCheck) {
  switch (check.status) {
    case "checking":
      return "Checking the live API for late evidence…";
    case "checked": {
      const at = new Date(check.checkedAtWallMs).toLocaleTimeString();
      return check.newRecords === 0
        ? `Checked at ${at}: no new evidence.`
        : `Checked at ${at}: ${check.newRecords} new record${check.newRecords === 1 ? "" : "s"} received.`;
    }
    case "failed":
      return `Check failed at ${new Date(check.checkedAtWallMs).toLocaleTimeString()}: ${check.message}`;
  }
}

/** The outcome of the last late-evidence check, announced when it changes. */
function LateCheckResult({ check }: { check: LateCheck | null }) {
  return (
    <p
      className={cn(
        "basis-full text-sm break-words empty:sr-only",
        check?.status === "failed" ? "text-warning" : "text-ink-muted",
      )}
      role="status"
      aria-live="polite"
      aria-label="Late evidence check"
    >
      {check && lateCheckText(check)}
    </p>
  );
}

/** The live API connection for the selected run, with retry and late-evidence checks. */
function ConnectionBanner({
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
  const checking = feed.lateCheck?.status === "checking";
  const canReadAgain =
    connection.status === "retrying" ||
    (connection.status === "connected" && closed);
  return (
    <InsetCard className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <Icon
        className={cn(
          "size-5 shrink-0 text-ink-muted",
          connection.status === "connecting" && "motion-safe:animate-spin",
        )}
        aria-hidden
      />
      <div
        className="min-w-48 flex-1 text-sm"
        role="status"
        aria-live="polite"
        aria-atomic
      >
        <span className="sr-only">Live API connection: </span>
        <StatusBadge label={text.label} tone={text.tone} appearance="dot" />
        <span className="ml-2 text-ink-muted">{text.detail}</span>
      </div>
      {canReadAgain && (
        <Button
          variant="glass"
          size="pill"
          onClick={onReadAgain}
          disabled={checking}
        >
          {checking ? (
            <LoaderCircle className="motion-safe:animate-spin" aria-hidden />
          ) : (
            <RefreshCw aria-hidden />
          )}
          {checking
            ? "Checking…"
            : closed
              ? "Check for late evidence"
              : "Retry now"}
        </Button>
      )}
      <LateCheckResult check={feed.lateCheck} />
    </InsetCard>
  );
}

/**
 * What to show instead of a run's tabs: not found, not received yet or still connecting.
 * Null once the run has data. A missing run is never replaced by another one.
 */
export function LiveRunNotice({ live }: { live: LiveRunState }) {
  const { feed, run, readAgain } = live;
  if (!feed)
    return (
      <Skeleton label="Contacting the live API" className="h-48 fit:flex-1" />
    );
  if (run) return null;
  const notFound =
    feed.connection.status === "stopped" &&
    feed.connection.kind === "not_found";
  return (
    // The connection stays visible, so a failed or retrying read says why there is no data.
    <div className="space-y-3">
      <ConnectionBanner feed={feed} onReadAgain={readAgain} />
      {notFound ? (
        <ErrorState
          title="Live run not found"
          detail={`The live API has no run ${feed.runId}. No other run is shown in its place.`}
        />
      ) : feed.connection.status === "connecting" ? (
        <Skeleton label="Reading the live run" className="h-48" />
      ) : (
        <UnavailableState
          title="Live run unavailable"
          reason={`No data has been received for run ${feed.runId} yet.`}
        />
      )}
    </div>
  );
}

function stepRow(
  row: NonNullable<ReturnType<typeof wallTimeline>>["rows"][number],
  state: StepState | null,
): TimelineChartRow {
  const { step } = row;
  return {
    id: step.stepId,
    label: step.stepId,
    detail: (
      <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        {stepTypeLabel(step.stepType)}
        {state && state !== "completed" && (
          <StepStateBadge
            state={state}
            appearance="dot"
            className="px-2 py-0 text-xs"
          />
        )}
      </span>
    ),
    startShare: row.startShare,
    segments: row.segments,
    open: row.open,
    failed: state === "failed" || state === "cancelled",
    value:
      row.wallSeconds === null ? (
        <span className="font-normal text-ink-muted">Running</span>
      ) : (
        duration(row.wallSeconds)
      ),
    description: `${stepTypeLabel(step.stepType)}, ${state ?? step.state}${row.wallSeconds === null ? ", in progress" : ""}`,
  };
}

/** Steps on the producer's wall clock beside the selected step's applied state. */
export function LiveTimelineView({ run }: { run: LiveFeedRun }) {
  const timeline = useMemo(() => wallTimeline(run.wallSteps), [run.wallSteps]);
  const steps = run.view.steps;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected =
    steps.find((step) => step.stepId === selectedId) ??
    steps.find((step) => step.state === "running") ??
    steps.at(-1);
  const charted = new Set(timeline?.rows.map((row) => row.step.stepId));
  const notStarted = steps.filter((step) => !charted.has(step.stepId));
  const stateOf = (stepId: string) =>
    steps.find((step) => step.stepId === stepId)?.state ?? null;

  return (
    <div className={cn(TAB_GRID, WIDE_MAIN, "fit:h-full")}>
      <GlassPanel
        as="section"
        aria-labelledby="live-timeline"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="live-timeline"
          icon={Waypoints}
          title="Steps over time"
          description={
            timeline
              ? `${timeline.rows.length} steps in a ${duration(timeline.spanSeconds)} window`
              : "No step has started yet."
          }
          className="fit:shrink-0"
        />
        <div className={PANEL_SCROLL}>
          {timeline ? (
            <TimelineChart
              rows={timeline.rows.map((row) =>
                stepRow(row, stateOf(row.step.stepId)),
              )}
              ticks={timeline.ticks}
              label="Step timeline"
              selectedId={selected?.stepId}
              onSelect={setSelectedId}
            />
          ) : (
            <EmptyState
              title="Nothing to chart yet"
              detail="Steps appear here once the producer reports them running."
            />
          )}
          {notStarted.length > 0 && (
            <p className="mt-3 px-3 text-sm text-ink-muted [overflow-wrap:anywhere]">
              Not on the chart (no running event yet):{" "}
              {notStarted.map((step) => step.stepId).join(", ")}
            </p>
          )}
        </div>
        <p className="mt-3 flex items-start gap-2 text-xs text-ink-muted fit:shrink-0">
          <Timer className="mt-px size-4 shrink-0" aria-hidden />
          Bars use the producer&apos;s wall clock (observed_at) and are
          approximate. The step details use the runner&apos;s monotonic ticks.
        </p>
      </GlassPanel>
      <GlassPanel
        as="section"
        aria-labelledby="live-step-detail"
        aria-live="polite"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="live-step-detail"
          icon={FileSearch}
          title="Selected step"
          description={
            <span className="[overflow-wrap:anywhere]">
              {selected?.stepId ?? "No step reported"}
            </span>
          }
          className="fit:shrink-0"
          actions={selected && <StepStateBadge state={selected.state} />}
        />
        {selected ? (
          <div
            tabIndex={0}
            role="group"
            aria-label={`Applied state of ${selected.stepId}`}
            className={cn(PANEL_SCROLL, "space-y-3")}
          >
            <InsetCard>
              <KeyValueList items={liveStepItems(selected)} dense />
            </InsetCard>
            {selected.reason && (
              <p className="rounded-2xl border border-danger-line bg-danger-soft px-4 py-3 text-sm break-words text-danger">
                Reason: {selected.reason}
              </p>
            )}
            <p className="text-xs text-ink-muted">
              Running steps show their start time only; elapsed time is not
              computed across clocks.
            </p>
          </div>
        ) : (
          <EmptyState
            title="No step reported"
            detail="No step has an applied state yet."
          />
        )}
      </GlassPanel>
    </div>
  );
}

/** "Range 35.1%–42.0% · mean 40.6%": the chart's values as text. */
function rangeText(series: MetricSeries) {
  const summary = summarizeSeries(series);
  if (!summary) return "No measured values.";
  const f = (value: number) => formatSignal(value, series.unit);
  return `Range ${f(summary.min)}–${f(summary.max)} · mean ${f(summary.mean)}`;
}

function measuredText(series: MetricSeries) {
  const measured = series.points.filter((p) => p.value !== null).length;
  const missing = series.notMeasured
    ? Object.entries(series.notMeasured).map(
        ([state, { count }]) => ` · ${count} ${state}`,
      )
    : series.unavailable > 0
      ? [` · ${series.unavailable} unavailable`]
      : [];
  return `${measured} measured${missing.join("")}`;
}

function MetricChart({
  series,
  timeline,
  modelCalls,
}: {
  series: MetricSeries;
  timeline: NonNullable<ReturnType<typeof wallTimeline>>;
  modelCalls: SignalBand[];
}) {
  const { window } = timeline;
  return (
    <SignalChart
      title={series.label}
      label={
        series.resource
          ? `${series.label}, resource ${series.resource.resourceId}`
          : series.label
      }
      unit={series.unit}
      color={seriesColors[series.metric] ?? CPU_SERIES_COLOR}
      spanSeconds={timeline.spanSeconds}
      ticks={timeline.ticks}
      points={series.points.map((p) => ({
        seconds: wallSecondsInto(p.atMs, window),
        value: p.value,
      }))}
      bands={modelCalls}
      bandLabel="Model calls"
      meta={`${series.unit} · ${measuredText(series)}`}
      footer={
        <>
          <p className="mt-2 text-xs text-ink-muted">{rangeText(series)}</p>
          {series.notMeasured &&
            Object.entries(series.notMeasured).map(([state, { reason }]) => (
              <p
                key={state}
                className="mt-1 text-xs break-words text-ink-muted"
              >
                Not measured ({state}){reason ? `: ${reason}` : ""}. Shown as
                gaps, never as zero.
              </p>
            ))}
        </>
      }
    />
  );
}

/** Charts from one resource and service (or one diagnostic source) sit under one heading. */
function groupByResource(series: readonly MetricSeries[]) {
  const groups = new Map<string, { title: string; series: MetricSeries[] }>();
  for (const entry of series) {
    const key = entry.resource
      ? `${entry.resource.resourceId}|${entry.resource.serviceId}`
      : entry.source;
    const title = entry.resource
      ? `${entry.resource.resourceId} · ${entry.resource.serviceId} · ${entry.source}`
      : entry.source;
    const group = groups.get(key) ?? { title, series: [] };
    group.series.push(entry);
    groups.set(key, group);
  }
  return [...groups.entries()].map(([key, group]) => ({ key, ...group }));
}

/** Every live sample a registered resource reported, aligned to the run's wall clock. */
export function LiveResourcesView({ run }: { run: LiveFeedRun }) {
  const timeline = useMemo(() => wallTimeline(run.wallSteps), [run.wallSteps]);
  const modelCalls: SignalBand[] = timeline
    ? run.wallSteps
        .filter((s) => s.stepType === "model_call")
        .map((call) => ({
          id: call.stepId,
          from: wallSecondsInto(call.startMs, timeline.window),
          to: wallSecondsInto(
            call.endMs ?? timeline.window.endMs,
            timeline.window,
          ),
        }))
    : [];
  const gpu = run.gpu && "series" in run.gpu ? run.gpu : null;
  const liveGpu = run.liveSeries.some((s) => s.metric.startsWith("gpu."));
  const charts = [
    ...(gpu?.series.filter((s) => s.key !== "gpu.mem") ?? []),
    ...run.liveSeries,
  ];
  return (
    <GlassPanel
      as="section"
      aria-labelledby="live-resources"
      className={cn(FILL_PANEL, "fit:h-full")}
    >
      <PanelHeader
        titleId="live-resources"
        icon={Activity}
        title="Resource signals"
        description="Whole-host or whole-device samples, not one step's, on the step timeline's clock."
        className="fit:shrink-0"
        actions={
          modelCalls.length > 0 && (
            <span className="flex items-center gap-2 text-sm text-ink">
              <i
                aria-hidden
                className="h-1.5 w-5 rounded-full"
                style={{ background: BAND_COLOR }}
              />
              Model calls (lane under each chart)
            </span>
          )
        }
      />
      <div
        tabIndex={0}
        role="group"
        aria-label="Resource signal charts"
        className={cn(PANEL_SCROLL, "space-y-3")}
      >
        {timeline && charts.length > 0 ? (
          groupByResource(charts).map((group) => (
            <section key={group.key} aria-label={group.title}>
              <h3
                className="mb-2 truncate font-mono text-xs text-ink-muted"
                title={group.title}
              >
                {group.title}
              </h3>
              <div className="grid grid-cols-1 gap-3 @container lg:grid-cols-2">
                {group.series.map((s) => (
                  <MetricChart
                    key={s.key}
                    series={s}
                    timeline={timeline}
                    modelCalls={modelCalls}
                  />
                ))}
              </div>
            </section>
          ))
        ) : (
          <UnavailableState
            title="No resource samples yet"
            reason={
              timeline
                ? "No registered resource has reported a live sample for this run."
                : "Signals are aligned to the steps' wall clock, and no step has started yet."
            }
          />
        )}
        {run.gpu && "missing" in run.gpu && !liveGpu && (
          <p className="text-sm text-ink-muted">
            GPU: not measured for this run. {run.gpu.missing}
          </p>
        )}
        <p className="text-xs break-words text-ink-muted">
          Registered resources:{" "}
          {run.resources.length
            ? run.resources
                .map((r) => `${r.resourceClass} (${r.resourceId})`)
                .join(", ")
            : "none"}
          . Retrieval has no registered live metric yet. Signals labeled
          nvidia-smi come from the A01 diagnostic, which is not ingested and is
          aligned only by wall clock.
        </p>
      </div>
    </GlassPanel>
  );
}

/** What can be downloaded for a live run, and why there is no saved-run file. */
export function LiveEvidenceView({
  runId,
  run,
}: {
  runId: string;
  run: LiveFeedRun;
}) {
  return (
    <div className={cn(TAB_GRID, "fit:h-full")}>
      <GlassPanel
        as="section"
        aria-labelledby="live-evidence"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="live-evidence"
          icon={FolderArchive}
          title="Live records"
          description={`${run.recordCount} record${run.recordCount === 1 ? "" : "s"} received for this run.`}
          className="fit:shrink-0"
        />
        <div className={cn(PANEL_SCROLL, "space-y-3")}>
          <InsetCard className="space-y-3">
            <p className="text-sm text-ink-muted">
              Every record the live API holds for run{" "}
              <span className="font-mono break-all text-ink">{runId}</span>,
              read page by page with its gaps and pending or expired records
              unchanged. Run state, steps and gaps are as of the time of
              reading.
            </p>
            <Button asChild variant="brand-soft" size="pill">
              <a href={routes.liveExport(runId)}>
                <Download aria-hidden />
                Download live records (JSON)
              </a>
            </Button>
          </InsetCard>
          <LiveGaps view={run.view} />
        </div>
      </GlassPanel>
      <GlassPanel
        as="section"
        aria-labelledby="live-saved-evidence"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="live-saved-evidence"
          icon={FileSearch}
          title="Saved evidence"
          description="Not available for live runs."
          className="fit:shrink-0"
        />
        <InsetCard>
          <p className="text-sm text-ink-muted">
            Live records are kept separately from saved run history, and agent
            runs report only to the live API, so this run has no results.json,
            spans.jsonl or samples.jsonl.
          </p>
        </InsetCard>
      </GlassPanel>
    </div>
  );
}

/** The Status tab: connection, state tiles, totals and policy. */
export function LiveStatusView({ live }: { live: LiveRunState }) {
  const { feed, run, readAgain } = live;
  if (!feed || !run) return null;
  return (
    <LiveRunStatus
      runId={feed.runId}
      run={run.run}
      view={run.view}
      receipt={feed.receipt}
      connection={<ConnectionBanner feed={feed} onReadAgain={readAgain} />}
    />
  );
}

/** The content of a live run's Status, Timeline, Resources and Evidence tabs; null without data. */
export function liveRunTabs(live: LiveRunState) {
  const { feed, run } = live;
  if (!feed || !run) return null;
  return {
    status: <LiveStatusView live={live} />,
    timeline: <LiveTimelineView run={run} />,
    resources: <LiveResourcesView run={run} />,
    evidence: <LiveEvidenceView runId={feed.runId} run={run} />,
  };
}

/**
 * One live run's status, timeline, resources and evidence stacked, read by UUID. For views
 * without tabs; the Live and Run pages show the same panels as tabs.
 */
export function LiveRunDetail({ runId }: { runId: string }) {
  const live = useLiveRun(runId);
  const tabs = liveRunTabs(live);
  return (
    <div className="space-y-4">
      {tabs ? (
        <>
          {tabs.status}
          {tabs.timeline}
          {tabs.resources}
          {tabs.evidence}
        </>
      ) : (
        <LiveRunNotice live={live} />
      )}
    </div>
  );
}

function ListStatus({ list, count }: { list: RunListFeed; count: number }) {
  const { connection } = list;
  return (
    <span role="status" aria-live="polite">
      {connection.status === "connecting"
        ? "Loading live runs…"
        : `Newest ${count} from the live API.`}
      {connection.status === "retrying" &&
        ` The list could not be refreshed (${connection.message}); retrying in ${seconds(connection.retryInMs)}.`}
      {connection.status === "stopped" &&
        ` The list could not be read: ${connection.message}`}
    </span>
  );
}

function RunList({
  runs,
  list,
  current,
}: {
  runs: LiveRunSummary[];
  list: RunListFeed;
  current: string | null;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="live-run-list"
      className={cn(FILL_PANEL, "fit:h-full")}
    >
      <PanelHeader
        titleId="live-run-list"
        icon={RadioTower}
        title="Recent live runs"
        description={<ListStatus list={list} count={runs.length} />}
        className="fit:shrink-0"
      />
      {runs.length > 0 ? (
        <nav aria-label="Recorded live runs" className={PANEL_SCROLL}>
          <ul className="space-y-1">
            {runs.map((run) => {
              const selected = run.runId === current;
              const tone = RUN_STATE_TONES[run.state] ?? "info";
              return (
                <li key={run.runId}>
                  <Link
                    href={routes.live(run.runId)}
                    aria-current={selected ? "page" : undefined}
                    className={cn(
                      "grid min-h-11 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-3 rounded-xl border border-transparent px-3 py-2 text-ink outline-none transition-colors hover:bg-glass-strong focus-visible:ring-[3px] focus-visible:ring-brand/40",
                      selected &&
                        "border-success-line bg-brand-soft shadow-glow",
                    )}
                  >
                    <i
                      aria-hidden
                      className={cn("size-2.5 rounded-full", TONE_DOT[tone])}
                    />
                    <span className="min-w-0" title={run.runId}>
                      <span className="block truncate font-mono text-sm font-medium">
                        {run.runId.slice(0, 8)}
                      </span>
                      <span className="block truncate text-xs text-ink-muted">
                        {run.taskId}
                      </span>
                    </span>
                    <span className="text-right text-xs text-ink-muted">
                      <span className="block capitalize">{run.state}</span>
                      <time dateTime={run.createdAt} suppressHydrationWarning>
                        {new Date(run.createdAt).toLocaleTimeString()}
                      </time>
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      ) : (
        list.connection.status !== "connecting" && (
          <EmptyState
            title="No live runs yet"
            detail="Start one with make agent-run; it appears here while it runs."
          />
        )
      )}
    </GlassPanel>
  );
}

function SelectedRun({ runId }: { runId: string }) {
  const live = useLiveRun(runId);
  const tabs = liveRunTabs(live);
  const state = live.run?.run?.state ?? null;
  return (
    <TabbedView
      label="Live run views"
      param="tab"
      header={
        <div className="flex flex-wrap items-center gap-2">
          <CopyChip value={runId} label="run ID" />
          {state && <StepStateBadge state={state} />}
          {live.run && (
            <p className="min-w-0 text-sm text-ink-muted">
              {live.run.recordCount} record
              {live.run.recordCount === 1 ? "" : "s"} received for run{" "}
              <span className="font-mono break-all">{runId}</span>
            </p>
          )}
          <Button asChild variant="glass" size="pill" className="ml-auto">
            <Link href={routes.run(runId)}>
              <ExternalLink aria-hidden />
              Open run page
            </Link>
          </Button>
        </div>
      }
      tabs={
        tabs
          ? [
              { id: "status", label: "Status", content: tabs.status },
              { id: "timeline", label: "Timeline", content: tabs.timeline },
              // Charts measure their container, so they render only while visible.
              {
                id: "resources",
                label: "Resources",
                content: tabs.resources,
                lazy: true,
              },
              { id: "evidence", label: "Evidence", content: tabs.evidence },
            ]
          : [
              {
                id: "status",
                label: "Status",
                content: <LiveRunNotice live={live} />,
              },
            ]
      }
    />
  );
}

/**
 * The Live runs page body: the newest runs beside the selected run's tabs. `requested` is
 * the `?liveRun=` UUID, or null to follow the newest run when the list first loads.
 */
export function LiveFeed({ requested }: { requested: string | null }) {
  const { list, firstNewest } = useRunList();
  const requestedValid = requested === null || isLiveRunId(requested);
  // An explicit selection is read directly by UUID, so it need not be on the newest page.
  const selectedId = requested ?? firstNewest;
  const runs = useMemo(
    () => (list.page ? liveRunSummaries(list.page) : []),
    [list.page],
  );
  return (
    <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)] fit:h-full fit:min-h-0 fit:grid-rows-[minmax(0,1fr)]">
      <RunList runs={runs} list={list} current={selectedId} />
      <div className="min-w-0 fit:min-h-0">
        {!requestedValid ? (
          <ErrorState
            title="Not a live run ID"
            detail={`“${requested}” is not a live run UUID. Choose a run from the list.`}
          />
        ) : selectedId ? (
          // Keyed so a new selection starts with fresh tab state and no previous data.
          <SelectedRun key={selectedId} runId={selectedId} />
        ) : list.connection.status === "connected" ? (
          <EmptyState
            title="No live run selected"
            detail="Start one with make agent-run, then choose it from the list."
          />
        ) : (
          <UnavailableState
            title={
              list.connection.status === "connecting"
                ? "Connecting to the live API"
                : "Live API unavailable"
            }
            reason={
              list.connection.status === "connecting"
                ? "Reading the newest live runs…"
                : "No run is selected and the run list cannot be read yet. It is retried automatically."
            }
          />
        )}
      </div>
    </div>
  );
}
