"use client";
import { useEffect, useState } from "react";
import {
  CircleAlert,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  Clock3,
  LoaderCircle,
  RadioTower,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { PanelHeading } from "@/components/run-views";
import { duration } from "@/lib/dashboard-data";
import {
  activeSteps,
  failedSteps,
  feedClockTicks,
  feedDisplay,
  runOutcome,
  stepTotals,
  type BrowserClockReading,
  type CpuTotal,
  type FeedFreshness,
  type FeedReceipt,
  type RunStatus,
  type SpanTotal,
  type TimingTotal,
} from "@/lib/live-status";
import type {
  LiveRunView,
  LiveStep,
  SequenceGap,
  StepState,
} from "@/lib/live-view";
import { cn } from "@/lib/utils";

const FRESHNESS_CHECK_INTERVAL_MS = 5_000;

const stateStyles: Record<
  StepState,
  { label: string; icon: typeof CircleCheck; className: string }
> = {
  queued: {
    label: "Queued",
    icon: CircleDashed,
    className: "border-white/10 text-muted-foreground",
  },
  running: {
    label: "Running",
    icon: LoaderCircle,
    className: "border-sky-400/20 bg-sky-400/5 text-sky-200",
  },
  completed: {
    label: "Completed",
    icon: CircleCheck,
    className: "border-emerald-400/20 bg-emerald-400/5 text-emerald-300",
  },
  failed: {
    label: "Failed",
    icon: CircleAlert,
    className: "border-rose-400/25 bg-rose-400/5 text-rose-200",
  },
  cancelled: {
    label: "Cancelled",
    icon: CircleSlash,
    className: "border-amber-400/20 bg-amber-400/5 text-amber-200",
  },
};

const freshnessText: Record<FeedFreshness, { label: string; detail: string }> =
  {
    fresh: { label: "Receiving", detail: "Records arrived recently" },
    stale: {
      label: "Stale",
      detail: "No new records recently; the run may still be open",
    },
    closed: { label: "Closed", detail: "Run reached a terminal state" },
    never: { label: "No records", detail: "Nothing has been received yet" },
  };

function StateBadge({ state }: { state: StepState }) {
  const style = stateStyles[state];
  return (
    <Badge variant="outline" className={style.className}>
      <style.icon className="size-3" aria-hidden />
      {style.label}
    </Badge>
  );
}

function Timestamp({ iso }: { iso: string }) {
  // Formatted in the viewer's locale; server and browser time zones can differ.
  return (
    <time dateTime={iso} suppressHydrationWarning>
      {new Date(iso).toLocaleTimeString()}
    </time>
  );
}

function StatusTile({
  label,
  children,
  detail,
  announce = false,
}: {
  label: string;
  children: React.ReactNode;
  detail?: React.ReactNode;
  /** Announce value changes to screen readers (polite live region). */
  announce?: boolean;
}) {
  return (
    <Card className="gap-0 p-4 sm:p-5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div
        className="mt-3 flex flex-wrap items-center gap-2 text-sm font-semibold"
        {...(announce && {
          role: "status",
          "aria-live": "polite" as const,
          "aria-atomic": true,
        })}
      >
        {announce && <span className="sr-only">{label}:</span>}
        {children}
      </div>
      {detail && (
        <p className="mt-2 text-[11px] leading-5 break-words text-muted-foreground">
          {detail}
        </p>
      )}
    </Card>
  );
}

function stepCountText(total: { measuredSteps: number }) {
  return `${total.measuredSteps} step${total.measuredSteps === 1 ? "" : "s"}`;
}

function TimingValue({
  label,
  total,
  final,
}: {
  label: string;
  total: TimingTotal;
  final: boolean;
}) {
  return (
    <div>
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="metric-number mt-1 text-xl font-semibold">
        {total.measuredSteps
          ? duration(total.seconds)
          : total.unmeasuredSteps
            ? "Not measured"
            : "No finished steps"}
        {!final && total.measuredSteps > 0 && (
          <span className="ml-1.5 text-xs font-normal text-amber-200">
            so far
          </span>
        )}
      </dd>
      <dd className="mt-1 text-[11px] text-muted-foreground">
        {stepCountText(total)}
        {total.unmeasuredSteps > 0 &&
          ` · excludes ${total.unmeasuredSteps} that never started`}
      </dd>
    </div>
  );
}

function SpanValue({ total, final }: { total: SpanTotal; final: boolean }) {
  return (
    <div>
      <dt className="text-[11px] text-muted-foreground">Step wall-time span</dt>
      <dd className="metric-number mt-1 text-xl font-semibold">
        {total.measuredSteps === 0
          ? "No finished steps"
          : total.seconds === null
            ? "Not combined"
            : duration(total.seconds)}
        {!final && total.seconds !== null && total.measuredSteps > 0 && (
          <span className="ml-1.5 text-xs font-normal text-amber-200">
            so far
          </span>
        )}
      </dd>
      <dd className="mt-1 text-[11px] text-muted-foreground">
        {total.seconds === null && total.measuredSteps > 0
          ? `Steps use ${total.clockDomains} clock domains, which are not compared`
          : `${stepCountText(total)} · overlapping parent and child windows count once`}
      </dd>
    </div>
  );
}

function CpuValue({ total, final }: { total: CpuTotal; final: boolean }) {
  const steps = total.measuredSteps + total.unmeasuredSteps;
  return (
    <div>
      <dt className="text-[11px] text-muted-foreground">Process CPU time</dt>
      <dd className="metric-number mt-1 text-xl font-semibold">
        {total.seconds !== null
          ? duration(total.seconds)
          : steps
            ? "Not measured"
            : "No finished steps"}
        {!final && total.seconds !== null && (
          <span className="ml-1.5 text-xs font-normal text-amber-200">
            so far
          </span>
        )}
      </dd>
      <dd className="mt-1 text-[11px] text-muted-foreground">
        {steps === 0
          ? "Waiting for a finished step"
          : total.seconds === null
            ? `${total.measuredSteps} of ${steps} finished steps measured${total.unmeasuredSteps === 0 && steps > 0 ? " in different scopes" : ""}`
            : `Scope: ${total.scope === "process" ? "process" : "process and children"}`}
      </dd>
    </div>
  );
}

function Totals({ view, run }: { view: LiveRunView; run: RunStatus }) {
  const totals = stepTotals(view, run);
  return (
    <Card className="gap-0 p-5 sm:p-6">
      <PanelHeading
        title={totals.final ? "Final step totals" : "Partial step totals"}
        detail={
          totals.final
            ? "Totals for every step in the closed run."
            : "Not final: these totals cover finished steps only and will change."
        }
      >
        <Badge
          variant="outline"
          className={
            totals.final
              ? "border-emerald-400/20 text-emerald-300"
              : "border-amber-400/20 text-amber-200"
          }
        >
          {totals.final ? "Final" : "Incomplete"}
        </Badge>
      </PanelHeading>
      {!totals.final && (
        <ul className="mb-5 list-disc space-y-1 pl-5 text-xs text-amber-200">
          {totals.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
      <dl className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <SpanValue total={totals.duration} final={totals.final} />
        <TimingValue
          label="Queue delay"
          total={totals.queue}
          final={totals.final}
        />
        <TimingValue
          label="Execution wall time"
          total={totals.execution}
          final={totals.final}
        />
        <CpuValue total={totals.cpu} final={totals.final} />
      </dl>
      {totals.parentSteps > 0 && (
        <p className="mt-4 text-[11px] leading-5 text-muted-foreground">
          Queue delay, execution and CPU time are summed over steps without
          children; {totals.parentSteps} parent step
          {totals.parentSteps === 1 ? " is" : "s are"} left out because its
          window contains its children&apos;s. Concurrent steps still add up.
        </p>
      )}
    </Card>
  );
}

function unfinishedText(step: LiveStep) {
  if (step.state === "queued") return "Not started";
  return step.state === "running" ? "In progress" : "Not reported";
}

function gapText(gap: SequenceGap) {
  const held = `${gap.pendingRevisions} later revision${gap.pendingRevisions === 1 ? "" : "s"}`;
  const subject =
    gap.missingSequence === 0
      ? `Step ${gap.stepId} has no applied state: revision 0 has not arrived`
      : `Step ${gap.stepId} is applied through revision ${gap.missingSequence - 1}`;
  return gap.expired
    ? `${subject}. Revisions from ${gap.firstPendingSequence} expired unapplied (${held}) and will never be applied; this run cannot complete with them.`
    : `${subject} and is waiting for revision ${gap.missingSequence}; ${held} from ${gap.firstPendingSequence} ${gap.pendingRevisions === 1 ? "is" : "are"} held as pending and not shown.`;
}

function stepTiming(
  step: LiveStep,
  field: "queueSeconds" | "executionSeconds",
) {
  if (!step.timings) return unfinishedText(step);
  const value = step.timings[field];
  return value === null ? "Never started" : duration(value);
}

function StepRow({ step }: { step: LiveStep }) {
  return (
    <li className="rounded-lg border p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs break-all">{step.stepId}</span>
        <span className="text-[11px] text-muted-foreground">
          {step.stepType.replace("_", " ")}
        </span>
        <span className="ml-auto">
          <StateBadge state={step.state} />
        </span>
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 text-xs sm:grid-cols-3 lg:grid-cols-5">
        <div>
          <dt className="text-[11px] text-muted-foreground">
            Started (producer time)
          </dt>
          <dd className="mt-0.5">
            {step.startedObservedAt ? (
              <Timestamp iso={step.startedObservedAt} />
            ) : step.started ? (
              "Not received"
            ) : (
              "Not started"
            )}
          </dd>
        </div>
        <div>
          <dt className="text-[11px] text-muted-foreground">Queue delay</dt>
          <dd className="mt-0.5">{stepTiming(step, "queueSeconds")}</dd>
        </div>
        <div>
          <dt className="text-[11px] text-muted-foreground">Execution</dt>
          <dd className="mt-0.5">{stepTiming(step, "executionSeconds")}</dd>
        </div>
        <div>
          <dt className="text-[11px] text-muted-foreground">Duration</dt>
          <dd className="mt-0.5">
            {step.timings
              ? duration(step.timings.durationSeconds)
              : unfinishedText(step)}
          </dd>
        </div>
        <div>
          <dt className="text-[11px] text-muted-foreground">CPU</dt>
          <dd className="mt-0.5">
            {step.cpuSeconds === null
              ? "Not measured"
              : duration(step.cpuSeconds)}
          </dd>
        </div>
      </dl>
      {step.reason && (
        <p className="mt-3 text-xs break-words text-rose-200">
          Reason: {step.reason}
        </p>
      )}
    </li>
  );
}

/**
 * Reads only the browser's clocks, after mount, so feed age never compares server and browser
 * clocks. Returns null during server render and hydration, keeping the first client render
 * identical to the server's.
 */
function useBrowserClock(ticking: boolean) {
  const [now, setNow] = useState<{
    nowMonotonicMs: number;
    nowWallMs: number;
  } | null>(null);
  useEffect(() => {
    const read = () =>
      setNow({ nowMonotonicMs: performance.now(), nowWallMs: Date.now() });
    const first = setTimeout(read, 0);
    const timer = ticking
      ? setInterval(read, FRESHNESS_CHECK_INTERVAL_MS)
      : undefined;
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [ticking]);
  return now;
}

/** Run state, feed freshness, gaps, totals and steps of one live run. */
export function LiveRunSteps({
  run,
  view,
  receipt,
}: {
  run: RunStatus;
  view: LiveRunView;
  /** When the browser last received new records for this run; null before any. */
  receipt: FeedReceipt | null;
}) {
  const outcome = runOutcome(run);
  const receivedAgoMs = receipt?.ageAtAnchorMs ?? null;
  const now = useBrowserClock(feedClockTicks(receivedAgoMs));
  const clock: BrowserClockReading =
    receipt && now
      ? { ...now, anchorMonotonicMs: receipt.anchorMonotonicMs }
      : null;
  const feed = feedDisplay({ receivedAgoMs, outcome, clock });
  const freshness = feed.freshness;
  const active = activeSteps(view);
  const failed = failedSteps(view);
  return (
    <>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatusTile
          label="Run state"
          detail={
            run?.reason ?? (run ? undefined : "Not reported by the source")
          }
        >
          {run ? <StateBadge state={run.state} /> : "Unknown"}
        </StatusTile>
        <StatusTile
          label="Feed"
          detail={freshnessText[freshness].detail}
          announce
        >
          {freshness === "stale" ? (
            <Clock3 className="size-4 text-amber-300" aria-hidden />
          ) : (
            <RadioTower className="size-4 text-muted-foreground" aria-hidden />
          )}
          <span className={cn(freshness === "stale" && "text-amber-200")}>
            {freshnessText[freshness].label}
          </span>
        </StatusTile>
        <StatusTile
          label="Last record received"
          detail={
            feed.ageText === null ? (
              "No step record has been received for this run."
            ) : (
              <>
                {feed.receivedAtWallMs !== null && (
                  <>
                    At{" "}
                    <Timestamp
                      iso={new Date(feed.receivedAtWallMs).toISOString()}
                    />{" "}
                    on this device&apos;s clock.{" "}
                  </>
                )}
                {view.lastObservedAt && (
                  <>
                    Latest producer observation{" "}
                    <Timestamp iso={view.lastObservedAt} />
                  </>
                )}
              </>
            )
          }
        >
          {feed.ageText ?? "None yet"}
        </StatusTile>
        <StatusTile
          label="Steps"
          detail={`${view.steps.length - active.length} finished of ${view.steps.length} reported`}
        >
          {active.length} active · {failed.length} failed or cancelled
        </StatusTile>
      </div>
      {(view.gaps.length > 0 || view.otherRunRecords > 0) && (
        <div
          role="status"
          className="space-y-1 rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-xs text-amber-200"
        >
          {view.gaps.map((gap) => (
            <p key={gap.stepId} className="break-words">
              {gapText(gap)}
            </p>
          ))}
          {view.otherRunRecords > 0 && (
            <p>
              {view.otherRunRecords} record
              {view.otherRunRecords === 1 ? " belongs" : "s belong"} to another
              run and {view.otherRunRecords === 1 ? "was" : "were"} not applied.
            </p>
          )}
        </div>
      )}
      <Totals view={view} run={run} />
      <Card className="gap-0 p-5 sm:p-6">
        <PanelHeading
          title="Steps"
          detail="Applied state of each step. Running steps show their start time only; elapsed time is not computed across clocks."
        />
        {view.steps.length ? (
          <ul className="space-y-3">
            {view.steps.map((step) => (
              <StepRow key={step.stepId} step={step} />
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">
            No step has an applied state yet.
          </p>
        )}
      </Card>
    </>
  );
}
