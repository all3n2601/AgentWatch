"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import {
  CircleAlert,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  Clock3,
  FlaskConical,
  LoaderCircle,
  RadioTower,
  Unplug,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { PanelHeading } from "@/components/run-views";
import { duration } from "@/lib/dashboard-data";
import type { LivePageData, LiveReplay } from "@/lib/live-replay";
import {
  activeSteps,
  failedSteps,
  feedFreshness,
  runOutcome,
  stepTotals,
  type CpuTotal,
  type FeedFreshness,
  type TimingTotal,
} from "@/lib/live-status";
import type { LiveStep, StepState } from "@/lib/step-events";
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
}: {
  label: string;
  children: React.ReactNode;
  detail?: React.ReactNode;
}) {
  return (
    <Card className="gap-0 p-4 sm:p-5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm font-semibold">
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

function ReplayBanner({ replay }: { replay: LiveReplay }) {
  return (
    <div
      role="note"
      aria-label="Development replay"
      className="flex items-start gap-3 rounded-lg border border-violet-400/30 bg-violet-400/10 p-3 text-xs leading-5 text-violet-100"
    >
      <FlaskConical className="mt-0.5 size-4 shrink-0" aria-hidden />
      <p>
        <strong className="font-semibold uppercase tracking-wide">
          Development replay
        </strong>{" "}
        — synthetic scripted events, not recorded evidence. Scenario:{" "}
        {replay.title}. {replay.description}
      </p>
    </div>
  );
}

function ScenarioPicker({
  scenarios,
  current,
}: {
  scenarios: LivePageData["scenarios"];
  current?: string;
}) {
  if (!scenarios.length) return null;
  return (
    <nav aria-label="Development replay scenarios">
      <p className="mb-2 text-[11px] text-muted-foreground">
        Development replay scenarios (synthetic):
      </p>
      <ul className="flex flex-wrap gap-2">
        {scenarios.map((scenario) => (
          <li key={scenario.id}>
            <Link
              href={`/live?replay=${scenario.id}`}
              aria-current={scenario.id === current ? "page" : undefined}
              className={cn(
                "inline-flex min-h-11 items-center rounded-md border px-3 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-primary",
                scenario.id === current
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {scenario.title}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function stepCountText(total: TimingTotal) {
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

function Totals({ replay }: { replay: LiveReplay }) {
  const totals = stepTotals(replay.view, replay.run);
  return (
    <Card className="gap-0 p-5 sm:p-6">
      <PanelHeading
        title={totals.final ? "Final step totals" : "Partial step totals"}
        detail={
          totals.final
            ? "Sums of per-step timings for every step in the closed run."
            : "Not final: these sums cover finished steps only and will change."
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
        <TimingValue
          label="Total step duration"
          total={totals.duration}
          final={totals.final}
        />
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
    </Card>
  );
}

function unfinishedText(step: LiveStep) {
  return step.state === "queued" ? "Not started" : "In progress";
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

function ReplayView({
  replay,
  renderedAtMs,
}: {
  replay: LiveReplay;
  renderedAtMs: number;
}) {
  // Starts from the server render time so hydration matches, then ticks so an
  // open run turns stale when no new records arrive.
  const [nowMs, setNowMs] = useState(renderedAtMs);
  const outcome = runOutcome(replay.run);
  const freshness = feedFreshness({
    lastReceivedAtMs: replay.receivedAtMs,
    nowMs,
    outcome,
  });
  useEffect(() => {
    if (freshness === "closed") return;
    const timer = setInterval(
      () => setNowMs(Date.now()),
      FRESHNESS_CHECK_INTERVAL_MS,
    );
    return () => clearInterval(timer);
  }, [freshness]);
  const { view, run } = replay;
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
        <StatusTile label="Feed" detail={freshnessText[freshness].detail}>
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
            view.lastObservedAt ? (
              <>
                Latest producer observation{" "}
                <Timestamp iso={view.lastObservedAt} />
              </>
            ) : undefined
          }
        >
          <Timestamp iso={new Date(replay.receivedAtMs).toISOString()} />
        </StatusTile>
        <StatusTile
          label="Steps"
          detail={`${view.steps.length - active.length} finished of ${view.steps.length} reported`}
        >
          {active.length} active · {failed.length} failed or cancelled
        </StatusTile>
      </div>
      {(view.gaps.length > 0 || view.unreadableRecords > 0) && (
        <div
          role="status"
          className="space-y-1 rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-xs text-amber-200"
        >
          {view.gaps.map((gap) => (
            <p key={gap.stepId} className="break-words">
              Step {gap.stepId} is waiting for revision {gap.missingSequence};{" "}
              {gap.pendingRevisions} later revision
              {gap.pendingRevisions === 1 ? " is" : "s are"} held as pending and
              not shown.
            </p>
          ))}
          {view.unreadableRecords > 0 && (
            <p>
              {view.unreadableRecords} record
              {view.unreadableRecords === 1 ? "" : "s"} could not be read and
              {view.unreadableRecords === 1 ? " was" : " were"} not applied.
            </p>
          )}
        </div>
      )}
      <Totals replay={replay} />
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

export function LiveRun({ data }: { data: LivePageData }) {
  if (data.replay)
    return (
      <div className="space-y-6">
        <ReplayBanner replay={data.replay} />
        <ScenarioPicker scenarios={data.scenarios} current={data.replay.id} />
        <ReplayView replay={data.replay} renderedAtMs={data.renderedAtMs} />
      </div>
    );
  return (
    <div className="space-y-6">
      <Card className="items-center gap-3 border-dashed p-8 text-center sm:p-12">
        <Unplug className="size-7 text-muted-foreground" aria-hidden />
        <h2 className="text-base font-semibold">Live feed not connected</h2>
        <p className="max-w-lg text-sm leading-6 text-muted-foreground">
          {data.requestedReplay
            ? `Replay scenario “${data.requestedReplay}” does not exist. Choose one below.`
            : "The dashboard does not read live runs yet. Completed runs remain available in Run history."}
        </p>
      </Card>
      <ScenarioPicker scenarios={data.scenarios} />
    </div>
  );
}
