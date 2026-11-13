"use client";
import { useEffect, useState } from "react";
import {
  CalendarClock,
  CircleAlert,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  Clock3,
  Cpu,
  GitBranch,
  Layers,
  ListOrdered,
  LoaderCircle,
  Play,
  RadioTower,
  Scale,
  Tag,
  Timer,
  type LucideIcon,
} from "lucide-react";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge, TONE_TEXT } from "@/components/ui/status-badge";
import {
  KeyValueList,
  type KeyValueItem,
} from "@/components/widgets/evidence-lists";
import { startPolling } from "@/components/use-live-feed";
import { duration } from "@/lib/dashboard-data";
import { createPolicyBrowserApi, type PolicyResult } from "@/lib/policy-api";
import type { PolicyControls, RunPolicyView } from "@/lib/policy-api-schema";
import {
  controlsConsistent,
  currentPolicyText,
  INCONSISTENT_POLICY_RETRY_MS,
  nextRunPolicyReadMs,
  runAppliedLimitsText,
  runPolicyText,
} from "@/lib/policy-view";
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
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

const FRESHNESS_CHECK_INTERVAL_MS = 5_000;
const policyApi = createPolicyBrowserApi();

export const STEP_STATES: Record<
  StepState,
  { label: string; icon: LucideIcon; tone: Tone }
> = {
  queued: { label: "Queued", icon: CircleDashed, tone: "info" },
  running: { label: "Running", icon: LoaderCircle, tone: "info" },
  completed: { label: "Completed", icon: CircleCheck, tone: "success" },
  failed: { label: "Failed", icon: CircleAlert, tone: "danger" },
  cancelled: { label: "Cancelled", icon: CircleSlash, tone: "warning" },
};

const freshnessText: Record<
  FeedFreshness,
  { label: string; detail: string; tone: Tone }
> = {
  fresh: {
    label: "Receiving",
    detail: "Records arrived recently",
    tone: "success",
  },
  stale: {
    label: "Stale",
    detail: "No new records recently; the run may still be open",
    tone: "warning",
  },
  closed: {
    label: "Closed",
    detail: "Run reached a terminal state",
    tone: "info",
  },
  never: {
    label: "No records",
    detail: "Nothing has been received yet",
    tone: "info",
  },
};

export function StepStateBadge({
  state,
  appearance = "soft",
  className,
}: {
  state: StepState;
  appearance?: "soft" | "dot";
  className?: string;
}) {
  const style = STEP_STATES[state];
  return (
    <StatusBadge
      label={style.label}
      tone={style.tone}
      icon={style.icon}
      appearance={appearance}
      className={className}
    />
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
  icon: Icon,
  label,
  children,
  detail,
  tone,
  announce = false,
}: {
  icon: LucideIcon;
  label: string;
  children: React.ReactNode;
  detail?: React.ReactNode;
  tone?: Tone;
  /** Announce value changes to screen readers (polite live region). */
  announce?: boolean;
}) {
  return (
    <InsetCard className="flex min-w-0 gap-3">
      <Icon
        className={cn(
          "mt-0.5 size-5 shrink-0",
          tone ? TONE_TEXT[tone] : "text-ink",
        )}
        aria-hidden
      />
      <div className="min-w-0">
        <p className="text-sm text-ink-muted">{label}</p>
        <div
          className="mt-0.5 flex flex-wrap items-center gap-2 text-[15px] font-semibold text-ink"
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
          <p className="mt-1 text-xs break-words text-ink-muted">{detail}</p>
        )}
      </div>
    </InsetCard>
  );
}

const stepCountText = (total: { measuredSteps: number }) =>
  `${total.measuredSteps} step${total.measuredSteps === 1 ? "" : "s"}`;

function SoFar({ show }: { show: boolean }) {
  return show ? (
    <span className="ml-1.5 text-xs font-normal text-warning">so far</span>
  ) : null;
}

function TotalValue({
  label,
  value,
  detail,
}: {
  label: string;
  value: React.ReactNode;
  detail: React.ReactNode;
}) {
  return (
    <div className="min-w-0">
      <dt className="text-sm text-ink-muted">{label}</dt>
      <dd className="mt-0.5 text-xl font-semibold tracking-tight text-ink tabular-nums">
        {value}
      </dd>
      <dd className="mt-0.5 text-xs text-ink-muted">{detail}</dd>
    </div>
  );
}

function timingValue(total: TimingTotal, final: boolean) {
  return (
    <>
      {total.measuredSteps
        ? duration(total.seconds)
        : total.unmeasuredSteps
          ? "Not measured"
          : "No finished steps"}
      <SoFar show={!final && total.measuredSteps > 0} />
    </>
  );
}

const timingDetail = (total: TimingTotal) =>
  `${stepCountText(total)}${total.unmeasuredSteps > 0 ? ` · excludes ${total.unmeasuredSteps} that never started` : ""}`;

function spanValue(total: SpanTotal, final: boolean) {
  return (
    <>
      {total.measuredSteps === 0
        ? "No finished steps"
        : total.seconds === null
          ? "Not combined"
          : duration(total.seconds)}
      <SoFar
        show={!final && total.seconds !== null && total.measuredSteps > 0}
      />
    </>
  );
}

const spanDetail = (total: SpanTotal) =>
  total.seconds === null && total.measuredSteps > 0
    ? `Steps use ${total.clockDomains} clock domains, which are not compared`
    : `${stepCountText(total)} · overlapping parent and child windows count once`;

function cpuValue(total: CpuTotal, final: boolean) {
  const steps = total.measuredSteps + total.unmeasuredSteps;
  return (
    <>
      {total.seconds !== null
        ? duration(total.seconds)
        : steps
          ? "Not measured"
          : "No finished steps"}
      <SoFar show={!final && total.seconds !== null} />
    </>
  );
}

function cpuDetail(total: CpuTotal) {
  const steps = total.measuredSteps + total.unmeasuredSteps;
  if (steps === 0) return "Waiting for a finished step";
  if (total.seconds === null)
    return `${total.measuredSteps} of ${steps} finished steps measured${total.unmeasuredSteps === 0 && steps > 0 ? " in different scopes" : ""}`;
  return `Scope: ${total.scope === "process" ? "process" : "process and children"}`;
}

function Totals({ view, run }: { view: LiveRunView; run: RunStatus }) {
  const totals = stepTotals(view, run);
  return (
    <InsetCard>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[15px] font-semibold text-ink">
          {totals.final ? "Final step totals" : "Partial step totals"}
        </h3>
        <StatusBadge
          label={totals.final ? "Final" : "Incomplete"}
          tone={totals.final ? "success" : "warning"}
        />
      </div>
      {!totals.final && (
        <ul className="mb-3 list-disc space-y-0.5 pl-5 text-sm text-warning">
          {totals.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
      <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 @3xl:grid-cols-4">
        <TotalValue
          label="Step wall-time span"
          value={spanValue(totals.duration, totals.final)}
          detail={spanDetail(totals.duration)}
        />
        <TotalValue
          label="Queue delay"
          value={timingValue(totals.queue, totals.final)}
          detail={timingDetail(totals.queue)}
        />
        <TotalValue
          label="Execution wall time"
          value={timingValue(totals.execution, totals.final)}
          detail={timingDetail(totals.execution)}
        />
        <TotalValue
          label="Process CPU time"
          value={cpuValue(totals.cpu, totals.final)}
          detail={cpuDetail(totals.cpu)}
        />
      </dl>
      {totals.parentSteps > 0 && (
        <p className="mt-3 text-xs text-ink-muted">
          Queue delay, execution and CPU time are summed over steps without
          children; {totals.parentSteps} parent step
          {totals.parentSteps === 1 ? " is" : "s are"} left out because its
          window contains its children&apos;s. Concurrent steps still add up.
        </p>
      )}
    </InsetCard>
  );
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

/** Sequence gaps and records for other runs; nothing is shown when there are none. */
export function LiveGaps({ view }: { view: LiveRunView }) {
  if (view.gaps.length === 0 && view.otherRunRecords === 0) return null;
  return (
    <div
      role="status"
      className="space-y-1 rounded-2xl border border-warning-line bg-warning-soft px-4 py-3 text-sm text-warning"
    >
      {view.gaps.map((gap) => (
        <p key={gap.stepId} className="break-words">
          {gapText(gap)}
        </p>
      ))}
      {view.otherRunRecords > 0 && (
        <p>
          {view.otherRunRecords} record
          {view.otherRunRecords === 1 ? " belongs" : "s belong"} to another run
          and {view.otherRunRecords === 1 ? "was" : "were"} not applied.
        </p>
      )}
    </div>
  );
}

/** Failed and cancelled steps with their reasons, so a failure is visible without selecting it. */
function FailedSteps({ steps }: { steps: LiveStep[] }) {
  if (!steps.length) return null;
  return (
    <InsetCard>
      <h3 className="mb-2 text-[15px] font-semibold text-ink">
        Failed or cancelled steps
      </h3>
      <ul className="divide-y divide-line">
        {steps.map((step) => (
          <li
            key={step.stepId}
            className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 text-sm"
          >
            <span className="font-mono text-ink [overflow-wrap:anywhere]">
              {step.stepId}
            </span>
            <StepStateBadge state={step.state} appearance="dot" />
            <span className="basis-full text-ink-muted break-words">
              {step.reason ? `Reason: ${step.reason}` : "No reason reported"}
            </span>
          </li>
        ))}
      </ul>
    </InsetCard>
  );
}

function unfinishedText(step: LiveStep) {
  if (step.state === "queued") return "Not started";
  return step.state === "running" ? "In progress" : "Not reported";
}

function stepTiming(
  step: LiveStep,
  field: "queueSeconds" | "executionSeconds",
) {
  if (!step.timings) return unfinishedText(step);
  const value = step.timings[field];
  return value === null ? "Never started" : duration(value);
}

export const stepTypeLabel = (type: string) => {
  const words = type.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
};

/** One live step's applied state and its monotonic-clock timings. */
export function liveStepItems(step: LiveStep): KeyValueItem[] {
  return [
    {
      id: "state",
      icon: STEP_STATES[step.state].icon,
      label: "State",
      value: STEP_STATES[step.state].label,
    },
    {
      id: "type",
      icon: Tag,
      label: "Type",
      value: stepTypeLabel(step.stepType),
    },
    {
      id: "started",
      icon: Play,
      label: "Started (producer time)",
      value: step.startedObservedAt ? (
        <Timestamp iso={step.startedObservedAt} />
      ) : step.started ? (
        "Not received"
      ) : (
        "Not started"
      ),
    },
    {
      id: "queue",
      icon: Layers,
      label: "Queue delay",
      value: stepTiming(step, "queueSeconds"),
    },
    {
      id: "execution",
      icon: Timer,
      label: "Execution",
      value: stepTiming(step, "executionSeconds"),
    },
    {
      id: "duration",
      icon: Clock3,
      label: "Duration",
      value: step.timings
        ? duration(step.timings.durationSeconds)
        : unfinishedText(step),
    },
    {
      id: "cpu",
      icon: Cpu,
      label: "Process CPU",
      value:
        step.cpuSeconds === null
          ? "Not measured"
          : `${duration(step.cpuSeconds)}${step.cpuScope ? ` (${step.cpuScope.replaceAll("_", " ")})` : ""}`,
    },
    {
      id: "parent",
      icon: GitBranch,
      label: "Parent step",
      value: step.parentStepId ?? "None",
    },
    {
      id: "revision",
      icon: ListOrdered,
      label: "Applied revision",
      value: step.appliedSequence,
    },
    {
      id: "observed",
      icon: CalendarClock,
      label: "Last observed",
      value: <Timestamp iso={step.lastObservedAt} />,
    },
  ];
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

type RunPolicyRead = {
  key: string;
  run: PolicyResult<RunPolicyView>;
  current: PolicyResult<PolicyControls>;
  /** Browser wall time of the latest read whose current policy was consistent. */
  currentReadAt: string | null;
  retrying: boolean;
};

/**
 * The policy versions linked to the run, what its runner applied, and the current policy.
 * Read again on the live view's poll loop while the run is open, since a policy change then
 * links a new version and replaces the current one, and once more when it closes.
 */
function RunPolicy({ runId, closed }: { runId: string; closed: boolean }) {
  const key = `${runId}:${closed}`;
  const [policy, setPolicy] = useState<RunPolicyRead | null>(null);
  useEffect(() => {
    let inconsistentReads = 0;
    let currentReadAt: string | null = null;
    const polling = startPolling(async (signal) => {
      const [run, current] = await Promise.all([
        policyApi.runPolicy(runId, signal),
        policyApi.current(signal),
      ]);
      if (signal.aborted) return null;
      const inconsistent = current.ok && !controlsConsistent(current.data);
      inconsistentReads = inconsistent ? inconsistentReads + 1 : 0;
      if (current.ok && !inconsistent) currentReadAt = new Date().toISOString();
      const next = nextRunPolicyReadMs({ closed, inconsistentReads });
      setPolicy({
        key,
        run,
        current,
        currentReadAt,
        retrying: inconsistent && next === INCONSISTENT_POLICY_RETRY_MS,
      });
      return next;
    });
    return () => polling.stop();
  }, [key, runId, closed]);
  // Results for a previously selected run are never shown for this one.
  const shown = policy?.key === key ? policy : null;
  const run = shown?.run ?? null;
  const notApplicable = run?.ok === true && !run.data.applicable;
  const current = shown?.current ?? null;
  const currentConsistent =
    current?.ok === true && controlsConsistent(current.data);
  const items: KeyValueItem[] = [
    {
      id: "linked",
      label: "Policy active while open",
      value: runPolicyText(run),
    },
    {
      id: "applied",
      label: "Limits applied by the runner",
      value: notApplicable ? "Not applicable" : "Not available",
      ...(!notApplicable && {
        status: { label: "Not recorded", tone: "warning" as const },
      }),
    },
    {
      id: "current",
      label: "Current policy",
      value: currentPolicyText(current, { retrying: shown?.retrying ?? false }),
      ...(current?.ok &&
        !currentConsistent && {
          status: { label: "Not confirmed", tone: "warning" as const },
        }),
    },
    {
      id: "current-read-at",
      // A dated snapshot: the policy can change between reads.
      label: currentConsistent
        ? "Current policy read at"
        : "Last confirmed read",
      value: shown?.currentReadAt ? (
        <Timestamp iso={shown.currentReadAt} />
      ) : (
        "Not read yet"
      ),
    },
  ];
  return (
    <InsetCard role="region" aria-labelledby={`run-policy-${runId}`}>
      <h3
        id={`run-policy-${runId}`}
        className="mb-2 flex items-center gap-2 text-[15px] font-semibold text-ink"
      >
        <Scale className="size-4 shrink-0" aria-hidden />
        Concurrency policy
      </h3>
      <KeyValueList items={items} dense className="[&_dd]:break-words" />
      <p className="mt-2 text-sm text-ink-muted">{runAppliedLimitsText(run)}</p>
    </InsetCard>
  );
}

/** Run state, feed freshness, step counts, gaps, totals and policy of one live run. */
export function LiveRunStatus({
  runId,
  run,
  view,
  receipt,
  connection,
}: {
  runId: string;
  run: RunStatus;
  view: LiveRunView;
  /** When the browser last received new records for this run; null before any. */
  receipt: FeedReceipt | null;
  /** The connection banner, shown above the tiles. */
  connection?: React.ReactNode;
}) {
  const outcome = runOutcome(run);
  const receivedAgoMs = receipt?.ageAtAnchorMs ?? null;
  const now = useBrowserClock(feedClockTicks(receivedAgoMs));
  const clock: BrowserClockReading =
    receipt && now
      ? { ...now, anchorMonotonicMs: receipt.anchorMonotonicMs }
      : null;
  const feed = feedDisplay({ receivedAgoMs, outcome, clock });
  const freshness = freshnessText[feed.freshness];
  const active = activeSteps(view);
  const failed = failedSteps(view);
  return (
    <GlassPanel as="section" aria-labelledby="live-run-status">
      <PanelHeader
        titleId="live-run-status"
        icon={RadioTower}
        title="Run status"
        description="As last read from the live API."
      />
      <div className="@container space-y-3">
        {connection}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 @3xl:grid-cols-4">
          <StatusTile
            icon={run ? STEP_STATES[run.state].icon : CircleDashed}
            tone={run ? STEP_STATES[run.state].tone : "info"}
            label="Run state"
            detail={
              run?.reason ?? (run ? undefined : "Not reported by the source")
            }
          >
            {run ? STEP_STATES[run.state].label : "Unknown"}
          </StatusTile>
          <StatusTile
            icon={feed.freshness === "stale" ? Clock3 : RadioTower}
            tone={freshness.tone}
            label="Feed"
            detail={freshness.detail}
            announce
          >
            {freshness.label}
          </StatusTile>
          <StatusTile
            icon={CalendarClock}
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
            icon={ListOrdered}
            tone={failed.length > 0 ? "danger" : undefined}
            label="Steps"
            detail={
              view.steps.length
                ? `${view.steps.length - active.length} finished of ${view.steps.length} reported`
                : "No step has an applied state yet."
            }
          >
            {active.length} active · {failed.length} failed or cancelled
          </StatusTile>
        </div>
        <LiveGaps view={view} />
        <FailedSteps steps={failed} />
        <Totals view={view} run={run} />
        <RunPolicy
          runId={runId}
          closed={outcome !== "active" && outcome !== "unknown"}
        />
      </div>
    </GlassPanel>
  );
}
