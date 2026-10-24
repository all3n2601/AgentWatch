import type {
  LiveRunSnapshot,
  LiveSavedRecord,
  LiveStepView,
} from "./live-api-schema";
import type { RunStatus } from "./live-status";
import {
  terminalStates,
  type LiveRunView,
  type LiveStep,
  type SequenceGap,
  type StepTimings,
} from "./step-events";

// Reads a B01 run snapshot into the live view. The service's `steps` (validated contiguous
// prefixes) and `gaps` are authoritative: pending and expired records are never applied here,
// and an expired revision stays a gap even after the revisions before it arrive.

function timingsOf(view: LiveStepView): StepTimings | null {
  const { timings, event } = view;
  if (
    !terminalStates.includes(event.state) ||
    timings.duration_seconds === null
  )
    return null;
  return {
    queueSeconds: timings.queue_seconds,
    executionSeconds: timings.execution_seconds,
    durationSeconds: timings.duration_seconds,
  };
}

const latest = (a: string | null, b: string) =>
  a === null || Date.parse(b) > Date.parse(a) ? b : a;

/**
 * Builds the view of one run from its latest snapshot and every record received for it.
 * Records only supply producer start times and pending-revision counts.
 */
export function readRunSnapshot(
  snapshot: Pick<LiveRunSnapshot, "run_id" | "steps" | "gaps">,
  records: readonly LiveSavedRecord[],
): LiveRunView {
  const runId = snapshot.run_id;
  const ownRecords = records.filter((r) => r.payload.run_id === runId);
  const stepRecords = ownRecords.flatMap((r) =>
    r.payload.record_type === "step_event"
      ? [{ disposition: r.disposition, event: r.payload }]
      : [],
  );
  const appliedStart = new Map<string, string>();
  for (const { disposition, event } of stepRecords)
    if (disposition === "applied" && event.state === "running")
      appliedStart.set(event.step_id, event.observed_at);

  const ordered = [...snapshot.steps].sort(
    (a, b) =>
      a.event.clock_id.localeCompare(b.event.clock_id) ||
      a.event.submitted_tick_ns - b.event.submitted_tick_ns ||
      a.event.step_id.localeCompare(b.event.step_id),
  );
  let lastObservedAt: string | null = null;
  const steps: LiveStep[] = ordered.map((view) => {
    const { event } = view;
    lastObservedAt = latest(lastObservedAt, event.observed_at);
    return {
      stepId: event.step_id,
      stepType: event.step_type,
      parentStepId: event.parent_step_id,
      state: event.state,
      appliedSequence: event.sequence,
      clockId: event.clock_id,
      started: event.started_tick_ns !== null,
      startedObservedAt: appliedStart.get(event.step_id) ?? null,
      lastObservedAt: event.observed_at,
      timings: timingsOf(view),
      cpuSeconds: event.cpu_seconds,
      cpuScope: event.cpu_scope,
      reason: event.reason,
    };
  });

  const gaps: SequenceGap[] = snapshot.gaps.map((gap) => ({
    stepId: gap.step_id,
    missingSequence: gap.missing_sequence,
    firstPendingSequence: gap.first_pending_sequence,
    expired: gap.expired,
    pendingRevisions: stepRecords.filter(
      ({ disposition, event }) =>
        event.step_id === gap.step_id && disposition !== "applied",
    ).length,
  }));

  return {
    runId,
    steps,
    gaps,
    unreadableRecords: 0,
    otherRunRecords: records.length - ownRecords.length,
    lastObservedAt,
  };
}

export const runStatusOf = (
  snapshot: Pick<LiveRunSnapshot, "state" | "reason">,
): RunStatus => ({ state: snapshot.state, reason: snapshot.reason });
