import { z } from "zod";

// The only module that reads the live-contracts-v1 StepEvent wire format
// (docs/shared-event-contracts.md). Validation is deliberately light: the B01
// service owns identity, conflict, retry and transition rejection. This adapter
// parses records safely, applies each step's contiguous revision prefix and
// reports gaps. The local schema matches the StepEvent type that B01 (#25)
// generates in @agentwatch/sdk. TODO(F02): when F02 connects the live read
// API, type the parsed records with the SDK's StepEvent instead of the local
// inferred type.

export const stepStates = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
] as const;
export type StepState = (typeof stepStates)[number];
export const terminalStates: readonly StepState[] = [
  "completed",
  "failed",
  "cancelled",
];

const NS_PER_SECOND = 1e9;
const tick = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER);
const nullableTick = tick.nullable();

const stepEventSchema = z
  .object({
    contract_version: z.literal("live-contracts-v1"),
    record_type: z.literal("step_event"),
    run_id: z.string().min(1),
    step_id: z.string().min(1),
    step_type: z.enum(["model_call", "tool", "retrieval", "other"]),
    parent_step_id: z.string().nullable(),
    sequence: z.number().int().nonnegative(),
    state: z.enum(stepStates),
    observed_at: z.iso.datetime({ offset: true }),
    clock_id: z.string().min(1),
    submitted_tick_ns: tick,
    started_tick_ns: nullableTick,
    finished_tick_ns: nullableTick,
    cpu_seconds: z.number().finite().nonnegative().nullable(),
    cpu_scope: z.enum(["process", "process_and_children"]).nullable(),
    reason: z.string().nullable(),
  })
  // Only ordering is checked here, so derived durations can never be negative.
  .refine(
    (e) =>
      (e.started_tick_ns === null ||
        e.submitted_tick_ns <= e.started_tick_ns) &&
      (e.finished_tick_ns === null ||
        (e.started_tick_ns ?? e.submitted_tick_ns) <= e.finished_tick_ns),
    { message: "Step ticks are out of order" },
  );
type StepEvent = z.infer<typeof stepEventSchema>;

/** Durations derived from ticks of one clock domain; null when not measured. */
export type StepTimings = {
  queueSeconds: number | null;
  executionSeconds: number | null;
  durationSeconds: number;
};

export type LiveStep = {
  stepId: string;
  stepType: StepEvent["step_type"];
  parentStepId: string | null;
  state: StepState;
  appliedSequence: number;
  clockId: string;
  /** Producer wall time of the applied running revision, if one exists. */
  startedObservedAt: string | null;
  lastObservedAt: string;
  /** Present only for terminal steps; running steps never get a computed duration. */
  timings: StepTimings | null;
  cpuSeconds: number | null;
  cpuScope: StepEvent["cpu_scope"];
  reason: string | null;
};

export type SequenceGap = {
  stepId: string;
  missingSequence: number;
  pendingRevisions: number;
};

export type LiveRunView = {
  runId: string;
  steps: LiveStep[];
  gaps: SequenceGap[];
  unreadableRecords: number;
  otherRunRecords: number;
  lastObservedAt: string | null;
};

const secondsBetween = (fromNs: number, toNs: number) =>
  (toNs - fromNs) / NS_PER_SECOND;

function timingsOf(event: StepEvent): StepTimings | null {
  if (!terminalStates.includes(event.state) || event.finished_tick_ns === null)
    return null;
  const started = event.started_tick_ns;
  return {
    queueSeconds:
      started === null
        ? null
        : secondsBetween(event.submitted_tick_ns, started),
    executionSeconds:
      started === null ? null : secondsBetween(started, event.finished_tick_ns),
    // For a step that never started this is submission to terminal
    // notification, not execution time.
    durationSeconds: secondsBetween(
      event.submitted_tick_ns,
      event.finished_tick_ns,
    ),
  };
}

const latest = (a: string | null, b: string) =>
  a === null || Date.parse(b) > Date.parse(a) ? b : a;

/** Reads raw StepEvent records for one run into the applied step states. */
export function readStepEvents(
  runId: string,
  records: readonly unknown[],
): LiveRunView {
  const revisionsByStep = new Map<string, Map<number, StepEvent>>();
  let unreadableRecords = 0;
  let otherRunRecords = 0;
  for (const record of records) {
    const parsed = stepEventSchema.safeParse(record);
    if (!parsed.success) {
      unreadableRecords += 1;
      continue;
    }
    const event = parsed.data;
    if (event.run_id !== runId) {
      otherRunRecords += 1;
      continue;
    }
    const revisions =
      revisionsByStep.get(event.step_id) ?? new Map<number, StepEvent>();
    // Retries reuse a sequence; the service rejects conflicting content, so the
    // first arrival is kept.
    if (!revisions.has(event.sequence)) revisions.set(event.sequence, event);
    revisionsByStep.set(event.step_id, revisions);
  }

  const steps: LiveStep[] = [];
  const gaps: SequenceGap[] = [];
  let lastObservedAt: string | null = null;
  for (const [stepId, revisions] of revisionsByStep) {
    let next = 0;
    let applied: StepEvent | undefined;
    let startedObservedAt: string | null = null;
    while (revisions.has(next)) {
      applied = revisions.get(next)!;
      if (applied.state === "running") startedObservedAt = applied.observed_at;
      lastObservedAt = latest(lastObservedAt, applied.observed_at);
      next += 1;
    }
    if (revisions.size > next) {
      gaps.push({
        stepId,
        missingSequence: next,
        pendingRevisions: revisions.size - next,
      });
    }
    if (!applied) continue;
    steps.push({
      stepId,
      stepType: applied.step_type,
      parentStepId: applied.parent_step_id,
      state: applied.state,
      appliedSequence: applied.sequence,
      clockId: applied.clock_id,
      startedObservedAt,
      lastObservedAt: applied.observed_at,
      timings: timingsOf(applied),
      cpuSeconds: applied.cpu_seconds,
      cpuScope: applied.cpu_scope,
      reason: applied.reason,
    });
  }
  return {
    runId,
    steps,
    gaps,
    unreadableRecords,
    otherRunRecords,
    lastObservedAt,
  };
}
