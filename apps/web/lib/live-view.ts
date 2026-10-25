import type { StepEventSnapshot } from "@agentwatch/sdk";

// The live view's model of one run's steps. lib/live-snapshot.ts builds it from the B01
// service's validated snapshot; components and totals read only this shape.

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

/** Durations derived from ticks of one clock domain; null when not measured. */
export type StepTimings = {
  queueSeconds: number | null;
  executionSeconds: number | null;
  durationSeconds: number;
};

export type LiveStep = {
  stepId: string;
  stepType: StepEventSnapshot["step_type"];
  parentStepId: string | null;
  state: StepState;
  appliedSequence: number;
  clockId: string;
  /** Monotonic ticks in `clockId`; only comparable with ticks of the same domain. */
  submittedTickNs: number;
  finishedTickNs: number | null;
  /** Whether the applied revision records a start tick. */
  started: boolean;
  /** Producer wall time of the applied running revision, if one was received. */
  startedObservedAt: string | null;
  lastObservedAt: string;
  /** Present only for terminal steps; running steps never get a computed duration. */
  timings: StepTimings | null;
  cpuSeconds: number | null;
  cpuScope: StepEventSnapshot["cpu_scope"];
  reason: string | null;
};

export type SequenceGap = {
  stepId: string;
  /** Next revision the step needs; 0 when it has no applied state. */
  missingSequence: number;
  firstPendingSequence: number;
  /** Held revisions expired unapplied; they are never applied, even if the gap fills. */
  expired: boolean;
  pendingRevisions: number;
};

export type LiveRunView = {
  runId: string;
  steps: LiveStep[];
  gaps: SequenceGap[];
  /** Records attributed to a different run; counted, never applied. */
  otherRunRecords: number;
  lastObservedAt: string | null;
};
