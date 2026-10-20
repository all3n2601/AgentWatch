// Synthetic, scripted StepEvent records for tests and the development replay.
// None of these values were measured; they must never be shown as recorded evidence.

import type { RunStatus } from "../lib/live-status";

export const SYNTHETIC_RUN_ID = "00000000-0000-4000-8000-00000000f001";
const NS_PER_SECOND = 1e9;

type SyntheticEventInput = {
  step_id: string;
  sequence: number;
  state: "queued" | "running" | "completed" | "failed" | "cancelled";
  observed_at: string;
  submitted_s?: number;
  started_s?: number | null;
  finished_s?: number | null;
  step_type?: "model_call" | "tool" | "retrieval" | "other";
  cpu_seconds?: number | null;
  cpu_scope?: "process" | "process_and_children" | null;
  reason?: string | null;
  clock_id?: string;
  run_id?: string;
};

const seconds = (value: number | null | undefined) =>
  value === null || value === undefined
    ? null
    : Math.round(value * NS_PER_SECOND);

/** Builds one wire-format StepEvent with tick fields given in seconds for readability. */
export function syntheticStepEvent(input: SyntheticEventInput) {
  return {
    contract_version: "live-contracts-v1",
    record_type: "step_event",
    run_id: input.run_id ?? SYNTHETIC_RUN_ID,
    task_id: "synthetic-replay-task-v1",
    event_id: `${input.run_id ?? SYNTHETIC_RUN_ID}:${input.step_id}:${input.sequence}`,
    step_id: input.step_id,
    step_type: input.step_type ?? "tool",
    parent_step_id: null,
    trace_id: "0000000000000000000000000000f001",
    span_id: "000000000000f001",
    resource_id: "synthetic-worker-1",
    service_id: "synthetic-sandbox-1",
    sequence: input.sequence,
    state: input.state,
    observed_at: input.observed_at,
    clock_id: input.clock_id ?? "synthetic-process-1",
    submitted_tick_ns: seconds(input.submitted_s ?? 0),
    started_tick_ns: seconds(input.started_s),
    finished_tick_ns: seconds(input.finished_s),
    cpu_seconds: input.cpu_seconds ?? null,
    cpu_scope: input.cpu_scope ?? null,
    reason: input.reason ?? null,
  };
}

export type ReplayScenario = {
  id: string;
  title: string;
  description: string;
  run: RunStatus;
  /**
   * How long before page load the replayed records are treated as received;
   * null when no record has been received.
   */
  receivedAgoMs: number | null;
  records: unknown[];
};

const REPLAY_EPOCH_MS = Date.parse("2026-10-14T12:00:00Z");
const at = (second: number) =>
  new Date(REPLAY_EPOCH_MS + second * 1000).toISOString();

function completedStep(
  step_id: string,
  submitted_s: number,
  started_s: number,
  finished_s: number,
  {
    step_type,
    cpu_seconds = 0.4,
    cpu_scope = cpu_seconds === null ? null : "process",
  }: Pick<SyntheticEventInput, "step_type" | "cpu_seconds" | "cpu_scope"> = {},
) {
  // Nonterminal revisions carry null CPU per the contract.
  return [
    syntheticStepEvent({
      step_id,
      sequence: 0,
      state: "queued",
      observed_at: at(submitted_s),
      submitted_s,
      step_type,
    }),
    syntheticStepEvent({
      step_id,
      sequence: 1,
      state: "running",
      observed_at: at(started_s),
      submitted_s,
      started_s,
      step_type,
    }),
    syntheticStepEvent({
      step_id,
      sequence: 2,
      state: "completed",
      observed_at: at(finished_s),
      submitted_s,
      started_s,
      finished_s,
      step_type,
      cpu_seconds,
      cpu_scope,
    }),
  ];
}

const running = { state: "running", reason: null } as const;
const completed = { state: "completed", reason: null } as const;

export const replayScenarios: ReplayScenario[] = [
  {
    id: "active-steps",
    title: "Active steps",
    description: "One step finished, one running and one still queued.",
    run: running,
    receivedAgoMs: 0,
    records: [
      ...completedStep("plan", 0, 0.1, 1.2, { step_type: "model_call" }),
      ...completedStep("run-tests", 1.3, 1.5, 3.5).slice(0, 2),
      syntheticStepEvent({
        step_id: "summarize",
        sequence: 0,
        state: "queued",
        observed_at: at(4),
        submitted_s: 3.6,
        step_type: "model_call",
      }),
    ],
  },
  {
    id: "completed-run",
    title: "Completed run",
    description:
      "Every step finished and the run is closed, so totals are final.",
    run: completed,
    receivedAgoMs: 0,
    records: [
      ...completedStep("plan", 0, 0.1, 1.2, { step_type: "model_call" }),
      ...completedStep("run-tests", 1.3, 1.5, 3.5),
      ...completedStep("summarize", 3.6, 3.7, 4.4, { step_type: "model_call" }),
    ],
  },
  {
    id: "failed-run",
    title: "Failed run",
    description: "A tool step timed out and the run closed as failed.",
    run: { state: "failed", reason: "Tool step run-tests failed" },
    receivedAgoMs: 0,
    records: [
      ...completedStep("plan", 0, 0.1, 1.2, { step_type: "model_call" }),
      ...completedStep("run-tests", 1.3, 1.5, 3.5).slice(0, 2),
      syntheticStepEvent({
        step_id: "run-tests",
        sequence: 2,
        state: "failed",
        observed_at: at(6),
        submitted_s: 1.3,
        started_s: 1.5,
        finished_s: 5.5,
        reason: "Tool timed out; cleanup status is recorded separately",
      }),
    ],
  },
  {
    id: "cancelled-before-start",
    title: "Cancelled before start",
    description:
      "A queued step was cancelled before any worker started it; it has no execution time or CPU.",
    run: { state: "cancelled", reason: "Operator cancelled the run" },
    receivedAgoMs: 0,
    records: [
      ...completedStep("plan", 0, 0.1, 1.2, { step_type: "model_call" }),
      syntheticStepEvent({
        step_id: "run-tests",
        sequence: 0,
        state: "queued",
        observed_at: at(1),
        submitted_s: 1.3,
      }),
      syntheticStepEvent({
        step_id: "run-tests",
        sequence: 1,
        state: "cancelled",
        observed_at: at(2),
        submitted_s: 1.3,
        finished_s: 2.1,
        reason: "Cancelled before a worker started",
      }),
    ],
  },
  {
    id: "out-of-order",
    title: "Out-of-order delivery",
    description:
      "A completion arrived before its running revision; it is held as pending, not shown as completed.",
    run: running,
    receivedAgoMs: 0,
    records: [
      ...completedStep("plan", 0, 0.1, 1.2, { step_type: "model_call" }),
      ...completedStep("run-tests", 1.3, 1.5, 3.5).filter(
        (_, index) => index !== 1,
      ),
    ],
  },
  {
    id: "no-records",
    title: "No records yet",
    description:
      "The run is registered and queued, but no step record has been received.",
    run: { state: "queued", reason: null },
    receivedAgoMs: null,
    records: [],
  },
  {
    id: "stale-feed",
    title: "Stale feed",
    description:
      "The run is still open but no new records have arrived for two minutes.",
    run: running,
    receivedAgoMs: 120_000,
    records: completedStep("run-tests", 0, 0.2, 2).slice(0, 2),
  },
  {
    id: "missing-cpu",
    title: "Missing CPU measurement",
    description:
      "The run completed but one step has no CPU measurement, so CPU time is not totalled.",
    run: completed,
    receivedAgoMs: 0,
    records: [
      ...completedStep("plan", 0, 0.1, 1.2, { step_type: "model_call" }),
      ...completedStep("run-tests", 1.3, 1.5, 3.5, { cpu_seconds: null }),
    ],
  },
  {
    id: "zero-duration",
    title: "Zero-duration step",
    description:
      "A step finished in the same tick it started: a measured zero.",
    run: completed,
    receivedAgoMs: 0,
    records: completedStep("cache-hit", 0, 0, 0, {
      step_type: "retrieval",
      cpu_seconds: 0,
    }),
  },
];

export const findReplayScenario = (id: string | undefined) =>
  replayScenarios.find((scenario) => scenario.id === id);
