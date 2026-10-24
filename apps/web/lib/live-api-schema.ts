import { z } from "zod";
import type {
  Gap,
  ResourceSampleSnapshot,
  RunPage,
  RunSnapshot,
  RunSummary,
  SavedRecord,
  StepEventSnapshot,
  StepView,
} from "@agentwatch/sdk";

// Runtime validation of B01 live read-API responses (docs/live-ingestion.md). The SDK types
// describe the wire format only, so every response is parsed here before use. Objects are
// loose: the service may add fields compatibly, and a JSON export must keep them as received.

const states = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
] as const;
const runState = z.enum(states);
const timestamp = z.iso.datetime({ offset: true });
/** Cursors and sequences are PostgreSQL bigints; JavaScript reads them exactly only up to this. */
export const MAX_LIVE_CURSOR = Number.MAX_SAFE_INTEGER;
const sequence = z.number().int().nonnegative().max(MAX_LIVE_CURSOR);
const tick = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const seconds = z.number().finite().nullable();
const resourceClass = z.enum(["cpu_sandbox", "inference", "retrieval"]);

const recordBase = {
  contract_version: z.literal("live-contracts-v1"),
  run_id: z.string(),
  task_id: z.string(),
  observed_at: timestamp,
  clock_id: z.string(),
};

export const stepEventSnapshotSchema = z.looseObject({
  ...recordBase,
  record_type: z.literal("step_event"),
  event_id: z.string(),
  step_id: z.string().min(1),
  step_type: z.enum(["model_call", "tool", "retrieval", "other"]),
  parent_step_id: z.string().nullable(),
  trace_id: z.string(),
  span_id: z.string(),
  resource_id: z.string().nullable(),
  service_id: z.string().nullable(),
  sequence,
  state: runState,
  submitted_tick_ns: tick,
  started_tick_ns: tick.nullable(),
  finished_tick_ns: tick.nullable(),
  cpu_seconds: z.number().finite().nonnegative().nullable(),
  cpu_scope: z.enum(["process", "process_and_children"]).nullable(),
  reason: z.string().nullable(),
}) satisfies z.ZodType<StepEventSnapshot>;

export const resourceSampleSnapshotSchema = z.looseObject({
  ...recordBase,
  record_type: z.literal("resource_sample"),
  sample_id: z.string(),
  resource_id: z.string(),
  service_id: z.string(),
  resource_class: resourceClass,
  tick_ns: tick,
  // Unfamiliar metric names are retained evidence, not rejected (docs/live-ingestion.md).
  metric: z.string(),
  unit: z.string(),
  value: z.number().finite().nullable(),
  availability: z.enum([
    "measured",
    "missing",
    "unsupported",
    "malformed",
    "unavailable",
  ]),
  reason: z.string().nullable(),
}) satisfies z.ZodType<ResourceSampleSnapshot>;

export const savedRecordSchema = z.looseObject({
  record_id: z.string(),
  disposition: z.enum(["pending", "applied", "expired"]),
  sequence,
  payload: z.discriminatedUnion("record_type", [
    stepEventSnapshotSchema,
    resourceSampleSnapshotSchema,
  ]),
  received_at: timestamp,
}) satisfies z.ZodType<SavedRecord>;

export const gapSchema = z.looseObject({
  step_id: z.string(),
  missing_sequence: sequence,
  first_pending_sequence: sequence,
  expired: z.boolean(),
}) satisfies z.ZodType<Gap>;

const stepViewSchema = z.looseObject({
  event: stepEventSnapshotSchema,
  timings: z.looseObject({
    queue_seconds: seconds,
    execution_seconds: seconds,
    duration_seconds: seconds,
    cpu_seconds: seconds,
    queue_fraction: seconds,
  }),
}) satisfies z.ZodType<StepView>;

const registrationSchema = z.looseObject({
  contract_version: z.literal("live-contracts-v1"),
  request_id: z.string(),
  task_id: z.string(),
  configuration_digest: z.string(),
  clock_ids: z.array(z.string()),
  resources: z.array(
    z.looseObject({
      resource_id: z.string(),
      service_id: z.string(),
      resource_class: resourceClass,
    }),
  ),
});

const runSummaryFields = {
  run_id: z.uuid(),
  registration: registrationSchema,
  state: runState,
  revision: z.number().int().nonnegative(),
  cleanup_requested: z.boolean(),
  created_at: timestamp,
  updated_at: timestamp,
  sequence,
};

export const runSummarySchema = z.looseObject(
  runSummaryFields,
) satisfies z.ZodType<RunSummary>;

export const runPageSchema = z.looseObject({
  runs: z.array(runSummarySchema),
  next_cursor: sequence.nullable(),
}) satisfies z.ZodType<RunPage>;

export const runSnapshotSchema = z.looseObject({
  ...runSummaryFields,
  request_id: z.string(),
  reason: z.string().nullable(),
  cleanup_confirmed: z.null(),
  unfinished_steps: z.array(z.string()),
  steps: z.array(stepViewSchema),
  changes: z.array(
    z.looseObject({
      payload: z.looseObject({
        change_id: z.string(),
        expected_revision: z.number().int().nonnegative(),
        state: runState,
        reason: z.string().nullable(),
      }),
      revision: z.number().int().nonnegative(),
    }),
  ),
  gaps: z.array(gapSchema),
  records: z.array(savedRecordSchema),
  next_cursor: sequence.nullable(),
}) satisfies z.ZodType<RunSnapshot>;

export type LiveRunPage = z.infer<typeof runPageSchema>;
export type LiveRunSnapshot = z.infer<typeof runSnapshotSchema>;
export type LiveSavedRecord = z.infer<typeof savedRecordSchema>;
export type LiveGap = z.infer<typeof gapSchema>;
export type LiveStepView = z.infer<typeof stepViewSchema>;
