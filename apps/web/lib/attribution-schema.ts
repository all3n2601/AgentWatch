import { z } from "zod";
import type {
  AttributionItem,
  AttributionRecord,
  ModelIdentityView,
  ModelPredictionView,
  ModelStatusView,
  RulesVerdictView,
} from "@agentwatch/sdk";
import { MAX_LIVE_CURSOR } from "./live-api-schema";

// Runtime validation of B03 attribution responses (api/openapi/attribution.json). The page
// envelope is validated as a whole, since its cursor and model status drive polling. Each row
// is validated on its own (lib/attribution-view.ts), so one malformed row is shown as malformed
// while the others still render. Objects are loose: the service may add fields compatibly.

const sequence = z.number().int().nonnegative().max(MAX_LIVE_CURSOR);
const timestamp = z.iso.datetime({ offset: true });
const finite = z.number().finite();
const fraction = finite.min(0).max(1);
const seconds = finite.min(0).nullable();

export const MODEL_STATES = [
  "not_configured",
  "not_promoted_refused",
  "ready_not_promoted",
  "ready_promoted",
  "load_failed",
] as const;
export type ModelState = (typeof MODEL_STATES)[number];

export const MODEL_OUTCOMES = [
  "predicted",
  "abstained",
  "not_scored_step_state",
  "unavailable",
] as const;
export type ModelOutcome = (typeof MODEL_OUTCOMES)[number];

export const TELEMETRY_STATUSES = [
  "present",
  "empty",
  "missing",
  "no_overlap",
  "unaligned",
] as const;
export type TelemetryStatus = (typeof TELEMETRY_STATUSES)[number];

/** The only feature definition this view reads; another version is shown as malformed. */
export const FEATURE_VERSION = "step-features-v1";

export const modelIdentitySchema = z.looseObject({
  training_version: z.string(),
  calibration_version: z.string(),
  classifier_sha256: z.string(),
  regressor_sha256: z.string(),
  preprocessing_sha256: z.string(),
  calibration_sha256: z.string(),
  temperature: finite,
  abstention_threshold: fraction,
  class_order: z.array(z.string()),
  blocks_promotion: z.boolean(),
}) satisfies z.ZodType<ModelIdentityView>;

export const modelStatusSchema = z.looseObject({
  state: z.enum(MODEL_STATES),
  reason: z.string().nullable(),
  identity: modelIdentitySchema.nullable(),
  not_promoted: z.boolean().nullable(),
  serving_key: z.string().min(1),
}) satisfies z.ZodType<ModelStatusView>;

const rulesVerdictSchema = z.looseObject({
  version: z.string().min(1),
  verdict: z.string(),
}) satisfies z.ZodType<RulesVerdictView>;

const modelPredictionSchema = z.looseObject({
  state: z.enum(MODEL_STATES),
  reason: z.string().nullable(),
  training_version: z.string().nullable(),
  calibration_version: z.string().nullable(),
  not_promoted: z.boolean().nullable(),
  outcome: z.enum(MODEL_OUTCOMES),
  predicted_class: z.string().nullable(),
  confidence: fraction.nullable(),
  abstention_threshold: fraction.nullable(),
  abstained: z.boolean().nullable(),
  calibrated_probabilities: z.record(z.string(), fraction).nullable(),
  excess_fraction_estimate: fraction.nullable(),
}) satisfies z.ZodType<ModelPredictionView>;

/**
 * The web app's copy of a `step-features-v1` FeatureRow (packages/features, `live.py`). The
 * API leaves `feature_row` untyped; ranges follow the Python model so an out-of-range value is
 * reported as malformed rather than displayed.
 */
export const featureRowSchema = z.looseObject({
  run_id: z.uuid(),
  step_id: z.string().min(1),
  feature_version: z.literal(FEATURE_VERSION),
  timings: z.looseObject({
    queue_seconds: seconds,
    execution_seconds: seconds,
    duration_seconds: seconds,
    cpu_seconds: seconds,
  }),
  telemetry: z.looseObject({
    status: z.enum(TELEMETRY_STATUSES),
    sample_count: z.number().int().nonnegative(),
    coverage: fraction.nullable(),
    declared_interval_seconds: finite.positive().nullable(),
    observed_interval_seconds: finite.positive().nullable(),
  }),
  features: z.looseObject({
    queue_seconds: seconds,
    execution_seconds: seconds,
    duration_seconds: seconds,
    cpu_seconds: seconds,
    queue_fraction: fraction.nullable(),
    cpu_pressure_max_avg10: finite.min(0).max(100).nullable(),
    cpu_pressure_available: z.boolean(),
    cgroup_throttled_usec_delta: finite.min(0).nullable(),
    cgroup_available: z.boolean(),
    telemetry_sample_count: z.number().int().nonnegative(),
    telemetry_coverage: fraction.nullable(),
  }),
});
export type FeatureRow = z.infer<typeof featureRowSchema>;

export const attributionRecordSchema = z.looseObject({
  run_id: z.uuid(),
  step_id: z.string().min(1),
  step_state: z.enum(["completed", "failed", "cancelled"]),
  feature_version: z.string().min(1),
  feature_row: z.record(z.string(), z.unknown()),
  missing_inputs: z.array(z.string()),
  telemetry_status: z.enum(TELEMETRY_STATUSES),
  rules: rulesVerdictSchema,
  model: modelPredictionSchema,
}) satisfies z.ZodType<AttributionRecord>;
export type AttributionRecordData = z.infer<typeof attributionRecordSchema>;

export const attributionItemSchema = z.looseObject({
  sequence,
  serving_key: z.string().min(1),
  committed_at: timestamp,
  step_terminal_received_at: timestamp,
  evidence_changed: z.boolean(),
  record: attributionRecordSchema,
}) satisfies z.ZodType<AttributionItem>;
export type AttributionItemData = z.infer<typeof attributionItemSchema>;

/**
 * A page as received. Only `sequence` is required of each row here: the cursor must advance
 * past every row, valid or not, and the rest is checked per row.
 */
export const attributionPageSchema = z.looseObject({
  run_id: z.uuid(),
  model: modelStatusSchema,
  attributions: z.array(z.looseObject({ sequence })),
  next_cursor: sequence.nullable(),
});
export type AttributionPageData = z.infer<typeof attributionPageSchema>;
export type ModelStatus = z.infer<typeof modelStatusSchema>;
export type ReceivedItem = AttributionPageData["attributions"][number];
