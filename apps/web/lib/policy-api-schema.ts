import { z } from "zod";
import type {
  PolicyChangeResult,
  PolicyPage,
  PolicyVersion,
  RunPolicy,
} from "@agentwatch/sdk";
import { MAX_LIVE_CURSOR } from "./live-api-schema";

// Runtime validation of the O02 concurrency policy API (docs/superpowers/specs/
// 2026-11-03-concurrency-policy-design.md). The SDK types describe the wire format only.

/** The service accepts limits from 1 to this many concurrent operations per control. */
export const MAX_POLICY_LIMIT = 1024;
/** `Identifier` and `Reason` lengths in agentwatch_sdk.live. */
export const MAX_OPERATOR_LENGTH = 128;
export const MAX_REASON_LENGTH = 512;

const limit = z.number().int().min(1).max(MAX_POLICY_LIMIT);
const version = z.number().int().nonnegative().max(MAX_LIVE_CURSOR);
const timestamp = z.iso.datetime({ offset: true });
const text = (max: number) =>
  z.string().min(1).max(max).regex(/\S/, "Must not be blank");

/** The only controls; an absent key means "no limit", and null is never a limit. */
const limitsShape = {
  inference_requests: limit.optional(),
  test_jobs: limit.optional(),
};
export const policyLimitsSchema = z.strictObject(limitsShape);

export const policyVersionSchema = z.looseObject({
  target: z.string(),
  version,
  action: z.enum(["baseline", "set", "disable", "restore"]),
  limits: policyLimitsSchema,
  restored_from: version.nullable(),
  operator: z.string().nullable(),
  reason: z.string().nullable(),
  request_id: z.uuid().nullable(),
  created_at: timestamp.nullable(),
  controlled_services: z.record(
    z.string(),
    z.looseObject({
      service_id: z.string(),
      resource_class: z.enum(["cpu_sandbox", "inference", "retrieval"]),
    }),
  ),
  enforcement_scope: z.literal("runner_process"),
}) satisfies z.ZodType<PolicyVersion>;

export const policyPageSchema = z.looseObject({
  versions: z.array(policyVersionSchema),
  next_before: version.nullable(),
}) satisfies z.ZodType<PolicyPage>;

export const policyChangeResultSchema = z.looseObject({
  created: z.boolean(),
  policy: policyVersionSchema,
}) satisfies z.ZodType<PolicyChangeResult>;

export const runPolicySchema = z.looseObject({
  run_id: z.uuid(),
  target: z.string(),
  applicable: z.boolean(),
  versions: z.array(policyVersionSchema),
}) satisfies z.ZodType<RunPolicy>;

/** What the web route returns for the controls screen: the active policy and one history page. */
export const policyControlsSchema = z.object({
  active: policyVersionSchema,
  history: policyPageSchema,
});

/**
 * A change request as the web route accepts it. Shape only: the service checks the
 * action-specific rules (set needs a limit, restore needs a source) and answers 422.
 */
export const policyChangeSchema = z.strictObject({
  request_id: z.uuid(),
  expected_version: version,
  action: z.enum(["set", "disable", "restore"]),
  limits: policyLimitsSchema.optional(),
  restored_from: version.optional(),
  operator: text(MAX_OPERATOR_LENGTH),
  reason: text(MAX_REASON_LENGTH),
});

export type PolicyVersionView = z.infer<typeof policyVersionSchema>;
export type PolicyLimitsView = z.infer<typeof policyLimitsSchema>;
export type PolicyPageView = z.infer<typeof policyPageSchema>;
export type PolicyChangeResultView = z.infer<typeof policyChangeResultSchema>;
export type RunPolicyView = z.infer<typeof runPolicySchema>;
export type PolicyControls = z.infer<typeof policyControlsSchema>;
export type PolicyChangeRequest = z.infer<typeof policyChangeSchema>;
