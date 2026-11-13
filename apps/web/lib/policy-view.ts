import type { PolicyFailure, PolicyResult } from "./policy-api";
import {
  MAX_POLICY_LIMIT,
  type PolicyControls,
  type PolicyLimitsView,
  type RunPolicyView,
} from "./policy-api-schema";

// Display text for the concurrency policy. A failed or pending read must never read as
// "No limits": only a policy the service returned can say that.

export function limitsText(limits: PolicyLimitsView) {
  const { inference_requests: inference, test_jobs: tests } = limits;
  if (inference === undefined && tests === undefined) return "No limits";
  return `Inference: ${inference ?? "no limit"} · Tests: ${tests ?? "no limit"}`;
}

/** Limits of a returned version; only the synthesized version 0 is the baseline. */
export function versionLimitsText(policy: {
  version: number;
  limits: PolicyLimitsView;
}) {
  const text = limitsText(policy.limits);
  return policy.version === 0 ? `${text} (baseline)` : text;
}

function failureText(failure: PolicyFailure) {
  if (failure.kind === "not_found") return "Policy not found";
  if (failure.kind === "invalid_response") return "Policy response is invalid";
  return "Policy service unavailable";
}

/** The active policy, or why it is not known; `null` while loading. */
export function policyStatusText(
  result: PolicyResult<{ version: number; limits: PolicyLimitsView }> | null,
) {
  if (result === null) return "Loading policy…";
  if (!result.ok) return failureText(result);
  return `Version ${result.data.version}: ${versionLimitsText(result.data)}`;
}

/** The policy versions active while a live run was open; `null` while loading. */
export function runPolicyText(result: PolicyResult<RunPolicyView> | null) {
  if (result === null) return "Loading policy…";
  if (!result.ok) return failureText(result);
  if (!result.data.applicable) return "Not an agent run";
  if (result.data.versions.length === 0)
    return "No recorded policy (registered before policy tracking)";
  // The caller labels this text, so it lists the versions alone.
  return result.data.versions
    .map((v) => `v${v.version} (${versionLimitsText(v)})`)
    .join(" → ");
}

/**
 * What a run's runner applied. Admission records stay in the runner's local evidence, so for
 * a coding-agent run this is unavailable whatever the links say; other runs admit nothing.
 */
export function runAppliedLimitsText(
  result: PolicyResult<RunPolicyView> | null,
) {
  if (result?.ok && !result.data.applicable)
    return "Not applicable: this run has no agent runner, so nothing was admitted under a policy.";
  return EFFECTIVE_LIMITS_UNAVAILABLE;
}

/** What a change request does, in words, e.g. to say what Retry will resend. */
export function changeText(change: {
  action: "set" | "disable" | "restore";
  expected_version: number;
  limits?: PolicyLimitsView;
  restored_from?: number;
}) {
  const what =
    change.action === "set"
      ? `Set limits to ${limitsText(change.limits ?? {})}`
      : change.action === "disable"
        ? "Disable all limits"
        : `Restore the limits of version ${change.restored_from ?? "unknown"}`;
  return `${what}, against version ${change.expected_version}`;
}

/** Form inputs holding a version's limits, so a `set` keeps controls the operator left alone. */
export function formFromLimits(limits: PolicyLimitsView) {
  return {
    inference: limits.inference_requests?.toString() ?? "",
    tests: limits.test_jobs?.toString() ?? "",
  };
}

const CONTROL_LABELS = {
  inference_requests: "Inference requests",
  test_jobs: "Test jobs",
} as const;

export type LimitChange = {
  control: string;
  active: string;
  proposed: string;
};

/** Controls whose limit differs between the active version and a proposed one. */
export function limitChanges(
  active: PolicyLimitsView,
  proposed: PolicyLimitsView,
): LimitChange[] {
  return (
    Object.keys(CONTROL_LABELS) as (keyof typeof CONTROL_LABELS)[]
  ).flatMap((control) =>
    active[control] === proposed[control]
      ? []
      : [
          {
            control: CONTROL_LABELS[control],
            active: active[control]?.toString() ?? "no limit",
            proposed: proposed[control]?.toString() ?? "no limit",
          },
        ],
  );
}

/**
 * Which of the two 409 causes a conflict is. The service has no machine-readable code yet, so
 * this matches its detail text (policy_store.py); anything else is reported as unknown.
 */
export function conflictCause(
  message: string,
): "stale_version" | "request_reused" | "unknown" {
  if (message.startsWith("Policy changed:")) return "stale_version";
  if (message.startsWith("Policy request reused")) return "request_reused";
  return "unknown";
}

/**
 * Whether one read of the controls is self-consistent. The route reads the active version and
 * the first history page separately, so a change stored between them leaves them disagreeing;
 * changes made against such a read could target a version the operator never saw.
 */
export function controlsConsistent(controls: {
  active: { version: number };
  history: { versions: { version: number }[] };
}) {
  return controls.history.versions[0]?.version === controls.active.version;
}

/**
 * The current policy from one consistent read, as display text; `null` while loading. An
 * inconsistent read is never shown as current: another change landed between its two reads.
 */
export function currentPolicyText(
  result: PolicyResult<PolicyControls> | null,
  { retrying }: { retrying: boolean },
) {
  if (result?.ok && !controlsConsistent(result.data))
    return retrying
      ? "Changed while it was being read; reading again…"
      : "Changed while it was being read; not confirmed";
  return policyStatusText(
    result?.ok ? { ok: true, data: result.data.active } : result,
  );
}

/** How often an open run's policy is read again, so "Current policy" follows changes. */
export const RUN_POLICY_REFRESH_MS = 10_000;
/** A read that changed underneath is retried quickly, a bounded number of times in a row. */
export const INCONSISTENT_POLICY_RETRY_MS = 1_000;
export const MAX_INCONSISTENT_POLICY_READS = 3;

/**
 * When to read a run's policy again: while the run is open, and soon after an inconsistent
 * read. A closed run links no new versions, so it stops once a read is settled.
 */
export function nextRunPolicyReadMs({
  closed,
  inconsistentReads,
}: {
  closed: boolean;
  /** Consecutive inconsistent reads, ending with the latest; 0 after any other result. */
  inconsistentReads: number;
}): number | null {
  if (
    inconsistentReads > 0 &&
    inconsistentReads < MAX_INCONSISTENT_POLICY_READS
  )
    return INCONSISTENT_POLICY_RETRY_MS;
  return closed ? null : RUN_POLICY_REFRESH_MS;
}

/** Shown wherever limits appear: no API reports which limits a runner actually applied. */
export const EFFECTIVE_LIMITS_UNAVAILABLE =
  "Not available from the API. Each agent runner records the limits it applied in its own local evidence, which this dashboard cannot read. A stored or linked version does not show that any work was limited.";

/** Strict whole numbers only: "1.5", "1e2" and " " are not limits. */
const WHOLE_NUMBER = /^\d+$/;

/** The form's limit inputs, so a validation message can name the inputs it is about. */
export type LimitField = "inference" | "tests";

function limitInput(field: LimitField, label: string, raw: string) {
  const value = raw.trim();
  if (value === "") return { ok: true as const, value: undefined };
  const parsed = Number(value);
  return WHOLE_NUMBER.test(value) && parsed >= 1 && parsed <= MAX_POLICY_LIMIT
    ? { ok: true as const, value: parsed }
    : {
        ok: false as const,
        message: `${label} must be a whole number from 1 to ${MAX_POLICY_LIMIT}, or empty for no limit.`,
        fields: [field],
      };
}

/** Limits for a `set` change from the form; an empty input means no limit for that control. */
export function limitsFromForm(form: {
  inference: string;
  tests: string;
}):
  | { ok: true; limits: PolicyLimitsView }
  | { ok: false; message: string; fields: LimitField[] } {
  const inference = limitInput(
    "inference",
    "Inference requests",
    form.inference,
  );
  if (!inference.ok) return inference;
  const tests = limitInput("tests", "Test jobs", form.tests);
  if (!tests.ok) return tests;
  if (inference.value === undefined && tests.value === undefined)
    return {
      ok: false,
      message:
        "Enter at least one limit, or use Disable limits to remove all limits.",
      fields: ["inference", "tests"],
    };
  return {
    ok: true,
    limits: {
      ...(inference.value !== undefined && {
        inference_requests: inference.value,
      }),
      ...(tests.value !== undefined && { test_jobs: tests.value }),
    },
  };
}
