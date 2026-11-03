import type { PolicyFailure, PolicyResult } from "./policy-api";
import {
  MAX_POLICY_LIMIT,
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
  return `Policy while open: ${result.data.versions.map((v) => `v${v.version}`).join(" → ")}`;
}

/** Strict whole numbers only: "1.5", "1e2" and " " are not limits. */
const WHOLE_NUMBER = /^\d+$/;

function limitInput(label: string, raw: string) {
  const value = raw.trim();
  if (value === "") return { ok: true as const, value: undefined };
  const parsed = Number(value);
  return WHOLE_NUMBER.test(value) && parsed >= 1 && parsed <= MAX_POLICY_LIMIT
    ? { ok: true as const, value: parsed }
    : {
        ok: false as const,
        message: `${label} must be a whole number from 1 to ${MAX_POLICY_LIMIT}, or empty for no limit.`,
      };
}

/** Limits for a `set` change from the form; an empty input means no limit for that control. */
export function limitsFromForm(form: {
  inference: string;
  tests: string;
}): { ok: true; limits: PolicyLimitsView } | { ok: false; message: string } {
  const inference = limitInput("Inference requests", form.inference);
  if (!inference.ok) return inference;
  const tests = limitInput("Test jobs", form.tests);
  if (!tests.ok) return tests;
  if (inference.value === undefined && tests.value === undefined)
    return {
      ok: false,
      message:
        "Enter at least one limit, or use Disable limits to remove all limits.",
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
