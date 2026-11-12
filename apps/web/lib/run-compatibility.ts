import type { Experiment } from "./experiment-schema";
import { formatQuantity } from "./measurement-format";

/**
 * The one compatibility rule for saved runs, shared by the Runs selection and
 * the Compare page. A setting is "unknown" when either run does not record it;
 * unknown is never counted as a match.
 */

/** Settings a saved run records; null where the run does not record one. */
export type RecordedConfiguration = {
  kind: Experiment["kind"];
  workload: string;
  model: string | null;
  workerCapacity: number;
  iterations: number;
  /** Contention intensity; older saved runs may not record it. */
  competingIterations: number | null;
  testsPerStep: number | null;
  baselineRepeats: number;
  cleanChecksum: string;
  ruleVersion: string;
};

export function configurationOf(run: Experiment): RecordedConfiguration {
  return {
    kind: run.kind,
    workload: run.experiment.workload,
    model: run.agent?.model ?? null,
    workerCapacity: run.experiment.worker_capacity,
    iterations: run.experiment.iterations,
    competingIterations: run.experiment.competing_iterations ?? null,
    testsPerStep: run.experiment.tests_per_step,
    baselineRepeats: run.experiment.baseline_repeats,
    cleanChecksum: run.clean.checksum,
    ruleVersion: run.clean.rule_version,
  };
}

export type CheckStatus = "same" | "differs" | "unknown";

export type CompatibilityCheck = {
  id: string;
  label: string;
  baseline: string;
  candidate: string;
  /** Full value for a shortened display, e.g. a checksum. */
  baselineTitle?: string;
  candidateTitle?: string;
  status: CheckStatus;
};

/**
 * "differs": a recorded setting differs, so no difference may be calculated.
 * "unverified": every recorded setting matches but at least one is unknown.
 * "match": every setting is recorded and equal.
 */
export type CompatibilityVerdict = "differs" | "unverified" | "match";

export type Compatibility = {
  checks: CompatibilityCheck[];
  verdict: CompatibilityVerdict;
  /** Labels of recorded settings that differ. */
  differing: string[];
  /** Labels of settings at least one run does not record. */
  unknown: string[];
};

export const NOT_RECORDED = "Not recorded";

/** Shown whenever the concurrency policy is unknown, which is true of every saved run. */
export const POLICY_NOT_RECORDED_WARNING =
  "Policy not recorded; full compatibility cannot be verified";

const POLICY_LABEL = "Concurrency policy";

const CHECKSUM_EDGE_LENGTH = 6;

const kindName = (kind: Experiment["kind"]) =>
  kind === "coding" ? "Coding" : "CPU";

const shortChecksum = (checksum: string) =>
  checksum.length > CHECKSUM_EDGE_LENGTH * 2 + 1
    ? `${checksum.slice(0, CHECKSUM_EDGE_LENGTH)}…${checksum.slice(-CHECKSUM_EDGE_LENGTH)}`
    : checksum;

const workers = (count: number) =>
  `${formatQuantity(count, "count")} ${count === 1 ? "worker" : "workers"}`;

const count = (value: number) => formatQuantity(value, "count");

type Setting = {
  id: string;
  label: string;
  /** Display value, or null when the run does not record the setting. */
  value: (config: RecordedConfiguration) => string | null;
  /** Full value compared and shown on hover when the display is shortened. */
  title?: (config: RecordedConfiguration) => string;
};

const SETTINGS: Setting[] = [
  { id: "kind", label: "Workload kind", value: (c) => kindName(c.kind) },
  { id: "workload", label: "Workload", value: (c) => c.workload },
  {
    id: "model",
    label: "Agent model",
    value: (c) => c.model ?? "None recorded",
  },
  {
    id: "capacity",
    label: "Worker capacity",
    value: (c) => workers(c.workerCapacity),
  },
  { id: "iterations", label: "Iterations", value: (c) => count(c.iterations) },
  {
    id: "contention",
    label: "Competing iterations (contention)",
    value: (c) =>
      c.competingIterations === null ? null : count(c.competingIterations),
  },
  {
    id: "tests",
    label: "Tests per step",
    // The CPU workload runs no tests by design; that is recorded, not missing.
    value: (c) =>
      c.testsPerStep === null ? "Not applicable" : count(c.testsPerStep),
  },
  {
    id: "repeats",
    label: "Baseline repeats",
    value: (c) => count(c.baselineRepeats),
  },
  {
    id: "checksum",
    label: "Clean output checksum",
    value: (c) => shortChecksum(c.cleanChecksum),
    title: (c) => c.cleanChecksum,
  },
  { id: "rule", label: "Attribution rule", value: (c) => c.ruleVersion },
  // Saved runs do not record the concurrency policy they ran under, so it is never
  // known. Read it here once the saved-run API links runs to policy versions.
  { id: "policy", label: POLICY_LABEL, value: () => null },
];

function checkSetting(
  setting: Setting,
  baseline: RecordedConfiguration,
  candidate: RecordedConfiguration,
): CompatibilityCheck {
  const base = setting.value(baseline);
  const cand = setting.value(candidate);
  const baseTitle = base === null ? undefined : setting.title?.(baseline);
  const candTitle = cand === null ? undefined : setting.title?.(candidate);
  const status: CheckStatus =
    base === null || cand === null
      ? "unknown"
      : (baseTitle ?? base) === (candTitle ?? cand)
        ? "same"
        : "differs";
  return {
    id: setting.id,
    label: setting.label,
    baseline: base ?? NOT_RECORDED,
    candidate: cand ?? NOT_RECORDED,
    ...(baseTitle !== undefined && { baselineTitle: baseTitle }),
    ...(candTitle !== undefined && { candidateTitle: candTitle }),
    status,
  };
}

export function compareConfiguration(
  baseline: RecordedConfiguration,
  candidate: RecordedConfiguration,
): Compatibility {
  const checks = SETTINGS.map((setting) =>
    checkSetting(setting, baseline, candidate),
  );
  const labels = (status: CheckStatus) =>
    checks.filter((check) => check.status === status).map((c) => c.label);
  const differing = labels("differs");
  const unknown = labels("unknown");
  return {
    checks,
    verdict:
      differing.length > 0
        ? "differs"
        : unknown.length > 0
          ? "unverified"
          : "match",
    differing,
    unknown,
  };
}

/** Whether the policy is among the unknown settings (always, for saved runs today). */
export const policyUnknown = (compatibility: Compatibility) =>
  compatibility.unknown.includes(POLICY_LABEL);
