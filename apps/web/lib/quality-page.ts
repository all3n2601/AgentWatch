import type { CheckState } from "../components/widgets/evidence-lists";
import type { DatasetSnapshot } from "./dataset-schema";
import { count, displayName, LABEL_RESOURCES } from "./dataset-view";

type Ready = Extract<DatasetSnapshot, { status: "ready" }>;

/**
 * Recorded statuses from the quality gate (`passed`/`failed`), the split-v2 checks
 * (`passed`/`blocked`/`inconclusive` for leakage, `met`/`not_met` for class balance and
 * held-out methods) and the evaluation contract (`accepted`). Anything else, including
 * `report_only` and `unavailable`, is not a verdict and maps to "unknown", never to a pass.
 */
const STATUS_STATES: Record<string, CheckState> = {
  passed: "pass",
  met: "pass",
  accepted: "pass",
  inconclusive: "warn",
  failed: "fail",
  blocked: "fail",
  not_met: "fail",
  rejected: "fail",
};

export function checkState(status: string): CheckState {
  return STATUS_STATES[status.trim().toLowerCase()] ?? "unknown";
}

/** A recorded machine status as text, e.g. "not_met" → "not met". */
export const statusText = (status: string) => status.replaceAll("_", " ");

export interface QualityCheck {
  id: string;
  label: string;
  /** Who recorded the check: the release's quality gate or its split checks. */
  group: "gate" | "split";
  state: CheckState;
  /** What was recorded, so the state can be traced back to the file. */
  detail: string;
}

export interface CountPart {
  id: string;
  label: string;
  value: number;
  color: string;
}

/**
 * Telemetry coverage states. They are three different situations, and neither missing state is
 * a measured zero load.
 */
const TELEMETRY_STATES: Record<string, { label: string; color: string }> = {
  present: { label: "Present", color: "var(--aw-success-solid)" },
  no_overlap: { label: "Missed step", color: "var(--aw-warning-solid)" },
  empty: { label: "No samples", color: "var(--aw-bar-track)" },
};
const TELEMETRY_ORDER = ["present", "no_overlap", "empty"];
const OTHER_COLOR = "var(--series-slate)";

/** Split names in reading order; any other split the file names follows, by name. */
const SPLIT_ORDER = ["train", "validation", "test"];

export interface SplitSummary {
  name: string;
  rows: number;
  runs: number;
  groups: number;
  labels: CountPart[];
}

export interface FieldHealthRow {
  field: string;
  missing: number;
  /** Rows with a value; null when the recorded missing count exceeds the row count. */
  recorded: number | null;
  /** Missing share of all rows, 0-1; null when the count is inconsistent with the rows. */
  share: number | null;
}

export interface AucRow {
  id: string;
  feature: string;
  /** Resource class for per-class leakage rows. */
  className?: string;
  auc: number;
  /** True when the AUC is above the leakage bound. */
  aboveBound?: boolean;
}

export interface QualityPageData {
  release: string;
  releaseName: string;
  schemaVersion: string;
  labelVersions: string[];
  rows: number;
  runs: number;
  tasks: number;
  /** Distinct experiments named by the release's runs. */
  experiments: number;
  gate: {
    status: string;
    state: CheckState;
    validatedAt: string;
    examples: number;
    gateVersion: string;
    tfdvVersion: string;
    anomalies: number;
    drift: number;
  };
  telemetry: { parts: CountPart[]; total: number; present: number };
  /** Rows with a resource label; null when the manifest does not count that column. */
  labelledRows: number | null;
  labels: CountPart[];
  labelBasis: CountPart[];
  kinds: CountPart[];
  conditions: CountPart[];
  split: {
    id: string;
    method: string;
    contract: { version: string; status: string; groupKey: string };
    parts: SplitSummary[];
    onlyInTest: string[];
    heldOutMethods: string[];
  };
  checks: QualityCheck[];
  leakage: { status: string; bound: number; rows: AucRow[] };
  designEncoding: { status: string; rows: number; features: AucRow[] };
  fieldHealth: { rows: FieldHealthRow[]; completeFields: number };
}

const CONDITION_COLORS = [
  "var(--series-slate)",
  "var(--series-emerald)",
  "var(--series-rose)",
  "var(--series-sky)",
];
const KIND_COLORS = [
  "var(--series-violet)",
  "var(--series-sky)",
  "var(--series-amber)",
  "var(--series-slate)",
];

/** Counts largest first, coloured in that order. */
function rankedParts(
  counts: Record<string, number>,
  colors: readonly string[],
): CountPart[] {
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([id, value], index) => ({
      id,
      label: displayName(id),
      value,
      color: colors[index % colors.length],
    }));
}

/** Label counts in the fixed class order, then any class the release adds. */
export function labelParts(counts: Record<string, number>): CountPart[] {
  const known = new Set<string>(LABEL_RESOURCES.map((entry) => entry.id));
  return [
    ...LABEL_RESOURCES.map((entry) => ({
      id: entry.id,
      label: entry.label,
      value: counts[entry.id] ?? 0,
      color: entry.color,
    })),
    ...rankedParts(
      Object.fromEntries(
        Object.entries(counts).filter(([id]) => !known.has(id)),
      ),
      [OTHER_COLOR],
    ),
  ];
}

export function telemetryParts(counts: Record<string, number>): CountPart[] {
  const known = TELEMETRY_ORDER.filter((id) => id in counts).map((id) => ({
    id,
    ...TELEMETRY_STATES[id],
    value: counts[id],
  }));
  const other = rankedParts(
    Object.fromEntries(
      Object.entries(counts).filter(([id]) => !(id in TELEMETRY_STATES)),
    ),
    [OTHER_COLOR],
  );
  return [...known, ...other];
}

/**
 * Columns with at least one empty value, highest missing share first (then by name), and how
 * many columns are complete. Empty means not recorded; it is never treated as a measured zero.
 */
export function fieldHealth(
  missing: Record<string, number>,
  rows: number,
): QualityPageData["fieldHealth"] {
  const entries = Object.entries(missing);
  const healthRows = entries
    .filter(([, value]) => value > 0)
    .map(([field, value]): FieldHealthRow => {
      const consistent = value <= rows;
      return {
        field,
        missing: value,
        recorded: consistent ? rows - value : null,
        share: consistent ? value / rows : null,
      };
    })
    .sort(
      (a, b) =>
        (b.share ?? Infinity) - (a.share ?? Infinity) ||
        a.field.localeCompare(b.field),
    );
  return {
    rows: healthRows,
    completeFields: entries.length - healthRows.length,
  };
}

export function splitSummaries(splits: Ready["split"]["splits"]) {
  const names = [
    ...SPLIT_ORDER.filter((name) => name in splits),
    ...Object.keys(splits)
      .filter((name) => !SPLIT_ORDER.includes(name))
      .sort(),
  ];
  return names.map((name): SplitSummary => ({
    name,
    rows: splits[name].rows,
    runs: splits[name].runs,
    groups: splits[name].groups,
    labels: labelParts(splits[name].label_resource),
  }));
}

const auc = (value: number) => value.toFixed(3);

/** A resource class by its label name ("CPU sandbox"); unknown classes keep their raw name. */
export const className = (id: string): string =>
  LABEL_RESOURCES.find((entry) => entry.id === id)?.label ?? displayName(id);

/** Validation and held-out checks, each with the status exactly as the file records it. */
export function qualityChecks(
  validation: Ready["validation"],
  split: Ready["split"],
): QualityCheck[] {
  const balance = split.class_balance;
  const perClass = Object.values(balance.per_class);
  const smallestClass = perClass.length ? Math.min(...perClass) : null;
  const strongest = split.leakage.features[0];
  const anomalies = validation.anomalies.length;
  const drift = validation.drift.length;
  const heldOut = split.held_out_injection_methods;

  return [
    {
      id: "gate",
      group: "gate",
      label: "Quality gate",
      state: checkState(validation.status),
      detail: `Recorded ${statusText(validation.status)} · ${count(validation.examples)} examples · gate ${validation.gate_version}, TFDV ${validation.tfdv_version}`,
    },
    {
      id: "anomalies",
      group: "gate",
      label: "Blocking schema anomalies",
      state: anomalies === 0 ? "pass" : "fail",
      detail:
        anomalies === 0
          ? "None recorded by the quality gate"
          : `${count(anomalies)} recorded by the quality gate`,
    },
    {
      id: "drift",
      group: "gate",
      label: "Drift against the previous passing release",
      // An empty list is also what the gate writes when it had no earlier release to compare
      // with, and the release it compared against is not read here, so zero is not a pass.
      state: drift === 0 ? "unknown" : "warn",
      detail:
        drift === 0
          ? "No drift findings recorded. This page does not read which release the gate compared against, so an empty list is not shown as a pass"
          : `${count(drift)} non-blocking drift ${drift === 1 ? "finding" : "findings"} recorded`,
    },
    {
      id: "leakage",
      group: "split",
      label: "Single-feature leakage",
      state: checkState(split.leakage.status),
      detail: `Recorded ${statusText(split.leakage.status)} · bound ${split.leakage.max_single_feature_auc.toFixed(2)} AUC${
        strongest
          ? ` · highest ${auc(strongest.auc)} (${className(strongest.class)}, ${strongest.feature})`
          : " · no per-class features recorded"
      }`,
    },
    {
      id: "class-balance",
      group: "split",
      label: "Class balance",
      state: checkState(balance.status),
      detail: `Recorded ${statusText(balance.status)} · clean ${count(balance.clean)} (minimum ${count(balance.minimums.clean)}) · mixed ${count(balance.mixed)} (minimum ${count(balance.minimums.mixed)})${
        smallestClass === null
          ? ""
          : ` · smallest class ${count(smallestClass)} (minimum ${count(balance.minimums.per_class)})`
      }`,
    },
    {
      id: "held-out",
      group: "split",
      label: "Held-out injection methods",
      state: checkState(heldOut.status),
      detail: `Recorded ${statusText(heldOut.status)} · ${
        heldOut.only_in_test.length
          ? `only in test: ${heldOut.only_in_test.join(", ")}`
          : "no method is held out of training"
      }`,
    },
    {
      id: "contract",
      group: "split",
      label: "Evaluation contract",
      state: checkState(split.contract.status),
      detail: `Recorded ${statusText(split.contract.status)} · ${split.contract.version} · grouped by ${split.contract.group_key}`,
    },
  ];
}

/**
 * A short tally of the checks for the Validation tab badge. Unknown and warning checks
 * count as not passed, so an unverified check never adds to "all passed".
 */
export function checkTally(checks: readonly QualityCheck[]): string {
  if (checks.length === 0) return "none recorded";
  const notPassed = checks.filter((check) => check.state !== "pass").length;
  return notPassed === 0 ? "all passed" : `${count(notPassed)} not passed`;
}

/** The `/quality` page's figures, all taken from one dataset release snapshot. */
export function buildQualityPage(snapshot: Ready): QualityPageData {
  const { manifest, split, validation } = snapshot;
  const telemetry = telemetryParts(manifest.telemetry_status);
  const experiments = new Set(
    snapshot.runs.flatMap((run) => (run.experiment ? [run.experiment] : [])),
  );
  const bound = split.leakage.max_single_feature_auc;

  return {
    release: manifest.dataset_version,
    releaseName: manifest.dataset_version.split("+")[0],
    schemaVersion: manifest.schema_version,
    labelVersions: manifest.label_version.split(",").filter(Boolean),
    rows: manifest.rows,
    runs: manifest.runs,
    tasks: manifest.tasks,
    experiments: experiments.size,
    gate: {
      status: validation.status,
      state: checkState(validation.status),
      validatedAt: validation.validated_at,
      examples: validation.examples,
      gateVersion: validation.gate_version,
      tfdvVersion: validation.tfdv_version,
      anomalies: validation.anomalies.length,
      drift: validation.drift.length,
    },
    telemetry: {
      parts: telemetry,
      total: telemetry.reduce((sum, part) => sum + part.value, 0),
      present: manifest.telemetry_status.present ?? 0,
    },
    labelledRows:
      "label_resource" in manifest.missing
        ? manifest.rows - manifest.missing.label_resource
        : null,
    labels: labelParts(manifest.label_resource),
    labelBasis: rankedParts(manifest.label_basis, [OTHER_COLOR]),
    kinds: rankedParts(manifest.kind, KIND_COLORS),
    conditions: rankedParts(manifest.condition, CONDITION_COLORS),
    split: {
      id: split.split_id,
      method: split.split_method,
      contract: {
        version: split.contract.version,
        status: split.contract.status,
        groupKey: split.contract.group_key,
      },
      parts: splitSummaries(split.splits),
      onlyInTest: split.held_out_injection_methods.only_in_test,
      heldOutMethods: split.held_out_injection_methods.methods,
    },
    checks: qualityChecks(validation, split),
    leakage: {
      status: split.leakage.status,
      bound,
      rows: split.leakage.features.map((feature) => ({
        id: `${feature.class}:${feature.feature}`,
        feature: feature.feature,
        className: className(feature.class),
        auc: feature.auc,
        aboveBound: feature.auc > bound,
      })),
    },
    designEncoding: {
      status: split.design_encoding.status,
      rows: split.design_encoding.rows,
      features: split.design_encoding.features.map((feature) => ({
        id: feature.feature,
        feature: feature.feature,
        auc: feature.auc,
      })),
    },
    fieldHealth: fieldHealth(manifest.missing, manifest.rows),
  };
}
