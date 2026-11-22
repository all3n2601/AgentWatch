import type { MeasurementUnit } from "./measurement-format";
import {
  attributionItemSchema,
  FEATURE_VERSION,
  featureRowSchema,
  type AttributionItemData,
  type AttributionRecordData,
  type FeatureRow,
  type ModelOutcome,
  type ModelState,
  type ModelStatus,
  type ReceivedItem,
  type TelemetryStatus,
} from "./attribution-schema";
import type { Tone } from "./tone";

// Display model for B03 step attributions (docs/attribution-serving.md, "Dashboard handoff").
// A row is either valid, with its rules verdict beside the model outcome, or malformed with the
// reasons it cannot be shown; a malformed row never hides the others. A verdict or prediction
// names the resource a step likely waited on; nothing here states a cause.

/** Classes the served model and rules can name (`class_order` of the current pair). */
const CLASS_LABELS: Record<string, string> = {
  none: "None identified",
  cpu_sandbox: "CPU sandbox",
  inference: "Inference",
  retrieval: "Retrieval",
};

export const isRecognisedClass = (value: string) => value in CLASS_LABELS;
export const classLabel = (value: string) => CLASS_LABELS[value] ?? value;

/** The rules verdict, or the raw string marked unrecognised. */
export function rulesVerdictLabel(verdict: string): string {
  return isRecognisedClass(verdict)
    ? classLabel(verdict)
    : `Unrecognised: ${verdict}`;
}

/**
 * `rules-v1` has no rule for inference or retrieval (docs/rule-baseline.md). Served versions
 * carry a hash suffix (`rules-v1+1bf54c19`); any `rules-v1` build gets the note, others none.
 */
export const RULES_V1_NOTE =
  "rules-v1 only detects CPU sandbox contention: it never names inference or retrieval (docs/rule-baseline.md).";
export const rulesCoverageNote = (version: string): string | null =>
  version === "rules-v1" || version.startsWith("rules-v1+")
    ? RULES_V1_NOTE
    : null;

const OUTCOME_LABELS: Record<ModelOutcome, string> = {
  predicted: "Predicted",
  abstained: "Uncertain (abstained)",
  not_scored_step_state: "Not scored",
  unavailable: "Unavailable",
};
export const outcomeLabel = (outcome: ModelOutcome) => OUTCOME_LABELS[outcome];

const MODEL_STATE_LABELS: Record<ModelState, { label: string; tone: Tone }> = {
  not_configured: { label: "No model configured", tone: "info" },
  not_promoted_refused: {
    label: "Model refused: not promoted",
    tone: "warning",
  },
  ready_not_promoted: { label: "Serving, not promoted", tone: "warning" },
  ready_promoted: { label: "Serving, promoted", tone: "success" },
  load_failed: { label: "Model failed to load", tone: "danger" },
};
export const modelStateDisplay = (state: ModelState) =>
  MODEL_STATE_LABELS[state];

const TELEMETRY_LABELS: Record<TelemetryStatus, string> = {
  present: "Present",
  empty: "No samples in the step window",
  missing: "No telemetry for the step",
  no_overlap: "No samples overlap the step window",
  unaligned: "Samples in another clock domain",
};
export const telemetryLabel = (status: TelemetryStatus) =>
  TELEMETRY_LABELS[status];

/** Why the model has no output for a step, from the record's `model.reason`. */
export function unavailableReason(reason: string | null): string {
  switch (reason) {
    case null:
      return "No reason was recorded.";
    case "not_configured":
      return "No model pair is configured; only the rules verdict is recorded.";
    case "not_promoted_refused":
      return "The model pair is not promoted and serving it is not allowed.";
    case "prediction_failed":
      return "The served model rejected this step's values.";
    case "non_finite_prediction":
      return "The model's output was not a finite number.";
    default:
      return `The model pair failed a load check (${reason}).`;
  }
}

/** How the two halves of one row relate. Both scored the same feature row in one pass. */
export type Comparison =
  | "agree"
  | "differ"
  | "model_abstained"
  | "no_model_output"
  | "verdict_unrecognised";

export type ValidRow = {
  kind: "valid";
  sequence: number;
  servingKey: string;
  /** Written under the serving key the worker uses now. */
  current: boolean;
  evidenceChanged: boolean;
  stepId: string;
  stepState: AttributionRecordData["step_state"];
  featureVersion: string;
  rules: AttributionRecordData["rules"];
  model: AttributionRecordData["model"];
  missingInputs: string[];
  telemetryStatus: TelemetryStatus;
  feature: FeatureRow;
  comparison: Comparison;
};

export type MalformedRow = {
  kind: "malformed";
  sequence: number;
  /** Only when it could be read; a malformed row may not name its step. */
  stepId: string | null;
  servingKey: string | null;
  current: boolean;
  reasons: string[];
};

export type AttributionRow = ValidRow | MalformedRow;

const COMPARISON_DISPLAY: Record<Comparison, { label: string; tone: Tone }> = {
  agree: { label: "Same resource", tone: "info" },
  differ: { label: "Rules and model differ", tone: "warning" },
  model_abstained: { label: "Model uncertain", tone: "caution" },
  no_model_output: { label: "Rules only", tone: "info" },
  verdict_unrecognised: { label: "Unrecognised verdict", tone: "warning" },
};
export const comparisonDisplay = (comparison: Comparison) =>
  COMPARISON_DISPLAY[comparison];

/** Mixed evidence: the rules and the model name different resources, or the evidence changed. */
export const hasMixedEvidence = (row: ValidRow) =>
  row.comparison === "differ" || row.evidenceChanged;

/** Probabilities within this of each other are equal; the worker sends float64 values. */
const PROBABILITY_TOLERANCE = 1e-6;
const TIMING_NAMES = [
  "queue_seconds",
  "execution_seconds",
  "duration_seconds",
  "cpu_seconds",
] as const;
const SCORED_STATES: ModelState[] = ["ready_not_promoted", "ready_promoted"];

function topClass(probabilities: Record<string, number>) {
  let best: [string, number] | null = null;
  for (const entry of Object.entries(probabilities))
    if (best === null || entry[1] > best[1]) best = entry;
  return best;
}

/** Contradictions inside the model half of a record, each as a reviewer-readable reason. */
function predictionContradictions(record: AttributionRecordData): string[] {
  const { model, step_state: stepState } = record;
  const reasons: string[] = [];
  const predictionFields = [
    model.predicted_class,
    model.confidence,
    model.abstention_threshold,
    model.abstained,
    model.calibrated_probabilities,
    model.excess_fraction_estimate,
  ];
  if (model.state === "ready_not_promoted" && model.not_promoted !== true)
    reasons.push("Model state is not promoted but `not_promoted` is not true.");
  if (model.state === "ready_promoted" && model.not_promoted !== false)
    reasons.push("Model state is promoted but `not_promoted` is not false.");

  if (model.outcome === "predicted" || model.outcome === "abstained") {
    if (stepState !== "completed")
      reasons.push(`A ${stepState} step has a model ${model.outcome} outcome.`);
    if (!SCORED_STATES.includes(model.state))
      reasons.push(`Model state ${model.state} cannot score a step.`);
    if (model.reason !== null)
      reasons.push("A scored outcome carries an unavailability reason.");
    const {
      confidence,
      abstention_threshold: threshold,
      calibrated_probabilities: probabilities,
    } = model;
    if (
      confidence === null ||
      threshold === null ||
      probabilities === null ||
      model.abstained === null ||
      model.excess_fraction_estimate === null
    ) {
      reasons.push(`A ${model.outcome} outcome is missing prediction fields.`);
      return reasons;
    }
    const top = topClass(probabilities);
    if (top === null) {
      reasons.push("Calibrated probabilities are empty.");
      return reasons;
    }
    const total = Object.values(probabilities).reduce((a, b) => a + b, 0);
    if (Math.abs(total - 1) > PROBABILITY_TOLERANCE)
      reasons.push("Calibrated probabilities do not sum to 1.");
    if (Math.abs(confidence - top[1]) > PROBABILITY_TOLERANCE)
      reasons.push("Confidence is not the largest calibrated probability.");
    // The worker abstains below the threshold; equal to it is a prediction.
    const abstains = confidence < threshold;
    if (model.abstained !== (model.outcome === "abstained"))
      reasons.push("`abstained` contradicts the outcome.");
    if (abstains !== (model.outcome === "abstained"))
      reasons.push("The outcome contradicts confidence against the threshold.");
    if (model.outcome === "abstained" && model.predicted_class !== null)
      reasons.push("An abstention names a predicted class.");
    if (model.outcome === "predicted" && model.predicted_class !== top[0])
      reasons.push("The predicted class is not the most probable class.");
  } else {
    if (predictionFields.some((value) => value !== null))
      reasons.push(`A ${model.outcome} outcome carries prediction values.`);
    if (model.outcome === "not_scored_step_state" && stepState === "completed")
      reasons.push("A completed step is marked not scored.");
    if (model.outcome === "unavailable" && model.reason === null)
      reasons.push("An unavailable outcome has no reason.");
  }
  return reasons;
}

/** Contradictions between the record and its feature row, and inside the feature row. */
function featureContradictions(
  record: AttributionRecordData,
  feature: FeatureRow,
): string[] {
  const reasons: string[] = [];
  const { timings, telemetry, features } = feature;
  if (feature.run_id !== record.run_id)
    reasons.push("The feature row belongs to another run.");
  if (feature.step_id !== record.step_id)
    reasons.push("The feature row belongs to another step.");
  if (feature.feature_version !== record.feature_version)
    reasons.push("The feature row's version differs from the record's.");
  if (telemetry.status !== record.telemetry_status)
    reasons.push(
      "Telemetry status differs between the record and its feature row.",
    );
  if (TIMING_NAMES.some((name) => timings[name] !== features[name]))
    reasons.push("Timing features differ from the row timings.");
  if (
    features.telemetry_sample_count !== telemetry.sample_count ||
    features.telemetry_coverage !== telemetry.coverage
  )
    reasons.push("Telemetry features differ from the row telemetry.");
  if (
    features.cpu_pressure_available !==
    (features.cpu_pressure_max_avg10 !== null)
  )
    reasons.push("CPU pressure availability contradicts its value.");
  if (
    features.cgroup_available !==
    (features.cgroup_throttled_usec_delta !== null)
  )
    reasons.push("cgroup availability contradicts its value.");
  return reasons;
}

function compare(record: AttributionRecordData): Comparison {
  const { model, rules } = record;
  if (!isRecognisedClass(rules.verdict)) return "verdict_unrecognised";
  if (model.outcome === "abstained") return "model_abstained";
  if (model.outcome !== "predicted" || model.predicted_class === null)
    return "no_model_output";
  return model.predicted_class === rules.verdict ? "agree" : "differ";
}

const issueText = (
  prefix: string,
  issues: { path: PropertyKey[]; message: string }[],
) =>
  issues
    .slice(0, 3)
    .map(
      (issue) =>
        `${prefix}${issue.path.length ? ` ${issue.path.map(String).join(".")}` : ""}: ${issue.message}`,
    );

/** Fields worth naming on a row that failed validation, read without trusting the rest. */
function looseIdentity(item: ReceivedItem) {
  const record = (item as { record?: unknown }).record;
  const stepId =
    record !== null &&
    typeof record === "object" &&
    typeof (record as { step_id?: unknown }).step_id === "string"
      ? (record as { step_id: string }).step_id
      : null;
  const key = (item as { serving_key?: unknown }).serving_key;
  return { stepId, servingKey: typeof key === "string" ? key : null };
}

/**
 * One received row as the view shows it. `runId` is the selected run: a record of any other
 * run is malformed, never shown under this one.
 */
export function attributionRow(
  item: ReceivedItem,
  runId: string,
  currentServingKey: string,
): AttributionRow {
  const parsed = attributionItemSchema.safeParse(item);
  if (!parsed.success) {
    const { stepId, servingKey } = looseIdentity(item);
    return {
      kind: "malformed",
      sequence: item.sequence,
      stepId,
      servingKey,
      current: servingKey === currentServingKey,
      reasons: issueText("Field", parsed.error.issues),
    };
  }
  return validatedRow(parsed.data, runId, currentServingKey);
}

function validatedRow(
  item: AttributionItemData,
  runId: string,
  currentServingKey: string,
): AttributionRow {
  const { record } = item;
  const malformed = (reasons: string[]): MalformedRow => ({
    kind: "malformed",
    sequence: item.sequence,
    stepId: record.step_id,
    servingKey: item.serving_key,
    current: item.serving_key === currentServingKey,
    reasons,
  });
  if (record.run_id !== runId)
    return malformed(["The record belongs to another run."]);
  if (record.feature_version !== FEATURE_VERSION)
    return malformed([
      `Feature version ${record.feature_version} is not ${FEATURE_VERSION}, the only one this view reads.`,
    ]);
  const feature = featureRowSchema.safeParse(record.feature_row);
  if (!feature.success)
    return malformed(issueText("Feature row", feature.error.issues));
  const reasons = [
    ...featureContradictions(record, feature.data),
    ...predictionContradictions(record),
  ];
  if (reasons.length > 0) return malformed(reasons);
  return {
    kind: "valid",
    sequence: item.sequence,
    servingKey: item.serving_key,
    current: item.serving_key === currentServingKey,
    evidenceChanged: item.evidence_changed,
    stepId: record.step_id,
    stepState: record.step_state,
    featureVersion: record.feature_version,
    rules: record.rules,
    model: record.model,
    missingInputs: record.missing_inputs,
    telemetryStatus: record.telemetry_status,
    feature: feature.data,
    comparison: compare(record),
  };
}

export type ComparisonCounts = Record<Comparison, number>;

/**
 * Rows written under one serving key and feature version: the same rules version, model
 * configuration and feature definition, so their comparisons can be counted together.
 */
export type ComparisonGroup = {
  servingKey: string;
  featureVersion: string;
  current: boolean;
  rulesVersions: string[];
  trainingVersions: (string | null)[];
  calibrationVersions: (string | null)[];
  rows: number;
  counts: ComparisonCounts;
  evidenceChanged: number;
};

const emptyCounts = (): ComparisonCounts => ({
  agree: 0,
  differ: 0,
  model_abstained: 0,
  no_model_output: 0,
  verdict_unrecognised: 0,
});

const pushUnique = <T>(values: T[], value: T) => {
  if (!values.includes(value)) values.push(value);
};

/** Compatible groups, the current serving key first; rows of different groups are never mixed. */
export function comparisonGroups(
  rows: readonly AttributionRow[],
): ComparisonGroup[] {
  const groups = new Map<string, ComparisonGroup>();
  for (const row of rows) {
    if (row.kind !== "valid") continue;
    const id = `${row.servingKey}\u0000${row.featureVersion}`;
    let group = groups.get(id);
    if (!group) {
      group = {
        servingKey: row.servingKey,
        featureVersion: row.featureVersion,
        current: row.current,
        rulesVersions: [],
        trainingVersions: [],
        calibrationVersions: [],
        rows: 0,
        counts: emptyCounts(),
        evidenceChanged: 0,
      };
      groups.set(id, group);
    }
    group.rows += 1;
    group.counts[row.comparison] += 1;
    if (row.evidenceChanged) group.evidenceChanged += 1;
    pushUnique(group.rulesVersions, row.rules.version);
    pushUnique(group.trainingVersions, row.model.training_version);
    pushUnique(group.calibrationVersions, row.model.calibration_version);
  }
  return [...groups.values()].sort(
    (a, b) => Number(b.current) - Number(a.current),
  );
}

/** Calibrated probabilities, most probable first, as text the table and screen readers share. */
export function probabilityEntries(
  probabilities: Record<string, number> | null,
): { id: string; label: string; value: number }[] {
  if (!probabilities) return [];
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([id, value]) => ({ id, label: classLabel(id), value }));
}

/** Probabilities, confidence and the threshold share one precision so they compare directly. */
export const formatProbability = (value: number) => value.toFixed(3);

/**
 * A duration in seconds for `Measurement`: milliseconds below one second (so a measured
 * 0.3 ms never reads as zero), seconds above. Null stays null.
 */
export function secondsReading(seconds: number | null): {
  value: number | null;
  unit: MeasurementUnit;
} {
  if (seconds === null) return { value: null, unit: "s" };
  return seconds >= 1
    ? { value: seconds, unit: "s" }
    : { value: seconds * 1000, unit: "ms" };
}

/** A 0–1 fraction as a percentage reading; null stays null. */
export const percentReading = (fraction: number | null) =>
  fraction === null ? null : fraction * 100;

export type { ModelStatus };
