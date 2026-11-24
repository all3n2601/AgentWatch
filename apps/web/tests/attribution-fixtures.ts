// Synthetic B03 attribution rows for unit and browser tests only; never recorded evidence.
// Shapes follow api/openapi/attribution.json and the worker's record builder
// (services/worker/src/agentwatch_worker/attribution.py).

export const CURRENT_KEY = "c".repeat(64);
export const EARLIER_KEY = "e".repeat(64);
export const RULES_VERSION = "rules-v1+1bf54c19";
export const THRESHOLD = 0.647;

const IDENTITY = {
  training_version: "training-v2+synthetic",
  calibration_version: "calibration-v2+synthetic",
  classifier_sha256: "1".repeat(64),
  regressor_sha256: "2".repeat(64),
  preprocessing_sha256: "3".repeat(64),
  calibration_sha256: "4".repeat(64),
  temperature: 0.897,
  abstention_threshold: THRESHOLD,
  class_order: ["none", "cpu_sandbox", "inference", "retrieval"],
  blocks_promotion: true,
};

type Json = Record<string, unknown>;
type ModelState =
  | "not_configured"
  | "not_promoted_refused"
  | "ready_not_promoted"
  | "ready_promoted"
  | "load_failed";

/** The page's model status. `ready_not_promoted` is the synthetic pair behind the flag. */
export function modelStatus(
  state: ModelState = "ready_not_promoted",
  { servingKey = CURRENT_KEY, reason = null as string | null } = {},
): Json {
  const loaded = state !== "not_configured" && state !== "load_failed";
  return {
    state,
    reason,
    identity: loaded
      ? { ...IDENTITY, blocks_promotion: state !== "ready_promoted" }
      : null,
    not_promoted: loaded ? state !== "ready_promoted" : null,
    serving_key: servingKey,
  };
}

type Timings = {
  queue_seconds: number | null;
  execution_seconds: number | null;
  duration_seconds: number | null;
  cpu_seconds: number | null;
};

/** A consistent `step-features-v1` row; `queue_fraction` is derived as the worker does. */
export function featureRow(
  runId: string,
  stepId: string,
  {
    timings = {
      queue_seconds: 0.25,
      execution_seconds: 1.5,
      duration_seconds: 1.75,
      cpu_seconds: 0.9,
    } as Timings,
    status = "present" as string,
    sampleCount = 4,
    coverage = 1 as number | null,
    pressure = 12.5 as number | null,
    throttledUsec = 1500 as number | null,
  } = {},
): Json {
  const { queue_seconds: queue, duration_seconds: duration } = timings;
  return {
    contract_version: "live-contracts-v1",
    record_type: "feature_row",
    run_id: runId,
    task_id: "synthetic-attribution-v1",
    step_id: stepId,
    step_type: "tool",
    feature_version: "step-features-v1",
    source_event_ids: ["00000000-0000-4000-8000-000000000001"],
    source_sample_ids:
      sampleCount > 0 ? ["00000000-0000-4000-8000-000000000002"] : [],
    window: {
      clock_id: "monotonic",
      started_tick_ns: 1000,
      finished_tick_ns: 2000,
    },
    timings,
    telemetry: {
      status,
      sample_count: sampleCount,
      coverage,
      declared_interval_seconds: coverage === null ? null : 0.25,
      observed_interval_seconds: sampleCount >= 2 ? 0.25 : null,
    },
    features: {
      ...timings,
      queue_fraction: queue !== null && duration ? queue / duration : null,
      cpu_pressure_max_avg10: pressure,
      cpu_pressure_available: pressure !== null,
      cgroup_throttled_usec_delta: throttledUsec,
      cgroup_available: throttledUsec !== null,
      telemetry_sample_count: sampleCount,
      telemetry_coverage: coverage,
    },
  };
}

const servedLabels = (state: ModelState) => {
  const loaded = state !== "not_configured" && state !== "load_failed";
  return {
    state,
    training_version: loaded ? IDENTITY.training_version : null,
    calibration_version: loaded ? IDENTITY.calibration_version : null,
    not_promoted: loaded ? state !== "ready_promoted" : null,
  };
};

const noPrediction = {
  predicted_class: null,
  confidence: null,
  abstention_threshold: null,
  abstained: null,
  calibrated_probabilities: null,
  excess_fraction_estimate: null,
};

/** A scored model half; the outcome follows confidence against the threshold, as served. */
export function scored(
  probabilities: Record<string, number>,
  {
    excess = 0.2 as number | null,
    state = "ready_not_promoted" as ModelState,
    trainingVersion = IDENTITY.training_version,
  } = {},
): Json {
  const [top, confidence] = Object.entries(probabilities).sort(
    (a, b) => b[1] - a[1],
  )[0];
  const abstained = confidence < THRESHOLD;
  return {
    ...servedLabels(state),
    training_version: trainingVersion,
    reason: null,
    outcome: abstained ? "abstained" : "predicted",
    predicted_class: abstained ? null : top,
    confidence,
    abstention_threshold: THRESHOLD,
    abstained,
    calibrated_probabilities: probabilities,
    excess_fraction_estimate: excess,
  };
}

/** No model output: `unavailable` with its reason, or `not_scored_step_state`. */
export function unscored(
  outcome: "unavailable" | "not_scored_step_state",
  {
    state = "ready_not_promoted" as ModelState,
    reason = null as string | null,
  } = {},
): Json {
  return { ...servedLabels(state), reason, outcome, ...noPrediction };
}

export const CONFIDENT_INFERENCE = {
  none: 0.1,
  cpu_sandbox: 0.08,
  inference: 0.72,
  retrieval: 0.1,
};
export const CONFIDENT_CPU = {
  none: 0.05,
  cpu_sandbox: 0.85,
  inference: 0.05,
  retrieval: 0.05,
};
export const CONFIDENT_NONE = {
  none: 0.9,
  cpu_sandbox: 0.04,
  inference: 0.03,
  retrieval: 0.03,
};
/** Below the 0.647 threshold: the model abstains. */
export const UNCERTAIN = {
  none: 0.3,
  cpu_sandbox: 0.58,
  inference: 0.07,
  retrieval: 0.05,
};

export function attributionRecord(
  runId: string,
  stepId: string,
  {
    verdict = "none",
    rulesVersion = RULES_VERSION,
    model = scored(CONFIDENT_NONE),
    stepState = "completed",
    feature = featureRow(runId, stepId),
    missingInputs = [
      "workload_iterations",
      "tests_per_step",
      "worker_capacity",
    ],
  }: {
    verdict?: string;
    rulesVersion?: string;
    model?: Json;
    stepState?: string;
    feature?: Json;
    missingInputs?: string[];
  } = {},
): Json {
  const telemetry = (feature as { telemetry?: { status?: string } }).telemetry;
  return {
    run_id: runId,
    step_id: stepId,
    step_state: stepState,
    feature_version: "step-features-v1",
    feature_row: feature,
    missing_inputs: missingInputs,
    telemetry_status: telemetry?.status ?? "present",
    rules: { version: rulesVersion, verdict },
    model,
  };
}

export function attributionItem(
  sequence: number,
  record: Json,
  {
    servingKey = CURRENT_KEY,
    evidenceChanged = false,
  }: { servingKey?: string; evidenceChanged?: boolean } = {},
): Json & { sequence: number } {
  return {
    sequence,
    serving_key: servingKey,
    committed_at: "2026-11-22T10:00:01.000Z",
    step_terminal_received_at: "2026-11-22T10:00:00.975Z",
    evidence_changed: evidenceChanged,
    record,
  };
}

export function attributionPage(
  runId: string,
  attributions: (Json & { sequence: number })[],
  { model = modelStatus(), nextCursor = null as number | null } = {},
) {
  return { run_id: runId, model, attributions, next_cursor: nextCursor };
}
