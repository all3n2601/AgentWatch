import { stepsOf } from "./dashboard-data";
import { displayName, LABEL_RESOURCES } from "./dataset-view";
import type { Experiment, Step } from "./experiment-schema";
import type { LiveRunSnapshot, LiveStepView } from "./live-api-schema";
import { formatMeasurement, type MeasurementUnit } from "./measurement-format";
import type { Tone } from "./tone";
import {
  diagnoseView,
  fromSaved,
  type DiagnosisStep,
  type RunState,
  type RunView,
} from "./unified-runs";

/**
 * View model for the Run detail page (`/runs/[runId]`). Everything here is derived from the
 * one run the page loaded; no other run, baseline or fixture is mixed in. Missing readings stay
 * null so the page renders them as unavailable, never as zero.
 */

const MS_PER_SECOND = 1000;
const NS_PER_MS = 1e6;

/** One row of "Evidence for selected step". Null values render as unavailable. */
export type EvidenceRow =
  | { id: string; label: string; kind: "seconds"; value: number | null }
  | { id: string; label: string; kind: "percent"; value: number | null }
  | { id: string; label: string; kind: "count"; value: number | null }
  | { id: string; label: string; kind: "text"; value: string | null };

export type StepFlag = { label: string; tone: Tone };

export type RunDetailStep = DiagnosisStep & {
  /** Human name: "Contended", "Baseline repeat 2 of 5", or the live step id. */
  label: string;
  /** Display name of the resource, when one is recorded. */
  resourceLabel: string | null;
  /** Short statement of what the resource claim rests on. */
  basisLabel: string;
  /** Row status pill: the rule outcome (saved) or the step state (live). */
  flag: StepFlag;
  evidence: EvidenceRow[];
};

export type RunHeadline = { tone: Tone; title: string; detail: string };

export type RunDetailModel = {
  runId: string;
  source: "live" | "saved";
  title: string;
  state: RunState;
  /** Timezone-aware ISO time and what it marks. */
  time: { label: string; iso: string };
  steps: RunDetailStep[];
  totals: {
    /** Sum of measured step durations; null when no step has a duration. */
    stepSeconds: number | null;
    /** Sum of measured queue delays; null when no step has a queue reading. */
    queueSeconds: number | null;
    measuredQueueSteps: number;
    totalSteps: number;
    /** Steps the saved rule attributed excess wait to; null for live runs (no attribution). */
    attributedSteps: number | null;
  };
  /** Contended over clean-control duration for saved runs; null when not applicable. */
  contendedOverClean: number | null;
  telemetry: { label: string; tone: Tone; detail: string };
  headline: RunHeadline | null;
  /** Statements the reader needs before trusting the figures. */
  notes: string[];
  limitations: string[];
};

const BASIS_LABELS: Record<DiagnosisStep["basis"], string> = {
  attributed: "Rule attribution recorded with the run",
  ran_on: "Where the step ran; not an attribution of waiting",
  none: "No resource recorded",
};

const resourceName = (resource: string) =>
  LABEL_RESOURCES.find((entry) => entry.id === resource)?.label ??
  displayName(resource);

const sentence = (value: string) =>
  value.charAt(0).toUpperCase() + value.slice(1);

/** UTC wall-clock text with millisecond precision, identical on server and client. */
export function formatUtc(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const text = date.toISOString();
  return `${text.slice(0, 10)} ${text.slice(11, 23)} UTC`;
}

/** Saved step timestamps are epoch nanoseconds; milliseconds survive the conversion. */
export const nsToIso = (ns: number) => new Date(ns / NS_PER_MS).toISOString();

/**
 * A step timing in the unit a reader expects: milliseconds below one second, seconds above.
 * Readings below one millisecond say so instead of rounding a measured value down to "0 ms".
 */
export function formatTiming(seconds: number | null): {
  text: string;
  accessibleText: string;
  hasValue: boolean;
} {
  if (seconds !== null && Number.isFinite(seconds) && seconds > 0) {
    if (seconds < 0.001)
      return {
        text: "< 1 ms",
        accessibleText: "less than 1 millisecond",
        hasValue: true,
      };
  }
  const unit: MeasurementUnit =
    seconds !== null && Math.abs(seconds) >= 1 ? "s" : "ms";
  const value =
    seconds === null ? null : unit === "ms" ? seconds * MS_PER_SECOND : seconds;
  const formatted = formatMeasurement(value, unit);
  return {
    text: formatted.text,
    accessibleText: formatted.accessibleText,
    hasValue: formatted.hasValue,
  };
}

/** Queue and execution as fractions of the longest step in the list, for the row bars. */
export function rowBarShares(
  step: Pick<DiagnosisStep, "queueSeconds" | "executionSeconds">,
  longestSeconds: number,
): { queue: number; execution: number } | null {
  const { queueSeconds: queue, executionSeconds: execution } = step;
  if (
    queue === null ||
    execution === null ||
    !Number.isFinite(queue) ||
    !Number.isFinite(execution) ||
    queue < 0 ||
    execution < 0 ||
    !(longestSeconds > 0)
  )
    return null;
  return {
    queue: queue / longestSeconds,
    execution: execution / longestSeconds,
  };
}

/** The longest queue-plus-execution among steps that have both readings. */
export function longestMeasured(steps: readonly DiagnosisStep[]): number {
  return steps.reduce(
    (longest, step) =>
      step.queueSeconds === null || step.executionSeconds === null
        ? longest
        : Math.max(longest, step.queueSeconds + step.executionSeconds),
    0,
  );
}

/**
 * Queue and execution for the selected-step bar, in one unit chosen from the step's size.
 * Null when either reading is missing, so the bar never draws a partial split as the whole.
 */
export function contextSegments(
  step: Pick<DiagnosisStep, "queueSeconds" | "executionSeconds">,
): {
  unit: "ms" | "s";
  segments: {
    id: string;
    label: string;
    value: number;
    kind: "wait" | "work";
  }[];
} | null {
  const { queueSeconds: queue, executionSeconds: execution } = step;
  if (queue === null || execution === null) return null;
  const unit = queue + execution >= 1 ? "s" : "ms";
  const scale = unit === "ms" ? MS_PER_SECOND : 1;
  return {
    unit,
    segments: [
      { id: "queue", label: "Queue", value: queue * scale, kind: "wait" },
      {
        id: "execution",
        label: "Execution (wall)",
        value: execution * scale,
        kind: "work",
      },
    ],
  };
}

function savedLabel(step: Step, run: Experiment): string {
  if (step.step_id === "clean") return "Clean control";
  if (step.step_id === "contended") return "Contended";
  const repeat = /^baseline-(\d+)$/.exec(step.step_id);
  if (repeat)
    return `Baseline repeat ${Number(repeat[1]) + 1} of ${run.baseline.steps.length}`;
  return step.step_id;
}

function savedEvidence(step: Step, run: Experiment): EvidenceRow[] {
  const telemetry = step.telemetry;
  return [
    { id: "role", label: "Step", kind: "text", value: savedLabel(step, run) },
    {
      id: "submitted",
      label: "Submitted",
      kind: "text",
      value: formatUtc(nsToIso(step.submitted_ns)),
    },
    {
      id: "started",
      label: "Started",
      kind: "text",
      value: formatUtc(nsToIso(step.started_ns)),
    },
    {
      id: "finished",
      label: "Finished",
      kind: "text",
      value: formatUtc(nsToIso(step.finished_ns)),
    },
    { id: "queue", label: "Queue", kind: "seconds", value: step.queue_seconds },
    {
      id: "execution",
      label: "Execution (wall)",
      kind: "seconds",
      value: step.execution_seconds,
    },
    {
      id: "cpu",
      label: "Process CPU time",
      kind: "seconds",
      value: step.cpu_seconds,
    },
    {
      id: "share",
      label: "Waiting share",
      kind: "percent",
      value: step.queue_fraction * 100,
    },
    {
      id: "samples",
      label: "Host samples",
      kind: "count",
      value: telemetry?.sample_count ?? null,
    },
    {
      id: "pressure",
      label: "Max CPU pressure (avg10)",
      kind: "percent",
      // No valid pressure sample means no reading, even if a zero was written.
      value:
        telemetry && telemetry.valid_pressure_samples > 0
          ? telemetry.max_cpu_pressure_avg10_percent
          : null,
    },
    {
      id: "telemetry",
      label: "Telemetry",
      kind: "text",
      value: telemetry ? displayName(telemetry.status) : null,
    },
    {
      id: "attribution",
      label: "Attribution",
      kind: "text",
      value:
        step.resource === "none"
          ? "No excess wait attributed"
          : resourceName(step.resource),
    },
    {
      id: "rule",
      label: "Attribution method",
      kind: "text",
      value: `${step.rule_version} · queue over ${formatTiming(step.threshold_seconds).text}`,
    },
    {
      id: "timing-source",
      label: "Timing source",
      kind: "text",
      value: step.evidence,
    },
  ];
}

const CPU_SCOPES: Record<string, string> = {
  process: "process",
  process_and_children: "process and children",
};

function liveEvidence(
  view: LiveStepView,
  snapshot: LiveRunSnapshot,
): EvidenceRow[] {
  const { event, timings } = view;
  const resource = snapshot.registration.resources.find(
    (entry) => entry.resource_id === event.resource_id,
  );
  return [
    {
      id: "type",
      label: "Step type",
      kind: "text",
      value: sentence(displayName(event.step_type)),
    },
    { id: "state", label: "State", kind: "text", value: sentence(event.state) },
    {
      id: "observed",
      label: "Last event observed",
      kind: "text",
      value: formatUtc(event.observed_at),
    },
    {
      id: "queue",
      label: "Queue",
      kind: "seconds",
      value: timings.queue_seconds,
    },
    {
      id: "execution",
      label: "Execution (wall)",
      kind: "seconds",
      value: timings.execution_seconds,
    },
    {
      id: "cpu",
      label: event.cpu_scope
        ? `CPU time (${CPU_SCOPES[event.cpu_scope] ?? event.cpu_scope})`
        : "CPU time",
      kind: "seconds",
      value: timings.cpu_seconds,
    },
    {
      id: "share",
      label: "Waiting share",
      kind: "percent",
      value:
        timings.queue_fraction === null ? null : timings.queue_fraction * 100,
    },
    {
      id: "ran-on",
      label: "Ran on",
      kind: "text",
      value: event.resource_id,
    },
    {
      id: "service",
      label: "Service",
      kind: "text",
      value: event.service_id ?? resource?.service_id ?? null,
    },
    {
      id: "class",
      label: "Resource class",
      kind: "text",
      value: resource ? resourceName(resource.resource_class) : null,
    },
    {
      id: "attribution",
      label: "Attribution",
      kind: "text",
      value: "None recorded for live runs",
    },
    {
      id: "clock",
      label: "Timing clock",
      kind: "text",
      value: `${event.clock_id} (monotonic ticks)`,
    },
    {
      id: "parent",
      label: "Parent step",
      kind: "text",
      value: event.parent_step_id,
    },
    { id: "reason", label: "Reason", kind: "text", value: event.reason },
  ];
}

const LIVE_STATE_FLAGS: Record<string, Tone> = {
  completed: "success",
  failed: "danger",
  running: "info",
  queued: "info",
  cancelled: "warning",
};

function savedSteps(run: Experiment): RunDetailStep[] {
  const byId = new Map(stepsOf(run).map((step) => [step.step_id, step]));
  return diagnoseView({ source: "saved", run }).map((diagnosis) => {
    const step = byId.get(diagnosis.stepId)!;
    return {
      ...diagnosis,
      label: savedLabel(step, run),
      resourceLabel: diagnosis.resource
        ? resourceName(diagnosis.resource)
        : null,
      basisLabel: BASIS_LABELS[diagnosis.basis],
      flag:
        diagnosis.basis === "attributed"
          ? { label: "Excess wait", tone: "warning" }
          : { label: "No excess wait", tone: "success" },
      evidence: savedEvidence(step, run),
    };
  });
}

function liveSteps(snapshot: LiveRunSnapshot): RunDetailStep[] {
  const byId = new Map(
    snapshot.steps.map((view) => [view.event.step_id, view]),
  );
  return diagnoseView({ source: "live", snapshot }).map((diagnosis) => {
    const view = byId.get(diagnosis.stepId)!;
    return {
      ...diagnosis,
      label: diagnosis.stepId,
      resourceLabel: diagnosis.resource
        ? resourceName(diagnosis.resource)
        : null,
      basisLabel: BASIS_LABELS[diagnosis.basis],
      flag: {
        label: sentence(diagnosis.state),
        tone: LIVE_STATE_FLAGS[diagnosis.state] ?? "info",
      },
      evidence: liveEvidence(view, snapshot),
    };
  });
}

function sumOrNull(values: (number | null)[]): number | null {
  const present = values.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  );
  return present.length === 0
    ? null
    : present.reduce((sum, value) => sum + value, 0);
}

/** The top-ranked step stated as an observation, with the qualifier it needs. */
export function headlineFor(
  steps: readonly RunDetailStep[],
  contendedOverClean: number | null,
): RunHeadline | null {
  if (steps.length === 0) return null;
  const top = steps[0];
  if (top.queueSeconds === null)
    return {
      tone: "info",
      title: "No step has a measured queue delay",
      detail:
        "Steps that never started, or recorded no start tick, cannot be ranked by waiting.",
    };
  if (top.queueSeconds === 0)
    return {
      tone: "success",
      title: "No step waited in queue",
      detail: "Every measured queue delay is zero.",
    };
  const ratio =
    contendedOverClean === null
      ? ""
      : `Contended step took ${contendedOverClean.toFixed(1)}× the clean control's time · `;
  return {
    tone: top.basis === "attributed" ? "warning" : "info",
    title: `${top.label} waited longest: ${formatTiming(top.queueSeconds).text} in queue`,
    detail: `${ratio}Ranked by measured queue delay · Observed, not causal`,
  };
}

function savedTelemetry(run: Experiment): RunDetailModel["telemetry"] {
  const steps = stepsOf(run);
  const withPressure = steps.filter(
    (step) => (step.telemetry?.valid_pressure_samples ?? 0) > 0,
  ).length;
  return {
    label:
      withPressure === steps.length
        ? "Host telemetry on every step"
        : withPressure === 0
          ? "Host telemetry unavailable"
          : "Telemetry limited",
    tone: withPressure === steps.length ? "success" : "warning",
    detail: `${withPressure} of ${steps.length} steps have a valid CPU-pressure sample`,
  };
}

export function buildRunDetail(view: RunView): RunDetailModel {
  if (view.source === "live") {
    const { snapshot } = view;
    const steps = liveSteps(snapshot);
    const notes = [
      "Live runs record which resource each step ran on and its timings. No per-step attribution of waiting is stored, so none is shown.",
    ];
    if (snapshot.reason) notes.push(`Run state reason: ${snapshot.reason}`);
    if (snapshot.gaps.length > 0)
      notes.push(
        `${snapshot.gaps.length} sequence gap(s) in the live records; some step timings may be incomplete.`,
      );
    if (snapshot.unfinished_steps.length > 0)
      notes.push(
        `${snapshot.unfinished_steps.length} step(s) have not finished: ${snapshot.unfinished_steps.join(", ")}.`,
      );
    return {
      runId: snapshot.run_id,
      source: "live",
      title: snapshot.registration.task_id,
      state: snapshot.state,
      time: { label: "Added", iso: snapshot.created_at },
      steps,
      totals: totalsFor(steps, null),
      contendedOverClean: null,
      telemetry: {
        label: "Resource samples on the live tab",
        tone: "info",
        detail:
          "Host samples are read from the live API on the Timeline, resources and evidence tab.",
      },
      headline: headlineFor(steps, null),
      notes,
      limitations: [],
    };
  }

  const { run } = view;
  const summary = fromSaved(run);
  const steps = savedSteps(run);
  const contendedOverClean =
    run.clean.duration_seconds > 0
      ? run.contended.duration_seconds / run.clean.duration_seconds
      : null;
  return {
    runId: run.run_id,
    source: "saved",
    title: summary.title,
    state: summary.state,
    time: { label: "Finished", iso: summary.at },
    steps,
    totals: totalsFor(
      steps,
      steps.filter((step) => step.basis === "attributed").length,
    ),
    contendedOverClean,
    telemetry: savedTelemetry(run),
    headline: headlineFor(steps, contendedOverClean),
    notes: [
      "Attributions on saved runs come from the rule recorded with the run. This page does not report how accurate that rule is.",
    ],
    limitations: run.limitations,
  };
}

function totalsFor(
  steps: RunDetailStep[],
  attributedSteps: number | null,
): RunDetailModel["totals"] {
  return {
    stepSeconds: sumOrNull(steps.map((step) => step.durationSeconds)),
    queueSeconds: sumOrNull(steps.map((step) => step.queueSeconds)),
    measuredQueueSteps: steps.filter((step) => step.queueSeconds !== null)
      .length,
    totalSteps: steps.length,
    attributedSteps,
  };
}
