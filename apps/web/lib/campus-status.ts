import { formatQuantity } from "./measurement-format";
import type { ModelOverview, PipelineStage } from "./model-overview";
import type { SceneAnchorId } from "./overview-scene";
import type { Tone } from "./tone";

/**
 * How the Overview's 3D campus turns measured counts into colour and traffic.
 * Every value here comes from `ModelOverview`; nothing is simulated. Thresholds
 * only choose emphasis and are not attribution rules or quality gates.
 */
export const EXCESS_SHARE_THRESHOLDS = {
  warning: 0.05,
  caution: 0.15,
  danger: 0.35,
} as const;
export const FAILURE_SHARE_THRESHOLDS = {
  warning: 0,
  caution: 0.05,
  danger: 0.15,
} as const;

export type StageId = SceneAnchorId;

export interface StageFact {
  label: string;
  value: string;
}

export interface StageStatus {
  id: StageId;
  name: string;
  /** Null when the stage has no steps to judge. */
  tone: Tone | null;
  /** One line for the label chip and screen readers. */
  summary: string;
  facts: StageFact[];
}

/** What a traffic bead stands for; the scene maps each kind to a colour. */
export type BeadKind = "normal" | "excess" | "passed" | "failed";

export interface RoadTraffic {
  base: BeadKind;
  flag: BeadKind;
  /** Share of beads drawn as `flag`, 0-1. */
  flagShare: number;
}

const count = (value: number) => formatQuantity(value, "count");
const percent = (share: number) => formatQuantity(share * 100, "percent");

export function excessTone(share: number): Tone {
  if (share >= EXCESS_SHARE_THRESHOLDS.danger) return "danger";
  if (share >= EXCESS_SHARE_THRESHOLDS.caution) return "caution";
  if (share >= EXCESS_SHARE_THRESHOLDS.warning) return "warning";
  return "success";
}

export function failureTone(share: number): Tone {
  if (share >= FAILURE_SHARE_THRESHOLDS.danger) return "danger";
  if (share >= FAILURE_SHARE_THRESHOLDS.caution) return "caution";
  if (share > FAILURE_SHARE_THRESHOLDS.warning) return "warning";
  return "success";
}

const STAGE_NAMES: Record<PipelineStage, { name: string; calls: string }> = {
  model: { name: "Model runtime", calls: "Model calls" },
  retrieval: { name: "Retrieval", calls: "Retrieval calls" },
  cpuTool: { name: "CPU tool", calls: "Tool calls" },
};

function pipelineStage(
  id: PipelineStage,
  overview: ModelOverview,
): StageStatus {
  const { calls, excessSteps } = overview.stages[id];
  const { name, calls: callsLabel } = STAGE_NAMES[id];
  const facts = [
    { label: callsLabel, value: count(calls) },
    { label: "With excess wait", value: count(excessSteps) },
  ];
  if (calls === 0)
    return {
      id,
      name,
      tone: null,
      summary: `No ${callsLabel.toLowerCase()} recorded`,
      facts,
    };
  const share = excessSteps / calls;
  return {
    id,
    name: id === "model" ? `${overview.model.name} runtime` : name,
    tone: excessTone(share),
    summary: `${percent(share)} of ${callsLabel.toLowerCase()} waited beyond the clean baseline`,
    facts: [...facts, { label: "Share", value: percent(share) }],
  };
}

function judged(overview: ModelOverview) {
  return overview.outcomes.passed + overview.outcomes.failed;
}

/** Status for each building, from the same counts the page's panels show. */
export function campusStages(
  overview: ModelOverview,
): Record<StageId, StageStatus> {
  const { passed, failed, unrecorded } = overview.outcomes;
  const total = judged(overview);
  const completeFacts = [
    { label: "Passed", value: count(passed) },
    { label: "Failed", value: count(failed) },
    ...(unrecorded > 0
      ? [{ label: "Outcome not recorded", value: count(unrecorded) }]
      : []),
  ];
  return {
    gateway: {
      id: "gateway",
      name: "Agent gateway",
      tone: "info",
      summary: `${count(overview.runs)} runs entered`,
      facts: [
        { label: "Runs", value: count(overview.runs) },
        { label: "Measured steps", value: count(overview.steps) },
      ],
    },
    model: pipelineStage("model", overview),
    retrieval: pipelineStage("retrieval", overview),
    cpuTool: pipelineStage("cpuTool", overview),
    complete:
      total === 0
        ? {
            id: "complete",
            name: "Complete",
            tone: null,
            summary: "No outcomes recorded",
            facts: completeFacts,
          }
        : {
            id: "complete",
            name: "Complete",
            tone: failureTone(failed / total),
            summary: `${count(failed)} of ${count(total)} runs failed`,
            facts: [
              ...completeFacts,
              { label: "Failed share", value: percent(failed / total) },
            ],
          },
  };
}

/** Traffic on each road, keyed by the road ids in `lib/overview-campus-3d.json`. */
export function campusTraffic(
  overview: ModelOverview,
): Record<string, RoadTraffic> {
  const excessShare = (id: PipelineStage) => {
    const { calls, excessSteps } = overview.stages[id];
    return calls === 0 ? 0 : excessSteps / calls;
  };
  const total = judged(overview);
  return {
    ingress: { base: "normal", flag: "normal", flagShare: 0 },
    model: { base: "normal", flag: "excess", flagShare: excessShare("model") },
    retrieval: {
      base: "normal",
      flag: "excess",
      flagShare: excessShare("retrieval"),
    },
    tool: { base: "normal", flag: "excess", flagShare: excessShare("cpuTool") },
    egress: {
      base: "passed",
      flag: "failed",
      flagShare: total === 0 ? 0 : overview.outcomes.failed / total,
    },
  };
}

/**
 * Which of `count` beads carry the flag, spread evenly so the stream reads as a
 * proportion. Any non-zero share shows at least one bead, so rare failures stay
 * visible; otherwise the flagged count is the share rounded to whole beads.
 */
export function spreadFlags(count: number, share: number): boolean[] {
  if (count <= 0) return [];
  const clamped = Number.isFinite(share) ? Math.min(1, Math.max(0, share)) : 0;
  const flagged = clamped === 0 ? 0 : Math.max(1, Math.round(count * clamped));
  return Array.from(
    { length: count },
    (_, i) =>
      Math.floor(((i + 1) * flagged) / count) >
      Math.floor((i * flagged) / count),
  );
}
