/**
 * Illustrative values for the component review page only. They mirror the
 * approved mockups so components can be compared with the design; they are
 * not recorded evidence and must never feed a product page.
 */
export const SAMPLE_CALL_TYPES = [
  { id: "model", label: "Model calls", value: 2847 },
  { id: "tool", label: "Tool calls", value: 2158 },
  { id: "retrieval", label: "Retrieval calls", value: 598 },
] as const;

export const SAMPLE_RESOURCE_LABELS = [
  { id: "inference", label: "Inference", value: 326 },
  { id: "retrieval", label: "Retrieval", value: 322 },
  { id: "cpu", label: "CPU sandbox", value: 205 },
  { id: "mixed", label: "Mixed", value: 146 },
] as const;

export const SAMPLE_OUTCOMES = [
  {
    id: "passed",
    label: "Passed",
    value: 585,
    color: "var(--aw-success-solid)",
  },
  { id: "failed", label: "Failed", value: 13, color: "var(--aw-danger-solid)" },
] as const;

export const SAMPLE_WORK_WAIT = [
  { id: "work", label: "Work", value: 8.48, kind: "work" },
  { id: "wait", label: "Wait (excess)", value: 18.76, kind: "wait" },
] as const;

export const SAMPLE_STEP_CONTEXT = [
  { id: "queue", label: "Queue", value: 0.069, kind: "idle" },
  { id: "execute", label: "Execute", value: 2.18, kind: "work" },
  { id: "wait", label: "Wait (contention)", value: 10.37, kind: "wait" },
  { id: "finalize", label: "Finalize", value: 0.17, kind: "idle" },
] as const;

export const SAMPLE_COVERAGE = [
  {
    id: "present",
    label: "Present",
    value: 777,
    color: "var(--aw-success-solid)",
  },
  {
    id: "missed",
    label: "Missed step",
    value: 1731,
    color: "var(--aw-warning-solid)",
  },
  {
    id: "none",
    label: "No samples",
    value: 3445,
    color: "var(--aw-bar-track)",
  },
] as const;
