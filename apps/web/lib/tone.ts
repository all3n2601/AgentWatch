/**
 * Semantic tones shared by every design-system status surface. A tone only
 * picks colour; components always pair it with an icon and a text label so
 * status is never communicated by colour alone.
 */
export const TONES = [
  "success",
  "warning",
  "caution",
  "danger",
  "info",
] as const;
export type Tone = (typeof TONES)[number];

/** Statuses that appear across the approved screens, mapped to their tone. */
const STATUS_TONES = {
  passed: "success",
  complete: "success",
  completed: "success",
  match: "success",
  connected: "success",
  full: "success",
  attention: "warning",
  limited: "warning",
  running: "info",
  queued: "info",
  degraded: "caution",
  mismatch: "danger",
  failed: "danger",
  disconnected: "danger",
} as const satisfies Record<string, Tone>;

export type KnownStatus = keyof typeof STATUS_TONES;

/** Unknown statuses are informational, never silently "success". */
export function statusTone(status: string): Tone {
  const key = status.trim().toLowerCase();
  return key in STATUS_TONES ? STATUS_TONES[key as KnownStatus] : "info";
}

/**
 * Display severity for an observed slowdown ratio (candidate / clean baseline).
 * These thresholds only choose emphasis in the UI; they are not attribution
 * rules and make no causal claim. Ratios at or below 1 mean "not slower".
 */
export const SLOWDOWN_TONE_THRESHOLDS = {
  warning: 1.1,
  caution: 1.5,
  danger: 8,
} as const;

export function slowdownTone(ratio: number): Tone | null {
  if (!Number.isFinite(ratio) || ratio < 0) return null;
  if (ratio <= 1) return "success";
  if (ratio >= SLOWDOWN_TONE_THRESHOLDS.danger) return "danger";
  if (ratio >= SLOWDOWN_TONE_THRESHOLDS.caution) return "caution";
  if (ratio >= SLOWDOWN_TONE_THRESHOLDS.warning) return "warning";
  return "info";
}

/**
 * Emphasis for a delta chip. Neutral deltas get none at any ratio: a faster
 * candidate (ratio below 1) is as uncontrolled as a slower one, so it must not
 * read as an improvement.
 */
export function deltaTone(ratio: number | null, neutral: boolean): Tone | null {
  if (neutral || ratio === null || !Number.isFinite(ratio)) return null;
  return slowdownTone(ratio);
}

/**
 * One colour per resource class, reused by every chart, chip and legend
 * (docs/ui-redesign-plan.md section 4). Values are CSS variables so both
 * themes stay in sync with app/globals.css.
 */
export const RESOURCE_CLASSES = {
  cpu: { label: "CPU sandbox", color: "var(--series-amber)" },
  inference: { label: "Inference", color: "var(--series-violet)" },
  retrieval: { label: "Retrieval", color: "var(--series-sky)" },
  mixed: { label: "Mixed", color: "var(--series-rose)" },
  none: { label: "No excess", color: "var(--series-slate)" },
} as const;

export type ResourceClass = keyof typeof RESOURCE_CLASSES;
