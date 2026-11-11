/**
 * Geometry for proportional widgets (segment bars, bar lists and donuts). The
 * math lives here so components only render, and so invalid readings are
 * rejected instead of being drawn as an empty or misleading share.
 */
export interface SegmentInput {
  id: string;
  value: number;
}

export type SegmentShare<T extends SegmentInput> = T & {
  /** Fraction of the total, 0-1. */
  share: number;
  /** Cumulative fraction before this segment, 0-1. */
  offset: number;
};

export type SegmentLayout<T extends SegmentInput> =
  | { status: "ok"; total: number; segments: SegmentShare<T>[] }
  | { status: "empty"; total: 0; segments: [] }
  | { status: "invalid"; reason: string };

export function layoutSegments<T extends SegmentInput>(
  items: readonly T[],
): SegmentLayout<T> {
  const ids = new Set<string>();
  for (const item of items) {
    if (!Number.isFinite(item.value) || item.value < 0) {
      return {
        status: "invalid",
        reason: `Segment "${item.id}" has no valid reading.`,
      };
    }
    if (ids.has(item.id)) {
      return {
        status: "invalid",
        reason: `Segment id "${item.id}" is repeated.`,
      };
    }
    ids.add(item.id);
  }
  const total = items.reduce((sum, item) => sum + item.value, 0);
  if (total === 0) return { status: "empty", total: 0, segments: [] };

  let offset = 0;
  const segments = items.map((item) => {
    const share = item.value / total;
    const segment = { ...item, share, offset };
    offset += share;
    return segment;
  });
  return { status: "ok", total, segments };
}

/**
 * Bar lengths relative to the largest value, as 0-1 fractions. Lists compare
 * magnitudes, so the longest bar fills the track. Invalid values yield null.
 */
export function relativeLengths(values: readonly number[]): (number | null)[] {
  const valid = values.filter((value) => Number.isFinite(value) && value >= 0);
  const max = Math.max(0, ...valid);
  return values.map((value) => {
    if (!Number.isFinite(value) || value < 0) return null;
    return max === 0 ? 0 : value / max;
  });
}

/** SVG stroke-dasharray/offset for one donut arc on a circle of `circumference`. */
export function donutArc(
  share: number,
  offset: number,
  circumference: number,
  gap = 0,
) {
  const length = Math.max(0, share * circumference - gap);
  return {
    dashArray: `${length} ${circumference - length}`,
    // SVG circles start at 3 o'clock; a quarter turn starts arcs at 12 o'clock.
    dashOffset: circumference * 0.25 - offset * circumference,
  };
}
