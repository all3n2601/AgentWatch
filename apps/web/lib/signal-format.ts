// Text for resource-signal readings (live CPU and GPU samples), whose units are free-form
// strings reported with the series ("%", "ms", "W", "MiB", "MHz").

/** Units written without a space before them. */
const ATTACHED_UNITS = new Set(["%"]);

/**
 * A reading with precision that suits its size: two decimals below 10, one below 100, none
 * above. `compact` (axis ticks) drops trailing zeros and abbreviates thousands.
 */
export function formatSignal(value: number, unit: string, compact = false) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  let text: string;
  if (compact && magnitude >= 1000)
    text = `${Number((value / 1000).toFixed(1))}K`;
  else {
    const decimals = magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : 2;
    text = compact
      ? String(Number(value.toFixed(decimals)))
      : value.toFixed(decimals);
  }
  if (!unit) return text;
  return ATTACHED_UNITS.has(unit) ? `${text}${unit}` : `${text} ${unit}`;
}

/** The last measured reading and its position; null when nothing was measured. */
export function latestReading(points: readonly { value: number | null }[]) {
  for (let index = points.length - 1; index >= 0; index--) {
    const value = points[index].value;
    if (value !== null) return { value, index };
  }
  return null;
}
