/**
 * Formatting for measured values. A reading that is missing, unsupported or
 * malformed must never render as zero, so every value carries a state.
 */
export const MEASUREMENT_STATES = [
  "measured",
  "estimated",
  "unavailable",
  "not_instrumented",
] as const;
export type MeasurementState = (typeof MEASUREMENT_STATES)[number];

export type MeasurementUnit = "ms" | "s" | "percent" | "ratio" | "count";

export const MISSING_VALUE = "—";

const STATE_LABELS: Record<MeasurementState, string> = {
  measured: "Measured",
  estimated: "Estimated",
  unavailable: "Unavailable",
  not_instrumented: "Not instrumented",
};

export function measurementStateLabel(state: MeasurementState): string {
  return STATE_LABELS[state];
}

const countFormat = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

/** Formats a finite number in its unit, with the unit's documented precision. */
/** A duration in seconds in the unit that keeps it readable: s, ms or µs. */
export function formatSeconds(seconds: number): string {
  const magnitude = Math.abs(seconds);
  if (magnitude >= 1) return formatQuantity(seconds, "s");
  if (magnitude >= 0.001 || seconds === 0)
    return formatQuantity(seconds * 1000, "ms");
  return `${Math.round(seconds * 1e6)} µs`;
}

export function formatQuantity(value: number, unit: MeasurementUnit): string {
  switch (unit) {
    case "ms":
      // A measured 0.3 ms must not read as a measured zero.
      if (value !== 0 && Math.abs(value) < 1)
        return `${Math.abs(value) < 0.01 ? "< 0.01" : value.toFixed(2)} ms`;
      if (value !== 0 && Math.abs(value) < 10) return `${value.toFixed(1)} ms`;
      return `${countFormat.format(Math.round(value))} ms`;
    case "s":
      return `${value.toFixed(2)} s`;
    case "percent":
      return `${value.toFixed(1)}%`;
    case "ratio":
      return `${value.toFixed(1)}×`;
    case "count":
      return countFormat.format(value);
  }
}

export interface FormattedMeasurement {
  /** Visible text: the value, or a dash when there is no reading. */
  text: string;
  /** Text for assistive technology, which cannot see the dash's meaning. */
  accessibleText: string;
  state: MeasurementState;
  hasValue: boolean;
}

/**
 * `value` of null, undefined or a non-finite number is treated as unavailable
 * regardless of the declared state; a recorded zero stays a measured zero.
 */
export function formatMeasurement(
  value: number | null | undefined,
  unit: MeasurementUnit,
  state: MeasurementState = "measured",
): FormattedMeasurement {
  const finite = typeof value === "number" && Number.isFinite(value);
  const effectiveState: MeasurementState =
    !finite && (state === "measured" || state === "estimated")
      ? "unavailable"
      : state;

  if (
    !finite ||
    effectiveState === "unavailable" ||
    effectiveState === "not_instrumented"
  ) {
    return {
      text: MISSING_VALUE,
      accessibleText: STATE_LABELS[effectiveState],
      state: effectiveState,
      hasValue: false,
    };
  }

  const quantity = formatQuantity(value, unit);
  const estimated = effectiveState === "estimated";
  return {
    text: estimated ? `~${quantity}` : quantity,
    accessibleText: estimated ? `Estimated ${quantity}` : quantity,
    state: effectiveState,
    hasValue: true,
  };
}
