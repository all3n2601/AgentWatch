import {
  formatMeasurement,
  measurementStateLabel,
  type MeasurementState,
  type MeasurementUnit,
} from "@/lib/measurement-format";
import { cn } from "@/lib/utils";

/**
 * Renders one reading with its state. Missing readings show a dash plus a
 * state label (never a zero); estimates carry a visible "~".
 */
export function Measurement({
  value,
  unit,
  state = "measured",
  showState = false,
  className,
}: {
  value: number | null | undefined;
  unit: MeasurementUnit;
  state?: MeasurementState;
  /** Also print the state beside missing or estimated values. */
  showState?: boolean;
  className?: string;
}) {
  const formatted = formatMeasurement(value, unit, state);
  const stateVisible = showState && formatted.state !== "measured";
  return (
    <span
      data-state={formatted.state}
      className={cn(
        "tabular-nums",
        !formatted.hasValue && "text-ink-subtle",
        className,
      )}
    >
      <span aria-hidden>{formatted.text}</span>
      <span className="sr-only">{formatted.accessibleText}</span>
      {stateVisible && (
        <span aria-hidden className="ml-1.5 text-xs font-normal text-ink-muted">
          {measurementStateLabel(formatted.state)}
        </span>
      )}
    </span>
  );
}
