import * as React from "react";
import { formatQuantity, type MeasurementUnit } from "@/lib/measurement-format";
import { layoutSegments } from "@/lib/segments";
import { cn } from "@/lib/utils";

/**
 * `work` is solid, `wait` is hatched, `idle` is a light neutral; `color`
 * overrides the fill for categorical bars such as telemetry coverage.
 */
export type SegmentKind = "work" | "wait" | "idle";

export interface Segment {
  id: string;
  label: string;
  value: number;
  kind?: SegmentKind;
  color?: string;
}

const KIND_FILL: Record<SegmentKind, string> = {
  work: "bg-bar",
  wait: "hatch",
  idle: "bg-bar-track",
};

function SegmentSwatch({ segment }: { segment: Segment }) {
  return (
    <i
      aria-hidden
      className={cn(
        "size-3.5 shrink-0 rounded-[4px] border border-line",
        !segment.color && KIND_FILL[segment.kind ?? "work"],
      )}
      style={segment.color ? { background: segment.color } : undefined}
    />
  );
}

/**
 * Proportional bar for one total split into parts ("Work versus wait",
 * "Telemetry coverage"). The legend lists every value and share as text.
 */
export function SegmentBar({
  segments,
  unit,
  label,
  showLabels = false,
  showLegend = true,
  className,
}: {
  segments: readonly Segment[];
  unit: MeasurementUnit;
  /** Accessible name, e.g. "Work versus wait for fix-window-count". */
  label: string;
  /** Print each segment's label and value above the bar (step context). */
  showLabels?: boolean;
  showLegend?: boolean;
  className?: string;
}) {
  const layout = layoutSegments(segments);

  if (layout.status !== "ok") {
    return (
      <p
        className={cn(
          "rounded-xl border border-dashed border-line px-4 py-3 text-sm text-ink-muted",
          className,
        )}
      >
        {label}:{" "}
        {layout.status === "empty" ? "no time recorded." : layout.reason}
      </p>
    );
  }

  return (
    <figure aria-label={label} className={cn("space-y-3", className)}>
      {showLabels && (
        <div className="flex text-sm text-ink" aria-hidden>
          {layout.segments.map((segment) => (
            <span
              key={segment.id}
              className="min-w-0 truncate pr-2 leading-tight"
              style={{ width: `${segment.share * 100}%`, minWidth: "4.5rem" }}
            >
              {segment.label}
              <br />
              <span className="tabular-nums">
                {formatQuantity(segment.value, unit)}
              </span>
            </span>
          ))}
        </div>
      )}
      <div
        className="flex h-8 gap-[3px] overflow-hidden rounded-lg"
        aria-hidden
      >
        {layout.segments.map((segment) => (
          <span
            key={segment.id}
            className={cn(
              "h-full first:rounded-l-lg last:rounded-r-lg",
              !segment.color && KIND_FILL[segment.kind ?? "work"],
            )}
            style={{
              width: `${segment.share * 100}%`,
              background: segment.color,
              minWidth: segment.value > 0 ? 3 : 0,
            }}
          />
        ))}
      </div>
      {showLegend && (
        <figcaption>
          <ul className="flex flex-wrap gap-x-8 gap-y-2 text-[15px] text-ink">
            {layout.segments.map((segment) => (
              <li key={segment.id} className="flex items-center gap-2">
                <SegmentSwatch segment={segment} />
                <span>{segment.label}</span>
                <span className="text-ink-muted tabular-nums">
                  {formatQuantity(segment.value, unit)} (
                  {formatQuantity(segment.share * 100, "percent")})
                </span>
              </li>
            ))}
          </ul>
        </figcaption>
      )}
    </figure>
  );
}

/** Compact legend for "Useful work / Waiting" beside a chart heading. */
/**
 * Legend for solid-versus-hatched bars. Execution wall time is not pure
 * useful work, so the default labels say what was measured.
 */
export function WorkWaitLegend({
  className,
  solidLabel = "Execution (wall)",
  hatchedLabel = "Queue wait",
}: {
  className?: string;
  solidLabel?: string;
  hatchedLabel?: string;
}) {
  return (
    <ul className={cn("flex items-center gap-5 text-sm text-ink", className)}>
      <li className="flex items-center gap-2">
        <i aria-hidden className="size-3.5 rounded-[4px] bg-bar" />
        {solidLabel}
      </li>
      <li className="flex items-center gap-2">
        <i
          aria-hidden
          className="hatch size-3.5 rounded-[4px] border border-line"
        />
        {hatchedLabel}
      </li>
    </ul>
  );
}
