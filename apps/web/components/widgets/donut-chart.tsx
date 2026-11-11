import * as React from "react";
import { formatQuantity } from "@/lib/measurement-format";
import { donutArc, layoutSegments } from "@/lib/segments";
import { cn } from "@/lib/utils";

const RADIUS = 42;
const STROKE_WIDTH = 13;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const ARC_GAP = 1.2;

export interface DonutSlice {
  id: string;
  label: string;
  value: number;
  /** CSS colour, e.g. "var(--aw-success-solid)". */
  color: string;
}

/** Share-of-whole ring ("Run outcomes") with counts as a visible legend. */
export function DonutChart({
  slices,
  label,
  size = 164,
  legend = "inside",
  className,
}: {
  slices: readonly DonutSlice[];
  label: string;
  size?: number;
  /** "side" puts the counts beside a smaller ring, for short panels. */
  legend?: "inside" | "side";
  className?: string;
}) {
  const layout = layoutSegments(slices);
  if (layout.status !== "ok") {
    return (
      <p className="text-sm text-ink-muted">
        {label}:{" "}
        {layout.status === "empty" ? "no runs recorded." : layout.reason}
      </p>
    );
  }

  return (
    <figure
      aria-label={label}
      className={cn(
        legend === "side"
          ? "flex w-fit items-center gap-4"
          : "relative grid w-fit place-items-center",
        className,
      )}
    >
      <svg viewBox="0 0 100 100" width={size} height={size} aria-hidden>
        <circle
          cx="50"
          cy="50"
          r={RADIUS}
          fill="none"
          stroke="var(--aw-bar-track)"
          strokeWidth={STROKE_WIDTH}
        />
        {layout.segments.map((slice) => {
          const arc = donutArc(
            slice.share,
            slice.offset,
            CIRCUMFERENCE,
            layout.segments.length > 1 ? ARC_GAP : 0,
          );
          return (
            <circle
              key={slice.id}
              cx="50"
              cy="50"
              r={RADIUS}
              fill="none"
              stroke={slice.color}
              strokeWidth={STROKE_WIDTH}
              strokeDasharray={arc.dashArray}
              strokeDashoffset={arc.dashOffset}
            />
          );
        })}
      </svg>
      <figcaption
        className={
          legend === "side"
            ? undefined
            : "absolute inset-0 grid place-items-center"
        }
      >
        <ul className="space-y-0.5 text-xs text-ink">
          {layout.segments.map((slice) => (
            <li key={slice.id} className="flex items-center gap-1.5">
              <i
                aria-hidden
                className="size-2.5 rounded-full"
                style={{ background: slice.color }}
              />
              <span>{slice.label}</span>
              <strong className="ml-auto pl-1.5 font-semibold tabular-nums">
                {formatQuantity(slice.value, "count")}
              </strong>
            </li>
          ))}
        </ul>
      </figcaption>
    </figure>
  );
}
