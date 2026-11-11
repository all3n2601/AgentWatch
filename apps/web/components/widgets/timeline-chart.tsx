"use client";

import * as React from "react";
import {
  tickLabel,
  type TimelineSegment,
  type TimelineTick,
} from "@/lib/run-timeline";
import { cn } from "@/lib/utils";

export interface TimelineChartRow {
  id: string;
  label: React.ReactNode;
  detail?: React.ReactNode;
  /** Bar start as a share of the window. */
  startShare: number;
  segments: readonly TimelineSegment[];
  /** Right-hand reading, e.g. the step's duration. */
  value: React.ReactNode;
  /** Still running: the bar pulses and reaches the window end. */
  open?: boolean;
  /** Failed or cancelled: the bar is drawn in the danger colour (the status says so in text). */
  failed?: boolean;
  /** Screen-reader text for the bar: start, split and state. */
  description: string;
}

const SEGMENT_FILL: Record<TimelineSegment["kind"], string> = {
  work: "bg-bar",
  wait: "hatch",
};

/** Smallest bar drawn, so a sub-pixel step stays visible and clickable. */
const MIN_BAR_PX = 3;

function Bar({ row }: { row: TimelineChartRow }) {
  const total = row.segments.reduce((sum, segment) => sum + segment.share, 0);
  return (
    <span
      aria-hidden
      className={cn(
        "absolute inset-y-1 flex gap-px overflow-hidden rounded-[5px]",
        row.open && "motion-safe:animate-pulse",
        row.failed && "ring-2 ring-danger-solid",
      )}
      style={{
        left: `${Math.min(row.startShare, 1) * 100}%`,
        width: `${Math.min(total, 1 - row.startShare) * 100}%`,
        minWidth: MIN_BAR_PX,
      }}
    >
      {row.segments.map((segment, index) =>
        segment.share > 0 ? (
          <span
            key={index}
            className={cn(
              "h-full",
              SEGMENT_FILL[segment.kind],
              row.failed && segment.kind === "work" && "bg-danger-solid/70",
              row.open && segment.kind === "work" && "bg-info-solid",
            )}
            style={{
              flexGrow: total > 0 ? segment.share / total : 1,
              flexBasis: 0,
              minWidth: 1,
            }}
          />
        ) : null,
      )}
    </span>
  );
}

function Gridlines({ ticks }: { ticks: readonly TimelineTick[] }) {
  return (
    <>
      {ticks.map((tick) => (
        <span
          key={tick.seconds}
          aria-hidden
          className="absolute inset-y-0 w-px bg-line"
          style={{ left: `${tick.share * 100}%` }}
        />
      ))}
    </>
  );
}

/**
 * Narrow containers put each bar on its own line under the step name, so the track keeps
 * the panel's full width; from `@xl` the name, bar and value share one line.
 */
const COLUMNS =
  "grid items-center gap-x-3 grid-cols-[1.5rem_minmax(0,1fr)_4.5rem] @xl:grid-cols-[1.5rem_minmax(5.5rem,9rem)_minmax(0,1fr)_4.5rem]";
const LABEL_CELL = "col-start-2 row-start-1 min-w-0";
const TRACK_CELL =
  "col-span-2 col-start-2 row-start-2 @xl:col-span-1 @xl:col-start-3 @xl:row-start-1";
const VALUE_CELL = "col-start-3 row-start-1 @xl:col-start-4";

/**
 * The run as a Gantt chart: one numbered row per step in time order on a shared axis,
 * hatched queue then solid execution. Rows are buttons when `onSelect` is given. Each bar
 * carries a text description, so the chart never relies on its geometry alone.
 */
export function TimelineChart({
  rows,
  ticks,
  label,
  selectedId,
  onSelect,
  className,
}: {
  rows: readonly TimelineChartRow[];
  ticks: readonly TimelineTick[];
  /** Accessible name for the step list. */
  label: string;
  selectedId?: string;
  onSelect?: (id: string) => void;
  className?: string;
}) {
  const last = ticks.length - 1;
  return (
    <div className={cn("@container min-w-0", className)}>
      <div className={cn(COLUMNS, "px-3 pb-1.5")} aria-hidden>
        <span
          className={cn(
            TRACK_CELL,
            "relative row-start-1 h-4 text-xs text-ink-muted tabular-nums",
          )}
        >
          {ticks.map((tick, index) => (
            <span
              key={tick.seconds}
              className={cn(
                "absolute top-0 whitespace-nowrap",
                index === 0
                  ? "translate-x-0"
                  : index === last && tick.share > 0.97
                    ? "-translate-x-full"
                    : "-translate-x-1/2",
                // Crowded middle labels drop on narrow tracks; the first and last stay.
                index !== 0 && index !== last && "hidden @sm:inline",
              )}
              style={{ left: `${tick.share * 100}%` }}
            >
              {tickLabel(tick, ticks)}
            </span>
          ))}
        </span>
      </div>
      <ol aria-label={label} className="space-y-1">
        {rows.map((row, index) => {
          const selected = row.id === selectedId;
          const cells = (
            <>
              <span className="col-start-1 row-start-1 text-sm text-ink-muted tabular-nums">
                {index + 1}.
              </span>
              <span className={LABEL_CELL}>
                <span className="block truncate text-sm font-medium">
                  {row.label}
                </span>
                {row.detail && (
                  <span className="block text-xs text-ink-muted [overflow-wrap:anywhere]">
                    {row.detail}
                  </span>
                )}
              </span>
              <span
                className={cn(
                  TRACK_CELL,
                  "relative mt-1.5 h-6 rounded-md bg-bar-track/60 @xl:mt-0 @xl:h-7",
                )}
              >
                <Gridlines ticks={ticks} />
                <Bar row={row} />
              </span>
              <span
                className={cn(
                  VALUE_CELL,
                  "text-right text-sm font-semibold tabular-nums",
                )}
              >
                {row.value}
              </span>
              <span className="sr-only">{row.description}</span>
            </>
          );
          const rowClass = cn(
            COLUMNS,
            "relative min-h-11 w-full rounded-xl border border-transparent px-3 py-1.5 text-left text-ink",
          );
          return (
            <li key={row.id}>
              {onSelect ? (
                <button
                  type="button"
                  aria-pressed={selected}
                  onClick={() => onSelect(row.id)}
                  className={cn(
                    rowClass,
                    "outline-none transition-colors hover:bg-glass-strong focus-visible:ring-[3px] focus-visible:ring-brand/40",
                    selected && "border-success-line bg-brand-soft shadow-glow",
                  )}
                >
                  {cells}
                </button>
              ) : (
                <div className={rowClass}>{cells}</div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
