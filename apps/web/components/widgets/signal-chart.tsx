"use client";

import * as React from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { tickLabel, type TimelineTick } from "@/lib/run-timeline";
import { formatSignal, latestReading } from "@/lib/signal-format";
import { cn } from "@/lib/utils";

export interface SignalPoint {
  /** Seconds from the run's origin. */
  seconds: number;
  /** Null for a reading that was not measured; drawn as a gap, never as zero. */
  value: number | null;
}

export interface SignalBand {
  id: string;
  from: number;
  to: number;
}

/** Step windows are context, not data, so the lane is neutral ink, not a series hue. */
export const BAND_COLOR = "var(--aw-ink-subtle)";

/** Plot-area insets shared by the chart and the band lane, so both use one x scale. */
const Y_AXIS_WIDTH = 40;
const MARGIN = { top: 8, right: 12, bottom: 0, left: 0 };

function SignalTooltip({
  active,
  payload,
  label,
  unit,
}: {
  active?: boolean;
  payload?: readonly { value?: unknown }[];
  label?: unknown;
  unit: string;
}) {
  if (!active || !payload?.length) return null;
  const value = payload[0]?.value;
  return (
    <div className="glass-raised rounded-xl px-3 py-2 text-sm text-ink shadow-lg">
      <p className="text-xs text-ink-muted tabular-nums">
        {Number(label).toFixed(2)} s
      </p>
      <p className="font-semibold tabular-nums">
        {typeof value === "number" ? formatSignal(value, unit) : "Not measured"}
      </p>
    </div>
  );
}

/**
 * One resource signal over the run: a 2px line over a 10% wash, round time ticks shared with
 * the step timeline, a crosshair tooltip, and an optional lane under the plot marking step
 * windows (model calls). Missing readings are gaps. The latest reading is printed at the
 * top right; the readings table carries every value for readers who cannot use the chart.
 */
export function SignalChart({
  title,
  label,
  unit,
  color,
  points,
  spanSeconds,
  ticks,
  bands = [],
  bandLabel,
  meta,
  footer,
  className,
}: {
  title: string;
  /** Accessible name when the title alone is ambiguous (several resources); defaults to it. */
  label?: string;
  unit: string;
  /** The series' mark colour; text always uses ink tokens. */
  color: string;
  points: readonly SignalPoint[];
  spanSeconds: number;
  ticks: readonly TimelineTick[];
  bands?: readonly SignalBand[];
  /** Names the band lane for assistive tech, e.g. "Model calls". */
  bandLabel?: string;
  /** Source and resource line under the title. */
  meta?: React.ReactNode;
  /** Notes under the chart (not-measured reasons). */
  footer?: React.ReactNode;
  className?: string;
}) {
  const gradientId = React.useId().replaceAll(":", "");
  const data = points.map((point) => ({ ...point }));
  const latest = latestReading(points);
  const measured = points.filter((point) => point.value !== null);
  // A reading with no measured neighbour has no line segment, so it gets a dot to stay visible.
  const isolated = new Set(
    points.flatMap((point, index) =>
      point.value !== null &&
      points[index - 1]?.value == null &&
      points[index + 1]?.value == null
        ? [index]
        : [],
    ),
  );

  return (
    <figure
      className={cn("glass-inset min-w-0 p-3 sm:p-4", className)}
      aria-label={`${label ?? title}, ${measured.length} measured readings`}
    >
      <figcaption className="mb-2 flex items-start justify-between gap-3">
        <span className="min-w-0">
          <span className="block text-sm font-semibold text-ink">{title}</span>
          {meta && (
            <span className="mt-0.5 block truncate text-xs text-ink-muted">
              {meta}
            </span>
          )}
        </span>
        <span className="shrink-0 text-right">
          <span className="block text-xs text-ink-muted">Latest</span>
          <span className="flex items-center justify-end gap-1.5 text-base font-semibold text-ink">
            <i
              aria-hidden
              className="size-2 rounded-full"
              style={{ background: color }}
            />
            {latest === null ? "—" : formatSignal(latest.value, unit)}
          </span>
        </span>
      </figcaption>

      <div className="h-28" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={MARGIN}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.18} />
                <stop offset="100%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid
              vertical={false}
              stroke="var(--aw-line)"
              strokeDasharray="0"
            />
            <XAxis
              dataKey="seconds"
              type="number"
              domain={[0, spanSeconds]}
              ticks={ticks.map((tick) => tick.seconds)}
              tickFormatter={(seconds: number) =>
                tickLabel({ seconds, share: 0 }, ticks)
              }
              allowDataOverflow
              axisLine={false}
              tickLine={false}
              tick={{ fill: "var(--aw-ink-muted)", fontSize: 11 }}
              height={20}
            />
            <YAxis
              width={Y_AXIS_WIDTH}
              domain={[0, "auto"]}
              tickCount={3}
              allowDecimals
              axisLine={false}
              tickLine={false}
              tick={{ fill: "var(--aw-ink-muted)", fontSize: 11 }}
              // Ticks carry the unit only when it is one character ("%"); the title line
              // names longer units, which would wrap in the axis column.
              tickFormatter={(value: number) =>
                formatSignal(value, unit.length === 1 ? unit : "", true)
              }
            />
            <Tooltip
              content={(props) => (
                <SignalTooltip
                  active={props.active}
                  payload={props.payload}
                  label={props.label}
                  unit={unit}
                />
              )}
              cursor={{ stroke: "var(--aw-ink-subtle)", strokeWidth: 1 }}
              isAnimationActive={false}
            />
            <Area
              dataKey="value"
              // Straight segments: samples are discrete, and a smoothed curve would draw
              // values between them that were never read.
              type="linear"
              stroke={color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              fill={`url(#${gradientId})`}
              connectNulls={false}
              isAnimationActive={false}
              dot={(props: { cx?: number; cy?: number; index?: number }) =>
                props.index !== undefined &&
                (isolated.has(props.index) || props.index === latest?.index) &&
                props.cx !== undefined &&
                props.cy !== undefined ? (
                  <circle
                    key={props.index}
                    cx={props.cx}
                    cy={props.cy}
                    r={4}
                    fill={color}
                    stroke="var(--aw-surface)"
                    strokeWidth={2}
                  />
                ) : (
                  <g key={props.index} />
                )
              }
              activeDot={{
                r: 5,
                fill: color,
                stroke: "var(--aw-surface)",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {bands.length > 0 && (
        // Same insets as the plot area, so a band sits under the samples it overlaps.
        <div
          role="img"
          aria-label={`${bandLabel ?? "Steps"}: ${bands.length} windows`}
          title={bandLabel}
          className="relative mt-1 h-1.5 rounded-full bg-bar-track"
          style={{ marginLeft: Y_AXIS_WIDTH, marginRight: MARGIN.right }}
        >
          {bands.map((band) => {
            const from = Math.max(band.from, 0);
            const to = Math.min(band.to, spanSeconds);
            return (
              <span
                key={band.id}
                className="absolute inset-y-0 rounded-full"
                style={{
                  left: `${(from / spanSeconds) * 100}%`,
                  width: `${(Math.max(to - from, 0) / spanSeconds) * 100}%`,
                  minWidth: 2,
                  background: BAND_COLOR,
                }}
              />
            );
          })}
        </div>
      )}

      {footer}
      <details className="group mt-1 text-xs">
        <summary className="inline-flex min-h-8 cursor-pointer items-center rounded-md text-ink-muted outline-none hover:text-ink focus-visible:ring-[3px] focus-visible:ring-brand/40">
          Show {points.length} readings
        </summary>
        <div className="mt-1 max-h-40 overflow-y-auto rounded-lg border border-line">
          <table className="w-full text-left tabular-nums">
            <caption className="sr-only">{title} readings</caption>
            <thead className="sticky top-0 bg-glass-strong text-ink-muted">
              <tr>
                <th scope="col" className="px-2 py-1 font-medium">
                  Time
                </th>
                <th scope="col" className="px-2 py-1 text-right font-medium">
                  Value
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line text-ink">
              {points.map((point, index) => (
                <tr key={index}>
                  <td className="px-2 py-1">{point.seconds.toFixed(2)} s</td>
                  <td className="px-2 py-1 text-right">
                    {point.value === null
                      ? "Not measured"
                      : formatSignal(point.value, unit)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
