import { ChartScatter, Table2 } from "lucide-react";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import {
  conditionShortName,
  ratioAxis,
  type ReplaySequence,
  type SequenceSlot,
} from "@/lib/experiments-page";
import { formatQuantity } from "@/lib/measurement-format";
import { cn } from "@/lib/utils";
import { conditionColor, roleName } from "./format";

// Chart geometry in viewBox units; the SVG scales with its container.
const WIDTH = 640;
const HEIGHT = 250;
const MARGIN = { top: 26, right: 12, bottom: 44, left: 44 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;
const MARK_RADIUS = 5;

const ratio = (value: number) => formatQuantity(value, "ratio");
const isClean = (slot: SequenceSlot) => slot.condition === "clean";

/** Baseline: filled circle; clean control: ring; injected: diamond. */
function Mark({ slot, x, y }: { slot: SequenceSlot; x: number; y: number }) {
  if (!isClean(slot)) {
    const r = MARK_RADIUS + 1.5;
    return (
      <path
        d={`M ${x} ${y - r} L ${x + r} ${y} L ${x} ${y + r} L ${x - r} ${y} Z`}
        fill={conditionColor(slot.condition)}
        stroke="var(--aw-ink)"
        strokeWidth={1}
      />
    );
  }
  return slot.role === "clean_control" ? (
    <circle
      cx={x}
      cy={y}
      r={MARK_RADIUS}
      fill="none"
      stroke={conditionColor("clean")}
      strokeWidth={2}
    />
  ) : (
    <circle cx={x} cy={y} r={MARK_RADIUS} fill={conditionColor("clean")} />
  );
}

function LegendMark({ kind }: { kind: "baseline" | "control" | "injected" }) {
  return (
    <svg viewBox="0 0 14 14" className="size-3.5" aria-hidden>
      {kind === "injected" ? (
        <path d="M 7 1 L 13 7 L 7 13 L 1 7 Z" fill="var(--aw-ink-muted)" />
      ) : kind === "control" ? (
        <circle
          cx={7}
          cy={7}
          r={5}
          fill="none"
          stroke={conditionColor("clean")}
          strokeWidth={2}
        />
      ) : (
        <circle cx={7} cy={7} r={5} fill={conditionColor("clean")} />
      )}
    </svg>
  );
}

function slotLabel(slot: SequenceSlot) {
  return slot.role ? roleName(slot.role) : conditionShortName(slot.condition);
}

/**
 * The replay order of one experiment, with each position's replay time over its
 * experiment's clean median: median dot and min–max line across experiments.
 */
export function ReplaySequencePanel({
  sequence,
  experiments,
  className,
}: {
  sequence: ReplaySequence;
  experiments: number;
  className?: string;
}) {
  const { slots } = sequence;
  const axis = ratioAxis(
    Math.max(...slots.map((slot) => slot.maxRatio ?? slot.medianRatio ?? 0)),
  );
  const step = PLOT_WIDTH / slots.length;
  const xOf = (index: number) => MARGIN.left + (index + 0.5) * step;
  const yOf = (value: number) =>
    MARGIN.top + PLOT_HEIGHT - (value / axis.max) * PLOT_HEIGHT;
  const hasRoles = slots.some((slot) => slot.role !== null);

  return (
    <GlassPanel
      as="section"
      aria-labelledby="replay-sequence"
      className={cn("min-w-0", className)}
    >
      <PanelHeader
        titleId="replay-sequence"
        icon={ChartScatter}
        title="Replay sequence"
        description={`Each replay's total step time over its experiment's clean median, in replay order. Dot: median of ${formatQuantity(sequence.matching, "count")} experiments; line: their range.`}
        actions={
          <ul
            aria-label="Legend"
            className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-muted"
          >
            {hasRoles && (
              <li className="flex items-center gap-1.5">
                <LegendMark kind="baseline" /> Baseline (clean)
              </li>
            )}
            <li className="flex items-center gap-1.5">
              <LegendMark kind="control" />{" "}
              {hasRoles ? "Clean control" : "Clean"}
            </li>
            <li className="flex items-center gap-1.5">
              <LegendMark kind="injected" /> Injected contention
            </li>
            <li className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="w-4 border-t-2 border-dashed border-ink-muted"
              />
              Clean median (1.0×)
            </li>
          </ul>
        }
      />
      {sequence.matching < experiments && (
        <p className="mb-2 text-sm text-ink-muted">
          {formatQuantity(experiments - sequence.matching, "count")} of{" "}
          {formatQuantity(experiments, "count")} experiments follow a different
          replay order and are not drawn.
        </p>
      )}
      <figure className="m-0">
        <div
          role="region"
          aria-label="Replay sequence chart"
          tabIndex={0}
          className="relative overflow-x-auto rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
        >
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="h-auto w-full min-w-[32rem]"
            role="img"
            aria-label="Replay time over clean median by replay position. Values are listed in the Replay sequence values table."
          >
            {axis.ticks.map((tick) => (
              <g key={tick}>
                <line
                  x1={MARGIN.left}
                  x2={WIDTH - MARGIN.right}
                  y1={yOf(tick)}
                  y2={yOf(tick)}
                  stroke="var(--aw-line)"
                  strokeWidth={1}
                />
                <text
                  x={MARGIN.left - 6}
                  y={yOf(tick)}
                  dy="0.32em"
                  textAnchor="end"
                  fontSize={11}
                  fill="var(--aw-ink-muted)"
                >
                  {ratio(tick)}
                </text>
              </g>
            ))}
            <line
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={yOf(1)}
              y2={yOf(1)}
              stroke="var(--aw-ink-muted)"
              strokeDasharray="5 4"
              strokeWidth={1.25}
            />
            {slots.map((slot, i) => {
              const x = xOf(i);
              return (
                <g key={slot.index}>
                  {slot.minRatio !== null && slot.maxRatio !== null && (
                    <line
                      x1={x}
                      x2={x}
                      y1={yOf(slot.minRatio)}
                      y2={yOf(slot.maxRatio)}
                      stroke={conditionColor(slot.condition)}
                      strokeOpacity={0.6}
                      strokeWidth={2}
                      strokeLinecap="round"
                    />
                  )}
                  {slot.medianRatio !== null && (
                    <Mark slot={slot} x={x} y={yOf(slot.medianRatio)} />
                  )}
                  {!isClean(slot) && slot.medianRatio !== null && (
                    <text
                      x={x}
                      y={yOf(slot.maxRatio ?? slot.medianRatio) - 8}
                      textAnchor="middle"
                      fontSize={11}
                      fontWeight={600}
                      fill="var(--aw-ink)"
                    >
                      {ratio(slot.medianRatio)}
                    </text>
                  )}
                  <text
                    x={x}
                    y={HEIGHT - MARGIN.bottom + 16}
                    textAnchor="middle"
                    fontSize={11}
                    fill="var(--aw-ink-muted)"
                  >
                    {slot.index}
                  </text>
                  {!isClean(slot) && (
                    <text
                      x={x}
                      y={HEIGHT - MARGIN.bottom + 30}
                      textAnchor="middle"
                      fontSize={10.5}
                      fontWeight={600}
                      fill="var(--aw-ink)"
                    >
                      {conditionShortName(slot.condition)}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
        <figcaption className="mt-1 text-center text-xs text-ink-muted">
          Replay position within an experiment
        </figcaption>
      </figure>
    </GlassPanel>
  );
}

/**
 * The chart's values as a table, so no reading depends on the chart alone. On
 * fit screens the panel fills its column and the rows scroll inside it.
 */
export function ReplaySequenceValues({
  sequence,
  className,
}: {
  sequence: ReplaySequence;
  className?: string;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="replay-sequence-values"
      className={cn(
        "min-w-0 fit:flex fit:h-full fit:min-h-0 fit:flex-col",
        className,
      )}
    >
      <PanelHeader
        titleId="replay-sequence-values"
        icon={Table2}
        title="Replay sequence values"
        description="Replay time over the clean median, by position."
      />
      <div
        role="region"
        aria-label="Replay sequence values table"
        tabIndex={0}
        className="relative -mx-1 overflow-auto rounded-xl px-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:min-h-0 fit:flex-1"
      >
        <table className="w-full min-w-[18rem] text-left text-sm text-ink">
          <thead className="sticky top-0 z-10 bg-glass-strong text-[13px] text-ink-muted backdrop-blur">
            <tr className="border-b border-line">
              <th scope="col" className="py-1.5 pr-3 font-medium">
                Position
              </th>
              <th scope="col" className="py-1.5 pr-3 font-medium">
                Replay
              </th>
              <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                Median vs clean
              </th>
              <th scope="col" className="py-1.5 text-right font-medium">
                Range
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line whitespace-nowrap tabular-nums">
            {sequence.slots.map((slot) => (
              <tr key={slot.index}>
                <th scope="row" className="py-1.5 pr-3 font-normal">
                  {slot.index}
                </th>
                <td className="py-1.5 pr-3">
                  {isClean(slot)
                    ? slotLabel(slot)
                    : `${conditionShortName(slot.condition)} contention`}
                </td>
                <td className="py-1.5 pr-3 text-right">
                  {slot.medianRatio === null ? "—" : ratio(slot.medianRatio)}
                </td>
                <td className="py-1.5 text-right text-ink-muted">
                  {slot.minRatio === null || slot.maxRatio === null
                    ? "—"
                    : `${ratio(slot.minRatio)} – ${ratio(slot.maxRatio)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </GlassPanel>
  );
}
