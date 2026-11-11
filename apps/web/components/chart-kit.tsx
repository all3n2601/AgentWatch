/** Shared chart styling so every chart reads the same in both themes. */
export const chartTooltipStyle = {
  background: "var(--chart-tooltip-bg)",
  border: "1px solid var(--chart-tooltip-border)",
  borderRadius: 8,
  boxShadow: "0 8px 24px -10px rgb(0 0 0 / 0.35)",
  fontSize: 12,
  padding: "8px 10px",
} as const;

export const chartGridStroke = "var(--chart-grid)";
export const chartAxisTick = {
  fill: "var(--chart-axis)",
  fontSize: 11,
} as const;
export const chartCursor = { fill: "var(--table-row-hover)" } as const;

/** A legend swatch with its label, for charts whose colours carry meaning. */
export function LegendKey({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <i
        className="size-2 rounded-sm"
        style={{ background: color }}
        aria-hidden
      />
      {label}
    </span>
  );
}
