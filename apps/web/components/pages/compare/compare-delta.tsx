import { DeltaChip } from "@/components/widgets/delta-chip";
import { Measurement } from "@/components/widgets/measurement";
import type { TimingDelta } from "@/lib/compare-page";
import { duration } from "@/lib/dashboard-data";
import { cn } from "@/lib/utils";

/**
 * Candidate minus baseline. DeltaChip prints sub-second differences in ms/µs
 * and keeps the absolute change when a measured-zero baseline has no ratio.
 */
export function CompareDelta({
  delta,
  className,
}: {
  delta: TimingDelta;
  className?: string;
}) {
  return (
    <DeltaChip
      deltaSeconds={delta.deltaSeconds}
      ratio={delta.ratio}
      className={className}
    />
  );
}

/**
 * One duration with the dashboard's precision (μs / ms / s). A missing reading
 * renders through Measurement as a dash plus "Unavailable", never as 0.
 */
export function TimingText({
  seconds,
  className,
}: {
  seconds: number | null;
  className?: string;
}) {
  if (seconds === null || !Number.isFinite(seconds))
    return (
      <Measurement value={null} unit="s" showState className={className} />
    );
  return (
    <span className={cn("tabular-nums", className)}>{duration(seconds)}</span>
  );
}
