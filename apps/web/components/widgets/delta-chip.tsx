import { TONE_SURFACE } from "@/components/ui/status-badge";
import {
  formatQuantity,
  formatSeconds,
  MISSING_VALUE,
} from "@/lib/measurement-format";
import { slowdownTone } from "@/lib/tone";
import { cn } from "@/lib/utils";

export const CHIP =
  "inline-flex w-fit items-center rounded-full border px-2.5 py-0.5 text-[14px] font-semibold whitespace-nowrap tabular-nums";

/**
 * Observed slowdown against the clean baseline ("11.4×"). Emphasis follows
 * `slowdownTone`; the ratio is a timing difference, not a causal claim.
 */
export function RatioChip({
  ratio,
  className,
}: {
  ratio: number | null;
  className?: string;
}) {
  const tone = ratio === null ? null : slowdownTone(ratio);
  if (ratio === null || tone === null) {
    return (
      <span className={cn(CHIP, "border-line text-ink-subtle", className)}>
        {MISSING_VALUE}
        <span className="sr-only">No baseline ratio</span>
      </span>
    );
  }
  return (
    <span className={cn(CHIP, TONE_SURFACE[tone], className)}>
      {formatQuantity(ratio, "ratio")}
    </span>
  );
}

/**
 * Absolute and relative change ("+24.85 s (11.4×)", "+0.30 ms (1.2×)"). With
 * no ratio (a measured-zero baseline) the absolute change still shows.
 * `neutral` drops the slowdown emphasis, for differences that must not read as
 * better or worse.
 */
export function DeltaChip({
  deltaSeconds,
  ratio,
  neutral = false,
  className,
}: {
  deltaSeconds: number | null;
  ratio: number | null;
  neutral?: boolean;
  className?: string;
}) {
  if (deltaSeconds === null || !Number.isFinite(deltaSeconds)) {
    return (
      <span className={cn(CHIP, "border-line text-ink-subtle", className)}>
        {MISSING_VALUE}
        <span className="sr-only">No comparable measurement</span>
      </span>
    );
  }
  const sign = deltaSeconds > 0 ? "+" : deltaSeconds < 0 ? "−" : "±";
  const hasRatio = ratio !== null && Number.isFinite(ratio);
  const tone = hasRatio && !neutral ? slowdownTone(ratio) : null;
  return (
    <span
      className={cn(
        CHIP,
        tone ? TONE_SURFACE[tone] : "border-line text-ink",
        className,
      )}
    >
      {sign}
      {formatSeconds(Math.abs(deltaSeconds))}
      {hasRatio && ` (${formatQuantity(ratio, "ratio")})`}
      {!hasRatio && (
        <span className="sr-only"> (no ratio: baseline is zero)</span>
      )}
    </span>
  );
}
