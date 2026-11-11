import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { GlassPanel } from "@/components/ui/glass-panel";
import { TONE_TEXT } from "@/components/ui/status-badge";
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

/**
 * Floating callout with a tone accent ("Run outcomes · 585 passed"). The tone
 * colours the accent and icon; the label text carries the meaning.
 */
export function StatCallout({
  icon: Icon,
  label,
  value,
  detail,
  tone = "success",
  className,
}: {
  icon: LucideIcon;
  label: string;
  value: React.ReactNode;
  /** Qualifier the figure needs, e.g. "Observed, not causal". */
  detail?: React.ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <GlassPanel
      tone={tone}
      padding="sm"
      className={cn("flex items-start gap-3 rounded-2xl pr-4", className)}
    >
      <Icon
        className={cn("mt-0.5 size-5 shrink-0", TONE_TEXT[tone])}
        aria-hidden
      />
      <div className="min-w-0">
        <p className="text-[13px] text-ink-muted">{label}</p>
        <p className="text-base leading-snug font-semibold tracking-tight text-ink tabular-nums">
          {value}
        </p>
        {detail && (
          <p className="mt-0.5 text-[11px] leading-snug text-ink-muted">
            {detail}
          </p>
        )}
      </div>
    </GlassPanel>
  );
}

export interface StatStripItem {
  id: string;
  icon: LucideIcon;
  /** Emphasised figure, e.g. "598". */
  value: React.ReactNode;
  /** Trailing text, e.g. "runs in scope". */
  label: React.ReactNode;
  tone?: Tone;
  /** Two-line form: small caption above the value ("Quality gate / Passed"). */
  caption?: string;
}

/** Summary strip with dividers ("598 runs in scope | 585 passed | 13 failed"). */
export function StatStrip({
  items,
  label,
  tone,
  className,
}: {
  items: readonly StatStripItem[];
  /** Accessible name for the group. */
  label: string;
  tone?: Tone;
  className?: string;
}) {
  return (
    <GlassPanel
      as="section"
      aria-label={label}
      padding="none"
      className={cn(
        "rounded-2xl",
        tone === "success" &&
          "border-success-line bg-linear-to-r from-success-soft to-transparent",
        className,
      )}
    >
      <ul className="flex flex-wrap items-stretch">
        {items.map((item) => (
          <li
            key={item.id}
            className="flex min-w-[11rem] flex-1 items-center gap-3 px-4 py-3 sm:border-l sm:border-line sm:first:border-l-0"
          >
            <item.icon
              className={cn(
                "size-5 shrink-0",
                item.tone ? TONE_TEXT[item.tone] : "text-ink",
              )}
              aria-hidden
            />
            {item.caption ? (
              <span className="flex flex-col leading-tight">
                <span className="text-sm text-ink-muted">{item.caption}</span>
                <span className="text-lg font-semibold tracking-tight text-ink tabular-nums">
                  {item.value}
                </span>
              </span>
            ) : (
              <span className="text-[15px] text-ink">
                <strong className="font-semibold tabular-nums">
                  {item.value}
                </strong>{" "}
                {item.label}
              </span>
            )}
          </li>
        ))}
      </ul>
    </GlassPanel>
  );
}
