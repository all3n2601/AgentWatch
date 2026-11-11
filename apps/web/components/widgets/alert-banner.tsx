import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { TONE_ICONS, TONE_TEXT } from "@/components/ui/status-badge";
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

const BANNER_SURFACE: Record<Tone, string> = {
  success: "border-success-line from-success-soft",
  warning: "border-warning-line from-warning-soft",
  caution: "border-caution-line from-caution-soft",
  danger: "border-danger-line from-danger-soft",
  info: "border-info-line from-info-soft",
};

/**
 * Headline finding for a page ("Mixed contention produced the largest observed
 * slowdown"). Use `detail` for the qualifier the claim needs, such as
 * "Observed, not causal".
 */
export function AlertBanner({
  tone,
  title,
  detail,
  aside,
  icon,
  headingLevel: Heading = "h2",
  className,
}: {
  tone: Tone;
  title: React.ReactNode;
  detail?: React.ReactNode;
  /** Secondary statement shown after a divider on wide screens. */
  aside?: React.ReactNode;
  icon?: LucideIcon;
  headingLevel?: "h2" | "h3";
  className?: string;
}) {
  const Icon = icon ?? TONE_ICONS[tone];
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className={cn(
        "glass-panel flex flex-wrap items-center gap-x-5 gap-y-2 rounded-2xl border-l-4 bg-linear-to-r to-glass px-5 py-4",
        BANNER_SURFACE[tone],
        className,
      )}
    >
      <Icon className={cn("size-8 shrink-0", TONE_TEXT[tone])} aria-hidden />
      <div className="min-w-0 flex-1">
        <Heading className="text-xl leading-tight font-semibold tracking-tight text-ink sm:text-2xl">
          {title}
        </Heading>
        {detail && <p className="mt-1 text-[15px] text-ink-muted">{detail}</p>}
      </div>
      {aside && (
        <p className="text-[15px] text-ink-muted sm:border-l sm:border-line sm:pl-5">
          {aside}
        </p>
      )}
    </div>
  );
}
