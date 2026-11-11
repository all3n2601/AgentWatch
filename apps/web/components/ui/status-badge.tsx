import * as React from "react";
import { cva } from "class-variance-authority";
import {
  CircleAlert,
  CircleCheck,
  CircleX,
  Info,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { statusTone, type Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

export const TONE_ICONS: Record<Tone, LucideIcon> = {
  success: CircleCheck,
  warning: TriangleAlert,
  caution: TriangleAlert,
  danger: CircleX,
  info: Info,
};

export const TONE_TEXT: Record<Tone, string> = {
  success: "text-success",
  warning: "text-warning",
  caution: "text-caution",
  danger: "text-danger",
  info: "text-info",
};

/** Text, soft background and border classes per tone, for any tinted surface. */
export const TONE_SURFACE: Record<Tone, string> = {
  success: "text-success bg-success-soft border-success-line",
  warning: "text-warning bg-warning-soft border-warning-line",
  caution: "text-caution bg-caution-soft border-caution-line",
  danger: "text-danger bg-danger-soft border-danger-line",
  info: "text-info bg-info-soft border-info-line",
};

export const TONE_DOT: Record<Tone, string> = {
  success: "bg-success-solid",
  warning: "bg-warning-solid",
  caution: "bg-caution-solid",
  danger: "bg-danger-solid",
  info: "bg-info-solid",
};

const statusBadgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center gap-1.5 font-medium whitespace-nowrap [&_svg]:shrink-0",
  {
    variants: {
      appearance: {
        // Tinted pill with an icon: "⚠ Degraded".
        soft: "rounded-full border px-2.5 py-0.5 text-[13px]",
        // Tinted pill with a status dot: "● Degraded".
        dot: "rounded-full border px-2.5 py-0.5 text-[13px]",
        // Icon and label only, for dense tables.
        plain: "text-sm text-ink [&_svg]:size-[18px]",
      },
    },
    defaultVariants: { appearance: "soft" },
  },
);

type StatusBadgeProps = Omit<React.ComponentProps<"span">, "children"> & {
  label: string;
  /** Defaults to the tone for `label` (see `statusTone`). */
  tone?: Tone;
  icon?: LucideIcon;
  appearance?: "soft" | "dot" | "plain";
};

/** Status is always icon or dot plus text, never colour alone. */
function StatusBadge({
  label,
  tone,
  icon,
  appearance = "soft",
  className,
  ...props
}: StatusBadgeProps) {
  const resolvedTone = tone ?? statusTone(label);
  const Icon = icon ?? TONE_ICONS[resolvedTone];
  return (
    <span
      data-slot="status-badge"
      data-tone={resolvedTone}
      className={cn(
        statusBadgeVariants({ appearance }),
        appearance !== "plain" && TONE_SURFACE[resolvedTone],
        className,
      )}
      {...props}
    >
      {appearance === "dot" ? (
        <i
          className={cn("size-2 rounded-full", TONE_DOT[resolvedTone])}
          aria-hidden
        />
      ) : (
        <Icon
          className={cn(
            "size-3.5",
            appearance === "plain" && TONE_TEXT[resolvedTone],
          )}
          aria-hidden
        />
      )}
      {label}
    </span>
  );
}

/** Icon-only status mark, for list rows where the row text carries the label. */
function StatusIcon({
  tone,
  label,
  className,
}: {
  tone: Tone;
  label: string;
  className?: string;
}) {
  const Icon = tone === "warning" ? CircleAlert : TONE_ICONS[tone];
  return (
    <Icon
      role="img"
      aria-label={label}
      className={cn("size-5 shrink-0", TONE_TEXT[tone], className)}
    />
  );
}

export { StatusBadge, StatusIcon, statusBadgeVariants };
