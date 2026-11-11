import * as React from "react";
import { formatQuantity, MISSING_VALUE } from "@/lib/measurement-format";
import type { CountPart } from "@/lib/quality-page";
import { cn } from "@/lib/utils";

const count = (value: number) => formatQuantity(value, "count");

/** Share of a whole as text; a dash when there is no whole to divide by. */
export function shareText(part: number, whole: number) {
  return whole > 0
    ? formatQuantity((part / whole) * 100, "percent")
    : MISSING_VALUE;
}

/**
 * Step counts with their share of the stated total. Labels wrap instead of
 * truncating, since names such as "Contention, exceeded threshold" carry the
 * meaning; the bar is decorative because the numbers are text.
 */
export function CountShares({
  parts,
  total,
  label,
  className,
}: {
  parts: readonly CountPart[];
  /** The whole the shares are taken of, e.g. all steps in the release. */
  total: number;
  /** Accessible name for the list. */
  label: string;
  className?: string;
}) {
  return (
    <ul aria-label={label} className={cn("space-y-2.5", className)}>
      {parts.map((part) => (
        <li key={part.id} className="text-sm text-ink">
          <div className="flex items-baseline justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2">
              <i
                aria-hidden
                className="size-2.5 shrink-0 rounded-[3px]"
                style={{ background: part.color }}
              />
              <span className="min-w-0 break-words">{part.label}</span>
            </span>
            <span className="shrink-0 text-right tabular-nums">
              <span className="font-medium">{count(part.value)}</span>{" "}
              <span className="text-ink-muted">
                {shareText(part.value, total)}
              </span>
            </span>
          </div>
          <span
            aria-hidden
            className="mt-1 block h-2 overflow-hidden rounded-full bg-bar-track"
          >
            <span
              className="block h-full rounded-full"
              style={{
                width: total > 0 ? `${(part.value / total) * 100}%` : 0,
                background: part.color,
                minWidth: part.value > 0 ? 3 : 0,
              }}
            />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** A wide table that scrolls inside its own labelled, focusable region. */
export function ScrollTable({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      role="region"
      aria-label={label}
      tabIndex={0}
      className={cn(
        "relative max-w-full overflow-x-auto rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40",
        className,
      )}
    >
      {children}
    </div>
  );
}
