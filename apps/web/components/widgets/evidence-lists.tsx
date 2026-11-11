import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { StatusBadge, StatusIcon } from "@/components/ui/status-badge";
import type { Tone } from "@/lib/tone";
import { cn } from "@/lib/utils";

export interface KeyValueItem {
  id: string;
  label: string;
  value: React.ReactNode;
  icon?: LucideIcon;
  /** Optional status pill after the value ("Match"). */
  status?: { label: string; tone?: Tone };
}

/**
 * Evidence and configuration rows ("Queue · 69 ms", "Workload · Match"), as a
 * description list so each value is announced with its label.
 */
export function KeyValueList({
  items,
  className,
  dense = false,
}: {
  items: readonly KeyValueItem[];
  className?: string;
  dense?: boolean;
}) {
  return (
    <dl className={cn("divide-y divide-line", className)}>
      {items.map((item) => (
        <div
          key={item.id}
          className={cn(
            "grid grid-cols-[auto_1fr_auto] items-center gap-x-4 text-sm",
            dense ? "py-1.5" : "py-2.5",
          )}
        >
          <dt className="flex items-center gap-3 text-ink">
            {item.icon && <item.icon className="size-5 shrink-0" aria-hidden />}
            {item.label}
          </dt>
          <dd
            className={cn(
              "text-right text-ink tabular-nums",
              !item.status && "col-span-2",
            )}
          >
            {item.value}
          </dd>
          {item.status && (
            <dd>
              <StatusBadge label={item.status.label} tone={item.status.tone} />
            </dd>
          )}
        </div>
      ))}
    </dl>
  );
}

export type CheckState = "pass" | "warn" | "fail" | "unknown";

const CHECK_TONES: Record<CheckState, Tone> = {
  pass: "success",
  warn: "warning",
  fail: "danger",
  unknown: "info",
};

const CHECK_LABELS: Record<CheckState, string> = {
  pass: "Passed",
  warn: "Needs attention",
  fail: "Failed",
  unknown: "Not checked",
};

export interface ChecklistItem {
  id: string;
  label: string;
  state: CheckState;
  detail?: string;
}

/** Acceptance and validation checks, with an explicit "not checked" state. */
export function Checklist({
  items,
  showStateBadge = false,
  className,
}: {
  items: readonly ChecklistItem[];
  /** Show a "Passed" pill at the row end (Validation panel). */
  showStateBadge?: boolean;
  className?: string;
}) {
  return (
    <ul className={cn("divide-y divide-line", className)}>
      {items.map((item) => (
        <li
          key={item.id}
          className="flex items-center gap-3 py-2 text-sm text-ink"
        >
          <StatusIcon
            tone={CHECK_TONES[item.state]}
            label={CHECK_LABELS[item.state]}
          />
          <span className="min-w-0 flex-1">
            {item.label}
            {item.detail && (
              <span className="block text-sm text-ink-muted">
                {item.detail}
              </span>
            )}
          </span>
          {showStateBadge && (
            <StatusBadge
              label={CHECK_LABELS[item.state]}
              tone={CHECK_TONES[item.state]}
              className="[&_svg]:hidden"
            />
          )}
        </li>
      ))}
    </ul>
  );
}

export interface RankedItem {
  id: string;
  label: string;
  value: React.ReactNode;
}

/** Ordered findings ("Largest differences"); order is the ranking. */
export function RankedList({
  items,
  className,
}: {
  items: readonly RankedItem[];
  className?: string;
}) {
  return (
    <ol className={cn("divide-y divide-line", className)}>
      {items.map((item, index) => (
        <li
          key={item.id}
          className="flex items-center gap-4 py-2.5 text-[15px] text-ink"
        >
          <span
            aria-hidden
            className="glass-raised flex size-8 shrink-0 items-center justify-center rounded-lg text-sm font-semibold"
          >
            {index + 1}
          </span>
          <span className="min-w-0 flex-1 font-semibold">{item.label}</span>
          {item.value}
        </li>
      ))}
    </ol>
  );
}
