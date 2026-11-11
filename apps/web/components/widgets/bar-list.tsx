import * as React from "react";
import type { LucideIcon } from "lucide-react";
import {
  formatQuantity,
  MISSING_VALUE,
  type MeasurementUnit,
} from "@/lib/measurement-format";
import { relativeLengths } from "@/lib/segments";
import { cn } from "@/lib/utils";

export interface BarListItem {
  id: string;
  label: string;
  value: number;
  icon?: LucideIcon;
  /** Leading status icon or other marker. */
  leading?: React.ReactNode;
  /** CSS colour for the bar; defaults to the neutral bar token. */
  color?: string;
}

/**
 * Labelled horizontal bars scaled to the largest value ("Call types",
 * "Resource labels"). The numbers are text, so the bars are decorative.
 */
export function BarList({
  items,
  unit = "count",
  label,
  className,
}: {
  items: readonly BarListItem[];
  unit?: MeasurementUnit;
  /** Accessible name for the list. */
  label: string;
  className?: string;
}) {
  const lengths = relativeLengths(items.map((item) => item.value));
  return (
    <ul aria-label={label} className={cn("space-y-2.5", className)}>
      {items.map((item, index) => {
        const length = lengths[index];
        return (
          <li
            key={item.id}
            className="grid grid-cols-[minmax(0,6.5rem)_minmax(2rem,1fr)_3rem] items-center gap-x-3 text-sm text-ink"
          >
            <span className="flex min-w-0 items-center gap-2.5">
              {item.leading}
              {item.icon && (
                <item.icon className="size-4 shrink-0" aria-hidden />
              )}
              <span className="truncate">{item.label}</span>
            </span>
            <span
              className="h-2.5 overflow-hidden rounded-full bg-bar-track"
              aria-hidden
            >
              {length !== null && (
                <span
                  className="block h-full rounded-full bg-bar"
                  style={{
                    width: `${length * 100}%`,
                    background: item.color,
                  }}
                />
              )}
            </span>
            <span className="text-right font-medium tabular-nums">
              {length === null
                ? MISSING_VALUE
                : formatQuantity(item.value, unit)}
              {length === null && <span className="sr-only">Unavailable</span>}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
