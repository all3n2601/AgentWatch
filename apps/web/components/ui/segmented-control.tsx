"use client";

import * as React from "react";
import { RadioGroup as RadioGroupPrimitive } from "radix-ui";
import { cn } from "@/lib/utils";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

/**
 * One-of-many choice rendered as a segmented pill (Theme, Density, Refresh).
 * Built on a radio group, so arrow keys move the selection and screen readers
 * announce it as a choice.
 */
function SegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  label,
  labelledBy,
  className,
  disabled,
}: {
  value: T;
  onValueChange: (value: T) => void;
  options: readonly SegmentedOption<T>[];
  /** Accessible name when no visible label references the group. */
  label?: string;
  labelledBy?: string;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <RadioGroupPrimitive.Root
      value={value}
      onValueChange={(next) => {
        const option = options.find((candidate) => candidate.value === next);
        if (option) onValueChange(option.value);
      }}
      aria-label={labelledBy ? undefined : label}
      aria-labelledby={labelledBy}
      orientation="horizontal"
      disabled={disabled}
      className={cn(
        "flex w-full gap-1 rounded-full border border-line bg-bar-track p-1",
        className,
      )}
    >
      {options.map((option) => (
        <RadioGroupPrimitive.Item
          key={option.value}
          value={option.value}
          className={cn(
            "h-11 min-w-0 flex-1 rounded-full border border-transparent px-3 text-sm text-ink-muted transition-colors outline-none hover:text-ink",
            "focus-visible:ring-[3px] focus-visible:ring-brand/40",
            "data-[state=checked]:border-success-line data-[state=checked]:bg-glass-strong data-[state=checked]:font-semibold data-[state=checked]:text-ink data-[state=checked]:shadow-glow",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {option.label}
        </RadioGroupPrimitive.Item>
      ))}
    </RadioGroupPrimitive.Root>
  );
}

export { SegmentedControl };
