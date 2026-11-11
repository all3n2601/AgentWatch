"use client";

import * as React from "react";
import type { LucideIcon } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface FilterOption {
  value: string;
  label: string;
}

/**
 * Glass filter trigger over the existing Radix select. `caption` renders the
 * two-line form ("Task / All tasks"); without it the label is the trigger text
 * until a value is chosen, and stays the accessible name either way.
 */
function FilterSelect({
  label,
  caption = false,
  icon: Icon,
  value,
  onValueChange,
  options,
  placeholder,
  className,
}: {
  label: string;
  caption?: boolean;
  icon?: LucideIcon;
  value: string | undefined;
  onValueChange: (value: string) => void;
  options: readonly FilterOption[];
  placeholder?: string;
  className?: string;
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        aria-label={label}
        className={cn(
          "glass-raised h-12! min-w-0 gap-3 rounded-2xl border-line px-4 text-[15px] text-ink dark:bg-glass-strong dark:hover:bg-glass",
          caption && "h-14! min-w-48 justify-between",
          className,
        )}
      >
        {Icon && <Icon className="size-5 text-ink!" aria-hidden />}
        {caption ? (
          <span className="flex min-w-0 flex-1 flex-col items-start leading-tight">
            <span className="text-xs text-ink-muted">{label}</span>
            <SelectValue
              placeholder={placeholder ?? `All ${label.toLowerCase()}`}
            />
          </span>
        ) : (
          <SelectValue placeholder={placeholder ?? label} />
        )}
      </SelectTrigger>
      <SelectContent className="rounded-xl border-line bg-glass-strong backdrop-blur-xl">
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export { FilterSelect };
