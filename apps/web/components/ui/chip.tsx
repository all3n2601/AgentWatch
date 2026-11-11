import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type ChipProps = React.ComponentProps<"span"> & {
  icon?: LucideIcon;
  size?: "sm" | "default";
};

/** Raised glass pill for scope and meta labels ("qwen3:1.7b · 598 runs"). */
function Chip({
  icon: Icon,
  size = "default",
  className,
  children,
  ...props
}: ChipProps) {
  return (
    <span
      data-slot="chip"
      className={cn(
        "glass-raised inline-flex w-fit max-w-full items-center gap-2 rounded-full text-ink",
        size === "sm"
          ? "px-3 py-1 text-[13px]"
          : "px-4 py-1.5 text-sm font-medium",
        className,
      )}
      {...props}
    >
      {Icon && <Icon className="size-4 shrink-0" aria-hidden />}
      <span className="min-w-0 truncate">{children}</span>
    </span>
  );
}

export { Chip };
