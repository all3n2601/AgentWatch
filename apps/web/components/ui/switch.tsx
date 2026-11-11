"use client";

import * as React from "react";
import { Switch as SwitchPrimitive } from "radix-ui";
import { cn } from "@/lib/utils";

/** On/off toggle. Pair it with a visible label; the state text is shown beside it. */
function Switch({
  className,
  showStateLabel = true,
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root> & {
  showStateLabel?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-3">
      <SwitchPrimitive.Root
        data-slot="switch"
        className={cn(
          "peer inline-flex h-7 w-12 shrink-0 items-center rounded-full border border-line bg-bar-track p-0.5 transition-colors outline-none",
          "focus-visible:ring-[3px] focus-visible:ring-brand/40",
          "data-[state=checked]:border-success-line data-[state=checked]:bg-brand",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        {...props}
      >
        <SwitchPrimitive.Thumb className="block size-6 rounded-full bg-white shadow-raised transition-transform data-[state=checked]:translate-x-5" />
      </SwitchPrimitive.Root>
      {showStateLabel && (
        <span
          aria-hidden
          className="text-sm text-ink peer-data-[state=checked]:hidden"
        >
          Off
        </span>
      )}
      {showStateLabel && (
        <span
          aria-hidden
          className="hidden text-sm text-ink peer-data-[state=checked]:inline"
        >
          On
        </span>
      )}
    </span>
  );
}

export { Switch };
