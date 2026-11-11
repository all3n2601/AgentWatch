"use client";

import * as React from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

const COPIED_FEEDBACK_MS = 1600;
const SHORT_ID_LENGTH = 8;

type CopyState = "idle" | "copied" | "failed";

/**
 * Shows a shortened identifier and copies the full value. The full value stays
 * reachable through the button's accessible name and title, so truncation
 * never hides the identifier.
 */
function CopyChip({
  value,
  label = "ID",
  display,
  className,
}: {
  value: string;
  label?: string;
  display?: string;
  className?: string;
}) {
  const [state, setState] = React.useState<CopyState>("idle");

  React.useEffect(() => {
    if (state === "idle") return;
    const timer = window.setTimeout(() => setState("idle"), COPIED_FEEDBACK_MS);
    return () => window.clearTimeout(timer);
  }, [state]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setState("copied");
    } catch {
      // Clipboard access can be denied; say so rather than pretending it worked.
      setState("failed");
    }
  }

  const Icon = state === "copied" ? Check : Copy;
  return (
    <button
      type="button"
      onClick={copy}
      title={value}
      aria-label={`Copy ${label} ${value}`}
      className={cn(
        "glass-raised inline-flex h-11 items-center gap-2 rounded-xl px-3.5 font-mono text-[15px] text-ink transition-colors hover:bg-glass",
        className,
      )}
    >
      <span>{display ?? value.slice(0, SHORT_ID_LENGTH)}</span>
      <Icon className="size-4 text-ink-muted" aria-hidden />
      <span className="sr-only" aria-live="polite">
        {state === "copied"
          ? "Copied"
          : state === "failed"
            ? "Copy failed"
            : ""}
      </span>
    </button>
  );
}

export { CopyChip };
