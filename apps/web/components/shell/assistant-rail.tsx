"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const WIDE_QUERY = "(min-width: 80rem)";

function subscribeWide(onChange: () => void) {
  const query = window.matchMedia(WIDE_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/**
 * Collapsible right rail for the investigation assistant. Collapsed, it is a
 * vertical tab ("Investigation Assistant"); expanded, it hosts `children`.
 * The content stays mounted while collapsed so a draft question survives.
 *
 * `defaultOpen="wide"` starts expanded only where the rail has its own column
 * (xl and up), so narrow screens are not pushed down by an open panel. The
 * user's own toggle always wins.
 */
export function AssistantRail({
  title = "Investigation Assistant",
  defaultOpen = false,
  children,
  className,
}: {
  title?: string;
  defaultOpen?: boolean | "wide";
  children: React.ReactNode;
  className?: string;
}) {
  const wide = React.useSyncExternalStore(
    subscribeWide,
    () => window.matchMedia(WIDE_QUERY).matches,
    () => false,
  );
  const [choice, setOpen] = React.useState<boolean | null>(null);
  const open = choice ?? (defaultOpen === "wide" ? wide : defaultOpen);
  const panelId = React.useId();

  return (
    <aside
      aria-label={title}
      className={cn(
        "glass-panel flex rounded-3xl",
        open ? "w-full flex-col xl:w-[20rem] 2xl:w-[22rem]" : "w-full xl:w-14",
        className,
      )}
    >
      {open ? (
        <div className="flex items-center gap-3 p-4 pb-2">
          <Sparkles className="size-5 shrink-0 text-ink" aria-hidden />
          <h2 className="flex-1 text-lg font-semibold tracking-tight text-ink">
            {title}
          </h2>
          <button
            type="button"
            aria-expanded
            aria-controls={panelId}
            onClick={() => setOpen(false)}
            className="glass-raised flex size-11 items-center justify-center rounded-full text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40"
          >
            <ChevronRight className="size-5" aria-hidden />
            <span className="sr-only">Collapse assistant</span>
          </button>
        </div>
      ) : (
        <button
          type="button"
          aria-expanded={false}
          aria-controls={panelId}
          onClick={() => setOpen(true)}
          className="flex w-full items-center justify-center gap-3 rounded-3xl px-3 py-4 text-ink outline-none hover:bg-glass-strong focus-visible:ring-[3px] focus-visible:ring-brand/40 xl:flex-col xl:py-5"
        >
          <Sparkles className="size-5" aria-hidden />
          <span className="text-[15px] font-medium xl:[writing-mode:vertical-rl]">
            {title}
          </span>
          <ChevronLeft className="size-5 max-xl:rotate-[-90deg]" aria-hidden />
        </button>
      )}
      <div
        id={panelId}
        hidden={!open}
        className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-4 pt-2"
      >
        {children}
      </div>
    </aside>
  );
}
