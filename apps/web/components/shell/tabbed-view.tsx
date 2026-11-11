"use client";

import * as React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

export interface TabbedViewTab {
  id: string;
  label: string;
  /** Optional count or short status after the label ("13", "4 differ"). */
  badge?: React.ReactNode;
  content: React.ReactNode;
  /**
   * Mount only while active. For content that measures itself (charts) or has no state
   * worth keeping; other tabs stay mounted so their selections survive switching.
   */
  lazy?: boolean;
}

/**
 * The no-scroll page body: a header, a row of glass tabs and the active tab's
 * panel, which fills the remaining height on `fit` screens. Content that is
 * still taller (a long table) scrolls inside the panel, never the page.
 *
 * The active tab is kept in the URL (`?<param>=<id>`, default `view`) with
 * `history.replaceState`, so a reload or shared link opens the same tab
 * without a server round trip. Inactive panels stay mounted (hidden), so
 * selections and drafts inside them survive switching tabs, unless `lazy`.
 */
export function TabbedView({
  tabs,
  header,
  initialTab,
  param = "view",
  label,
  className,
}: {
  tabs: readonly TabbedViewTab[];
  /** Page title, scope and any always-visible summary above the tabs. */
  header?: React.ReactNode;
  /** The tab named in the URL; unknown ids fall back to the first tab. */
  initialTab?: string;
  param?: string;
  /** Accessible name for the tab list, e.g. "Run views". */
  label: string;
  className?: string;
}) {
  const fallback = tabs[0]?.id ?? "";
  const [active, setActive] = React.useState(
    tabs.some((tab) => tab.id === initialTab) ? initialTab! : fallback,
  );

  function select(id: string) {
    setActive(id);
    try {
      const url = new URL(window.location.href);
      if (id === fallback) url.searchParams.delete(param);
      else url.searchParams.set(param, id);
      window.history.replaceState(window.history.state, "", url);
    } catch {
      // The URL is a convenience; the tab still switches.
    }
  }

  return (
    <div
      className={cn("flex flex-col gap-3 fit:h-full fit:min-h-0", className)}
    >
      {header && <div className="shrink-0">{header}</div>}
      <Tabs
        value={active}
        onValueChange={select}
        className="gap-3 fit:min-h-0 fit:flex-1"
      >
        <TabsList variant="glass" aria-label={label} className="shrink-0">
          {tabs.map((tab) => (
            <TabsTrigger key={tab.id} value={tab.id} className="gap-2">
              {tab.label}
              {tab.badge !== undefined && (
                <span className="rounded-full bg-bar-track px-1.5 text-xs font-medium text-ink-muted tabular-nums">
                  {tab.badge}
                </span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>
        {tabs.map((tab) => (
          <TabsContent
            key={tab.id}
            value={tab.id}
            forceMount={tab.lazy ? undefined : true}
            className="min-w-0 data-[state=inactive]:hidden fit:min-h-0 fit:flex-1 fit:overflow-y-auto fit:pr-1"
          >
            {tab.content}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
