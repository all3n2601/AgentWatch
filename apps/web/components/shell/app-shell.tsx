import * as React from "react";
import { NavRail } from "@/components/shell/nav-rail";
import { WORKSPACE_NAVIGATION } from "@/components/shell/navigation";
import { TopBar } from "@/components/shell/top-bar";
import { WorkspacePreferences } from "@/components/shell/workspace-preferences";
import { cn } from "@/lib/utils";

/**
 * The persistent workspace frame, rendered once by `app/(workspace)/layout.tsx`:
 * header and sidebar stay mounted while pages change, so the sidebar never
 * moves or resizes between pages. On `fit` screens (≥ 1280×760) the frame
 * fills the viewport and pages lay out inside it without page scroll; below
 * that the document scrolls.
 */
export function WorkspaceFrame({ children }: { children: React.ReactNode }) {
  return (
    <div className="aw-canvas min-h-dvh fit:flex fit:h-dvh fit:flex-col fit:overflow-hidden">
      <WorkspacePreferences />
      <a
        href="#main"
        className="glass-raised fixed top-3 left-3 z-50 -translate-y-24 rounded-xl px-4 py-2 text-ink focus:translate-y-0"
      >
        Skip to content
      </a>
      <TopBar workspace="Agent Flow Studio" />
      <div className="flex flex-col gap-4 px-4 pt-4 pb-6 sm:px-6 md:flex-row md:items-start md:gap-6 fit:min-h-0 fit:flex-1 fit:items-stretch fit:pb-3">
        <NavRail
          items={WORKSPACE_NAVIGATION}
          className="md:sticky md:top-4 md:max-h-[calc(100dvh-2rem)] md:shrink-0 fit:static fit:h-full fit:max-h-none"
        />
        <div className="min-w-0 flex-1 fit:h-full fit:min-h-0">{children}</div>
      </div>
    </div>
  );
}

/**
 * One page inside the workspace frame: the main column and an optional right
 * rail (usually an AssistantRail). On `fit` screens both fill the frame's
 * height; a page should lay itself out to fit (see `TabbedView`), and main
 * scrolls internally only as a last resort.
 */
export function WorkspacePage({
  rail,
  children,
  className,
}: {
  rail?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    // Below xl the rail sits above the content as a collapsed bar, so it never
    // narrows the main column; from xl it moves to the right.
    <div
      className={cn(
        "grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_auto] xl:gap-6 fit:h-full fit:items-stretch",
        className,
      )}
    >
      {rail && (
        <div className="xl:sticky xl:top-4 xl:col-start-2 xl:row-start-1 fit:static fit:h-full fit:min-h-0 fit:[&>aside]:h-full">
          {rail}
        </div>
      )}
      <main
        id="main"
        className="min-w-0 xl:col-start-1 xl:row-start-1 fit:flex fit:h-full fit:min-h-0 fit:flex-col fit:overflow-y-auto"
      >
        {children}
      </main>
    </div>
  );
}
