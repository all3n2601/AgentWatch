"use client";

import { useState, type ReactNode } from "react";
import { Bot, type LucideIcon } from "lucide-react";
import { AgentRun } from "@/components/agent-run";
import { LiveRunNotice, liveRunTabs } from "@/components/live-feed";
import { useLiveRun } from "@/components/use-live-feed";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { RunDetailModel } from "@/lib/run-detail";
import type { RunTab } from "@/lib/routes";
import type { RunView } from "@/lib/unified-runs";
import { DiagnosisTab } from "./diagnosis-tab";
import { EvidenceTab } from "./evidence-tab";
import { ResourcesTab } from "./resources-tab";
import { SummaryTab } from "./summary-tab";
import { TimelineTab } from "./timeline-tab";

/**
 * The page's tabs. "summary" (totals, AI summary, scope) exists only here: it was split off
 * the Diagnosis tab so that tab fits one screen, and nothing links to it directly.
 */
type RunDetailTab = RunTab | "summary";

const SUMMARY_TAB: [RunDetailTab, string] = ["summary", "Summary and scope"];

/** Only tabs the run can support are offered; a live run has a status tab instead of Agent. */
function tabsFor(view: RunView): [RunDetailTab, string][] {
  if (view.source === "live")
    return [
      ["diagnosis", "Diagnosis"],
      SUMMARY_TAB,
      ["status", "Live status"],
      ["timeline", "Timeline"],
      ["resources", "Resources"],
      ["evidence", "Evidence"],
    ];
  return [
    ["diagnosis", "Diagnosis"],
    SUMMARY_TAB,
    ["timeline", "Timeline"],
    ...(view.run.agent
      ? ([["agent", "Agent"]] as [RunDetailTab, string][])
      : []),
    ["resources", "Resources"],
    ["evidence", "Evidence"],
  ];
}

/**
 * A tab panel that fills the height left under the tab bar on `fit` screens and scrolls
 * inside itself, matching TabbedView, so the page never scrolls there.
 */
const TAB_PANEL = "min-w-0 fit:min-h-0 fit:flex-1 fit:overflow-y-auto";

/**
 * The existing tab views, framed as glass panels. On `fit` screens the panel fills the tab
 * and its body scrolls under a fixed panel header.
 */
function WrappedTab({
  id,
  icon,
  title,
  description,
  children,
}: {
  id: string;
  icon: LucideIcon;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby={id}
      className="min-w-0 fit:flex fit:h-full fit:min-h-0 fit:flex-col"
    >
      <PanelHeader
        titleId={id}
        icon={icon}
        title={title}
        description={description}
        className="fit:shrink-0"
      />
      {/*
        Focusable so keyboard users can scroll a body taller than the panel; `relative` keeps
        absolutely positioned descendants (sr-only text) inside its scroll clip.
      */}
      <div
        tabIndex={0}
        role="group"
        aria-label={`${title} content`}
        className="relative min-w-0 rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:-m-1 fit:min-h-0 fit:flex-1 fit:overflow-y-auto fit:p-1"
      >
        {children}
      </div>
    </GlassPanel>
  );
}

/**
 * A live run's Status, Timeline, Resources and Evidence tabs. One poll of the live API feeds
 * all four; until the run has data each tab shows the same notice.
 */
function LiveTabs({ runId }: { runId: string }) {
  const live = useLiveRun(runId);
  const tabs = liveRunTabs(live);
  const notice = <LiveRunNotice live={live} />;
  return (
    <>
      {(["status", "timeline", "resources", "evidence"] as const).map((id) => (
        <TabsContent key={id} value={id} className={TAB_PANEL}>
          {tabs ? tabs[id] : notice}
        </TabsContent>
      ))}
    </>
  );
}

export function RunTabs({
  view,
  model,
  initialTab,
  insightsEnabled,
}: {
  view: RunView;
  model: RunDetailModel;
  initialTab?: string;
  insightsEnabled: boolean;
}) {
  const tabs = tabsFor(view);
  const [tab, setTabState] = useState<RunDetailTab>(
    tabs.find(([id]) => id === initialTab)?.[0] ?? "diagnosis",
  );
  function setTab(next: string) {
    const match = tabs.find(([id]) => id === next);
    if (!match) return;
    setTabState(match[0]);
    // Keep the tab in the address so a link opens the same view without a server round trip.
    const url = new URL(window.location.href);
    url.searchParams.set("tab", match[0]);
    window.history.replaceState(null, "", url);
  }

  return (
    <Tabs
      value={tab}
      onValueChange={setTab}
      className="gap-4 fit:min-h-0 fit:flex-1 fit:gap-3"
    >
      <TabsList variant="glass" aria-label="Run views" className="shrink-0">
        {tabs.map(([id, label]) => (
          <TabsTrigger key={id} value={id}>
            {label}
          </TabsTrigger>
        ))}
      </TabsList>

      {/* Diagnosis fills the panel itself; its list and evidence scroll in their own panels. */}
      <TabsContent value="diagnosis" className={TAB_PANEL}>
        <DiagnosisTab model={model} />
      </TabsContent>

      <TabsContent value="summary" className={TAB_PANEL}>
        <SummaryTab model={model} insightsEnabled={insightsEnabled} />
      </TabsContent>

      {view.source === "live" && <LiveTabs runId={view.snapshot.run_id} />}

      {view.source === "saved" && (
        <>
          <TabsContent value="timeline" className={TAB_PANEL}>
            <TimelineTab key={view.run.run_id} run={view.run} />
          </TabsContent>
          {view.run.agent && (
            <TabsContent value="agent" className={TAB_PANEL}>
              <WrappedTab
                id="run-agent"
                icon={Bot}
                title="Agent"
                description="The saved agent conversation and model calls (existing agent view)."
              >
                <AgentRun
                  experiment={view.run}
                  onInspectTool={() => setTab("timeline")}
                />
              </WrappedTab>
            </TabsContent>
          )}
          <TabsContent value="resources" className={TAB_PANEL}>
            <ResourcesTab run={view.run} />
          </TabsContent>
          <TabsContent value="evidence" className={TAB_PANEL}>
            <EvidenceTab run={view.run} />
          </TabsContent>
        </>
      )}
    </Tabs>
  );
}
