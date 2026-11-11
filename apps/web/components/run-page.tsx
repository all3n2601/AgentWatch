"use client";

import { routes } from "@/lib/routes";
import Link from "next/link";
import { ArrowDownToLine, GitCompareArrows } from "lucide-react";
import { useState } from "react";
import { AgentRun } from "@/components/agent-run";
import { DiagnosisPanel } from "@/components/diagnosis-panel";
import { InsightPanel } from "@/components/insight-panel";
import { LiveRunDetail } from "@/components/live-feed";
import { StateBadge } from "@/components/runs-list";
import { Resources, TraceExplorer } from "@/components/run-views";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { evidenceUrl, percent } from "@/lib/dashboard-data";
import type { Experiment } from "@/lib/experiment-schema";
import {
  diagnoseView,
  fromSaved,
  type RunState,
  type RunView,
} from "@/lib/unified-runs";
import { cn } from "@/lib/utils";

type Tab =
  "diagnosis" | "timeline" | "agent" | "resources" | "evidence" | "live";

const SAVED_FILES = ["results.json", "spans.jsonl", "samples.jsonl"] as const;

function tabsFor(view: RunView): [Tab, string][] {
  if (view.source === "live")
    return [
      ["diagnosis", "Diagnosis"],
      ["live", "Timeline, resources and evidence"],
    ];
  return [
    ["diagnosis", "Diagnosis"],
    ["timeline", "Timeline"],
    ...(view.run.agent ? ([["agent", "Agent"]] as [Tab, string][]) : []),
    ["resources", "Resources"],
    ["evidence", "Evidence"],
  ];
}

function useTab(view: RunView, initial?: string) {
  const available = tabsFor(view).map(([id]) => id);
  const [tab, setTab] = useState<Tab>(
    available.includes(initial as Tab) ? (initial as Tab) : "diagnosis",
  );
  return [
    tab,
    (next: Tab) => {
      setTab(next);
      // Keep the tab in the address so a link opens the same view without a server round trip.
      const url = new URL(window.location.href);
      url.searchParams.set("tab", next);
      window.history.replaceState(null, "", url);
    },
    tabsFor(view),
  ] as const;
}

function Header({ view }: { view: RunView }) {
  const live = view.source === "live";
  const saved = view.source === "saved" ? fromSaved(view.run) : null;
  const state: RunState = live ? view.snapshot.state : saved!.state;
  const title = live ? view.snapshot.registration.task_id : saved!.title;
  const id = live ? view.snapshot.run_id : view.run.run_id;
  const added = live ? view.snapshot.created_at : saved!.at;
  return (
    <Card className="flex-row flex-wrap items-center justify-between gap-3 px-5 py-4">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2 text-sm font-medium">
          {title}
          <StateBadge state={state} />
          <Badge variant="outline" className="font-normal">
            {live ? "Live run" : "Saved run"}
          </Badge>
        </div>
        <p className="mt-2 break-all font-mono text-[10px] text-muted-foreground">
          {id}
        </p>
        {live && view.snapshot.reason && (
          <p className="mt-2 text-xs text-amber-200">{view.snapshot.reason}</p>
        )}
      </div>
      <dl className="flex gap-6 text-xs">
        <div>
          <dt className="text-muted-foreground">Added</dt>
          <dd className="mt-1 font-mono text-sm">
            <time dateTime={added} suppressHydrationWarning>
              {new Date(added).toLocaleString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </time>
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Steps</dt>
          <dd className="mt-1 font-mono text-sm">
            {live ? view.snapshot.steps.length : saved!.steps}
          </dd>
        </div>
        {saved && saved.waitingShare !== null && (
          <div>
            <dt className="text-muted-foreground">Waiting</dt>
            <dd className="mt-1 font-mono text-sm">
              {percent(saved.waitingShare)}
            </dd>
          </div>
        )}
      </dl>
    </Card>
  );
}

function SavedEvidence({ run }: { run: Experiment }) {
  return (
    <Card className="gap-0 p-5">
      <h2 className="text-sm font-semibold tracking-tight">Evidence files</h2>
      <p className="mb-4 mt-1 text-xs text-muted-foreground">
        The artifacts saved with this run, downloaded as recorded.
      </p>
      <div className="flex flex-wrap gap-2">
        {SAVED_FILES.map((file) => (
          <Button key={file} asChild variant="outline" size="sm">
            <a href={evidenceUrl(run, file)}>
              <ArrowDownToLine className="size-3.5" />
              {file}
            </a>
          </Button>
        ))}
        <Button asChild variant="ghost" size="sm">
          <Link href={routes.compare(run.run_id)}>
            <GitCompareArrows className="size-3.5" />
            Compare with another run
          </Link>
        </Button>
      </div>
    </Card>
  );
}

/** One page per run, whichever store it came from; only tabs the run cannot support are absent. */
export function RunPage({
  view,
  initialTab,
  insightsEnabled = false,
}: {
  view: RunView;
  initialTab?: string;
  insightsEnabled?: boolean;
}) {
  const [tab, setTab, tabs] = useTab(view, initialTab);
  return (
    <div className="space-y-4">
      <Header view={view} />
      <div
        role="tablist"
        aria-label="Run views"
        className="flex flex-wrap gap-0.5 rounded-lg border bg-card p-0.5"
      >
        {tabs.map(([id, label]) => (
          <button
            key={id}
            role="tab"
            id={`run-tab-${id}`}
            aria-selected={tab === id}
            aria-controls="run-tabpanel"
            onClick={() => setTab(id)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-primary",
              tab === id
                ? "bg-accent text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <div
        id="run-tabpanel"
        role="tabpanel"
        aria-labelledby={`run-tab-${tab}`}
        className="min-w-0"
      >
        {tab === "diagnosis" && (
          <div className="space-y-4">
            {insightsEnabled && (
              <InsightPanel
                request={{
                  kind: "run",
                  runId:
                    view.source === "live"
                      ? view.snapshot.run_id
                      : view.run.run_id,
                }}
              />
            )}
            <DiagnosisPanel scope={view.source} steps={diagnoseView(view)} />
          </div>
        )}
        {tab === "live" && view.source === "live" && (
          <LiveRunDetail runId={view.snapshot.run_id} />
        )}
        {view.source === "saved" && tab === "timeline" && (
          <TraceExplorer key={view.run.run_id} run={view.run} />
        )}
        {view.source === "saved" && tab === "agent" && view.run.agent && (
          <AgentRun
            experiment={view.run}
            onInspectTool={() => setTab("timeline")}
          />
        )}
        {view.source === "saved" && tab === "resources" && (
          <Resources key={view.run.run_id} run={view.run} />
        )}
        {view.source === "saved" && tab === "evidence" && (
          <SavedEvidence run={view.run} />
        )}
      </div>
    </div>
  );
}
