import * as React from "react";
import { AssistantChat } from "@/components/assistant-chat";
import { WorkspacePage } from "@/components/shell/app-shell";
import { AssistantRail } from "@/components/shell/assistant-rail";
import { PageHeader } from "@/components/shell/page-header";
import { GlassPanel } from "@/components/ui/glass-panel";
import { Skeleton } from "@/components/widgets/state-views";
import { formatQuantity } from "@/lib/measurement-format";
import type { RunsPageData, RunsQuery, RunsView } from "@/lib/runs-page";
import { loadRunsPage } from "@/lib/runs-page-load";
import { RunsBrowser } from "./runs-browser";

const TITLE = "Runs";
const count = (value: number) => formatQuantity(value, "count");

function scopeLine(data: RunsPageData) {
  const saved = `${count(data.saved.rows.length)} ${
    data.saved.source === "database"
      ? "saved runs (newest history page)"
      : "local artifact files (history unavailable)"
  }`;
  const live =
    data.live.status === "ready"
      ? `${count(data.live.rows.length)} live runs (${
          data.liveBefore === null ? "newest page" : "older page"
        })`
      : "live runs unavailable";
  return `${saved} · ${live} · filters apply to loaded runs`;
}

/** Glass shell for the Runs page, with the investigation assistant collapsed. */
export function RunsPageFrame({
  chatEnabled,
  children,
}: {
  chatEnabled: boolean;
  children: React.ReactNode;
}) {
  return (
    <WorkspacePage
      rail={
        <AssistantRail>
          <AssistantChat
            enabled={chatEnabled}
            suggestions={[
              "Which recent runs failed their checks?",
              "Which runs spent the largest share of step time queued?",
            ]}
          />
        </AssistantRail>
      }
    >
      {children}
    </WorkspacePage>
  );
}

/** Reads both run stores, then hands serializable rows to the client browser. */
export async function RunsData({
  query,
  initialView,
}: {
  query: RunsQuery;
  initialView: RunsView;
}) {
  const data = await loadRunsPage(query);
  return (
    <RunsBrowser
      data={data}
      initialQuery={query.query}
      initialView={initialView}
      header={<PageHeader title={TITLE} scope={scopeLine(data)} />}
    />
  );
}

/** Shown while run history and the live API are read; fits one screen too. */
export function RunsLoading() {
  return (
    <div className="space-y-4 fit:flex fit:h-full fit:min-h-0 fit:flex-col fit:space-y-0 fit:gap-3 fit:overflow-hidden">
      <PageHeader title={TITLE} scope="Reading run history and live runs…" />
      <Skeleton className="h-14 w-full shrink-0" label="Loading runs" />
      <div className="flex shrink-0 flex-wrap gap-2" aria-hidden>
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton key={index} className="h-12 min-w-44 flex-1" />
        ))}
      </div>
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(22rem,26rem)] fit:min-h-0 fit:flex-1 fit:grid-rows-[minmax(0,1fr)] fit:items-stretch">
        <GlassPanel className="space-y-2 fit:overflow-hidden" aria-hidden>
          {Array.from({ length: 5 }, (_, index) => (
            <Skeleton key={index} className="h-16 w-full" />
          ))}
        </GlassPanel>
        <GlassPanel className="space-y-2 fit:overflow-hidden" aria-hidden>
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-40 w-full" />
        </GlassPanel>
      </div>
    </div>
  );
}
