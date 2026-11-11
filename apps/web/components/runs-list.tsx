"use client";

import { routes } from "@/lib/routes";
import type { Route } from "next";
import Link from "next/link";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import { PanelHeading } from "@/components/run-views";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { percent } from "@/lib/dashboard-data";
import {
  byNewest,
  filterRuns,
  type RunFilters,
  type RunSource,
  type RunState,
  type UnifiedRun,
} from "@/lib/unified-runs";
import { displayName } from "@/lib/dataset-view";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 25;

const STATE_TONE: Record<RunState, string> = {
  queued: "bg-sky-400",
  running: "bg-sky-400",
  completed: "bg-emerald-400",
  passed: "bg-emerald-400",
  failed: "bg-rose-400",
  cancelled: "bg-slate-400",
  inconclusive: "bg-amber-400",
};
const SOURCE_LABEL: Record<RunSource, string> = {
  live: "Live",
  saved: "Saved",
};

export function StateBadge({ state }: { state: RunState }) {
  return (
    <Badge variant="outline" className="gap-1.5 font-normal capitalize">
      <i
        className={cn("size-1.5 rounded-full", STATE_TONE[state])}
        aria-hidden
      />
      {state}
    </Badge>
  );
}

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: [T, string][];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="flex flex-wrap gap-0.5 rounded-lg border bg-card p-0.5"
    >
      {options.map(([id, text]) => (
        <button
          key={id}
          aria-pressed={value === id}
          onClick={() => onChange(id)}
          className={cn(
            "rounded-md px-3 py-1 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-primary",
            value === id
              ? "bg-accent text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

/** Every run in one list. Which store a run came from is a column and a filter, not a page. */
export function RunsList({
  runs,
  liveLoaded,
  savedLoaded,
  olderLive,
  liveError,
  loadOlderSaved,
  loadingSaved,
  initialQuery = "",
}: {
  runs: UnifiedRun[];
  liveLoaded: number;
  savedLoaded: number;
  olderLive?: Route;
  liveError?: string;
  loadOlderSaved?: () => void;
  loadingSaved?: boolean;
  /** Search text from the URL (`/runs?q=`), e.g. from the header search box. */
  initialQuery?: string;
}) {
  const [filters, setFilters] = useState<RunFilters>({
    source: "all",
    group: "all",
    kind: "all",
    query: initialQuery,
  });
  const [page, setPage] = useState(0);
  const update = (next: Partial<RunFilters>) => {
    setFilters({ ...filters, ...next });
    setPage(0);
  };
  const filtered = useMemo(
    () => filterRuns(runs, filters).sort(byNewest),
    [runs, filters],
  );
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, pageCount - 1);
  const visible = filtered.slice(
    current * PAGE_SIZE,
    (current + 1) * PAGE_SIZE,
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-0 basis-full sm:max-w-sm sm:flex-1 sm:basis-auto">
          <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
          <Input
            className="h-9 pl-9 text-xs"
            aria-label="Search runs"
            placeholder="Search run ID, task, or workload…"
            value={filters.query}
            onChange={(event) => update({ query: event.target.value })}
          />
        </div>
        <Segmented
          label="Run source"
          value={filters.source}
          onChange={(source) => update({ source })}
          options={[
            ["all", "All sources"],
            ["live", "Live"],
            ["saved", "Saved"],
          ]}
        />
        <Segmented
          label="Run state"
          value={filters.group}
          onChange={(group) => update({ group })}
          options={[
            ["all", "Any state"],
            ["in_progress", "In progress"],
            ["done", "Done"],
            ["needs_attention", "Needs attention"],
          ]}
        />
        <Segmented
          label="Run kind"
          value={filters.kind}
          onChange={(kind) => update({ kind })}
          options={[
            ["all", "All kinds"],
            ["agent", "Agent"],
            ["coding", "Coding"],
            ["cpu", "CPU"],
          ]}
        />
      </div>
      <Card className="gap-0 overflow-hidden py-0">
        <div className="p-5 pb-0">
          <PanelHeading
            title="Runs"
            detail={`${filtered.length.toLocaleString("en-US")} of ${runs.length.toLocaleString("en-US")} loaded · ${liveLoaded.toLocaleString("en-US")} live, ${savedLoaded.toLocaleString("en-US")} saved`}
          />
        </div>
        {liveError && (
          <p
            role="alert"
            className="mx-5 mb-3 rounded-md border border-amber-400/25 bg-amber-400/5 px-3 py-2 text-xs text-amber-200"
          >
            Live runs could not be read: {liveError}
          </p>
        )}
        <div className="overflow-x-auto">
          <table className="aw-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Source</th>
                <th>State</th>
                <th>Steps</th>
                <th>Waiting</th>
                <th>Main resource</th>
                <th>Added</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((run) => (
                <tr key={run.id}>
                  <td>
                    <Link
                      href={routes.run(run.id)}
                      className="font-medium text-foreground hover:text-primary"
                    >
                      {run.title}
                    </Link>
                    <div className="font-mono text-[10px] text-muted-foreground">
                      {run.id.slice(0, 8)} · {displayName(run.kind)}
                    </div>
                  </td>
                  <td>{SOURCE_LABEL[run.source]}</td>
                  <td>
                    <StateBadge state={run.state} />
                  </td>
                  <td>{run.steps ?? "—"}</td>
                  <td className="font-mono">
                    {run.waitingShare === null
                      ? "—"
                      : percent(run.waitingShare)}
                  </td>
                  <td>
                    {run.topResource === null ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      displayName(run.topResource)
                    )}
                  </td>
                  <td>
                    <time dateTime={run.at} suppressHydrationWarning>
                      {new Date(run.at).toLocaleString("en-US", {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </time>
                  </td>
                </tr>
              ))}
              {visible.length === 0 && (
                <tr>
                  <td
                    colSpan={7}
                    className="py-10 text-center text-muted-foreground"
                  >
                    No runs match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-3 text-xs text-muted-foreground">
          <span>
            Waiting is the queue share of step time; live runs show it on their
            page.
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={current === 0}
              onClick={() => setPage(current - 1)}
            >
              Previous
            </Button>
            <span>
              Page {current + 1} of {pageCount}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={current >= pageCount - 1}
              onClick={() => setPage(current + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      </Card>
      <div className="flex flex-wrap gap-2">
        {olderLive && (
          <Button asChild variant="outline" size="sm">
            <Link href={olderLive}>Older live runs</Link>
          </Button>
        )}
        {loadOlderSaved && (
          <Button
            variant="outline"
            size="sm"
            disabled={loadingSaved}
            onClick={loadOlderSaved}
          >
            Older saved runs
          </Button>
        )}
      </div>
    </div>
  );
}
