"use client";

import * as React from "react";
import Link from "next/link";
import {
  Boxes,
  CircleCheck,
  Cpu,
  Database,
  Layers,
  Loader,
  Search,
  TriangleAlert,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { FilterSelect } from "@/components/ui/filter-select";
import { TabbedView, type TabbedViewTab } from "@/components/shell/tabbed-view";
import { GlassPanel } from "@/components/ui/glass-panel";
import { AlertBanner } from "@/components/widgets/alert-banner";
import { StatStrip } from "@/components/widgets/stat";
import {
  EmptyState,
  ErrorState,
  UnavailableState,
} from "@/components/widgets/state-views";
import { displayName } from "@/lib/dataset-view";
import { formatQuantity } from "@/lib/measurement-format";
import { routes } from "@/lib/routes";

import {
  byNewestRow,
  DEFAULT_FILTERS,
  filterOptions,
  filterRows,
  MAX_QUERY_LENGTH,
  mergeOlderSaved,
  parseOlderSaved,
  RUNS_PAGE_SIZE,
  summarizeRows,
  toggleCompare,
  UNATTRIBUTED,
  type RunsPageData,
  type RunsPageFilters,
  type RunsView,
} from "@/lib/runs-page";
import type { RunSource, StateGroup } from "@/lib/unified-runs";
import {
  CompareControls,
  CompareNote,
  CompareSelectionView,
} from "./run-compare";
import { RunPreview } from "./run-preview";
import { RunRow, RunRowHeader } from "./run-row";

const count = (value: number) => formatQuantity(value, "count");

/** Matches the xl breakpoint, where the preview sits beside the list. */
const SIDE_BY_SIDE_QUERY = "(min-width: 80rem)";

type OlderSavedState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string };

const SOURCE_OPTIONS: { value: RunSource | "all"; label: string }[] = [
  { value: "all", label: "All sources" },
  { value: "live", label: "Live" },
  { value: "saved", label: "Saved" },
];
const STATE_OPTIONS: { value: StateGroup | "all"; label: string }[] = [
  { value: "all", label: "Any state" },
  { value: "in_progress", label: "In progress" },
  { value: "done", label: "Done" },
  { value: "needs_attention", label: "Needs attention" },
];

function isSource(value: string): value is RunSource | "all" {
  return SOURCE_OPTIONS.some((option) => option.value === value);
}
function isGroup(value: string): value is StateGroup | "all" {
  return STATE_OPTIONS.some((option) => option.value === value);
}

/** Keeps `?q=` in the address bar so a refresh or shared link keeps the search. */
function replaceQueryParam(query: string) {
  const url = new URL(window.location.href);
  if (query.trim()) url.searchParams.set("q", query);
  else url.searchParams.delete("q");
  window.history.replaceState(window.history.state, "", url);
}

/**
 * The Runs page body: the header, filters and summary above two tabs, the run
 * list with a preview of the selected run, and the runs ticked for
 * comparison. Filters apply to the runs loaded so far. On `fit` screens the
 * page does not scroll; the list and the preview scroll inside their panels.
 */
export function RunsBrowser({
  data,
  initialQuery,
  initialView,
  header,
}: {
  data: RunsPageData;
  initialQuery: string;
  initialView: RunsView;
  /** Server-rendered page title and scope line. */
  header: React.ReactNode;
}) {
  const [filters, setFilters] = React.useState<RunsPageFilters>({
    ...DEFAULT_FILTERS,
    query: initialQuery,
  });
  const [page, setPage] = React.useState(0);
  const [saved, setSaved] = React.useState(data.saved.rows);
  const [savedCursor, setSavedCursor] = React.useState(data.saved.nextCursor);
  const [older, setOlder] = React.useState<OlderSavedState>({
    status: "idle",
  });
  const [previewKey, setPreviewKey] = React.useState<string | null>(null);
  const [compareKeys, setCompareKeys] = React.useState<string[]>([]);
  const olderRequest = React.useRef<AbortController | null>(null);
  const previewHeading = React.useRef<HTMLHeadingElement>(null);
  const previewId = React.useId();
  const listHeadingId = React.useId();
  const searchId = React.useId();

  React.useEffect(() => () => olderRequest.current?.abort(), []);

  const liveRows = React.useMemo(
    () => (data.live.status === "ready" ? data.live.rows : []),
    [data.live],
  );
  const rows = React.useMemo(() => [...saved, ...liveRows], [saved, liveRows]);
  const options = React.useMemo(() => filterOptions(rows), [rows]);
  const filtered = React.useMemo(
    () => filterRows(rows, filters).sort(byNewestRow),
    [rows, filters],
  );
  const summary = React.useMemo(() => summarizeRows(filtered), [filtered]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / RUNS_PAGE_SIZE));
  const current = Math.min(page, pageCount - 1);
  const visible = filtered.slice(
    current * RUNS_PAGE_SIZE,
    (current + 1) * RUNS_PAGE_SIZE,
  );
  // A filtered-out selection falls back to the first visible run.
  const previewRow =
    filtered.find((row) => row.key === previewKey) ?? visible[0];
  const compareRows = compareKeys.flatMap(
    (key) => rows.find((row) => row.key === key) ?? [],
  );

  const update = (next: Partial<RunsPageFilters>) => {
    setFilters((old) => ({ ...old, ...next }));
    setPage(0);
  };

  function preview(key: string) {
    setPreviewKey(key);
    // Below xl the preview is under the list; bring it into view.
    if (!window.matchMedia(SIDE_BY_SIDE_QUERY).matches) {
      const reduced = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      previewHeading.current?.scrollIntoView({
        behavior: reduced ? "auto" : "smooth",
        block: "start",
      });
      previewHeading.current?.focus({ preventScroll: true });
    }
  }

  async function loadOlderSaved() {
    if (savedCursor === null || older.status === "loading") return;
    olderRequest.current?.abort();
    const controller = new AbortController();
    olderRequest.current = controller;
    setOlder({ status: "loading" });
    try {
      const response = await fetch(`/api/experiments?before=${savedCursor}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) {
        setOlder({
          status: "error",
          message: `Older saved runs could not be loaded (HTTP ${response.status}).`,
        });
        return;
      }
      const result = parseOlderSaved(await response.json());
      if (!result.ok) {
        setOlder({
          status: "error",
          message: "Older saved runs were returned in an unexpected format.",
        });
        return;
      }
      setSaved((loaded) => mergeOlderSaved(loaded, result.rows));
      setSavedCursor(result.nextCursor);
      setOlder({ status: "idle" });
    } catch (error) {
      if (controller.signal.aborted) return;
      setOlder({
        status: "error",
        message:
          error instanceof Error
            ? `Older saved runs could not be loaded: ${error.message}`
            : "Older saved runs could not be loaded.",
      });
    }
  }

  const liveUnavailable = data.live.status === "unavailable";
  const liveNext =
    data.live.status === "ready" && data.live.nextCursor !== null
      ? routes.runs({
          q: filters.query.trim() || undefined,
          liveBefore: String(data.live.nextCursor),
        })
      : null;

  const clearCompare = () => setCompareKeys([]);

  const filterBar = (
    <div
      role="search"
      aria-label="Filter runs"
      className="grid grid-cols-2 items-center gap-2 md:flex md:flex-wrap"
    >
      <div className="relative col-span-2 min-w-0 md:max-w-sm md:flex-1">
        <label htmlFor={searchId} className="sr-only">
          Search loaded runs by ID, task, workload or model
        </label>
        <Search
          className="pointer-events-none absolute top-1/2 left-4 size-5 -translate-y-1/2 text-ink-muted"
          aria-hidden
        />
        <input
          id={searchId}
          type="search"
          value={filters.query}
          maxLength={MAX_QUERY_LENGTH}
          onChange={(event) => {
            update({ query: event.target.value });
            replaceQueryParam(event.target.value);
          }}
          placeholder="Search ID, task, workload or model…"
          className="glass-raised h-12 w-full rounded-2xl pr-4 pl-12 text-[15px] text-ink outline-none placeholder:text-ink-subtle focus-visible:ring-[3px] focus-visible:ring-brand/40"
        />
      </div>
      <FilterSelect
        caption
        label="Source"
        icon={Database}
        value={filters.source}
        onValueChange={(value) => isSource(value) && update({ source: value })}
        options={SOURCE_OPTIONS}
        className="w-full min-w-0 md:w-auto md:min-w-44 md:flex-none"
      />
      <FilterSelect
        caption
        label="State"
        icon={CircleCheck}
        value={filters.group}
        onValueChange={(value) => isGroup(value) && update({ group: value })}
        options={STATE_OPTIONS}
        className="w-full min-w-0 md:w-auto md:min-w-44 md:flex-none"
      />
      <FilterSelect
        caption
        label="Kind"
        icon={Boxes}
        value={filters.kind}
        onValueChange={(kind) => update({ kind })}
        options={[
          { value: "all", label: "All kinds" },
          ...options.kinds.map((kind) => ({
            value: kind,
            label: displayName(kind),
          })),
        ]}
        className="w-full min-w-0 md:w-auto md:min-w-44 md:flex-none"
      />
      <FilterSelect
        caption
        label="Top resource"
        icon={Cpu}
        value={filters.resource}
        onValueChange={(resource) => update({ resource })}
        options={[
          { value: "all", label: "Any resource" },
          ...options.resources.map((resource) => ({
            value: resource,
            label: displayName(resource),
          })),
          ...(options.unattributed
            ? [{ value: UNATTRIBUTED, label: "None attributed" }]
            : []),
        ]}
        className="w-full min-w-0 md:w-auto md:min-w-44 md:flex-none"
      />
    </div>
  );

  const listBody =
    visible.length > 0 ? (
      <div className="fit:flex fit:min-h-0 fit:flex-1 fit:flex-col">
        <RunRowHeader />
        {/* The inset padding keeps row focus rings clear of the scroll edge. */}
        <ul
          className="space-y-1.5 fit:-mx-1 fit:min-h-0 fit:flex-1 fit:overflow-y-auto fit:px-1 fit:py-1"
          aria-label="Runs"
        >
          {visible.map((row) => (
            <RunRow
              key={row.key}
              row={row}
              previewed={row.key === previewRow?.key}
              previewId={previewId}
              onPreview={preview}
              compare={{
                checked: compareKeys.includes(row.key),
                disabled:
                  compareKeys.length >= 2 && !compareKeys.includes(row.key),
                onToggle: (key) =>
                  setCompareKeys((keys) => toggleCompare(keys, key)),
              }}
            />
          ))}
        </ul>
      </div>
    ) : (
      <div className="fit:min-h-0 fit:flex-1 fit:overflow-y-auto">
        {filters.source === "live" && liveUnavailable ? (
          <UnavailableState
            title="Live runs unavailable"
            reason={data.live.status === "unavailable" ? data.live.reason : ""}
          />
        ) : rows.length === 0 ? (
          <EmptyState
            title="No runs recorded yet"
            detail="Neither run history nor the live API returned any runs."
          />
        ) : (
          <EmptyState
            title="No runs match these filters"
            detail="Filters apply to the runs loaded so far. Load older runs or clear a filter."
            action={
              <Button
                variant="glass"
                size="pill"
                onClick={() => {
                  update(DEFAULT_FILTERS);
                  replaceQueryParam("");
                }}
              >
                Clear filters
              </Button>
            }
          />
        )}
      </div>
    );

  const listFooter = (
    <div className="shrink-0 space-y-3 border-t border-line pt-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <nav
          aria-label="Run list pages"
          className="flex flex-wrap items-center gap-2"
        >
          <span className="text-sm text-ink-muted tabular-nums">
            Page {current + 1} of {pageCount}
          </span>
          <Button
            variant="glass"
            size="pill"
            disabled={current === 0}
            onClick={() => setPage(current - 1)}
          >
            Previous
          </Button>
          <Button
            variant="glass"
            size="pill"
            disabled={current >= pageCount - 1}
            onClick={() => setPage(current + 1)}
          >
            Next
          </Button>
        </nav>
        <div className="flex flex-wrap items-center gap-2">
          {data.liveBefore !== null && (
            <Button asChild variant="glass" size="pill">
              <Link
                href={routes.runs({ q: filters.query.trim() || undefined })}
              >
                Newest live runs
              </Link>
            </Button>
          )}
          {liveNext && (
            <Button asChild variant="glass" size="pill">
              <Link href={liveNext}>Older live runs</Link>
            </Button>
          )}
          {savedCursor !== null && (
            <Button
              variant="glass"
              size="pill"
              disabled={older.status === "loading"}
              aria-busy={older.status === "loading"}
              onClick={() => void loadOlderSaved()}
            >
              {older.status === "loading"
                ? "Loading older saved runs…"
                : "Load older saved runs"}
            </Button>
          )}
        </div>
      </div>
      {older.status === "error" && (
        <ErrorState
          title="Older saved runs not loaded"
          detail={`${older.message} The runs already shown are unchanged.`}
          action={
            <Button
              variant="glass"
              size="pill"
              onClick={() => void loadOlderSaved()}
            >
              Try again
            </Button>
          }
        />
      )}
    </div>
  );

  const allRuns = (
    // On fit screens the one grid row takes the panel height; the list and
    // the preview each scroll inside their own panel.
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(22rem,26rem)] fit:h-full fit:grid-rows-[minmax(0,1fr)] fit:items-stretch">
      <GlassPanel
        as="section"
        aria-labelledby={listHeadingId}
        className="@container flex min-w-0 flex-col gap-3 fit:min-h-0"
      >
        {/* The All runs tab names this panel visually; the heading keeps the
            landmark labelled while the row of controls saves a line. */}
        <h2 id={listHeadingId} className="sr-only">
          Run list
        </h2>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <CompareControls selected={compareRows} onClear={clearCompare} />
          <p className="min-w-[min(100%,14rem)] flex-1 text-xs text-ink-muted">
            Newest first. Select a run to preview it; tick two saved runs to
            compare.
          </p>
        </div>
        <CompareNote selected={compareRows} hideHint />
        {listBody}
        {listFooter}
      </GlassPanel>

      <RunPreview
        row={previewRow}
        id={previewId}
        headingRef={previewHeading}
        className="scroll-mt-4 xl:sticky xl:top-4 fit:static fit:min-h-0 fit:overflow-y-auto"
      />
    </div>
  );

  const tabs: (TabbedViewTab & { id: RunsView })[] = [
    {
      id: "all",
      label: "All runs",
      badge: count(filtered.length),
      content: allRuns,
    },
    {
      id: "compare",
      label: "Compare selection",
      badge: `${count(compareRows.length)} of 2`,
      content: (
        <CompareSelectionView selected={compareRows} onClear={clearCompare} />
      ),
    },
  ];

  return (
    <TabbedView
      label="Run views"
      initialTab={initialView}
      tabs={tabs}
      header={
        <div className="space-y-3">
          {header}
          {data.saved.source === "local" && (
            <AlertBanner
              tone="warning"
              title="Run history is unavailable"
              detail={`Saved runs below are the latest local artifact files, not history. ${data.saved.warnings.join(" ")}`}
            />
          )}
          {data.saved.source === "database" &&
            data.saved.warnings.length > 0 && (
              <AlertBanner
                tone="warning"
                title="Some saved runs could not be read"
                detail={data.saved.warnings.join(" ")}
              />
            )}
          {liveUnavailable && (
            <AlertBanner
              tone="warning"
              title="Live runs could not be read"
              detail={`${data.live.status === "unavailable" ? data.live.reason : ""} The list shows saved runs only.`}
            />
          )}
          {filterBar}
          <StatStrip
            label="Runs matching the filters"
            items={[
              {
                id: "total",
                icon: Layers,
                value: count(summary.total),
                label: `of ${count(rows.length)} match`,
              },
              {
                id: "done",
                icon: CircleCheck,
                tone: "success",
                value: count(summary.done),
                label: "done",
              },
              {
                id: "attention",
                icon: TriangleAlert,
                tone: summary.needsAttention > 0 ? "warning" : undefined,
                value: count(summary.needsAttention),
                label: "need attention",
              },
              {
                id: "progress",
                icon: Loader,
                tone: "info",
                value: count(summary.inProgress),
                label: "in progress",
              },
            ]}
          />
        </div>
      }
    />
  );
}
