"use client";

import { routes } from "@/lib/routes";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import Link from "next/link";
import {
  ArrowDownToLine,
  ArrowRight,
  GitCompareArrows,
  Info,
  Layers3,
  LayoutDashboard,
  Menu,
  FlaskConical,
  Orbit,
  PanelLeftClose,
  PanelLeftOpen,
  RadioTower,
  RefreshCw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ConcurrencyControls } from "@/components/concurrency-controls";
import { DatasetSummary } from "@/components/dataset-views";
import { CommandCenter } from "@/components/command-center";
import { LiveFeed } from "@/components/live-feed";
import { ExperimentsView } from "@/components/experiments-view";
import { InsightPanel } from "@/components/insight-panel";
import { RunPage } from "@/components/run-page";
import { RunsList } from "@/components/runs-list";
import { ThemeToggle } from "@/components/theme-toggle";
import { PanelHeading, Status } from "@/components/run-views";
import {
  aggregate,
  duration,
  evidenceUrl,
  executionOf,
  matchingWorkload,
  queueOf,
  runLabel,
  runTime,
  scopedRuns,
  stepsOf,
  type Page,
} from "@/lib/dashboard-data";
import type { DatasetSnapshot } from "@/lib/dataset-schema";
import type { LiveRunSummary } from "@/lib/live-feed-data";
import type { OverviewModel } from "@/lib/overview-model";
import { fromLive, fromSaved, type RunView } from "@/lib/unified-runs";
import type { DashboardData, Experiment } from "@/lib/experiment-schema";
import { cn } from "@/lib/utils";

const navigation = [
  {
    id: "overview",
    label: "Overview",
    icon: LayoutDashboard,
    group: "Investigate",
    detail: "Where waiting time goes across your runs",
  },
  {
    id: "runs",
    label: "Runs",
    icon: Layers3,
    group: "Investigate",
    detail: "Every run, live or saved, in one list",
  },
  {
    id: "experiments",
    label: "Experiments",
    icon: FlaskConical,
    group: "Investigate",
    detail: "Replays under injected contention, against their baseline",
  },
  {
    id: "compare",
    label: "Compare",
    icon: GitCompareArrows,
    group: "Investigate",
    detail: "Timing differences between recorded workloads",
  },
  {
    id: "quality",
    label: "Data quality",
    icon: ShieldCheck,
    group: "Trust",
    detail: "Telemetry coverage, labels, splits and quality checks",
  },
  {
    id: "controls",
    label: "Controls",
    icon: SlidersHorizontal,
    group: "Operate",
    detail: "Versioned manual concurrency limits for coding-agent runs",
  },
  {
    id: "settings",
    label: "Settings",
    icon: Settings2,
    group: "Operate",
    detail: "Display preferences and the data connection",
  },
  {
    // Reachable by URL but not listed: the Runs page opens a live run in its own tabs.
    id: "live",
    label: "Live runs",
    icon: RadioTower,
    group: "Hidden",
    detail: "Active steps, failures and feed status for runs in progress",
  },
] as const;
const navigationGroups = ["Investigate", "Trust", "Operate"] as const;
const defaultPreferences = JSON.stringify({
  density: "comfortable",
  refresh: 0,
  sidebar: "expanded",
});
function preferenceSnapshot() {
  try {
    return localStorage.getItem("agentwatch.preferences") || defaultPreferences;
  } catch {
    return defaultPreferences;
  }
}
function subscribePreferences(fn: () => void) {
  window.addEventListener("agentwatch:preferences", fn);
  window.addEventListener("storage", fn);
  return () => {
    window.removeEventListener("agentwatch:preferences", fn);
    window.removeEventListener("storage", fn);
  };
}
function usePreferences() {
  const raw = useSyncExternalStore(
    subscribePreferences,
    preferenceSnapshot,
    () => defaultPreferences,
  );
  let prefs: { density: string; refresh: number; sidebar: string };
  try {
    // Preferences saved before a field existed fall back to its default.
    prefs = { ...JSON.parse(defaultPreferences), ...JSON.parse(raw) };
    if (
      !["comfortable", "compact"].includes(prefs.density) ||
      ![0, 15, 30, 60].includes(prefs.refresh) ||
      !["expanded", "collapsed"].includes(prefs.sidebar)
    )
      throw Error();
  } catch {
    prefs = JSON.parse(defaultPreferences);
  }
  return [
    prefs,
    (next: typeof prefs) => {
      localStorage.setItem("agentwatch.preferences", JSON.stringify(next));
      window.dispatchEvent(new Event("agentwatch:preferences"));
    },
  ] as const;
}
function Notice({
  role,
  children,
}: {
  role: "alert" | "status";
  children: React.ReactNode;
}) {
  return (
    <div
      role={role}
      className="flex items-center gap-2 rounded-md border border-amber-400/25 bg-amber-400/5 px-3 py-2 text-xs text-amber-200"
    >
      <Info className="size-3.5 shrink-0" aria-hidden />
      <span className="min-w-0 break-words">{children}</span>
    </div>
  );
}
function Filters({
  options,
  value,
  onChange,
  label,
  compact = false,
}: {
  options: [string, string][];
  value: string;
  onChange: (value: string) => void;
  label: string;
  compact?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap gap-1 rounded-lg border bg-card p-1",
        compact && "gap-0.5 p-0.5",
      )}
      role="group"
      aria-label={label}
    >
      {options.map(([id, text]) => (
        <button
          key={id}
          aria-pressed={value === id}
          onClick={() => onChange(id)}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-primary",
            compact && "py-1",
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
function RunDate({ run }: { run: Experiment }) {
  return (
    <span>
      {new Date(runTime(run)).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })}
    </span>
  );
}
function Empty({ title, detail }: { title: string; detail: string }) {
  return (
    <Card className="items-center gap-3 border-dashed p-12 text-center">
      <Layers3 className="size-7 text-muted-foreground" />
      <h2 className="text-base font-semibold">{title}</h2>
      <p className="max-w-lg text-sm leading-6 text-muted-foreground">
        {detail}
      </p>
    </Card>
  );
}

export function Dashboard({
  initialData,
  page = "overview",
  initialRunId,
  initialBaselineId,
  live,
  dataset,
  liveRuns,
  runView,
  initialTab,
  insightsEnabled = false,
  overview,
  initialQuery,
}: {
  initialData: DashboardData;
  page?: Page;
  initialRunId?: string;
  initialBaselineId?: string;
  /** On the Runs list: search text from `/runs?q=`. */
  initialQuery?: string;
  /** On the live page: the `?liveRun=` UUID, or null to follow the newest run. */
  live?: { requested: string | null };
  /** On the dataset page: the dataset release read from the local DVC checkout. */
  dataset?: DatasetSnapshot;
  /** On the Runs list: one page of live runs, read alongside the saved runs. */
  liveRuns?: {
    runs: LiveRunSummary[];
    nextCursor: number | null;
    error?: string;
  };
  /** On a run's own page: the run, from whichever store holds it. */
  runView?: RunView;
  initialTab?: string;
  /** True when an AI endpoint and model are configured on the server. */
  insightsEnabled?: boolean;
  /** On Overview: the dataset-derived panels, or why they are unavailable. */
  overview?: OverviewModel | { unavailable: string };
}) {
  const [data, setData] = useState(initialData);
  const [range, setRange] = useState("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [mobile, setMobile] = useState(false);
  const [prefs, setPrefs] = usePreferences();
  // Collapsing only applies from the lg breakpoint up; the mobile drawer is always full width.
  const collapsed = prefs.sidebar === "collapsed";
  const inflight = useRef(false);
  const meta = navigation.find((item) => item.id === page)!;
  const runs = scopedRuns(data.experiments, range);
  const totals = aggregate(runs);
  const selected = runs.find((run) => run.run_id === initialRunId) ?? runs[0];
  const refresh = useCallback(async () => {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/experiments", { cache: "no-store" });
      if (!response.ok) throw Error();
      const next: DashboardData = await response.json();
      setData((old) => ({
        ...next,
        experiments: [
          ...next.experiments,
          ...old.experiments.filter(
            (r) => !next.experiments.some((n) => n.run_id === r.run_id),
          ),
        ],
        nextCursor: old.nextCursor ?? next.nextCursor,
      }));
    } catch {
      setError(
        "Refresh failed. Previously loaded evidence is still available.",
      );
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }, []);
  useEffect(() => {
    if (!prefs.refresh) return;
    const timer = setInterval(() => void refresh(), prefs.refresh * 1000);
    return () => clearInterval(timer);
  }, [prefs.refresh, refresh]);
  async function loadOlder() {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    setError("");
    try {
      const response = await fetch(
        `/api/experiments?before=${data.nextCursor}`,
      );
      if (!response.ok) throw Error();
      const next: DashboardData = await response.json();
      setData((old) => ({
        ...next,
        experiments: [
          ...old.experiments,
          ...next.experiments.filter(
            (r) => !old.experiments.some((n) => n.run_id === r.run_id),
          ),
        ],
        // Older pages do not refresh the newest history.
        loadedAt: old.loadedAt,
      }));
    } catch {
      setError("Older history could not be retrieved.");
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }
  const baseline = data.experiments.find((r) => r.run_id === initialBaselineId);
  const source =
    page === "live" && live
      ? { badge: "LIVE API", description: "live run read API" }
      : page === "controls"
        ? { badge: "POLICY API", description: "concurrency policy API" }
        : page === "experiments" || page === "quality"
          ? {
              badge: "DATASET FILES",
              description: "local DVC dataset release files",
            }
          : data.source === "database"
            ? { badge: "HISTORY", description: "saved run history" }
            : { badge: "LOCAL FILES", description: "local artifact fallback" };
  return (
    <div
      className={cn(
        "min-h-screen",
        prefs.density === "compact" && "aw-compact",
      )}
    >
      {mobile && (
        <button
          aria-label="Close navigation overlay"
          onClick={() => setMobile(false)}
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
        />
      )}
      <aside
        id="app-sidebar"
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[232px] flex-col overflow-y-auto border-r bg-sidebar px-4 transition-[transform,width] lg:translate-x-0",
          collapsed && "lg:w-[72px]",
          mobile ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <Link
          href="/"
          aria-label="AgentWatch home"
          className={cn(
            "flex h-14 shrink-0 items-center gap-2.5 px-2",
            collapsed && "lg:justify-center lg:px-0",
          )}
        >
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Orbit className="size-5" />
          </span>
          <span
            className={cn(
              "text-lg font-semibold tracking-tight",
              collapsed && "lg:hidden",
            )}
          >
            agentwatch<span className="text-primary">.</span>
          </span>
        </Link>
        <button
          className="absolute right-4 top-7 lg:hidden"
          aria-label="Close navigation"
          onClick={() => setMobile(false)}
        >
          <X className="size-4" />
        </button>
        <nav aria-label="Main navigation" className="space-y-5">
          {navigationGroups.map((group) => (
            <div key={group} className="space-y-1">
              <p
                className={cn(
                  "mb-2 px-3 text-[10px] font-medium uppercase tracking-[.16em] text-muted-foreground",
                  collapsed && "lg:hidden",
                )}
              >
                {group}
              </p>
              {navigation
                .filter((item) => item.group === group)
                .map((item) => (
                  <Link
                    key={item.id}
                    href={routes.section(item.id)}
                    onClick={() => setMobile(false)}
                    aria-current={page === item.id ? "page" : undefined}
                    title={collapsed ? item.label : undefined}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13px] transition-colors",
                      collapsed && "lg:justify-center",
                      page === item.id
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-accent hover:text-foreground",
                    )}
                  >
                    <item.icon className="size-4 shrink-0" />
                    <span className={cn(collapsed && "lg:sr-only")}>
                      {item.label}
                    </span>
                  </Link>
                ))}
            </div>
          ))}
        </nav>
        <div className="mt-auto space-y-3 pb-5 pt-4">
          <button
            type="button"
            aria-expanded={!collapsed}
            aria-controls="app-sidebar"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            onClick={() =>
              setPrefs({
                ...prefs,
                sidebar: collapsed ? "expanded" : "collapsed",
              })
            }
            className={cn(
              "hidden w-full items-center gap-3 rounded-lg px-3 py-2.5 text-[13px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground lg:flex",
              collapsed && "justify-center",
            )}
          >
            {collapsed ? (
              <PanelLeftOpen className="size-4 shrink-0" />
            ) : (
              <PanelLeftClose className="size-4 shrink-0" />
            )}
            {!collapsed && <span>Collapse</span>}
          </button>
          <div className={cn(collapsed && "lg:hidden")}>
            <ThemeToggle />
          </div>
          {collapsed && (
            <div className="hidden lg:block">
              <ThemeToggle compact />
            </div>
          )}
        </div>
      </aside>
      <div
        className={cn(
          "transition-[margin]",
          collapsed ? "lg:ml-[72px]" : "lg:ml-[232px]",
        )}
      >
        <header className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur">
          <div className="mx-auto flex min-h-14 max-w-[1680px] flex-wrap items-center gap-x-4 gap-y-2 px-5 py-2 sm:px-6">
            <button
              aria-label="Open navigation"
              onClick={() => setMobile(true)}
              className="-ml-1.5 flex size-9 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground lg:hidden"
            >
              <Menu className="size-5" />
            </button>
            <div className="flex min-w-0 flex-1 items-baseline gap-3">
              <h1 className="shrink-0 text-[17px] font-semibold tracking-tight">
                {meta.label}
              </h1>
              <p
                title={meta.detail}
                className="hidden min-w-0 truncate text-[13px] text-muted-foreground md:block"
              >
                {meta.detail}
              </p>
            </div>
            {page === "compare" && (
              <div className="order-last w-full overflow-x-auto sm:order-none sm:w-auto">
                <Filters
                  compact
                  label="Time range"
                  options={[
                    ["all", "All recorded"],
                    ["24h", "Last 24h"],
                    ["7d", "Last 7 days"],
                  ]}
                  value={range}
                  onChange={setRange}
                />
              </div>
            )}
            <div className="flex items-center gap-2 sm:gap-3">
              <Badge
                variant="outline"
                aria-label={`Data source: ${source.description}`}
                className="text-[9px] tracking-wide text-muted-foreground"
              >
                {source.badge}
              </Badge>
              {data.loadedAt && (
                <span className="hidden text-[11px] text-muted-foreground sm:inline">
                  Updated{" "}
                  <time dateTime={data.loadedAt} suppressHydrationWarning>
                    {new Date(data.loadedAt).toLocaleTimeString()}
                  </time>
                </span>
              )}
              <Button
                variant="ghost"
                size="sm"
                aria-label="Refresh data"
                disabled={busy}
                onClick={() => void refresh()}
              >
                <RefreshCw className={cn("size-3.5", busy && "animate-spin")} />
                <span className="hidden sm:inline">Refresh</span>
              </Button>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1680px] space-y-4 px-5 py-5 sm:px-6">
          {error && <Notice role="alert">{error}</Notice>}
          {data.warnings.length > 0 && (
            <Notice role="status">{data.warnings.join(" ")}</Notice>
          )}
          {page === "live" && live && <LiveFeed requested={live.requested} />}
          {page === "overview" && (
            <>
              {insightsEnabled && (
                <InsightPanel request={{ kind: "overview" }} />
              )}
              <CommandCenter
                model={overview && "release" in overview ? overview : undefined}
                unavailable={
                  overview && "unavailable" in overview
                    ? overview.unavailable
                    : undefined
                }
                saved={{
                  runs: totals.runs,
                  passed: totals.passed,
                  queueFraction: totals.queueFraction,
                }}
              />
              <RunTable
                runs={runs.slice(0, 6)}
                title="Recent activity"
                detail="Latest recorded experiments and agent sessions"
              />
              <div className="flex justify-end">
                <Button asChild variant="outline" size="sm">
                  <Link href="/runs">
                    View all runs <ArrowRight className="size-3.5" />
                  </Link>
                </Button>
              </div>
            </>
          )}
          {page === "runs" && runView && (
            <RunPage
              view={runView}
              initialTab={initialTab}
              insightsEnabled={insightsEnabled}
            />
          )}
          {page === "runs" && !runView && (
            <RunsList
              runs={[
                ...data.experiments.map(fromSaved),
                ...(liveRuns?.runs.map(fromLive) ?? []),
              ]}
              liveLoaded={liveRuns?.runs.length ?? 0}
              savedLoaded={data.experiments.length}
              liveError={liveRuns?.error}
              olderLive={
                liveRuns?.nextCursor
                  ? routes.runs({ liveBefore: String(liveRuns.nextCursor) })
                  : undefined
              }
              loadOlderSaved={data.nextCursor ? loadOlder : undefined}
              loadingSaved={busy}
              initialQuery={initialQuery}
            />
          )}
          {page === "experiments" && dataset && (
            <ExperimentsView snapshot={dataset} />
          )}
          {page === "quality" && dataset?.status === "ready" && (
            <DatasetSummary snapshot={dataset} />
          )}
          {(page === "experiments" || page === "quality") &&
            dataset?.status === "unavailable" && (
              <Empty title="Dataset unavailable" detail={dataset.reason} />
            )}
          {page === "controls" && <ConcurrencyControls />}
          {page === "compare" && (
            <ComparePage runs={runs} current={selected} baseline={baseline} />
          )}
          {page === "settings" && (
            <div className="grid min-w-0 grid-cols-1 gap-5 xl:grid-cols-2">
              <Card className="gap-0 p-5">
                <PanelHeading
                  title="Display preferences"
                  detail="Saved in this browser and applied across every page"
                />
                <label className="mb-3 block text-xs font-medium">
                  Table density
                </label>
                <Filters
                  label="Table density"
                  options={[
                    ["comfortable", "Comfortable"],
                    ["compact", "Compact"],
                  ]}
                  value={prefs.density}
                  onChange={(density) => setPrefs({ ...prefs, density })}
                />
                <p className="mt-3 text-xs text-muted-foreground">
                  Controls row padding in run and span tables.
                </p>
                <label className="mb-3 mt-7 block text-xs font-medium">
                  Refresh saved history
                </label>
                <Filters
                  label="Automatic refresh interval"
                  options={[
                    ["0", "Manual"],
                    ["15", "15 seconds"],
                    ["30", "30 seconds"],
                    ["60", "60 seconds"],
                  ]}
                  value={String(prefs.refresh)}
                  onChange={(value) =>
                    setPrefs({ ...prefs, refresh: Number(value) })
                  }
                />
                <p className="mt-3 text-xs leading-6 text-muted-foreground">
                  Checks for completed runs at the selected interval. Runs
                  currently executing are not streamed.
                </p>
                <Button
                  className="mt-6"
                  variant="outline"
                  size="sm"
                  onClick={() => setPrefs(JSON.parse(defaultPreferences))}
                >
                  Reset preferences
                </Button>
              </Card>
              <Card className="gap-0 p-5">
                <PanelHeading
                  title="Data connection"
                  detail="Observed application source"
                />
                <dl className="space-y-5 text-xs">
                  {[
                    [
                      "Source",
                      data.source === "database"
                        ? "PostgreSQL-backed ingestion API"
                        : "Local artifact fallback",
                    ],
                    ["Loaded runs", data.experiments.length],
                    [
                      "History pagination",
                      data.nextCursor
                        ? "More runs available"
                        : "All returned runs loaded",
                    ],
                    [
                      "Dashboard refresh",
                      prefs.refresh
                        ? `Every ${prefs.refresh} seconds`
                        : "Manual",
                    ],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="flex justify-between gap-5 border-b pb-3"
                    >
                      <dt className="text-muted-foreground">{label}</dt>
                      <dd className="text-right">{value}</dd>
                    </div>
                  ))}
                </dl>
                <div className="mt-6 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy}
                    onClick={() => void refresh()}
                  >
                    <RefreshCw className="size-3.5" />
                    Check connection
                  </Button>
                  <Button asChild size="sm" variant="outline">
                    <Link href="/runs">Browse retained evidence</Link>
                  </Button>
                </div>
              </Card>
              <Card className="gap-0 p-5 xl:col-span-2">
                <PanelHeading
                  title="Measurement contract"
                  detail="What this workspace currently verifies"
                />
                <div className="grid gap-5 sm:grid-cols-3">
                  {[
                    [
                      "Worker queue attribution",
                      "Rule-based classification from measured submission/start timestamps.",
                    ],
                    [
                      "Model telemetry",
                      "CPU inference request timing and server token counts. No inferred GPU or inference queue split.",
                    ],
                    [
                      "Persisted evidence",
                      "Completed runs and their artifacts. General live-agent ingestion and learned attribution remain planned.",
                    ],
                  ].map(([label, detail]) => (
                    <div key={label}>
                      <h3 className="text-xs font-medium">{label}</h3>
                      <p className="mt-2 text-xs leading-6 text-muted-foreground">
                        {detail}
                      </p>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          )}
          <footer className="flex flex-wrap items-center justify-between gap-2 border-t pt-5 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="size-3" />
              Recorded measurements · rule-based attribution
            </span>
            <span>
              {runs.length} runs in scope ·{" "}
              {data.source === "database"
                ? "Persistent history"
                : "Local artifacts"}
              {prefs.refresh ? ` · Refresh ${prefs.refresh}s` : ""}
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function RunTable({
  runs,
  title,
  detail,
}: {
  runs: Experiment[];
  title: string;
  detail: string;
}) {
  const [sort, setSort] = useState({ field: "recorded", descending: true });
  const accessor =
    sort.field === "queue"
      ? queueOf
      : sort.field === "execution"
        ? executionOf
        : runTime;
  const sorted = [...runs].sort(
    (a, b) => (sort.descending ? -1 : 1) * (accessor(a) - accessor(b)),
  );
  function sortHeader(field: string, label: string) {
    return (
      <th
        aria-sort={
          sort.field === field
            ? sort.descending
              ? "descending"
              : "ascending"
            : "none"
        }
      >
        <button
          className="flex items-center gap-1.5 hover:text-foreground"
          onClick={() =>
            setSort({
              field,
              descending: sort.field === field ? !sort.descending : true,
            })
          }
        >
          {label}
          <span className="text-[10px]">
            {sort.field === field ? (sort.descending ? "↓" : "↑") : "↕"}
          </span>
        </button>
      </th>
    );
  }
  return (
    <Card className="gap-0 overflow-hidden py-0">
      <div className="p-5 pb-0">
        <PanelHeading title={title} detail={detail} />
      </div>
      <div className="overflow-x-auto">
        <table className="aw-table">
          <thead>
            <tr>
              <th>Run / workload</th>
              <th>Status</th>
              {sortHeader("recorded", "Recorded")}
              <th>Steps</th>
              {sortHeader("execution", "Tool execution")}
              {sortHeader("queue", "Worker queue")}
              <th>Model requests</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {sorted.map((run) => (
              <tr key={run.run_id}>
                <td>
                  <Link
                    href={routes.run(run.run_id)}
                    className="font-medium text-foreground hover:text-primary"
                  >
                    {runLabel(run)}
                    <span className="mt-1 block font-mono text-[10px] font-normal text-muted-foreground">
                      {run.run_id.slice(0, 8)} · {run.kind}
                    </span>
                  </Link>
                </td>
                <td>
                  <Status passed={run.checks.passed} />
                </td>
                <td className="text-muted-foreground">
                  <RunDate run={run} />
                </td>
                <td className="font-mono">{stepsOf(run).length}</td>
                <td className="font-mono">{duration(executionOf(run))}</td>
                <td className="font-mono text-amber-200">
                  {duration(queueOf(run))}
                </td>
                <td className="font-mono">
                  {run.agent?.inference.length ?? "—"}
                </td>
                <td>
                  <Button variant="ghost" size="icon-sm" asChild>
                    <a
                      aria-label={`Export run ${run.run_id.slice(0, 8)}`}
                      href={evidenceUrl(run)}
                    >
                      <ArrowDownToLine className="size-3.5" />
                    </a>
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!runs.length && (
        <p className="p-12 text-center text-sm text-muted-foreground">
          No runs match this selection.
        </p>
      )}
    </Card>
  );
}
function ComparePage({
  runs,
  current,
  baseline,
}: {
  runs: Experiment[];
  current?: Experiment;
  baseline?: Experiment;
}) {
  const previous =
    baseline ??
    runs.find((r) => r.run_id !== current?.run_id && r.kind === current?.kind);
  if (!current || !previous)
    return (
      <Empty
        title="Two runs are needed to compare"
        detail="Record another experiment of the same workload, then return here."
      />
    );
  const match = matchingWorkload(current, previous);
  const rows: [string, number, number][] = [
    ["Clean queue", previous.clean.queue_seconds, current.clean.queue_seconds],
    [
      "Clean execution",
      previous.clean.execution_seconds,
      current.clean.execution_seconds,
    ],
    [
      "Busy queue",
      previous.contended.queue_seconds,
      current.contended.queue_seconds,
    ],
    [
      "Busy execution",
      previous.contended.execution_seconds,
      current.contended.execution_seconds,
    ],
    ["Total tool execution", executionOf(previous), executionOf(current)],
    ["Total worker waiting", queueOf(previous), queueOf(current)],
  ];
  return (
    <div className="space-y-5">
      <div className="grid gap-4 lg:grid-cols-2">
        {[
          ["Baseline", previous],
          ["Candidate", current],
        ].map(([label, r]) => {
          const run = r as Experiment;
          return (
            <Card key={String(label)} className="gap-3 p-5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                {String(label)}
              </span>
              <span className="font-medium">{runLabel(run)}</span>
              <span className="font-mono text-xs text-muted-foreground">
                {run.run_id.slice(0, 8)} · <RunDate run={run} />
              </span>
              <details className="rounded-lg border">
                <summary className="cursor-pointer px-3 py-2 text-xs text-muted-foreground">
                  Change {String(label).toLowerCase()}
                </summary>
                <div className="max-h-64 space-y-1 overflow-y-auto border-t p-2">
                  {runs
                    .filter(
                      (item) =>
                        item.run_id !== current.run_id &&
                        item.run_id !== previous.run_id,
                    )
                    .map((item) => (
                      <Link
                        key={item.run_id}
                        className="block rounded-lg p-3 hover:bg-accent"
                        href={routes.compare(
                          label === "Candidate" ? item.run_id : current.run_id,
                          label === "Baseline" ? item.run_id : previous.run_id,
                        )}
                      >
                        <span className="block text-xs font-medium">
                          {runLabel(item)}
                        </span>
                        <span className="mt-1 block font-mono text-[10px] text-muted-foreground">
                          {item.run_id.slice(0, 8)} · <RunDate run={item} />
                        </span>
                      </Link>
                    ))}
                  {runs.length <= 2 && (
                    <p className="p-3 text-xs text-muted-foreground">
                      All available runs are selected.
                    </p>
                  )}
                </div>
              </details>
            </Card>
          );
        })}
      </div>
      <div
        className={cn(
          "rounded-lg border px-4 py-3 text-xs",
          match
            ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-200"
            : "border-amber-400/20 bg-amber-400/5 text-amber-200",
        )}
      >
        {match
          ? "Matching workload: checksum, iterations, and worker capacity agree."
          : "Workload settings differ. Timing differences do not establish a performance regression."}
      </div>
      <Card className="gap-0 overflow-hidden py-0">
        <div className="p-5 pb-0">
          <PanelHeading
            title="Measured timing differences"
            detail="Candidate minus baseline · positive values indicate longer duration"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="aw-table">
            <thead>
              <tr>
                <th>Measurement</th>
                <th>Baseline</th>
                <th>Candidate</th>
                <th>Difference</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, a, b]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td className="font-mono">{duration(a)}</td>
                  <td className="font-mono">{duration(b)}</td>
                  <td
                    className={cn(
                      "font-mono",
                      b > a ? "text-amber-200" : "text-emerald-300",
                    )}
                  >
                    {b >= a ? "+" : "−"}
                    {duration(Math.abs(b - a))}
                  </td>
                  <td className="font-mono text-muted-foreground">
                    {a > 0
                      ? `${b >= a ? "+" : ""}${(((b - a) / a) * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="flex gap-2">
        <Button asChild variant="outline" size="sm">
          <Link href={routes.run(current.run_id)}>
            Inspect candidate trace <ArrowRight className="size-3.5" />
          </Link>
        </Button>
        <Button asChild variant="outline" size="sm">
          <a href={evidenceUrl(previous)}>Export baseline</a>
        </Button>
      </div>
    </div>
  );
}
