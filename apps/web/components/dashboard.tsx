"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import Link from "next/link";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Bot,
  Check,
  ChevronRight,
  Clock3,
  Cpu,
  Database,
  GitCompareArrows,
  Layers3,
  LayoutDashboard,
  ListTree,
  Menu,
  Orbit,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { AgentRun } from "@/components/agent-run";
import {
  PanelHeading,
  Resources,
  Status,
  TraceExplorer,
} from "@/components/run-views";
import {
  aggregate,
  duration,
  evidenceUrl,
  executionOf,
  matchingWorkload,
  percent,
  queueOf,
  runLabel,
  runTime,
  scopedRuns,
  stepsOf,
  type Page,
} from "@/lib/dashboard-data";
import type { DashboardData, Experiment } from "@/lib/experiment-schema";
import { cn } from "@/lib/utils";

const navigation = [
  {
    id: "overview",
    label: "Overview",
    icon: LayoutDashboard,
    detail: "Activity across your workspace",
  },
  {
    id: "runs",
    label: "Run history",
    icon: Layers3,
    detail: "Search, inspect, and export recorded experiments",
  },
  {
    id: "traces",
    label: "Trace explorer",
    icon: ListTree,
    detail: "Follow each step from submission to completion",
  },
  {
    id: "agents",
    label: "Agents",
    icon: Bot,
    detail: "Model requests, tool feedback, and conversation evidence",
  },
  {
    id: "resources",
    label: "Resources",
    icon: Cpu,
    detail: "Measurement coverage, pressure readings, and experiment integrity",
  },
  {
    id: "compare",
    label: "Compare runs",
    icon: GitCompareArrows,
    detail: "Evaluate timing differences between recorded workloads",
  },
  {
    id: "settings",
    label: "Workspace settings",
    icon: Settings2,
    detail: "Configure this dashboard and inspect your data connection",
  },
] as const;
const defaultPreferences = JSON.stringify({
  density: "comfortable",
  refresh: 0,
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
  let prefs: { density: string; refresh: number };
  try {
    prefs = JSON.parse(raw);
    if (
      !["comfortable", "compact"].includes(prefs.density) ||
      ![0, 15, 30, 60].includes(prefs.refresh)
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
function Metric({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: React.ReactNode;
  detail: string;
  icon: typeof Activity;
}) {
  return (
    <Card className="gap-0 p-5">
      <div className="flex min-h-8 items-start justify-between gap-2">
        <span className="text-xs text-muted-foreground">{label}</span>
        <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      </div>
      <div className="metric-number mt-4 text-[30px] font-semibold">
        {value}
      </div>
      <p className="mt-2 text-[11px] text-muted-foreground">{detail}</p>
    </Card>
  );
}
function Filters({
  options,
  value,
  onChange,
  label,
}: {
  options: [string, string][];
  value: string;
  onChange: (value: string) => void;
  label: string;
}) {
  return (
    <div
      className="flex flex-wrap gap-1 rounded-lg border bg-card p-1"
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
}: {
  initialData: DashboardData;
  page?: Page;
  initialRunId?: string;
  initialBaselineId?: string;
}) {
  const [data, setData] = useState(initialData);
  const [range, setRange] = useState("all");
  const [search, setSearch] = useState("");
  const [type, setType] = useState("all");
  const [status, setStatus] = useState("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [mobile, setMobile] = useState(false);
  const [prefs, setPrefs] = usePreferences();
  const inflight = useRef(false);
  const [updated, setUpdated] = useState<string | null>(null);
  const meta = navigation.find((item) => item.id === page)!;
  const runs = scopedRuns(data.experiments, range);
  const totals = aggregate(runs);
  const eligible = page === "agents" ? runs.filter((run) => run.agent) : runs;
  const selected =
    eligible.find((run) => run.run_id === initialRunId) ?? eligible[0];
  const visible = runs.filter(
    (run) =>
      `${run.run_id} ${runLabel(run)} ${run.kind}`
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (type === "all" ||
        (type === "agent" ? !!run.agent : run.kind === type && !run.agent)) &&
      (status === "all" ||
        (status === "passed" ? run.checks.passed : !run.checks.passed)),
  );
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
      setUpdated(new Date().toLocaleTimeString());
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
      }));
    } catch {
      setError("Older history could not be retrieved.");
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }
  const path = (target: string, run?: Experiment, baseline?: Experiment) => {
    const targetRun =
      target === "agents" && !run?.agent ? runs.find((r) => r.agent) : run;
    return `/${target}${targetRun ? `?run=${targetRun.run_id}${baseline ? `&baseline=${baseline.run_id}` : ""}` : ""}`;
  };
  const baseline = data.experiments.find((r) => r.run_id === initialBaselineId);
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
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[232px] flex-col border-r bg-[#111720] px-4 transition-transform lg:translate-x-0",
          mobile ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <Link href="/" className="flex h-[76px] items-center gap-2.5 px-2">
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Orbit className="size-5" />
          </span>
          <span className="text-lg font-semibold tracking-tight">
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
        <div className="mb-7 rounded-lg border bg-white/[.02] p-3">
          <div className="flex items-center gap-2 text-xs font-medium">
            <Database className="size-3.5 text-primary" />
            AgentWatch workspace
          </div>
          <p className="mt-1.5 pl-5.5 text-[10px] text-muted-foreground">
            Local development · measured evidence
          </p>
        </div>
        <p className="mb-3 px-3 text-[10px] font-medium uppercase tracking-[.16em] text-muted-foreground">
          Observability
        </p>
        <nav aria-label="Main navigation" className="space-y-1">
          {navigation
            .filter((n) => n.id !== "settings")
            .map((item) => (
              <Link
                key={item.id}
                href={path(item.id, selected)}
                onClick={() => setMobile(false)}
                aria-current={page === item.id ? "page" : undefined}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13px] transition-colors",
                  page === item.id
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <item.icon className="size-4" />
                {item.label}
                {item.id === "runs" && (
                  <span className="ml-auto rounded bg-white/5 px-1.5 text-[10px]">
                    {data.experiments.length}
                  </span>
                )}
              </Link>
            ))}
        </nav>
        <div className="mt-8 border-t pt-5">
          <Link
            href="/settings"
            aria-current={page === "settings" ? "page" : undefined}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13px]",
              page === "settings"
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Settings2 className="size-4" />
            Workspace settings
          </Link>
        </div>
        <div className="mt-auto mb-5 rounded-lg border bg-background/30 p-3">
          <div className="flex items-center gap-2 text-xs">
            <span
              className={cn(
                "size-1.5 rounded-full",
                data.source === "database" ? "bg-emerald-400" : "bg-amber-400",
              )}
            />
            {data.source === "database"
              ? "History connected"
              : "Local file fallback"}
          </div>
          <p className="mt-2 text-[11px] leading-5 text-muted-foreground">
            {data.experiments.length} loaded runs ·{" "}
            {aggregate(data.experiments).agentRuns} agent runs
          </p>
          <Link
            href="/settings"
            className="mt-2 flex items-center gap-1 text-[11px] text-primary"
          >
            Connection details <ChevronRight className="size-3" />
          </Link>
        </div>
        <div className="mb-5 flex items-center gap-2 px-2 text-[10px] text-muted-foreground">
          <ShieldCheck className="size-3.5" />
          Evidence-based attribution<span className="ml-auto">v0.1</span>
        </div>
      </aside>
      <div className="lg:ml-[232px]">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between gap-3 border-b bg-background/95 px-5 backdrop-blur sm:px-8">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <button
              aria-label="Open navigation"
              onClick={() => setMobile(true)}
              className="mr-1 lg:hidden"
            >
              <Menu className="size-5" />
            </button>
            <span className="hidden sm:inline">Workspace</span>
            <ChevronRight className="hidden size-3 sm:inline" />
            <span className="text-foreground">{meta.label}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-[11px] text-muted-foreground md:inline">
              {updated ? `Updated ${updated}` : "Saved experiment history"}
            </span>
            <Badge
              variant="outline"
              className="text-[9px] tracking-wide text-muted-foreground"
            >
              LOCAL
            </Badge>
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
        </header>
        <main className="mx-auto max-w-[1680px] space-y-6 px-5 py-7 sm:px-8">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="mb-2 text-[10px] font-medium uppercase tracking-[.18em] text-primary">
                Agent observability
              </p>
              <h1 className="text-[28px] font-semibold tracking-tight">
                {meta.label}
              </h1>
              <p className="mt-1.5 text-[13px] text-muted-foreground">
                {meta.detail}
              </p>
            </div>
            {page !== "settings" && (
              <Filters
                label="Time range"
                options={[
                  ["all", "All recorded"],
                  ["24h", "Last 24h"],
                  ["7d", "Last 7 days"],
                ]}
                value={range}
                onChange={setRange}
              />
            )}
          </div>
          {error && (
            <div
              role="alert"
              className="rounded-lg border border-amber-400/20 p-3 text-xs text-amber-200"
            >
              {error}
            </div>
          )}
          {data.warnings.length > 0 && (
            <div
              role="status"
              className="rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-xs text-amber-200"
            >
              {data.warnings.join(" ")}
            </div>
          )}
          {page === "overview" && (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <Metric
                  icon={Layers3}
                  label="Recorded runs"
                  value={totals.runs}
                  detail={`${totals.passed} passed · ${totals.runs - totals.passed} inconclusive`}
                />
                <Metric
                  icon={ListTree}
                  label="Profiled tool steps"
                  value={totals.steps}
                  detail={`${totals.attributed} attributed to CPU worker queue`}
                />
                <Metric
                  icon={Clock3}
                  label="Worker queue time"
                  value={duration(totals.queue)}
                  detail={`${percent(totals.queueFraction)} of profiled tool-step time`}
                />
                <Metric
                  icon={Bot}
                  label="Model requests"
                  value={totals.modelCalls}
                  detail={`${totals.agentRuns} agent runs · ${duration(totals.inferenceWall)} inference wall time`}
                />
              </div>
              <div className="grid gap-5 min-[1100px]:grid-cols-[1.7fr_1fr]">
                <Card className="gap-0 p-6">
                  <PanelHeading
                    title="Tool time by run"
                    detail="Measured execution and queue · latest 20 runs"
                  >
                    <div className="flex gap-3 text-[10px] text-muted-foreground">
                      <span className="flex items-center gap-1.5">
                        <i className="size-2 rounded bg-violet-400" />
                        Execution
                      </span>
                      <span className="flex items-center gap-1.5">
                        <i className="size-2 rounded bg-amber-400" />
                        Queue
                      </span>
                    </div>
                  </PanelHeading>
                  {runs.length ? (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={runs
                            .slice(0, 20)
                            .reverse()
                            .map((r) => ({
                              run: r.run_id.slice(0, 6),
                              execution: executionOf(r),
                              queue: queueOf(r),
                            }))}
                          barCategoryGap="30%"
                        >
                          <CartesianGrid vertical={false} stroke="#273140" />
                          <XAxis
                            dataKey="run"
                            axisLine={false}
                            tickLine={false}
                          />
                          <YAxis unit="s" axisLine={false} tickLine={false} />
                          <Tooltip
                            cursor={{ fill: "#ffffff05" }}
                            contentStyle={{
                              background: "#171d27",
                              border: "1px solid #303948",
                              borderRadius: 8,
                            }}
                            formatter={(v) => `${Number(v).toFixed(3)} s`}
                          />
                          <Bar
                            dataKey="execution"
                            stackId="time"
                            fill="#a78bfa"
                            isAnimationActive={false}
                          />
                          <Bar
                            dataKey="queue"
                            stackId="time"
                            fill="#fbbf24"
                            radius={[3, 3, 0, 0]}
                            isAnimationActive={false}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="py-20 text-center text-sm text-muted-foreground">
                      No runs in this time range.
                    </p>
                  )}
                  <p className="mt-4 text-[11px] text-muted-foreground">
                    Sums baseline and evaluation tool steps. Model request time
                    is measured separately.
                  </p>
                </Card>
                <Card className="gap-0 p-6">
                  <PanelHeading
                    title="Resource coverage"
                    detail="Signals present in loaded history"
                  />
                  <div className="space-y-5">
                    {[
                      {
                        name: "CPU worker queue",
                        value: `${totals.steps} steps`,
                        detail: "Direct enqueue and worker-start timestamps",
                        color: "bg-emerald-400",
                      },
                      {
                        name: "CPU inference",
                        value: `${totals.modelCalls} requests`,
                        detail: "Ollama request wall time and server timings",
                        color: "bg-violet-400",
                      },
                      {
                        name: "GPU / retrieval",
                        value: "Not instrumented",
                        detail: "No measured evidence in these runs",
                        color: "bg-slate-500",
                      },
                    ].map((item) => (
                      <div
                        key={item.name}
                        className="border-b pb-5 last:border-0 last:pb-0"
                      >
                        <div className="flex items-center justify-between gap-3 text-xs">
                          <span className="flex items-center gap-2">
                            <i
                              className={`size-1.5 rounded-full ${item.color}`}
                            />
                            {item.name}
                          </span>
                          <span className="font-mono text-[10px] text-muted-foreground">
                            {item.value}
                          </span>
                        </div>
                        <p className="mt-2 pl-3.5 text-[11px] leading-5 text-muted-foreground">
                          {item.detail}
                        </p>
                      </div>
                    ))}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    asChild
                    className="mt-5 w-full"
                  >
                    <Link href={path("resources", selected)}>
                      Inspect resource evidence{" "}
                      <ArrowRight className="size-3.5" />
                    </Link>
                  </Button>
                </Card>
              </div>
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
          {page === "runs" && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="relative min-w-0 flex-1 sm:max-w-sm">
                  <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
                  <Input
                    className="h-9 pl-9 text-xs"
                    aria-label="Search run history"
                    placeholder="Search run ID, model, or workload…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                </div>
                <Filters
                  label="Workload filter"
                  options={[
                    ["all", "All workloads"],
                    ["agent", "Agent"],
                    ["coding", "Test tool"],
                    ["cpu", "CPU"],
                  ]}
                  value={type}
                  onChange={setType}
                />
                <Filters
                  label="Run status"
                  options={[
                    ["all", "Any status"],
                    ["passed", "Passed"],
                    ["inconclusive", "Inconclusive"],
                  ]}
                  value={status}
                  onChange={setStatus}
                />
              </div>
              <RunTable
                runs={visible}
                title="Recorded experiments"
                detail={`${visible.length} matching runs · ${data.nextCursor ? "more history available" : "all loaded history"}`}
              />
              {data.nextCursor && (
                <Button disabled={busy} variant="outline" onClick={loadOlder}>
                  Load older runs
                </Button>
              )}
            </>
          )}
          {page === "agents" && (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <Metric
                  icon={Bot}
                  label="Agent sessions"
                  value={totals.agentRuns}
                  detail="Real tool-calling runs"
                />
                <Metric
                  icon={Activity}
                  label="Inference requests"
                  value={totals.modelCalls}
                  detail="Includes correction attempts"
                />
                <Metric
                  icon={Clock3}
                  label="Inference wall time"
                  value={duration(totals.inferenceWall)}
                  detail="Sum of elapsed model requests"
                />
                <Metric
                  icon={Layers3}
                  label="Generated tokens"
                  value={totals.outputTokens ?? "Unavailable"}
                  detail="Reported server counts; missing stays unavailable"
                />
              </div>
            </>
          )}
          {["traces", "agents", "resources"].includes(page) &&
            (selected && eligible.length ? (
              <div className="grid items-start gap-5 min-[1100px]:grid-cols-[250px_minmax(0,1fr)]">
                <RunNavigator runs={eligible} selected={selected} page={page} />
                <div className="min-w-0 space-y-5">
                  <RunContext run={selected} />
                  {page === "traces" && (
                    <TraceExplorer key={selected.run_id} run={selected} />
                  )}{" "}
                  {page === "agents" && (
                    <AgentRun
                      experiment={selected}
                      onInspectTool={() =>
                        window.location.assign(path("traces", selected))
                      }
                    />
                  )}{" "}
                  {page === "resources" && (
                    <Resources key={selected.run_id} run={selected} />
                  )}
                </div>
              </div>
            ) : (
              <Empty
                title="No recorded evidence in this view"
                detail={
                  page === "agents"
                    ? "Record a tool-calling run with make record-llm, then refresh."
                    : "Record an experiment and refresh, or choose a wider time range."
                }
              />
            ))}
          {page === "compare" && (
            <ComparePage runs={runs} current={selected} baseline={baseline} />
          )}
          {page === "settings" && (
            <div className="grid min-w-0 grid-cols-1 gap-5 xl:grid-cols-2">
              <Card className="gap-0 p-6">
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
              <Card className="gap-0 p-6">
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
                    ["Environment", "Local development"],
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
              <Card className="gap-0 p-6 xl:col-span-2">
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
    <Card className="gap-0 overflow-hidden">
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
                    href={`/traces?run=${run.run_id}`}
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
function RunNavigator({
  runs,
  selected,
  page,
}: {
  runs: Experiment[];
  selected: Experiment;
  page: Page;
}) {
  const [query, setQuery] = useState("");
  const visible = runs.filter((r) =>
    `${r.run_id} ${runLabel(r)}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <Card className="gap-0 overflow-hidden min-[1100px]:sticky min-[1100px]:top-21">
      <div className="border-b p-4">
        <div className="mb-3 flex justify-between text-xs font-medium">
          <span>Recorded runs</span>
          <span className="text-muted-foreground">{runs.length}</span>
        </div>
        <div className="relative">
          <Search className="absolute top-2.5 left-2.5 size-3 text-muted-foreground" />
          <Input
            aria-label="Find a recorded run"
            className="h-8 pl-7 text-xs"
            placeholder="Find run or model…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      </div>
      <div className="max-h-[280px] overflow-y-auto p-2 min-[1100px]:max-h-[620px]">
        {visible.map((run) => (
          <Link
            key={run.run_id}
            href={`/${page}?run=${run.run_id}`}
            className={cn(
              "mb-1 block rounded-lg border p-3 transition-colors",
              selected.run_id === run.run_id
                ? "border-primary/25 bg-primary/5"
                : "border-transparent hover:bg-accent",
            )}
          >
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "size-1.5 rounded-full",
                  run.checks.passed ? "bg-emerald-400" : "bg-amber-400",
                )}
              />
              <span className="truncate text-xs font-medium">
                {runLabel(run)}
              </span>
              {run.run_id === selected.run_id && (
                <Check className="ml-auto size-3 shrink-0 text-primary" />
              )}
            </div>
            <div className="mt-2 flex justify-between gap-2 font-mono text-[10px] text-muted-foreground">
              <span>{run.run_id.slice(0, 8)}</span>
              <span>{duration(queueOf(run))} queued</span>
            </div>
            <div className="mt-2 text-[10px] text-muted-foreground">
              <RunDate run={run} />
            </div>
          </Link>
        ))}
        {!visible.length && (
          <p className="p-5 text-xs text-muted-foreground">No matching runs</p>
        )}
      </div>
    </Card>
  );
}
function RunContext({ run }: { run: Experiment }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-card px-5 py-4">
      <div>
        <div className="flex flex-wrap items-center gap-2 text-sm font-medium">
          {runLabel(run)}
          <Status passed={run.checks.passed} />
        </div>
        <p className="mt-2 font-mono text-[10px] text-muted-foreground">
          {run.run_id.slice(0, 8)} · <RunDate run={run} /> ·{" "}
          {stepsOf(run).length} tool steps
        </p>
      </div>
      <Button variant="outline" size="sm" asChild>
        <a href={evidenceUrl(run)}>
          <ArrowDownToLine className="size-3.5" />
          Export evidence
        </a>
      </Button>
    </div>
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
                        href={`/compare?run=${label === "Candidate" ? item.run_id : current.run_id}&baseline=${label === "Baseline" ? item.run_id : previous.run_id}`}
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
      <Card className="gap-0 overflow-hidden">
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
          <Link href={`/traces?run=${current.run_id}`}>
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
