"use client";
import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  Check,
  Clock3,
  Cpu,
  Download,
  FileJson,
  Search,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { z } from "zod";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import type { Experiment, Step } from "@/lib/experiment-schema";
import { duration, percent, evidenceUrl, stepsOf } from "@/lib/dashboard-data";

export function Status({ passed }: { passed: boolean }) {
  return (
    <Badge
      variant="outline"
      className={
        passed
          ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-300"
          : "border-amber-400/20 bg-amber-400/5 text-amber-200"
      }
    >
      <span
        className={`size-1.5 rounded-full ${passed ? "bg-emerald-400" : "bg-amber-400"}`}
      />
      {passed ? "Passed" : "Inconclusive"}
    </Badge>
  );
}
export function PanelHeading({
  title,
  detail,
  children,
}: {
  title: string;
  detail?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
        {detail && (
          <p className="mt-1 text-xs text-muted-foreground">{detail}</p>
        )}
      </div>
      {children}
    </div>
  );
}
export function TraceExplorer({ run }: { run: Experiment }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState<Step | null>(null);
  const steps = stepsOf(run);
  const visible = steps.filter(
    (s) =>
      s.step_id.toLowerCase().includes(search.toLowerCase()) &&
      (filter === "all" || s.resource !== "none"),
  );
  const start = Math.min(...steps.map((s) => s.submitted_ns));
  const end = Math.max(...steps.map((s) => s.finished_ns));
  return (
    <div className="space-y-5">
      <Card className="gap-0 p-5 sm:p-6">
        <PanelHeading
          title="Worker trace"
          detail={`${steps.length} recorded spans · ${duration((end - start) / 1e9)} run window`}
        >
          <div className="flex gap-4 text-[11px] text-muted-foreground">
            <span>
              <i className="mr-1 inline-block size-2 rounded bg-amber-400" />
              Queue
            </span>
            <span>
              <i className="mr-1 inline-block size-2 rounded bg-violet-400" />
              Execution
            </span>
          </div>
        </PanelHeading>
        <div className="space-y-2">
          {steps.map((s) => (
            <button
              key={s.step_id}
              onClick={() => setSelected(s)}
              className="grid w-full grid-cols-[80px_1fr_55px] items-center gap-3 rounded-lg p-2 text-left text-xs transition-colors hover:bg-accent sm:grid-cols-[150px_1fr_75px]"
            >
              <span className="truncate text-muted-foreground">
                {s.step_id.replaceAll("-", " ")}
              </span>
              <span className="trace-grid relative h-10 rounded bg-background/40">
                <span
                  className="absolute top-2 flex h-6 min-w-px overflow-hidden rounded"
                  style={{
                    left: `${((s.submitted_ns - start) / (end - start)) * 100}%`,
                    width: `${((s.finished_ns - s.submitted_ns) / (end - start)) * 100}%`,
                  }}
                >
                  <span
                    className="bg-amber-400"
                    style={{ width: percent(s.queue_fraction) }}
                  />
                  <span className="flex-1 bg-violet-400" />
                </span>
              </span>
              <span className="text-right font-mono">
                {duration(s.duration_seconds)}
              </span>
            </button>
          ))}
        </div>
        <p className="mt-4 text-[11px] text-muted-foreground">
          Queue measured from submission to worker start. Execution is wall time
          after start.
        </p>
      </Card>
      <Card className="gap-0 overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 p-5">
          <PanelHeading
            title="Span inventory"
            detail="Select a span for measurements and attribution evidence"
          />
          <div className="flex flex-wrap gap-2">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 size-3.5 text-muted-foreground" />
              <Input
                aria-label="Search spans"
                placeholder="Search spans…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="h-8 w-44 pl-8 text-xs"
              />
            </div>
            <Button
              size="sm"
              variant={filter === "queued" ? "secondary" : "outline"}
              onClick={() => setFilter(filter === "all" ? "queued" : "all")}
            >
              Contended only
            </Button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="aw-table">
            <thead>
              <tr>
                <th>Span / operation</th>
                <th>Attribution</th>
                <th>Duration</th>
                <th>Queue</th>
                <th>Execution</th>
                <th>Process CPU</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((s) => (
                <tr key={s.step_id}>
                  <td>
                    <button
                      className="text-left font-medium text-foreground hover:text-primary"
                      onClick={() => setSelected(s)}
                    >
                      {s.step_id}
                      <span className="mt-1 block text-[11px] font-normal text-muted-foreground">
                        {run.kind === "coding"
                          ? "sandbox.run_tests"
                          : "sandbox.cpu_task"}
                      </span>
                    </button>
                  </td>
                  <td>
                    <Badge
                      variant="outline"
                      className={
                        s.resource !== "none"
                          ? "text-amber-200"
                          : "text-muted-foreground"
                      }
                    >
                      {s.resource === "none" ? "None" : "CPU worker queue"}
                    </Badge>
                  </td>
                  <td className="font-mono">{duration(s.duration_seconds)}</td>
                  <td className="font-mono text-amber-200">
                    {duration(s.queue_seconds)}
                  </td>
                  <td className="font-mono">{duration(s.execution_seconds)}</td>
                  <td className="font-mono">{duration(s.cpu_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!visible.length && (
          <p className="p-8 text-center text-sm text-muted-foreground">
            No spans match these filters.
          </p>
        )}
      </Card>
      <Sheet
        open={!!selected}
        onOpenChange={(open) => {
          if (!open) setSelected(null);
        }}
      >
        <SheetContent className="w-full overflow-y-auto sm:max-w-lg">
          <SheetHeader>
            <SheetTitle>{selected?.step_id}</SheetTitle>
            <SheetDescription>
              Recorded timing and attribution evidence
            </SheetDescription>
          </SheetHeader>
          {selected && (
            <div className="space-y-6 px-6 pb-6">
              <div className="grid grid-cols-2 gap-3">
                {[
                  ["Total", selected.duration_seconds],
                  ["Queue", selected.queue_seconds],
                  ["Execution", selected.execution_seconds],
                  ["Process CPU", selected.cpu_seconds],
                ].map(([label, value]) => (
                  <Card key={String(label)} className="gap-2 p-4">
                    <span className="text-xs text-muted-foreground">
                      {label}
                    </span>
                    <strong className="text-xl">
                      {duration(Number(value))}
                    </strong>
                  </Card>
                ))}
              </div>
              <p className="text-sm leading-6">{selected.evidence}</p>
              <dl className="space-y-4 text-xs">
                {[
                  ["Attribution", selected.resource],
                  ["Queue share", percent(selected.queue_fraction)],
                  ["Rule", selected.rule_version],
                  ["Threshold", duration(selected.threshold_seconds)],
                  [
                    "Submitted",
                    new Date(selected.submitted_ns / 1e6).toISOString(),
                  ],
                  [
                    "Started",
                    new Date(selected.started_ns / 1e6).toISOString(),
                  ],
                  [
                    "Finished",
                    new Date(selected.finished_ns / 1e6).toISOString(),
                  ],
                  ["Checksum", selected.checksum],
                ].map(([label, value]) => (
                  <div key={label} className="space-y-1">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="break-all font-mono">{value}</dd>
                  </div>
                ))}
              </dl>
              <Button asChild variant="outline">
                <a href={evidenceUrl(run)}>
                  <FileJson className="size-4" />
                  Download source evidence
                </a>
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
}

const sampleSchema = z.object({
  timestamp_ns: z.number().finite().nonnegative(),
  cpu_pressure: z
    .record(z.string(), z.record(z.string(), z.number().finite().nonnegative()))
    .nullable(),
  cgroup_cpu_stat: z
    .record(z.string(), z.number().finite().nonnegative())
    .nullable(),
  unavailable: z.array(z.string()),
});
export function Resources({ run }: { run: Experiment }) {
  const [state, setState] = useState<{
    samples: z.infer<typeof sampleSchema>[];
    loading: boolean;
    error: string;
  }>({ samples: [], loading: true, error: "" });
  useEffect(() => {
    const abort = new AbortController();
    fetch(evidenceUrl(run, "samples.jsonl"), { signal: abort.signal })
      .then(async (r) => {
        if (!r.ok) throw Error("Recorded samples could not be retrieved");
        return r.text();
      })
      .then((raw) => {
        const samples = raw
          .split("\n")
          .filter(Boolean)
          .map((line) => sampleSchema.parse(JSON.parse(line)));
        setState({ samples, loading: false, error: "" });
      })
      .catch((e) => {
        if (!abort.signal.aborted)
          setState({ samples: [], loading: false, error: e.message });
      });
    return () => abort.abort();
  }, [run]);
  const readings = state.samples.filter(
    (s) => s.cpu_pressure?.some?.avg10 !== undefined,
  );
  const chart = readings.map((s) => ({
    time: new Date(s.timestamp_ns / 1e6).toLocaleTimeString(),
    pressure: s.cpu_pressure!.some.avg10,
  }));
  const cgroup = state.samples.filter((s) => s.cgroup_cpu_stat);
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-3">
        {[
          [
            "Collected samples",
            state.loading ? "Loading…" : state.samples.length,
            "Raw sampler observations",
          ],
          ["Valid PSI readings", readings.length, "CPU pressure on Linux"],
          ["cgroup readings", cgroup.length, "Container CPU accounting"],
        ].map(([label, value, detail]) => (
          <Card key={String(label)} className="gap-2 p-5">
            <span className="text-xs text-muted-foreground">{label}</span>
            <strong className="text-2xl">{value}</strong>
            <span className="text-[11px] text-muted-foreground">{detail}</span>
          </Card>
        ))}
      </div>
      {state.error && (
        <p
          role="alert"
          className="rounded-lg border border-amber-400/20 p-4 text-sm text-amber-200"
        >
          {state.error}
        </p>
      )}
      <Card className="gap-0 p-6">
        <PanelHeading
          title="CPU pressure"
          detail="PSI avg10 · percentage of time waiting for CPU · host-wide signal"
        >
          <Button size="sm" asChild variant="outline">
            <a href={evidenceUrl(run, "samples.jsonl")}>
              <Download className="size-3.5" />
              Raw samples
            </a>
          </Button>
        </PanelHeading>
        {chart.length ? (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chart}>
                <CartesianGrid vertical={false} stroke="#28313c" />
                <XAxis dataKey="time" />
                <YAxis unit="%" />
                <Tooltip
                  contentStyle={{
                    background: "#171d27",
                    border: "1px solid #303948",
                    borderRadius: 8,
                  }}
                />
                <Area
                  dataKey="pressure"
                  stroke="#7dd3fc"
                  fill="#7dd3fc"
                  fillOpacity={0.12}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="flex min-h-44 flex-col items-center justify-center rounded-lg border border-dashed bg-background/30 text-center">
            <Cpu className="mb-3 size-6 text-muted-foreground" />
            <h3 className="text-sm font-medium">
              {state.loading
                ? "Reading recorded samples"
                : "No valid CPU pressure readings"}
            </h3>
            <p className="mt-2 max-w-md text-xs leading-6 text-muted-foreground">
              Linux PSI readings were unavailable in this run. Measured worker
              queue times remain available in Traces.
            </p>
          </div>
        )}
      </Card>
      <Card className="gap-0 p-6">
        <PanelHeading
          title="Evaluation coverage"
          detail="Samples within each measured step window"
        />
        <div className="grid gap-4 sm:grid-cols-2">
          {[run.clean, run.contended].map((s) => (
            <div key={s.step_id} className="rounded-lg border p-4">
              <div className="mb-4 flex items-center gap-2">
                <Clock3 className="size-4 text-muted-foreground" />
                <span className="text-sm capitalize">
                  {s.step_id} evaluation
                </span>
              </div>
              <div className="grid grid-cols-3 gap-3">
                {[
                  ["Samples", s.telemetry?.sample_count],
                  ["Valid PSI", s.telemetry?.valid_pressure_samples],
                  [
                    "Max avg10",
                    s.telemetry?.max_cpu_pressure_avg10_percent == null
                      ? null
                      : `${s.telemetry.max_cpu_pressure_avg10_percent.toFixed(2)}%`,
                  ],
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <p className="text-[10px] text-muted-foreground">{label}</p>
                    <p className="mt-1 font-mono">{value ?? "—"}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
      <Card className="gap-0 p-6">
        <PanelHeading
          title="Experiment checks"
          detail="Acceptance checks for this saved run"
        />
        <div className="grid gap-3 sm:grid-cols-2">
          {Object.entries(run.checks)
            .filter(([name]) => name !== "passed")
            .map(([name, passed]) => (
              <div
                key={name}
                className="flex items-center justify-between gap-3 rounded-lg border px-4 py-3 text-xs"
              >
                <span className="capitalize">{name.replaceAll("_", " ")}</span>
                <Status passed={passed} />
              </div>
            ))}
        </div>
      </Card>
      <Card className="gap-0 p-6">
        <PanelHeading title="Evidence & measurement limits" />
        <ul className="space-y-3 text-xs leading-6 text-muted-foreground">
          {run.limitations.map((item, i) => (
            <li key={i} className="flex gap-2">
              <Check className="mt-1 size-3 shrink-0" />
              {item}
            </li>
          ))}
        </ul>
        <div className="mt-5 flex flex-wrap gap-2">
          {["results.json", "spans.jsonl", "samples.jsonl"].map((file) => (
            <Button key={file} size="sm" variant="outline" asChild>
              <a href={evidenceUrl(run, file)}>
                <ArrowDownToLine className="size-3.5" />
                {file}
              </a>
            </Button>
          ))}
        </div>
      </Card>
    </div>
  );
}
