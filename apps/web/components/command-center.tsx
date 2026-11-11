import Link from "next/link";
import { routes } from "@/lib/routes";
import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { duration, percent } from "@/lib/dashboard-data";
import { count, replayConditionName, sharePercent } from "@/lib/dataset-view";
import type { Dot, OverviewModel, WorstRun } from "@/lib/overview-model";
import { cn } from "@/lib/utils";

const CONDITION_COLOR: Record<string, string> = {
  clean: "var(--series-slate)",
  cpu: "var(--series-amber)",
  retrieval: "var(--series-sky)",
  inference: "var(--series-violet)",
  mixed: "var(--series-rose)",
};
const CONDITIONS = ["clean", "cpu", "retrieval", "inference", "mixed"] as const;

function Tile({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail: string;
}) {
  return (
    <Card className="gap-0 px-3 py-2.5">
      <span className="text-[11px] text-muted-foreground">{label}</span>
      <span className="metric-number text-xl font-semibold leading-tight">
        {value}
      </span>
      <span className="text-[11px] leading-snug text-muted-foreground">
        {detail}
      </span>
    </Card>
  );
}

function Panel({
  title,
  note,
  children,
  className,
}: {
  title: string;
  note?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn("gap-0 p-3", className)}>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="text-xs font-semibold tracking-tight">{title}</h2>
        {note && (
          <span className="truncate text-[11px] text-muted-foreground">
            {note}
          </span>
        )}
      </div>
      {children}
    </Card>
  );
}

/** A stable 0..1 value from an id, so dots keep their place between renders. */
const unit = (id: string) => {
  let hash = 0;
  for (const char of id) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return (hash % 1000) / 1000;
};

const WIDTH = 760;
const HEIGHT = 210;
const PAD = { left: 34, right: 10, top: 10, bottom: 34 };

function niceTicks(max: number) {
  const steps = [2, 5, 10, 20, 40].filter((step) => step <= max);
  return steps.length ? steps : [max];
}

/**
 * Every replay run as one dot: replay order across, total time up. Colour is the injected
 * condition, so the contention spikes and their clean controls read at a glance.
 */
function DotPlot({ dots }: { dots: Dot[] }) {
  const maxY = Math.max(...dots.map((dot) => dot.y), 5) * 1.05;
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const orders = [...new Set(dots.map((dot) => dot.x))].sort((a, b) => a - b);
  const slot = plotW / Math.max(orders.length, 1);
  const px = (dot: Dot) =>
    PAD.left +
    (orders.indexOf(dot.x) + 0.5) * slot +
    (unit(dot.id) - 0.5) * slot * 0.8;
  const py = (value: number) => PAD.top + plotH - (value / maxY) * plotH;
  const injectedAt = (order: number) => {
    const found = dots.find(
      (dot) => dot.x === order && dot.condition !== "clean",
    );
    return found
      ? replayConditionName(found.condition).replace(" contention", "")
      : null;
  };
  const counts = CONDITIONS.map(
    (id) => [id, dots.filter((dot) => dot.condition === id).length] as const,
  );
  return (
    <figure>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full"
        role="group"
        aria-label="Total time of every replay run by replay order, coloured by condition"
      >
        {niceTicks(maxY).map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={py(tick)}
              y2={py(tick)}
              stroke="var(--chart-grid)"
              strokeDasharray="3 3"
            />
            <text
              x={PAD.left - 6}
              y={py(tick) + 3}
              textAnchor="end"
              fontSize="10"
              fill="var(--chart-axis)"
            >
              {tick}s
            </text>
          </g>
        ))}
        {orders.map((order, index) => {
          const label = injectedAt(order);
          const cx = PAD.left + (index + 0.5) * slot;
          return (
            <g key={order}>
              <text
                x={cx}
                y={HEIGHT - 20}
                textAnchor="middle"
                fontSize="10"
                fill="var(--chart-axis)"
              >
                {order}
              </text>
              {label && (
                <text
                  x={cx}
                  y={HEIGHT - 7}
                  textAnchor="middle"
                  fontSize="9"
                  fill="var(--chart-axis)"
                >
                  {label}
                </text>
              )}
            </g>
          );
        })}
        {[...dots]
          .sort(
            (a, b) =>
              Number(a.condition !== "clean") - Number(b.condition !== "clean"),
          )
          .map((dot) => {
            const circle = (
              <circle
                cx={px(dot)}
                cy={py(dot.y)}
                r={2.6}
                fill={CONDITION_COLOR[dot.condition] ?? "var(--series-slate)"}
                fillOpacity={dot.condition === "clean" ? 0.55 : 0.9}
              >
                <title>{`${replayConditionName(dot.condition)} · replay ${dot.x} · ${duration(dot.y)}${dot.ratio === null ? "" : ` · ×${dot.ratio.toFixed(1)} vs clean`}`}</title>
              </circle>
            );
            return dot.liveId ? (
              <a key={dot.id} href={routes.run(dot.liveId)}>
                {circle}
              </a>
            ) : (
              <g key={dot.id}>{circle}</g>
            );
          })}
      </svg>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        {counts.map(([id, n]) => (
          <span key={id} className="flex items-center gap-1.5">
            <i
              className="size-2 rounded-full"
              style={{ background: CONDITION_COLOR[id] }}
              aria-hidden
            />
            {replayConditionName(id)} {count(n)}
          </span>
        ))}
        <span>
          Replays 1–5 are clean baselines; each condition is followed by a clean
          control.
        </span>
      </figcaption>
    </figure>
  );
}

function ConditionBars({ model }: { model: OverviewModel }) {
  const max = Math.max(...model.conditions.map((c) => c.ratio), 1);
  return (
    <Panel title="Slowdown by condition" note="median × clean">
      <ul className="space-y-2">
        {model.conditions.map((entry) => (
          <li
            key={entry.id}
            className="grid grid-cols-[78px_1fr_44px] items-center gap-2 text-[11px]"
          >
            <span className="truncate">
              {entry.label.replace(" contention", "")}
            </span>
            <span className="h-2 rounded-full bg-muted">
              <span
                className="block h-full rounded-full"
                style={{
                  width: `${(entry.ratio / max) * 100}%`,
                  background: CONDITION_COLOR[entry.id],
                }}
              />
            </span>
            <span className="text-right font-mono font-semibold">
              ×{entry.ratio.toFixed(1)}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] leading-4 text-muted-foreground">
        Across {model.experiments} experiments, each against its own clean
        replays. Observed, not a controlled comparison.
      </p>
    </Panel>
  );
}

function ResourceMix({ model }: { model: OverviewModel }) {
  const total = model.labelMix.reduce((sum, part) => sum + part.value, 0);
  return (
    <Panel title="Excess wait by resource" note={`${count(total)} steps`}>
      <div className="flex h-2 gap-0.5 overflow-hidden rounded-full">
        {model.labelMix.map((part) => (
          <span
            key={part.id}
            style={{ flexGrow: part.value, background: part.color }}
            title={`${part.label}: ${count(part.value)}`}
          />
        ))}
      </div>
      <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
        Share of steps with excess wait, by the resource the label names.
      </p>
      <ul className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
        {model.labelMix.map((part) => (
          <li key={part.id} className="flex items-center gap-1.5">
            <i
              className="size-2 shrink-0 rounded-full"
              style={{ background: part.color }}
              aria-hidden
            />
            <span className="flex-1 truncate">{part.label}</span>
            <span className="font-mono text-muted-foreground">
              {count(part.value)}
            </span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function TelemetryRing({ model }: { model: OverviewModel }) {
  const radius = 20;
  const circumference = 2 * Math.PI * radius;
  const share = model.telemetryPresent / model.steps;
  return (
    <Panel title="Host telemetry" note={`${count(model.steps)} steps`}>
      <div className="flex items-center gap-3">
        <div
          className="relative size-14 shrink-0"
          role="img"
          aria-label={`${sharePercent(model.telemetryPresent, model.steps)} of steps overlap a host sample`}
        >
          <svg viewBox="0 0 48 48" className="size-full -rotate-90">
            <circle
              cx="24"
              cy="24"
              r={radius}
              fill="none"
              stroke="var(--border)"
              strokeWidth="6"
            />
            <circle
              cx="24"
              cy="24"
              r={radius}
              fill="none"
              stroke="var(--series-emerald)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${Math.max(share, 0.01) * circumference} ${circumference}`}
            />
          </svg>
          <span className="absolute inset-0 grid place-items-center text-xs font-semibold">
            {sharePercent(model.telemetryPresent, model.steps)}
          </span>
        </div>
        <ul className="min-w-0 flex-1 space-y-0.5 text-[11px]">
          <li className="flex justify-between gap-2">
            <span className="text-muted-foreground">Overlap a sample</span>
            <span className="font-mono">{count(model.telemetryPresent)}</span>
          </li>
          <li className="flex justify-between gap-2">
            <span className="text-muted-foreground">Missed the step</span>
            <span className="font-mono">{count(model.telemetryMissed)}</span>
          </li>
          <li className="flex justify-between gap-2">
            <span className="text-muted-foreground">No samples</span>
            <span className="font-mono">
              {count(
                model.steps - model.telemetryPresent - model.telemetryMissed,
              )}
            </span>
          </li>
        </ul>
      </div>
      <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
        Steps without a sample have no host reading; that is not zero load.
      </p>
    </Panel>
  );
}

function DataHealth({ model }: { model: OverviewModel }) {
  const rows: [string, string][] = [
    [
      "Quality gate",
      model.health.gate.charAt(0).toUpperCase() + model.health.gate.slice(1),
    ],
    ["Anomalies", String(model.health.anomalies)],
    [
      "Train / test rows",
      `${count(model.health.trainRows)} / ${count(model.health.testRows)}`,
    ],
    ["Validation rows", count(model.health.validationRows)],
    ["Attribution accuracy", "Not reported"],
  ];
  return (
    <Panel title="Data health" note={model.release.split("+")[0]}>
      <dl className="space-y-1 text-[11px]">
        {rows.map(([name, value]) => (
          <div
            key={name}
            className="flex justify-between gap-2 border-b pb-1 last:border-0 last:pb-0"
          >
            <dt className="text-muted-foreground">{name}</dt>
            <dd className="font-medium">{value}</dd>
          </div>
        ))}
      </dl>
      <Link
        href="/quality"
        className="mt-2 inline-block text-[11px] text-primary hover:underline"
      >
        Open data quality
      </Link>
    </Panel>
  );
}

function WorstRuns({ runs }: { runs: WorstRun[] }) {
  return (
    <Panel
      title="Slowest runs"
      note="against each experiment's clean replays"
      className="overflow-hidden"
    >
      <div className="-mx-3 overflow-x-auto">
        <table className="aw-table aw-dense">
          <thead>
            <tr>
              <th>Run</th>
              <th>Task</th>
              <th>Condition</th>
              <th className="text-right">Time</th>
              <th className="text-right">vs clean</th>
              <th className="text-right">Queue</th>
              <th className="text-right">Slow steps</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.runId}>
                <td className="font-mono">
                  {run.liveId ? (
                    <Link
                      href={routes.run(run.liveId)}
                      className="text-foreground hover:text-primary"
                    >
                      {run.liveId.slice(0, 8)}
                    </Link>
                  ) : (
                    <span title="No live run is mapped to this dataset run">
                      {run.runId.slice(0, 8)}
                    </span>
                  )}
                </td>
                <td>{run.task}</td>
                <td>
                  <span className="inline-flex items-center gap-1.5">
                    <i
                      className="size-1.5 rounded-full"
                      style={{ background: CONDITION_COLOR[run.condition] }}
                      aria-hidden
                    />
                    {replayConditionName(run.condition).replace(
                      " contention",
                      "",
                    )}
                  </span>
                </td>
                <td className="text-right font-mono">
                  {duration(run.seconds)}
                </td>
                <td className="text-right font-mono font-bold">
                  ×{run.ratio.toFixed(1)}
                </td>
                <td className="text-right font-mono text-muted-foreground">
                  {duration(run.queueSeconds)}
                </td>
                <td className="text-right text-muted-foreground">
                  {run.slowSteps}/{run.steps}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function TaskTable({ model }: { model: OverviewModel }) {
  const injected = ["cpu", "retrieval", "inference", "mixed"] as const;
  return (
    <Panel
      title="Experiments by task"
      note="median × clean"
      className="overflow-hidden"
    >
      <div className="-mx-3 overflow-x-auto">
        <table className="aw-table aw-dense">
          <thead>
            <tr>
              <th>Task</th>
              {injected.map((id) => (
                <th key={id} className="text-right">
                  {replayConditionName(id).replace(" contention", "")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {model.tasks.map((task) => (
              <tr key={task.task}>
                <td>{task.task}</td>
                {injected.map((id) => (
                  <td key={id} className="text-right font-mono">
                    {task.ratios[id] === undefined
                      ? "—"
                      : `×${task.ratios[id].toFixed(1)}`}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Link
        href="/experiments"
        className="mt-2 inline-block text-[11px] text-primary hover:underline"
      >
        All {model.experiments} experiments
      </Link>
    </Panel>
  );
}

/**
 * The home screen: small tiles and panels, each answering one question about time lost against
 * the clean baseline. Saved-run figures come from the database, the rest from the dataset release.
 */
export function CommandCenter({
  model,
  saved,
  unavailable,
}: {
  model?: OverviewModel;
  saved: { runs: number; passed: number; queueFraction: number };
  unavailable?: string;
}) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {model ? (
          <>
            <Tile
              label="Dataset runs"
              value={count(model.runs)}
              detail={`${count(model.replayRuns)} replays, ${model.experiments} experiments`}
            />
            <Tile
              label="Labelled steps"
              value={count(model.steps)}
              detail={model.release.split("+")[0]}
            />
            <Tile
              label="Slowest vs clean"
              value={model.slowest ? `×${model.slowest.ratio.toFixed(1)}` : "—"}
              detail={
                model.slowest
                  ? `${replayConditionName(model.slowest.condition)}, ${duration(model.slowest.seconds)}`
                  : "no injected runs"
              }
            />
            <Tile
              label="Steps with excess wait"
              value={sharePercent(model.excessSteps, model.steps)}
              detail={`${count(model.excessSteps)} of ${count(model.steps)}`}
            />
            <Tile
              label="Host telemetry"
              value={sharePercent(model.telemetryPresent, model.steps)}
              detail="steps with a sample"
            />
          </>
        ) : null}
        <Tile
          label="Saved runs"
          value={count(saved.runs)}
          detail={`${count(saved.passed)} passed · ${percent(saved.queueFraction)} queued`}
        />
      </div>
      {unavailable && (
        <p
          role="status"
          className="rounded-md border border-amber-400/25 bg-amber-400/5 px-3 py-2 text-xs text-amber-200"
        >
          Dataset panels unavailable: {unavailable}
        </p>
      )}
      {model && (
        <>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            <Panel
              title="Every replay run"
              note="total time · one dot per run"
              className="xl:col-span-2"
            >
              <DotPlot dots={model.dots} />
            </Panel>
            <ConditionBars model={model} />
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <ResourceMix model={model} />
            <TelemetryRing model={model} />
            <DataHealth model={model} />
          </div>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            <div className="min-w-0 xl:col-span-2">
              <WorstRuns runs={model.worst} />
            </div>
            <div className="min-w-0">
              <TaskTable model={model} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
