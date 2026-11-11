"use client";

import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  chartAxisTick,
  chartCursor,
  chartGridStroke,
  chartTooltipStyle,
  LegendKey,
} from "@/components/chart-kit";
import { PanelHeading } from "@/components/run-views";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { DatasetSnapshot } from "@/lib/dataset-schema";
import {
  count,
  displayName,
  labelSeries,
  LABEL_RESOURCES,
  missingColumns,
  segments,
  sharePercent,
  type Segment,
} from "@/lib/dataset-view";

const SEGMENT_COLORS = [
  "var(--series-violet)",
  "var(--series-sky)",
  "var(--series-amber)",
  "var(--series-slate)",
];

/** One horizontal bar split by share, with a legend that repeats the numbers as text. */
export function ShareBar({
  label,
  parts,
  colors = SEGMENT_COLORS,
}: {
  label: string;
  parts: Segment[];
  colors?: string[];
}) {
  const total = parts.reduce((sum, part) => sum + part.value, 0);
  return (
    <figure aria-label={label} className="space-y-3">
      <div className="flex h-2.5 gap-0.5 overflow-hidden rounded-full">
        {parts.map((part, index) => (
          <div
            key={part.id}
            title={`${part.label}: ${count(part.value)}`}
            className="min-w-px first:rounded-l-full last:rounded-r-full"
            style={{
              flexGrow: part.value,
              background: colors[index % colors.length],
            }}
          />
        ))}
      </div>
      <ul className="grid gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
        {parts.map((part, index) => (
          <li key={part.id} className="flex items-center gap-2">
            <i
              className="size-2 shrink-0 rounded-sm"
              style={{ background: colors[index % colors.length] }}
              aria-hidden
            />
            <span className="min-w-0 flex-1 truncate">{part.label}</span>
            <span className="font-mono tabular-nums text-muted-foreground">
              {count(part.value)} · {sharePercent(part.value, total)}
            </span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

function Stat({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: ReactNode;
  detail: string;
  tone?: "good";
}) {
  return (
    <div className="bg-card p-4 sm:px-5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="metric-number mt-2 flex items-center gap-2 text-[26px] font-semibold leading-none">
        {value}
        {tone === "good" && (
          <span className="size-2 rounded-full bg-emerald-400" aria-hidden />
        )}
      </div>
      <p className="mt-2 truncate text-[11px] text-muted-foreground">
        {detail}
      </p>
    </div>
  );
}

export function HorizontalBars({
  data,
  height,
  label,
  labelWidth = 150,
  unit = "",
}: {
  data: { name: string; value: number; color: string }[];
  height: number;
  label: string;
  labelWidth?: number;
  unit?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <div
        role="img"
        aria-label={label}
        style={{ height, minWidth: labelWidth + 300 }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 8, right: 56, top: 4, bottom: 4 }}
          >
            <CartesianGrid horizontal={false} stroke={chartGridStroke} />
            <XAxis type="number" hide />
            <YAxis
              type="category"
              dataKey="name"
              width={labelWidth}
              tickLine={false}
              axisLine={false}
              tick={chartAxisTick}
            />
            <Tooltip
              cursor={chartCursor}
              contentStyle={chartTooltipStyle}
              formatter={(value) => `${count(Number(value))}${unit}`}
            />
            <Bar
              isAnimationActive={false}
              dataKey="value"
              radius={[0, 4, 4, 0]}
              barSize={16}
              label={{
                position: "right",
                fill: "var(--chart-axis)",
                fontSize: 11,
                formatter: (value: unknown) => `${count(Number(value))}${unit}`,
              }}
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SplitComposition({
  split,
}: {
  split: Extract<DatasetSnapshot, { status: "ready" }>["split"];
}) {
  const order = ["train", "validation", "test"].filter(
    (name) => name in split.splits,
  );
  const rows = order.map((name) => {
    const part = split.splits[name];
    return {
      name,
      rows: part.rows,
      runs: part.runs,
      groups: part.groups,
      ...Object.fromEntries(
        LABEL_RESOURCES.map((entry) => [
          entry.id,
          part.label_resource[entry.id] ?? 0,
        ]),
      ),
    };
  });
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {LABEL_RESOURCES.map((entry) => (
          <LegendKey key={entry.id} color={entry.color} label={entry.label} />
        ))}
      </div>
      <ul className="space-y-4">
        {rows.map((row) => (
          <li key={row.name}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3 text-xs">
              <span className="font-medium capitalize">{row.name}</span>
              <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                {row.rows === 0
                  ? "no rows assigned"
                  : `${count(row.rows)} rows · ${count(row.runs)} runs · ${count(row.groups)} task ${row.groups === 1 ? "group" : "groups"}`}
              </span>
            </div>
            <div
              className="flex h-3 gap-0.5 overflow-hidden rounded-full bg-muted"
              role="img"
              aria-label={`${row.name} label mix: ${LABEL_RESOURCES.map(
                (entry) =>
                  `${entry.label} ${count(Number(row[entry.id as keyof typeof row]))}`,
              ).join(", ")}`}
            >
              {LABEL_RESOURCES.map((entry) => (
                <div
                  key={entry.id}
                  title={`${entry.label}: ${count(Number(row[entry.id as keyof typeof row]))}`}
                  style={{
                    flexGrow: Number(row[entry.id as keyof typeof row]),
                    background: entry.color,
                  }}
                />
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function AucBars({
  data,
  label,
  labelWidth = 170,
}: {
  data: { name: string; auc: number }[];
  label: string;
  labelWidth?: number;
}) {
  return (
    <div className="overflow-x-auto">
      <div
        role="img"
        aria-label={label}
        style={{ height: data.length * 26 + 28, minWidth: labelWidth + 280 }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 8, right: 40, top: 4, bottom: 4 }}
          >
            <CartesianGrid horizontal={false} stroke={chartGridStroke} />
            <XAxis
              type="number"
              domain={[0.4, 1]}
              ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]}
              tickLine={false}
              axisLine={false}
              tick={chartAxisTick}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={labelWidth}
              tickLine={false}
              axisLine={false}
              tick={chartAxisTick}
            />
            <ReferenceLine
              x={0.5}
              stroke="var(--chart-axis)"
              strokeDasharray="3 3"
            />
            <Tooltip
              cursor={chartCursor}
              contentStyle={chartTooltipStyle}
              formatter={(value) => Number(value).toFixed(3)}
            />
            <Bar
              isAnimationActive={false}
              dataKey="auc"
              radius={[0, 4, 4, 0]}
              barSize={12}
              fill="var(--series-violet)"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function DatasetSummary({ snapshot }: { snapshot: DatasetSnapshot }) {
  if (snapshot.status === "unavailable") {
    return (
      <Card className="items-center gap-3 border-dashed p-12 text-center">
        <h2 className="text-base font-semibold">Dataset unavailable</h2>
        <p className="max-w-lg text-sm leading-6 text-muted-foreground">
          {snapshot.reason}
        </p>
      </Card>
    );
  }
  const { manifest, split, validation } = snapshot;
  const telemetry = segments(manifest.telemetry_status);
  const present = manifest.telemetry_status.present ?? 0;
  const labels = labelSeries(manifest.label_resource);
  const excess = manifest.rows - (manifest.label_resource.none ?? 0);
  const missing = missingColumns(manifest, 12);
  const designFeatures = split.design_encoding.features.slice(0, 8);
  const leakage = split.leakage.features.slice(0, 6);
  const gatePassed =
    validation.status === "passed" && validation.anomalies.length === 0;

  const emptyValues = (
    <Card className="gap-0 p-5">
      <PanelHeading
        title="Empty values by column"
        detail="Share of rows where the column has no value. Some columns apply to one workload kind only."
      />
      <ul className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
        {missing.map((column) => (
          <li key={column.column} className="text-xs">
            <div className="mb-1 flex justify-between gap-3">
              <span className="truncate font-mono">{column.column}</span>
              <span className="font-mono tabular-nums text-muted-foreground">
                {count(column.missing)} ·{" "}
                {sharePercent(column.missing, manifest.rows)}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-[var(--series-amber)]"
                style={{ width: `${column.share * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-4 text-xs text-muted-foreground">
        Empty means not recorded. It is never charted or averaged as zero.
      </p>
    </Card>
  );

  return (
    <div className="space-y-4">
      <Card className="grid grid-cols-2 gap-px overflow-hidden bg-border p-0 lg:grid-cols-5 [&>*:last-child]:col-span-2 lg:[&>*:last-child]:col-span-1">
        <Stat
          label="Runs"
          value={count(snapshot.runs.length)}
          detail={`${count(manifest.rows)} labelled steps · ${manifest.tasks} tasks`}
        />
        <Stat
          label="Steps with excess wait"
          value={count(excess)}
          detail={`${sharePercent(excess, manifest.rows)} of all steps`}
        />
        <Stat
          label="Host telemetry present"
          value={sharePercent(present, manifest.rows)}
          detail={`${count(present)} steps overlap a sample`}
        />
        <Stat
          label="Quality gate"
          value={gatePassed ? "Passed" : displayName(validation.status)}
          tone={gatePassed ? "good" : undefined}
          detail={`${validation.anomalies.length} anomalies · TFDV ${validation.tfdv_version}`}
        />
        <Stat
          label="Release"
          value={manifest.schema_version}
          detail={`${split.split_method} · ${manifest.dataset_version.split("+")[1]}`}
        />
      </Card>

      <div className="grid grid-cols-1 gap-4 min-[1100px]:grid-cols-[1.5fr_1fr]">
        <Card className="gap-0 p-5">
          <PanelHeading
            title="Which resource caused the wait"
            detail="Rows per label. Mixed rows are not one of the four prediction classes."
          />
          <HorizontalBars
            label="Rows per resource label"
            height={labels.length * 40 + 12}
            data={labels.map((entry) => ({
              name: entry.label,
              value: entry.value,
              color: entry.color,
            }))}
          />
          <p className="mt-3 text-xs text-muted-foreground">
            {sharePercent(manifest.label_resource.none ?? 0, manifest.rows)} of
            steps show no excess wait, so any classifier must beat a strong
            majority class.
          </p>
        </Card>
        <Card className="gap-0 p-5">
          <PanelHeading
            title="Workload mix"
            detail="Where the rows were recorded"
          />
          <div className="space-y-6">
            <div>
              <p className="mb-2 text-xs font-medium">Workload kind</p>
              <ShareBar
                label="Rows by workload kind"
                parts={segments(manifest.kind)}
              />
            </div>
            <div>
              <p className="mb-2 text-xs font-medium">Experiment condition</p>
              <ShareBar
                label="Rows by experiment condition"
                parts={segments(manifest.condition)}
                colors={[
                  "var(--series-slate)",
                  "var(--series-emerald)",
                  "var(--series-rose)",
                ]}
              />
            </div>
            <p className="border-t pt-4 text-[11px] leading-5 text-muted-foreground">
              Labelled with {manifest.label_version.replaceAll(",", " and ")}.
            </p>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 min-[1100px]:grid-cols-2">
        <Card className="gap-0 p-5">
          <PanelHeading
            title="Why each row got its label"
            detail="The rule outcome behind the label, from the label basis column"
          />
          <HorizontalBars
            label="Rows per label basis"
            labelWidth={210}
            height={Object.keys(manifest.label_basis).length * 40 + 12}
            data={segments(manifest.label_basis).map((part) => ({
              name: part.label,
              value: part.value,
              color: part.id.startsWith("mixed")
                ? "var(--series-rose)"
                : part.id.startsWith("contention")
                  ? "var(--series-amber)"
                  : "var(--series-slate)",
            }))}
          />
        </Card>
        <div className="flex min-w-0 flex-col gap-4">
          <Card className="gap-0 p-5">
            <PanelHeading
              title="Host telemetry coverage"
              detail="Whether a resource sample overlapped the step"
            />
            <ShareBar
              label="Rows by telemetry status"
              parts={telemetry}
              colors={[
                "var(--series-slate)",
                "var(--series-amber)",
                "var(--series-emerald)",
              ]}
            />
            <p className="mt-4 text-xs leading-5 text-muted-foreground">
              Steps without an overlapping sample have no host reading. They are
              not zero load, and the dataset keeps them as empty values.
            </p>
          </Card>

          <Card className="gap-0 p-5">
            <PanelHeading
              title="Held-out evidence"
              detail="Checks that keep test data independent of training"
            />
            <dl className="space-y-3 text-xs">
              {[
                ["Class balance", split.class_balance.status],
                [
                  "Held-out injection method",
                  split.held_out_injection_methods.status,
                ],
                ["Single-feature leakage check", split.leakage.status],
              ].map(([name, status]) => (
                <div
                  key={name}
                  className="flex items-center justify-between gap-3 border-b pb-3"
                >
                  <dt className="text-muted-foreground">{name}</dt>
                  <dd>
                    <Badge variant="outline" className="capitalize">
                      {status.replaceAll("_", " ")}
                    </Badge>
                  </dd>
                </div>
              ))}
            </dl>
            <p className="mb-2 mt-4 text-xs font-medium">
              Only in the test split
            </p>
            <div className="flex flex-wrap gap-1.5">
              {split.held_out_injection_methods.only_in_test.map((method) => (
                <Badge
                  key={method}
                  variant="secondary"
                  className="font-mono text-[10px]"
                >
                  {method}
                </Badge>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <div>
        <Card className="gap-0 p-5">
          <PanelHeading
            title="How the rows were split"
            detail={`${split.split_id.split("+")[0]} · grouped by ${split.contract.group_key} · ${split.contract.version} ${split.contract.status}`}
          />
          <SplitComposition split={split} />
          {split.splits.validation?.rows === 0 && (
            <p className="mt-4 text-xs text-muted-foreground">
              This split assigns no rows to validation. Choose thresholds before
              evaluating on test.
            </p>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 min-[1100px]:grid-cols-2">
        <Card className="gap-0 p-5">
          <PanelHeading
            title="Feature signal"
            detail={`AUC of each feature alone against the label, ${count(split.design_encoding.rows)} rows. 0.5 means no signal; report only.`}
          />
          <AucBars
            label="Single-feature AUC"
            data={designFeatures.map((f) => ({ name: f.feature, auc: f.auc }))}
          />
        </Card>
        <Card className="gap-0 p-5">
          <PanelHeading
            title="Strongest per-class signals"
            detail={`Leakage check: no single feature may exceed ${split.leakage.max_single_feature_auc.toFixed(2)} AUC`}
          />
          <AucBars
            label="Strongest feature per class, AUC"
            labelWidth={280}
            data={leakage.map((f) => ({
              name: `${displayName(f.class)} · ${f.feature}`,
              auc: f.auc,
            }))}
          />
        </Card>
      </div>
      {emptyValues}
    </div>
  );
}
