import { Activity, Boxes, Database, Network, Shield, Tag } from "lucide-react";
import { Chip } from "@/components/ui/chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { Checklist, KeyValueList } from "@/components/widgets/evidence-lists";
import { SegmentBar } from "@/components/widgets/segment-bar";
import { formatQuantity } from "@/lib/measurement-format";
import type {
  QualityCheck,
  QualityPageData,
  SplitSummary,
} from "@/lib/quality-page";
import { CountShares, shareText } from "./count-shares";

const count = (value: number) => formatQuantity(value, "count");

const TELEMETRY_MEANINGS: Record<string, string> = {
  present: "A host resource sample overlapped the step.",
  no_overlap: "Samples were taken, but none overlapped the step.",
  empty: "No host sample was recorded for the step.",
};

interface PanelProps {
  data: QualityPageData;
}

export function TelemetryPanel({ data }: PanelProps) {
  const { telemetry, rows } = data;
  return (
    <GlassPanel as="section" aria-labelledby="quality-telemetry">
      <PanelHeader
        titleId="quality-telemetry"
        icon={Activity}
        title="Telemetry coverage"
        description="Whether a host resource sample overlapped each step"
        actions={<Chip size="sm">{count(telemetry.total)} steps</Chip>}
      />
      <InsetCard className="space-y-4">
        <SegmentBar
          label={`Host telemetry coverage across ${count(telemetry.total)} steps`}
          segments={telemetry.parts}
          unit="count"
        />
        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
          {telemetry.parts.map((part) => (
            <div key={part.id}>
              <dt className="font-medium text-ink">{part.label}</dt>
              <dd className="text-ink-muted">
                {TELEMETRY_MEANINGS[part.id] ??
                  "Status recorded by the release; not described here."}
              </dd>
            </div>
          ))}
        </dl>
      </InsetCard>
      <p className="mt-3 text-sm text-ink-muted">
        Missed step and No samples have no host reading. They are not zero load,
        and the release keeps them as empty values.
        {telemetry.total !== rows &&
          ` Coverage counts ${count(telemetry.total)} steps, but the release has ${count(rows)}.`}
      </p>
    </GlassPanel>
  );
}

// The longer label-basis list runs in two columns on one-screen desktops so
// both lists fit the tab.
const TWO_COLUMNS_ON_FIT =
  "fit:grid fit:grid-cols-2 fit:gap-x-6 fit:gap-y-2.5 fit:space-y-0";

export function LabelsPanel({ data }: PanelProps) {
  const none = data.labels.find((part) => part.id === "none")?.value ?? 0;
  return (
    <GlassPanel as="section" aria-labelledby="quality-labels">
      <PanelHeader
        titleId="quality-labels"
        icon={Tag}
        title="Labels"
        description={`Which resource caused the wait, and why each step got its label · ${data.labelVersions.join(", ")}`}
      />
      <div className="grid gap-4 md:grid-cols-2 fit:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <InsetCard>
          <h3 className="mb-3 text-[15px] font-semibold text-ink">
            Resource label{" "}
            <span className="text-xs font-normal text-ink-muted">
              steps · share of {count(data.rows)}
            </span>
          </h3>
          <CountShares
            label="Steps per resource label"
            parts={data.labels}
            total={data.rows}
          />
          <p className="mt-3 text-sm text-ink-muted">
            {shareText(none, data.rows)} of steps show no excess wait. Mixed is
            not one of the four prediction classes.
          </p>
        </InsetCard>
        <InsetCard>
          <h3 className="mb-3 text-[15px] font-semibold text-ink">
            Label basis{" "}
            <span className="text-xs font-normal text-ink-muted">
              rule outcome · steps
            </span>
          </h3>
          <CountShares
            label="Steps per label basis"
            parts={data.labelBasis}
            total={data.rows}
            className={TWO_COLUMNS_ON_FIT}
          />
        </InsetCard>
      </div>
    </GlassPanel>
  );
}

export function WorkloadPanel({ data }: PanelProps) {
  return (
    <GlassPanel as="section" aria-labelledby="quality-workload">
      <PanelHeader
        titleId="quality-workload"
        icon={Boxes}
        title="Workload mix"
        description={`Where the ${count(data.rows)} steps were recorded`}
      />
      <div className="grid gap-4 md:grid-cols-2">
        <InsetCard>
          <h3 className="mb-3 text-[15px] font-semibold text-ink">
            Workload kind
          </h3>
          <SegmentBar
            label="Steps by workload kind"
            segments={data.kinds}
            unit="count"
          />
        </InsetCard>
        <InsetCard>
          <h3 className="mb-3 text-[15px] font-semibold text-ink">
            Experiment condition
          </h3>
          <SegmentBar
            label="Steps by experiment condition"
            segments={data.conditions}
            unit="count"
          />
        </InsetCard>
      </div>
    </GlassPanel>
  );
}

export function CompositionPanel({ data }: PanelProps) {
  return (
    <GlassPanel as="section" aria-labelledby="quality-composition">
      <PanelHeader
        titleId="quality-composition"
        icon={Database}
        title="Dataset composition"
      />
      <KeyValueList
        dense
        items={[
          { id: "runs", label: "Runs", value: count(data.runs) },
          { id: "steps", label: "Steps", value: count(data.rows) },
          { id: "tasks", label: "Tasks", value: count(data.tasks) },
          {
            id: "experiments",
            label: "Experiments named by runs",
            value: count(data.experiments),
          },
          { id: "schema", label: "Schema", value: data.schemaVersion },
          {
            id: "release",
            label: "Release",
            value: <span className="break-all">{data.release}</span>,
          },
        ]}
      />
    </GlassPanel>
  );
}

function SplitCard({ part, total }: { part: SplitSummary; total: number }) {
  const labelText = part.labels
    .filter((label) => label.value > 0)
    .map((label) => `${label.label} ${count(label.value)}`)
    .join(" · ");
  return (
    <InsetCard className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="text-[15px] font-semibold text-ink capitalize">
          {part.name}
        </h3>
        <span className="text-sm text-ink-muted tabular-nums">
          {part.rows === 0
            ? "No rows assigned"
            : `${count(part.rows)} rows (${shareText(part.rows, total)}) · ${count(part.runs)} runs · ${count(part.groups)} task ${part.groups === 1 ? "group" : "groups"}`}
        </span>
      </div>
      {part.rows > 0 && (
        <>
          <SegmentBar
            label={`${part.name} label mix: ${labelText}`}
            segments={part.labels}
            unit="count"
            showLegend={false}
            className="[&>div]:h-3"
          />
          <p className="text-xs text-ink-muted tabular-nums">{labelText}</p>
        </>
      )}
    </InsetCard>
  );
}

export function SplitPanel({ data }: PanelProps) {
  const { split } = data;
  const legend = data.labels.filter((label) =>
    split.parts.some((part) =>
      part.labels.some((entry) => entry.id === label.id && entry.value > 0),
    ),
  );
  const validation = split.parts.find((part) => part.name === "validation");
  return (
    <GlassPanel as="section" aria-labelledby="quality-splits">
      <PanelHeader
        titleId="quality-splits"
        icon={Network}
        title="Split composition"
        description={`${split.id} · grouped by ${split.contract.groupKey}`}
      />
      <ul
        aria-label="Label colours"
        className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink"
      >
        {legend.map((label) => (
          <li key={label.id} className="flex items-center gap-1.5">
            <i
              aria-hidden
              className="size-2.5 rounded-[3px]"
              style={{ background: label.color }}
            />
            {label.label}
          </li>
        ))}
      </ul>
      <div className="grid gap-3 lg:grid-cols-[repeat(auto-fit,minmax(16rem,1fr))]">
        {split.parts.map((part) => (
          <SplitCard key={part.name} part={part} total={data.rows} />
        ))}
      </div>
      {validation?.rows === 0 && (
        <p className="mt-3 text-sm text-ink-muted">
          This split assigns no rows to validation. Choose thresholds before
          evaluating on test.
        </p>
      )}
      <div className="mt-3">
        <h3 className="mb-2 text-sm font-medium text-ink">
          Injection methods only in test{" "}
          <span className="font-normal text-ink-muted">
            of {count(split.heldOutMethods.length)} recorded
          </span>
        </h3>
        {split.onlyInTest.length ? (
          <ul className="flex flex-wrap gap-2">
            {split.onlyInTest.map((method) => (
              <li key={method}>
                <Chip size="sm" className="font-mono">
                  {method}
                </Chip>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-ink-muted">None recorded.</p>
        )}
      </div>
    </GlassPanel>
  );
}

const CHECK_GROUPS: readonly { id: QualityCheck["group"]; title: string }[] = [
  { id: "gate", title: "Quality gate" },
  { id: "split", title: "Split checks" },
];

const validatedFormat = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: "UTC",
});

function validatedLabel(timestamp: string) {
  const moment = new Date(timestamp);
  return Number.isNaN(moment.getTime())
    ? `at an unreadable time (${timestamp})`
    : `${validatedFormat.format(moment)} UTC`;
}

export function ValidationPanel({ data }: PanelProps) {
  return (
    <GlassPanel as="section" aria-labelledby="quality-validation">
      <PanelHeader
        titleId="quality-validation"
        icon={Shield}
        title="Validation"
        description={`Quality gate and split checks as recorded · validated ${validatedLabel(data.gate.validatedAt)}`}
      />
      <div className="grid gap-4 lg:grid-cols-2">
        {CHECK_GROUPS.map((group) => {
          const items = data.checks.filter((check) => check.group === group.id);
          return (
            <InsetCard key={group.id}>
              <h3 className="text-[15px] font-semibold text-ink">
                {group.title}
              </h3>
              {items.length ? (
                <Checklist items={items} showStateBadge />
              ) : (
                <p className="mt-2 text-sm text-ink-muted">
                  No checks recorded.
                </p>
              )}
            </InsetCard>
          );
        })}
      </div>
    </GlassPanel>
  );
}
