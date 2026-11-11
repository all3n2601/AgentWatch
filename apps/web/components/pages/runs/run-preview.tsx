import * as React from "react";
import Link from "next/link";
import {
  Activity,
  Clock,
  Database,
  Download,
  FileText,
  GitCompareArrows,
  Layers,
  ListOrdered,
  Radio,
  Scale,
  Stethoscope,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { CopyChip } from "@/components/ui/copy-chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { RatioChip } from "@/components/widgets/delta-chip";
import { Checklist, KeyValueList } from "@/components/widgets/evidence-lists";
import { Measurement } from "@/components/widgets/measurement";
import { SegmentBar } from "@/components/widgets/segment-bar";
import { EmptyState, UnavailableState } from "@/components/widgets/state-views";
import { displayName } from "@/lib/dataset-view";
import { formatQuantity } from "@/lib/measurement-format";
import { routes } from "@/lib/routes";
import {
  contendedOverClean,
  STATE_PRESENTATION,
  type RunsPageRow,
  type SavedRunEvidence,
} from "@/lib/runs-page";
import { cn } from "@/lib/utils";
import { RecordedTime, SourceChip } from "./run-row";

const MS_PER_SECOND = 1000;
const count = (value: number) => formatQuantity(value, "count");

function Stat({
  icon: Icon,
  value,
  label,
  className,
}: {
  icon: typeof Clock;
  value: React.ReactNode;
  label: React.ReactNode;
  className?: string;
}) {
  return (
    <InsetCard className={cn("flex items-start gap-2.5 p-3 sm:p-3", className)}>
      <Icon className="mt-0.5 size-5 shrink-0 text-ink" aria-hidden />
      <p className="min-w-0 text-sm text-ink-muted">
        <span className="block text-base font-semibold text-ink tabular-nums">
          {value}
        </span>
        {label}
      </p>
    </InsetCard>
  );
}

function RunIdentity({ row }: { row: RunsPageRow }) {
  const state = STATE_PRESENTATION[row.state];
  return (
    <InsetCard className="flex items-start gap-3">
      <FileText className="mt-1 size-6 shrink-0 text-ink" aria-hidden />
      <div className="min-w-0 flex-1">
        <h3 className="text-lg leading-tight font-semibold break-words text-ink">
          {row.title}
        </h3>
        <p className="mt-0.5 text-sm text-ink-muted">
          {displayName(row.kind)} ·{" "}
          {row.source === "live" ? "registered" : "finished"}{" "}
          <RecordedTime at={row.at} />
        </p>
        <div className="mt-2">
          <SourceChip row={row} />
        </div>
      </div>
      <StatusBadge label={state.label} tone={state.tone} />
    </InsetCard>
  );
}

function SavedEvidence({
  row,
  evidence,
}: {
  row: RunsPageRow;
  evidence: SavedRunEvidence;
}) {
  const ratio = contendedOverClean(evidence);
  const passedChecks = evidence.checks.filter((check) => check.passed).length;
  const resource = row.topResource ? displayName(row.topResource) : null;
  return (
    <>
      <InsetCard className="flex items-center gap-3 py-2.5 sm:py-2.5">
        <Layers className="size-5 shrink-0 text-ink" aria-hidden />
        <p className="min-w-0 text-sm text-ink">{evidence.workload}</p>
      </InsetCard>

      <div className="grid grid-cols-1 gap-2 min-[25rem]:grid-cols-2">
        <Stat
          icon={Clock}
          value={
            <Measurement
              value={evidence.queueSeconds + evidence.executionSeconds}
              unit="s"
            />
          }
          label={`step time, sum of ${count(evidence.stepCount)} steps`}
        />
        <Stat
          icon={Scale}
          value={<RatioChip ratio={ratio} className="text-sm" />}
          label="contended vs clean step · observed, not causal"
        />
        <Stat
          icon={ListOrdered}
          value={
            <Measurement
              value={evidence.contendedQueueSeconds * MS_PER_SECOND}
              unit="ms"
            />
          }
          label="queue wait in the contended step"
        />
        <Stat
          icon={Activity}
          value={`${count(evidence.attributedSteps)} of ${count(evidence.stepCount)}`}
          label={
            resource
              ? `steps attributed to ${resource} (rule ${evidence.ruleVersions.join(", ")})`
              : `steps attributed (rule ${evidence.ruleVersions.join(", ")})`
          }
        />
        <Stat
          icon={Database}
          value={`${count(evidence.telemetrySteps)} of ${count(evidence.stepCount)}`}
          label="steps with host telemetry"
          className="min-[25rem]:col-span-2"
        />
      </div>

      <InsetCard>
        <h3 className="mb-3 text-[15px] font-semibold text-ink">
          Execution versus queue wait
        </h3>
        <SegmentBar
          label={`Execution versus queue wait for ${row.title}, summed over ${count(evidence.stepCount)} steps`}
          unit="s"
          segments={[
            {
              id: "execution",
              label: "Execution",
              value: evidence.executionSeconds,
              kind: "work",
            },
            {
              id: "queue",
              label: "Queue wait",
              value: evidence.queueSeconds,
              kind: "wait",
            },
          ]}
        />
        <p className="mt-2 text-xs text-ink-muted">
          Summed over {count(evidence.baselineSteps)} baseline, 1 clean and 1
          contended step. Execution is wall time and can include contention, so
          it is not pure useful work.
        </p>
      </InsetCard>

      <InsetCard className="py-1 sm:py-1">
        <KeyValueList
          dense
          items={[
            {
              id: "model",
              label: "Model",
              value: evidence.model ?? "None (tool run)",
            },
            {
              id: "capacity",
              label: "Worker capacity",
              value: count(evidence.workerCapacity),
            },
            {
              id: "iterations",
              label: "Iterations",
              value: count(evidence.iterations),
            },
          ]}
        />
      </InsetCard>

      {evidence.checks.length > 0 && (
        <details className="glass-inset group px-3 py-1 sm:px-4">
          <summary className="flex min-h-11 cursor-pointer items-center rounded-lg text-[15px] font-semibold text-ink outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40">
            Acceptance checks · {count(passedChecks)} of{" "}
            {count(evidence.checks.length)} passed
          </summary>
          <Checklist
            className="pb-2"
            items={evidence.checks.map((check) => ({
              id: check.id,
              label: displayName(check.id),
              state: check.passed ? "pass" : "fail",
            }))}
          />
        </details>
      )}
    </>
  );
}

function PreviewActions({ row }: { row: RunsPageRow }) {
  const evidence = row.evidence;
  return (
    <div className="flex flex-wrap gap-2">
      <Button asChild variant="brand-soft" size="pill">
        <Link href={routes.run(row.id)}>
          <Stethoscope aria-hidden />
          Open diagnosis
        </Link>
      </Button>
      {evidence ? (
        <Button asChild variant="glass" size="pill">
          <Link href={routes.compare(row.id)}>
            <GitCompareArrows aria-hidden />
            Compare
          </Link>
        </Button>
      ) : (
        <Button asChild variant="glass" size="pill">
          <Link href={routes.live(row.id)}>
            <Radio aria-hidden />
            Live view
          </Link>
        </Button>
      )}
      <Button asChild variant="glass" size="pill">
        <a
          href={evidence ? evidence.evidenceHref : routes.liveExport(row.id)}
          download
        >
          <Download aria-hidden />
          {evidence && !evidence.persisted
            ? "Export local file"
            : "Export evidence"}
        </a>
      </Button>
    </div>
  );
}

/**
 * A run's evidence: the selected run beside the list from xl (below it
 * before), and each selected run on the Compare selection tab.
 */
export function RunPreview({
  row,
  id,
  title = "Run preview",
  headingRef,
  className,
}: {
  row: RunsPageRow | undefined;
  id: string;
  title?: string;
  headingRef?: React.Ref<HTMLHeadingElement>;
  className?: string;
}) {
  return (
    <GlassPanel
      as="section"
      id={id}
      aria-labelledby={`${id}-title`}
      className={cn("space-y-3", className)}
    >
      <PanelHeader
        className="mb-0"
        icon={Layers}
        title={
          <span ref={headingRef} tabIndex={-1} className="outline-none">
            {title}
          </span>
        }
        titleId={`${id}-title`}
        actions={row ? <CopyChip value={row.id} label="run ID" /> : undefined}
      />
      {!row && (
        <EmptyState
          title="No run selected"
          detail="Choose a run in the list to preview its evidence."
        />
      )}
      {row && <RunIdentity row={row} />}
      {row?.evidence && <SavedEvidence row={row} evidence={row.evidence} />}
      {row && !row.evidence && (
        <UnavailableState
          title="Step timings not in the live list"
          reason="The live run list reports each run's state only. Execution, queue wait, attribution and telemetry are read per run on its own page."
        />
      )}
      {row && <PreviewActions row={row} />}
    </GlassPanel>
  );
}
