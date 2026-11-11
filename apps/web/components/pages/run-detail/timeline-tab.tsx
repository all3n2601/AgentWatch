"use client";

import { useMemo, useState } from "react";
import {
  ArrowDownToLine,
  CalendarClock,
  Cpu,
  FileSearch,
  Gauge,
  Play,
  Ruler,
  ScrollText,
  Square,
  Tag,
  Timer,
  Waypoints,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { CopyChip } from "@/components/ui/copy-chip";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  KeyValueList,
  type KeyValueItem,
} from "@/components/widgets/evidence-lists";
import { SegmentBar, WorkWaitLegend } from "@/components/widgets/segment-bar";
import {
  TimelineChart,
  type TimelineChartRow,
} from "@/components/widgets/timeline-chart";
import { evidenceUrl } from "@/lib/dashboard-data";
import type { Experiment, Step } from "@/lib/experiment-schema";
import { formatQuantity } from "@/lib/measurement-format";
import { formatTiming, formatUtc, nsToIso } from "@/lib/run-detail";
import {
  SAVED_PHASE_LABELS,
  savedTimeline,
  secondsSinceOrigin,
  type SavedTimelineRow,
} from "@/lib/run-timeline";
import { cn } from "@/lib/utils";
import {
  FILL_PANEL,
  PANEL_SCROLL,
  TAB_GRID,
} from "@/components/shell/panel-layout";
import { StepTiming } from "./step-timing";

const MS_PER_SECOND = 1000;

const attributed = (step: Step) => step.resource !== "none";

function chartRow(row: SavedTimelineRow, originNs: number): TimelineChartRow {
  const { step } = row;
  const start = secondsSinceOrigin(step.submitted_ns, originNs);
  return {
    id: step.step_id,
    label: step.step_id,
    // Only attributed spans carry a badge, so the one that matters stands out in the list.
    detail: (
      <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        {SAVED_PHASE_LABELS[row.phase]}
        {attributed(step) && (
          <StatusBadge
            label="CPU queue"
            tone="warning"
            appearance="dot"
            className="px-2 py-0 text-xs"
          />
        )}
      </span>
    ),
    startShare: row.startShare,
    segments: row.segments,
    value: <StepTiming seconds={step.duration_seconds} />,
    description: `submitted at ${start.toFixed(3)} s; queue ${formatTiming(step.queue_seconds).accessibleText}, execution ${formatTiming(step.execution_seconds).accessibleText}`,
  };
}

/** Queue then execution for one span, in the unit its total reads best in. */
function spanSegments(step: Step) {
  const inMs = step.duration_seconds < 1;
  const scale = inMs ? MS_PER_SECOND : 1;
  return {
    unit: inMs ? ("ms" as const) : ("s" as const),
    segments: [
      {
        id: "queue",
        label: "Queue",
        value: step.queue_seconds * scale,
        kind: "wait" as const,
      },
      {
        id: "execution",
        label: "Execution",
        value: step.execution_seconds * scale,
        kind: "work" as const,
      },
    ],
  };
}

function spanItems(step: Step): KeyValueItem[] {
  return [
    {
      id: "submitted",
      icon: CalendarClock,
      label: "Submitted",
      value: formatUtc(nsToIso(step.submitted_ns)),
    },
    {
      id: "started",
      icon: Play,
      label: "Started",
      value: formatUtc(nsToIso(step.started_ns)),
    },
    {
      id: "finished",
      icon: Square,
      label: "Finished",
      value: formatUtc(nsToIso(step.finished_ns)),
    },
    {
      id: "cpu",
      icon: Cpu,
      label: "Process CPU",
      value: <StepTiming seconds={step.cpu_seconds} />,
    },
    {
      id: "share",
      icon: Gauge,
      label: "Queue share",
      value: formatQuantity(step.queue_fraction * 100, "percent"),
    },
    {
      id: "threshold",
      icon: Ruler,
      label: "Queue threshold",
      value: <StepTiming seconds={step.threshold_seconds} />,
    },
    {
      id: "attribution",
      icon: Tag,
      label: "Attribution",
      value: attributed(step) ? "CPU worker queue" : "None",
    },
    {
      id: "rule",
      icon: ScrollText,
      label: "Rule",
      value: step.rule_version,
    },
  ];
}

function SelectedSpan({ run, step }: { run: Experiment; step: Step }) {
  const split = spanSegments(step);
  return (
    <GlassPanel
      as="section"
      aria-labelledby="run-span-evidence"
      aria-live="polite"
      className={FILL_PANEL}
    >
      <PanelHeader
        titleId="run-span-evidence"
        icon={FileSearch}
        title="Selected span"
        description={step.step_id}
        className="fit:shrink-0"
        actions={<CopyChip value={step.checksum} label="checksum" />}
      />
      <div
        tabIndex={0}
        role="group"
        aria-label={`Recorded readings for ${step.step_id}`}
        className={cn(PANEL_SCROLL, "space-y-3")}
      >
        <InsetCard>
          <SegmentBar
            segments={split.segments}
            unit={split.unit}
            label={`Queue versus execution for ${step.step_id}`}
            showLabels
            showLegend={false}
          />
        </InsetCard>
        <InsetCard>
          <KeyValueList items={spanItems(step)} dense />
        </InsetCard>
        <InsetCard>
          <h3 className="mb-1 text-sm font-semibold text-ink">
            Recorded evidence
          </h3>
          <p className="text-sm text-ink-muted [overflow-wrap:anywhere]">
            {step.evidence}
          </p>
        </InsetCard>
        <Button asChild variant="glass" size="pill">
          <a href={evidenceUrl(run, "spans.jsonl")}>
            <ArrowDownToLine aria-hidden />
            Download spans.jsonl
          </a>
        </Button>
      </div>
    </GlassPanel>
  );
}

/**
 * Every recorded worker span on one wall-clock axis, beside the selected span's timestamps
 * and attribution evidence. Spans are in submission order, so the numbering is time order.
 */
export function TimelineTab({ run }: { run: Experiment }) {
  const timeline = useMemo(() => savedTimeline(run), [run]);
  const [selectedId, setSelectedId] = useState(run.contended.step_id);
  const selected =
    timeline.rows.find((row) => row.step.step_id === selectedId)?.step ??
    run.contended;
  const queued = timeline.rows.filter((row) => attributed(row.step)).length;

  return (
    <div className={cn(TAB_GRID, "fit:h-full")}>
      <GlassPanel
        as="section"
        aria-labelledby="run-timeline"
        className={FILL_PANEL}
      >
        <PanelHeader
          titleId="run-timeline"
          icon={Waypoints}
          title="Worker spans over time"
          description={`${timeline.rows.length} spans in a ${formatTiming(timeline.spanSeconds).text} window · ${queued} attributed to the CPU worker queue`}
          className="fit:shrink-0"
          actions={
            <WorkWaitLegend
              hatchedLabel="Queue"
              className="flex-wrap gap-x-5 gap-y-1"
            />
          }
        />
        <div className={PANEL_SCROLL}>
          <TimelineChart
            rows={timeline.rows.map((row) => chartRow(row, timeline.originNs))}
            ticks={timeline.ticks}
            label="Worker spans in submission order"
            selectedId={selected.step_id}
            onSelect={setSelectedId}
          />
        </div>
        <p className="mt-3 flex items-center gap-2 text-xs text-ink-muted fit:shrink-0">
          <Timer className="size-4 shrink-0" aria-hidden />
          Wall clock from the first submission. Queue is submission to worker
          start; execution is wall time after start, not pure useful work.
        </p>
      </GlassPanel>
      <SelectedSpan run={run} step={selected} />
    </div>
  );
}
