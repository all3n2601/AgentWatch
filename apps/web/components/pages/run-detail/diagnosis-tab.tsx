"use client";

import { useState } from "react";
import { BarChart3, ChevronRight, FileSearch, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { AlertBanner } from "@/components/widgets/alert-banner";
import { RatioChip } from "@/components/widgets/delta-chip";
import {
  KeyValueList,
  type KeyValueItem,
} from "@/components/widgets/evidence-lists";
import { Measurement } from "@/components/widgets/measurement";
import { SegmentBar, WorkWaitLegend } from "@/components/widgets/segment-bar";
import { EmptyState } from "@/components/widgets/state-views";
import {
  contextSegments,
  formatTiming,
  longestMeasured,
  rowBarShares,
  type EvidenceRow,
  type RunDetailModel,
  type RunDetailStep,
} from "@/lib/run-detail";
import { cn } from "@/lib/utils";
import { StepTiming } from "./step-timing";
import {
  FILL_PANEL,
  PANEL_SCROLL,
  TAB_GRID,
} from "@/components/shell/panel-layout";

/** Rows shown before "Show all steps"; long live runs stay scannable. */
const INITIAL_ROWS = 10;

function RowBar({
  step,
  longestSeconds,
}: {
  step: RunDetailStep;
  longestSeconds: number;
}) {
  const shares = rowBarShares(step, longestSeconds);
  if (!shares)
    return (
      <span className="text-[13px] text-ink-subtle">Split not measured</span>
    );
  return (
    <span
      aria-hidden
      className="flex h-3.5 w-full gap-[2px] overflow-hidden rounded-[5px] bg-bar-track"
    >
      <span
        className="h-full bg-bar"
        style={{
          width: `${shares.execution * 100}%`,
          minWidth: shares.execution > 0 ? 2 : 0,
        }}
      />
      <span
        className="hatch h-full"
        style={{
          width: `${shares.queue * 100}%`,
          minWidth: shares.queue > 0 ? 2 : 0,
        }}
      />
    </span>
  );
}

function RankedSteps({
  steps,
  selectedId,
  onSelect,
}: {
  steps: RunDetailStep[];
  selectedId: string;
  onSelect: (stepId: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? steps : steps.slice(0, INITIAL_ROWS);
  const longestSeconds = longestMeasured(steps);
  return (
    <>
      <div className={PANEL_SCROLL}>
        <ol
          className="@container space-y-1.5"
          aria-label="Steps ranked by queue delay"
        >
          {shown.map((step, index) => {
            const selected = step.stepId === selectedId;
            return (
              <li key={step.stepId}>
                <button
                  type="button"
                  aria-pressed={selected}
                  onClick={() => onSelect(step.stepId)}
                  className={cn(
                    "grid min-h-11 w-full grid-cols-[1.75rem_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1.5 rounded-xl border border-transparent px-3 py-2 text-left text-sm text-ink outline-none transition-colors hover:bg-glass-strong focus-visible:ring-[3px] focus-visible:ring-brand/40",
                    "@2xl:grid-cols-[1.75rem_minmax(0,1.3fr)_5rem_minmax(0,1.4fr)_minmax(0,1.1fr)_auto_1rem]",
                    selected && "border-success-line bg-brand-soft shadow-glow",
                  )}
                >
                  <span className="font-medium text-ink-muted tabular-nums">
                    {index + 1}.
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate font-medium">
                      {step.label}
                    </span>
                    {step.label !== step.stepId && (
                      <span className="block truncate text-xs text-ink-muted">
                        {step.stepId}
                      </span>
                    )}
                  </span>
                  <StepTiming
                    seconds={step.queueSeconds}
                    className="text-right font-semibold"
                  />
                  <span className="col-span-3 flex items-center @2xl:col-span-1">
                    <RowBar step={step} longestSeconds={longestSeconds} />
                  </span>
                  <span className="col-span-2 min-w-0 @2xl:col-span-1">
                    <span className="block truncate">
                      {step.resourceLabel ?? "No resource"}
                    </span>
                    <span className="block truncate text-xs text-ink-muted">
                      {step.basisLabel}
                    </span>
                  </span>
                  <StatusBadge
                    label={step.flag.label}
                    tone={step.flag.tone}
                    appearance="dot"
                    className="justify-self-end"
                  />
                  <ChevronRight
                    aria-hidden
                    className="hidden size-4 text-ink-muted @2xl:block"
                  />
                  <span className="sr-only">
                    {`queued; execution ${formatTiming(step.executionSeconds).accessibleText}`}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
      {steps.length > INITIAL_ROWS && (
        <Button
          type="button"
          variant="glass"
          size="pill"
          className="mt-3 self-start fit:shrink-0"
          aria-expanded={showAll}
          onClick={() => setShowAll((value) => !value)}
        >
          {showAll
            ? `Show top ${INITIAL_ROWS}`
            : `Show all ${steps.length} steps`}
        </Button>
      )}
    </>
  );
}

function evidenceValue(row: EvidenceRow) {
  switch (row.kind) {
    case "seconds":
      return <StepTiming seconds={row.value} />;
    case "percent":
      return <Measurement value={row.value} unit="percent" showState />;
    case "count":
      return <Measurement value={row.value} unit="count" showState />;
    case "text":
      return row.value === null ? (
        <span className="text-ink-subtle">Not recorded</span>
      ) : (
        <span className="[overflow-wrap:anywhere]">{row.value}</span>
      );
  }
}

const evidenceItems = (step: RunDetailStep): KeyValueItem[] =>
  step.evidence.map((row) => ({
    id: row.id,
    label: row.label,
    value: evidenceValue(row),
  }));

function SelectedContext({ step }: { step: RunDetailStep }) {
  const split = contextSegments(step);
  return (
    <InsetCard>
      <h3 className="mb-3 flex min-w-0 items-center gap-2 text-[15px] font-semibold text-ink">
        <BarChart3 className="size-4 shrink-0" aria-hidden />
        Queue versus execution
      </h3>
      {split ? (
        <SegmentBar
          segments={split.segments}
          unit={split.unit}
          label={`Queue versus execution for ${step.label}`}
          showLabels
        />
      ) : (
        <p className="rounded-xl border border-dashed border-line px-4 py-3 text-sm text-ink-muted">
          Queue and execution were not both measured for this step, so no split
          is drawn.
        </p>
      )}
      <p className="mt-3 text-xs text-ink-muted">
        Queue is measured delay before the step started. Execution is wall time
        on the resource, not pure useful work.
      </p>
    </InsetCard>
  );
}

/** The selected step's split and the readings it rests on; scrolls inside its column. */
function SelectedEvidence({ step }: { step: RunDetailStep }) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="run-step-evidence"
      aria-live="polite"
      className={FILL_PANEL}
    >
      <PanelHeader
        titleId="run-step-evidence"
        icon={FileSearch}
        title="Evidence for selected step"
        className="fit:shrink-0"
        description={
          <span className="[overflow-wrap:anywhere]">
            {step.label}
            {step.resourceLabel && ` · ${step.resourceLabel}`}
          </span>
        }
      />
      {/* Focusable so keyboard users can scroll the readings when they overflow the column. */}
      <div
        tabIndex={0}
        role="group"
        aria-label={`Evidence readings for ${step.label}`}
        className={cn(PANEL_SCROLL, "space-y-3")}
      >
        <SelectedContext step={step} />
        <InsetCard>
          <KeyValueList items={evidenceItems(step)} dense />
        </InsetCard>
      </div>
    </GlassPanel>
  );
}

/**
 * Diagnosis-first view: the headline, steps ranked by measured queue delay beside the
 * selected step's split and evidence. On `fit` screens it fills the tab panel and the step
 * list and evidence scroll inside their own panels. No confidence figure is shown; none is
 * calibrated for this data.
 */
export function DiagnosisTab({ model }: { model: RunDetailModel }) {
  const [selectedId, setSelectedId] = useState(model.steps[0]?.stepId ?? "");
  const selected =
    model.steps.find((step) => step.stepId === selectedId) ?? model.steps[0];

  return (
    <div className="space-y-4 fit:flex fit:h-full fit:min-h-0 fit:flex-col fit:gap-3 fit:space-y-0">
      {model.headline && (
        <AlertBanner
          tone={model.headline.tone}
          title={model.headline.title}
          detail={model.headline.detail}
          // The banner's text column has a zero flex basis, so on phones the aside would squeeze
          // it to a word per line; a minimum width makes the aside wrap below it instead.
          className="fit:shrink-0 fit:py-3 [&>div]:min-w-48"
          aside={
            model.contendedOverClean !== null ? (
              <span className="flex items-center gap-2">
                Contended vs clean
                <RatioChip ratio={model.contendedOverClean} />
              </span>
            ) : undefined
          }
        />
      )}

      {selected ? (
        <div className={cn(TAB_GRID, "fit:min-h-0 fit:flex-1")}>
          <GlassPanel
            as="section"
            aria-labelledby="run-ranked-steps"
            className={FILL_PANEL}
          >
            <PanelHeader
              titleId="run-ranked-steps"
              icon={Layers}
              title="Steps ranked by queue delay"
              description="Longest measured wait first. Select a step to see its evidence."
              className="fit:shrink-0"
              actions={
                <WorkWaitLegend
                  hatchedLabel="Queue"
                  className="flex-wrap gap-x-5 gap-y-1"
                />
              }
            />
            <RankedSteps
              steps={model.steps}
              selectedId={selected.stepId}
              onSelect={setSelectedId}
            />
          </GlassPanel>
          <SelectedEvidence step={selected} />
        </div>
      ) : (
        <EmptyState
          title="No steps recorded"
          detail="This run has no step events yet, so there is nothing to rank."
        />
      )}
    </div>
  );
}
