"use client";

import * as React from "react";
import Link from "next/link";
import { GitCompareArrows } from "lucide-react";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/widgets/state-views";
import { formatQuantity } from "@/lib/measurement-format";
import { routes } from "@/lib/routes";
import {
  policyUnknown,
  POLICY_NOT_RECORDED_WARNING,
} from "@/lib/run-compatibility";
import {
  compareSelection,
  type CompareSelection,
  type RunsPageRow,
} from "@/lib/runs-page";
import { RunPreview } from "./run-preview";

const count = (value: number) => formatQuantity(value, "count");

/** What the current selection allows, and why when it cannot be compared. */
function compareNote(selection: CompareSelection): string {
  if (selection.status === "incomplete")
    return "Tick two saved runs to compare them; differences are shown only when their recorded settings do not differ. Live runs have no saved timings to compare.";
  if (selection.status === "incompatible") return selection.reason;
  const { verdict, differing, unknown } = selection.compatibility;
  if (verdict === "differs")
    return `Baseline is the older run. Recorded settings differ: ${differing.join(", ")}. Compare lists them and calculates no difference.`;
  if (verdict === "unverified")
    return policyUnknown(selection.compatibility)
      ? `Baseline is the older run. Recorded settings match. ${POLICY_NOT_RECORDED_WARNING}.`
      : `Baseline is the older run. Recorded settings match, but ${unknown.join(", ").toLowerCase()} is not recorded; full compatibility cannot be verified.`;
  return "Baseline is the older run. Recorded settings match; differences are observed, not causal.";
}

/** Selection count, plus the compare link once two compatible runs are ticked. */
export function CompareControls({
  selected,
  onClear,
}: {
  selected: readonly RunsPageRow[];
  onClear: () => void;
}) {
  const selection = compareSelection(selected);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span
        className="glass-raised inline-flex h-11 shrink-0 items-center gap-2 rounded-xl px-3.5 text-sm text-ink"
        aria-live="polite"
      >
        <GitCompareArrows className="size-4" aria-hidden />
        {count(selected.length)} of 2 selected
      </span>
      {selection.status === "ready" && (
        <Button asChild variant="brand-soft" size="pill">
          <Link
            href={routes.compare(selection.candidate.id, selection.baseline.id)}
          >
            Compare runs
          </Link>
        </Button>
      )}
      {selected.length > 0 && (
        <Button variant="glass" size="pill" onClick={onClear}>
          Clear selection
        </Button>
      )}
    </div>
  );
}

/**
 * Why the selection can or cannot be compared. `hideHint` drops the
 * "tick two runs" hint where the surrounding text already gives it.
 */
export function CompareNote({
  selected,
  hideHint = false,
}: {
  selected: readonly RunsPageRow[];
  hideHint?: boolean;
}) {
  const selection = compareSelection(selected);
  if (hideHint && selection.status === "incomplete") return null;
  return (
    <p className="text-xs text-ink-muted" aria-live="polite">
      {compareNote(selection)}
    </p>
  );
}

/**
 * The Compare selection tab: the ticked runs' evidence side by side, baseline
 * (older) first once the pair is comparable.
 */
export function CompareSelectionView({
  selected,
  onClear,
}: {
  selected: readonly RunsPageRow[];
  onClear: () => void;
}) {
  const idPrefix = React.useId();
  const selection = compareSelection(selected);
  const panels =
    selection.status === "ready"
      ? [
          { row: selection.baseline, title: "Baseline · older run" },
          { row: selection.candidate, title: "Candidate · newer run" },
        ]
      : selected.map((row, index) => ({
          row,
          title: `Selected run ${index + 1}`,
        }));

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <CompareControls selected={selected} onClear={onClear} />
        <CompareNote selected={selected} />
      </div>
      {panels.length === 0 ? (
        <EmptyState
          title="No runs selected for comparison"
          detail="Tick the box beside two saved runs on the All runs tab to see their evidence side by side here."
        />
      ) : (
        <div className="grid items-start gap-4 lg:grid-cols-2">
          {panels.map(({ row, title }) => (
            <RunPreview
              key={row.key}
              row={row}
              title={title}
              id={`${idPrefix}-${row.key}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
