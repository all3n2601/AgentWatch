"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { ArrowLeftRight, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CopyChip } from "@/components/ui/copy-chip";
import { FilterSelect } from "@/components/ui/filter-select";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  COMPARE_TAB_PARAM,
  compareHref,
  type RunOption,
  type RunSummary,
} from "@/lib/compare-page";

function RunMeta({
  run,
  role,
  storedAt,
}: {
  run: RunSummary;
  role: string;
  storedAt: string | null;
}) {
  return (
    // One line on `fit` screens (the save time truncates, with the full value
    // in its title) so the picker stays short above the tabs.
    <div className="flex flex-wrap items-center gap-2 text-[13px] text-ink-muted fit:flex-nowrap">
      <CopyChip value={run.id} label={`${role} run ID`} />
      <span
        className="min-w-0 truncate tabular-nums"
        title={storedAt ?? undefined}
      >
        {storedAt ?? "Local artifact"}
      </span>
      <StatusBadge
        label={run.passed ? "Checks passed" : "Checks failed"}
        tone={run.passed ? "success" : "danger"}
      />
    </div>
  );
}

type Slot = {
  options: readonly RunOption[];
  /** Selected id, only when it names a loaded run. */
  value: string | undefined;
  run: RunSummary | null;
  /** Display time for the selected run, formatted on the server. */
  storedAt: string | null;
};

/**
 * Chooses the baseline and candidate. Each choice writes `?run=&baseline=`
 * (keeping the open `?view=` tab); the server then loads exactly those runs.
 */
export function RunPicker({
  baseline,
  candidate,
  status,
}: {
  baseline: Slot;
  candidate: Slot;
  /** Configuration status shown between the two choices. */
  status?: React.ReactNode;
}) {
  const router = useRouter();
  const [pending, startTransition] = React.useTransition();

  function choose(next: { run?: string; baseline?: string }) {
    // TabbedView records the open tab with history.replaceState, which
    // useSearchParams does not observe, so read it from the live URL here.
    const view = new URLSearchParams(window.location.search).get(
      COMPARE_TAB_PARAM,
    );
    startTransition(() => {
      router.push(
        compareHref(
          next.run ?? candidate.value,
          next.baseline ?? baseline.value,
          view,
        ),
      );
    });
  }

  const canSwap = baseline.value !== undefined && candidate.value !== undefined;

  return (
    <div className="glass-panel grid gap-3 p-3 sm:p-4 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] lg:items-start">
      <div className="min-w-0 space-y-2">
        <FilterSelect
          label="Baseline"
          caption
          icon={Layers}
          value={baseline.value}
          onValueChange={(value) => choose({ baseline: value })}
          options={baseline.options.map((o) => ({
            value: o.id,
            label: o.label,
          }))}
          placeholder="Choose a baseline run"
          className="w-full [&_[data-slot=select-value]]:block [&_[data-slot=select-value]]:max-w-full [&_[data-slot=select-value]]:truncate"
        />
        {baseline.run && (
          <RunMeta
            run={baseline.run}
            role="Baseline"
            storedAt={baseline.storedAt}
          />
        )}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 lg:h-14">
        {canSwap ? (
          <Button
            type="button"
            variant="glass"
            size="icon-round"
            aria-label="Swap baseline and candidate"
            title="Swap baseline and candidate"
            disabled={pending}
            onClick={() =>
              choose({ run: baseline.value, baseline: candidate.value })
            }
          >
            <ArrowLeftRight aria-hidden />
          </Button>
        ) : null}
        {status}
        <span
          role="status"
          aria-live="polite"
          className="text-[13px] text-ink-muted"
        >
          {pending ? "Loading comparison…" : ""}
        </span>
      </div>
      <div className="min-w-0 space-y-2">
        <FilterSelect
          label="Candidate"
          caption
          icon={Layers}
          value={candidate.value}
          onValueChange={(value) => choose({ run: value })}
          options={candidate.options.map((o) => ({
            value: o.id,
            label: o.label,
          }))}
          placeholder="Choose a candidate run"
          className="w-full [&_[data-slot=select-value]]:block [&_[data-slot=select-value]]:max-w-full [&_[data-slot=select-value]]:truncate"
        />
        {candidate.run && (
          <RunMeta
            run={candidate.run}
            role="Candidate"
            storedAt={candidate.storedAt}
          />
        )}
      </div>
    </div>
  );
}
