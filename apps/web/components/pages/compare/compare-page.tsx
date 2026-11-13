import Link from "next/link";
import { History } from "lucide-react";
import { WorkspacePage } from "@/components/shell/app-shell";
import { PageHeader } from "@/components/shell/page-header";
import { TabbedView, type TabbedViewTab } from "@/components/shell/tabbed-view";
import { Button } from "@/components/ui/button";
import { Chip } from "@/components/ui/chip";
import { GlassPanel } from "@/components/ui/glass-panel";
import { StatusBadge, StatusIcon } from "@/components/ui/status-badge";
import { AlertBanner } from "@/components/widgets/alert-banner";
import { EmptyState, UnavailableState } from "@/components/widgets/state-views";
import {
  formatStoredAt,
  type ComparePageModel,
  type CompareTab,
  type Comparison,
  type RunOption,
  type RunSlot,
} from "@/lib/compare-page";
import { formatQuantity } from "@/lib/measurement-format";
import { routes } from "@/lib/routes";
import {
  policyUnknown,
  POLICY_NOT_RECORDED_WARNING,
  VERDICT_TONE,
  verdictBadge,
  type Compatibility,
} from "@/lib/run-compatibility";
import {
  AlignedStepsPanel,
  ConfigurationPanel,
  InterpretationPanel,
  LargestDifferencesPanel,
  QualityPanel,
  TimeChangesPanel,
} from "./compare-panels";
import { RunPicker } from "./run-picker";

const ROLE_NAMES = { candidate: "Candidate", baseline: "Baseline" } as const;

function sourceLabel(source: ComparePageModel["source"]) {
  return source === "database"
    ? "Saved run history"
    : "Local artifact fallback";
}

function pickerSlot(slot: RunSlot, options: readonly RunOption[]) {
  const run = slot.status === "selected" ? slot.run : null;
  return {
    options,
    value: run?.id,
    run,
    storedAt: formatStoredAt(run?.storedAt),
  };
}

/** The configuration verdict; unknown settings are never labelled a match. */
export function verdictStatus(compatibility: Compatibility) {
  const tone = VERDICT_TONE[compatibility.verdict];
  if (compatibility.verdict === "differs")
    return { label: "Settings differ", tone };
  if (compatibility.verdict === "unverified")
    return {
      label: policyUnknown(compatibility)
        ? "Policy not recorded"
        : "Settings not recorded",
      tone,
    };
  return { label: "Configuration match", tone };
}

function ConfigurationStatus({
  comparison,
  className,
}: {
  comparison: Comparison;
  className?: string;
}) {
  const status = verdictStatus(comparison.compatibility);
  return (
    <StatusBadge
      label={status.label}
      tone={status.tone}
      className={className}
    />
  );
}

/** Visible, role-specific notices; a missing run is never replaced. */
function SelectionNotices({ model }: { model: ComparePageModel }) {
  const missing = (["baseline", "candidate"] as const).flatMap((role) => {
    const slot = model[role];
    return slot.status === "not-found"
      ? [{ role, requestedId: slot.requestedId }]
      : [];
  });
  return (
    <>
      {missing.map(({ role, requestedId }) => (
        <AlertBanner
          key={role}
          tone="warning"
          title={`${ROLE_NAMES[role]} run not found`}
          detail={
            <>
              No loaded run has ID{" "}
              <code className="font-mono break-all">{requestedId}</code>. Choose
              another {role} run; no other run is shown in its place.
            </>
          }
        />
      ))}
      {model.sameRun && (
        <AlertBanner
          tone="warning"
          title="Choose two different runs"
          detail="The baseline and candidate are the same run, so there is nothing to compare."
        />
      )}
      {model.warnings.length > 0 && (
        <GlassPanel tone="warning" padding="sm" role="status">
          <h2 className="sr-only">Run history notices</h2>
          {/* Capped on one-screen layouts so many notices cannot push the
              comparison off screen; the list scrolls instead. */}
          <ul className="space-y-1.5 fit:max-h-20 fit:overflow-y-auto">
            {model.warnings.map((warning) => (
              <li
                key={warning}
                className="flex items-start gap-2 text-sm text-ink"
              >
                <StatusIcon tone="warning" label="Warning" />
                <span className="min-w-0 break-words">{warning}</span>
              </li>
            ))}
          </ul>
        </GlassPanel>
      )}
    </>
  );
}

/**
 * The verdict before any number: differing settings block the comparison; an
 * unknown setting allows the neutral observed total with a visible warning.
 */
function ComparisonHeadline({ comparison }: { comparison: Comparison }) {
  const { compatibility, headline } = comparison;
  if (compatibility.verdict === "differs")
    return (
      <AlertBanner
        tone="danger"
        title="Not compared: recorded settings differ"
        detail={`Differing settings: ${compatibility.differing.join(", ")}. No difference or ratio is calculated; each run's values are shown side by side.`}
        className="fit:py-3"
      />
    );
  const total = comparison.metrics.find((m) => m.id === "total");
  const detail =
    compatibility.verdict === "unverified"
      ? policyUnknown(compatibility)
        ? `${POLICY_NOT_RECORDED_WARNING}. Every recorded setting matches.`
        : `Not recorded on both runs: ${compatibility.unknown.join(", ")}; full compatibility cannot be verified.`
      : `Total step time over ${total?.scope ?? "all steps"}; recorded settings match.`;
  return (
    <AlertBanner
      tone={compatibility.verdict === "unverified" ? "warning" : "info"}
      title={headline?.title ?? "Total step time could not be compared"}
      detail={detail}
      aside="Observed difference, not causal"
      // On phones the aside (the banner's only direct <p>) takes its own row
      // instead of squeezing the headline into a one-word column.
      className="max-sm:[&>p]:basis-full fit:py-3"
    />
  );
}

function comparisonTabs(
  comparison: Comparison,
): (TabbedViewTab & { id: CompareTab })[] {
  const { compatibility } = comparison;
  return [
    {
      id: "summary",
      label: "Summary",
      // On `fit` screens the grid fills the tab: the timing panel scrolls its
      // rows and the side column scrolls as a whole, so neither moves the page.
      content: (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.65fr)_minmax(0,1fr)] lg:items-start fit:h-full fit:grid-rows-[minmax(0,1fr)] fit:items-stretch">
          <TimeChangesPanel
            metrics={comparison.metrics}
            differencesShown={comparison.differencesShown}
          />
          <div
            role="region"
            aria-label="Interpretation, quality and largest differences"
            tabIndex={0}
            className="relative min-w-0 space-y-4 rounded-2xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:h-full fit:min-h-0 fit:overflow-y-auto"
          >
            <InterpretationPanel comparison={comparison} />
            <QualityPanel rows={comparison.quality} />
            <LargestDifferencesPanel comparison={comparison} />
          </div>
        </div>
      ),
    },
    {
      id: "configuration",
      label: "Configuration",
      badge: verdictBadge(compatibility).label,
      content: (
        <ConfigurationPanel
          compatibility={compatibility}
          className="max-w-4xl"
        />
      ),
    },
    {
      id: "steps",
      label: "Steps",
      badge: formatQuantity(comparison.steps.length, "count"),
      content: (
        <AlignedStepsPanel
          steps={comparison.steps}
          differencesShown={comparison.differencesShown}
        />
      ),
    },
  ];
}

function NoComparison({ model }: { model: ComparePageModel }) {
  if (model.loadedRuns === 0)
    return (
      <UnavailableState
        title="No runs available to compare"
        reason="Neither saved run history nor local artifacts returned a run. Record an experiment, then return here."
        action={
          <Button asChild variant="glass" size="pill">
            <Link href={routes.section("runs")}>
              <History aria-hidden />
              Open runs
            </Link>
          </Button>
        }
      />
    );
  if (
    model.sameRun ||
    model.candidate.status === "not-found" ||
    model.baseline.status === "not-found"
  )
    return null;
  const chosen = [model.baseline, model.candidate].filter(
    (slot) => slot.status === "selected",
  ).length;
  return (
    <EmptyState
      title={
        chosen === 0
          ? "Choose two runs to compare"
          : `Choose a ${model.baseline.status === "selected" ? "candidate" : "baseline"} run`
      }
      detail={
        <>
          Pick a baseline and a candidate above. Compare runs of the same
          workload kind; the page shows which settings differ. You can also find
          runs on the Runs page and copy their IDs.
        </>
      }
      action={
        <Button asChild variant="glass" size="pill">
          <Link href={routes.section("runs")}>
            <History aria-hidden />
            Browse runs
          </Link>
        </Button>
      }
    />
  );
}

/**
 * The Compare page: two chosen saved runs, their compatibility and observed
 * deltas. The picker and headline stay above the tabs, so the compatibility
 * verdict is read before any delta on every tab. On `fit` screens the page
 * does not scroll; the open tab's panel does.
 */
export function ComparePageView({
  model,
  initialTab,
}: {
  model: ComparePageModel;
  initialTab?: CompareTab;
}) {
  const { comparison } = model;
  const header = (
    <div className="space-y-3">
      <PageHeader
        title="Compare runs"
        description="Measure observed differences between compatible workloads"
        scope={
          <>
            {sourceLabel(model.source)} ·{" "}
            {formatQuantity(model.loadedRuns, "count")} runs loaded
            {comparison &&
              ` · candidate ${comparison.candidate.id.slice(0, 8)} vs baseline ${comparison.baseline.id.slice(0, 8)}`}
          </>
        }
        actions={
          model.source !== "database" ? (
            <Chip size="sm">Local files, not saved history</Chip>
          ) : undefined
        }
      />
      <RunPicker
        baseline={pickerSlot(model.baseline, model.baselineOptions)}
        candidate={pickerSlot(model.candidate, model.candidateOptions)}
        status={
          comparison ? (
            // On `fit` screens the headline directly below states the same
            // compatibility verdict; hiding the duplicate here leaves the run
            // details room for one line, keeping the tabs taller.
            <ConfigurationStatus
              comparison={comparison}
              className="fit:hidden"
            />
          ) : undefined
        }
      />
      <SelectionNotices model={model} />
      {comparison && <ComparisonHeadline comparison={comparison} />}
    </div>
  );
  return (
    <WorkspacePage>
      {comparison ? (
        <TabbedView
          label="Comparison views"
          header={header}
          initialTab={initialTab}
          tabs={comparisonTabs(comparison)}
        />
      ) : (
        <div className="space-y-4 fit:flex fit:h-full fit:min-h-0 fit:flex-col fit:gap-3 fit:space-y-0">
          <div className="shrink-0">{header}</div>
          <div className="fit:min-h-0 fit:flex-1 fit:overflow-y-auto">
            <NoComparison model={model} />
          </div>
        </div>
      )}
    </WorkspacePage>
  );
}
