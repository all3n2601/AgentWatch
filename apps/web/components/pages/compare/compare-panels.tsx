import Link from "next/link";
import {
  ChartColumn,
  ClipboardCheck,
  Download,
  ExternalLink,
  Info,
  Layers,
  ListOrdered,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  KeyValueList,
  RankedList,
  type KeyValueItem,
} from "@/components/widgets/evidence-lists";
import { Measurement } from "@/components/widgets/measurement";
import type {
  AlignedStep,
  Comparison,
  MetricDelta,
  QualityRow,
  RunSummary,
  StepTiming,
} from "@/lib/compare-page";
import { routes } from "@/lib/routes";
import {
  policyUnknown,
  POLICY_NOT_RECORDED_WARNING,
  type CheckStatus,
  type Compatibility,
  type CompatibilityCheck,
} from "@/lib/run-compatibility";
import { relativeLengths } from "@/lib/segments";
import { cn } from "@/lib/utils";
import { CompareDelta, TimingText } from "./compare-delta";

/** "Match" / "Mismatch" use the shared status tones (lib/tone.ts); unknown is neither. */
const CHECK_STATUS: Record<CheckStatus, { label: string; tone?: "info" }> = {
  same: { label: "Match" },
  differs: { label: "Mismatch" },
  unknown: { label: "Unknown", tone: "info" },
};

/** Baseline and candidate values, one per line, with the full value on hover. */
function PairValue({
  baseline,
  candidate,
  baselineTitle,
  candidateTitle,
}: {
  baseline: string;
  candidate: string;
  baselineTitle?: string;
  candidateTitle?: string;
}) {
  return (
    <span className="block space-y-0.5 text-right [overflow-wrap:anywhere]">
      <span className="block" title={baselineTitle}>
        <span className="text-ink-muted">Baseline </span>
        {baseline}
      </span>
      <span className="block" title={candidateTitle}>
        <span className="text-ink-muted">Candidate </span>
        {candidate}
      </span>
    </span>
  );
}

function CheckValue({ check }: { check: CompatibilityCheck }) {
  if (check.status === "same")
    return (
      <span
        title={check.candidateTitle}
        className="block text-right [overflow-wrap:anywhere]"
      >
        {check.candidate}
      </span>
    );
  return <PairValue {...check} />;
}

export function ConfigurationPanel({
  compatibility,
  className,
}: {
  compatibility: Compatibility;
  className?: string;
}) {
  const items: KeyValueItem[] = compatibility.checks.map((check) => ({
    id: check.id,
    label: check.label,
    value: <CheckValue check={check} />,
    status: CHECK_STATUS[check.status],
  }));
  const verdict =
    compatibility.verdict === "differs"
      ? {
          label: `${compatibility.differing.length} differ`,
          tone: "danger" as const,
        }
      : compatibility.verdict === "unverified"
        ? {
            label: `${compatibility.unknown.length} unknown`,
            tone: "warning" as const,
          }
        : { label: "Configuration match", tone: "success" as const };
  return (
    <GlassPanel
      as="section"
      aria-labelledby="compare-configuration"
      className={cn("fit:flex fit:h-full fit:min-h-0 fit:flex-col", className)}
    >
      <PanelHeader
        titleId="compare-configuration"
        icon={ShieldCheck}
        title="Configuration check"
        description="Settings recorded on both runs"
        actions={<StatusBadge label={verdict.label} tone={verdict.tone} />}
        className="shrink-0"
      />
      <div
        role="region"
        aria-labelledby="compare-configuration"
        tabIndex={0}
        className="relative -mx-1 rounded-xl px-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:min-h-0 fit:flex-1 fit:overflow-y-auto"
      >
        <KeyValueList items={items} dense />
      </div>
      {policyUnknown(compatibility) && (
        <p className="mt-3 shrink-0 text-sm text-ink-muted">
          {POLICY_NOT_RECORDED_WARNING}: saved runs do not record which
          concurrency policy they ran under.
        </p>
      )}
    </GlassPanel>
  );
}

const BAR_MIN_VISIBLE_PERCENT = 1;

/** Shown where a difference would be when recorded settings differ. */
function NotCompared({ className }: { className?: string }) {
  return (
    <span className={cn("text-sm text-ink-subtle", className)}>
      <span aria-hidden>—</span> Not compared
    </span>
  );
}

function PairedBar({
  label,
  seconds,
  length,
  tone,
}: {
  label: string;
  seconds: number | null;
  length: number | null;
  tone: "baseline" | "candidate";
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="sr-only">{label}</span>
      <span
        aria-hidden
        className="relative h-2.5 min-w-0 flex-1 overflow-hidden rounded-full bg-bar-track"
      >
        {length !== null && (
          <span
            className={cn(
              "absolute inset-y-0 left-0 rounded-full",
              tone === "baseline" ? "bg-bar/40" : "bg-bar",
            )}
            style={{
              width: `${Math.max(length * 100, length > 0 ? BAR_MIN_VISIBLE_PERCENT : 0)}%`,
            }}
          />
        )}
      </span>
      <TimingText
        seconds={seconds}
        className="w-20 shrink-0 text-right text-[13px] text-ink-muted"
      />
    </div>
  );
}

function MetricRow({
  metric,
  differencesShown,
}: {
  metric: MetricDelta;
  differencesShown: boolean;
}) {
  const [baselineLength, candidateLength] = relativeLengths([
    metric.baselineSeconds ?? Number.NaN,
    metric.candidateSeconds ?? Number.NaN,
  ]);
  return (
    <li className="grid gap-x-4 gap-y-1.5 py-2.5 fit:py-2 sm:grid-cols-[minmax(9rem,11rem)_minmax(0,1fr)_auto] sm:items-center">
      <div className="min-w-0">
        <p className="text-sm font-medium text-ink">{metric.label}</p>
        <p className="text-xs text-ink-muted">{metric.scope}</p>
      </div>
      <div className="min-w-0 space-y-1">
        <PairedBar
          label="Baseline"
          seconds={metric.baselineSeconds}
          length={baselineLength ?? null}
          tone="baseline"
        />
        <PairedBar
          label="Candidate"
          seconds={metric.candidateSeconds}
          length={candidateLength ?? null}
          tone="candidate"
        />
      </div>
      {differencesShown ? (
        <CompareDelta delta={metric} className="sm:justify-self-end" />
      ) : (
        <NotCompared className="sm:justify-self-end" />
      )}
    </li>
  );
}

function PairLegend() {
  return (
    <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-ink-muted">
      <span className="inline-flex items-center gap-1.5">
        <i aria-hidden className="h-2.5 w-4 rounded-full bg-bar/40" />
        Baseline (upper bar)
      </span>
      <span className="inline-flex items-center gap-1.5">
        <i aria-hidden className="h-2.5 w-4 rounded-full bg-bar" />
        Candidate (lower bar)
      </span>
    </p>
  );
}

/** On `fit` screens the panel fills its column and the rows scroll inside it. */
export function TimeChangesPanel({
  metrics,
  differencesShown,
}: {
  metrics: readonly MetricDelta[];
  differencesShown: boolean;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="compare-changes"
      className="fit:flex fit:h-full fit:min-h-0 fit:flex-col"
    >
      <PanelHeader
        titleId="compare-changes"
        icon={ChartColumn}
        title={differencesShown ? "Where time changed" : "Recorded times"}
        description={
          differencesShown
            ? "Candidate minus baseline · bars scale within each row"
            : "Each run's values · not compared because recorded settings differ"
        }
        actions={<PairLegend />}
        className="shrink-0"
      />
      <div
        role="region"
        aria-labelledby="compare-changes"
        tabIndex={0}
        className="relative -mx-1 rounded-xl px-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:min-h-0 fit:flex-1 fit:overflow-y-auto"
      >
        <ul className="divide-y divide-line">
          {metrics.map((metric) => (
            <MetricRow
              key={metric.id}
              metric={metric}
              differencesShown={differencesShown}
            />
          ))}
        </ul>
      </div>
    </GlassPanel>
  );
}

function StepCell({ timing }: { timing: StepTiming | null }) {
  if (!timing)
    return (
      <span className="text-sm text-ink-subtle">
        <span aria-hidden>—</span> Not recorded
      </span>
    );
  return (
    <span className="block">
      <TimingText
        seconds={timing.durationSeconds}
        className="font-medium text-ink"
      />
      <span className="block text-xs text-ink-muted">
        queue <TimingText seconds={timing.queueSeconds} /> · exec{" "}
        <TimingText seconds={timing.executionSeconds} />
      </span>
    </span>
  );
}

function WaitShare({ timing }: { timing: StepTiming | null }) {
  const percent = timing ? timing.queueFraction * 100 : null;
  return (
    <span className="flex items-center gap-2">
      <span
        aria-hidden
        className="relative h-2 w-16 shrink-0 overflow-hidden rounded-full bg-bar-track"
      >
        {percent !== null && (
          <span
            className="hatch absolute inset-y-0 left-0 border-r border-line"
            style={{ width: `${percent}%` }}
          />
        )}
      </span>
      <Measurement
        value={percent}
        unit="percent"
        className="text-sm text-ink"
      />
    </span>
  );
}

const TH = "px-3 py-2 text-left text-[13px] font-medium text-ink-muted";
const TD = "px-3 py-2.5 align-top";

/**
 * On `fit` screens the panel fills its tab and the table scrolls inside it
 * with a sticky header; below that the page scrolls and only wide tables
 * scroll sideways.
 */
export function AlignedStepsPanel({
  steps,
  differencesShown,
}: {
  steps: readonly AlignedStep[];
  differencesShown: boolean;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="compare-steps"
      className="fit:flex fit:h-full fit:min-h-0 fit:flex-col"
    >
      <PanelHeader
        titleId="compare-steps"
        icon={Layers}
        title="Aligned step comparison"
        description="Steps paired by recorded step ID · wait share is the candidate's queue delay over its step duration (hatched)"
        className="shrink-0"
      />
      <div
        role="region"
        aria-labelledby="compare-steps"
        tabIndex={0}
        className="relative -mx-1 overflow-x-auto rounded-xl px-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:min-h-0 fit:flex-1 fit:overflow-y-auto"
      >
        <table className="w-full min-w-[40rem] text-sm text-ink">
          <thead className="sticky top-0 z-10 border-b border-line bg-glass-strong backdrop-blur">
            <tr>
              <th scope="col" className={TH}>
                Step
              </th>
              <th scope="col" className={TH}>
                Baseline
                <span className="block text-xs font-normal">
                  duration · queue · exec
                </span>
              </th>
              <th scope="col" className={TH}>
                Candidate
                <span className="block text-xs font-normal">
                  duration · queue · exec
                </span>
              </th>
              <th scope="col" className={TH}>
                Candidate wait share
              </th>
              {differencesShown && (
                <th scope="col" className={cn(TH, "text-right")}>
                  Duration delta
                </th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {steps.map((step) => (
              <tr key={step.stepId}>
                <th scope="row" className={cn(TD, "text-left font-medium")}>
                  {step.label}
                </th>
                <td className={TD}>
                  <StepCell timing={step.baseline} />
                </td>
                <td className={TD}>
                  <StepCell timing={step.candidate} />
                </td>
                <td className={TD}>
                  <WaitShare timing={step.candidate} />
                </td>
                {differencesShown && (
                  <td className={cn(TD, "text-right")}>
                    <CompareDelta delta={step.duration} />
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </GlassPanel>
  );
}

function RunActions({ run, role }: { run: RunSummary; role: string }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Button asChild variant="glass" size="pill">
        <Link href={routes.run(run.id)}>
          <ExternalLink aria-hidden />
          Open {role}
        </Link>
      </Button>
      <Button asChild variant="glass" size="pill">
        <a href={run.evidenceHref} download>
          <Download aria-hidden />
          {run.persisted
            ? `Export ${role} evidence`
            : `Export ${role} local artifact`}
        </a>
      </Button>
    </div>
  );
}

export function LargestDifferencesPanel({
  comparison,
}: {
  comparison: Comparison;
}) {
  const ranked = comparison.largestStepDifferences;
  return (
    <GlassPanel as="section" aria-labelledby="compare-largest">
      <PanelHeader
        titleId="compare-largest"
        icon={ListOrdered}
        title="Largest differences"
        description="Steps recorded on both runs, by absolute duration change"
      />
      {!comparison.differencesShown ? (
        <p className="text-sm text-ink-muted">
          Not calculated: recorded settings differ.
        </p>
      ) : ranked.length > 0 ? (
        <RankedList
          items={ranked.map((step) => ({
            id: step.stepId,
            label: step.label,
            value: <CompareDelta delta={step.duration} />,
          }))}
        />
      ) : (
        <p className="text-sm text-ink-muted">
          No step was recorded on both runs, so no step difference is available.
        </p>
      )}
      <div className="mt-4 space-y-2 border-t border-line pt-4">
        <RunActions run={comparison.baseline} role="baseline" />
        <RunActions run={comparison.candidate} role="candidate" />
      </div>
    </GlassPanel>
  );
}

/** Quality and coverage per run, side by side; never differenced. */
export function QualityPanel({ rows }: { rows: readonly QualityRow[] }) {
  return (
    <GlassPanel as="section" aria-labelledby="compare-quality">
      <PanelHeader
        titleId="compare-quality"
        icon={ClipboardCheck}
        title="Quality and coverage"
        description="Recorded on each run, shown side by side"
      />
      <KeyValueList
        dense
        items={rows.map((row) => ({
          id: row.id,
          label: row.label,
          value: (
            <PairValue baseline={row.baseline} candidate={row.candidate} />
          ),
        }))}
      />
    </GlassPanel>
  );
}

export function InterpretationPanel({
  comparison,
}: {
  comparison: Comparison;
}) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="compare-interpretation"
      tone="info"
    >
      {/* Side by side on small screens; stacked from lg, where the panel sits
          in the narrow summary column. */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-5 lg:flex-col lg:gap-2">
        <h2
          id="compare-interpretation"
          className="flex shrink-0 items-center gap-2 text-base font-semibold text-ink sm:text-lg"
        >
          <Info className="size-5 text-info" aria-hidden />
          Interpretation
        </h2>
        <div className="min-w-0 space-y-2 text-sm text-ink sm:border-l sm:border-line sm:pl-5 lg:border-l-0 lg:pl-0">
          <p>{comparison.interpretation}</p>
          {comparison.limitations.length > 0 && (
            <details className="text-ink-muted">
              <summary className="cursor-pointer py-1 outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40">
                Recorded limitations ({comparison.limitations.length})
              </summary>
              <ul className="mt-1 list-disc space-y-0.5 pl-5">
                {comparison.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </div>
    </GlassPanel>
  );
}
