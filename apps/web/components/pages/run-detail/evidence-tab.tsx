import Link from "next/link";
import {
  ArrowDownToLine,
  FileJson,
  FileText,
  GitCompareArrows,
  ListChecks,
  Ruler,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  GlassPanel,
  InsetCard,
  PanelHeader,
} from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { Checklist, KeyValueList } from "@/components/widgets/evidence-lists";
import { evidenceUrl } from "@/lib/dashboard-data";
import type { Experiment } from "@/lib/experiment-schema";
import { formatQuantity } from "@/lib/measurement-format";
import { routes } from "@/lib/routes";
import { cn } from "@/lib/utils";
import { StepTiming } from "./step-timing";
import {
  FILL_PANEL,
  PANEL_SCROLL,
  TAB_GRID,
} from "@/components/shell/panel-layout";

const SAVED_FILES: {
  file: string;
  icon: LucideIcon;
  contents: string;
}[] = [
  {
    file: "results.json",
    icon: FileJson,
    contents:
      "Run summary: configuration, every step's timings, acceptance checks and limitations.",
  },
  {
    file: "spans.jsonl",
    icon: FileText,
    contents: "One worker span per step: submit, start and finish timestamps.",
  },
  {
    file: "samples.jsonl",
    icon: FileText,
    contents:
      "Host sampler readings: CPU pressure, cgroup CPU and unavailable signals.",
  },
];

/** "stable_baseline" → "Stable baseline". */
const checkLabel = (name: string) => {
  const words = name.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
};

function EvidenceFiles({ run }: { run: Experiment }) {
  return (
    <GlassPanel
      as="section"
      aria-labelledby="run-evidence-files"
      className={FILL_PANEL}
    >
      <PanelHeader
        titleId="run-evidence-files"
        icon={FileJson}
        title="Evidence files"
        description="The artifacts saved with this run, downloaded exactly as recorded."
        className="fit:shrink-0"
      />
      <div className={cn(PANEL_SCROLL, "space-y-3")}>
        <ul className="divide-y divide-line rounded-2xl border border-line">
          {SAVED_FILES.map(({ file, icon: Icon, contents }) => (
            <li
              key={file}
              className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3"
            >
              <span className="glass-raised flex size-10 shrink-0 items-center justify-center rounded-xl">
                <Icon className="size-5 text-ink" aria-hidden />
              </span>
              <span className="min-w-48 flex-1">
                <span className="block font-mono text-[15px] text-ink">
                  {file}
                </span>
                <span className="block text-sm text-ink-muted">{contents}</span>
              </span>
              <Button asChild variant="glass" size="pill">
                <a
                  href={evidenceUrl(run, file)}
                  aria-label={`Download ${file}`}
                >
                  <ArrowDownToLine aria-hidden />
                  Download
                </a>
              </Button>
            </li>
          ))}
        </ul>
        <InsetCard className="flex flex-wrap items-center justify-between gap-3">
          <span className="min-w-48 flex-1 text-sm text-ink-muted">
            Set this run beside another of the same workload. Differences are
            observed, not causal.
          </span>
          <Button asChild variant="brand-soft" size="pill">
            <Link href={routes.compare(run.run_id)}>
              <GitCompareArrows aria-hidden />
              Compare with another run
            </Link>
          </Button>
        </InsetCard>
      </div>
    </GlassPanel>
  );
}

function Method({ run }: { run: Experiment }) {
  const { experiment, baseline } = run;
  return (
    <InsetCard>
      <h3 className="mb-1 flex items-center gap-2 text-sm font-semibold text-ink">
        <Ruler className="size-4" aria-hidden />
        How attribution was decided
      </h3>
      <KeyValueList
        dense
        items={[
          { id: "rule", label: "Rule", value: run.contended.rule_version },
          {
            id: "threshold",
            label: "Queue threshold",
            value: <StepTiming seconds={baseline.queue_threshold_seconds} />,
          },
          {
            id: "repeats",
            label: "Baseline repeats",
            value: formatQuantity(experiment.baseline_repeats, "count"),
          },
          {
            id: "cv",
            label: "Baseline execution CV",
            value: formatQuantity(baseline.execution_cv * 100, "percent"),
          },
          {
            id: "capacity",
            label: "Worker capacity",
            value: formatQuantity(experiment.worker_capacity, "count"),
          },
          { id: "workload", label: "Workload", value: experiment.workload },
        ]}
      />
    </InsetCard>
  );
}

function Checks({ run }: { run: Experiment }) {
  const checks = Object.entries(run.checks).filter(
    ([name]) => name !== "passed",
  );
  const passed = checks.filter(([, ok]) => ok).length;
  return (
    <GlassPanel
      as="section"
      aria-labelledby="run-checks"
      className={FILL_PANEL}
    >
      <PanelHeader
        titleId="run-checks"
        icon={ListChecks}
        title="Acceptance checks"
        description={`${passed} of ${checks.length} passed`}
        className="fit:shrink-0"
        actions={
          <StatusBadge
            label={run.checks.passed ? "Passed" : "Inconclusive"}
            tone={run.checks.passed ? "success" : "warning"}
          />
        }
      />
      <div
        tabIndex={0}
        role="group"
        aria-label="Acceptance checks and method"
        className={cn(PANEL_SCROLL, "space-y-3")}
      >
        <InsetCard>
          <Checklist
            items={checks.map(([name, ok]) => ({
              id: name,
              label: checkLabel(name),
              state: ok ? "pass" : "fail",
            }))}
          />
        </InsetCard>
        <Method run={run} />
      </div>
    </GlassPanel>
  );
}

/** What was saved with the run, the checks it was held to, and how attribution was decided. */
export function EvidenceTab({ run }: { run: Experiment }) {
  return (
    <div className={cn(TAB_GRID, "fit:h-full")}>
      <EvidenceFiles run={run} />
      <Checks run={run} />
    </div>
  );
}
