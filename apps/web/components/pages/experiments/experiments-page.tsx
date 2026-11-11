import { ShieldCheck, TriangleAlert } from "lucide-react";
import { WorkspacePage } from "@/components/shell/app-shell";
import { PageHeader } from "@/components/shell/page-header";
import { TabbedView, type TabbedViewTab } from "@/components/shell/tabbed-view";
import { GlassPanel, PanelHeader } from "@/components/ui/glass-panel";
import { AlertBanner } from "@/components/widgets/alert-banner";
import { Checklist } from "@/components/widgets/evidence-lists";
import { EmptyState, UnavailableState } from "@/components/widgets/state-views";
import {
  checkTally,
  type ExperimentsPage as PageModel,
  type ReplaySequence,
} from "@/lib/experiments-page";
import type { ExperimentsPageState } from "@/lib/experiments-page-load";
import { formatQuantity } from "@/lib/measurement-format";
import { slowdownTone } from "@/lib/tone";
import { BusinessReadingPanel } from "./business-reading-panel";
import { ConditionImpactPanel } from "./condition-impact-panel";
import { ExperimentTable } from "./experiment-table";
import {
  ReplaySequencePanel,
  ReplaySequenceValues,
} from "./replay-sequence-panel";
import { ReplayStructurePanel } from "./replay-structure-panel";

const count = (value: number) => formatQuantity(value, "count");
const TITLE = "Experiments";
const DESCRIPTION = "Replay agent workloads against their own clean baseline";

function HeadlineBanner({ page }: { page: PageModel }) {
  const { largest } = page;
  return largest ? (
    <AlertBanner
      tone={slowdownTone(largest.medianRatio) ?? "info"}
      icon={TriangleAlert}
      title={`${largest.shortLabel} contention shows the largest observed impact: ${formatQuantity(largest.medianRatio, "ratio")} vs clean`}
      detail={`Median across ${count(largest.experiments)} experiments of each experiment's ${largest.shortLabel.toLowerCase()} replay over its own clean replays.`}
      aside="Observed, not causal"
    />
  ) : (
    <AlertBanner
      tone="info"
      title="No condition has a clean baseline to compare against"
      aside="Observed, not causal"
    />
  );
}

function SummaryTab({ page }: { page: PageModel }) {
  return (
    <div className="space-y-4">
      <HeadlineBanner page={page} />
      {page.conditions.length > 0 && (
        <ConditionImpactPanel conditions={page.conditions} />
      )}
    </div>
  );
}

/** Chart and its values side by side; the values scroll inside their panel. */
function SequenceTab({
  sequence,
  experiments,
}: {
  sequence: ReplaySequence;
  experiments: number;
}) {
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(19rem,25rem)] xl:items-start fit:h-full fit:items-stretch">
      <ReplaySequencePanel
        sequence={sequence}
        experiments={experiments}
        className="fit:self-start"
      />
      <ReplaySequenceValues sequence={sequence} />
    </div>
  );
}

function StructureTab({ page }: { page: PageModel }) {
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(19rem,22rem)] xl:items-start">
      <ReplayStructurePanel page={page} />
      {page.reading && (
        <BusinessReadingPanel
          reading={page.reading}
          models={page.models}
          tasks={page.groups.length}
        />
      )}
    </div>
  );
}

function ChecksTab({ checks }: { checks: PageModel["checks"] }) {
  return (
    <GlassPanel as="section" aria-labelledby="acceptance-checks">
      <PanelHeader
        titleId="acceptance-checks"
        icon={ShieldCheck}
        title="Acceptance checks"
        description="Results recorded in the release and its build report. Checks without a recorded result read Not checked."
      />
      {/* Two columns on wide screens so all checks fit without scrolling. */}
      <Checklist
        items={checks}
        className="xl:grid xl:grid-cols-2 xl:gap-x-6 xl:divide-y-0 xl:[&>li]:border-t xl:[&>li]:border-line"
      />
    </GlassPanel>
  );
}

function Ready({ page, initialTab }: { page: PageModel; initialTab?: string }) {
  const tally = checkTally(page.checks);
  const tabs: TabbedViewTab[] = [
    { id: "summary", label: "Summary", content: <SummaryTab page={page} /> },
    ...(page.sequence
      ? [
          {
            id: "sequence",
            label: "Replay sequence",
            content: (
              <SequenceTab
                sequence={page.sequence}
                experiments={page.experiments}
              />
            ),
          },
        ]
      : []),
    {
      id: "structure",
      label: "Replay structure",
      content: <StructureTab page={page} />,
    },
    {
      id: "checks",
      label: "Acceptance checks",
      badge: `${count(tally.passed)} of ${count(tally.total)} passed`,
      content: <ChecksTab checks={page.checks} />,
    },
    {
      id: "experiments",
      label: "Experiments",
      badge: count(page.experiments),
      content: (
        <ExperimentTable groups={page.groups} experiments={page.experiments} />
      ),
    },
  ];
  return (
    <TabbedView
      label="Experiment views"
      initialTab={initialTab}
      header={
        <PageHeader
          title={TITLE}
          description={DESCRIPTION}
          scope={
            <>
              {count(page.experiments)} experiments · {count(page.replayRuns)}{" "}
              replay runs · {count(page.replaySteps)} steps ·{" "}
              <span title={page.release} className="whitespace-nowrap">
                {page.releaseName}
              </span>
            </>
          }
        />
      }
      tabs={tabs}
    />
  );
}

/**
 * The Experiments page: replay experiments of the current dataset release.
 * `initialTab` is the `?view=` tab named in the URL.
 */
export function ExperimentsPage({
  state,
  initialTab,
}: {
  state: ExperimentsPageState;
  initialTab?: string;
}) {
  return (
    <WorkspacePage>
      {state.status === "ready" && (
        <Ready page={state.page} initialTab={initialTab} />
      )}
      {state.status === "unavailable" && (
        <div className="space-y-6">
          <PageHeader title={TITLE} description={DESCRIPTION} />
          <UnavailableState
            title="Dataset unavailable"
            reason={
              <>
                {state.reason} To read another checkout, set{" "}
                <code>AGENTWATCH_DATASET_DIR</code>.
              </>
            }
          />
        </div>
      )}
      {state.status === "empty" && (
        <div className="space-y-6">
          <PageHeader
            title={TITLE}
            description={DESCRIPTION}
            scope={state.release}
          />
          <EmptyState
            title="No replay experiments"
            detail="This release has no agent replay experiments, so there is no clean baseline to compare conditions against."
          />
        </div>
      )}
    </WorkspacePage>
  );
}
